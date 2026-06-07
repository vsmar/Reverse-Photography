"""
Decoder for time-multiplexed BINARY structured-light captures.

Capture model (from projection_control.py):
  - Frame 0 is the all-white reference (every cell ON).
  - Frames 1..K each light ~half the cells; a cell's ON/OFF state across
    these K frames is its unique binary CODE (verified: all 65536 cells in the
    256x256 grid get distinct 16-bit codes).

Decoding does NOT solve A x = y. Each projector cell has a unique code, so we
read each camera pixel's observed bit-sequence and look up which cell it maps
to -- a codebook lookup. Exact, no solver, no sparsity prior.

Pipeline per camera:
  decode_pixels(stack, matrix) -> (code_map, valid_mask)   per-pixel cell id
  dual_photo(code_map, valid_mask, reference, matrix)      projector-PoV image
"""

import os
import glob
import argparse
import numpy as np
import cv2
from camera_controls import CameraController as Cam, CAMERA_SERIALS, OUTPUT_DIR


def build_codebook(matrix):
    """From the saved (F, C) pattern matrix, return:
      - coded:    (K, C) the coded frames (reference row 0 dropped)
      - code_int: (C,) integer code per cell, for fast lookup
      - lut:      dict {code_int -> cell_id}
    Assumes row 0 is the all-white reference frame.
    """
    coded = matrix[1:].astype(np.uint8)          # (K, C)
    K, C = coded.shape
    weights = (1 << np.arange(K))[::-1].astype(np.uint64)
    code_int = (coded.T.astype(np.uint64) @ weights)  # (C,)
    lut = {int(code): cell for cell, code in enumerate(code_int)}
    return coded, code_int, lut


def decode_pixels(stack, matrix, contrast_frac=0.2, ref_floor=30.0,
                  median_clean=False, complementary=False, margin=3.0):
    """Assign each camera pixel the projector cell that illuminates it.

    stack:          (F, H, W) grayscale frames. Row 0 = all-white reference,
                    then either K coded frames (threshold mode) or K code/
                    inverse PAIRS (complementary mode). The black calibration
                    frame is NOT part of this stack -- it is subtracted upstream
                    in load_frame_stack.
    matrix:         (F, C) saved pattern matrix (white row + code rows; in
                    complementary mode the code rows are interleaved with
                    inverses just like the frames).
    contrast_frac:  relative-contrast validity threshold (threshold mode only).
    ref_floor:      absolute min reference brightness (0-255) to trust a pixel.
    median_clean:   3x3 median cleanup of the code map.
    complementary:  if True, frames after the white reference are
                    [code0, ~code0, code1, ~code1, ...] and each bit is decided
                    by (code_frame > inverse_frame) -- robust to gloss/specular.
    margin:         complementary mode: a bit is "uncertain" (pixel invalid) if
                    |code - inverse| <= margin.

    Returns (code_map, valid_mask, confidence), each (H, W).
    confidence is a per-pixel 0..1 reliability used later for fusion.
    """
    F, H, W = stack.shape
    obs = stack.reshape(F, H * W).astype(np.float64)  # (F, P)
    reference = obs[0]               # all-white frame: per-pixel "fully lit"

    if complementary:
        pair_frames = obs[1:]                       # (2K, P)
        n_pairs = pair_frames.shape[0] // 2
        code_frames = pair_frames[0::2][:n_pairs]   # (K, P)
        inv_frames = pair_frames[1::2][:n_pairs]    # (K, P)
        K = n_pairs

        diff = code_frames - inv_frames             # (K, P)
        bits = (diff > 0).astype(np.uint64)
        uncertain = (np.abs(diff) <= margin)        # (K, P)
        any_uncertain = uncertain.any(axis=0)       # (P,)

        # Confidence: how decisively each bit was called, worst bit per pixel,
        # normalised by the pixel's own dynamic range. Higher = more reliable.
        denom = np.maximum(reference, 1e-9)
        per_bit_conf = np.abs(diff) / denom[None, :]   # (K, P)
        confidence_flat = per_bit_conf.min(axis=0)     # (P,) limited by worst bit

        # Codebook from the PATTERN rows only (white + code rows, drop inverses)
        code_matrix = np.vstack([matrix[0:1], matrix[1::2]])
        _, code_int, _ = build_codebook(code_matrix)
    else:
        coded_obs = obs[1:]              # (K, P)
        K = coded_obs.shape[0]
        thresh = reference * 0.5
        bits = (coded_obs > thresh[None, :]).astype(np.uint64)
        any_uncertain = np.zeros(H * W, dtype=bool)
        # Confidence: how far each bit sits from the threshold, worst bit.
        denom = np.maximum(reference, 1e-9)
        per_bit_conf = np.abs(coded_obs - thresh[None, :]) / denom[None, :]
        confidence_flat = per_bit_conf.min(axis=0)
        _, code_int, _ = build_codebook(matrix)

    # Pack bits into an integer code per pixel (MSB = first coded frame).
    weights = (1 << np.arange(K))[::-1].astype(np.uint64)
    pixel_codes = (bits.T @ weights)                        # (P,)

    # Map each pixel's observed code to a cell id via the codebook.
    max_code = int(code_int.max())
    table = np.full(max_code + 1, -1, dtype=np.int64)
    table[code_int.astype(np.int64)] = np.arange(len(code_int))
    safe = np.minimum(pixel_codes.astype(np.int64), max_code)
    code_map_flat = table[safe]                             # -1 if no match

    # Validity
    if complementary:
        valid_flat = (
            (reference >= ref_floor)
            & (code_map_flat >= 0)
            & (~any_uncertain)
        )
    else:
        darkest = obs[1:].min(axis=0)
        contrast = reference - darkest
        valid_flat = (
            (contrast > contrast_frac * np.maximum(reference, 1e-9))
            & (reference >= ref_floor)
            & (code_map_flat >= 0)
        )

    # Invalid pixels get zero confidence.
    confidence_flat = np.where(valid_flat, confidence_flat, 0.0)
    confidence_flat = np.clip(confidence_flat, 0.0, 1.0)

    code_map = code_map_flat.reshape(H, W)
    valid_mask = valid_flat.reshape(H, W)
    confidence = confidence_flat.reshape(H, W)

    if median_clean:
        code_map, valid_mask = _median_clean_codes(code_map, valid_mask)
        confidence = np.where(valid_mask, confidence, 0.0)

    return code_map, valid_mask, confidence


def _median_clean_codes(code_map, valid_mask, ksize=3):
    """3x3 median filter on the code map, applied only where valid. Isolated
    misdecodes differ from their neighbours and get pulled back."""
    cm = code_map.astype(np.float64)
    cm[~valid_mask] = np.nan
    H, W = cm.shape
    pad = ksize // 2
    padded = np.pad(cm, pad, constant_values=np.nan)
    out = code_map.copy()
    stacks = []
    for dy in range(ksize):
        for dx in range(ksize):
            stacks.append(padded[dy:dy + H, dx:dx + W])
    neigh = np.stack(stacks, axis=0)
    with np.errstate(all="ignore"):
        med = np.nanmedian(neigh, axis=0)
    has_med = ~np.isnan(med)
    replace = valid_mask & has_med
    out[replace] = np.round(med[replace]).astype(np.int64)
    return out, valid_mask


def dual_photo(code_map, valid_mask, reference_frame, matrix, grid_meta):
    """Construct the projector's-point-of-view image.

    For each projector cell, average the reference-frame brightness of all
    camera pixels that decoded to that cell, and place it at the cell's [i,j].

    reference_frame: (H, W) the all-white capture (frame 0), grayscale or color.
    Returns a (grid_dim, grid_dim) or (grid_dim, grid_dim, 3) image.
    """
    g = grid_meta["grid_dimensions"]
    n_cells = grid_meta["n_cells"]

    color = reference_frame.ndim == 3
    chans = 3 if color else 1
    ref = reference_frame.reshape(-1, chans).astype(np.float64) if color \
        else reference_frame.reshape(-1, 1).astype(np.float64)

    code_flat = code_map.reshape(-1)
    valid_flat = valid_mask.reshape(-1)

    sums = np.zeros((n_cells, chans))
    counts = np.zeros(n_cells)
    idx = code_flat[valid_flat]
    np.add.at(sums, idx, ref[valid_flat])
    np.add.at(counts, idx, 1)

    nonzero = counts > 0
    sums[nonzero] /= counts[nonzero, None]

    img = sums.reshape(g, g, chans)
    img = img - img.min()
    if img.max() > 0:
        img = img / img.max()
    img = (img * 255).astype(np.uint8)
    return img[..., 0] if not color else img


# --------------------------------------------------------------------------- #
#  Loading + driver (wires the decoder to the capture folder layout)          #
# --------------------------------------------------------------------------- #

def _frame_index(path):
    """Extract the integer i from a '.../frame_<i>.png' path for sorting."""
    name = os.path.splitext(os.path.basename(path))[0]   # frame_<i>
    return int(name.split("_")[-1])


def load_frame_stack(cam_dir, color=False):
    """Load the capture stack from a camera folder.

    Frame files on disk are expected as:
      frame_black.png   (optional) projector-off calibration shot
      frame_0.png       all-white reference (first PATTERN frame)
      frame_1.png ...   coded / code+inverse frames

    If frame_black.png exists it is subtracted from every pattern frame to
    remove ambient + projector dark-level leakage (Sen et al. 3.4).

    Returns:
      gray_stack: (F, H, W) float64 for decoding (black-subtracted), where
                  row 0 is the white reference.
      ref_out:    the white-reference frame for the dual photo (colour if
                  color=True), also black-subtracted.
    """
    # Black calibration frame (optional)
    black_path = os.path.join(cam_dir, "frame_black.png")
    black_gray = None
    if os.path.exists(black_path):
        bimg = cv2.imread(black_path, cv2.IMREAD_COLOR)
        if bimg is not None:
            black_gray = cv2.cvtColor(bimg, cv2.COLOR_BGR2GRAY).astype(np.float64)

    # Pattern frames frame_0..frame_N, sorted numerically
    paths = glob.glob(os.path.join(cam_dir, "frame_*.png"))
    paths = [p for p in paths if "frame_black" not in os.path.basename(p)]
    if not paths:
        raise FileNotFoundError(f"No frame_*.png found in {cam_dir}")
    paths.sort(key=_frame_index)

    gray = []
    ref_out = None
    for k, p in enumerate(paths):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read {p}")
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
        if black_gray is not None:
            g = np.clip(g - black_gray, 0, 255)
        gray.append(g)
        if k == 0:  # the white reference
            if color:
                ref_img = img.astype(np.float64)
                if black_gray is not None:
                    ref_img = np.clip(ref_img - black_gray[..., None], 0, 255)
                ref_out = ref_img.astype(np.uint8)
            else:
                ref_out = g
    return np.stack(gray, axis=0), ref_out


def decode_camera(cam_dir, matrix, grid_meta, contrast_frac=0.2, color=False,
                  ref_floor=30.0, median_clean=False, margin=3.0):
    """Decode one camera folder into a dual photo. Returns
    (dual_img, code_map, valid_mask, confidence)."""
    stack, ref = load_frame_stack(cam_dir, color=color)
    complementary = bool(grid_meta.get("complementary", False))

    if stack.shape[0] != matrix.shape[0]:
        print(f"  WARNING: {stack.shape[0]} pattern frames but matrix has "
              f"{matrix.shape[0]} rows in {cam_dir}. Using min of the two.")
        n = min(stack.shape[0], matrix.shape[0])
        stack, matrix_use = stack[:n], matrix[:n]
    else:
        matrix_use = matrix

    code_map, valid, conf = decode_pixels(
        stack, matrix_use, contrast_frac, ref_floor=ref_floor,
        median_clean=median_clean, complementary=complementary, margin=margin
    )
    dual = dual_photo(code_map, valid, ref, matrix_use, grid_meta)
    return dual, code_map, valid, conf


def find_matrix(patterns_root, run_name, pattern):
    """Locate pattern_matrix.npy / grid_meta.npy saved by the projector."""
    pdir = os.path.join(patterns_root, run_name, pattern)
    mpath = os.path.join(pdir, "pattern_matrix.npy")
    gpath = os.path.join(pdir, "grid_meta.npy")
    if not (os.path.exists(mpath) and os.path.exists(gpath)):
        raise FileNotFoundError(
            f"Could not find pattern_matrix.npy / grid_meta.npy in {pdir}"
        )
    matrix = np.load(mpath)
    grid_meta = np.load(gpath, allow_pickle=True).item()
    return matrix, grid_meta


def run_decode(run_name, pattern, serials=None,
               captures_root="captures", patterns_root="patterns",
               contrast_frac=0.2, color=False, only_serial=None,
               ref_floor=30.0, median_clean=False, margin=3.0):
    """Decode camera(s) in a run and save dual photos.

    Reads frames from  captures_root/<run_name>/<serial>/frame_*.png
    Reads matrix from   patterns_root/<run_name>/<pattern>/pattern_matrix.npy
    Saves dual photos to captures_root/<run_name>/<serial>/dual_photo.png
    Also saves code_map.npy, valid_mask.npy, confidence.npy per camera.

    only_serial: if given, decode just that one camera; otherwise decode all.
    Returns dict {serial: dual_image}.
    """
    matrix, grid_meta = find_matrix(patterns_root, run_name, pattern)
    comp = bool(grid_meta.get("complementary", False))
    print(f"Loaded matrix {matrix.shape}, grid "
          f"{grid_meta['grid_dimensions']}x{grid_meta['grid_dimensions']}, "
          f"complementary={comp}")

    if serials is None:
        serials = CAMERA_SERIALS

    if only_serial is not None:
        if only_serial not in serials:
            print(f"  Note: {only_serial} not in known serials {serials}; "
                  f"decoding it anyway.")
        serials = [only_serial]

    results = {}
    for serial in serials:
        cam_dir = os.path.join(captures_root, run_name, serial)
        if not os.path.isdir(cam_dir):
            print(f"  Skipping {serial}: {cam_dir} not found")
            continue
        print(f"Decoding camera {serial} ...")
        dual, code_map, valid, conf = decode_camera(
            cam_dir, matrix, grid_meta, contrast_frac, color,
            ref_floor=ref_floor, median_clean=median_clean, margin=margin
        )
        out_path = os.path.join(cam_dir, "dual_photo.png")
        cv2.imwrite(out_path, dual)
        np.save(os.path.join(cam_dir, "code_map.npy"), code_map)
        np.save(os.path.join(cam_dir, "valid_mask.npy"), valid)
        np.save(os.path.join(cam_dir, "confidence.npy"), conf)
        print(f"  Saved {out_path}  ({int(valid.sum())} valid pixels)")
        results[serial] = dual
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Decode time-multiplexed binary structured-light captures "
                    "into projector's-point-of-view dual photos."
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--pattern",        default="structured")
    parser.add_argument("--serials",        nargs="+", required=False, default=None)
    parser.add_argument("--serial", default=None,
                        help="Decode only this one serial (must be among --serials, "
                             "or any folder that exists). Omit to decode all.")
    parser.add_argument("--captures-root", default="captures")
    parser.add_argument("--patterns-root", default="patterns")
    parser.add_argument("--contrast-frac", type=float, default=0.2,
                        help="Min contrast (fraction of reference) to trust a pixel.")
    parser.add_argument("--ref-floor", type=float, default=30.0,
                        help="Min absolute reference brightness (0-255) to trust a pixel.")
    parser.add_argument("--margin", type=float, default=3.0,
                        help="Complementary mode: drop bits where |code-inverse| <= margin.")
    parser.add_argument("--no-median", action="store_true",
                        help="Disable 3x3 median cleanup of the code map.")
    parser.add_argument("--color", action="store_true",
                        help="Build a color dual photo from the reference frame.")
    args = parser.parse_args()

    run_decode(
        run_name=args.run_name,
        pattern=args.pattern,
        serials=args.serials,
        captures_root=args.captures_root,
        patterns_root=args.patterns_root,
        contrast_frac=args.contrast_frac,
        color=args.color,
        only_serial=args.serial,
        ref_floor=args.ref_floor,
        median_clean=not args.no_median,
        margin=args.margin,
    )