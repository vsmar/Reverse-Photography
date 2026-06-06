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


def decode_pixels(stack, matrix, contrast_frac=0.2):
    """Assign each camera pixel the projector cell that illuminates it.

    stack:          (F, H, W) grayscale frames in the SAME order as `matrix`
                    rows (frame 0 = all-white reference).
    matrix:         (F, C) saved pattern matrix.
    contrast_frac:  a pixel is valid only if (reference - darkest) exceeds
                    this fraction of the reference brightness; filters
                    background/shadow that never gets lit.

    Returns (code_map, valid_mask), both (H, W). code_map holds the cell id;
    valid_mask is True where the decode is trustworthy.
    """
    F, H, W = stack.shape
    obs = stack.reshape(F, H * W).astype(np.float64)  # (F, P)

    reference = obs[0]               # all-white frame: per-pixel "fully lit"
    coded_obs = obs[1:]              # (K, P) the coded frames
    K = coded_obs.shape[0]

    # Bit = 1 where the pixel is brighter than half its reference brightness.
    # Using reference/2 as a per-pixel threshold handles uneven illumination
    # far better than one global threshold.
    thresh = reference * 0.5
    bits = (coded_obs > thresh[None, :]).astype(np.uint64)  # (K, P)

    # Pack bits into an integer code per pixel (MSB = first coded frame),
    # matching build_codebook's weighting.
    weights = (1 << np.arange(K))[::-1].astype(np.uint64)
    pixel_codes = (bits.T @ weights)                        # (P,)

    # Map each pixel's observed code to a cell id via the codebook.
    _, code_int, lut = build_codebook(matrix)
    # Vectorized lookup: build a dense table indexed by code value.
    max_code = int(code_int.max())
    table = np.full(max_code + 1, -1, dtype=np.int64)
    table[code_int.astype(np.int64)] = np.arange(len(code_int))
    safe = np.minimum(pixel_codes.astype(np.int64), max_code)
    code_map_flat = table[safe]                             # -1 if no match

    # Validity: enough contrast AND the observed code actually exists.
    darkest = coded_obs.min(axis=0)
    contrast = reference - darkest
    valid_flat = (contrast > contrast_frac * np.maximum(reference, 1e-9)) \
        & (code_map_flat >= 0)

    return code_map_flat.reshape(H, W), valid_flat.reshape(H, W)


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
    """Load frame_0.png, frame_1.png, ... from a camera folder, sorted
    numerically (so frame_2 precedes frame_10). Returns:
      gray_stack: (F, H, W) float64 for decoding
      ref_color:  (H, W) or (H, W, 3) the reference frame (index 0) for the
                  dual photo, in color if color=True.
    """
    paths = glob.glob(os.path.join(cam_dir, "frame_*.png"))
    if not paths:
        raise FileNotFoundError(f"No frame_*.png found in {cam_dir}")
    paths.sort(key=_frame_index)

    gray = []
    ref_color = None
    for k, p in enumerate(paths):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read {p}")
        gray.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64))
        if k == 0:
            ref_color = img if color else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.stack(gray, axis=0), ref_color


def decode_camera(cam_dir, matrix, grid_meta, contrast_frac=0.2, color=False):
    """Decode one camera folder into a dual photo. Returns (dual_img,
    code_map, valid_mask)."""
    stack, ref = load_frame_stack(cam_dir, color=color)

    if stack.shape[0] != matrix.shape[0]:
        print(f"  WARNING: {stack.shape[0]} frames but matrix has "
              f"{matrix.shape[0]} rows in {cam_dir}. Using min of the two.")
        n = min(stack.shape[0], matrix.shape[0])
        stack, matrix_use = stack[:n], matrix[:n]
    else:
        matrix_use = matrix

    code_map, valid = decode_pixels(stack, matrix_use, contrast_frac)
    dual = dual_photo(code_map, valid, ref, matrix_use, grid_meta)
    return dual, code_map, valid


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


def run_decode(run_name, pattern, serials=FileNotFoundError,
               captures_root="captures", patterns_root="patterns",
               contrast_frac=0.2, color=False, only_serial=None):
    """Decode camera(s) in a run and save dual photos.

    Reads frames from  captures_root/<run_name>/<serial>/frame_*.png
    Reads matrix from   patterns_root/<run_name>/<pattern>/pattern_matrix.npy
    Saves dual photos to captures_root/<run_name>/<serial>/dual_photo.png

    only_serial: if given, decode just that one camera; otherwise decode all
                 serials in the list.

    Returns dict {serial: dual_image}.
    """
    matrix, grid_meta = find_matrix(patterns_root, run_name, pattern)
    print(f"Loaded matrix {matrix.shape}, grid "
          f"{grid_meta['grid_dimensions']}x{grid_meta['grid_dimensions']}")
    
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
        dual, code_map, valid = decode_camera(
            cam_dir, matrix, grid_meta, contrast_frac, color
        )
        out_path = os.path.join(cam_dir, "dual_photo.png")
        cv2.imwrite(out_path, dual)
        # Save the raw correspondence too (useful for later 3D / fusion)
        np.save(os.path.join(cam_dir, "code_map.npy"), code_map)
        np.save(os.path.join(cam_dir, "valid_mask.npy"), valid)
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
    )