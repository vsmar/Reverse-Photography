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
  decode_pixels(stack, matrix) -> (code_map, valid_mask, confidence)
  dual_photo(code_map, valid_mask, reference, matrix, confidence)

Weighting in dual_photo combines three independent factors:
  w_confidence : how far each bit-read was from the ambiguous midpoint,
                 averaged over all K coded frames -- high when every bit
                 was a clear 0 or 1, low when reads were near 50 % of ref.
  w_brightness : reference-frame brightness, down-weighted by brightness_pow
                 (< 1 compresses dynamic range so bright pixels don't dominate).
  w_locality   : Gaussian falloff from the centroid of all camera pixels that
                 decoded to the same projector cell -- suppresses outlier
                 pixels that may be misreads (specular hits, pattern edges).
"""

import os
import glob
import argparse
import numpy as np
import cv2
from camera_controls import CameraController as Cam, CAMERA_SERIALS, OUTPUT_DIR


# --------------------------------------------------------------------------- #
#  Codebook                                                                    #
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
#  Pixel decoding with per-pixel confidence scoring                            #
# --------------------------------------------------------------------------- #

def decode_pixels(stack, matrix, contrast_frac=0.2):
    """Assign each camera pixel the projector cell that illuminates it.

    stack:          (F, H, W) grayscale frames in the SAME order as `matrix`
                    rows (frame 0 = all-white reference).
    matrix:         (F, C) saved pattern matrix.
    contrast_frac:  a pixel is valid only if (reference - darkest) exceeds
                    this fraction of the reference brightness.

    Returns
    -------
    code_map   : (H, W) int64  -- projector cell id per pixel (-1 if invalid)
    valid_mask : (H, W) bool   -- True where the decode is trustworthy
    confidence : (H, W) float64 in [0, 1]
                 Mean per-frame confidence: how far each bit-read was from the
                 ambiguous midpoint (reference/2), normalised by reference/2.
                 1.0 = every bit perfectly clear; 0.0 = every bit ambiguous.
                 Only meaningful where valid_mask is True.
    """
    F, H, W = stack.shape
    obs = stack.reshape(F, H * W).astype(np.float64)  # (F, P)

    reference = obs[0]               # all-white frame: per-pixel "fully lit"
    coded_obs = obs[1:]              # (K, P) the coded frames
    K = coded_obs.shape[0]

    # Per-pixel, per-frame threshold: half the reference brightness.
    thresh = reference * 0.5         # (P,)

    # Hard bit decision (same as before).
    bits = (coded_obs > thresh[None, :]).astype(np.uint64)  # (K, P)

    # --- Confidence scoring ---------------------------------------------------
    # For each frame and pixel, measure how far the observed brightness is from
    # the decision boundary (thresh).  Normalise by thresh so the score is
    # scale-independent:
    #
    #   frame_conf[k, p] = |coded_obs[k,p] - thresh[p]| / max(thresh[p], 1e-9)
    #
    # A pixel sitting right at the boundary scores 0 (maximally ambiguous).
    # A pixel at 0 or at reference scores 1 (maximally confident).
    # Average over all K frames to get one scalar per pixel.
    safe_thresh = np.maximum(thresh, 1e-9)
    frame_conf = np.abs(coded_obs - thresh[None, :]) / safe_thresh[None, :]
    frame_conf = np.clip(frame_conf, 0.0, 1.0)       # (K, P)
    confidence_flat = frame_conf.mean(axis=0)         # (P,)  in [0, 1]

    # Pack bits into integer codes (MSB = first coded frame).
    weights = (1 << np.arange(K))[::-1].astype(np.uint64)
    pixel_codes = (bits.T @ weights)                  # (P,)

    # Map each pixel's observed code to a cell id via the codebook.
    _, code_int, lut = build_codebook(matrix)
    max_code = int(code_int.max())
    table = np.full(max_code + 1, -1, dtype=np.int64)
    table[code_int.astype(np.int64)] = np.arange(len(code_int))
    safe = np.minimum(pixel_codes.astype(np.int64), max_code)
    code_map_flat = table[safe]                       # -1 if no match

    # Validity: enough contrast AND the observed code actually exists.
    darkest = coded_obs.min(axis=0)
    contrast = reference - darkest
    valid_flat = (contrast > contrast_frac * np.maximum(reference, 1e-9)) \
        & (code_map_flat >= 0)

    return (
        code_map_flat.reshape(H, W),
        valid_flat.reshape(H, W),
        confidence_flat.reshape(H, W),
    )


# --------------------------------------------------------------------------- #
#  Weighted dual-photo reconstruction                                          #
# --------------------------------------------------------------------------- #

def dual_photo(code_map, valid_mask, reference_frame, matrix, grid_meta,
               confidence=None,
               sigma=None,
               brightness_pow=1.0):
    """Construct the projector's-point-of-view image with weighted averaging.

    For each projector cell the final pixel value is a weighted average of the
    reference-frame brightness of all valid camera pixels that decoded to it:

        value[cell] = Σ (w_conf * w_bright * w_local * brightness)
                    / Σ (w_conf * w_bright * w_local)

    Parameters
    ----------
    code_map        : (H, W) int64   -- output of decode_pixels
    valid_mask      : (H, W) bool
    reference_frame : (H, W) or (H, W, 3)
    matrix          : (F, C) pattern matrix (used only for grid_meta fallback)
    grid_meta       : dict with 'grid_dimensions' and 'n_cells'
    confidence      : (H, W) float64 in [0,1] from decode_pixels, or None
                      (None disables confidence weighting → uniform weight 1)
    sigma           : float, pixels.  Gaussian locality kernel width.
                      None disables locality weighting.
                      A reasonable starting value is ~5-15 camera pixels.
    brightness_pow  : float ≥ 0.  Reference brightness is raised to this power
                      before being used as a weight.
                      1.0 = linear weighting by brightness.
                      0.5 = square-root (compresses dynamic range).
                      0.0 = disabled (uniform brightness weight).

    Returns
    -------
    (grid_dim, grid_dim) uint8  or  (grid_dim, grid_dim, 3) uint8
    """
    g = grid_meta["grid_dimensions"]
    n_cells = grid_meta["n_cells"]

    color = reference_frame.ndim == 3
    chans = 3 if color else 1
    ref = (reference_frame.reshape(-1, chans).astype(np.float64)
           if color else reference_frame.reshape(-1, 1).astype(np.float64))

    H, W = code_map.shape
    code_flat = code_map.reshape(-1)
    valid_flat = valid_mask.reshape(-1)

    # Pixel row/col coordinates (needed for locality weighting).
    rows, cols = np.mgrid[0:H, 0:W]
    rows_flat = rows.reshape(-1).astype(np.float64)
    cols_flat = cols.reshape(-1).astype(np.float64)

    # --- Base weight: confidence ---------------------------------------------
    if confidence is not None:
        w_conf = confidence.reshape(-1)          # (P,) in [0, 1]
    else:
        w_conf = np.ones(H * W, dtype=np.float64)

    # --- Brightness weight ---------------------------------------------------
    ref_gray = reference_frame if not color \
        else cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
    ref_gray_flat = ref_gray.reshape(-1).astype(np.float64)
    if brightness_pow > 0:
        safe_ref = np.maximum(ref_gray_flat, 1e-9)
        w_bright = (safe_ref / safe_ref.max()) ** brightness_pow
    else:
        w_bright = np.ones(H * W, dtype=np.float64)

    # Combined weight before locality (locality needs a two-pass approach).
    w_pre = w_conf * w_bright                    # (P,)

    # Select valid pixels only for efficiency.
    v_idx    = np.where(valid_flat)[0]
    v_cells  = code_flat[v_idx]
    v_ref    = ref[v_idx]                        # (V, chans)
    v_rows   = rows_flat[v_idx]
    v_cols   = cols_flat[v_idx]
    v_wpre   = w_pre[v_idx]                      # (V,)

    # --- Locality weighting (two-pass) ---------------------------------------
    if sigma is not None and sigma > 0:
        # Pass 1: compute centroid of camera-pixel coordinates per cell.
        centroid_row = np.zeros(n_cells)
        centroid_col = np.zeros(n_cells)
        centroid_w   = np.zeros(n_cells)
        np.add.at(centroid_row, v_cells, v_rows * v_wpre)
        np.add.at(centroid_col, v_cells, v_cols * v_wpre)
        np.add.at(centroid_w,   v_cells, v_wpre)
        hit = centroid_w > 0
        centroid_row[hit] /= centroid_w[hit]
        centroid_col[hit] /= centroid_w[hit]

        # Pass 2: Gaussian weight from each pixel to its cell's centroid.
        dr = v_rows - centroid_row[v_cells]
        dc = v_cols - centroid_col[v_cells]
        dist2 = dr ** 2 + dc ** 2
        w_local = np.exp(-dist2 / (2.0 * sigma ** 2))
    else:
        w_local = np.ones(len(v_idx), dtype=np.float64)

    # Final combined weight.
    w_total = v_wpre * w_local                   # (V,)

    # --- Weighted accumulation -----------------------------------------------
    sums   = np.zeros((n_cells, chans))
    counts = np.zeros(n_cells)
    np.add.at(sums,   v_cells, v_ref * w_total[:, None])
    np.add.at(counts, v_cells, w_total)

    nonzero = counts > 0
    sums[nonzero] /= counts[nonzero, None]

    img = sums.reshape(g, g, chans)
    img = img - img.min()
    if img.max() > 0:
        img = img / img.max()
    img = (img * 255).astype(np.uint8)
    return img[..., 0] if not color else img


# --------------------------------------------------------------------------- #
#  Loading + driver                                                            #
# --------------------------------------------------------------------------- #

def _frame_index(path):
    name = os.path.splitext(os.path.basename(path))[0]
    return int(name.split("_")[-1])


def load_frame_stack(cam_dir, color=False):
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


def decode_camera(cam_dir, matrix, grid_meta,
                  contrast_frac=0.2, color=False,
                  sigma=None, brightness_pow=1.0):
    """Decode one camera folder into a weighted dual photo."""
    stack, ref = load_frame_stack(cam_dir, color=color)

    if stack.shape[0] != matrix.shape[0]:
        print(f"  WARNING: {stack.shape[0]} frames but matrix has "
              f"{matrix.shape[0]} rows in {cam_dir}. Using min.")
        n = min(stack.shape[0], matrix.shape[0])
        stack, matrix = stack[:n], matrix[:n]

    code_map, valid, confidence = decode_pixels(stack, matrix, contrast_frac)
    dual = dual_photo(
        code_map, valid, ref, matrix, grid_meta,
        confidence=confidence,
        sigma=sigma,
        brightness_pow=brightness_pow,
    )
    return dual, code_map, valid, confidence


def find_matrix(patterns_root, run_name, pattern):
    pdir = os.path.join(patterns_root, run_name, pattern)
    mpath = os.path.join(pdir, "pattern_matrix.npy")
    gpath = os.path.join(pdir, "grid_meta.npy")
    if not (os.path.exists(mpath) and os.path.exists(gpath)):
        raise FileNotFoundError(
            f"Could not find pattern_matrix.npy / grid_meta.npy in {pdir}"
        )
    matrix   = np.load(mpath)
    grid_meta = np.load(gpath, allow_pickle=True).item()
    return matrix, grid_meta


def run_decode(run_name, pattern, serials=None,
               captures_root="captures", patterns_root="patterns",
               contrast_frac=0.2, color=False,
               sigma=None, brightness_pow=1.0,
               only_serial=None):
    """Decode camera(s) in a run and save weighted dual photos."""
    matrix, grid_meta = find_matrix(patterns_root, run_name, pattern)
    print(f"Loaded matrix {matrix.shape}, grid "
          f"{grid_meta['grid_dimensions']}x{grid_meta['grid_dimensions']}")
    print(f"Weights: confidence=ON  brightness_pow={brightness_pow}"
          f"  locality sigma={sigma}")

    if serials is None:
        serials = CAMERA_SERIALS

    if only_serial is not None:
        serials = [only_serial]

    results = {}
    for serial in serials:
        cam_dir = os.path.join(captures_root, run_name, serial)
        if not os.path.isdir(cam_dir):
            print(f"  Skipping {serial}: {cam_dir} not found")
            continue
        print(f"Decoding camera {serial} ...")
        dual, code_map, valid, confidence = decode_camera(
            cam_dir, matrix, grid_meta,
            contrast_frac=contrast_frac,
            color=color,
            sigma=sigma,
            brightness_pow=brightness_pow,
        )
        out_path = os.path.join(cam_dir, "dual_photo.png")
        cv2.imwrite(out_path, dual)
        np.save(os.path.join(cam_dir, "code_map.npy"),    code_map)
        np.save(os.path.join(cam_dir, "valid_mask.npy"),  valid)
        np.save(os.path.join(cam_dir, "confidence.npy"),  confidence)
        print(f"  Saved {out_path}  ({int(valid.sum())} valid pixels, "
              f"mean confidence {confidence[valid].mean():.3f})")
        results[serial] = dual
    return results


# --------------------------------------------------------------------------- #
#  CLI                                                                         #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Decode time-multiplexed binary structured-light captures "
                    "into projector's-point-of-view dual photos."
    )
    parser.add_argument("--run-name",       required=True)
    parser.add_argument("--pattern",        default="structured")
    parser.add_argument("--serials",        nargs="+", required=False, default=None)
    parser.add_argument("--serial",         default=None)
    parser.add_argument("--captures-root",  default="captures")
    parser.add_argument("--patterns-root",  default="patterns")
    parser.add_argument("--contrast-frac",  type=float, default=0.2)
    parser.add_argument("--color",          action="store_true")
    parser.add_argument("--sigma",          type=float, default=None,
                        help="Gaussian locality kernel width in camera pixels. "
                             "Omit to disable locality weighting.")
    parser.add_argument("--brightness-pow", type=float, default=1.0,
                        help="Brightness weight exponent. "
                             "1.0=linear, 0.5=sqrt, 0.0=disabled.")
    args = parser.parse_args()

    run_decode(
        run_name=args.run_name,
        pattern=args.pattern,
        serials=args.serials,
        captures_root=args.captures_root,
        patterns_root=args.patterns_root,
        contrast_frac=args.contrast_frac,
        color=args.color,
        sigma=args.sigma,
        brightness_pow=args.brightness_pow,
        only_serial=args.serial,
    )