"""
Fuse the two cameras' dual photos into a single, better projector's-eye view.

Both cameras decode into the SAME projector cell grid, so fusion needs no
alignment or calibration -- the images are already in the same coordinate
space. We produce two outputs:

  fused_clean.png  -- confidence-weighted combination. Each cell is taken
                      from whichever camera decoded it more reliably; cells
                      both saw well are averaged (which reduces noise). This
                      fills each camera's dropouts with the other's good data
                      and yields one clean projector-POV image.

  fused_sum.png    -- the two views added together. In dual photography each
                      camera acts as a virtual light source, so summing the
                      two dual photos simulates lighting the scene with BOTH
                      virtual lights at once (a relighting-style result).

Inputs per camera (written by reconstruction_control.run_decode):
  captures/<run>/<serial>/dual_photo.png   (normalized 0-255 dual photo)
  captures/<run>/<serial>/confidence.npy   (per-cell-pixel 0..1 reliability)

NOTE on confidence resolution: confidence.npy is per CAMERA PIXEL (H x W),
while the dual photo is per CELL (grid x grid). We reduce the pixel-confidence
to a per-cell confidence using the saved code_map (averaging the confidence of
all camera pixels that decoded to each cell). If code_map is missing we fall
back to a flat confidence.
"""

import os
import argparse
import numpy as np
import cv2


def _per_cell_confidence(cam_dir, grid_dim, n_cells):
    """Reduce per-pixel confidence to per-cell by averaging over the pixels
    that decoded to each cell (via code_map). Returns (grid, grid) in 0..1."""
    conf_path = os.path.join(cam_dir, "confidence.npy")
    code_path = os.path.join(cam_dir, "code_map.npy")
    valid_path = os.path.join(cam_dir, "valid_mask.npy")

    if not (os.path.exists(conf_path) and os.path.exists(code_path)):
        # Fall back to uniform confidence if we can't compute per-cell.
        return np.ones((grid_dim, grid_dim), dtype=np.float64)

    conf = np.load(conf_path).astype(np.float64).reshape(-1)
    code = np.load(code_path).astype(np.int64).reshape(-1)
    if os.path.exists(valid_path):
        valid = np.load(valid_path).astype(bool).reshape(-1)
    else:
        valid = code >= 0

    sums = np.zeros(n_cells)
    counts = np.zeros(n_cells)
    sel = valid & (code >= 0) & (code < n_cells)
    np.add.at(sums, code[sel], conf[sel])
    np.add.at(counts, code[sel], 1)
    cell_conf = np.zeros(n_cells)
    nz = counts > 0
    cell_conf[nz] = sums[nz] / counts[nz]
    return cell_conf.reshape(grid_dim, grid_dim)


def _load_dual(cam_dir):
    """Load a camera's dual photo as float64. Returns (img, is_color)."""
    path = os.path.join(cam_dir, "dual_photo.png")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No dual_photo.png in {cam_dir}")
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read {path}")
    is_color = img.ndim == 3
    return img.astype(np.float64), is_color


def fuse(run_name, serials, captures_root="captures"):
    """Produce fused_clean.png and fused_sum.png for a run."""
    if len(serials) < 2:
        raise ValueError("Need two camera serials to fuse.")

    cam_dirs = [os.path.join(captures_root, run_name, s) for s in serials]
    for d in cam_dirs:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Camera folder not found: {d}")

    dualA, colorA = _load_dual(cam_dirs[0])
    dualB, colorB = _load_dual(cam_dirs[1])
    if dualA.shape != dualB.shape:
        raise ValueError(f"Dual photos differ in shape: {dualA.shape} vs {dualB.shape}")

    grid_dim = dualA.shape[0]
    n_cells = grid_dim * grid_dim
    is_color = colorA and colorB

    confA = _per_cell_confidence(cam_dirs[0], grid_dim, n_cells)
    confB = _per_cell_confidence(cam_dirs[1], grid_dim, n_cells)

    if is_color:
        confA = confA[..., None]
        confB = confB[..., None]

    # ---- Confidence-weighted clean fusion ----
    denom = confA + confB
    safe = denom > 1e-9
    fused_clean = np.zeros_like(dualA)
    wsum = confA * dualA + confB * dualB
    fused_clean = np.where(safe, wsum / np.where(safe, denom, 1.0), 0.0)
    fused_clean = np.clip(fused_clean, 0, 255).astype(np.uint8)

    # ---- Two-light sum ----
    # Add the two virtual-light views, then renormalize to 0-255 so the
    # brighter combined result doesn't clip everything.
    summed = dualA + dualB
    m = summed.max()
    if m > 0:
        summed = summed / m * 255.0
    fused_sum = np.clip(summed, 0, 255).astype(np.uint8)

    out_dir = os.path.join(captures_root, run_name)
    clean_path = os.path.join(out_dir, "fused_clean.png")
    sum_path = os.path.join(out_dir, "fused_sum.png")
    cv2.imwrite(clean_path, fused_clean)
    cv2.imwrite(sum_path, fused_sum)

    # Report coverage gain
    covA = (confA > 0).sum()
    covB = (confB > 0).sum()
    covFused = (denom > 1e-9).sum()
    print(f"Fused {serials[0]} + {serials[1]}")
    print(f"  cells covered:  A={int(covA)}  B={int(covB)}  fused={int(covFused)} "
          f"of {n_cells}")
    print(f"  saved {clean_path}")
    print(f"  saved {sum_path}")
    return fused_clean, fused_sum


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fuse two cameras' dual photos into a cleaner POV and a "
                    "two-light sum."
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--serials", nargs="+", required=True,
                        help="Two camera serial folder names to fuse.")
    parser.add_argument("--captures-root", default="captures")
    args = parser.parse_args()

    fuse(args.run_name, args.serials, captures_root=args.captures_root)