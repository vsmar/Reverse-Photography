import argparse
import os

import cv2
import numpy as np
from decode import _norm_u8, load_run


def _load_dual(cam_dir):
    path = os.path.join(cam_dir, "dual_photo.png")
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read {path}")
    return img.astype(np.float64)


def _cell_confidence(cam_dir, meta):
    conf_path = os.path.join(cam_dir, "confidence.npy")
    code_path = os.path.join(cam_dir, "code_map.npy")
    valid_path = os.path.join(cam_dir, "valid_mask.npy")

    g = int(meta["grid_dimensions"])
    n_cells = int(meta["n_cells"])

    if not (os.path.exists(conf_path) and os.path.exists(code_path)):
        return np.ones((g, g), dtype=np.float64)

    conf = np.load(conf_path).astype(np.float64).reshape(-1)
    code = np.load(code_path).astype(np.int64).reshape(-1)
    valid = np.load(valid_path).astype(bool).reshape(-1) if os.path.exists(valid_path) else code >= 0

    keep = valid & (code >= 0) & (code < n_cells)
    sums = np.zeros(n_cells, dtype=np.float64)
    counts = np.zeros(n_cells, dtype=np.float64)

    np.add.at(sums, code[keep], conf[keep])
    np.add.at(counts, code[keep], 1)

    out = np.zeros(n_cells, dtype=np.float64)
    seen = counts > 0
    out[seen] = sums[seen] / counts[seen]
    return out.reshape(g, g)


def _match_channels(conf, dual):
    return conf[..., None] if dual.ndim == 3 else conf


def fuse(run_name, captures_root="captures"):
    run_dir, _, meta = load_run(run_name, captures_root)
    serials = list(meta["camera_serials"])

    if len(serials) < 2:
        raise ValueError("metadata.json must list at least two camera_serials to fuse.")

    duals = []
    confs = []
    used_serials = []

    for serial in serials:
        cam_dir = os.path.join(run_dir, serial)
        if not os.path.isdir(cam_dir):
            print(f"  Skipping {serial}: missing folder")
            continue

        dual = _load_dual(cam_dir)
        conf = _cell_confidence(cam_dir, meta)

        if dual.shape[:2] != conf.shape:
            raise ValueError(f"Shape mismatch for {serial}: dual={dual.shape}, conf={conf.shape}")

        duals.append(dual)
        confs.append(_match_channels(conf, dual))
        used_serials.append(serial)

    if len(duals) < 2:
        raise ValueError("Need at least two decoded camera folders with dual_photo.png.")

    first_shape = duals[0].shape
    if any(d.shape != first_shape for d in duals):
        raise ValueError("All dual photos must have the same shape.")

    weighted_sum = np.zeros_like(duals[0], dtype=np.float64)
    weight_sum = np.zeros_like(confs[0], dtype=np.float64)
    relight_sum = np.zeros_like(duals[0], dtype=np.float64)

    for dual, conf in zip(duals, confs):
        weighted_sum += dual * conf
        weight_sum += conf
        relight_sum += dual

    fused_clean = np.where(weight_sum > 1e-9, weighted_sum / np.maximum(weight_sum, 1e-9), 0)
    fused_clean = np.clip(fused_clean, 0, 255).astype(np.uint8)
    fused_sum = _norm_u8(relight_sum)

    clean_path = os.path.join(run_dir, "fused_clean.png")
    sum_path = os.path.join(run_dir, "fused_sum.png")
    cv2.imwrite(clean_path, fused_clean)
    cv2.imwrite(sum_path, fused_sum)

    covered = (np.squeeze(weight_sum) > 1e-9).sum()
    print(f"Fused {len(used_serials)} cameras from metadata: {', '.join(used_serials)}")
    print(f"  cells covered: {int(covered)} of {int(meta['n_cells'])}")
    print(f"  saved {clean_path}")
    print(f"  saved {sum_path}")

    return fused_clean, fused_sum


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--captures-root", default="captures")
    args = parser.parse_args()

    fuse(run_name=args.run_name, captures_root=args.captures_root)
