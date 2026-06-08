import os
import glob
import json
import argparse
import numpy as np
import cv2


def _frame_index(path):
    name = os.path.splitext(os.path.basename(path))[0]
    return int(name.split("_")[-1])


def _norm_u8(img):
    arr = img.astype(np.float32)
    arr = arr - np.nanmin(arr)
    mx = np.nanmax(arr)
    if mx > 0:
        arr = arr * 255.0 / mx
    return np.clip(arr, 0, 255).astype(np.uint8)


def _load_meta(run_dir):
    with open(os.path.join(run_dir, "metadata.json"), "r") as f:
        return json.load(f)


def _add_derived_meta(meta, matrix):
    n_cells = int(matrix.shape[1])
    grid_dimensions = int(np.sqrt(n_cells))
    if grid_dimensions * grid_dimensions != n_cells:
        raise ValueError(f"Matrix has {n_cells} cells, not a square grid")

    out = dict(meta)
    out["n_cells"] = n_cells
    out["grid_dimensions"] = grid_dimensions
    out["num_base_patterns"] = int(matrix.shape[0])
    return out


def load_run(run_name, captures_root="captures"):
    run_dir = os.path.join(captures_root, run_name)

    meta = _load_meta(run_dir)
    if "camera_serials" not in meta:
        raise KeyError(f"metadata.json in {run_dir} is missing camera_serials")

    matrix_path = os.path.join(run_dir, "pattern_matrix.npy")
    if not os.path.exists(matrix_path):
        raise FileNotFoundError(f"Missing pattern_matrix.npy in {run_dir}")

    matrix = np.load(matrix_path)
    return run_dir, matrix, _add_derived_meta(meta, matrix)


def build_codebook(matrix):
    coded = matrix[1:].astype(np.uint8)
    K, _ = coded.shape
    weights = (1 << np.arange(K))[::-1].astype(np.uint64)
    return coded.T.astype(np.uint64) @ weights


def load_processed_stack(cam_dir, color=False):
    proc_dir = os.path.join(cam_dir, "processed")
    search_dir = proc_dir if os.path.isdir(proc_dir) else cam_dir

    paths = glob.glob(os.path.join(search_dir, "frame_*.png"))
    paths = [p for p in paths if "frame_black" not in os.path.basename(p)]
    if not paths:
        raise FileNotFoundError(f"No frame_*.png found in {search_dir}")

    paths.sort(key=_frame_index)

    gray = []
    ref = None
    for k, path in enumerate(paths):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)

        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
        gray.append(g)

        if k == 0:
            ref = img if color else g

    return np.stack(gray, axis=0), ref


def decode_pixels(stack, matrix, meta, contrast_frac=0.2, ref_floor=30.0,
                  median_clean=True):
    F, H, W = stack.shape

    if F != matrix.shape[0]:
        n = min(F, matrix.shape[0])
        print(f"  WARNING: {F} frames, {matrix.shape[0]} matrix rows; using {n}")
        stack = stack[:n]
        matrix = matrix[:n]

    obs = stack.reshape(stack.shape[0], -1).astype(np.float64)
    ref = obs[0]
    coded_obs = obs[1:]
    K = coded_obs.shape[0]

    if meta.get("complement", False):
        bits = (coded_obs > 127).astype(np.uint64)
        confidence = np.ones(obs.shape[1], dtype=np.float64)
        valid = ref >= ref_floor
    else:
        thresh = 0.5 * ref
        bits = (coded_obs > thresh[None, :]).astype(np.uint64)
        per_bit_conf = np.abs(coded_obs - thresh[None, :]) / np.maximum(ref[None, :], 1e-9)
        confidence = per_bit_conf.min(axis=0)
        contrast = ref - obs[1:].min(axis=0)
        valid = (contrast > contrast_frac * np.maximum(ref, 1e-9)) & (ref >= ref_floor)

    weights = (1 << np.arange(K))[::-1].astype(np.uint64)
    pixel_codes = bits.T @ weights

    code_int = build_codebook(matrix)
    max_code = int(max(code_int.max(), pixel_codes.max(initial=0)))
    table = np.full(max_code + 1, -1, dtype=np.int64)
    table[code_int.astype(np.int64)] = np.arange(len(code_int))

    code_map = table[pixel_codes.astype(np.int64)]
    valid &= code_map >= 0

    code_map = code_map.reshape(H, W)
    valid_mask = valid.reshape(H, W)
    confidence = np.where(valid, confidence, 0.0).reshape(H, W)

    if median_clean:
        code_map, valid_mask = median_clean_codes(code_map, valid_mask)
        confidence = np.where(valid_mask, confidence, 0.0)

    return code_map, valid_mask, np.clip(confidence, 0.0, 1.0)


def median_clean_codes(code_map, valid_mask, ksize=3):
    cm = code_map.astype(np.float64)
    cm[~valid_mask] = np.nan

    H, W = cm.shape
    pad = ksize // 2
    padded = np.pad(cm, pad, constant_values=np.nan)
    neigh = [padded[y:y + H, x:x + W] for y in range(ksize) for x in range(ksize)]

    with np.errstate(all="ignore"):
        med = np.nanmedian(np.stack(neigh, axis=0), axis=0)

    out = code_map.copy()
    replace = valid_mask & ~np.isnan(med)
    out[replace] = np.round(med[replace]).astype(np.int64)
    return out, valid_mask


def dual_photo(code_map, valid_mask, reference_frame, meta):
    g = int(meta["grid_dimensions"])
    n_cells = int(meta["n_cells"])

    color = reference_frame.ndim == 3
    chans = 3 if color else 1
    ref = reference_frame.reshape(-1, chans).astype(np.float64) if color else reference_frame.reshape(-1, 1)

    valid_flat = valid_mask.reshape(-1)
    idx = code_map.reshape(-1)[valid_flat]

    sums = np.zeros((n_cells, chans), dtype=np.float64)
    counts = np.zeros(n_cells, dtype=np.float64)
    np.add.at(sums, idx, ref[valid_flat])
    np.add.at(counts, idx, 1)

    hit = counts > 0
    sums[hit] /= counts[hit, None]

    img = sums.reshape(g, g, chans)
    return _norm_u8(img[..., 0] if not color else img)


def code_map_color(code_map, valid_mask, meta):
    g = int(meta["grid_dimensions"])
    y = np.clip(code_map // g, 0, g - 1)
    x = np.clip(code_map % g, 0, g - 1)

    b = (255 * x / max(g - 1, 1)).astype(np.uint8)
    gr = (255 * y / max(g - 1, 1)).astype(np.uint8)
    r = np.where(valid_mask, 255, 0).astype(np.uint8)

    out = np.dstack([b, gr, r])
    out[~valid_mask] = 0
    return out


def coverage_map(code_map, valid_mask, meta):
    g = int(meta["grid_dimensions"])
    n_cells = int(meta["n_cells"])

    counts = np.zeros(n_cells, dtype=np.float64)
    np.add.at(counts, code_map.reshape(-1)[valid_mask.reshape(-1)], 1)
    return _norm_u8(counts.reshape(g, g))


def save_delta_overview(cam_dir, meta):
    if not meta.get("complement", False):
        return

    proc_dir = os.path.join(cam_dir, "processed")
    delta_paths = sorted(glob.glob(os.path.join(proc_dir, "delta_*.npy")), key=_frame_index)
    if not delta_paths:
        return

    sample = np.load(delta_paths[0])
    min_delta = np.full_like(sample, np.inf, dtype=np.float32)
    max_delta = np.full_like(sample, -np.inf, dtype=np.float32)
    min_abs = np.full_like(sample, np.inf, dtype=np.float32)

    for path in delta_paths:
        d = np.load(path).astype(np.float32)
        min_delta = np.minimum(min_delta, d)
        max_delta = np.maximum(max_delta, d)
        min_abs = np.minimum(min_abs, np.abs(d))

    cv2.imwrite(os.path.join(proc_dir, "delta_min.png"), _norm_u8(min_delta))
    cv2.imwrite(os.path.join(proc_dir, "delta_max.png"), _norm_u8(max_delta))
    cv2.imwrite(os.path.join(proc_dir, "delta_min_abs.png"), _norm_u8(min_abs))


def decode_camera(cam_dir, matrix, meta, contrast_frac=0.2, color=False,
                  ref_floor=30.0, median_clean=True):
    stack, ref = load_processed_stack(cam_dir, color=color)
    code_map, valid, conf = decode_pixels(
        stack,
        matrix,
        meta,
        contrast_frac=contrast_frac,
        ref_floor=ref_floor,
        median_clean=median_clean,
    )
    dual = dual_photo(code_map, valid, ref, meta)
    return dual, code_map, valid, conf


def run_decode(run_name, captures_root="captures", serials=None,
               contrast_frac=0.2, color=False, only_serial=None,
               ref_floor=30.0, median_clean=True):
    run_dir, matrix, meta = load_run(run_name, captures_root)

    if serials is None:
        serials = list(meta["camera_serials"])
    if only_serial is not None:
        serials = [only_serial]

    print(
        f"Loaded {matrix.shape}, grid "
        f"{meta['grid_dimensions']}x{meta['grid_dimensions']}, "
        f"complement={meta.get('complement', False)}"
    )

    results = {}
    for serial in serials:
        cam_dir = os.path.join(run_dir, serial)
        if not os.path.isdir(cam_dir):
            print(f"  Skipping {serial}: missing folder")
            continue

        print(f"Decoding camera {serial} ...")
        save_delta_overview(cam_dir, meta)

        dual, code_map, valid, conf = decode_camera(
            cam_dir,
            matrix,
            meta,
            contrast_frac=contrast_frac,
            color=color,
            ref_floor=ref_floor,
            median_clean=median_clean,
        )

        cv2.imwrite(os.path.join(cam_dir, "dual_photo.png"), dual)
        cv2.imwrite(os.path.join(cam_dir, "code_map_color.png"), code_map_color(code_map, valid, meta))
        cv2.imwrite(os.path.join(cam_dir, "coverage_map.png"), coverage_map(code_map, valid, meta))
        cv2.imwrite(os.path.join(cam_dir, "valid_mask.png"), valid.astype(np.uint8) * 255)
        cv2.imwrite(os.path.join(cam_dir, "confidence.png"), _norm_u8(conf))

        np.save(os.path.join(cam_dir, "code_map.npy"), code_map)
        np.save(os.path.join(cam_dir, "valid_mask.npy"), valid)
        np.save(os.path.join(cam_dir, "confidence.npy"), conf)

        print(f"  Saved reconstruction products ({int(valid.sum())} valid pixels)")
        results[serial] = dual

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--captures-root", default="captures")
    parser.add_argument("--serials", nargs="+", default=None)
    parser.add_argument("--serial", default=None, help="Decode only one serial")
    parser.add_argument("--contrast-frac", type=float, default=0.2)
    parser.add_argument("--ref-floor", type=float, default=30.0)
    parser.add_argument("--no-median", action="store_true")
    parser.add_argument("--color", action="store_true")
    args = parser.parse_args()

    run_decode(
        run_name=args.run_name,
        captures_root=args.captures_root,
        serials=args.serials,
        only_serial=args.serial,
        contrast_frac=args.contrast_frac,
        color=args.color,
        ref_floor=args.ref_floor,
        median_clean=not args.no_median,
    )