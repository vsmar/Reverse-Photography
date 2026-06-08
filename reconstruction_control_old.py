import os
import glob
import numpy as np
import cv2

def build_codebook(matrix):
    coded = matrix[1:].astype(np.uint8)
    K, C = coded.shape
    weights = (1 << np.arange(K))[::-1].astype(np.uint64)
    code_int = (coded.T.astype(np.uint64) @ weights)
    lut = {int(code): cell for cell, code in enumerate(code_int)}
    return coded, code_int, lut

def decode_pixels(stack, matrix, contrast_frac=0.2):
    F, H, W = stack.shape
    obs = stack.reshape(F, H * W).astype(np.float64)
    reference = obs[0]
    coded_obs = obs[1:]
    K = coded_obs.shape[0]

    thresh = reference * 0.5
    bits = (coded_obs > thresh[None, :]).astype(np.uint64)
    weights = (1 << np.arange(K))[::-1].astype(np.uint64)
    pixel_codes = (bits.T @ weights)

    _, code_int, _ = build_codebook(matrix)
    max_code = int(code_int.max())
    table = np.full(max_code + 1, -1, dtype=np.int64)
    table[code_int.astype(np.int64)] = np.arange(len(code_int))
    safe = np.minimum(pixel_codes.astype(np.int64), max_code)
    code_map_flat = table[safe]

    darkest = coded_obs.min(axis=0)
    contrast = reference - darkest
    valid_flat = (contrast > contrast_frac * np.maximum(reference, 1e-9)) & (code_map_flat >= 0)

    return code_map_flat.reshape(H, W), valid_flat.reshape(H, W)

def dual_photo(code_map, valid_mask, reference_frame, grid_dimensions):
    g = grid_dimensions
    n_cells = g * g
    
    color = reference_frame.ndim == 3
    chans = 3 if color else 1
    ref = reference_frame.reshape(-1, chans).astype(np.float64) if color else reference_frame.reshape(-1, 1).astype(np.float64)
    
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
    if img.max() > 0: img = img / img.max()
    img = (img * 255).astype(np.uint8)
    return img[..., 0] if not color else img

def run_decode(run_name, pattern, serials, captures_root="captures", contrast_frac=0.2, color=False):
    run_dir = os.path.join(captures_root, run_name)
    mpath = os.path.join(run_dir, "pattern_matrix.npy")
    if not os.path.exists(mpath):
        print(f"Missing matrix at {mpath}")
        return
        
    matrix = np.load(mpath)
    grid_dim = int(np.sqrt(matrix.shape[1]))
    
    for serial in serials:
        cam_dir = os.path.join(run_dir, serial)
        if not os.path.exists(cam_dir): continue
        
        paths = glob.glob(os.path.join(cam_dir, "frame_*.png"))
        paths.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split("_")[-1]))
        if not paths: continue
        
        gray = []
        ref_color = None
        for k, p in enumerate(paths):
            img = cv2.imread(p)
            gray.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64))
            if k == 0: ref_color = img if color else gray[-1]
            
        stack = np.stack(gray, axis=0)
        n = min(stack.shape[0], matrix.shape[0])
        stack, matrix_use = stack[:n], matrix[:n]
        
        code_map, valid = decode_pixels(stack, matrix_use, contrast_frac)
        dual = dual_photo(code_map, valid, ref_color, grid_dim)
        
        cv2.imwrite(os.path.join(cam_dir, "dual_photo.png"), dual)
        np.save(os.path.join(cam_dir, "code_map.npy"), code_map)
        np.save(os.path.join(cam_dir, "valid_mask.npy"), valid)
        print(f"{serial}: saved dual photo (valid pxl: {int(valid.sum())})")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="test_inverse")
    parser.add_argument("--pattern", default="structured")
    args = parser.parse_args()
    
    # Assuming standard serials for testing if none provided
    CAMERA_SERIALS = ["105322251697", "046322251346"]
    run_decode(args.run_name, args.pattern, CAMERA_SERIALS)
