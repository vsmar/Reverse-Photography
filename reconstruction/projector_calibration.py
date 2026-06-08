"""Calibrate the projector as a THIRD camera (no extra capture needed).

A projector is an inverse camera. We already know, per camera pixel, which
projector cell lit it (code_map); with the two cameras calibrated we triangulate
that pixel to a 3D point. Each decoded pixel therefore gives

    (3D point, camera-1 frame)  <->  (projector pixel)

i.e. the 3D-2D data cv2.calibrateCamera needs. Pool reliable pairs from several
fixed-projector captures and solve for the projector intrinsics + pose.

IMPORTANT / honest limitation
-----------------------------
Our calibration points come from scanned OBJECTS, which occupy a thin depth slab
and a limited field. That makes the projector's focal length and distance hard to
separate (depth degeneracy), so we constrain the model to the stable part:
square pixels (fx = fy), principal point at the image centre, no distortion, and
solve only focal length + pose. The result is an APPROXIMATE projector model --
good enough to demonstrate the reverse / round-trip pipeline, but for a rigorous
intrinsic you would project a checkerboard onto a flat plane at several depths
(the standard projector-calibration capture).

Output: reconstruction/calibration/projector_calib.npz (K_p, dist_p, R_p, T_p,
proj_w, proj_h).
"""
import os
import sys
import glob
import numpy as np
import cv2

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
_CAPTURES = os.path.join(_REPO_ROOT, "captures")
_CALIB_DIR = os.path.join(_SCRIPT_DIR, "calibration")

sys.path.insert(0, _SCRIPT_DIR)
from reconstruct_metric import load_calibration, triangulate_metric

SERIALS = ("105322251697", "046322251346")


# Projector resolution for this rig (DLP, constant). Update if the projector
# or its output resolution ever changes.
PROJ_W, PROJ_H = 1920, 1080

def grid_meta_for(run):
    """Projector grid geometry for a run. Old captures saved it under
    patterns/<run>/*/grid_meta.npy; the newer capture front-end only writes
    captures/<run>/metadata.json, so derive it from there when needed. The grid
    is centred on the projector (matches projection_control's grid_x/grid_y)."""
    g = glob.glob(os.path.join(_REPO_ROOT, "patterns", run, "*", "grid_meta.npy"))
    if g:
        return np.load(g[0], allow_pickle=True).item()
    meta = os.path.join(_CAPTURES, run, "metadata.json")
    if os.path.exists(meta):
        import json
        m = json.load(open(meta))
        gd = int(m["grid_dimensions"]); res = int(m["pattern_res_pxl"])
        gp = gd * res
        return {"grid_dimensions": gd, "n_cells": gd * gd, "grid_px": gp,
                "grid_x": (PROJ_W - gp) // 2, "grid_y": (PROJ_H - gp) // 2,
                "proj_width": PROJ_W, "proj_height": PROJ_H}
    return None


def cell_to_proj_pixel(cells, gm):
    """Projector CELL id -> projector PIXEL at the cell centre."""
    g = gm["grid_dimensions"]
    sq = gm["grid_px"] / g
    u = gm["grid_x"] + (cells % g + 0.5) * sq
    v = gm["grid_y"] + (cells // g + 0.5) * sq
    return np.column_stack([u, v]).astype(np.float32)


def compact_centroids(code, valid, std_max=10.0, nmin=3):
    """Per-cell centroid in the camera image, KEEPING only cells whose pixels
    form a compact blob (spatial std < std_max). A real projector cell lights a
    small patch; a mis-decoded background cell is scattered, so this throws out
    the speckle that otherwise wrecks the calibration."""
    ys, xs = np.nonzero(valid)
    cells = code[ys, xs]
    o = np.argsort(cells, kind="stable")
    cells, xs, ys = cells[o], xs[o], ys[o]
    uniq, start = np.unique(cells, return_index=True)
    cnt = np.diff(np.append(start, len(cells)))
    sx = np.add.reduceat(xs.astype(float), start)
    sy = np.add.reduceat(ys.astype(float), start)
    sxx = np.add.reduceat(xs.astype(float) ** 2, start)
    syy = np.add.reduceat(ys.astype(float) ** 2, start)
    cx, cy = sx / cnt, sy / cnt
    std = np.sqrt(np.maximum(sxx / cnt - cx ** 2, 0) +
                  np.maximum(syy / cnt - cy ** 2, 0))
    keep = (cnt >= nmin) & (std < std_max)
    return {int(c): (x, y) for c, x, y, k in zip(uniq, cx, cy, keep) if k}


def collect_correspondences(run, calib, std_max=10.0, max_reproj=5.0):
    # NOTE: max_reproj is looser than the dense pipeline's 1.5 px on purpose --
    # a projector cell's footprint looks slightly different from each camera, so
    # its cam1 vs cam2 centroid are not the exact same 3D point. 5 px keeps
    # thousands of solid pairs; tightening to ~1.5 starves the calibration.
    """One reliable (3D, projector-pixel) pair per trustworthy projector cell."""
    K1, d1, K2, d2, R, T = calib
    d1d = os.path.join(_CAPTURES, run, SERIALS[0])
    d2d = os.path.join(_CAPTURES, run, SERIALS[1])
    c1 = np.load(os.path.join(d1d, "code_map.npy"))
    v1 = np.load(os.path.join(d1d, "valid_mask.npy"))
    c2 = np.load(os.path.join(d2d, "code_map.npy"))
    v2 = np.load(os.path.join(d2d, "valid_mask.npy"))
    a = compact_centroids(c1, v1, std_max)
    b = compact_centroids(c2, v2, std_max)
    gm = grid_meta_for(run)
    shared = [c for c in a if c in b and c >= 0]
    if not shared:
        return np.empty((0, 3), np.float32), np.empty((0, 2), np.float32), gm
    p1 = np.array([a[c] for c in shared])
    p2 = np.array([b[c] for c in shared])
    X, keep, err = triangulate_metric(p1, p2, K1, d1, K2, d2, R, T, max_reproj)
    cells = np.array(shared)[keep]
    proj_px = cell_to_proj_pixel(cells, gm)
    return X.astype(np.float32), proj_px, gm


def calibrate_projector(runs, calib_path=None, reject_pct=15):
    calib_path = calib_path or os.path.join(_CALIB_DIR, "stereo_calib.npz")
    calib = load_calibration(calib_path)

    Xs, Ps, gm = [], [], None
    for run in runs:
        X, p, g = collect_correspondences(run, calib)
        if len(X):
            Xs.append(X); Ps.append(p); gm = gm or g
            print(f"  {run}: {len(X)} reliable 3D<->projector pairs")
        else:
            print(f"  {run}: 0 reliable pairs (skipped)")
    X = np.vstack(Xs); P = np.vstack(Ps)
    pw, ph = gm["proj_width"], gm["proj_height"]

    # Stable model: square pixels, principal point centred, no distortion.
    f0 = 1.2 * pw
    K0 = np.array([[f0, 0, pw / 2.0], [0, f0, ph / 2.0], [0, 0, 1.0]])
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO |
             cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_ZERO_TANGENT_DIST |
             cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)

    rms, K, dist, rv, tv = cv2.calibrateCamera([X], [P], (pw, ph),
                                               K0.copy(), None, flags=flags)
    # One robust refit: drop the worst-reprojecting pairs and re-solve.
    Pr = K @ np.hstack([cv2.Rodrigues(rv[0])[0], tv[0]])
    pr = (Pr @ np.vstack([X.T, np.ones(len(X))])); pr = (pr[:2] / pr[2]).T
    e = np.linalg.norm(pr - P, axis=1)
    m = e < np.percentile(e, 100 - reject_pct)
    rms, K, dist, rv, tv = cv2.calibrateCamera([X[m]], [P[m]], (pw, ph),
                                               K.copy(), None, flags=flags)

    R_p = cv2.Rodrigues(rv[0])[0]
    T_p = tv[0].reshape(3)
    os.makedirs(_CALIB_DIR, exist_ok=True)
    out = os.path.join(_CALIB_DIR, "projector_calib.npz")
    np.savez(out, K_p=K, dist_p=dist, R_p=R_p, T_p=T_p, proj_w=pw, proj_h=ph)

    print(f"\nApproximate projector calibration "
          f"({int(m.sum())} pairs, RMS = {rms:.1f} projector px):")
    print(f"  K_p fx=fy={K[0,0]:.0f}  pp=({K[0,2]:.0f},{K[1,2]:.0f}) [fixed]")
    print(f"  T_p (projector vs cam1) = {np.round(T_p,1)} mm  "
          f"|T_p|={np.linalg.norm(T_p):.0f} mm")
    print(f"  saved -> {out}")
    return out


if __name__ == "__main__":
    runs = sys.argv[1:] or ["attempt2", "scene2", "scene3"]
    print(f"Calibrating projector from runs: {runs}")
    calibrate_projector(runs)
