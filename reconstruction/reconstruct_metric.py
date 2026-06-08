"""Metric 3D reconstruction using the real stereo calibration.

This is the upgrade from self_calibrated.py: instead of guessing the focal
length and recovering the camera pose up to scale, we load the actual stereo
calibration (calibration.py -> stereo_calib.npz) and triangulate in real-world
millimeters.

Inputs it consumes (already produced by the capture + decode step):
    captures/<run>/<serial>/code_map.npy     camera-pixel -> projector-cell id
    captures/<run>/<serial>/valid_mask.npy   which decodes are trustworthy
    captures/<run>/<serial>/frame_0.png      reference frame, for point color
    calibration/stereo_calib.npz             K1,d1,K2,d2,R,T

Pipeline:
  1. dense_matches: every valid camera-1 pixel -> the camera-2 centroid of its
     projector cell (shared cell == same world point).
  2. undistort both pixel sets, then triangulate with the calibrated
     P1 = K1[I|0], P2 = K2[R|T] -> 3D points in millimeters, camera-1 frame.
  3. keep points in front of both cameras and within max_reproj px of their
     observations; trim depth outliers; color from the reference frame.

Run (from the reconstruction/ folder, no cameras needed):
    python reconstruct_metric.py <run_name>
"""
import os
import sys
import numpy as np
import cv2

from self_calibrated import dense_matches, save_ply   # reuse the matcher + PLY


# Anchor all paths to the repo layout, independent of the current directory.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))      # .../reconstruction
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)                     # repo root
_CAPTURES = os.path.join(_REPO_ROOT, "captures")
_CALIB = os.path.join(_SCRIPT_DIR, "calibration", "stereo_calib.npz")


def load_calibration(path=_CALIB):
    d = np.load(path)
    return d["K1"], d["d1"], d["K2"], d["d2"], d["R"], d["T"]


def triangulate_metric(p1, p2, K1, d1, K2, d2, R, T, max_reproj=2.0):
    """Calibrated triangulation in millimeters.

    p1, p2 are matched pixels (distorted) in cam1 / cam2. We undistort them to
    ideal pixel coordinates (P=K so they stay in pixels), then intersect the two
    calibrated rays. Camera 1 is the world origin.
    """
    P1 = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K2 @ np.hstack([R, T.reshape(3, 1)])
    u1 = cv2.undistortPoints(p1.reshape(-1, 1, 2), K1, d1, P=K1).reshape(-1, 2)
    u2 = cv2.undistortPoints(p2.reshape(-1, 1, 2), K2, d2, P=K2).reshape(-1, 2)

    Xh = cv2.triangulatePoints(P1, P2, u1.T, u2.T)
    X = (Xh[:3] / Xh[3]).T                              # (N,3) mm, cam-1 frame

    X2 = (R @ X.T + T.reshape(3, 1)).T
    front = (X[:, 2] > 0) & (X2[:, 2] > 0)
    pr1 = (P1 @ np.vstack([X.T, np.ones(len(X))])); pr1 = (pr1[:2] / pr1[2]).T
    pr2 = (P2 @ np.vstack([X.T, np.ones(len(X))])); pr2 = (pr2[:2] / pr2[2]).T
    err = (np.linalg.norm(pr1 - u1, axis=1) +
           np.linalg.norm(pr2 - u2, axis=1)) / 2
    keep = front & (err < max_reproj)
    return X[keep], keep, err[keep]


def run(run_name, serials, captures_root=_CALIB and _CAPTURES,
        calib_path=_CALIB,
        max_reproj=2.0, depth_clip=(1, 99)):
    s1, s2 = serials
    d1dir = os.path.join(captures_root, run_name, s1)
    d2dir = os.path.join(captures_root, run_name, s2)
    code1 = np.load(os.path.join(d1dir, "code_map.npy"))
    valid1 = np.load(os.path.join(d1dir, "valid_mask.npy"))
    code2 = np.load(os.path.join(d2dir, "code_map.npy"))
    valid2 = np.load(os.path.join(d2dir, "valid_mask.npy"))
    H, W = code1.shape

    K1, dd1, K2, dd2, R, T = load_calibration(calib_path)

    dp1, dp2 = dense_matches(code1, valid1, code2, valid2)
    X, keep, err = triangulate_metric(dp1, dp2, K1, dd1, K2, dd2, R, T, max_reproj)
    in1 = dp1[keep]

    if len(X):
        z = X[:, 2]
        lo, hi = np.percentile(z, depth_clip)
        m = (z >= lo) & (z <= hi)
        X, in1, err = X[m], in1[m], err[m]

    colors = None
    ref = cv2.imread(os.path.join(d1dir, "frame_0.png"), cv2.IMREAD_COLOR)
    if ref is not None and len(in1):
        xs = np.clip(in1[:, 0].round().astype(int), 0, W - 1)
        ys = np.clip(in1[:, 1].round().astype(int), 0, H - 1)
        colors = ref[ys, xs][:, ::-1]

    out = os.path.join(captures_root, run_name, "reconstruction_metric.ply")
    save_ply(out, X, colors)

    bbox = X.max(0) - X.min(0) if len(X) else np.zeros(3)
    return {
        "run": run_name, "points": len(X),
        "reproj_med": float(np.median(err)) if len(err) else None,
        "depth_mm": (float(X[:, 2].min()), float(np.median(X[:, 2])),
                     float(X[:, 2].max())) if len(X) else None,
        "bbox_mm": tuple(np.round(bbox, 1)),
        "ply": out,
    }


if __name__ == "__main__":
    serials = ("105322251697", "046322251346")
    runs = sys.argv[1:] or ["scene1"]
    for r in runs:
        st = run(r, serials)
        dm = st["depth_mm"]
        print(f"[{st['run']}] points={st['points']} "
              f"reproj_med={st['reproj_med']:.2f}px")
        print(f"   depth(mm) min/med/max = {dm[0]:.0f}/{dm[1]:.0f}/{dm[2]:.0f}")
        bb=st["bbox_mm"]; print(f"   object bounding box (mm) = {bb[0]:.0f} x {bb[1]:.0f} x {bb[2]:.0f}")
        print(f"   -> {st['ply']}")
