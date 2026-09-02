"""Camera calibration: intrinsics from RealSense, extrinsics from a checkerboard.

The intrinsics (K, distortion) come straight off each RealSense device's factory
calibration. The extrinsics -- the rotation R and translation T relating the two
cameras -- are NOT known to the cameras (they are two independent devices), so we
recover them from checkerboard images that BOTH cameras saw at the same moment.

This module is the only one that touches the hardware (to read intrinsics). Run
it once with both cameras attached; the result is cached to disk and the main
reconstruction then runs fully offline.
"""
import glob
import os

import cv2
import numpy as np


def get_realsense_intrinsics(serial: str, width: int, height: int):
    """Read the factory color intrinsics for one RealSense camera.

    Returns (K, dist): the 3x3 camera matrix and the distortion coefficients.
    Requires the camera to be plugged in.
    """
    import pyrealsense2 as rs

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
    profile = pipe.start(cfg)
    try:
        intr = (profile.get_stream(rs.stream.color)
                .as_video_stream_profile().get_intrinsics())
    finally:
        pipe.stop()

    K = np.array([[intr.fx, 0.0,     intr.ppx],
                  [0.0,     intr.fy, intr.ppy],
                  [0.0,     0.0,     1.0]])
    dist = np.array(intr.coeffs, dtype=np.float64)
    return K, dist


def _object_points(cols, rows, square_mm):
    """3D coordinates of the checkerboard inner corners, z=0, in millimeters.

    The board is flat, so every corner has z=0; (x, y) march across the grid in
    units of one square, scaled to real millimeters.
    """
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_mm
    return objp


def stereo_calibrate(cfg):
    """Compute R, T between the two cameras from paired checkerboard images.

    Expects matching file names in
        <checkerboard_dir>/<serial1>/   and   <checkerboard_dir>/<serial2>/
    so the i-th image in each folder is the same moment in time.

    Saves K1, d1, K2, d2, R, T to cfg.calibration_path and returns them.
    """
    s1, s2 = cfg.camera_serials
    files1 = sorted(glob.glob(os.path.join(cfg.checkerboard_dir, s1, "*.png")))
    files2 = sorted(glob.glob(os.path.join(cfg.checkerboard_dir, s2, "*.png")))
    if not files1 or len(files1) != len(files2):
        raise ValueError("Need the same (nonzero) number of images per camera.")

    pattern = (cfg.checkerboard_cols, cfg.checkerboard_rows)
    objp = _object_points(*pattern, cfg.checkerboard_square_mm)

    objpoints, imgpoints1, imgpoints2 = [], [], []
    for f1, f2 in zip(files1, files2):
        g1 = cv2.imread(f1, cv2.IMREAD_GRAYSCALE)
        g2 = cv2.imread(f2, cv2.IMREAD_GRAYSCALE)
        ok1, c1 = cv2.findChessboardCorners(g1, pattern, None)
        ok2, c2 = cv2.findChessboardCorners(g2, pattern, None)
        if ok1 and ok2:
            objpoints.append(objp)
            imgpoints1.append(c1)
            imgpoints2.append(c2)

    if not objpoints:
        raise RuntimeError("No checkerboard detected in any image pair.")
    print(f"Using {len(objpoints)} valid image pairs for calibration.")

    # Start from each camera's factory intrinsics and keep them fixed; we only
    # solve for the relative pose, which is far more stable than solving for
    # everything at once.
    K1, d1 = get_realsense_intrinsics(s1, cfg.image_width, cfg.image_height)
    K2, d2 = get_realsense_intrinsics(s2, cfg.image_width, cfg.image_height)

    _, K1, d1, K2, d2, R, T, _, _ = cv2.stereoCalibrate(
        objpoints, imgpoints1, imgpoints2,
        K1, d1, K2, d2,
        imageSize=(cfg.image_width, cfg.image_height),
        flags=cv2.CALIB_FIX_INTRINSIC,
    )

    os.makedirs(os.path.dirname(cfg.calibration_path), exist_ok=True)
    np.savez(cfg.calibration_path, K1=K1, d1=d1, K2=K2, d2=d2, R=R, T=T)
    return K1, d1, K2, d2, R, T


def load_calibration(path: str):
    """Load a previously computed stereo calibration from disk."""
    data = np.load(path)
    return (data["K1"], data["d1"], data["K2"], data["d2"],
            data["R"], data["T"])


if __name__ == "__main__":
    from config import CONFIG

    stereo_calibrate(CONFIG)
    print(f"Saved stereo calibration to {CONFIG.calibration_path}")
