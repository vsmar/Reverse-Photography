"""Step 3: reconstruct again from the PROJECTOR view + ONE camera.

Having calibrated the projector as a third camera, we can now triangulate using
the projector and a single real camera (camera 2), completely independent of the
camera1+camera2 stereo that started everything. Correspondences are free: each
camera-2 pixel's decoded cell IS a projector pixel.

    projector pixel  <->  camera-2 pixel   --triangulate-->  3D point

Comparing this cloud to the original camera1+camera2 cloud closes the loop and
measures how well the whole reverse pipeline holds together.

    python reconstruction/reconstruct_projcam.py <run_name>
"""
import os
import sys
import numpy as np
import cv2

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
_CAPTURES = os.path.join(_REPO_ROOT, "captures")
_CALIB_DIR = os.path.join(_SCRIPT_DIR, "calibration")
sys.path.insert(0, _SCRIPT_DIR)

from reconstruct_metric import load_calibration
from projector_calibration import compact_centroids, cell_to_proj_pixel, grid_meta_for
from synthesize_projector_view import load_projector
from self_calibrated import save_ply

SERIALS = ("105322251697", "046322251346")


def triangulate(Pa, Pb, pa, pb):
    Xh = cv2.triangulatePoints(Pa, Pb, pa.T, pb.T)
    return (Xh[:3] / Xh[3]).T


def run(run_name, cam_index=1, std_max=10.0, max_reproj=3.0):
    """cam_index: which real camera to pair with the projector (0 or 1)."""
    serial = SERIALS[cam_index]
    K1, d1, K2, d2, R, T = load_calibration(
        os.path.join(_CALIB_DIR, "stereo_calib.npz"))
    Kc, dc = (K1, d1) if cam_index == 0 else (K2, d2)
    # camera projection in cam-1 world frame
    if cam_index == 0:
        Pc = Kc @ np.hstack([np.eye(3), np.zeros((3, 1))])
        Rc, Tc = np.eye(3), np.zeros(3)
    else:
        Pc = Kc @ np.hstack([R, T.reshape(3, 1)])
        Rc, Tc = R, T.reshape(3)

    K_p, dist_p, R_p, T_p, pw, ph = load_projector()
    Pp = K_p @ np.hstack([R_p, T_p.reshape(3, 1)])

    d = os.path.join(_CAPTURES, run_name, serial)
    code = np.load(os.path.join(d, "code_map.npy"))
    valid = np.load(os.path.join(d, "valid_mask.npy"))
    gm = grid_meta_for(run_name)

    # Reliable camera pixels (compact cells) -> their projector pixels.
    cents = compact_centroids(code, valid, std_max)
    cells = np.array(sorted(c for c in cents if c >= 0))
    cam_px = np.array([cents[int(c)] for c in cells], np.float64)
    proj_px = cell_to_proj_pixel(cells, gm).astype(np.float64)

    # Undistort both, then triangulate projector <-> camera.
    uc = cv2.undistortPoints(cam_px.reshape(-1, 1, 2), Kc, dc, P=Kc).reshape(-1, 2)
    up = cv2.undistortPoints(proj_px.reshape(-1, 1, 2), K_p, dist_p, P=K_p).reshape(-1, 2)
    X = triangulate(Pp, Pc, up, uc)

    # cheirality (in front of camera and projector) + reprojection filter
    zc = (Rc @ X.T + Tc.reshape(3, 1))[2]
    zp = (R_p @ X.T + T_p.reshape(3, 1))[2]
    prc = (Pc @ np.vstack([X.T, np.ones(len(X))])); prc = (prc[:2] / prc[2]).T
    prp = (Pp @ np.vstack([X.T, np.ones(len(X))])); prp = (prp[:2] / prp[2]).T
    err = (np.linalg.norm(prc - uc, axis=1) + np.linalg.norm(prp - up, axis=1)) / 2
    keep = (zc > 0) & (zp > 0) & (err < max_reproj)
    X = X[keep]; cam_px = cam_px[keep]

    if len(X):
        z = X[:, 2]; lo, hi = np.percentile(z, [1, 99])
        m = (z >= lo) & (z <= hi); X, cam_px = X[m], cam_px[m]

    # color from that camera's reference frame
    ref = cv2.imread(os.path.join(d, "frame_0.png"), cv2.IMREAD_COLOR)
    colors = None
    if ref is not None and len(cam_px):
        xs = np.clip(cam_px[:, 0].round().astype(int), 0, code.shape[1] - 1)
        ys = np.clip(cam_px[:, 1].round().astype(int), 0, code.shape[0] - 1)
        colors = ref[ys, xs][:, ::-1]

    out = os.path.join(_CAPTURES, run_name, "reconstruction_projcam.ply")
    save_ply(out, X, colors)
    print(f"[{run_name}] projector + cam{cam_index} -> {len(X)} pts "
          f"(depth {X[:,2].min():.0f}-{X[:,2].max():.0f} mm)  -> {out}")
    return out


if __name__ == "__main__":
    for _name in (sys.argv[1:] or ["attempt2"]):
        run(_name)
