"""Step 2 (the "reverse"): render the projector's-eye 2D view from the 3D cloud.

Given the camera1+camera2 metric reconstruction and the projector calibration,
we project every 3D point into the projector's image plane (cv2.projectPoints)
and splat its color, nearest-point-wins. The result is a synthetic photo from
the projector's vantage point -- literally "what the projector sees" -- rebuilt
from geometry rather than captured.

    python src/reverse_photography/reconstruction/synthesize_projector_view.py <run_name>
"""
import os
import sys
import numpy as np
import cv2

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))  # .../src/reverse_photography/reconstruction -> repo root
_CAPTURES = os.path.join(_REPO_ROOT, "captures")
_CALIB_DIR = os.path.join(_SCRIPT_DIR, "calibration")


def load_ply(path):
    lines = open(path).read().splitlines()
    i = lines.index("end_header") + 1
    P, C = [], []
    for ln in lines[i:]:
        v = ln.split()
        if len(v) < 3:
            continue
        P.append([float(v[0]), float(v[1]), float(v[2])])
        if len(v) >= 6:
            C.append([int(v[3]), int(v[4]), int(v[5])])
    return np.array(P), (np.array(C) if C else None)


def load_projector(path=None):
    d = np.load(path or os.path.join(_CALIB_DIR, "projector_calib.npz"))
    return (d["K_p"], d["dist_p"], d["R_p"], d["T_p"],
            int(d["proj_w"]), int(d["proj_h"]))


def synthesize(run, splat=3):
    K_p, dist_p, R_p, T_p, pw, ph = load_projector()
    X, C = load_ply(os.path.join(_CAPTURES, run, "reconstruction_metric.ply"))
    if C is None:
        C = np.full((len(X), 3), 200, np.uint8)

    # 3D (cam-1 frame) -> projector pixels. depth = z in the projector frame.
    rvec = cv2.Rodrigues(R_p)[0]
    uv, _ = cv2.projectPoints(X.reshape(-1, 1, 3).astype(np.float64),
                              rvec, T_p.astype(np.float64), K_p, dist_p)
    uv = uv.reshape(-1, 2)
    depth = (R_p @ X.T + T_p.reshape(3, 1))[2]

    img = np.zeros((ph, pw, 3), np.uint8)
    zbuf = np.full((ph, pw), np.inf)
    u = np.round(uv[:, 0]).astype(int)
    v = np.round(uv[:, 1]).astype(int)
    ok = (u >= 0) & (u < pw) & (v >= 0) & (v < ph) & (depth > 0)
    order = np.argsort(-depth[ok])          # far first so near overwrites
    ui, vi = u[ok][order], v[ok][order]
    ci = C[ok][order][:, ::-1]              # RGB->BGR for cv2
    di = depth[ok][order]
    for r in range(-splat, splat + 1):      # small splat to fill gaps
        for c in range(-splat, splat + 1):
            vv = np.clip(vi + r, 0, ph - 1); uu = np.clip(ui + c, 0, pw - 1)
            img[vv, uu] = ci

    out = os.path.join(_CAPTURES, run, "projector_view_synth.png")
    cv2.imwrite(out, img)
    inb = int(ok.sum())
    print(f"[{run}] projected {inb}/{len(X)} points into the projector view "
          f"({pw}x{ph}) -> {out}")
    return out


if __name__ == "__main__":
    for run in (sys.argv[1:] or ["attempt2"]):
        synthesize(run)
