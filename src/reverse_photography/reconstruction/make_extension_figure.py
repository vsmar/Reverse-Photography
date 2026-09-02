"""Assemble the full extension storyboard for one capture, as image panels:

  (1) Camera 1 raw view          (2) Camera 2 raw view
  (3) cam1+cam2 -> 3D            (4) 3D -> synthesized PROJECTOR view (reverse)
  (5) projector+camera -> 3D     (6) round-trip overlay (cam-stereo vs proj-cam)

Run AFTER the reconstruction steps for the run have been produced:
    python src/reverse_photography/reconstruction/make_extension_figure.py <run_name>

It also drops the individual panels in captures/<run>/figure_panels/ so you can
use any single one on a slide.
"""
import os
import sys
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))  # .../src/reverse_photography/reconstruction -> repo root
_CAPTURES = os.path.join(_REPO_ROOT, "captures")
SERIALS = ("105322251697", "046322251346")


def load_ply(path):
    if not os.path.exists(path):
        return np.empty((0, 3)), None
    L = open(path).read().splitlines()
    i = L.index("end_header") + 1
    P, C = [], []
    for ln in L[i:]:
        v = ln.split()
        if len(v) < 3:
            continue
        P.append([float(v[0]), float(v[1]), float(v[2])])
        if len(v) >= 6:
            C.append([int(v[3]), int(v[4]), int(v[5])])
    return np.array(P), (np.array(C) / 255.0 if C else None)


def rgb(path):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if im is not None else None


def scatter3d(ax, X, C, title, sub=20000):
    if len(X) == 0:
        ax.text(0.5, 0.5, 0.5, "(missing)"); ax.set_title(title); return
    if len(X) > sub:
        idx = np.random.default_rng(0).choice(len(X), sub, replace=False)
        X, C = X[idx], (C[idx] if C is not None else None)
    ax.scatter(X[:, 0], X[:, 2], -X[:, 1], s=1,
               c=(C if C is not None else X[:, 2]))
    ax.set_title(title)
    ax.set_xlabel("X mm"); ax.set_ylabel("Z mm")
    try:
        ax.set_box_aspect((np.ptp(X[:, 0]), np.ptp(X[:, 2]), np.ptp(X[:, 1])))
    except Exception:
        pass
    ax.view_init(elev=18, azim=-70)


def main(run):
    cdir = os.path.join(_CAPTURES, run)
    cam1 = rgb(os.path.join(cdir, SERIALS[0], "frame_0.png"))
    cam2 = rgb(os.path.join(cdir, SERIALS[1], "frame_0.png"))
    proj = rgb(os.path.join(cdir, "projector_view_synth.png"))
    Xm, Cm = load_ply(os.path.join(cdir, "reconstruction_metric.ply"))
    Xp, Cp = load_ply(os.path.join(cdir, "reconstruction_projcam.ply"))

    fig = plt.figure(figsize=(18, 10))
    ax = fig.add_subplot(2, 3, 1); ax.imshow(cam1); ax.set_title("1) Camera 1 view"); ax.axis("off")
    ax = fig.add_subplot(2, 3, 2); ax.imshow(cam2); ax.set_title("1) Camera 2 view"); ax.axis("off")
    ax = fig.add_subplot(2, 3, 3, projection="3d"); scatter3d(ax, Xm, Cm, "2) cam1+cam2 -> 3D")
    ax = fig.add_subplot(2, 3, 4)
    if proj is not None:
        ax.imshow(proj)
    ax.set_title("3) 3D -> synthesized PROJECTOR view"); ax.axis("off")
    ax = fig.add_subplot(2, 3, 5, projection="3d"); scatter3d(ax, Xp, Cp, "4) projector+camera -> 3D")
    ax = fig.add_subplot(2, 3, 6)
    if len(Xm) and len(Xp):
        ax.scatter(Xm[:, 0], Xm[:, 2], s=1, c="tab:blue", alpha=.35, label="cam1+cam2")
        ax.scatter(Xp[:, 0], Xp[:, 2], s=1, c="tab:red", alpha=.35, label="projector+cam")
        ax.legend(markerscale=6); ax.set_aspect("equal")
    ax.set_title("round-trip overlay (top view, mm)"); ax.set_xlabel("X mm"); ax.set_ylabel("Z mm")

    fig.suptitle(f"Reverse-photography extension storyboard - run '{run}'", fontsize=15)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(_CAPTURES, f"{run}_extension_storyboard.png")
    plt.savefig(out, dpi=95)
    print(f"saved {out}")

    # also save individual panels
    pdir = os.path.join(cdir, "figure_panels"); os.makedirs(pdir, exist_ok=True)
    for name, img in [("1_cam1.png", cam1), ("1_cam2.png", cam2),
                      ("3_projector_view.png", proj)]:
        if img is not None:
            cv2.imwrite(os.path.join(pdir, name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"individual panels -> {pdir}")
    return out


if __name__ == "__main__":
    for r in (sys.argv[1:] or ["attempt2"]):
        main(r)
