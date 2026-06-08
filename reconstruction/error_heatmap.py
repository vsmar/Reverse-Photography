"""Round-trip error heatmap: colour the projector+camera cloud by how far each
point sits from the camera1+camera2 surface (cloud-to-cloud distance, mm).

Outputs (for run <name>):
  captures/<name>/error_heatmap.png            slide figure (3 views + histogram)
  captures/<name>/reconstruction_projcam_error.ply   coloured cloud for CloudCompare

    python reconstruction/error_heatmap.py <run_name> [clip_mm]
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.spatial import cKDTree

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CAPTURES = os.path.join(os.path.dirname(_SCRIPT_DIR), "captures")


def load_ply(path):
    L = open(path).read().splitlines()
    i = L.index("end_header") + 1
    return np.array([[float(x) for x in ln.split()[:3]]
                     for ln in L[i:] if len(ln.split()) >= 3])


def save_ply_rgb(path, P, C):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(P)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(P, C):
            f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")


def main(run, clip=15.0):
    A = load_ply(os.path.join(_CAPTURES, run, "reconstruction_metric.ply"))   # cam1+cam2
    B = load_ply(os.path.join(_CAPTURES, run, "reconstruction_projcam.ply"))  # projector+cam
    err, _ = cKDTree(A).query(B)                      # mm, point-to-surface

    print(f"[{run}] round-trip error (projector+cam -> cam-stereo): "
          f"median={np.median(err):.1f}  mean={err.mean():.1f}  "
          f"p90={np.percentile(err,90):.1f} mm")

    # green (good) -> red (bad), clipped so the colour scale is readable.
    norm = np.clip(err / clip, 0, 1)
    cmap = cm.get_cmap("RdYlGn_r")
    rgb = (cmap(norm)[:, :3] * 255).astype(np.uint8)
    save_ply_rgb(os.path.join(_CAPTURES, run, "reconstruction_projcam_error.ply"),
                 B, rgb)

    fig = plt.figure(figsize=(17, 5))
    views = [("front (X-Y)", 0, 1, True), ("side (Z-Y)", 2, 1, True),
             ("top (X-Z)", 0, 2, False)]
    for k, (title, ai, bi, flipy) in enumerate(views):
        ax = fig.add_subplot(1, 4, k + 1)
        yv = -B[:, bi] if flipy else B[:, bi]
        sc = ax.scatter(B[:, ai], yv, c=err, cmap="RdYlGn_r", s=2,
                        vmin=0, vmax=clip)
        ax.set_title(title); ax.set_aspect("equal"); ax.set_xlabel("mm")
    cb = fig.colorbar(sc, ax=fig.axes, fraction=0.025, pad=0.02)
    cb.set_label("distance to cam-stereo surface (mm)")
    axh = fig.add_subplot(1, 4, 4)
    axh.hist(np.clip(err, 0, clip * 2), bins=60, color="tab:gray")
    axh.axvline(np.median(err), color="green", lw=2,
                label=f"median {np.median(err):.1f} mm")
    axh.axvline(err.mean(), color="red", lw=2, ls="--",
                label=f"mean {err.mean():.1f} mm")
    axh.set_title("error distribution"); axh.set_xlabel("mm"); axh.legend()
    fig.suptitle(f"Round-trip error heatmap - run '{run}'  "
                 f"(projector+camera vs camera-stereo)", fontsize=14)
    out = os.path.join(_CAPTURES, f"{run}_error_heatmap.png")
    plt.savefig(out, dpi=95, bbox_inches="tight")
    print(f"saved {out}")
    print(f"saved coloured cloud -> captures/{run}/reconstruction_projcam_error.ply")


if __name__ == "__main__":
    run = sys.argv[1] if len(sys.argv) > 1 else "dino"
    clip = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0
    main(run, clip)
