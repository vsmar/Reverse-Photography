"""Self-calibrated 3D reconstruction from structured-light code maps.

When the stereo calibration (R, T between the two cameras) is NOT available,
we still recover the scene SHAPE up to an unknown global scale, directly from
the dense cross-camera correspondences that structured-light decoding gives us.

Pipeline
--------
1. reconstruction_control.py decoded every camera pixel to the projector cell
   that lit it (code_map). Two pixels -- one per camera -- sharing the same
   projector cell look at the same world point.
2. From clean one-match-per-cell centroids we estimate the essential matrix E
   (RANSAC) under an ASSUMED pinhole K, and recoverPose gives rotation R and a
   UNIT translation t (length unknown -> "up to scale").
3. With that pose fixed we DENSIFY: every valid camera-1 pixel is triangulated
   against the camera-2 centroid of its projector cell, then filtered by
   cheirality (in front of both cameras) and reprojection error.

DEMO-grade: geometry is metric-up-to-scale and the focal length is a guess, so
absolute distances are not meaningful. Feed real checkerboard calibration into
the main pipeline (calibration.py) for a true metric cloud.
"""
import os
import sys
import numpy as np
import cv2


def assumed_intrinsics(width, height, fov_deg=60.0):
    """Plausible pinhole K when no factory calibration is on hand.
    focal(px) = (width/2)/tan(fov/2); principal point at the image center."""
    f = (width / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    return np.array([[f, 0, width / 2.0],
                     [0, f, height / 2.0],
                     [0, 0, 1.0]], dtype=np.float64)


def _centroids(code, valid):
    """Per projector-cell centroid of the valid pixels carrying that cell id."""
    ys, xs = np.nonzero(valid)
    cells = code[ys, xs]
    order = np.argsort(cells, kind="stable")
    cells, xs, ys = cells[order], xs[order], ys[order]
    uniq, start = np.unique(cells, return_index=True)
    grp = np.diff(np.append(start, len(cells)))
    cx = np.add.reduceat(xs.astype(np.float64), start) / grp
    cy = np.add.reduceat(ys.astype(np.float64), start) / grp
    return uniq, cx, cy


def correspondences_from_codemaps(code1, valid1, code2, valid2):
    """One clean match per projector cell seen by both cameras (centroids)."""
    u1, x1, y1 = _centroids(code1, valid1)
    u2, x2, y2 = _centroids(code2, valid2)
    d1 = {int(c): (x, y) for c, x, y in zip(u1, x1, y1)}
    d2 = {int(c): (x, y) for c, x, y in zip(u2, x2, y2)}
    shared = [c for c in sorted(set(d1) & set(d2)) if c >= 0]
    pts1 = np.array([d1[c] for c in shared], dtype=np.float64)
    pts2 = np.array([d2[c] for c in shared], dtype=np.float64)
    return pts1, pts2


def reconstruct(pts1, pts2, K):
    """Essential matrix (RANSAC) -> recoverPose -> (R, t) up to scale."""
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC,
                                   prob=0.999, threshold=1.0)
    mask = mask.ravel().astype(bool)
    _, R, t, _ = cv2.recoverPose(E, pts1[mask], pts2[mask], K)
    return R, t, int(mask.sum())


def dense_matches(code1, valid1, code2, valid2):
    """Every valid camera-1 pixel paired to the camera-2 centroid of its cell."""
    u2, cx2, cy2 = _centroids(code2, valid2)
    maxc = int(u2.max())
    cenx = np.full(maxc + 1, np.nan)
    ceny = np.full(maxc + 1, np.nan)
    cenx[u2] = cx2
    ceny[u2] = cy2
    ys, xs = np.nonzero(valid1)
    c = code1[ys, xs]
    ok = (c >= 0) & (c <= maxc)
    ys, xs, c = ys[ok], xs[ok], c[ok]
    mx, my = cenx[c], ceny[c]
    have = ~np.isnan(mx)
    pts1 = np.column_stack([xs[have], ys[have]]).astype(np.float64)
    pts2 = np.column_stack([mx[have], my[have]]).astype(np.float64)
    return pts1, pts2


def triangulate_with_pose(pts1, pts2, K, R, t, max_reproj=2.0):
    """Triangulate against a known pose; keep cheirality + low-reproj points."""
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t.reshape(3, 1)])
    Xh = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    X = (Xh[:3] / Xh[3]).T
    X2 = (R @ X.T + t.reshape(3, 1)).T
    front = (X[:, 2] > 0) & (X2[:, 2] > 0)
    pr1 = (P1 @ np.vstack([X.T, np.ones(len(X))]))
    pr1 = (pr1[:2] / pr1[2]).T
    pr2 = (P2 @ np.vstack([X.T, np.ones(len(X))]))
    pr2 = (pr2[:2] / pr2[2]).T
    err = (np.linalg.norm(pr1 - pts1, axis=1) +
           np.linalg.norm(pr2 - pts2, axis=1)) / 2
    keep = front & (err < max_reproj)
    return X[keep], pts1[keep], err[keep]


def save_ply(path, points, colors=None):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write("element vertex %d\n" % len(points))
        f.write("property float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(len(points)):
            x, y, z = points[i]
            if colors is not None:
                r, g, b = colors[i]
                f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")
            else:
                f.write(f"{x} {y} {z}\n")


def run_scene(scene, serials, captures_root="captures", fov_deg=60.0,
              depth_clip=(1, 99)):
    s1, s2 = serials
    d1 = os.path.join(captures_root, scene, s1)
    d2 = os.path.join(captures_root, scene, s2)
    code1 = np.load(os.path.join(d1, "code_map.npy"))
    valid1 = np.load(os.path.join(d1, "valid_mask.npy"))
    code2 = np.load(os.path.join(d2, "code_map.npy"))
    valid2 = np.load(os.path.join(d2, "valid_mask.npy"))
    H, W = code1.shape
    K = assumed_intrinsics(W, H, fov_deg)

    # 1) Pose from clean one-per-cell centroid matches (robust, low noise).
    pts1, pts2 = correspondences_from_codemaps(code1, valid1, code2, valid2)
    R, t, e_inliers = reconstruct(pts1, pts2, K)

    # 2) Densify: triangulate EVERY valid cam-1 pixel against that pose.
    dp1, dp2 = dense_matches(code1, valid1, code2, valid2)
    X, in1, err = triangulate_with_pose(dp1, dp2, K, R, t)

    # Trim depth outliers (a few wild points always survive).
    if len(X):
        z = X[:, 2]
        lo, hi = np.percentile(z, depth_clip)
        keep = (z >= lo) & (z <= hi)
        X, in1, err = X[keep], in1[keep], err[keep]

    # Color each point from the reference (all-white) frame of camera 1.
    colors = None
    ref = cv2.imread(os.path.join(d1, "frame_0.png"), cv2.IMREAD_COLOR)
    if ref is not None and len(in1):
        xs = np.clip(in1[:, 0].round().astype(int), 0, W - 1)
        ys = np.clip(in1[:, 1].round().astype(int), 0, H - 1)
        colors = ref[ys, xs][:, ::-1]               # BGR -> RGB

    out = os.path.join(captures_root, scene, "reconstruction_selfcal.ply")
    save_ply(out, X, colors)
    return {
        "scene": scene, "centroid_matches": len(pts1), "e_inliers": e_inliers,
        "dense_matches": len(dp1), "points": len(X),
        "reproj_med": float(np.median(err)) if len(err) else None,
        "z_min": float(X[:, 2].min()) if len(X) else None,
        "z_med": float(np.median(X[:, 2])) if len(X) else None,
        "z_max": float(X[:, 2].max()) if len(X) else None,
        "t": t.ravel(), "ply": out,
    }


if __name__ == "__main__":
    serials = ("105322251697", "046322251346")
    scenes = sys.argv[1:] or ["scene1", "scene2", "scene3"]
    for sc in scenes:
        st = run_scene(sc, serials)
        print(f"[{st['scene']}] dense_matches={st['dense_matches']} "
              f"points={st['points']} reproj_med={st['reproj_med']:.2f}px "
              f"z(up-to-scale) {st['z_min']:.2f}/{st['z_med']:.2f}/{st['z_max']:.2f}")
        print(f"          baseline dir t={np.round(st['t'], 3)} -> {st['ply']}")
