"""Decode structured-light captures into cross-camera pixel correspondences.

This is the ONLY stage whose math depends on the projector pattern, so it is
isolated here. The rest of the pipeline just consumes the (pts1, pts2) point
pairs this module returns and never needs to know which pattern was used.

Idea: each captured frame lit up some set of projector cells. By looking at how
a single camera pixel's brightness changes across the whole sequence, we work
out which projector cell illuminates that pixel -- its "code". Two pixels (one
per camera) that decode to the SAME projector cell are looking at the same point
in the world, which is exactly the correspondence triangulation needs.
"""
from collections import defaultdict

import numpy as np


def pixel_codes(stack, measurement_matrix=None, min_contrast=30):
    """Assign each pixel the projector cell that best illuminates it.

    stack:               (F, H, W) grayscale frames, one per projected pattern.
    measurement_matrix:  (F, C) 0/1 array of which cells were ON per frame.
                         None is allowed only for the raster pattern, where each
                         frame lights exactly one cell (the matrix is identity),
                         so the frame index already IS the cell id.
    min_contrast:        pixels whose intensity barely changes across the
                         sequence are background/shadow and get marked invalid.

    Returns (code_map, valid_mask), both (H, W). code_map holds an integer cell
    id per pixel; valid_mask is True where that id is trustworthy.
    """
    F, H, W = stack.shape
    obs = stack.reshape(F, H * W).astype(np.float64)  # (F, P) one column/pixel

    if measurement_matrix is None:
        # Raster: frame index == cell id, so the single brightest frame wins.
        code_flat = np.argmax(obs, axis=0)             # (P,)
    else:
        # General case (hadamard / random): each frame mixes many cells, so we
        # un-mix. With M the measurement matrix, the per-cell response r at a
        # pixel satisfies  obs = M @ r. Solving that least-squares for every
        # pixel at once gives the response to each cell; the strongest cell is
        # the one illuminating the pixel.
        #     r = pinv(M) @ obs        ->  (C, P)
        M = np.asarray(measurement_matrix, dtype=np.float64)
        response = np.linalg.pinv(M) @ obs             # (C, P)
        code_flat = np.argmax(response, axis=0)        # (P,)

    contrast = obs.max(axis=0) - obs.min(axis=0)       # (P,)
    valid_flat = contrast > min_contrast

    return code_flat.reshape(H, W), valid_flat.reshape(H, W)


def match_codes(code1, valid1, code2, valid2):
    """Pair up pixels from the two cameras that share the same projector cell.

    For each projector cell seen by both cameras, take the centroid of the
    camera-2 pixels carrying that code as the match for each camera-1 pixel.
    (A stricter version could enforce the epipolar constraint, but the centroid
    is enough for a first reconstruction.)

    Returns (pts1, pts2): two (N, 2) float arrays of matched (x, y) pixels.
    """
    # Where does each cell id appear in camera 2?
    cell_to_xy2 = defaultdict(list)
    ys, xs = np.nonzero(valid2)
    for y, x in zip(ys, xs):
        cell_to_xy2[int(code2[y, x])].append((x, y))
    cell_centroid2 = {c: np.mean(v, axis=0) for c, v in cell_to_xy2.items()}

    pts1, pts2 = [], []
    ys, xs = np.nonzero(valid1)
    for y, x in zip(ys, xs):
        c = int(code1[y, x])
        if c in cell_centroid2:
            pts1.append((x, y))
            pts2.append(cell_centroid2[c])

    return np.array(pts1, dtype=np.float64), np.array(pts2, dtype=np.float64)


def build_correspondences(stack1, stack2, measurement_matrix=None, min_contrast=30):
    """End-to-end: two image stacks -> matched pixel pairs (pts1, pts2)."""
    code1, valid1 = pixel_codes(stack1, measurement_matrix, min_contrast)
    code2, valid2 = pixel_codes(stack2, measurement_matrix, min_contrast)
    return match_codes(code1, valid1, code2, valid2)
