"""
Decoder for time-multiplexed BINARY structured-light captures.

Capture model (from projection_control.py):
  - Frame 0 is the all-white reference (every cell ON).
  - Frames 1..K each light ~half the cells; a cell's ON/OFF state across
    these K frames is its unique binary CODE (verified: all 65536 cells in the
    256x256 grid get distinct 16-bit codes).

Pipeline per camera:
  decode_pixels()      -> code_map, valid_mask, confidence
  build_clusters()     -> per-cell list of spatially coherent Clusters
  disambiguate()       -> outlier clusters removed via projector-neighbor prior
  render_dual_photo()  -> weighted projector-PoV image

Weighting combines three factors:
  w_confidence : how clearly each bit-read cleared the threshold, averaged
                 over all K coded frames.
  w_brightness : reference-frame brightness ^ brightness_pow.
  Locality is handled structurally by clustering -- only pixels belonging to
  surviving clusters contribute, and within a cluster the Gaussian locality
  weight from the previous version is still applied.
"""

import os
import glob
import argparse
from dataclasses import dataclass, field
from collections import deque
from typing import List, Optional
from camera_controls import CameraController as Cam, CAMERA_SERIALS, OUTPUT_DIR

import numpy as np
import cv2


# =========================================================================== #
#  Codebook                                                                    #
# =========================================================================== #

def build_codebook(matrix):
    """(F,C) pattern matrix -> coded(K,C), code_int(C,), lut{int->cell_id}."""
    coded = matrix[1:].astype(np.uint8)
    K, C  = coded.shape
    weights  = (1 << np.arange(K))[::-1].astype(np.uint64)
    code_int = (coded.T.astype(np.uint64) @ weights)
    lut = {int(c): cell for cell, c in enumerate(code_int)}
    return coded, code_int, lut


# =========================================================================== #
#  Pixel decoding + confidence                                                 #
# =========================================================================== #

def decode_pixels(stack, matrix, contrast_frac=0.2):
    """Decode every camera pixel to a projector cell id.

    Parameters
    ----------
    stack         : (F, H, W) float64 grayscale, frame 0 = all-white ref
    matrix        : (F, C) pattern matrix
    contrast_frac : min (ref - darkest) / ref to trust a pixel

    Returns
    -------
    code_map   : (H, W) int64   cell id, -1 if invalid
    valid_mask : (H, W) bool
    confidence : (H, W) float64 in [0,1]
                 Mean normalised distance of each bit-read from the decision
                 boundary.  1 = every bit crystal-clear; 0 = all ambiguous.
    """
    F, H, W = stack.shape
    obs = stack.reshape(F, H * W).astype(np.float64)

    reference  = obs[0]
    coded_obs  = obs[1:]
    K          = coded_obs.shape[0]

    thresh     = reference * 0.5
    safe_thresh = np.maximum(thresh, 1e-9)

    # Hard bit decision
    bits = (coded_obs > thresh[None, :]).astype(np.uint64)

    # Per-frame, per-pixel confidence: normalised distance from boundary
    frame_conf = np.abs(coded_obs - thresh[None, :]) / safe_thresh[None, :]
    frame_conf = np.clip(frame_conf, 0.0, 1.0)
    confidence_flat = frame_conf.mean(axis=0)

    # Pack bits -> integer code
    weights      = (1 << np.arange(K))[::-1].astype(np.uint64)
    pixel_codes  = (bits.T @ weights)

    # Codebook lookup
    _, code_int, _ = build_codebook(matrix)
    max_code = int(code_int.max())
    table    = np.full(max_code + 1, -1, dtype=np.int64)
    table[code_int.astype(np.int64)] = np.arange(len(code_int))
    safe         = np.minimum(pixel_codes.astype(np.int64), max_code)
    code_map_flat = table[safe]

    darkest  = coded_obs.min(axis=0)
    contrast = reference - darkest
    valid_flat = (
        (contrast > contrast_frac * np.maximum(reference, 1e-9))
        & (code_map_flat >= 0)
    )

    return (
        code_map_flat.reshape(H, W),
        valid_flat.reshape(H, W),
        confidence_flat.reshape(H, W),
    )


# =========================================================================== #
#  Supernode clustering                                                        #
# =========================================================================== #

@dataclass
class Cluster:
    cell_id:       int
    pixel_indices: np.ndarray   # indices into the flat valid-pixel array
    centroid:      np.ndarray   # shape (2,)  [row, col] camera-space
    mean_conf:     float
    mean_bright:   float        # mean reference brightness, for fallback sort


def _union_find_clusters(rows, cols, connect_radius):
    """Group points into connected components using union-find.

    Two points are in the same component if their Euclidean distance is <=
    connect_radius.  We use a grid-bucketed approach so we only compare each
    point against its local neighbourhood rather than all O(N²) pairs.

    Parameters
    ----------
    rows, cols       : (N,) int arrays of camera-pixel coordinates
    connect_radius   : float, max distance to be considered connected

    Returns
    -------
    labels : (N,) int array, component id per point
    """
    N = len(rows)
    if N == 0:
        return np.empty(0, dtype=np.int64)

    # Union-Find with path compression + union by rank
    parent = np.arange(N, dtype=np.int64)
    rank   = np.zeros(N, dtype=np.int64)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]   # path halving
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    # Bucket points into grid cells of size connect_radius so we only check
    # nearby buckets (3x3 neighbourhood) for candidate pairs.
    r = connect_radius
    bucket = {}
    for i in range(N):
        gx = int(cols[i] // r)
        gy = int(rows[i] // r)
        bucket.setdefault((gx, gy), []).append(i)

    r2 = connect_radius ** 2
    for (gx, gy), members in bucket.items():
        # Check within bucket and against 8 neighbours
        for dgx in (-1, 0, 1):
            for dgy in (-1, 0, 1):
                neighbours = bucket.get((gx + dgx, gy + dgy), [])
                for i in members:
                    for j in neighbours:
                        if j <= i:
                            continue
                        dr = rows[i] - rows[j]
                        dc = cols[i] - cols[j]
                        if dr * dr + dc * dc <= r2:
                            union(i, j)

    # Flatten labels
    labels = np.array([find(i) for i in range(N)], dtype=np.int64)
    # Re-label 0..M-1
    unique, labels = np.unique(labels, return_inverse=True)
    return labels.astype(np.int64)


def build_clusters(code_map, valid_mask, confidence, reference_frame,
                   grid_meta,
                   connect_radius=10,
                   min_cluster_conf=0.3,
                   min_cluster_size=3):
    """Group valid pixels into spatially coherent clusters per projector cell.

    Parameters
    ----------
    code_map        : (H, W) int64
    valid_mask      : (H, W) bool
    confidence      : (H, W) float64
    reference_frame : (H, W) grayscale float or uint8
    grid_meta       : dict  'n_cells', 'grid_dimensions'
    connect_radius  : float, camera pixels -- max distance to be in same cluster
    min_cluster_conf: float -- drop clusters whose mean confidence is below this
    min_cluster_size: int   -- drop clusters with fewer pixels than this

    Returns
    -------
    cell_clusters : list of length n_cells
                    Each entry is a (possibly empty) list of Cluster objects.
    """
    H, W      = code_map.shape
    n_cells   = grid_meta["n_cells"]

    ref_gray  = (reference_frame if reference_frame.ndim == 2
                 else cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY))
    ref_flat  = ref_gray.reshape(-1).astype(np.float64)

    conf_flat  = confidence.reshape(-1)
    code_flat  = code_map.reshape(-1)
    valid_flat = valid_mask.reshape(-1)

    # Pixel coordinates
    rr, cc    = np.mgrid[0:H, 0:W]
    rows_flat = rr.reshape(-1)
    cols_flat = cc.reshape(-1)

    # Valid pixel indices
    v_idx    = np.where(valid_flat)[0]
    v_cells  = code_flat[v_idx]
    v_rows   = rows_flat[v_idx]
    v_cols   = cols_flat[v_idx]
    v_conf   = conf_flat[v_idx]
    v_bright = ref_flat[v_idx]

    # Group valid-pixel positions by cell id
    # argsort by cell for contiguous grouping
    order    = np.argsort(v_cells, kind="stable")
    sv_cells = v_cells[order]
    sv_idx   = v_idx[order]
    sv_rows  = v_rows[order]
    sv_cols  = v_cols[order]
    sv_conf  = v_conf[order]
    sv_bright= v_bright[order]

    cell_clusters = [[] for _ in range(n_cells)]

    # Find cell boundaries in the sorted array
    boundaries = np.searchsorted(sv_cells, np.arange(n_cells + 1))

    for cell_id in range(n_cells):
        lo, hi = int(boundaries[cell_id]), int(boundaries[cell_id + 1])
        if hi - lo == 0:
            continue

        px_rows  = sv_rows[lo:hi]
        px_cols  = sv_cols[lo:hi]
        px_conf  = sv_conf[lo:hi]
        px_bright= sv_bright[lo:hi]
        px_idx   = sv_idx[lo:hi]

        labels   = _union_find_clusters(px_rows, px_cols, connect_radius)
        n_labels = int(labels.max()) + 1

        for lbl in range(n_labels):
            mask = labels == lbl
            if mask.sum() < min_cluster_size:
                continue
            mc = float(px_conf[mask].mean())
            if mc < min_cluster_conf:
                continue
            c_rows = px_rows[mask].astype(np.float64)
            c_cols = px_cols[mask].astype(np.float64)
            centroid = np.array([c_rows.mean(), c_cols.mean()])
            cell_clusters[cell_id].append(Cluster(
                cell_id       = cell_id,
                pixel_indices = px_idx[mask],
                centroid      = centroid,
                mean_conf     = mc,
                mean_bright   = float(px_bright[mask].mean()),
            ))

    return cell_clusters


# =========================================================================== #
#  Cluster disambiguation via projector-neighbour continuity                  #
# =========================================================================== #

def disambiguate_clusters(cell_clusters, grid_meta,
                          anchor_k=12,
                          outlier_sigma=2.0):
    """Remove spatially inconsistent clusters using projector-neighbour priors.

    Algorithm
    ---------
    1. Cells with exactly one cluster are anchors -- their camera-space
       centroid is ground truth.
    2. For each ambiguous cell (>1 cluster), find the anchor_k nearest
       anchor cells in projector-space (BFS outward on the grid).
    3. Compute an expected camera-space centroid by inverse-distance-squared
       weighting of the anchor centroids.
    4. Score each candidate cluster by its distance from the expected centroid.
       Compute the spread (std) of those distances; drop clusters that are
       more than outlier_sigma * spread away from the expected position.
    5. If fewer than 1 cluster survives the filter, fall back to the single
       highest-confidence cluster (occlusion-tolerant: we never leave a cell
       completely empty if it had candidates).

    Parameters
    ----------
    cell_clusters : list[list[Cluster]]  from build_clusters
    grid_meta     : dict  'grid_dimensions', 'n_cells'
    anchor_k      : int   how many anchor neighbours to gather via BFS
    outlier_sigma : float rejection threshold in units of distance std-dev

    Returns
    -------
    resolved : list[list[Cluster]]  same shape, outlier clusters removed
    """
    g       = grid_meta["grid_dimensions"]
    n_cells = grid_meta["n_cells"]

    def cell_to_ij(cell_id):
        return divmod(cell_id, g)   # (row, col) in projector grid

    def ij_to_cell(i, j):
        return i * g + j

    # --- Identify anchors ----------------------------------------------------
    anchor_centroid = {}   # cell_id -> np.array([row, col])
    for cell_id, clusters in enumerate(cell_clusters):
        if len(clusters) == 1:
            anchor_centroid[cell_id] = clusters[0].centroid

    # --- BFS helper: find k nearest anchors in projector-grid space ----------
    def find_nearest_anchors(cell_id, k):
        """BFS from cell_id outward; collect up to k anchor cells."""
        ci, cj    = cell_to_ij(cell_id)
        visited   = {cell_id}
        queue     = deque([(ci, cj, 0)])
        found     = []   # (proj_dist, centroid)
        while queue and len(found) < k:
            i, j, dist = queue.popleft()
            cid = ij_to_cell(i, j)
            if cid in anchor_centroid and cid != cell_id:
                found.append((dist if dist > 0 else 1, anchor_centroid[cid]))
            for di, dj in ((-1,0),(1,0),(0,-1),(0,1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < g and 0 <= nj < g:
                    ncid = ij_to_cell(ni, nj)
                    if ncid not in visited:
                        visited.add(ncid)
                        queue.append((ni, nj, dist + 1))
        return found   # list of (projector_dist, camera_centroid)

    # --- Disambiguate --------------------------------------------------------
    resolved = [list(clusters) for clusters in cell_clusters]  # shallow copy

    for cell_id, clusters in enumerate(cell_clusters):
        if len(clusters) <= 1:
            continue   # nothing to disambiguate

        anchors = find_nearest_anchors(cell_id, anchor_k)

        if not anchors:
            # No anchor context at all -- keep highest-confidence cluster only
            best = max(clusters, key=lambda c: c.mean_conf)
            resolved[cell_id] = [best]
            continue

        # Expected camera centroid via inverse-distance-squared weighting
        weights = np.array([1.0 / (d ** 2) for d, _ in anchors])
        weights /= weights.sum()
        expected = sum(
            w * cent for w, (_, cent) in zip(weights, anchors)
        )

        # Distance of each candidate cluster from expected position
        dists = np.array([
            np.linalg.norm(cl.centroid - expected) for cl in clusters
        ])

        if len(dists) < 2:
            # Only one distance value, nothing to compare against
            resolved[cell_id] = list(clusters)
            continue

        dist_mean = dists.mean()
        dist_std  = dists.std()

        if dist_std < 1e-6:
            # All clusters are equally close -- keep all (probably fine)
            resolved[cell_id] = list(clusters)
            continue

        # Keep clusters within outlier_sigma of the mean distance
        survivors = [
            cl for cl, d in zip(clusters, dists)
            if d <= dist_mean + outlier_sigma * dist_std
        ]

        if not survivors:
            # Fallback: keep closest cluster to expected position
            best = clusters[int(dists.argmin())]
            resolved[cell_id] = [best]
        else:
            resolved[cell_id] = survivors

    return resolved


# =========================================================================== #
#  Weighted dual-photo from resolved clusters                                  #
# =========================================================================== #

def render_dual_photo(resolved_clusters, reference_frame, grid_meta,
                      confidence_map,
                      brightness_pow=1.0,
                      locality_sigma=None):
    """Build the projector-PoV image from disambiguated clusters.

    For each projector cell, all pixels from all surviving clusters contribute
    to a weighted average:

        w = w_confidence * w_brightness [* w_locality if sigma given]

    Parameters
    ----------
    resolved_clusters : list[list[Cluster]]  from disambiguate_clusters
    reference_frame   : (H, W) or (H, W, 3)
    grid_meta         : dict
    confidence_map    : (H, W) float64
    brightness_pow    : float  see dual_photo docstring
    locality_sigma    : float or None  Gaussian locality width in camera pixels

    Returns
    -------
    (grid_dim, grid_dim) uint8  or  (grid_dim, grid_dim, 3) uint8
    """
    g       = grid_meta["grid_dimensions"]
    n_cells = grid_meta["n_cells"]

    color = reference_frame.ndim == 3
    chans = 3 if color else 1
    ref   = (reference_frame.reshape(-1, chans).astype(np.float64)
             if color
             else reference_frame.reshape(-1, 1).astype(np.float64))

    ref_gray = (reference_frame if not color
                else cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY))
    ref_gray_flat = ref_gray.reshape(-1).astype(np.float64)
    conf_flat     = confidence_map.reshape(-1)

    sums   = np.zeros((n_cells, chans))
    counts = np.zeros(n_cells)

    for cell_id, clusters in enumerate(resolved_clusters):
        if not clusters:
            continue

        for cl in clusters:
            idx    = cl.pixel_indices
            r_vals = ref[idx]                      # (N, chans)
            c_vals = conf_flat[idx]                # (N,)
            b_vals = ref_gray_flat[idx]            # (N,)

            # Confidence weight
            w = c_vals.copy()

            # Brightness weight
            if brightness_pow > 0:
                safe_b = np.maximum(b_vals, 1e-9)
                bmax   = ref_gray_flat.max()
                w     *= (safe_b / max(bmax, 1e-9)) ** brightness_pow

            # Locality weight: Gaussian from cluster centroid
            if locality_sigma is not None and locality_sigma > 0:
                # Recover pixel coordinates from flat indices
                H = reference_frame.shape[0]
                W = reference_frame.shape[1]
                p_rows = (idx // W).astype(np.float64)
                p_cols = (idx  % W).astype(np.float64)
                dr = p_rows - cl.centroid[0]
                dc = p_cols - cl.centroid[1]
                dist2 = dr ** 2 + dc ** 2
                w *= np.exp(-dist2 / (2.0 * locality_sigma ** 2))

            np.add.at(sums,   cell_id, (r_vals * w[:, None]).sum(axis=0))
            counts[cell_id] += w.sum()

    nonzero = counts > 0
    sums[nonzero] /= counts[nonzero, None]

    img = sums.reshape(g, g, chans)
    img = img - img.min()
    if img.max() > 0:
        img = img / img.max()
    img = (img * 255).astype(np.uint8)
    return img[..., 0] if not color else img


# =========================================================================== #
#  Loading helpers                                                             #
# =========================================================================== #

def _frame_index(path):
    name = os.path.splitext(os.path.basename(path))[0]
    return int(name.split("_")[-1])


def load_frame_stack(cam_dir, color=False):
    paths = glob.glob(os.path.join(cam_dir, "frame_*.png"))
    if not paths:
        raise FileNotFoundError(f"No frame_*.png found in {cam_dir}")
    paths.sort(key=_frame_index)

    gray      = []
    ref_color = None
    for k, p in enumerate(paths):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read {p}")
        gray.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64))
        if k == 0:
            ref_color = img if color else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.stack(gray, axis=0), ref_color


def find_matrix(patterns_root, run_name, pattern):
    pdir  = os.path.join(patterns_root, run_name, pattern)
    mpath = os.path.join(pdir, "pattern_matrix.npy")
    gpath = os.path.join(pdir, "grid_meta.npy")
    if not (os.path.exists(mpath) and os.path.exists(gpath)):
        raise FileNotFoundError(
            f"Could not find pattern_matrix.npy / grid_meta.npy in {pdir}"
        )
    matrix    = np.load(mpath)
    grid_meta = np.load(gpath, allow_pickle=True).item()
    return matrix, grid_meta


# =========================================================================== #
#  Top-level decode driver                                                     #
# =========================================================================== #

def decode_camera(cam_dir, matrix, grid_meta,
                  contrast_frac=0.2,
                  color=False,
                  connect_radius=10,
                  min_cluster_conf=0.3,
                  min_cluster_size=3,
                  anchor_k=12,
                  outlier_sigma=2.0,
                  brightness_pow=1.0,
                  locality_sigma=None):
    """Full supernode pipeline for one camera folder.

    Returns
    -------
    dual        : (G, G) or (G, G, 3) uint8
    code_map    : (H, W) int64
    valid_mask  : (H, W) bool
    confidence  : (H, W) float64
    resolved    : list[list[Cluster]]
    """
    stack, ref = load_frame_stack(cam_dir, color=color)

    if stack.shape[0] != matrix.shape[0]:
        print(f"  WARNING: frame/matrix mismatch in {cam_dir}; using min.")
        n      = min(stack.shape[0], matrix.shape[0])
        stack  = stack[:n]
        matrix = matrix[:n]

    code_map, valid, confidence = decode_pixels(stack, matrix, contrast_frac)

    print(f"    Building clusters ...")
    clusters = build_clusters(
        code_map, valid, confidence, ref, grid_meta,
        connect_radius   = connect_radius,
        min_cluster_conf = min_cluster_conf,
        min_cluster_size = min_cluster_size,
    )

    n_ambiguous = sum(1 for cl in clusters if len(cl) > 1)
    print(f"    Disambiguating {n_ambiguous} ambiguous cells ...")
    resolved = disambiguate_clusters(
        clusters, grid_meta,
        anchor_k     = anchor_k,
        outlier_sigma= outlier_sigma,
    )

    dual = render_dual_photo(
        resolved, ref, grid_meta, confidence,
        brightness_pow = brightness_pow,
        locality_sigma = locality_sigma,
    )
    return dual, code_map, valid, confidence, resolved


def run_decode(run_name, pattern, 
               serials=None,
               captures_root="captures",
               patterns_root="patterns",
               contrast_frac=0.2,
               color=False,
               connect_radius=10,
               min_cluster_conf=0.3,
               min_cluster_size=3,
               anchor_k=12,
               outlier_sigma=2.0,
               brightness_pow=1.0,
               locality_sigma=None,
               only_serial=None):
    """Decode all (or one) cameras in a run."""
    matrix, grid_meta = find_matrix(patterns_root, run_name, pattern)
    g = grid_meta["grid_dimensions"]
    print(f"Loaded matrix {matrix.shape}, grid {g}x{g}")
    print(f"Params: connect_r={connect_radius}  min_conf={min_cluster_conf}"
          f"  min_size={min_cluster_size}  anchor_k={anchor_k}"
          f"  outlier_sigma={outlier_sigma}  brightness_pow={brightness_pow}"
          f"  locality_sigma={locality_sigma}")
    
    if serials is None:
        serials = CAMERA_SERIALS

    if only_serial is not None:
        serials = [only_serial]

    results = {}
    for serial in serials:
        cam_dir = os.path.join(captures_root, run_name, serial)
        if not os.path.isdir(cam_dir):
            print(f"  Skipping {serial}: {cam_dir} not found")
            continue
        print(f"Decoding camera {serial} ...")

        dual, code_map, valid, confidence, resolved = decode_camera(
            cam_dir, matrix, grid_meta,
            contrast_frac    = contrast_frac,
            color            = color,
            connect_radius   = connect_radius,
            min_cluster_conf = min_cluster_conf,
            min_cluster_size = min_cluster_size,
            anchor_k         = anchor_k,
            outlier_sigma    = outlier_sigma,
            brightness_pow   = brightness_pow,
            locality_sigma   = locality_sigma,
        )

        out_path = os.path.join(cam_dir, "dual_photo.png")
        cv2.imwrite(out_path, dual)
        np.save(os.path.join(cam_dir, "code_map.npy"),   code_map)
        np.save(os.path.join(cam_dir, "valid_mask.npy"), valid)
        np.save(os.path.join(cam_dir, "confidence.npy"), confidence)

        n_valid     = int(valid.sum())
        n_cells_hit = sum(1 for cl in resolved if cl)
        n_ambig_rem = sum(1 for cl in resolved if len(cl) > 1)
        mean_conf   = float(confidence[valid].mean()) if n_valid else 0.0
        print(f"  Saved {out_path}")
        print(f"  {n_valid} valid px | {n_cells_hit} cells covered"
              f" | {n_ambig_rem} still multi-cluster"
              f" | mean conf {mean_conf:.3f}")

        results[serial] = dual
    return results


# =========================================================================== #
#  CLI                                                                         #
# =========================================================================== #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Supernode structured-light decoder."
    )
    parser.add_argument("--run-name",          required=True)
    parser.add_argument("--pattern",           default="structured")
    parser.add_argument("--serials",           nargs="+", required=False, default=None)
    parser.add_argument("--serial",            default=None)
    parser.add_argument("--captures-root",     default="captures")
    parser.add_argument("--patterns-root",     default="patterns")
    parser.add_argument("--contrast-frac",     type=float, default=0.2)
    parser.add_argument("--color",             action="store_true")
    # Clustering
    parser.add_argument("--connect-radius",    type=float, default=10,
                        help="Max camera-pixel distance to be in same cluster.")
    parser.add_argument("--min-cluster-conf",  type=float, default=0.3,
                        help="Drop clusters with mean confidence below this.")
    parser.add_argument("--min-cluster-size",  type=int,   default=3,
                        help="Drop clusters with fewer pixels than this.")
    # Disambiguation
    parser.add_argument("--anchor-k",          type=int,   default=12,
                        help="Projector-neighbour anchors for expected-position"
                             " interpolation.")
    parser.add_argument("--outlier-sigma",     type=float, default=2.0,
                        help="Reject clusters beyond this many std-devs from"
                             " the neighbour-interpolated expected position.")
    # Rendering weights
    parser.add_argument("--brightness-pow",    type=float, default=1.0)
    parser.add_argument("--locality-sigma",    type=float, default=None,
                        help="Intra-cluster Gaussian locality width (px).")
    args = parser.parse_args()

    run_decode(
        run_name         = args.run_name,
        pattern          = args.pattern,
        serials          = args.serials,
        captures_root    = args.captures_root,
        patterns_root    = args.patterns_root,
        contrast_frac    = args.contrast_frac,
        color            = args.color,
        connect_radius   = args.connect_radius,
        min_cluster_conf = args.min_cluster_conf,
        min_cluster_size = args.min_cluster_size,
        anchor_k         = args.anchor_k,
        outlier_sigma    = args.outlier_sigma,
        brightness_pow   = args.brightness_pow,
        locality_sigma   = args.locality_sigma,
        only_serial      = args.serial,
    )