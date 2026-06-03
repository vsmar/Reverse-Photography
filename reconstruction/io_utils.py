"""I/O helpers: loading image stacks / matrices and writing point clouds."""
import glob
import os

import cv2
import numpy as np


def load_image_stack(folder: str, template: str = "frame_{i}.png") -> np.ndarray:
    """Load frame_0.png, frame_1.png, ... into an (F, H, W) grayscale array.

    Frames are sorted by their integer index, not lexicographically, so that
    frame_2 comes before frame_10 (string sorting would put frame_10 first).
    """
    pattern = os.path.join(folder, template.replace("{i}", "*"))
    paths = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(f"No frames matching '{template}' in {folder}")

    def frame_index(path):
        name = os.path.basename(path)
        digits = "".join(ch for ch in name if ch.isdigit())
        return int(digits)

    paths.sort(key=frame_index)
    frames = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in paths]
    return np.stack(frames, axis=0)


def load_measurement_matrix(path: str) -> np.ndarray:
    """Load the (num_frames, num_cells) matrix of which cells were ON per frame.

    Adjust this to match however projection_control.save_pattern_matrix wrote
    the file (the default assumes a .npy file; falls back to CSV).
    """
    if path.endswith(".npy"):
        return np.load(path)
    return np.genfromtxt(path, delimiter=",")


def save_ply(path: str, points: np.ndarray, colors: np.ndarray = None) -> None:
    """Write an (N, 3) point cloud to an ASCII .ply file.

    If colors (N, 3) uint8 are given, they are written as per-vertex RGB so the
    cloud shows up colored in MeshLab / CloudCompare.
    """
    points = np.asarray(points)
    n = len(points)
    has_color = colors is not None

    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if has_color:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            x, y, z = points[i]
            if has_color:
                r, g, b = colors[i]
                f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")
            else:
                f.write(f"{x} {y} {z}\n")
