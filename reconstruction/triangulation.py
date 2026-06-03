"""Triangulate matched pixel pairs into 3D points.

Once we know where the two cameras are relative to each other (R, T) and how
each maps pixels to rays (the intrinsics K, dist), a point seen in both images
is just the intersection of the two back-projected rays. OpenCV does the linear
solve for us.
"""
import cv2
import numpy as np


def triangulate(pts1, pts2, K1, d1, K2, d2, R, T):
    """Reconstruct 3D points from corresponding pixels in the two cameras.

    Camera 1 is taken as the world origin:   P1 = K1 [I | 0].
    Camera 2 sits at (R, T) relative to it:  P2 = K2 [R | T].

    Returns an (N, 3) array of 3D points in camera-1 coordinates. Units are
    millimeters, inherited from the checkerboard square size used in calibration.
    """
    if len(pts1) == 0:
        return np.empty((0, 3))

    P1 = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K2 @ np.hstack([R, T.reshape(3, 1)])

    # Undistort, then re-express in pixel coordinates (P=K) so the points line
    # up with the projection matrices above.
    u1 = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K1, d1, P=K1).reshape(-1, 2)
    u2 = cv2.undistortPoints(pts2.reshape(-1, 1, 2), K2, d2, P=K2).reshape(-1, 2)

    # triangulatePoints returns homogeneous (4, N); divide by w for real xyz.
    X_h = cv2.triangulatePoints(P1, P2, u1.T, u2.T)
    X = (X_h[:3] / X_h[3]).T
    return X
