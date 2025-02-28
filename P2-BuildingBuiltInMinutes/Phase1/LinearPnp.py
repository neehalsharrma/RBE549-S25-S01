import numpy as np


def linear_PnP(K, points2D, points3D):
    """
    Linear PnP to estimate the 3D points.
    @ K: The intrinsic camera matrix in the shape of (3, 3)

    @ R: The rotation matrix of the camera in the shape of (3, 3)
    @ C: The center of the camera in the shape of (3, 1)
    @ points2D: The 2D points from the image in the shape of (n, 2)
    @ points3D: The 3D points in the shape of (n, 3)


    """
