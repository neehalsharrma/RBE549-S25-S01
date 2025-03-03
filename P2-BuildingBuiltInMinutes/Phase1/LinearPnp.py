"""
This module provides a function to perform Linear Perspective-n-Point (PnP) to estimate the camera pose.

Functions
---------
linear_PnP(
    K: np.ndarray, 
    points2D: np.ndarray, 
    points3D: np.ndarray
import numpy as np

def linear_PnP(
import numpy as np


def linear_PnP(
    K: np.ndarray, points2D: np.ndarray, points3D: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform Linear PnP to estimate the camera pose.

    Parameters
    ----------
    K : np.ndarray
        The intrinsic camera matrix in the shape of (3, 3).
    points2D : np.ndarray
        The 2D points from the image in the shape of (n, 2).
    points3D : np.ndarray
        The 3D points in the shape of (n, 3).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The rotation matrix (3, 3) and the camera center (3, 1).
    """
    # Number of points
    # Calculate the number of 2D points

    # Convert 2D points to homogeneous coordinates
    points2d_homogeneous = np.hstack((points2D, np.ones((num_points, 1))))

    # Construct the matrix A for the linear system
    A = np.zeros((2 * num_points, 12))
    for i in range(num_points):
        X, Y, Z = points3D[i]
        u, v = points2d_homogeneous[i, :2]
        A[2 * i] = [X, Y, Z, 1, 0, 0, 0, 0, -u * X, -u * Y, -u * Z, -u]
        A[2 * i + 1] = [0, 0, 0, 0, X, Y, Z, 1, -v * X, -v * Y, -v * Z, -v]

    # Solve the linear system using SVD
    _, _, VT = np.linalg.svd(A)
    P = VT[-1].reshape(3, 4)

    # Decompose the projection matrix to get R and C
    R = P[:, :3]
    t = P[:, 3]
    C = -np.linalg.inv(R) @ t

    return R, C.reshape(3, 1)
    # Ensure R is a valid rotation matrix using QR decomposition
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt

    return R, C.reshape(3, 1)
