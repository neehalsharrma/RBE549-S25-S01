"""
This module provides functions for linear triangulation to estimate 3D points
from 2D correspondences in two images.

Functions
---------
linear_triangulation(K, R1, C1, R2, C2, points1, points2)
    Estimates 3D points using linear triangulation.
linear_triangulation2(K, R1, C1, R2, C2, points1, points2)
    Estimates 3D points using an alternative method of linear triangulation.
"""

import numpy as np


def linear_triangulation(
    K: np.ndarray,
    R1: np.ndarray,
    C1: np.ndarray,
    R2: np.ndarray,
    C2: np.ndarray,
    points1: np.ndarray,
    points2: np.ndarray,
) -> np.ndarray:
    """
    Linear Triangulation to estimate the 3D points.

    Parameters
    ----------
    K : np.ndarray
        The intrinsic camera matrix in the shape of (3, 3).
    R1 : np.ndarray
        The rotation matrix of the first camera in the shape of (3, 3).
    C1 : np.ndarray
        The center of the first camera in the shape of (3, 1).
    R2 : np.ndarray
        The rotation matrix of the second camera in the shape of (3, 3).
    C2 : np.ndarray
        The center of the second camera in the shape of (3, 1).
    points1 : np.ndarray
        The 2D points from the first image in the shape of (n, 2).
    points2 : np.ndarray
        The 2D points from the second image in the shape of (n, 2).

    Returns
    -------
    np.ndarray
        The estimated 3D points in the shape of (n, 4).
    """
    # Create the pose matrices for the cameras
    P1 = K @ R1 @ np.hstack((np.eye(3), -C1))
    P2 = K @ R2 @ np.hstack((np.eye(3), -C2))

    # Extract rows of the projection matrices
    p1_1 = P1[0, :].reshape(1, 4)
    p1_2 = P1[1, :].reshape(1, 4)
    p1_3 = P1[2, :].reshape(1, 4)

    p2_1 = P2[0, :].reshape(1, 4)
    p2_2 = P2[1, :].reshape(1, 4)
    p2_3 = P2[2, :].reshape(1, 4)

    points3D = []
    # Iterate through each point
    for i in range(points1.shape[0]):
        x1, y1 = points1[i]
        x2, y2 = points2[i]
        # From Page 312 of Hartley and Zisserman
        A = np.array(
            [
                [x1 * p1_3 - p1_1],
                [y1 * p1_3 - p1_2],
                [x2 * p2_3 - p2_1],
                [y2 * p2_3 - p2_2],
            ]
        ).reshape(4, 4)
        # Solve the linear system
        _, _, VT = np.linalg.svd(A)
        V = VT.T
        X = V[:, -1]
        X = X / X[3]
        points3D.append(X)

    return np.array(points3D).reshape(-1, 4)


def linear_triangulation2(
    K: np.ndarray,
    R1: np.ndarray,
    C1: np.ndarray,
    R2: np.ndarray,
    C2: np.ndarray,
    points1: np.ndarray,
    points2: np.ndarray,
) -> np.ndarray:
    """
    Linear Triangulation to estimate the 3D points.

    Parameters
    ----------
    K : np.ndarray
        The intrinsic camera matrix in the shape of (3, 3).
    R1 : np.ndarray
        The rotation matrix of the first camera in the shape of (3, 3).
    C1 : np.ndarray
        The center of the first camera in the shape of (3, 1).
    R2 : np.ndarray
        The rotation matrix of the second camera in the shape of (3, 3).
    C2 : np.ndarray
        The center of the second camera in the shape of (3, 1).
    points1 : np.ndarray
        The 2D points from the first image in the shape of (n, 2).
    points2 : np.ndarray
        The 2D points from the second image in the shape of (n, 2).

    Returns
    -------
    np.ndarray
        The estimated 3D points in the shape of (n, 4).
    """

    def skew_matrix(x: np.ndarray) -> np.ndarray:
        """
        Takes a (3, 1) vector and returns the skew-symmetric 3 x 3 matrix.

        Parameters
        ----------
        x : np.ndarray
            A 3x1 vector.

        Returns
        -------
        np.ndarray
            The skew-symmetric matrix of shape (3, 3).
        """
        return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])

    # Create the pose matrices for the cameras
    P1 = K @ R1 @ np.hstack((np.eye(3), -C1))
    P2 = K @ R2 @ np.hstack((np.eye(3), -C2))

    # Convert points to homogeneous coordinates
    points1 = np.hstack((points1, np.ones((points1.shape[0], 1))))
    points2 = np.hstack((points2, np.ones((points2.shape[0], 1))))

    points3D = []
    # Iterate through each point
    for i in range(points1.shape[0]):
        p1 = skew_matrix(points1[i])
        p2 = skew_matrix(points2[i])
        A = np.vstack((p1 @ P1, p2 @ P2))
        _, _, V = np.linalg.svd(A)
        X = V[-1, :]
        X = X / X[3]
        points3D.append(X)

    return np.array(points3D).reshape(-1, 4)
