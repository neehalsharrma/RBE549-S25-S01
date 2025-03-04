"""
Module for building a visibility matrix for 3D points projected onto 2D images.

This module contains functions to compute a binary mask visibility matrix that indicates
whether a 3D point is visible from a given camera.
"""

import numpy as np
import sys
sys.dont_write_bytecode = True

def build_visibility_matrix(C_matrices, R_matrices, K, world_X_points, img_x_points) -> np.array:
    """
    Build a visibility matrix for 3D points projected onto 2D images.

    Parameters
    ----------
    C_matrices : list of np.array
        The camera centers in the shape of list[(n, 3)] of length num_imgs.
    R_matrices : list of np.array
        The rotation matrices in the shape of list[(3, 3)] of length num_imgs.
    K : np.array
        The intrinsic camera matrix in the shape of (3, 3).
    world_X_points : np.array
        The homogenized 3D points for the world coordinate system in the shape of (n, 4).
    img_x_points : list of np.array
        The homogenized 2D points from each image in the shape of list[(n, 3)] of length num_imgs.

    Returns
    -------
    np.array
        The binary mask visibility matrix in the shape of (num_imgs, n) where Vij
        is one if the jth point is visible from the ith camera and zero otherwise.
    """
    num_imgs = len(C_matrices)
    n = world_X_points.shape[0]
    visibility_matrix = np.zeros((num_imgs, n))

    for i in range(num_imgs):
        C = C_matrices[i]
        R = R_matrices[i]
        # Compute the projection matrix P
        P = K @ np.hstack((R, -R @ C))
        # Calculate the reprojection of 3D points onto the 2D image plane
        x_hat = P @ world_X_points.T
        x_hat = x_hat / x_hat[2, :]  # Normalize by the third (homogeneous) coordinate
        x_hat = x_hat[:2, :].T  # Use only the first two coordinates and transpose

        # Ensure the number of points matches
        if x_hat.shape[0] != img_x_points[i].shape[0]:
            raise ValueError(f"Number of points in image {i} does not match number of world points")

        # Compute the difference between actual and projected 2D points
        diffs = img_x_points[i][:, :2] - x_hat  # Shape: (n, 2)
        # Calculate the Euclidean distance of the reprojection error
        errors = np.linalg.norm(diffs, axis=1)  # Shape: (n,)
        # Determine visibility based on a threshold
        visibility_matrix[i] = (errors < 0.1).astype(int)

    return visibility_matrix