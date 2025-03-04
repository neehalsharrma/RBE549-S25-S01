"""
Module for extracting camera pose from an essential matrix.

This module provides functionality to extract the camera translation and rotation 
matrices from a given essential matrix using Singular Value Decomposition (SVD).

Functions
---------
extract_camera_pose(essential: np.array) -> tuple[np.array, np.array]
    Extracts the camera pose from the essential matrix.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
sys.dont_write_bytecode = True


def extract_camera_pose(essential: np.array) -> tuple[np.array, np.array]:
    """
    Extract the camera pose from the essential matrix.

    This function uses Singular Value Decomposition (SVD) to decouple the rotation and
    the translation matrices from the essential matrix. It ensures that the rotation
    matrices have a positive determinant.

    Parameters
    ----------
    essential : np.array
        The essential matrix in the shape of (3, 3).

    Returns
    -------
    tuple[np.array, np.array]
        The camera translation and the rotation matrices.
        The shape is (4, 3) for the camera center translations and (4, 3, 3) for the rotation matrices.

    Raises
    ------
    ValueError
        If the essential matrix is not of shape (3, 3).

    Notes
    -----
    The essential matrix is decomposed into its singular values and vectors. The rotation
    matrices are derived using a predefined matrix W. The translation vectors are derived
    from the third column of the U matrix from the SVD. The function ensures that the
    rotation matrices have a positive determinant by negating them if necessary.
    """
    if essential.shape != (3, 3):
        raise ValueError("The essential matrix must be of shape (3, 3).")

    C_out = []
    R_out = []
    # SVD to decouple the rotation and the translation matrices from the essential matrix
    U, D, VT = np.linalg.svd(essential)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    # Get the rotation matrices and the translation vectors
    R1 = U @ W @ VT
    R2 = U @ W.T @ VT
    C1_3 = U[:, 2].reshape(3, 1) / np.linalg.norm(U[:, 2])
    C2_4 = -U[:, 2].reshape(3, 1) / np.linalg.norm(U[:, 2])
    # Arrays to store the rotation matrices and the translation vectors
    # C is 3x1 and R is 3x3
    C_matrices = [C1_3, C2_4, C1_3, C2_4]
    R_matrices = [R1, R1, R2, R2]

    for C, R in zip(C_matrices, R_matrices):
        if np.linalg.det(R) < 0:
            R = -R
            C = -C
        C_out.append(C)
        R_out.append(R)
    C_out = np.array(C_out).reshape(4, 3)
    R_out = np.array(R_out).reshape(4, 3, 3)
    print("Extracted Camera Centers and Rotations")
    return C_out, R_out
