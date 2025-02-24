import numpy as np
import cv2
import matplotlib.pyplot as plt

def extract_camera_pose(essential: np.array) -> tuple[list[np.array, np.array]]:
    """
    Extract the camera pose from the essential matrix.
    @ essential: The essential matrix.
    """
    C_out = []
    R_out = []
    # SVD to decouple the rotation and the translation matrices from the essential matrix
    U, D, VT = np.linalg.svd(essential)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    # Get the rotation matrices and the translation vectors
    R1 = U @ W @ VT
    R2 = U @ W.T @ VT
    C1_3 = U[:, 2].reshape(3, 1)
    C2_4 = -U[:, 2].reshape(3, 1)
    # Arrays to store the rotation matrices and the translation vectors
    C_matrices = [C1_3, C2_4, C1_3, C2_4]
    R_matrices = [R1, R1, R2, R2]

    for C, R in zip(C_matrices, R_matrices):
        if np.linalg.det(R) < 0:
            R = -R
            C = -C
        C_out.append(C)
        R_out.append(R)

    return C_out, R_out




