"""
This module provides a function to disambiguate the correct camera pose from a set of possible poses.

Functions
---------
getCorrectPose(K: np.ndarray, C_out: np.ndarray, R_out: np.ndarray, points1: np.ndarray, points2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]
    Get the correct pose from the set of poses.
"""

import numpy as np
from LinearTriangulation import linear_triangulation


# Was going to call LinearTriangualtion multiple times to create x_set
# Linear Triangulation with (C_out[i], [0,0,0], R_out[i], np.eye(3)) sort of thing
# But we can also directly start over and call linear triangulation functions here instead like in this method
def getCorrectPose(
    K, C_out: np.array, R_out: np.array, points1: np.array, points2: np.array
):
    """
    Get the correct pose from the set of poses.

    Parameters
    ----------
    K : np.ndarray
        The intrinsic camera matrix in the shape of (3, 3).
    C_out : np.ndarray
        The camera centers in the shape of (4, 3).
    R_out : np.ndarray
        The rotation matrices in the shape of (4, 3, 3).
    points1 : np.ndarray
        The 2D points from the first image in the shape of (n, 2).
    points2 : np.ndarray
        The 2D points from the second image in the shape of (n, 2).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        The correct camera center and rotation matrix based on the number of points in front of the camera.
        Also returns the indices of the inliers in front of the camera.
    """
    best_count = 0
    index = 0

    disambiguated_inliers = []
    C1 = np.zeros((3, 1))
    R1 = np.eye(3)
    # Loop through each camera pose candidate
    for i in range(C_out.shape[0]):
        count = 0
        C = C_out[i].reshape(3, 1)  # shape (3, 1)
        R = R_out[i]
        r3 = R[-1, :].reshape(1, 3)  # shape (1, 3)

        # World points based on the camera pose
        x_set = linear_triangulation(K, R1, C1, R, C, points1, points2)  # shape (n, 4)
        triangulated_set = []

        # Check the number of points in front of the camera
        for j in range(x_set.shape[0]):
            X = x_set[j, :3].reshape(3, 1)  # shape (3, 1)
            if np.dot(r3, (X - C)) > 0 and X[2] > 0:
                count += 1
                triangulated_set.append(j)

        # Update the best count and the index
        if count > best_count:
            best_count = count
            index = i
            disambiguated_inliers = np.array(triangulated_set)

    print("Best Disambiguation Count: ", best_count)

    return C_out[index].reshape(3, 1), R_out[index], disambiguated_inliers
