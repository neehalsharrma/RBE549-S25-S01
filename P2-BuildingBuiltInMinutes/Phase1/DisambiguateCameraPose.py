"""
Module for disambiguating the correct camera pose from a set of possible poses.

This module provides a function to determine the correct camera pose based on the number of points
that lie in front of the camera. It uses linear triangulation to compute the 3D points and checks
their positions relative to the camera poses.

Functions
---------
get_correct_pose(K, C_out, R_out, points1, points2)
    Get the correct pose from the set of poses.
"""

import numpy as np
from LinearTriangulation import linearTriangulation

def get_correct_pose(K, C_out: np.array, R_out: np.array, points1: np.array, points2: np.array):
    """
    Get the correct pose from the set of poses.

    Parameters
    ----------
    K : np.array
        The intrinsic camera matrix in the shape of (3, 3).
    C_out : np.array
        The camera centers in the shape of (4, 3).
    R_out : np.array
        The rotation matrices in the shape of (4, 3, 3).
    points1 : np.array
        The 2D points from the first image in the shape of (n, 2).
    points2 : np.array
        The 2D points from the second image in the shape of (n, 2).

    Returns
    -------
    np.array
        The correct camera center in the shape of (3, 1).
    np.array
        The correct rotation matrix in the shape of (3, 3).
    np.array
        The indices of the inliers in front of the camera.
    """
    best_count = 0
    index = 0

    disambiguated_inliers = []
    C1 = np.zeros((3, 1))
    R1 = np.eye(3)
    
    # Iterate over each possible camera pose
    for i in range(C_out.shape[0]):
        count = 0
        C = C_out[i].reshape(3, 1)  # shape (3, 1)
        R = R_out[i]
        r3 = R[-1, :].reshape(1, 3)  # shape (1, 3)
        
        # Perform linear triangulation to get the 3D points
        x_set = linearTriangulation(K, R1, C1, R, C, points1, points2)  # shape (n, 4)
        triangulated_set = []
        
        # Check the number of points in front of the camera
        for j in range(x_set.shape[0]):
            X = x_set[j, :3].reshape(3, 1)  # shape (3, 1)
            if np.dot(r3, (X - C)) > 0 and X[2] > 0:
                count += 1
                triangulated_set.append(j)

        # Update the best pose if the current one has more points in front of the camera
        if count > best_count:
            best_count = count
            index = i
            disambiguated_inliers = np.array(triangulated_set)
    
    print("Best Disambiguation Count: ", best_count)
    return C_out[index].reshape(3, 1), R_out[index], disambiguated_inliers
