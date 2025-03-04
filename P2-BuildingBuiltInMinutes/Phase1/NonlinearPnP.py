"""
This module provides a function for Nonlinear Perspective-n-Point (PnP) problem.
It estimates the 3D points using nonlinear triangulation and refines the camera
center and rotation matrix.

Functions
---------
nonlinear_PnP(K, R1, C1, img_x, world_X)
    Nonlinear Triangulation to estimate the 3D points.
"""

from typing import Any
import numpy as np
from numpy import floating
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def nonlinear_PnP(K, R1, C1, img_x, world_X):
    """
    Nonlinear Triangulation to estimate the 3D points.

    Parameters
    ----------
    K : ndarray of shape (3, 3)
        The intrinsic camera matrix.
    R1 : ndarray of shape (3, 3)
        The rotation matrix of the predicted camera.
    C1 : ndarray of shape (3, 1)
        The center of the predicted camera.
    img_x : ndarray of shape (n, 3)
        The homogenized 2D points from the image.
    world_X : ndarray of shape (n, 3)
        The homogenized 3D points.

    Returns
    -------
    C_opt : ndarray of shape (3, 1)
        The refined camera center.
    R_opt : ndarray of shape (3, 3)
        The refined rotation matrix.
    """

    def loss_fnc(center_and_quat: np.array, x: np.array, X: np.array) -> floating[Any]:
        """
        Calculate the loss for the non-linear triangulation.

        Parameters
        ----------
        center_and_quat : ndarray of shape (7,)
            A 1D array containing the center of the camera and the quaternion representation of the rotation.
        x : ndarray of shape (n, 3)
            The homogenized 2D points from the image.
        X : ndarray of shape (n, 3)
            The homogenized 3D points.

        Returns
        -------
        error : float
            The norm of the error between the observed and predicted 2D points.
        """
        # Extract camera center and quaternion from the input array
        C = center_and_quat[:3].reshape(3, 1)
        quat = center_and_quat[3:]

        # Convert quaternion to rotation matrix
        R = Rotation.from_quat(quat).as_matrix()

        # Compute the projection matrix
        P = K @ np.hstack((R, -R @ C))

        # Project the 3D points to 2D
        x_hat = (P @ X.T).T
        x_hat = x_hat / x_hat[:, 2, np.newaxis]  # Normalize by the third coordinate

        # Compute the reprojection error
        error = x - x_hat
        return np.linalg.norm(error)

    # Convert initial rotation matrix to quaternion
    quaternion = Rotation.from_matrix(R1).as_quat()

    # Combine camera center and quaternion into a single array
    cq = np.append(C1, quaternion)

    print("Running Nonlinear PnP")

    # Optimize the camera parameters using least squares
    optimized = least_squares(loss_fnc, cq, args=(img_x, world_X))

    print("Finished Nonlinear PnP")

    # Extract optimized camera center and rotation matrix
    C_opt = optimized.x[:3].reshape(3, 1)
    quaternion = np.array(optimized.x[3:])
    R_opt = Rotation.from_quat(quaternion).as_matrix()

    return C_opt, R_opt
