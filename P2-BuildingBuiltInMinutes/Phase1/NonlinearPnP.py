from typing import Any

import numpy as np
from numpy import floating
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation




def nonlinearPnP(K, R1, C1, img_x, world_X):
    """
    Nonlinear Triangulation to estimate the 3D points.
    @ K: The intrinsic camera matrix in the shape of (3, 3)
    @ R1: The rotation matrix of the predicted camera in the shape of (3, 3)
    @ C1: The center of the predicted camera in the shape of (3, 1)
    @ img_x: The homogenized 2D points from the image in the shape of (n, 3)
    @ world_X: The homogenized 3D points in the shape of (n, 3)
    @ return: The refined camera center and rotation matrix.
    """
    def loss_fnc(center_and_quat:np.array, x:np.array, X:np.array) -> floating[Any]:
        """
        Calculate the loss for the non-linear triangulation.
        @ center_and_quat: A (7,1) array containing the center of the camera and the quaternion representation of the rotation.
        @ x: The homogenized 2D points from the image in the shape of (n, 3)
        @ X: The homogenized 3D points in the shape of (n, 3)
        """
        C = center_and_quat[:3].reshape(3, 1)
        quat = center_and_quat[3:]
        R = Rotation.from_quat(quat).as_matrix()
        P = K @ np.hstack((R, -R @ C))
        x_hat = (P @ X.T).T
        x_hat = x_hat / x_hat[:, 2, np.newaxis]  # divide by the last row of P.T @ X
        error = x - x_hat
        return np.linalg.norm(error)


    quaternion = Rotation.from_matrix(R1).as_quat()
    cq = np.append(C1, quaternion)
    print("Running Nonlinear PnP")
    optimized = least_squares(loss_fnc, cq, args=(img_x, world_X))
    print("Finished Nonlinear PnP")
    C_opt = optimized.x[:3].reshape(3, 1)
    quaternion = np.array(optimized.x[3:])
    R_opt = Rotation.from_quat(quaternion).as_matrix()

    return C_opt, R_opt

