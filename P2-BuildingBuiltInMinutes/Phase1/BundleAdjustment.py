"""
Module for performing bundle adjustment to refine camera poses and 3D points.

This module provides functions to perform bundle adjustment, which is an optimization
technique used to refine the camera parameters and 3D point positions to minimize
the re-projection error.

Functions
---------
bundle_adjustment(C_matrices, R_matrices, K, visibility_matrix, world_X_points, points_2d)
    Refines the camera centers, rotation matrices, and 3D points.
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation
import sys
sys.dont_write_bytecode = True


def bundle_adjustment(
    C_matrices: list[np.array],
    R_matrices: list[np.array],
    K: list[np.array],
    visibility_matrix: np.array,
    world_X_points: np.array,
    points_2d: list[np.array],
) -> tuple[list[np.array], list[np.array], np.array]:
    """
    Bundle Adjustment to refine the camera pose and the 3D points.

    Parameters
    ----------
    C_matrices : list of np.array
        The camera centers in the shape of list[(n, 3)] of length num_imgs.
    R_matrices : list of np.array
        The rotation matrices in the shape of list[(3, 3)] of length num_imgs.
    K : list of np.array
        The intrinsic camera matrix in the shape of (3, 3).
    visibility_matrix : np.array
        The binary mask visibility matrix in the shape of (num_imgs, n) where Vij
        is one if the jth point is visible from the ith camera and zero otherwise.
    world_X_points : np.array
        The homogenized 3D points for the world coordinate system (n, 4).
    points_2d : list of np.array
        The homogenized 2D points from the image in the shape of list[(n, 3)] of length num_imgs.

    Returns
    -------
    tuple
        The refined camera centers, rotation matrices, and the 3D points.
    """

    def unpack_params(
        n_cams: int, params: np.array
    ) -> tuple[np.array, np.array, np.array]:
        """
        Unpack the optimized parameters.

        Parameters
        ----------
        n_cams : int
            The number of cameras.
        params : np.array
            The 1D array containing the parameters of the cameras, rotation matrices, and 3D points.
            C Matrix: (n_cameras, 3)
            R Matrix: (n_cameras, 4)
            3D Points: (n_points, 4)

        Returns
        -------
        tuple
            The camera centers, rotation matrices, and 3D points.
        """
        opt_cam_params = params[: n_cams * 7].reshape((n_cams, 7))
        c_params = params[:, :3].reshape((n_cams, 3))
        r_params = opt_cam_params[:, 3:].reshape((n_cams, 4))
        points_3d = params[n_cams * 7 :]
        return c_params, r_params, points_3d

    def loss_func(params: np.array, n_cameras: int, img_pts: list[int]) -> np.array:
        """
        Calculate the loss function for the bundle adjustment.

        Parameters
        ----------
        params : np.array
            The parameters to be optimized. Contains the flattened C matrices, the flattened R matrices as quaternions, and the homogenized 3D points.
        n_cameras : int
            The number of cameras.
        img_pts : list of int
            The number of points in each image.

        Returns
        -------
        np.array
            The concatenated pose error and projection error.
        """
        # Unpack the parameters into camera centers, rotation matrices, and 3D points
        c_mats, r_mats, points_3d = unpack_params(n_cameras, params)
        points_proj = []
        pose_error_vec = []
        offset = 0
        error_vec = []
        for cam, n_points in enumerate(img_pts):
            # Get the camera center and rotation matrix for the current camera
            C = c_mats[cam].reshape(3, 1)
            R = Rotation.from_quat(r_mats[cam]).as_matrix().reshape(3, 3)
            # Compute the projection matrix
            P = K @ np.hstack((R, -R @ C))
            # Extract the 2D points
            x_cam = points_2d[cam]
            # Extract the 3D world points
            X = points_3d[offset : offset + n_points * 4].reshape(n_points, 4)
            # Project the 3D points to 2D
            x_hat = P @ X.T
            # Get the projection error
            errors = x_cam - x_hat
            error_vec = error_vec.append(errors)
            # Calculate the pose error as a scalar and apply it uniformly to the translation and rotation matrices
            pose_error = np.ones(7) * np.linalg.norm(errors)
            pose_error_vec = pose_error_vec.append(pose_error)
            offset += n_points * 4

        # Concatenate the pose error and projection error
        error = np.array(error_vec).ravel()
        pose_error = np.array(pose_error_vec).ravel()
        return np.concatenate((pose_error, error))

    # Gets the number of cameras and points
    n_cameras = len(C_matrices)
    n_points_per_img = [points_2d[i].shape[0] for i in range(n_cameras)]
    # Takes the camera centers and rotation matrices and flattens them
    poses = np.array(
        [
            np.hstack(
                (C_matrices[i].ravel(), Rotation.from_matrix(R_matrices[i]).as_quat())
            )
            for i in range(n_cameras)
        ]
    )
    # Flattens the 3D points
    points = np.array([world_X_points[i].ravel() for i in range(n_cameras)])

    # Concatenate the initial parameters into a single array
    X0 = np.concatenate((poses.ravel(), points.ravel()))
    # Runs the optimization
    print("Running Bundle Adjustment")
    optimized = least_squares(
        loss_func,
        X0,
        args=(n_cameras, n_points_per_img),
        method="lm",
        jac_sparsity=visibility_matrix,
    )
    print("Finished Bundle Adjustment")
    # Extracts the optimized parameters
    optimized_params = optimized.x
    # Extracts the image data
    return unpack_params(n_cameras, optimized_params)
