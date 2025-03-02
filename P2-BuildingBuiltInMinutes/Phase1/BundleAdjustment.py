import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation


def bundle_adjustment(C_matrices: list[np.array], R_matrices: list[np.array], K: list[np.array],
                      visibility_matrix: np.array,
                      world_X_points: np.array,
                      points_2d: list[np.array]) -> tuple[list[np.array], list[np.array], np.array]:
    """
    Bundle Adjustment to refine the camera pose and the 3D points.
    @ C_matrices: The camera centers in the shape of list[(n, 3)] of length num_imgs
    @ R_matrices: The rotation matrices in the shape of list[(3, 3)] of length num_imgs
    @ K: The intrinsic camera matrix in the shape of (3, 3)
    @ world_X_points: The homogenized 3D points for the world coordinate system (n, 4)
    @ points_2d: The homogenized 2D points from the image in the shape of list[(n, 3)] of length num_imgs
    @ visibility_matrix: The binary mask visibility matrix in the shape of (num_imgs, n) where Vij
                         is one if the jth point is visible from the ith camera and zero otherwise
    @  K: The intrinsic camera matrix in the shape of (3, 3)
    returns the refined camera centers, rotation matrices, and the 3D points.
    """

    def unpack_params(n_cams: int, params:np.array) -> tuple[np.array, np.array, np.array]:
        """
        Unpack the optimized parameters.
        @ n_cams: The number of cameras
        @params: The 1D array contain the parameters of the cameras, rotation matrices, and 3D points
                C Matrix: (n_cameras, 3)
                R Matrix: (n_cameras, 4)
                3D Points: (n_points, 4)
        @ return: The camera centers, rotation matrices, and 3D points

        """
        opt_cam_params = params[:n_cams * 7].reshape((n_cams, 7))
        c_params = params[:, :3].reshape((n_cams, 3))
        r_params = opt_cam_params[:, 3:].reshape((n_cams, 4))
        points_3d = params[n_cams * 7:]
        return c_params, r_params, points_3d

    def loss_func(params: np.array, n_cameras: int, img_pts: list[int]) -> np.array:
        """
        Calculate the loss function for the bundle adjustment.
        @ params: The parameters to be optimized.
            Contains: The flattened C matrices, the flattened R matrices as quaternions, and the homogenized3D points
        @ n_cameras: The number of cameras
        @ n_points: The number of points
        @ camera_indices: The indices of the cameras
        @ point_indices: The indices of the points
        """
        c_mats, r_mats, points_3d = unpack_params(n_cameras, params)
        points_proj = []
        pose_error_vec = []
        offset = 0
        error_vec = []
        for cam, n_points in enumerate(img_pts):
            C = c_mats[cam].reshape(3, 1)
            R = Rotation.from_quat(r_mats[cam]).as_matrix().reshape(3, 3)
            P = K @ np.hstack((R, -R @ C))
            # Extract the 2D points
            x_cam = points_2d[cam]
            # Extract the 3D world points
            X = points_3d[offset:offset + n_points * 4].reshape(n_points, 4)
            x_hat = P @ X.T
            # Get the projection error
            errors = x_cam - x_hat
            error_vec = error_vec.append(errors)
            # Calculate the pose error as a scalar and apply it uniformly to the translation and rotation matrices
            pose_error = np.ones(7) * np.linalg.norm(errors)
            pose_error_vec = pose_error_vec.append(pose_error)
            offset += n_points * 4

        error = np.array(error_vec).ravel()
        pose_error = np.array(pose_error_vec).ravel()
        return np.concatenate((pose_error, error))


        # Gets the number of cameras and points
    n_cameras = len(C_matrices)
    n_points_per_img = [points_2d[i].shape[0] for i in range(n_cameras)]
    # Takes the camera centers and rotation matrices and flattens them

    poses = np.array([np.hstack((C_matrices[i].ravel(), Rotation.from_matrix(R_matrices[i]).as_quat())) for i in range(n_cameras)])
    # Flattens the 3D points
    points = np.array([world_X_points[i].ravel() for i in range(n_cameras)])

    X0 = np.concatenate((poses.ravel(), points.ravel()))
    # Runs the optimization
    print("Running Bundle Adjustment")
    optimized = least_squares(loss_func, X0, args=(n_cameras, n_points_per_img), method='lm', jac_sparsity=visibility_matrix)
    print("Finished Bundle Adjustment")
    # Extracts the optimized parameters
    optimized_params = optimized.x
    # Extracts the image data
    return unpack_params(n_cameras, optimized_params)


