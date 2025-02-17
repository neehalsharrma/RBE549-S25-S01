"""
This module provides functions to perform automatic camera calibration.

Functions:
    find_chessboard_corners(images, world_pts, pattern_size, output_dir)
    calculate_v_matrix(m, n, H)
    compute_intrinsic_matrix(H_arr)
    compute_extrinsic_parameters(A, H_arr)
    project_image_points(A, extrinsics, img_pts, world_pts, k1, k2)
    optimize_parameters(A, img_pts, world_pts, extrinsics)
    loss_function(params, img_pts, world_pts, extrinsics)
    generate_world_points(rows, cols, checker_size)
    main()

TODO: Figure out why the type hints are not working in the functions.
"""

import cv2
import glob
import numpy as np
import scipy.optimize as opt
import os
from typing import List, Tuple


def find_chessboard_corners(
    images: List[np.ndarray],
    world_pts: np.ndarray,
    pattern_size: Tuple[int, int],
    output_dir: str,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Find chessboard corners in a list of images and save the corner visualization images.

    Args:
        images (List[np.ndarray]): List of input images.
        world_pts (np.ndarray): Array of world points in mm.
        pattern_size (Tuple[int, int]): Number of inner corners (rows, cols) in the chessboard pattern.
        output_dir (str): Directory path to save corner visualization images.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]: List of homography matrices and image points.
    """
    # Create a copy of the images to avoid modifying the original images
    copy = np.copy(images)
    H_arr = []
    img_pts = []

    # Iterate through each image to find chessboard corners
    for i, img in enumerate(copy):
        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners in the grayscale image
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret:
            # Reshape corners to a 2D array
            corners = corners.reshape(-1, 2)

            # Compute the homography matrix using the world points and detected corners
            H, _ = cv2.findHomography(world_pts, corners, cv2.RANSAC, 5.0)
            H_arr.append(H)
            img_pts.append(corners)

            # Draw the detected corners on the image
            cv2.drawChessboardCorners(img, pattern_size, corners, True)

            # Resize the image for visualization and save it
            img = cv2.resize(img, (int(img.shape[1] / 3), int(img.shape[0] / 3)))
            cv2.imwrite(os.path.join(output_dir, f"{i}_corners.png"), img)

    # Return the list of homography matrices and image points
    return H_arr, img_pts


def calculate_v_matrix(m: int, n: int, h: np.ndarray) -> np.ndarray:
    """
    Create the v_matrix required for intrinsic matrix calculation based on Zhang's method.

    Args:
        m (int): Index m for the v_matrix.
        n (int): Index n for the v_matrix.
        H (np.ndarray): Homography matrix.

    Returns:
        np.ndarray: Array of shape (6,) containing elements of the v_matrix.
    """
    return np.array(
        [
            h[0][m] * h[0][n],
            h[0][m] * h[1][n] + h[1][m] * h[0][n],
            h[1][m] * h[1][n],
            h[2][m] * h[0][n] + h[0][m] * h[2][n],
            h[2][m] * h[1][n] + h[1][m] * h[2][n],
            h[2][m] * h[2][n],
        ],
        dtype=np.float32,
    )


def camera_intrinsics(h_arr: List[np.ndarray]) -> np.ndarray:
    """
    Compute the intrinsic matrix A from a list of homography matrices using Zhang's method.

    Args:
        H_arr (List[np.ndarray]): List of homography matrices.

    Returns:
        np.ndarray: Intrinsic matrix.
    """
    # Initialize the V matrix
    v = []

    # Populate the V matrix using homography matrices
    for h in h_arr:
        v.append(calculate_v_matrix(0, 1, h))
        v.append(calculate_v_matrix(0, 0, h) - calculate_v_matrix(1, 1, h))

    # Convert V to a numpy array
    v = np.array(v)

    # Check if V matrix is empty
    if v.size == 0:
        raise ValueError(
            "V matrix is empty. Ensure that homographies are correctly computed."
        )

    # Reshape V if it is a 1D array
    if v.ndim == 1:
        v = v.reshape(1, -1)

    # Perform Singular Value Decomposition (SVD) on V
    _, _, vt = np.linalg.svd(v)

    # Extract the last row of V^T (corresponding to the smallest singular value)
    b = vt[-1][:]

    # Extract elements of the B matrix from vector b
    B11, B12, B22, B13, B23, B33 = b[0], b[1], b[2], b[3], b[4], b[5]

    # Compute the intrinsic parameters using the elements of B
    v0 = (B12 * B13 - B11 * B23) / (B11 * B22 - B12**2)
    lamda = B33 - (B13**2 + v0 * (B12 * B13 - B11 * B23)) / B11
    alpha = np.sqrt(lamda / B11)
    beta = np.sqrt(lamda * B11 / (B11 * B22 - B12**2))
    gamma = -1 * B12 * (alpha**2) * beta / lamda
    u0 = (gamma * v0 / beta) - (B13 * (alpha**2) / lamda)

    # Construct the intrinsic matrix A
    A = np.array([[alpha, gamma, u0], [0, beta, v0], [0, 0, 1]])

    # Return the intrinsic matrix
    return A


def camera_extrinsics(A: np.ndarray, H_arr: List[np.ndarray]) -> List[np.ndarray]:
    """
    Compute the extrinsic parameters (rotation and translation) from intrinsic matrix A and homography matrices.

    Args:
        A (np.ndarray): Intrinsic matrix.
        H_arr (List[np.ndarray]): List of homography matrices.

    Returns:
        List[np.ndarray]: List of rotation and translation matrices.
    """
    # Compute the inverse of the intrinsic matrix A
    A_inv = np.linalg.inv(A)
    extrinsics = []

    # Iterate through each homography matrix to compute extrinsic parameters
    for H in H_arr:
        # Extract columns from the homography matrix
        h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]

        # Compute the scaling factor lambda
        lamda_e = 1 / np.linalg.norm((A_inv @ h1), 2)

        # Compute the rotation vectors r1 and r2
        r1 = lamda_e * (A_inv @ h1)
        r2 = lamda_e * (A_inv @ h2)

        # Compute the translation vector t
        t = lamda_e * (A_inv @ h3)

        # Form the extrinsic matrix [R|t] by stacking r1, r2, and t
        Rt = np.vstack((r1, r2, t)).T

        # Append the extrinsic matrix to the list
        extrinsics.append(Rt)

    # Return the list of extrinsic matrices
    return extrinsics


def project_image_points(
    A: np.ndarray,
    extrinsics: List[np.ndarray],
    img_pts: List[np.ndarray],
    world_pts: np.ndarray,
    k1: float,
    k2: float,
) -> Tuple[np.ndarray, List[List[Tuple[int, int]]]]:
    """
    Project image points using the intrinsic matrix, extrinsic parameters, and distortion coefficients.

    Args:
        A (np.ndarray): Intrinsic matrix.
        extrinsics (List[np.ndarray]): List of rotation and translation matrices.
        img_pts (List[np.ndarray]): List of image points (corners).
        world_pts (np.ndarray): Array of world points in mm.
        k1 (float): First distortion coefficient.
        k2 (float): Second distortion coefficient.

    Returns:
        Tuple[np.ndarray, List[List[Tuple[int, int]]]]: Array of reprojection errors and list of projected points.
    """
    # Extract the principal point coordinates from the intrinsic matrix
    u0, v0 = A[0, 2], A[1, 2]
    projected_points = []
    errors = []

    # Iterate through each set of image points
    for i, img_pt in enumerate(img_pts):
        # Compute the product of the intrinsic matrix and the extrinsic matrix
        A_Rt = A @ extrinsics[i]
        error_sum = 0
        proj_pts_img = []

        # Iterate through each world point
        for j in range(world_pts.shape[0]):
            world_pt = world_pts[j]
            M = np.array([world_pt[0], world_pt[1], 1]).reshape(3, 1)

            # Project the world point using the extrinsic matrix
            proj_pts = extrinsics[i] @ M
            x, y = proj_pts[0][0] / proj_pts[2][0], proj_pts[1][0] / proj_pts[2][0]

            # Project the world point using the combined intrinsic and extrinsic matrix
            N = A_Rt @ M
            u, v = N[0][0] / N[2][0], N[1][0] / N[2][0]

            # Get the corresponding image point
            mij = img_pt[j]
            mij = np.array([mij[0], mij[1], 1], dtype=np.float32)

            # Compute the radial distortion
            t = x**2 + y**2
            u_cap = u + (u - u0) * (k1 * t + k2 * (t**2))
            v_cap = v + (v - v0) * (k1 * t + k2 * (t**2))

            # Append the projected point to the list
            proj_pts_img.append([int(u_cap), int(v_cap)])

            # Compute the reprojection error
            mij_cap = np.array([u_cap, v_cap, 1], dtype=np.float32)
            error = np.linalg.norm((mij - mij_cap), 2)
            error_sum += error

        # Append the projected points and average error for the current image
        projected_points.append(proj_pts_img)
        errors.append(error_sum / len(world_pts))

    # Return the array of reprojection errors and the list of projected points
    return np.array(errors), projected_points


def optimize_parameters(
    A: np.ndarray,
    img_pts: List[np.ndarray],
    world_pts: np.ndarray,
    extrinsics: List[np.ndarray],
) -> Tuple[np.ndarray, float, float]:
    """
    Optimize intrinsic matrix A and distortion coefficients k1, k2 using Levenberg-Marquardt optimization.

    Args:
        A (np.ndarray): Initial intrinsic matrix.
        img_pts (List[np.ndarray]): List of image points (corners).
        world_pts (np.ndarray): Array of world points in mm.
        extrinsics (List[np.ndarray]): List of rotation and translation matrices.

    Returns:
        Tuple[np.ndarray, float, float]: Optimized intrinsic matrix and distortion coefficients.
    """
    # Extract initial intrinsic parameters from matrix A
    alpha, gamma, u0 = A[0, 0], A[0, 1], A[0, 2]
    beta, v0 = A[1, 1], A[1, 2]

    # Initialize distortion coefficients
    k1, k2 = 0.0, 0.0

    # Perform Levenberg-Marquardt optimization to minimize the reprojection error
    optimized = opt.least_squares(
        fun=loss_function,
        x0=[alpha, gamma, beta, u0, v0, k1, k2],
        method="lm",
        args=(img_pts, world_pts, extrinsics),
    )

    # Extract optimized parameters from the result
    alpha_opt, gamma_opt, beta_opt, u0_opt, v0_opt, k1_opt, k2_opt = optimized.x

    # Construct the optimized intrinsic matrix A
    A_opt = np.array([[alpha_opt, gamma_opt, u0_opt], [0, beta_opt, v0_opt], [0, 0, 1]])

    # Return the optimized intrinsic matrix and distortion coefficients
    return A_opt, k1_opt, k2_opt


def loss_function(
    params: List[float],
    img_pts: List[np.ndarray],
    world_pts: np.ndarray,
    extrinsics: List[np.ndarray],
) -> np.ndarray:
    """
    Loss function for optimization: calculates reprojection errors given parameters.

    Args:
        params (List[float]): List of parameters [alpha, gamma, beta, u0, v0, k1, k2].
        img_pts (List[np.ndarray]): List of image points (corners).
        world_pts (np.ndarray): Array of world points in mm.
        extrinsics (List[np.ndarray]): List of rotation and translation matrices.

    Returns:
        np.ndarray: List of reprojection errors.
    """
    # Extract intrinsic parameters and distortion coefficients from params
    alpha, gamma, beta, u0, v0, k1, k2 = params

    # Construct the intrinsic matrix A using the extracted parameters
    A = np.array([[alpha, gamma, u0], [0, beta, v0], [0, 0, 1]])

    # Compute reprojection errors using the current parameters
    errors, _ = project_image_points(A, extrinsics, img_pts, world_pts, k1, k2)

    # Return the reprojection errors
    return errors


def create_checkerboard_points(rows: int, cols: int, checker_size: float) -> np.ndarray:
    """
    Generate world points for a chessboard pattern.

    Args:
        rows (int): Number of rows in the chessboard.
        cols (int): Number of columns in the chessboard.
        checker_size (float): Size of the checkerboard square in mm.

    Returns:
        np.ndarray: Array of world points in mm, shape (num_points, 2).
    """
    # Create a grid of points representing the chessboard corners
    grid_x, grid_y = np.meshgrid(range(cols), range(rows))

    # Combine the grid points into a single array of shape (num_points, 2)
    world_pts = np.hstack((grid_x.reshape(-1, 1), grid_y.reshape(-1, 1)))

    # Convert the world points to float32 and scale by the checker size
    world_pts = world_pts.astype(np.float32) * checker_size

    # Return the array of world points
    return world_pts


def main() -> None:
    """
    Main function to perform camera calibration.
    """
    calibration_imgs_dir = "Data/"
    output_dir = "Output/"

    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    checker_size = 21.5  # chessboard square in mm
    rows, cols = 9, 6  # Number of rows and columns in the chessboard

    # Load calibration images
    images = [
        cv2.imread(file)
        for file in glob.glob(os.path.join(calibration_imgs_dir, "*.jpg"))
    ]

    # Generate world points
    world_pts = create_checkerboard_points(rows, cols, checker_size)

    # Compute homographies and image points
    H_arr, img_pts = find_chessboard_corners(
        images, world_pts, (rows, cols), output_dir
    )

    # Check if H_arr is empty
    if not H_arr:
        raise ValueError(
            "No homographies found. Ensure that chessboard corners are detected in the images."
        )

    # Compute initial intrinsic matrix
    A = camera_intrinsics(H_arr)

    # Compute extrinsic parameters (rotation and translation)
    extrinsics = camera_extrinsics(A, H_arr)

    # Optimize intrinsic matrix A and distortion coefficients k1, k2
    A_opt, k1_opt, k2_opt = optimize_parameters(A, img_pts, world_pts, extrinsics)

    # Compute reprojection errors with optimized parameters
    rp_avg_error, reproj_pts = project_image_points(
        A_opt, extrinsics, img_pts, world_pts, k1_opt, k2_opt
    )

    # Print results
    print("Optimized Intrinsics:\n", A_opt)
    print("Optimized Distortion Coefficients (k1, k2):\n", k1_opt, k2_opt)
    print("Average Reprojection Error: ", rp_avg_error)

    # Draw circles on reprojected points in images and save
    for i, img in enumerate(images):
        for pt in reproj_pts[i]:
            cv2.circle(img, (pt[0], pt[1]), 10, (0, 0, 255), -1)
        cv2.imwrite(os.path.join(output_dir, f"rectified_{i}.png"), img)


if __name__ == "__main__":
    main()
