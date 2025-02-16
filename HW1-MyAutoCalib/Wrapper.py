"""
This module provides functions to perform camera calibration using Zhang's method. 
It includes functions to find chessboard corners, compute intrinsic and extrinsic parameters, 
project image points, and optimize calibration parameters.

Functions:
    - find_chessboard_corners(images, world_pts, pattern_size, output_dir): Find chessboard corners in a list of images and save the corner visualization images.
    - calculate_v_matrix(m, n, H): Create the v_matrix required for intrinsic matrix calculation based on Zhang's method.
    - compute_intrinsic_matrix(H_arr): Compute the intrinsic matrix A from a list of homography matrices using Zhang's method.
    - compute_extrinsic_parameters(A, H_arr): Compute the extrinsic parameters (rotation and translation) from intrinsic matrix A and homography matrices.
    - project_image_points(A, extrinsics, img_pts, world_pts, k1, k2): Project image points using the intrinsic matrix, extrinsic parameters, and distortion coefficients.
    - optimize_parameters(A, img_pts, world_pts, extrinsics): Optimize intrinsic matrix A and distortion coefficients k1, k2 using Levenberg-Marquardt optimization.
    - loss(params, img_pts, world_pts, extrinsics): Loss function for optimization: calculates reprojection errors given parameters.
    - generate_world_points(rows, cols, checker_size): Generate world points for a chessboard pattern.
    - main(): Main function to perform camera calibration.
"""

import cv2
import glob
import numpy as np
import scipy.optimize as opt
import os


def find_chessboard_corners(images, world_pts, pattern_size, output_dir):
    """
    Find chessboard corners in a list of images and save the corner visualization images.

    Parameters:
    - images: List of input images.
    - pattern_size: Tuple specifying the number of inner corners (rows, cols) in the chessboard pattern.
    - output_dir: Directory path to save corner visualization images.

    Returns:
    - H_arr: List of homography matrices.
    - img_pts: List of image points corresponding to chessboard corners.
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


def calculate_v_matrix(m, n, H):
    """
    Create the v_matrix required for intrinsic matrix calculation based on Zhang's method.

    Parameters:
    - m, n: Indices for the v_matrix.
    - H: Homography matrix.

    Returns:
    - v_matrix: Array of shape (6,) containing elements of the v_matrix.
    """
    return np.array(
        [
            H[0][m] * H[0][n],
            H[0][m] * H[1][n] + H[1][m] * H[0][n],
            H[1][m] * H[1][n],
            H[2][m] * H[0][n] + H[0][m] * H[2][n],
            H[2][m] * H[1][n] + H[1][m] * H[2][n],
            H[2][m] * H[2][n],
        ]
    )


def compute_intrinsic_matrix(H_arr):
    """
    Compute the intrinsic matrix A from a list of homography matrices using Zhang's method.

    Parameters:
    - H_arr: List of homography matrices.

    Returns:
    - A: Intrinsic matrix.
    """
    V = []  # Initialize list to store v_matrix values
    for h in H_arr:
        # Append v_matrix for (0,1) and (0,0)-(1,1) to V
        V.append(calculate_v_matrix(0, 1, h))
        V.append(calculate_v_matrix(0, 0, h) - calculate_v_matrix(1, 1, h))
    V = np.array(V)  # Convert V to a numpy array

    # Perform Singular Value Decomposition (SVD) on V
    _, _, vt = np.linalg.svd(V)
    b = vt[-1][
        :
    ]  # Extract the last row of V^T (corresponding to the smallest singular value)

    # Extract elements of the intrinsic parameter matrix B
    B11, B12, B22, B13, B23, B33 = b[0], b[1], b[2], b[3], b[4], b[5]

    # Compute the principal point v0
    v0 = (B12 * B13 - B11 * B23) / (B11 * B22 - B12**2)

    # Compute the scaling factor lambda
    lamda = B33 - (B13**2 + v0 * (B12 * B13 - B11 * B23)) / B11

    # Compute the focal lengths alpha and beta
    alpha = np.sqrt(lamda / B11)
    beta = np.sqrt(lamda * B11 / (B11 * B22 - B12**2))

    # Compute the skew coefficient gamma
    gamma = -1 * B12 * (alpha**2) * beta / lamda

    # Compute the principal point u0
    u0 = (gamma * v0 / beta) - (B13 * (alpha**2) / lamda)

    # Form the intrinsic matrix A
    A = np.array([[alpha, gamma, u0], [0, beta, v0], [0, 0, 1]])

    return A  # Return the intrinsic matrix


def compute_extrinsic_parameters(A, H_arr):
    """
    Compute the extrinsic parameters (rotation and translation) from intrinsic matrix A and homography matrices.

    Parameters:
    - A: Intrinsic matrix.
    - H_arr: List of homography matrices.

    Returns:
    - extrinsics: List of rotation and translation matrices.
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

        # Stack the rotation vectors and translation vector to form the extrinsic matrix
        Rt = np.vstack((r1, r2, t)).T

        # Append the extrinsic matrix to the list
        extrinsics.append(Rt)

    # Return the list of extrinsic matrices
    return extrinsics


def project_image_points(A, extrinsics, img_pts, world_pts, k1, k2):
    """
    Project image points using the intrinsic matrix, extrinsic parameters, and distortion coefficients.

    Parameters:
    - A: Intrinsic matrix.
    - extrinsics: List of rotation and translation matrices.
    - img_pts: List of image points (corners).
    - world_pts: Array of world points in mm.
    - k1: First distortion coefficient.
    - k2: Second distortion coefficient.

    Returns:
    - errors: Array of reprojection errors.
    - projected_points: List of projected points.
    """
    u0, v0 = (
        A[0, 2],
        A[1, 2],
    )  # Extract principal point coordinates from intrinsic matrix
    projected_points = []  # List to store projected points for each image
    errors = []  # List to store reprojection errors for each image

    for i, img_pt in enumerate(img_pts):
        A_Rt = (
            A @ extrinsics[i]
        )  # Compute the product of intrinsic matrix and extrinsic parameters
        error_sum = 0  # Initialize error sum for the current image
        proj_pts_img = []  # List to store projected points for the current image

        for j in range(world_pts.shape[0]):
            # Zhang's method
            world_pt = world_pts[j]  # Get the current world point
            M = np.array([world_pt[0], world_pt[1], 1]).reshape(
                3, 1
            )  # Convert world point to homogeneous coordinates
            proj_pts = (
                extrinsics[i] @ M
            )  # Project world point using extrinsic parameters
            x, y = (
                proj_pts[0][0] / proj_pts[2][0],
                proj_pts[1][0] / proj_pts[2][0],
            )  # Normalize projected points
            N = (
                A_Rt @ M
            )  # Project world point using combined intrinsic and extrinsic parameters
            u, v = N[0][0] / N[2][0], N[1][0] / N[2][0]  # Normalize projected points
            mij = img_pt[j]  # Get the corresponding image point
            mij = np.array(
                [mij[0], mij[1], 1], dtype=np.float32
            )  # Convert image point to homogeneous coordinates
            t = x**2 + y**2  # Compute radial distortion factor
            u_cap = u + (u - u0) * (
                k1 * t + k2 * (t**2)
            )  # Apply radial distortion to u coordinate
            v_cap = v + (v - v0) * (
                k1 * t + k2 * (t**2)
            )  # Apply radial distortion to v coordinate
            proj_pts_img.append(
                [int(u_cap), int(v_cap)]
            )  # Append distorted projected point to list
            mij_cap = np.array(
                [u_cap, v_cap, 1], dtype=np.float32
            )  # Convert distorted projected point to homogeneous coordinates
            error = np.linalg.norm((mij - mij_cap), 2)  # Compute reprojection error
            error_sum += error  # Accumulate error for the current image

        projected_points.append(
            proj_pts_img
        )  # Append projected points for the current image to list
        errors.append(
            error_sum / len(world_pts)
        )  # Compute average reprojection error for the current image

    return (
        np.array(errors),
        projected_points,
    )  # Return reprojection errors and projected points


def optimize_parameters(A, img_pts, world_pts, extrinsics):
    """
    Optimize intrinsic matrix A and distortion coefficients k1, k2 using Levenberg-Marquardt optimization.

    Parameters:
    - A: Initial intrinsic matrix.
    - img_pts: List of image points (corners).
    - world_pts: Array of world points in mm.
    - extrinsics: List of rotation and translation matrices.

    Returns:
    - A_opt: Optimized intrinsic matrix.
    - k1_opt, k2_opt: Optimized distortion coefficients.
    """
    # Extract initial intrinsic parameters from matrix A
    alpha, gamma, u0 = A[0, 0], A[0, 1], A[0, 2]
    beta, v0 = A[1, 1], A[1, 2]

    # Initialize distortion coefficients
    k1, k2 = 0.0, 0.0

    # Perform Levenberg-Marquardt optimization to minimize reprojection errors
    optimized = opt.least_squares(
        fun=loss,  # Loss function to minimize
        x0=[alpha, gamma, beta, u0, v0, k1, k2],  # Initial parameter estimates
        method="lm",  # Optimization method
        args=(
            img_pts,
            world_pts,
            extrinsics,
        ),  # Additional arguments for the loss function
    )

    # Extract optimized parameters from the result
    alpha_opt, gamma_opt, beta_opt, u0_opt, v0_opt, k1_opt, k2_opt = optimized.x

    # Form the optimized intrinsic matrix A
    A_opt = np.array([[alpha_opt, gamma_opt, u0_opt], [0, beta_opt, v0_opt], [0, 0, 1]])

    # Return the optimized intrinsic matrix and distortion coefficients
    return A_opt, k1_opt, k2_opt


def loss(params, img_pts, world_pts, extrinsics):
    """
    Loss function for optimization: calculates reprojection errors given parameters.

    Parameters:
    - params: List of parameters [alpha, gamma, beta, u0, v0, k1, k2].
    - img_pts: List of image points (corners).
    - world_pts: Array of world points in mm.
    - extrinsics: List of rotation and translation matrices.

    Returns:
    - errors: List of reprojection errors.
    """
    # Extract intrinsic parameters and distortion coefficients from params
    alpha, gamma, beta, u0, v0, k1, k2 = params

    # Form the intrinsic matrix A using the extracted parameters
    A = np.array([[alpha, gamma, u0], [0, beta, v0], [0, 0, 1]])

    # Compute reprojection errors using the current parameters
    errors, _ = project_image_points(A, extrinsics, img_pts, world_pts, k1, k2)

    # Return the reprojection errors
    return errors


def generate_world_points(rows, cols, checker_size):
    """
    Generate world points for a chessboard pattern.

    Parameters:
    - rows: Number of rows in the chessboard.
    - cols: Number of columns in the chessboard.
    - checker_size: Size of the checkerboard square in mm.

    Returns:
    - world_pts: Array of world points in mm, shape (num_points, 2).
    """
    # Generate grid points using meshgrid
    grid_x, grid_y = np.meshgrid(range(cols), range(rows))

    # Reshape and stack grid coordinates
    world_pts = np.hstack((grid_x.reshape(-1, 1), grid_y.reshape(-1, 1)))

    # Scale coordinates to real-world size
    world_pts = world_pts.astype(np.float32) * checker_size

    return world_pts


def main():
    """
    Main function to perform camera calibration.
    """
    calibration_imgs_dir = "Data/"
    output_dir = "Output/"

    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    checker_size = 21.5  # chessboard square in mm
    rows, cols = 6, 9  # Number of rows and columns in the chessboard

    # Load calibration images
    images = [
        cv2.imread(file)
        for file in glob.glob(os.path.join(calibration_imgs_dir, "*.jpg"))
    ]

    # Generate world points
    world_pts = generate_world_points(rows, cols, checker_size)

    # Compute homographies and image points
    H_arr, img_pts = find_chessboard_corners(
        images, world_pts, (cols, rows), output_dir
    )

    # Compute initial intrinsic matrix
    A = compute_intrinsic_matrix(H_arr)

    # Compute extrinsic parameters (rotation and translation)
    extrinsics = compute_extrinsic_parameters(A, H_arr)

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
