"""
This module provides functions to perform Linear Perspective-n-Point (PnP) 
and PnP RANSAC for estimating the camera pose from 2D-3D point correspondences.

Functions
---------
calc_loss(x, P, X)
    Calculate the loss for the non-linear triangulation.
get_inliers(x, X, P, threshold)
    Get inliers based on the reprojection error.
calc_inliers(x, X, P, threshold)
    Calculate the number of inliers based on the reprojection error.
get_equation(point3D, point2D)
    Get the linear equation for the PnP problem.
linear_PnP(K, points2D, points3D)
    Linear PnP to estimate the 3D points.
PnPRansac(K, points2D, points3D, threshold, acc_thresh)
    Perform PnP RANSAC to estimate the camera pose.
"""

import numpy as np


def calc_loss(x: np.array, P: np.array, X: np.array) -> float:
    """
    Calculate the loss for the non-linear triangulation.

    Parameters
    ----------
    x : np.array
        The 2D points from the image in the shape of (n, 3) since it's homogenized.
    P : np.array
        The projection matrix in the shape of (3, 4).
    X : np.array
        The 3D points in the shape of (n, 3).

    Returns
    -------
    float
        The loss for the non-linear triangulation.
    """
    # Project the 3D points to 2D using the projection matrix
    x_hat = P @ X.T
    x_hat = x_hat / x_hat[:, 2, np.newaxis]  # Normalize by the last row of P.T @ X
    error = x - x_hat  # Calculate the error
    return np.linalg.norm(error)  # Return the norm of the error


def get_inliers(x, X, P, threshold):
    """
    Get inliers based on the reprojection error.

    Parameters
    ----------
    x : np.array
        The 2D points from the image.
    X : np.array
        The 3D points.
    P : np.array
        The projection matrix.
    threshold : float
        The threshold to determine inliers.

    Returns
    -------
    points2d : list
        List of inlier 2D points.
    points3d : list
        List of inlier 3D points.
    """
    points2d = []  # Initialize list to store inlier 2D points
    points3d = []  # Initialize list to store inlier 3D points
    for i in range(x.shape[0]):
        # Check if the reprojection error for the point is below the threshold
        if calc_loss(x[i], P, X[i]) < threshold:
            points2d.append(x[i])  # Add the 2D point to the inliers list
            points3d.append(X[i])  # Add the 3D point to the inliers list
    return points2d, points3d  # Return the lists of inlier 2D and 3D points


def calc_inliers(x, X, P, threshold):
    """
    Calculate the number of inliers based on the reprojection error.

    Parameters
    ----------
    x : np.array
        The 2D points from the image.
    X : np.array
        The 3D points.
    P : np.array
        The projection matrix.
    threshold : float
        The threshold to determine inliers.

    Returns
    -------
    int
        The number of inliers.
    """
    tot = 0  # Initialize the total number of inliers
    for i in range(x.shape[0]):
        # Check if the reprojection error for the point is below the threshold
        if calc_loss(x[i], P, X[i]) < threshold:
            tot += 1  # Increment the inlier count
    return tot  # Return the total number of inliers


def get_equation(point3D, point2D):
    """
    Get the linear equation for the PnP problem.

    Parameters
    ----------
    point3D : np.array
        A single 3D point in homogeneous coordinates (X, Y, Z, 1).
    point2D : np.array
        A single 2D point in homogeneous coordinates (x, y, 1).

    Returns
    -------
    np.array
        The linear equation in the form of a matrix with shape (2, 12).
    """
    # Extract the coordinates from the 3D and 2D points
    X, Y, Z, _ = point3D
    x, y, _ = point2D
    
    # Construct the linear equations for the PnP problem
    # Each 3D-2D correspondence gives two equations
    return np.array(
        [
            [X, Y, Z, 1, 0, 0, 0, 0, -x * X, -x * Y, -x * Z, -x],  # Equation for x-coordinate
            [0, 0, 0, 0, X, Y, Z, 1, -y * X, -y * Y, -y * Z, -y],  # Equation for y-coordinate
        ]
    )


def linear_PnP(K, points2D, points3D):
    """
    Linear PnP to estimate the 3D points.

    Parameters
    ----------
    K : np.array
        The intrinsic camera matrix in the shape of (3, 3).
    points2D : np.array
        The 2D points from the image in the shape of (n, 2).
    points3D : np.array
        The 3D points in the shape of (n, 3).

    Returns
    -------
    R : np.array
        The estimated rotation matrix of the camera.
    C : np.array
        The estimated camera center.
    """
    # Number of points
    n = points3D.shape[0]
    A = None  # Initialize the matrix A

    # Construct the matrix A using the linear equations from the 2D-3D correspondences
    for i in range(n):
        a = get_equation(points3D[i], points2D[i])
        if i > 0:
            A = np.vstack((A, a))  # Stack the equations vertically
        else:
            A = a  # Initialize A with the first equation

    # Perform Singular Value Decomposition (SVD) on A
    U, S, VT = np.linalg.svd(A)
    P = VT.T[-1, :].reshape((3, 4))  # The last row of V.T gives the solution

    # Compute the inverse of the intrinsic matrix K
    inv_K = np.linalg.inv(K)

    # Extract the rotation matrix R and translation vector T from P
    R = inv_K @ P[:, :3]  # Inverse of K times the first 3 columns of P
    T = inv_K @ P[:, -1]  # Inverse of K times the last column of P

    # Perform SVD on R to ensure it is a valid rotation matrix
    UR, DR, VTR = np.linalg.svd(R)
    R = UR @ VTR

    # Ensure the determinant of R is positive
    if np.linalg.det(R) < 0:
        R = -R

    # Compute the camera center C
    C = -R.T @ T

    return R, C


def PnPRansac(K, points2D, points3D, threshold=float(5), acc_thresh=0.85):
    """
    Perform PnP RANSAC to estimate the camera pose.

    Parameters
    ----------
    K : np.array
        The intrinsic camera matrix in the shape of (3, 3).
    points2D : np.array
        The 2D points from the image in the shape of (n, 2).
    points3D : np.array
        The 3D points in the shape of (n, 3).
    threshold : float, optional
        The threshold to determine inliers (default is 5).
    acc_thresh : float, optional
        The accuracy threshold to stop RANSAC (default is 0.85).

    Returns
    -------
    best_R : np.array
        The best estimated rotation matrix of the camera.
    best_C : np.array
        The best estimated camera center.
    """
    rng = np.random.default_rng()  # Initialize random number generator
    best_percent = 0  # Initialize the best percentage of inliers

    tot_size = points2D.shape[0]  # Total number of points
    while best_percent < acc_thresh:  # Continue until the accuracy threshold is met
        # Randomly sample 6 points
        random_samples = rng.integers(0, tot_size, size=6, replace=False)
        point2D = points2D[random_samples]
        point3D = points3D[random_samples]
        
        # Estimate the pose using the sampled points
        R, C = linear_PnP(K, point2D, point3D)
        P = np.hstack((R, C.reshape(-1, 1)))  # Form the projection matrix
        
        # Calculate the number of inliers
        num_inliers = calc_inliers(points2D, points3D, P, threshold)
        percent_match = num_inliers / tot_size  # Calculate the percentage of inliers
        
        # If a better match is found, update the best match
        if percent_match > best_percent:
            best_percent = percent_match
            best_points2D, best_points3D = get_inliers(points2D, points3D, P, threshold)
            print(f"Best Percent: {best_percent}")

    print(f"Original No. of features: {tot_size}")
    print(f"No. of inliers: {len(best_points2D)}")
    
    # Recompute the pose using all inliers
    best_R, best_C = linear_PnP(K, np.array(best_points2D), np.array(best_points3D))

    return best_R, best_C
