"""
Module for performing Perspective-n-Point (PnP) with RANSAC.

This module includes functions to calculate the error function for the PnP problem,
determine inliers and outliers based on a threshold, and perform PnP with RANSAC
to estimate the camera pose.

Functions
---------
error_func(P, X, x, threshold)
    Calculate the error function for the PnP problem.
get_inliers(P, X, x, threshold)
    Get the inliers given the threshold.
PNP_RANSAC(world_X, img_x, K, threshold=100, acc_thresh=0.85, max_iters=1000)
    Perform PnP with RANSAC to estimate the camera pose.
"""

import sys
sys.dont_write_bytecode = True

import numpy as np
from LinearPnp import linear_PnP

def error_func(P: np.array, X: np.array, x: np.array, threshold: float) -> float:
    """
    Calculate the error function for the PnP problem.

    Parameters
    ----------
    P : np.array
        The projection matrix in the shape of (3, 4).
    X : np.array
        The homogenized 3D points in the shape of (n, 4).
    x : np.array
        The homogenized 2D points in the shape of (n, 3).
    threshold : float
        The threshold to determine inliers.

    Returns
    -------
    float
        The number of inliers.
    """
    # Project the 3D points to 2D using the projection matrix
    x_hat = (P @ X.T).T
    x_hat = x_hat / x_hat[:, 2, np.newaxis]  # Normalize by the third coordinate
    diffs = x - x_hat  # Calculate the difference between actual and projected points
    # Calculate the Euclidean distance of the reprojection error
    errors = np.linalg.norm(diffs, axis=1)  # Shape: (n,)
    return np.sum(errors < threshold)  # Count the number of inliers

def get_inliers(P, X, x, threshold: float) -> tuple[np.array, np.array]:
    """
    Get the inliers given the threshold.

    Parameters
    ----------
    P : np.array
        The projection matrix in the shape of (3, 4).
    X : np.array
        The homogenized 3D points in the shape of (n, 4).
    x : np.array
        The homogenized 2D points in the shape of (n, 3).
    threshold : float
        The threshold to determine inliers.

    Returns
    -------
    tuple[np.array, np.array]
        The inlier and outlier indices.
    """
    # Project the 3D points to 2D using the projection matrix
    x_hat = (P @ X.T).T
    x_hat = x_hat / x_hat[:, 2, np.newaxis]  # Normalize by the third coordinate
    diffs = x - x_hat  # Calculate the difference between actual and projected points
    # Calculate the Euclidean distance of the reprojection error
    errors = np.linalg.norm(diffs, axis=1)  # Shape: (n,)
    inliers = np.argwhere(errors < threshold).flatten()  # Indices of inliers
    outliers = np.argwhere(errors >= threshold).flatten()  # Indices of outliers
    return inliers, outliers

def PNP_RANSAC(world_X, img_x, K, threshold=100, acc_thresh=0.85, max_iters=1000) -> tuple[np.array, np.array, np.array, np.array]:
    """
    Perform PnP with RANSAC to estimate the camera pose.

    Parameters
    ----------
    world_X : np.array
        The homogenized 3D points in the shape of (n, 4).
    img_x : np.array
        The homogenized 2D points in the shape of (n, 3).
    K : np.array
        The intrinsic camera matrix in the shape of (3, 3).
    threshold : float, optional
        The threshold to determine if a point is an inlier or outlier (default is 100).
    acc_thresh : float, optional
        The threshold to determine if the current model is the best model (default is 0.85).
    max_iters : int, optional
        The maximum number of iterations to perform (default is 1000).

    Returns
    -------
    tuple[np.array, np.array, np.array, np.array]
        The best camera pose and rotation based on PnP estimation,
        and the indices of the inliers and outliers.
    """
    num_features = world_X.shape[0]  # Number of features
    best_acc = 0  # Best accuracy found
    inliers = None  # Inliers indices
    outliers = None  # Outliers indices

    best_C = np.zeros((3, 1))  # Best camera center
    best_R = np.eye(3)  # Best rotation matrix
    print("Running PnP RANSAC")
    
    # Iterate until the best accuracy is reached or max iterations are exhausted
    while best_acc < acc_thresh and max_iters > 0:
        # Randomly select 6 points
        idx = np.random.choice(num_features, 6, replace=False)
        x_sample = img_x[idx]  # Sampled 2D points
        X_sample = world_X[idx]  # Sampled 3D points
        
        # Estimate pose using linear PnP
        R, C = linear_PnP(K, x_sample, X_sample)
        
        # Compute projection matrix
        P = K @ np.hstack((R, -R @ C.reshape(3, 1)))
        
        # Calculate the number of inliers
        num_inliers = error_func(P, world_X, img_x, threshold)
        acc = num_inliers / num_features  # Calculate accuracy
        
        # Update the best model if current model is better
        if acc > best_acc:
            print("Best accuracy PnP RANSAC: ", acc)
            best_acc = acc
            best_C = C
            best_R = R
            inliers, outliers = get_inliers(P, world_X, img_x, threshold)
        
        max_iters -= 1  # Decrement the iteration counter
    
    print(f'Original No. of features: {num_features}')
    print(f"No. of inliers: {len(inliers)}")
    
    return best_C, best_R, inliers, outliers
