"""
Module for RANSAC algorithms to estimate the Fundamental Matrix using Sampson distance and the 7-point algorithm.

Functions
---------
normalization_matrix(points)
    Compute the similarity transformation (normalization matrix) that translates the centroid of the points to the origin and scales the RMS distance to sqrt(2).
sampson_dist_Threshold(points1, points2, threshold, F)
    Compute the number of inliers given the threshold using Sampson distance.
getInliers(points1, points2, threshold, best_F)
    Get the inliers and outliers based on the best Fundamental Matrix.
RANSAC_Sampson(correspondences, threshold=5, acc_thresh=0.85)
    RANSAC Algorithm to find the best set of inliers using Sampson distance.
RANSAC_7pt(correspondences, threshold=5, acc_thresh=0.85)
    RANSAC Algorithm to find the best set of inliers using the 7-point algorithm.
"""

import time
import numpy as np
from EstimateFundamentalMatrix import estimate_F, estimate_F_7pt


def normalization_matrix(points):
    """
    Compute the similarity transformation (normalization matrix) that
    translates the centroid of the points to the origin and scales the RMS distance to sqrt(2).

    Parameters
    ----------
    points : ndarray
        (n x 2) array of points.

    Returns
    -------
    T : ndarray
        3x3 normalization matrix.
    """
    # Compute the centroid of the points (mean of x and y coordinates)
    centroid = np.mean(points, axis=0)

    # Compute the RMS distance from the centroid to the points
    rms_dist = np.sqrt(np.mean(np.sum((points - centroid) ** 2, axis=1)))

    # Compute the scaling factor to make the RMS distance sqrt(2)
    scale = np.sqrt(2) / rms_dist

    # Construct the normalization matrix
    T = np.array(
        [
            [scale, 0, -scale * centroid[0]],  # Scale and translate x coordinates
            [0, scale, -scale * centroid[1]],  # Scale and translate y coordinates
            [0, 0, 1],  # Homogeneous coordinate
        ]
    )

    return T


def sampson_dist_Threshold(points1, points2, threshold, F) -> int:
    """
    Compute the number of inliers given the threshold using Sampson distance.

    Parameters
    ----------
    points1 : ndarray
        The points in the first image as an n x 2 array.
    points2 : ndarray
        The points in the second image as an n x 2 array.
    threshold : float
        The threshold to be used.
    F : ndarray
        The Fundamental Matrix as 3x3.

    Returns
    -------
    num_inliers : int
        Number of inliers.
    """
    # Convert points to homogeneous coordinates by adding a column of ones
    p1 = np.hstack((points1, np.ones((points1.shape[0], 1))))  # n x 3
    p2 = np.hstack((points2, np.ones((points2.shape[0], 1))))  # n x 3

    # Compute the epipolar lines for points in the first and second images
    F_p1 = F @ p1.T  # 3 x n
    F_p2 = F.T @ p2.T  # 3 x n

    # Compute the Sampson distance errors
    errors = np.sum(p2 * F_p1.T, axis=1) ** 2 / (
        F_p1[0] ** 2 + F_p1[1] ** 2 + F_p2[0] ** 2 + F_p2[1] ** 2
    )

    # Count the number of inliers where the error is below the threshold
    num_inliers = np.sum(errors < threshold**2)

    return num_inliers


def getInliers(points1, points2, threshold, best_F) -> tuple[np.array, np.array]:
    """
    Get the inliers and outliers based on the best Fundamental Matrix.

    Parameters
    ----------
    points1 : ndarray
        The points in the first image as an n x 2 array.
    points2 : ndarray
        The points in the second image as an n x 2 array.
    threshold : float
        The threshold to be used.
    best_F : ndarray
        The best Fundamental Matrix as 3x3.

    Returns
    -------
    inliers : ndarray
        Array of inliers.
    outliers : ndarray
        Array of outliers.
    """
    # Convert points to homogeneous coordinates by adding a column of ones
    p1 = np.hstack((points1, np.ones((points1.shape[0], 1))))  # n x 3
    p2 = np.hstack((points2, np.ones((points2.shape[0], 1))))  # n x 3

    # Compute the epipolar lines for points in the first and second images
    F_p1 = best_F @ p1.T  # 3 x n
    F_p2 = best_F.T @ p2.T  # 3 x n

    # Compute the Sampson distance errors
    errors = np.sqrt(
        np.sum(p2 * F_p1.T, axis=1) ** 2
        / (F_p1[0] ** 2 + F_p1[1] ** 2 + F_p2[0] ** 2 + F_p2[1] ** 2)
    )

    # Identify inliers where the error is below the threshold
    inliers = np.hstack((points1[errors < threshold], points2[errors < threshold]))

    # Identify outliers where the error is above or equal to the threshold
    outliers = np.hstack((points1[errors >= threshold], points2[errors >= threshold]))

    return inliers, outliers


def RANSAC_Sampson(
    correspondences: np.array, threshold: float = 5, acc_thresh=0.85
) -> np.array:
    """
    RANSAC Algorithm to find the best set of inliers using Sampson distance.

    Parameters
    ----------
    correspondences : ndarray
        The correspondences between the two images, np.array of shape (n, 4).
    threshold : float, optional
        The threshold to be used for determining inliers, by default 5.
    acc_thresh : float, optional
        The threshold for the percentage of inliers to stop the algorithm, by default 0.85.

    Returns
    -------
    F_hat : ndarray
        The best Fundamental Matrix.
    best_inliers : ndarray
        The best set of inliers.
    outliers : ndarray
        The set of outliers.
    """
    start_time = time.time()  # Start the timer
    num_features = correspondences.shape[0]  # Number of correspondences
    best_percent = 0  # Best percentage of inliers found
    best_inliers = None  # Best set of inliers
    outliers = None  # Set of outliers
    points1 = correspondences[:, 0:2]  # Points from the first image
    points2 = correspondences[:, 2:4]  # Corresponding points from the second image
    rng = np.random.default_rng()  # Random number generator

    # Loop until the best percentage of inliers exceeds the accuracy threshold
    while best_percent < acc_thresh:
        # Randomly sample 8 correspondences
        random_samples = rng.choice(
            a=correspondences, size=8, replace=False, axis=0, shuffle=False
        )
        samples1 = random_samples[:, 0:2]  # Sampled points from the first image
        samples2 = random_samples[
            :, 2:4
        ]  # Corresponding sampled points from the second image

        # Estimate the Fundamental Matrix using the sampled points
        F = estimate_F(samples1, samples2)

        # Compute the number of inliers using Sampson distance
        num_inliers = sampson_dist_Threshold(points1, points2, threshold, F)
        percent_match = (
            num_inliers / num_features
        )  # Calculate the percentage of inliers

        # If a better match is found, update the best match
        if percent_match > best_percent:
            best_percent = percent_match
            best_inliers, outliers = getInliers(points1, points2, threshold, F)
            print(f"Best Percent: {best_percent}")

    # Print the time taken and the number of inliers found
    print(f"Time taken: {time.time() - start_time}")
    print(f"Original No. of features: {num_features}")
    print(f"No. of inliers: {best_inliers.shape[0]}")

    # Re-estimate the Fundamental Matrix using the best set of inliers
    F_hat = estimate_F(best_inliers[:, 0:2], best_inliers[:, 2:4])

    return F_hat, best_inliers, outliers


def RANSAC_7pt(
    correspondences: np.array, threshold: float = 5, acc_thresh=0.85
) -> np.array:
    """
    RANSAC Algorithm to find the best set of inliers using the 7-point algorithm.

    Parameters
    ----------
    correspondences : ndarray
        The correspondences between the two images, np.array of shape (n, 4).
    threshold : float, optional
        The threshold to be used for determining inliers, by default 5.
    acc_thresh : float, optional
        The threshold for the percentage of inliers to stop the algorithm, by default 0.85.

    Returns
    -------
    F_hat : ndarray
        The best Fundamental Matrix.
    best_inliers : ndarray
        The best set of inliers.
    outliers : ndarray
        The set of outliers.
    """
    start_time = time.time()  # Start the timer
    num_features = correspondences.shape[0]  # Number of correspondences
    best_percent = 0  # Best percentage of inliers found
    best_inliers = None  # Best set of inliers
    outliers = None  # Set of outliers
    points1 = correspondences[:, 0:2]  # Points from the first image
    points2 = correspondences[:, 2:4]  # Corresponding points from the second image
    rng = np.random.default_rng()  # Random number generator

    # Loop until the best percentage of inliers exceeds the accuracy threshold
    while best_percent < acc_thresh:
        # Randomly sample 7 correspondences
        random_samples = rng.choice(
            a=correspondences, size=7, replace=False, axis=0, shuffle=False
        )
        samples1 = random_samples[:, 0:2]  # Sampled points from the first image
        samples2 = random_samples[
            :, 2:4
        ]  # Corresponding sampled points from the second image

        # Estimate the Fundamental Matrix using the 7-point algorithm
        F = estimate_F_7pt(samples1, samples2)

        # Iterate over all possible solutions of F
        for i in range(F.shape[2]):
            # Compute the number of inliers using Sampson distance
            num_inliers = sampson_dist_Threshold(
                points1, points2, threshold, F[:, :, i]
            )
            percent_match = (
                num_inliers / num_features
            )  # Calculate the percentage of inliers

            # If a better match is found, update the best match
            if percent_match > best_percent:
                best_percent = percent_match
                best_inliers, outliers = getInliers(
                    points1, points2, threshold, F[:, :, i]
                )
                print(f"Best Percent: {best_percent}")

    # Print the time taken and the number of inliers found
    print(f"Time taken: {time.time() - start_time}")
    print(f"Original No. of features: {num_features}")
    print(f"No. of inliers: {best_inliers.shape[0]}")

    # Re-estimate the Fundamental Matrix using the best set of inliers
    F_hat = estimate_F(best_inliers[:, 0:2], best_inliers[:, 2:4])

    return F_hat, best_inliers, outliers
