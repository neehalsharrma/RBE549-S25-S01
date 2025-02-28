import time

import numpy as np
import cv2
import matplotlib.pyplot as plt
from EstimateFundamentalMatrix import estimateF, estimateF_7pt
from LoadData import loadImage
import concurrent.futures
import threading



def normalization_matrix(points):
    """
    Compute the similarity transformation (normalization matrix) that
    translates the centroid of the points to the origin and scales the RMS distance to sqrt(2).
    @ points: (n x 2) array of points.
    @ return: 3x3 normalization matrix.
    """
    centroid = np.mean(points, axis=0)  # Compute centroid (x̄, ȳ)

    # Compute the RMS distance from the centroid
    rms_dist = np.sqrt(np.mean(np.sum((points - centroid) ** 2, axis=1)))
    scale = np.sqrt(2) / rms_dist
    # Normalization matrix
    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1]
    ])

    return T



# Using the Sampson distance to compute the number of inliers based on the algorithm from
# equation 11.9 in Multiple View Geometry in Computer Vision, Second Edition
def sampson_dist_Threshold(points1, points2, threshold, F) -> int:
    """
    Compute the number of inliers given the threshold using Sampson distance.
    @ points1: The points in the first image as an n x 2 array.
    @ points2: The points in the second image as an n x 2 array.
    @ threshold: The threshold to be used.
    @ F: The Fundamental Matrix as 3x3
    """

    p1 = np.hstack((points1, np.ones((points1.shape[0], 1))))  # n x 3
    p2 = np.hstack((points2, np.ones((points2.shape[0], 1))))  # n x 3

    F_p1 = F @ p1.T  # 3 x n
    F_p2 = F.T @ p2.T  # 3 x n
    errors = np.sum(p2 * F_p1.T, axis=1) ** 2 / (F_p1[0] ** 2 + F_p1[1] ** 2 + F_p2[0] ** 2 + F_p2[1] ** 2)
    num_inliers = np.sum(errors < threshold ** 2)
    return num_inliers



def getInliers(points1, points2, threshold, best_F) -> tuple[np.array, np.array]:
    p1 = np.hstack((points1, np.ones((points1.shape[0], 1))))  # n x 3
    p2 = np.hstack((points2, np.ones((points2.shape[0], 1))))  # n x 3
    F_p1 = best_F @ p1.T  # 3 x n
    F_p2 = best_F.T @ p2.T  # 3 x n
    errors = np.sqrt(np.sum(p2 * F_p1.T, axis=1) ** 2 / (F_p1[0] ** 2 + F_p1[1] ** 2 + F_p2[0] ** 2 + F_p2[1] ** 2))
    inliers = np.hstack((points1[errors < threshold], points2[errors < threshold]))
    outliers = np.hstack((points1[errors >= threshold], points2[errors >= threshold]))
    return inliers, outliers



# Single Threaded RANSAC written by Nikesh
def RANSAC_Sampson(correspondences: np.array, threshold: float = 5, acc_thresh=0.85) -> np.array:
    """
    RANSAC Algorithm to find the best set of inliers.
    @ correspondences: The correspondences between the two images, np.array of shape (n, 4).
    @ threshold: The threshold to be used for determining inliers.
    @ acc_thresh: The threshold for the percentage of inliers to stop the algorithm.
    @ return: The best Fundamental Matrix and the best inliers.
    """
    start_time = time.time()
    num_features = correspondences.shape[0]
    best_percent = 0
    best_inliers = None
    outliers = None
    points1 = correspondences[:, 0:2]
    points2 = correspondences[:, 2:4]
    rng = np.random.default_rng()
    while best_percent < acc_thresh:
        random_samples = rng.choice(a=correspondences, size=8, replace=False, axis=0, shuffle=False)
        samples1 = random_samples[:, 0:2]
        samples2 = random_samples[:, 2:4]
        F = estimateF(samples1, samples2)
        num_inliers = sampson_dist_Threshold(points1, points2, threshold, F)
        percent_match = num_inliers / num_features
        # if a better match is found, update the best match
        if percent_match > best_percent:
            best_percent = percent_match
            best_inliers, outliers = getInliers(points1, points2, threshold, F)
            print(f"Best Percent: {best_percent}")

    print(f"Time taken: {time.time() - start_time}")
    print(f'Original No. of features: {num_features}')
    print(f"No. of inliers: {best_inliers.shape[0]}")
    F_hat = estimateF(best_inliers[:, 0:2], best_inliers[:, 2:4])

    return F_hat, best_inliers, outliers


def RANSAC_7pt(correspondences: np.array, threshold: float = 5, acc_thresh=0.85) -> np.array:
    """
    RANSAC Algorithm to find the best set of inliers.
    @ correspondences: The correspondences between the two images, np.array of shape (n, 4).
    @ threshold: The threshold to be used for determining inliers.
    @ acc_thresh: The threshold for the percentage of inliers to stop the algorithm.
    @ return: The best Fundamental Matrix and the best inliers.
    """
    start_time = time.time()
    num_features = correspondences.shape[0]
    best_percent = 0
    best_inliers = None
    outliers = None
    points1 = correspondences[:, 0:2]
    points2 = correspondences[:, 2:4]
    rng = np.random.default_rng()
    while best_percent < acc_thresh:
        random_samples = rng.choice(a=correspondences, size=7, replace=False, axis=0, shuffle=False)
        samples1 = random_samples[:, 0:2]
        samples2 = random_samples[:, 2:4]
        F = estimateF_7pt(samples1, samples2)
        for i in range(F.shape[2]):
            num_inliers = sampson_dist_Threshold(points1, points2, threshold, F[:, :, i])
            percent_match = num_inliers / num_features
            # if a better match is found, update the best match
            if percent_match > best_percent:
                best_percent = percent_match
                best_inliers, outliers = getInliers(points1, points2, threshold, F[:, :, i])
                print(f"Best Percent: {best_percent}")
    print(f"Time taken: {time.time() - start_time}")
    print(f'Original No. of features: {num_features}')
    print(f"No. of inliers: {best_inliers.shape[0]}")
    F_hat = estimateF(best_inliers[:, 0:2], best_inliers[:, 2:4])

    return F_hat, best_inliers, outliers
