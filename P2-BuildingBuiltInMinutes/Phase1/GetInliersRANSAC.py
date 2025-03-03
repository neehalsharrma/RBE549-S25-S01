"""
This module provides functions to perform RANSAC for finding inliers and estimating the fundamental matrix.

Functions
---------
normalization_matrix(points: np.ndarray) -> np.ndarray
    Compute the similarity transformation (normalization matrix) that translates the centroid of the points to the origin and scales the RMS distance to sqrt(2).

ssd_threshold(p1: np.ndarray, p2: np.ndarray, threshold: float, F: np.ndarray) -> int
    Compute the number of inliers given the threshold.

get_inliers(points1: np.ndarray, points2: np.ndarray, threshold: float, best_F: np.ndarray) -> tuple[np.ndarray, np.ndarray]
    Get the inliers and outliers given the threshold.

RANSAC(correspondences: np.ndarray, threshold: float = 5, acc_thresh: float = 0.85, num_iterations: int = 1000) -> tuple[np.ndarray, np.ndarray, np.ndarray]
    RANSAC Algorithm to find the best set of inliers.

RANSAC_threaded(correspondences: np.ndarray, threshold: float = 5, acc_thresh: float = 0.85, num_threads: int = 12) -> tuple[np.ndarray, np.ndarray, np.ndarray]
    Multithreaded RANSAC Algorithm that stops all threads when a good enough model is found.

cv2_RANSAC(correspondences: np.ndarray, threshold: float = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]
    RANSAC Algorithm to find the best set of inliers using OpenCV.

show_RANSAC(image1: int, image2: int, inliers: np.ndarray, outliers: np.ndarray, save: bool = False, save_path: str = '../Results/', title: str = None) -> None
    Display the images with the inliers.
"""

import concurrent.futures
import threading
import time

import cv2
from EstimateFundamentalMatrix import estimateF
import concurrent.futures
import threading


def normalization_matrix(points: np.ndarray) -> np.ndarray:
    """
    Compute the similarity transformation (normalization matrix) that translates the centroid of the points to the origin and scales the RMS distance to sqrt(2).

    Parameters
    ----------
    points : np.ndarray
        (n x 2) array of points.

    Returns
    -------
    np.ndarray
        3x3 normalization matrix.
    """
    centroid = np.mean(points, axis=0)  # Compute centroid (x̄, ȳ)

    # Compute the RMS distance from the centroid
    rms_dist = np.sqrt(np.mean(np.sum((points - centroid) ** 2, axis=1)))
    scale = np.sqrt(2) / rms_dist
    # Normalization matrix
    T = np.array(
        [[scale, 0, -scale * centroid[0]], [0, scale, -scale * centroid[1]], [0, 0, 1]]
    )

    return T


def ssdThreshold(p1: np.array, p2: np.array, threshold: float, F: np.array) -> tuple[int, np.array, np.array]:
    """
    Compute the number of inliers given the threshold.
    @ p1: The points in the first image as an n x 3 array.
    @ p2: The points in the second image as an n x 3 array.
    @ threshold: The threshold to be used.
    @ F: The Fundamental Matrix as 3x3
    @ return: The number of inliers and the indices of the inliers.
    """
    errors = np.abs(np.sum(p2 * (F @ p1.T).T, axis=1))
    inliers_idx = np.argwhere(errors < threshold).squeeze()
    outliers_idx = np.argwhere(errors >= threshold).squeeze()
    num_inliers = inliers_idx.shape[0]

    return num_inliers, inliers_idx, outliers_idx


# Single Threaded RANSAC written by Nikesh
def RANSAC(correspondences: np.array, threshold: float = 5, acc_thresh=0.85, num_iterations=1000) \
        -> tuple[np.array, np.array, np.array]:
    """
    RANSAC Algorithm to find the best set of inliers.
    @ correspondences: The correspondences between the two images, np.array of shape (n, 4).
    @ threshold: The threshold to be used for determining inliers.
    @ acc_thresh: The threshold for the percentage of inliers to stop the algorithm.
    @ return: The best Fundamental Matrix and the indices of the best inliers.
    """
    start_time = time.time()
    num_features = correspondences.shape[0]
    best_percent = 0
    best_inliers = None
    outliers = None
    points1 = correspondences[:, 0:2]  # Extract points from the first image
    points2 = correspondences[:, 2:4]  # Extract points from the second image

    # Homogenize and normalize the points
    normed1 = np.hstack((points1, np.ones((points1.shape[0], 1))))
    normed2 = np.hstack((points2, np.ones((points2.shape[0], 1))))
    # 3x3 normalization matrices
    T1 = normalization_matrix(normed1)
    T2 = normalization_matrix(normed2)
    # Normalize the points
    normed1 = (T1 @ normed1.T).T
    normed2 = (T2 @ normed2.T).T
    iters = 0
    # Run RANSAC iterations
    while best_percent < acc_thresh and iters < num_iterations:
        iters += 1
        random_samples = np.random.choice(points1.shape[0], size=8, replace=False)
        samples1 = normed1[random_samples, :]
        samples2 = normed2[random_samples, :]

        F = estimateF(samples1, samples2)
        num_inliers, inliers_idx, outliers_idx = ssdThreshold(normed1, normed2, threshold, F)
        percent_match = num_inliers / num_features

        # If a better match is found, update the best match
        if percent_match > best_percent:
            best_percent = percent_match
            best_inliers = inliers_idx
            outliers = outliers_idx
            print(f"Best Percent: {best_percent}")

    # Print the results
    print(f"Time taken: {time.time() - start_time}")
    print(f"Iterations: {iters}")
    print(f"Original No. of features: {num_features}")
    print(f"No. of inliers: {best_inliers.shape[0]}")

    # Estimate the final F using the best inliers and the normalization matrices
    F_hat = estimateF(normed1[best_inliers, :], normed2[best_inliers, :])
    # Denormalize the fundamental matrix
    F_hat = T2.T @ F_hat @ T1
    inliers = np.hstack((points1[best_inliers, :], points2[best_inliers, :]))
    outliers = np.hstack((points1[outliers, :], points2[outliers, :]))
    return F_hat, inliers, outliers


def RANSAC_threaded(
    correspondences: np.ndarray,
    threshold: float = 5,
    acc_thresh: float = 0.85,
    num_threads: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Multithreaded RANSAC Algorithm that stops all threads when a good enough model is found.

    Parameters
    ----------
    correspondences : np.ndarray
        The correspondences between the two images, np.array of shape (n, 4).
    threshold : float, optional
        The threshold to be used for determining inliers (default is 5).
    acc_thresh : float, optional
        The threshold for the percentage of inliers to stop the algorithm (default is 0.85).
    num_threads : int, optional
        The number of threads to be used (default is 12).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        The best Fundamental Matrix, the best inliers, and the outliers.
    """
    start_time = time.time()  # Record the start time
    num_features = correspondences.shape[0]  # Number of features

    best_percent = 0
    best_inliers = None
    best_outliers = None
    best_lock = threading.Lock()
    stop_event = threading.Event()

    # Extract points from the correspondences
    points1 = correspondences[:, :2]
    points2 = correspondences[:, 2:]
    rng = np.random.default_rng()

    def ransac_iteration():
        """
        Perform a single RANSAC iteration.
        """
        nonlocal best_percent, best_inliers, best_outliers

        while not stop_event.is_set():
            random_samples = rng.choice(
                a=correspondences, size=8, replace=False, axis=0, shuffle=False
            )
            # Extract the sampled points from the image
            samples1 = random_samples[:, 0:2]
            samples2 = random_samples[:, 2:4]
            F = estimateF(samples1, samples2)  # Estimate the fundamental matrix
            num_inliers = ssd_threshold(points1, points2, threshold, F)
            percent_match = num_inliers / num_features

            # Check and update best match safely
            with best_lock:
                if percent_match > best_percent:
                    best_percent = percent_match
                    best_inliers, best_outliers = get_inliers(
                        points1, points2, threshold, F
                    )
                    print(f"Best Percent: {best_percent}")

                    # Stop all threads if we exceed the percentage threshold
                    if best_percent >= acc_thresh:
                        stop_event.set()
                        return

    # Run RANSAC iterations in multiple threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(ransac_iteration) for _ in range(num_threads)]
        concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)

    print(f"Time taken: {time.time() - start_time}")
    print(f"Original No. of features: {num_features}")
    print(f"No. of inliers: {best_inliers.shape[0]}")

    # Compute final refined F using the best inliers found
    F_hat = estimateF(best_inliers[:, 0:2], best_inliers[:, 2:4])

    return F_hat, best_inliers, best_outliers


def cv2_RANSAC(
    correspondences: np.ndarray, threshold: float = 5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    RANSAC Algorithm to find the best set of inliers using OpenCV.

    Parameters
    ----------
    correspondences : np.ndarray
        The correspondences between the two images, np.array of shape (n, 4).
    threshold : float, optional
        The threshold to be used for determining inliers (default is 5).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        The best Fundamental Matrix, the best inliers, and the outliers.
    """
    pts2 = correspondences[:, 2:]
    # Estimate the fundamental matrix using OpenCV
    fundamental, mask = cv2.findFundamentalMat(
        pts1, pts2, cv2.FM_RANSAC, threshold, 0.99
    )
    if fundamental is None or mask is None:
        raise ValueError("Failed to estimate a valid fundamental matrix using OpenCV.")
    inliers1 = pts1[mask.ravel() == 1]
    inliers2 = pts2[mask.ravel() == 1]
    outliers1 = pts1[mask.ravel() == 0]
    outliers2 = pts2[mask.ravel() == 0]
    print(f"Original No. of features: {correspondences.shape[0]}")
    print(f"No. of inliers: {inliers1.shape[0]}")
    print(f"Original No. of features: {correspondences.shape[0]}")
    print(f"No. of inliers: {inliers1.shape[0]}")

    return fundamental, np.hstack((inliers1, inliers2)), np.hstack((outliers1, outliers2))
