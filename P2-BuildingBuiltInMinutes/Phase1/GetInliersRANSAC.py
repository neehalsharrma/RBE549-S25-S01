import time

import numpy as np
import cv2
import matplotlib.pyplot as plt
from EstimateFundamentalMatrix import estimateF, estimateF_7pt
from LoadData import load_image
import concurrent.futures
import threading


def normalization_matrix(points: np.array) -> np.array:
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


def ssdThreshold(p1: np.array, p2: np.array, threshold: float, F: np.array) -> int:
    """
    Compute the number of inliers given the threshold.
    @ p1: The points in the first image as an n x 3 array.
    @ p2: The points in the second image as an n x 3 array.
    @ threshold: The threshold to be used.
    @ F: The Fundamental Matrix as 3x3
    """
    errors = np.abs(np.sum(p2 * (F @ p1.T).T, axis=1))
    num_inliers = np.sum(errors < threshold)

    return num_inliers


def getInliers(points1: np.array, points2: np.array, threshold: float, best_F: np.array) -> tuple[np.array, np.array]:
    """
    Get the inliers and outliers given the threshold.
    @ points1: The points in the first image as an n x 3 array.
    @ points2: The points in the second image as an n x 3 array.
    @ threshold: The threshold to be used.
    @ best_F: The best Fundamental Matrix as 3x3
    @ return: The inlier and outlier indices
    """
    errors = np.abs(np.sum(points2 * (best_F @ points1.T).T, axis=1))
    inliers = np.argwhere(errors < threshold).squeeze()
    outliers = np.argwhere(errors >= threshold).squeeze()

    return inliers, outliers


# Single Threaded RANSAC written by Nikesh
def RANSAC(correspondences: np.array, threshold: float = 5, acc_thresh=0.85, num_iterations=1000) -> np.array:
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

    # Homogenize and normalize the points
    normed1 = np.hstack((points1, np.ones((points1.shape[0], 1))))  # n x 3
    normed2 = np.hstack((points2, np.ones((points2.shape[0], 1))))  # n x 3
    T1 = normalization_matrix(normed1)  # 3x3
    T2 = normalization_matrix(normed2)  # 3x3
    normed1 = (T1 @ normed1.T).T  # n x 3
    normed2 = (T2 @ normed2.T).T  # n x 3
    iters = 0
    while best_percent < acc_thresh and iters < num_iterations:
        iters += 1
        random_samples = np.random.choice(points1.shape[0], size=8, replace=False)
        samples1 = normed1[random_samples, :]
        samples2 = normed2[random_samples, :]

        F = estimateF(samples1, samples2)
        num_inliers = ssdThreshold(normed1, normed2, threshold, F)
        percent_match = num_inliers / num_features

        # if a better match is found, update the best match
        if percent_match > best_percent:
            best_percent = percent_match
            best_inliers, outliers = getInliers(normed1, normed2, threshold, F)
            print(f"Best Percent: {best_percent}")

    print(f"Time taken: {time.time() - start_time}")
    print(f'Iterations: {iters}')
    print(f'Original No. of features: {num_features}')
    print(f"No. of inliers: {best_inliers.shape[0]}")
    inliers = np.hstack((points1[best_inliers], points2[best_inliers])).reshape(-1, 4)
    outliers = np.hstack((points1[outliers], points2[outliers])).reshape(-1, 4)
    # Estimate the final F using the best inliers and the normalization matrices
    F_hat = estimateF(normed1[best_inliers, :], normed2[best_inliers,:])
    # Denormalize the fundamental matrix
    F_hat = T2.T @ F_hat @ T1

    return F_hat, inliers, outliers


# Modified multithreaded RANSAC modified from the previous by ChatGPT
def RANSAC_Threaded(correspondences: np.array, threshold: float = 5, acc_thresh=0.85,
                    num_threads: int = 12) -> np.array:
    """
    Multithreaded RANSAC Algorithm that stops all threads when a good enough model is found.

    @ correspondences: The correspondences between the two images, np.array of shape (n, 4).
    @ threshold: The threshold to be used for determining inliers.
    @ acc_thresh: The threshold for the percentage of inliers to stop the algorithm.
    @ num_threads: The number of threads to be used.
    @ return: The best Fundamental Matrix and the best inliers.
    """
    start_time = time.time()
    num_features = correspondences.shape[0]

    best_percent = 0
    best_inliers = None
    best_outliers = None
    best_lock = threading.Lock()
    stop_event = threading.Event()

    points1 = correspondences[:, :2]
    points2 = correspondences[:, 2:]
    rng = np.random.default_rng()

    def ransac_iteration():
        nonlocal best_percent, best_inliers, best_outliers

        while not stop_event.is_set():
            random_samples = rng.choice(a=correspondences, size=8, replace=False, axis=0,
                                        shuffle=False)
            samples1 = random_samples[:, 0:2]
            samples2 = random_samples[:, 2:4]
            F = estimateF(samples1, samples2)
            num_inliers = ssdThreshold(points1, points2, threshold, F)
            percent_match = num_inliers / num_features

            # Check and update best match safely
            with best_lock:
                if percent_match > best_percent:
                    best_percent = percent_match
                    best_inliers, best_outliers = getInliers(points1, points2, threshold, F)
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
    print(f'Original No. of features: {num_features}')
    print(f"No. of inliers: {best_inliers.shape[0]}")

    # Compute final refined F using the best inliers found
    F_hat = estimateF(best_inliers[:, 0:2], best_inliers[:, 2:4])

    return F_hat, best_inliers, best_outliers


def cv2RANSAC(correspondences: np.array, threshold: float = 5):
    """
    RANSAC Algorithm to find the best set of inliers using OpenCV.
    Use this as a ground truth to compare your implementation.
    """
    pts1 = correspondences[:, :2]
    pts2 = correspondences[:, 2:]
    fundamental, mask = cv2.findFundamentalMat(pts1, pts2, cv2.RANSAC, threshold, 0.99)
    inliers1 = pts1[mask.ravel() == 1]
    inliers2 = pts2[mask.ravel() == 1]
    outliers1 = pts1[mask.ravel() == 0]
    outliers2 = pts2[mask.ravel() == 0]
    print(f'Original No. of features: {correspondences.shape[0]}')
    print(f"No. of inliers: {inliers1.shape[0]}")

    return fundamental, np.hstack((inliers1, inliers2)), np.hstack((outliers1, outliers2))


def showRANSAC(image1: int, image2: int, inliers: np.array, outliers: np.array, save: bool = False,
               save_path: str = '../Results/', title=None) -> None:
    """
    Display the images with the inliers.
    @ img1: The first image.
    @ img2: The second image.
    @ inliers: The inliers as an n x 4 array.
    """
    img1 = load_image(image1)
    img2 = load_image(image2)
    img = np.concatenate((img1, img2), axis=1)
    for i in range(inliers.shape[0]):
        # Hide some of the inliers to make the image easier to see
        if i % 5 != 0:
            continue
        x1, y1, x2, y2 = inliers[i]
        cv2.circle(img, (int(x1), int(y1)), 5, (0, 0, 255), -1)
        cv2.circle(img, (int(x2) + img1.shape[1], int(y2)), 5, (0, 0, 255), -1)
        cv2.line(img, (int(x1), int(y1)), (int(x2) + img1.shape[1], int(y2)), (0, 127, 0), 1)
    if outliers is not None:
        for i in range(outliers.shape[0]):
            if i % 5 != 0:
                continue
            x1, y1, x2, y2 = outliers[i]
            cv2.circle(img, (int(x1), int(y1)), 5, (0, 0, 255), -1)
            cv2.circle(img, (int(x2) + img1.shape[1], int(y2)), 5, (0, 0, 255), -1)
            cv2.line(img, (int(x1), int(y1)), (int(x2) + img1.shape[1], int(y2)), (255, 0, 0), 1)
    plt.figure(figsize=(10, 10))
    if title is not None:
        plt.title(title)
    else:
        plt.title('RANSAC Inliers')
    plt.imshow(img)
    if save:
        plt.savefig(save_path + 'RANSAC' + '_' + str(image1) + '_' + str(image2) + '.png')
    plt.axis('off')
    plt.show()
