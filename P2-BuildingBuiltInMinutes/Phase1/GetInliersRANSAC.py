import time

import numpy as np
import cv2
import matplotlib.pyplot as plt
from EstimateFundamentalMatrix import estimateF
from LoadData import loadImage
import concurrent.futures
import threading

import numpy as np


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
# def RANSAC(correspondences: np.array, threshold: float = 5, acc_thresh=0.85) -> np.array:
#     """
#     RANSAC Algorithm to find the best set of inliers.
#     @ correspondences: The correspondences between the two images, np.array of shape (n, 4).
#     @ threshold: The threshold to be used for determining inliers.
#     @ acc_thresh: The threshold for the percentage of inliers to stop the algorithm.
#     @ return: The best Fundamental Matrix and the best inliers.
#     """
#     start_time = time.time()
#     num_features = correspondences.shape[0]
#     best_percent = 0
#     best_inliers = None
#     outliers = None
#     points1 = correspondences[:, 0:2]
#     points2 = correspondences[:, 2:4]
#     rng = np.random.default_rng()
#     while best_percent < acc_thresh:
#         random_samples = rng.choice(a=correspondences, size=8, replace=False, axis=0, shuffle=False)
#         samples1 = random_samples[:, 0:2]
#         samples2 = random_samples[:, 2:4]
#         F = estimateF(samples1, samples2)
#         num_inliers = sampson_dist_Threshold(points1, points2, threshold, F)
#         percent_match = num_inliers / num_features
#         # if a better match is found, update the best match
#         if percent_match > best_percent:
#             best_percent = percent_match
#             best_inliers, outliers = getInliers(points1, points2, threshold, F)
#             print(f"Best Percent: {best_percent}")
#
#     print(f"Time taken: {time.time() - start_time}")
#     print(f'Original No. of features: {num_features}')
#     print(f"No. of inliers: {best_inliers.shape[0]}")
#     F_hat = estimateF(best_inliers[:, 0:2], best_inliers[:, 2:4])
#
#     return F_hat, best_inliers, outliers


# Modified multithreaded RANSAC modified from the previous by ChatGPT
def RANSAC(correspondences: np.array, threshold: float = 5, acc_thresh = 0.85,
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
            num_inliers = sampson_dist_Threshold(points1, points2, threshold, F)
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
    @ inliers: The inliers as a n x 4 array.
    """
    img1 = loadImage(image1)
    img2 = loadImage(image2)
    img = np.concatenate((img1, img2), axis=1)
    for i in range(inliers.shape[0]):
        # Hide some of the inliers to make the image easier to see
        if i % 5 == 0:
            continue
        x1, y1, x2, y2 = inliers[i]
        cv2.circle(img, (int(x1), int(y1)), 5, (0, 0, 255), -1)
        cv2.circle(img, (int(x2) + img1.shape[1], int(y2)), 5, (0, 0, 255), -1)
        cv2.line(img, (int(x1), int(y1)), (int(x2) + img1.shape[1], int(y2)), (0, 127, 0), 1)
    if outliers is not None:
        for i in range(outliers.shape[0]):
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
