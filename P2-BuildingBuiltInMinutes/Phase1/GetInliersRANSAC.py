import numpy as np
import cv2
import matplotlib.pyplot as plt
from EstimateFundamentalMatrix import estimateF
from LoadData import loadImage


def ssdThreshold(points1, points2, threshold, F) -> int:
    num_inliers = 0
    for i in range(points1.shape[0]):
        p1 = np.array([points1[i, 0], points1[i, 1], 1])
        p2 = np.array([points2[i, 0], points2[i, 1], 1])
        error = np.abs(p2 @ F @ p1)
        if error < threshold:
            num_inliers += 1
    return num_inliers


def getInliers(points1, points2, threshold, best_F) -> np.array:
    inliers = np.ndarray((0, 4))
    for i in range(points1.shape[0]):
        p1 = np.array([points1[i, 0], points1[i, 1], 1])
        p2 = np.array([points2[i, 0], points2[i, 1], 1])
        error = np.abs(p2 @ best_F @ p1)
        if error < threshold:
            points = np.array((points1[i], points2[i])).reshape(1, 4)
            inliers = np.append(inliers, points, axis=0)
    return inliers


def RANSAC(correspondences: np.array, threshold: float = 10, num_iterations: int = 2000) -> np.array:
    """
    RANSAC Algorithm to find the best set of inliers.
    @ correspondences: The correspondences between the two images, np.array of shape (n, 4).
    @ threshold: The threshold to be used for determining inliers.
    @ num_iterations: The number of iterations to be used.
    @ return: The best Fundamental Matrix and the best inliers.
    """
    num_features = correspondences.shape[0]
    best_percent = 0
    best_inliers = None
    points1 = correspondences[:, 0:2]
    points2 = correspondences[:, 2:4]
    # for i in range(num_iterations):
    while best_percent < .9:
        random_samples = np.random.default_rng().choice(a=correspondences,size=8, replace=False, axis=0, shuffle=False)
        samples1 = random_samples[:, 0:2]
        samples2 = random_samples[:, 2:4]
        F = estimateF(samples1, samples2)
        num_inliers = ssdThreshold(points1, points2, threshold, F)
        percent_match = num_inliers / num_features
        # if a better match is found, update the best match
        if percent_match > best_percent:
            best_percent = percent_match
            best_inliers = getInliers(points1, points2, threshold, F)
            print(f"Best Percent: {best_percent}")

    F_hat = estimateF(best_inliers[:, 0:2], best_inliers[:, 2:4])

    return F_hat, best_inliers


def showRANSAC(image1: int, image2: int, inliers: np.array, save: bool = False, save_path: str = '../Results/') -> None:
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
        x1, y1, x2, y2 = inliers[i]
        cv2.circle(img, (int(x1), int(y1)), 5, (0, 255, 0), -1)
        cv2.circle(img, (int(x2) + img1.shape[1], int(y2)), 5, (0, 255, 0), -1)
        cv2.line(img, (int(x1), int(y1)), (int(x2) + img1.shape[1], int(y2)), (255, 0, 0), 2)
    plt.figure(figsize=(10, 10))
    plt.title('RANSAC Inliers')
    plt.imshow(img)
    if save:
        plt.savefig(save_path + 'RANSAC' + '_' + str(image1) + '_' + str(image2) + '.png')
    plt.axis('off')
    plt.show()
