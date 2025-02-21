# Create Your Own Starter Code :)
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import LoadData
from EstimateFundamentalMatrix import estimateF
from GetInliersRANSAC import RANSAC, showRANSAC

if __name__ == '__main__':
    img1 = LoadData.loadImage(1)
    img2 = LoadData.loadImage(2)
    img = np.concatenate((img1, img2), axis=1)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    LoadData.showMatches(1, 2)
    # LoadData.showMatches(2, 3)
    # LoadData.showMatches(3, 4)
    # LoadData.showMatches(4, 5)
    correspondences = LoadData.loadCorrespondences(1, 2)
    points1 = correspondences[:, 0:2]
    points2 = correspondences[:, 2:4]
    F_hat, best_inliers = RANSAC(correspondences, num_iterations=3000)
    showRANSAC(1, 2, best_inliers)

