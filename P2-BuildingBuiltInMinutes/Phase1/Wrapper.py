# Create Your Own Starter Code :)
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import LoadData
from EstimateFundamentalMatrix import estimateF, plot_epipolar_lines
from EssentialMatrixFromFundamentalMatrix import *
from GetInliersRANSAC import RANSAC, showRANSAC, cv2RANSAC
from GetInliersRANSAC import RANSAC_7pt

from ExtractCameraPose import *
from LinearTriangulation import *

if __name__ == '__main__':
    K = loadCalibrationMatrix()
    
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
    F, best_inliers, outliers = RANSAC_7pt(correspondences, threshold=6, acc_thresh=0.80)

    # showRANSAC(1, 2, best_inliers, outliers)
    #
    # F, best_inliers, outliers = cv2RANSAC(correspondences, threshold=2)
    # showRANSAC(1, 2, best_inliers, outliers, title="OpenCV RANSAC")
    random_samples = np.random.default_rng().choice(a=correspondences, size=10, replace=False, axis=0,
                                                    shuffle=False)
    plot_epipolar_lines(F, random_samples[:20, 0:2], random_samples[:20, 2:4], img1, img2)

    print("Fundamental Matrix- ", F)
    EssentialMatrix= estimateE(F)
    print("Essential Matrix- ", EssentialMatrix)
    Cout, Rout= extract_camera_pose(EssentialMatrix)
    print("Cout, Rout", Cout, Rout)

    linearPoints= linearTriangulation(K, Rout[0], Cout[0], Rout[1], Cout[1], points1, points2)
    print("LinearPoints", linearPoints)
    seeTriangulation(linearPoints)
