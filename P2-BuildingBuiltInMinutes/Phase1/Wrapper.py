# Create Your Own Starter Code :)
import argparse

import numpy as np

from Phase1.Utils.LoadData import *
from EstimateFundamentalMatrix import estimateF
from GetInliersRANSAC import RANSAC
from ExtractCameraPose import extract_camera_pose
from EssentialMatrixFromFundamentalMatrix import estimateE
from DisambiguateCameraPose import getCorrectPose
from LinearTriangulation import linearTriangulation
from NonLinearTriangulation import non_linear_triangulation

from Utils.Plotting import *


def get_remaining_2d_3d_correspondences(j, inliers:list[np.array], ):
    """
    Get the remaining 2D and 3D correspondences for the j-th image.
    """

    pass


if __name__ == '__main__':
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--NumImgs', help='The number of images', type=int, default=5)
    Parser.add_argument('--PlotAll', help='Plot all the images', type=bool, default=False)
    Parser.add_argument('--RANSACThreshold', help='The threshold for RANSAC', type=float, default=0.125)
    Parser.add_argument('--AccThreshold', help='The threshold for the accuracy of RANSAC', type=float, default=0.85)

    args = Parser.parse_args()
    num_images = args.NumImgs
    thresh = args.RANSACThreshold
    acc_thresh = args.AccThreshold
    plot = args.PlotAll

    K = loadCalibrationMatrix()

    # Clean up all the image point correspondences
    # Store the RANSAC correspondences as a list[list[np.array]]
    # Due to how the data is stored in the calibration files,
    # the current image stores it correspondences with all subsequent images
    # i.e. image 0 stores its correspondences with images 1, 2, 3, 4
    # i.e. image 1 stores its correspondences with images 2, 3, 4
    # Thus, the number of columns in each correspondence matrix is (num_images - i) * 2 + 5
    # --> 5 for the image number, 2 for the 2D points, and 2 for the matching 2D points
    # The image inliers stores the inliers for each image with all subsequent images
    image_inliers = []
    image_correspondences = []
    for i in range(1, num_images):
        RANSAC_correspondences = []
        for j in range(i + 1, num_images):
            correspondences = loadCorrespondences(i, j)
            _, inliers_ij, _ = RANSAC(correspondences, threshold=thresh, acc_thresh=acc_thresh)
            RANSAC_correspondences.append(inliers_ij)
        image_inliers.append(RANSAC_correspondences)
        image_correspondences.append(loadDataSparse(i, num_images=num_images))
    ##########################################################################################
    # Perform the start of SfM using the first two images as the basepoint
    inliers_1_2 = image_inliers[0][0]
    # Get only the elements that have a correspondence with the first two images
    img_1_2_points = np.argwhere(image_correspondences[0][:, 5:7].sum(axis=1)).squeeze()
    img_1_2_points = image_correspondences[0][img_1_2_points,3:7]

    points1 = img_1_2_points[inliers_1_2, :2]
    points2 = img_1_2_points[inliers_1_2, 2:]
    ################################################################################################
    # Show the features and the matches of the first two images
    _, _, outliers = RANSAC(img_1_2_points, threshold=thresh, acc_thresh=acc_thresh)
    img1 = loadImage(1)
    img2 = loadImage(2)
    showMatches2(1, 2, points1, points2)
    showRANSAC(1, 2, img_1_2_points[inliers_1_2, :], img_1_2_points[outliers, :])
    ################################################################################################

    F = estimateF(points1, points2)
    # random_samples = np.random.default_rng().choice(a=correspondences, size=10, replace=False, axis=0,
    #                                                 shuffle=False)
    # plot_epipolar_lines(F, random_samples[:20, 0:2], random_samples[:20, 2:4], img1, img2)

    essential = estimateE(F, K)
    ##########################################################################################
    # Ground zero camera pose
    R0, C0 = np.eye(3), np.zeros((3, 1))
    C_out, R_out = extract_camera_pose(essential)
    if plot:
        plot_linear_triangulation(K, C_out, R_out, points1, points2)
    ##########################################################################################
    # Chierality check
    C, R, inlier_idx = getCorrectPose(K, C_out, R_out, points1, points2)
    points1 = points1[inlier_idx]
    points2 = points2[inlier_idx]
    triangulated_points = linearTriangulation(K, R0, C0, R, C, points1, points2)
    if plot:
        plot_linear_triangulation(K, C, R, points1, points2)
    ##########################################################################################
    optimized = non_linear_triangulation(K, R0, C0, R, C, points1, points2, triangulated_points)
    if plot:
        P1 = K @ np.hstack((R0, -C0))
        P2 = K @ np.hstack((R, -R @ C))
        plot_non_linear_triangulation(img1, img2, P1, P2, optimized, triangulated_points, image_correspondences[0][0])
    ##########################################################################################

    # Perform the rest of the SfM using the rest of the images in reference to image 1
    C_set = [C]
    R_set = [R]
    for i in range(1, num_images + 1):
