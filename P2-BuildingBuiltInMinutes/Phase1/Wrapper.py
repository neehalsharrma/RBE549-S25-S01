# Create Your Own Starter Code :)
import argparse
import numpy as np

from Phase1.BundleAdjustment import bundle_adjustment
from PnPRANSAC import PNP_RANSAC
from Phase1.Utils.LoadData import *
from EstimateFundamentalMatrix import estimateF
from GetInliersRANSAC import RANSAC
from ExtractCameraPose import extract_camera_pose
from EssentialMatrixFromFundamentalMatrix import estimateE
from DisambiguateCameraPose import getCorrectPose
from LinearTriangulation import linearTriangulation
from NonLinearTriangulation import non_linear_triangulation
from NonlinearPnP import nonlinearPnP
from Utils.Plotting import *
from BuildVisibilityMatrix import buildVisibilityMatrix

if __name__ == '__main__':
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--NumImgs', help='The number of images', type=int, default=5)
    Parser.add_argument('--PlotAll', help='Plot all the images', type=bool, default=True)
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
    image_inliers = []
    image_outliers = []
    for i in range(1, num_images):
        RANSAC_correspondences = []
        outliers_correspondences = []
        for j in range(i + 1, num_images):
            correspondences = loadCorrespondences(i, j)
            _, inliers_ij, outliers = RANSAC(correspondences, threshold=thresh, acc_thresh=acc_thresh)
            RANSAC_correspondences.append(inliers_ij)
            outliers_correspondences.append(outliers)
        image_inliers.append(RANSAC_correspondences)
        image_outliers.append(outliers_correspondences)
    ##########################################################################################
    # Perform the start of SfM using the first two images as the basepoint
    inliers_1_2 = image_inliers[0][0]
    outliers_1_2 = image_outliers[0][0]
    # Get only the elements that have a correspondence with the first two images
    points1 = inliers_1_2[:, :2]
    points2 = inliers_1_2[:, 2:]
    ################################################################################################
    # Show the features and the matches of the first two images
    correspondences = loadCorrespondences(1, 2)
    img1 = loadImage(1)
    img2 = loadImage(2)
    showMatches2(1, 2, points1, points2)
    showRANSAC(1, 2, inliers_1_2, outliers_1_2)
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
        plot_linear_triangulation(K, np.expand_dims(C,axis=0), np.expand_dims(R, axis=0), points1, points2)
    ##########################################################################################
    optimized, _ = non_linear_triangulation(K, R0, C0, R, C, points1, points2, triangulated_points)
    if plot:
        P1 = K @ np.hstack((R0, -C0))
        P2 = K @ np.hstack((R, -R @ C))
        plot_non_linear_triangulation(img1, img2, P1, P2, optimized, triangulated_points, np.hstack((points1, points2)))
    ##########################################################################################
    # Create an array that stores the 2D and 3D correspondences for first image to the world points
    img_X_x_matching = np.hstack((points1, optimized))

    # Perform the rest of the SfM using the rest of the images in reference to image 1
    C_set = [C]
    R_set = [R]
    world_points = optimized
    for i in range(1, num_images + 1):
        img_i_correspondences = image_inliers[0][i]
        # Assuming img_i_correspondences and points1 are already defined
        # Find the indices of elements in img_i_correspondences[:, :2] that are not in points1
        indices = np.argwhere(~np.isin(img_i_correspondences[:, :2], points1).all(axis=1)).ravel()
        # Get the world points that exist in the current image and the first image and then use this to make the correspondences
        pnp_world_points = img_X_x_matching[indices, 2:]

        pnp_img_points = img_i_correspondences[indices, 2:]
        pnp_img_points = np.hstack((pnp_img_points, np.ones((pnp_img_points.shape[0], 1))))
        base_img_points = img_i_correspondences[indices, :2]

        C_pnp, R_pnp, inliers_pnp, outliers_pnp = PNP_RANSAC(pnp_world_points, pnp_img_points, K, threshold=0.1,
                                                             acc_thresh=0.85, max_iters=1000)
        points = pnp_world_points[inliers_pnp]
        opt_points = pnp_img_points[inliers_pnp]

        C_pnp_opt, R_pnp_opt = nonlinearPnP(K, R_pnp, C_pnp, pnp_img_points, pnp_world_points)

        P = K @ np.hstack((R_pnp_opt, -R_pnp_opt @ C_pnp_opt))
        opt_points = np.hstack((opt_points, np.ones((opt_points.shape[0], 1))))
        opt_points = P @ opt_points

        plot_optimizations(C_pnp, C_pnp_opt, points, opt_points)

        C_set.append(C_pnp)
        R_set.append(R_pnp)
        X_new = linearTriangulation(K, R0, C0, R_pnp_opt, C_pnp_opt, base_img_points, pnp_img_points)
        X_new = non_linear_triangulation(K, R0, C0, R_pnp_opt, C_pnp_opt, base_img_points, pnp_img_points, X_new)

        img_X_x_matching = np.vstack((img_X_x_matching, np.hstack((base_img_points, X_new))))
        world_points = np.vstack((world_points, X_new))

        vis = buildVisibilityMatrix(C_set, R_set, K, world_points, image_inliers[0])

        # Bundle Adjustment
        C_set, R_set, world_points_new = bundle_adjustment(C_set, R_set, K, vis, world_points, image_inliers[0])
        world_points = world_points_new
        plot_bundle_adj(C_set, R_set, world_points)

    # Plot the final 3D reconstruction
    plot_reconstruction(C_set, world_points)