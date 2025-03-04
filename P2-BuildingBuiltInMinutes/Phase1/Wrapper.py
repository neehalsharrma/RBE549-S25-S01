# Create Your Own Starter Code :)
import argparse

import numpy as np
from BuildVisibilityMatrix import build_visibility_matrix
from DisambiguateCameraPose import get_correct_pose
from EssentialMatrixFromFundamentalMatrix import estimate_E
from EstimateFundamentalMatrix import estimate_F, plot_epipolar_lines
from ExtractCameraPose import extract_camera_pose
from GetInliersRANSAC import RANSAC
from LinearTriangulation import linearTriangulation
from NonlinearPnP import nonlinear_PnP
from NonLinearTriangulation import non_linear_triangulation
from BundleAdjustment import bundle_adjustment
from Utils.LoadData import (
    load_calibration_matrix,
    load_correspondences,
    load_image,
    show_matches2,
)
from PnPRANSAC import PNP_RANSAC
from Utils.Plotting import (
    plot_bundle_adj,
    plot_linear_triangulation,
    plot_non_linear_triangulation,
    plot_reconstruction,
    show_RANSAC,
)

if __name__ == "__main__":
    # Parse command-line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument("--NumImgs", help="The number of images", type=int, default=5)
    Parser.add_argument(
        "--PlotAll", help="Plot all the images", type=bool, default=True
    )
    Parser.add_argument(
        "--RANSACThreshold", help="The threshold for RANSAC", type=float, default=0.125
    )
    Parser.add_argument(
        "--AccThreshold",
        help="The threshold for the accuracy of RANSAC",
        type=float,
        default=0.85,
    )

    args = Parser.parse_args()
    num_images = args.NumImgs
    thresh = args.RANSACThreshold
    acc_thresh = args.AccThreshold
    plot = args.PlotAll

    # Load the camera calibration matrix
    K = load_calibration_matrix()

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
        for j in range(i + 1, num_images + 1):
            # Load correspondences between image i and image j
            correspondences = load_correspondences(i, j)
            # Perform RANSAC to get inliers and outliers
            _, inliers_ij, outliers = RANSAC(
                correspondences, threshold=thresh, acc_thresh=acc_thresh
            )
            # Store the inliers and outliers
            RANSAC_correspondences.append(inliers_ij)
            outliers_correspondences.append(outliers)
        # Store the inliers and outliers for the current image
        image_inliers.append(RANSAC_correspondences)
        image_outliers.append(outliers_correspondences)

    # Perform the start of SfM using the first two images as the basepoint
    inliers_1_2 = image_inliers[0][0]
    outliers_1_2 = image_outliers[0][0]
    # Get only the elements that have a correspondence with the first two images
    points1 = inliers_1_2[:, :2]
    points2 = inliers_1_2[:, 2:]

    # Show the features and the matches of the first two images
    correspondences = load_correspondences(1, 2)
    img1 = load_image(1)
    img2 = load_image(2)
    show_matches2(1, 2, points1, points2)
    show_RANSAC(1, 2, inliers_1_2, outliers_1_2)

    # Estimate the fundamental matrix
    F = estimate_F(points1, points2)
    random_samples = np.random.default_rng().choice(
        a=correspondences, size=10, replace=False, axis=0, shuffle=False
    )
    plot_epipolar_lines(
        F, random_samples[:20, 0:2], random_samples[:20, 2:4], img1, img2
    )

    # Estimate the essential matrix
    essential = estimate_E(F, K)
    ##########################################################################################
    # Ground zero camera pose
    R0, C0 = np.eye(3), np.zeros((3, 1))
    # Extract camera poses from the essential matrix
    C_out, R_out = extract_camera_pose(essential)
    if plot:
        plot_linear_triangulation(K, C_out, R_out, points1, points2)
    ##########################################################################################
    # Chierality check to get the correct camera pose
    C, R, inlier_idx = get_correct_pose(K, C_out, R_out, points1, points2)
    points1 = points1[inlier_idx]
    points2 = points2[inlier_idx]
    # Perform linear triangulation
    triangulated_points = linearTriangulation(K, R0, C0, R, C, points1, points2)
    if plot:
        plot_linear_triangulation(
            K, np.expand_dims(C, axis=0), np.expand_dims(R, axis=0), points1, points2
        )
    ##########################################################################################
    # Homogenize the 2D points to be an (n, 3) matrix
    x1 = np.hstack((points1, np.ones((points1.shape[0], 1))))
    x2 = np.hstack((points2, np.ones((points1.shape[0], 1))))
    # Perform non-linear triangulation
    optimized, _ = non_linear_triangulation(
        K, R0, C0, R, C, x1, x2, triangulated_points
    )
    if plot:
        P1 = K @ np.hstack((R0, -C0))
        P2 = K @ np.hstack((R, -R @ C))
        plot_non_linear_triangulation(
            img1,
            img2,
            P1,
            P2,
            optimized,
            triangulated_points,
            np.hstack((points1, points2)),
        )
    ##########################################################################################
    # Create an array that stores the 2D and 3D correspondences for first image to the world points
    img_X_x_matching = np.hstack((points1, optimized))

    # Perform the rest of the SfM using the rest of the images in reference to image 1
    C_set = [C]
    R_set = [R]
    world_points = optimized
    for i in range(1, num_images):
        img_i_correspondences = image_inliers[0][i]

        # Assuming img_i_correspondences and points1 are already defined
        # Find the indices of elements in img_i_correspondences[:, :2] that are in points1
        indices = np.argwhere(
            np.isin(img_i_correspondences[:, :2], points1).all(axis=1)
        ).ravel()
        # Get the world points that exist in the current image
        # and the first image and then use this to make the correspondences
        pnp_world_points = img_X_x_matching[indices, 2:]

        base_img_points = img_i_correspondences[indices, :2]
        base_img_points = np.hstack(
            (base_img_points, np.ones((base_img_points.shape[0], 1)))
        )
        pnp_img_points = img_i_correspondences[indices, 2:]
        pnp_img_points = np.hstack(
            (pnp_img_points, np.ones((pnp_img_points.shape[0], 1)))
        )

        # Perform PnP RANSAC to get camera pose
        C_pnp, R_pnp, inliers_pnp, outliers_pnp = PNP_RANSAC(
            pnp_world_points,
            pnp_img_points,
            K,
            threshold=100,
            acc_thresh=0.85,
            max_iters=10000,
        )
        points = pnp_world_points[inliers_pnp]
        opt_points = pnp_img_points[inliers_pnp]

        # Refine camera pose using non-linear PnP
        C_pnp_opt, R_pnp_opt = nonlinear_PnP(
            K, R_pnp, C_pnp, pnp_img_points, pnp_world_points
        )

        C_set.append(C_pnp)
        R_set.append(R_pnp)
        # Perform linear triangulation for new points
        X_new = linearTriangulation(
            K, R0, C0, R_pnp_opt, C_pnp_opt, base_img_points, pnp_img_points[:, :2]
        )
        # Perform non-linear triangulation for new points
        X_new, _ = non_linear_triangulation(
            K, R0, C0, R_pnp_opt, C_pnp_opt, base_img_points, pnp_img_points, X_new
        )
        P_opt = K @ np.hstack((R_pnp_opt, -R_pnp_opt @ C_pnp_opt))
        points_2d = (P_opt @ X_new.T).T
        points_2d = points_2d / points_2d[:, 2, np.newaxis]

        # Update the 2D-3D correspondences
        img_X_x_matching = np.vstack(
            (img_X_x_matching, np.hstack((points_2d[:, :2], X_new)))
        )
        world_points = np.vstack((world_points, X_new))

        # Build visibility matrix
        vis = build_visibility_matrix(
            C_set, R_set, K, world_points, [img[:, :2] for img in image_inliers[0]]
        )

        # Perform bundle adjustment
        C_set, R_set, world_points_new = bundle_adjustment(
            C_set, R_set, K, vis, world_points, image_inliers[0]
        )
        world_points = world_points_new
        plot_bundle_adj(C_set, R_set, world_points)

    # Plot the final 3D reconstruction
    plot_reconstruction(C_set, world_points)
