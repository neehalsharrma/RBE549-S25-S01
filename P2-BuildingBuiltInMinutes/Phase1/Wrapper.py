import cv2
import matplotlib.pyplot as plt
import numpy as np
from DisambiguateCameraPose import getCorrectPose
from EssentialMatrixFromFundamentalMatrix import estimateE
from ExtractCameraPose import extract_camera_pose
from GetInliersRANSAC import RANSAC, showRANSAC
from LinearTriangulation import linearTriangulation
from LoadData import loadCalibrationMatrix, loadImage, loadCorrespondences
from NonLinearTriangulation import non_linear_triangulation


def main() -> None:
    """
    Main function to perform 3D reconstruction from two images.
    """
    # Load the calibration matrix
    K = loadCalibrationMatrix()

    # Load the images
    img1 = loadImage(1)
    img2 = loadImage(2)

    # Load the correspondences between the images
    correspondences = loadCorrespondences(1, 2)

    # Perform RANSAC to find the best fundamental matrix and inliers
    F, best_inliers, outliers = RANSAC(
        correspondences, threshold=0.125, acc_thresh=0.85
    )
    # Visualize the RANSAC results
    # Visualize the RANSAC results with image indices 1 and 2
    showRANSAC(image1=1, image2=2, inliers=best_inliers, outliers=outliers)

    # Extract the inlier points
    points1 = best_inliers[:, 0:2]
    points2 = best_inliers[:, 2:4]

    # Estimate the essential matrix from the fundamental matrix
    essential = estimateE(F, K)

    # Extract the camera poses from the essential matrix
    C_out, R_out = extract_camera_pose(essential)

    # Plot the initial 3D points using linear triangulation
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = ["r", "c", "b", "y"]

    for camera_index in range(C_out.shape[0]):
        points = linearTriangulation(
            K,
            np.eye(3),
            np.zeros((3, 1)),
            R_out[camera_index],
            C_out[camera_index].reshape(3, 1),
            points1,
            points2,
        )
        ax.scatter(
            points[:, 0], points[:, 2], marker="o", label=f"Camera {camera_index + 1}", c=colors[camera_index]
        )
        ax.plot(C_out[camera_index, 0], C_out[camera_index, 2], marker="x", color="k", markersize=10)
    ax.plot([0], [0], marker="x", color="k", markersize=10)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("3D Reconstruction")
    ax.legend()
    plt.show()

    # Disambiguate the correct camera pose
    C, R, inlier_idx = getCorrectPose(K, C_out, R_out, points1, points2)
    points1 = points1[inlier_idx]
    points2 = points2[inlier_idx]

    # Compute the projection matrices
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ R @ np.hstack((np.eye(3), -C))

    # Perform linear triangulation with the correct pose
    triangulated_points = linearTriangulation(
        K, np.eye(3), np.zeros((3, 1)), R, C, points1, points2
    )
    # Plot the triangulated 3D points
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(triangulated_points[:, 0], triangulated_points[:, 2], c="b", marker="o")
    ax.plot([0], [0], marker="x", color="k", markersize=10)
    ax.plot(C[0], C[2], marker="o", color="g", markersize=10)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.show()

    # Draw circles on the original images for visualization
    one = img1.copy()
    two = img2.copy()
    for point_index in range(points1.shape[0]):
        one = cv2.circle(one, (int(points1[point_index, 0]), int(points1[point_index, 1])), 5, (0, 0, 255))
        two = cv2.circle(two, (int(points2[point_index, 0]), int(points2[point_index, 1])), 5, (0, 0, 255))
    lin1 = one.copy()
    lin2 = two.copy()

    # Perform non-linear triangulation to optimize the 3D points
    optimized_points, costs = non_linear_triangulation(
        K, np.eye(3), np.zeros((3, 1)), R, C, points1, points2, triangulated_points
    )
    opt1 = one.copy()
    opt2 = two.copy()

    # Reproject the points and draw circles for linear and non-linear triangulation
    for point_index in range(points1.shape[0]):
        t_pt = triangulated_points[point_index]
        reproj_linear_1 = np.dot(P1, t_pt)
        reproj_linear_1 /= reproj_linear_1[2]
        reproj_linear_2 = np.dot(P2, t_pt)
        reproj_linear_2 /= reproj_linear_2[2]
        lin1 = cv2.circle(
            lin1, (int(reproj_linear_1[0]), int(reproj_linear_1[1])), 5, (0, 255, 0)
        )
        lin2 = cv2.circle(
            lin2, (int(reproj_linear_2[0]), int(reproj_linear_2[1])), 5, (0, 255, 0)
        )

        o_pt = optimized_points[point_index]
        reproj_optimized_1 = np.dot(P1, o_pt)
        reproj_optimized_1 /= reproj_optimized_1[2]
        reproj_optimized_2 = np.dot(P2, o_pt)
        reproj_optimized_2 /= reproj_optimized_2[2]
        opt1 = cv2.circle(
            opt1,
            (int(reproj_optimized_1[0]), int(reproj_optimized_1[1])),
            5,
            (255, 0, 0),
        )
        opt2 = cv2.circle(
            opt2,
            (int(reproj_optimized_2[0]), int(reproj_optimized_2[1])),
            5,
            (255, 0, 0),
        )

    # Save the results of linear and non-linear triangulation
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(lin1)
    ax[0, 0].axis("off")
    ax[0, 0].set_title("Linear Triangulation 1")
    ax[0, 1].imshow(lin2)
    ax[0, 1].axis("off")
    ax[0, 1].set_title("Linear Triangulation 2")
    ax[1, 0].imshow(opt1)
    ax[1, 0].axis("off")
    ax[1, 0].set_title("Non-Linear Triangulation 1")
    ax[1, 1].imshow(opt2)
    ax[1, 1].axis("off")
    ax[1, 1].set_title("Non-Linear Triangulation 2")
    plt.savefig('Outputs/triangulation_results.png')
    plt.close()

    # Save the 3D points from linear and non-linear triangulation
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(optimized_points[:, 0], optimized_points[:, 2], c="r", marker="o")
    ax.scatter(triangulated_points[:, 0], triangulated_points[:, 2], c="b", marker="o")
    ax.plot([0], [0], marker="x", color="k", markersize=10)
    ax.plot(C[0], C[2], marker="o", color="g", markersize=10)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("3D Reconstruction")
    plt.savefig('Outputs/3d_reconstruction.png')
    plt.close()


if __name__ == "__main__":
    main()
