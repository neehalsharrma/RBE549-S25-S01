"""
This module contains functions for plotting various aspects of the 3D reconstruction process,
including RANSAC inliers, linear and non-linear triangulation, camera center optimization,
bundle adjustment, and the final 3D reconstruction.

Functions
---------
show_RANSAC(image1, image2, inliers, outliers, save=True, save_path='Outputs/', title=None)
    Display the images with the inliers and outliers.

plot_linear_triangulation(K, C_out, R_out, points1, points2, save_path='Outputs/', save=True)
    Plot linear triangulation points on a plane.

plot_non_linear_triangulation(img1, img2, P1, P2, optimized, linear, img_points, save_path='Outputs/', save=True)
    Plot non-linear triangulation points on a plane.

plot_optimizations(C_old, C_new, points_old, points_new)
    Plot the camera center optimization.

plot_bundle_adj(C_mats, world_points_old, world_points)
    Plot the bundle adjustment.

plot_reconstruction(C_mats, world_points)
    Plot the final 3D reconstruction.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from Utils.LoadData import load_image
from LinearTriangulation import linearTriangulation
import os


def show_RANSAC(
    image1: int,
    image2: int,
    inliers: np.array,
    outliers: np.array,
    save: bool = True,
    save_path: str = "Outputs/",
    title=None,
) -> None:
    """
    Display the images with the inliers and outliers.

    Parameters
    ----------
    image1 : int
        The first image index.
    image2 : int
        The second image index.
    inliers : np.array
        The inliers as an n x 4 array.
    outliers : np.array
        The outliers as an n x 4 array.
    save : bool, optional
        Whether to save the plot (default is True).
    save_path : str, optional
        The path to save the plot (default is 'Outputs/').
    title : str, optional
        The title of the plot (default is None).

    Returns
    -------
    None
    """
    # Load images and concatenate them side by side
    img1 = load_image(image1)
    img2 = load_image(image2)
    img = np.concatenate((img1, img2), axis=1)
    # Draw inliers
    for i in range(inliers.shape[0]):
        # Hide some of the inliers to make the image easier to see
        if i % 5 != 0:
            continue
        x1, y1, x2, y2 = inliers[i]
        cv2.circle(img, (int(x1), int(y1)), 5, (0, 0, 255), -1)
        cv2.circle(img, (int(x2) + img1.shape[1], int(y2)), 5, (0, 0, 255), -1)
        cv2.line(
            img, (int(x1), int(y1)), (int(x2) + img1.shape[1], int(y2)), (0, 127, 0), 1
        )
    # Draw outliers
    if outliers is not None:
        for i in range(outliers.shape[0]):
            if i % 5 != 0:
                continue
            x1, y1, x2, y2 = outliers[i]
            cv2.circle(img, (int(x1), int(y1)), 5, (0, 0, 255), -1)
            cv2.circle(img, (int(x2) + img1.shape[1], int(y2)), 5, (0, 0, 255), -1)
            cv2.line(
                img,
                (int(x1), int(y1)),
                (int(x2) + img1.shape[1], int(y2)),
                (255, 0, 0),
                1,
            )
    # Display and save the plot
    plt.figure(figsize=(10, 10))
    if title is not None:
        plt.title(title)
    else:
        plt.title("RANSAC Inliers")
    plt.imshow(img)
    if save:
        plt.savefig(
            save_path + "RANSAC" + "_" + str(image1) + "_" + str(image2) + ".png"
        )
    plt.axis("off")
    plt.show()


def plot_linear_triangulation(
    K, C_out, R_out, points1, points2, save_path: str = "Outputs/", save: bool = True
) -> None:
    """
    Plot linear triangulation points on a plane.

    Parameters
    ----------
    K : np.array
        The intrinsic camera matrix in the shape of (3, 3).
    C_out : np.array
        The camera centers as a 4 x n x 3 array.
    R_out : np.array
        The rotation matrices as a 4 x 3 x 3 array.
    points1 : np.array
        The points from the first image in the shape of (n, 3).
    points2 : np.array
        The points from the second image in the shape of (n, 3).
    save_path : str, optional
        The path to save the plot (default is 'Outputs/').
    save : bool, optional
        Whether to save the plot (default is True).

    Returns
    -------
    None
    """
    # Plot triangulated points for each camera
    fig = plt.figure()
    ax = fig.add_subplot(111)

    colors = ["r", "c", "b", "y"]
    for i in range(C_out.shape[0]):
        points = linearTriangulation(
            K,
            np.eye(3),
            np.zeros((3, 1)),
            R_out[i],
            C_out[i].reshape(3, 1),
            points1,
            points2,
        )
        ax.scatter(
            points[:, 0], points[:, 2], marker="o", label=f"Camera {i + 1}", c=colors[i]
        )
        ax.plot(C_out[i, 0], C_out[i, 2], marker="x", color="k", markersize=10)
    ax.plot([0], [0], marker="x", color="k", markersize=10)

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_title("3D Reconstruction")
    # Display and save the plot
    if save:
        plt.savefig(os.path.join(save_path, "LinearTriangulation.png"))
    plt.show()


def plot_non_linear_triangulation(
    img1,
    img2,
    P1: np.array,
    P2: np.array,
    optimized: np.array,
    linear: np.array,
    img_points: np.array,
    save_path: str = "Outputs/",
    save: bool = True,
) -> None:
    """
    Plot non-linear triangulation points on a plane.

    Parameters
    ----------
    img1 : np.array
        The first image.
    img2 : np.array
        The second image.
    P1 : np.array
        The first camera matrix in the shape of (3, 4).
    P2 : np.array
        The second camera matrix in the shape of (3, 4).
    optimized : np.array
        The optimized 3D points in the shape of (n, 4).
    linear : np.array
        The linear 3D points in the shape of (n, 4).
    img_points : np.array
        The image points in the shape of (n, 4).
    save_path : str, optional
        The path to save the plot (default is 'Outputs/').
    save : bool, optional
        Whether to save the plot (default is True).

    Returns
    -------
    None
    """
    # Plot detected feature points on 2D images
    one = img1.copy()
    two = img2.copy()
    points1, points2 = img_points[:, :2], img_points[:, 2:]
    for i in range(img_points.shape[0]):
        one = cv2.circle(one, (int(points1[i, 0]), int(points1[i, 1])), 5, (0, 0, 255))
        two = cv2.circle(two, (int(points2[i, 0]), int(points2[i, 1])), 5, (0, 0, 255))
    lin1 = one.copy()
    lin2 = two.copy()
    opt1 = one.copy()
    opt2 = two.copy()
    # Reproject linear and non-linear 3D points on images
    for i in range(linear.shape[0]):
        t_pt = linear[i]
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

        o_pt = optimized[i]
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

    # Display and save the plot
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
    if save:
        plt.savefig(os.path.join(save_path, "NonLinearTriangulation.png"))
    plt.show()


def plot_optimizations(C_old, C_new, points_old, points_new):
    """
    Plot the camera center optimization.

    Parameters
    ----------
    C_old : np.array
        The old camera center.
    C_new : np.array
        The new camera center.
    points_old : np.array
        The old points in the shape of (n, 4).
    points_new : np.array
        The new points in the shape of (n, 4).

    Returns
    -------
    None
    """
    # Plot old and new camera centers and points
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(
        points_old[:, 0], points_old[:, 2], c="r", marker="o", label="Old Points"
    )
    ax.scatter(
        points_new[:, 0], points_new[:, 2], c="b", marker="o", label="New Points"
    )
    ax.scatter(C_old[0], C_old[2], c="r", marker="x", label="Old Camera Center")
    ax.scatter(C_new[0], C_new[2], c="b", marker="x", label="New Camera Center")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")

    ax.set_title("Camera Center Optimization")
    ax.legend()
    # Display the plot
    plt.show()


def plot_bundle_adj(C_mats, world_points_old, world_points):
    """
    Plot the bundle adjustment.

    Parameters
    ----------
    C_mats : list of np.array
        The camera centers as a list of n x 3 arrays.
    world_points_old : np.array
        The old world points as an n x 4 array.
    world_points : np.array
        The world points as an n x 4 array.

    Returns
    -------
    None
    """
    # Plot old and new world points and camera centers
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(len(C_mats)):
        ax.scatter(world_points[i, 0], world_points[i, 2], c="r", marker="o")
        ax.scatter(C_mats[i][0], C_mats[i][2], c="b", marker="x")
        ax.scatter(C_mats[i][0], C_mats[i][2], c="b", marker="x")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_title("Bundle Adjustment")
    # Display the plot
    plt.show()


def plot_reconstruction(C_mats, world_points):
    """
    Plot the final 3D reconstruction.

    Parameters
    ----------
    C_mats : list of np.array
        The camera centers as a list of n x 3 arrays.
    world_points : np.array
        The world points as an n x 4 array.

    Returns
    -------
    None
    """
    # Plot world points and camera centers in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for i in range(len(C_mats)):
        ax.scatter(
            world_points[i, 0],
            world_points[i, 1],
            world_points[i, 2],
            c="r",
            marker="o",
        )
        ax.scatter(C_mats[i][0], C_mats[i][1], C_mats[i][2], c="b", marker="x")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Reconstruction")
    # Display the plot
    plt.show()
