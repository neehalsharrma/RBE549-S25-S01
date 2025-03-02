import numpy as np
import matplotlib.pyplot as plt
import cv2
from Phase1.Utils.LoadData import loadImage
from Phase1.LinearTriangulation import linearTriangulation
import os


def showRANSAC(image1: int, image2: int, inliers: np.array, outliers: np.array, save: bool = False,
               save_path: str = '../../Results/', title=None) -> None:
    """
    Display the images with the inliers.
    @ img1: The first image.
    @ img2: The second image.
    @ inliers: The inliers as an n x 4 array.
    """
    img1 = loadImage(image1)
    img2 = loadImage(image2)
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


def plot_linear_triangulation(K, C_out, R_out, points1, points2, save_path: str = '../../Results/',
                              save: bool = False) -> None:
    """
    Plot linear triangulation points on a plane
    @ C_out: The camera centers as an 4 x n x 3 array.
    @ R_out: The rotation matrices as an 4 x 3 x 3 array.
    @ points1: The points from the first image in the shape of (n, 3)
    @ points2: The points from the second image.in the shape of (n, 3)
    @ K: The intrinsic camera matrix in the shape of (3, 3)
    @ return: None
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    colors = ['r', 'c', 'b', 'y']
    for i in range(C_out.shape[0]):
        points = linearTriangulation(K, np.eye(3), np.zeros((3, 1)),
                                     R_out[i], C_out[i].reshape(3, 1),
                                     points1, points2)
        ax.scatter(points[:, 0], points[:, 2], marker='o', label=f'Camera {i + 1}', c=colors[i])
        ax.plot(C_out[i, 0], C_out[i, 2], marker='x', color='k', markersize=10)
    ax.plot([0], [0], marker='x', color='k', markersize=10)

    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_title('3D Reconstruction')
    if save:
        plt.savefig(os.path.join(save_path, 'LinearTriangulation.png'))
    plt.show()


def plot_non_linear_triangulation(img1, img2, P1: np.array, P2: np.array, optimized: np.array, linear: np.array,
                                  img_points: np.array, save_path: str = '../../Results/', save: bool = False) -> None:
    """
    Plot non-linear triangulation points on a plane
    @ img1: The first image
    @ img2: The second image
    @ P1: The first camera matrix in the shape of (3, 4)
    @ P2: The second camera matrix in the shape of (3, 4)
    @ optimized: The optimized 3D points in the shape of (n, 4)
    @ linear: The linear 3D points in the shape of (n, 4)
    @ img_points: The image points in the shape of (n, 4)
    @ return: None
    """
    # Plot detected feature points on 2d image
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
    # Reproject linear and nonlinear homogeneous 3d points on images with detected feature points
    # Each reprojected point is 2d homogeneous
    for i in range(linear.shape[0]):
        t_pt = linear[i]
        reproj_linear_1 = (np.dot(P1, t_pt))
        reproj_linear_1 /= reproj_linear_1[2]
        reproj_linear_2 = (np.dot(P2, t_pt))
        reproj_linear_2 /= reproj_linear_2[2]
        lin1 = cv2.circle(lin1, (int(reproj_linear_1[0]), int(reproj_linear_1[1])), 5, (0, 255, 0))
        lin2 = cv2.circle(lin2, (int(reproj_linear_2[0]), int(reproj_linear_2[1])), 5, (0, 255, 0))

        o_pt = optimized[i]
        reproj_optimized_1 = (np.dot(P1, o_pt))
        reproj_optimized_1 /= reproj_optimized_1[2]
        reproj_optimized_2 = (np.dot(P2, o_pt))
        reproj_optimized_2 /= reproj_optimized_2[2]
        opt1 = cv2.circle(opt1, (int(reproj_optimized_1[0]), int(reproj_optimized_1[1])), 5, (255, 0, 0))
        opt2 = cv2.circle(opt2, (int(reproj_optimized_2[0]), int(reproj_optimized_2[1])), 5, (255, 0, 0))

    fig, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(lin1)
    ax[0, 0].axis('off')
    ax[0, 0].set_title('Linear Triangulation 1')
    ax[0, 1].imshow(lin2)
    ax[0, 1].axis('off')
    ax[0, 1].set_title('Linear Triangulation 2')
    ax[1, 0].imshow(opt1)
    ax[1, 0].axis('off')
    ax[1, 0].set_title('Non-Linear Triangulation 1')
    ax[1, 1].imshow(opt2)
    ax[1, 1].axis('off')
    ax[1, 1].set_title('Non-Linear Triangulation 2')
    if save:
        plt.savefig(os.path.join(save_path, 'NonLinearTriangulation.png'))
    plt.show()


def plot_optimizations(C_old, C_new, points_old, points_new):
    """
    Plot the camera center optimization.
    @ C_old: The old camera center.
    @ C_new: The new camera center.
    @ points_old: The old points shape of (n, 4).
    @ points_new: The new points shape of (n, 4).
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(points_old[:, 0], points_old[:, 2], c='r', marker='o', label='Old Points')
    ax.scatter(points_new[:, 0], points_new[:, 2], c='b', marker='o', label='New Points')
    ax.scatter(C_old[0], C_old[2], c='r', marker='x', label='Old Camera Center')
    ax.scatter(C_new[0], C_new[2], c='b', marker='x', label='New Camera Center')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Center Optimization')
    ax.legend()
    plt.show()

def plot_bundle_adj(C_mats, world_points_old, world_points):
    """
    Plot the bundle adjustment.
    @ C_mats: The camera centers as a list of n x 3 arrays.
    @ world_points_old: The old world points as an n x 4 array.
    @ world_points: The world points as an n x 4 array.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(len(C_mats)):
        ax.scatter(world_points[i, 0], world_points[i, 2], c='r', marker='o')
        ax.scatter(C_mats[i][0], C_mats[i][2], c='b', marker='x')
        ax.scatter(C_mats[i][0], C_mats[i][2], c='b', marker='x')
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_title('Bundle Adjustment')
    plt.show()


def plot_reconstruction(C_mats, world_points):
    """
    Plot the final 3D reconstruction.
    @ C_mats: The camera centers as a list of n x 3 arrays.
    @ R_mats: The rotation matrices as a list of 3 x 3 arrays.
    @ world_points: The world points as an n x 4 array.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(C_mats)):
        ax.scatter(world_points[i, 0], world_points[i, 1], world_points[i, 2], c='r', marker='o')
        ax.scatter(C_mats[i][0], C_mats[i][1], C_mats[i][2], c='b', marker='x')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Reconstruction')
    plt.show()