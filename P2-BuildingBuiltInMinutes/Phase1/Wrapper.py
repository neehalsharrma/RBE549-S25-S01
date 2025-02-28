# Create Your Own Starter Code :)
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from LoadData import *
from EstimateFundamentalMatrix import estimateF, plot_epipolar_lines
from GetInliersRANSAC import RANSAC, showRANSAC, cv2RANSAC, RANSAC_7pt
from ExtractCameraPose import extract_camera_pose

from EssentialMatrixFromFundamentalMatrix import estimateE
from DisambiguateCameraPose import getCorrectPose
from LinearTriangulation import linearTriangulation2
from NonLinearTriangulation import non_linear_triangulation

if __name__ == '__main__':
    img1 = loadImage(1)
    img2 = loadImage(2)
    # img = np.concatenate((img1, img2), axis=1)
    # plt.imshow(img)
    # plt.axis('off')
    # plt.show()

    # showMatches(1, 2)
    # LoadData.showMatches(2, 3)
    # LoadData.showMatches(3, 4)
    # LoadData.showMatches(4, 5)
    correspondences = loadCorrespondences(1, 2)

    # F, best_inliers, outliers = RANSAC_7pt(correspondences, threshold=6, acc_thresh=0.80)

    # showRANSAC(1, 2, best_inliers, outliers)
    #
    F, best_inliers, outliers = cv2RANSAC(correspondences, threshold=2)
    points1 = best_inliers[:, 0:2]
    points2 = best_inliers[:, 2:4]
    # showRANSAC(1, 2, best_inliers, outliers, title="OpenCV RANSAC")
    K = loadCalibrationMatrix('../P2Data/')
    essential = estimateE(F, K)

    C_out, R_out = extract_camera_pose(essential)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = ['r', 'g', 'b', 'y']

    for i in range(len(C_out)):
        points = linearTriangulation2(K, np.eye(3), np.zeros((3, 1)),
                                      R_out[i].reshape(3, 3), C_out[i].reshape(3, 1),
                                      points1, points2)
        ax.scatter(points[:, 0], points[:, 1], marker='o', label=f'Camera {i + 1}', c=colors[i])
    ax.plot([0], [0], marker='x', color='k', markersize=10)
    ax.plot(C_out[:, 0], C_out[:, 2], marker='o', color='g', markersize=10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('3D Reconstruction')
    plt.show()

    C, R, inlier_idx = getCorrectPose(K, C_out, R_out, points1, points2)
    points1 = points1[inlier_idx]
    points2 = points2[inlier_idx]
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, -R @ C))

    triangulated_points = linearTriangulation2(K, np.eye(3), np.zeros((3, 1)), R, C, points1, points2)
    optimized_points, costs = non_linear_triangulation(K, np.eye(3), np.zeros((3, 1)), R, C, points1, points2,
                                                       triangulated_points)
    one = img1.copy()
    two = img2.copy()
    for i in range(points1.shape[0]):
        one = cv2.circle(one, (int(points1[i, 0]), int(points1[i, 1])), 5, (0, 0, 255))
        two = cv2.circle(two, (int(points2[i, 0]), int(points2[i, 1])), 5, (0, 0, 255))
    lin1 = one.copy()
    lin2 = two.copy()
    opt1 = one.copy()
    opt2 = two.copy()
    for i in range(points1.shape[0]):
        t_pt = np.concatenate((triangulated_points[i], [1]))
        o_pt = np.concatenate((optimized_points[i], [1]))
        reproj_linear_1 = (np.dot(P1, t_pt))
        reproj_optimized_1 = (np.dot(P1, o_pt))
        reproj_linear_2 = (np.dot(P2, t_pt))
        reproj_optimized_2 = (np.dot(P2, o_pt))

        lin1 = cv2.circle(lin1, (int(reproj_linear_1[0]), int(reproj_linear_1[1])), 5, (0, 255, 0))
        lin2 = cv2.circle(lin2, (int(reproj_linear_2[0]), int(reproj_linear_2[1])), 5, (0, 255, 0))
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
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(optimized_points[:, 0], optimized_points[:, 1], optimized_points[:, 2], c='r', marker='o')
    ax.scatter(triangulated_points[:, 0], triangulated_points[:, 1], triangulated_points[:, 2], c='b', marker='o')
    ax.plot([0], [0], [0], marker='x', color='k', markersize=10)
    ax.plot(C[0], C[1], C[2], marker='o', color='g', markersize=10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Reconstruction')
    plt.show()

    # random_samples = np.random.default_rng().choice(a=correspondences, size=10, replace=False, axis=0,
    #                                                 shuffle=False)
    # plot_epipolar_lines(F, random_samples[:20, 0:2], random_samples[:20, 2:4], img1, img2)
