"""
Module for estimating the Fundamental Matrix and plotting epipolar lines.

This module provides functions to estimate the Fundamental Matrix from point correspondences
between two images, estimate the epipole, and plot the epipolar lines on the images.

Functions
---------
getEquation(Point1, Point2)
    Construct the equation for the Fundamental Matrix estimation.
estimate_F(points1, points2)
    Estimate the Fundamental Matrix from the given points.
estimate_F_7pt(points1, points2)
    Estimate the Fundamental Matrix using the 7-point algorithm.
estimate_epipole(F)
    Estimate the epipole from the Fundamental Matrix.
plot_epipolar_lines(F, points1, points2, img1, img2)
    Plot the epipolar lines on the images.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

def getEquation(Point1, Point2):
    """
    Construct the equation for the Fundamental Matrix estimation.

    Parameters
    ----------
    Point1 : array-like
        The coordinates of the point in the first image (x1, y1).
    Point2 : array-like
        The coordinates of the point in the second image (x2, y2).

    Returns
    -------
    numpy.ndarray
        The equation coefficients as a 1D array.
    """
    x1, y1 = Point1
    x2, y2 = Point2
    return np.array([x1 * x2, y1 * x2, x2, x1 * y2, y1 * y2, y2, x1, y1, 1])

def estimate_F(points1, points2):
    """
    Estimate the Fundamental Matrix from the given points.

    Parameters
    ----------
    points1 : numpy.ndarray
        The points from the first image in the shape of (n, 2).
    points2 : numpy.ndarray
        The points from the second image in the shape of (n, 2).

    Returns
    -------
    numpy.ndarray
        The estimated Fundamental Matrix.
    """
    n = points1.shape[0]
    A = np.zeros((n, 9))
    for i in range(n):
        A[i] = getEquation(points1[i, :2], points2[i, :2])
    U, S, VT = np.linalg.svd(A)
    V = VT.T
    F = V[:, -1].reshape(3, 3)  # Last row of V is the row of F

    # Enforcing Rank 2
    U, S, VT = np.linalg.svd(F)
    S[-1] = 0
    F = U @ np.diag(S) @ VT

    return F

def estimate_F_7pt(points1, points2):
    """
    Estimate the Fundamental Matrix using the 7-point algorithm.

    Parameters
    ----------
    points1 : numpy.ndarray
        The points from the first image in the shape of (n, 2).
    points2 : numpy.ndarray
        The points from the second image in the shape of (n, 2).

    Returns
    -------
    numpy.ndarray
        The estimated Fundamental Matrix. Either 3x3 or 3x3xN.
    """
    n = points1.shape[0]
    A = np.zeros((n, 9))

    points1 = np.append(points1, np.ones((n, 1)), axis=1)  # n x 3
    points2 = np.append(points2, np.ones((n, 1)), axis=1)  # n x 3
    for i in range(n):
        A[i] = getEquation(points1[i, :2], points2[i, :2])
    U, S, VT = np.linalg.svd(A)
    V = VT.T
    # With only 7 points, the number of F's may be 3
    num_F = V.shape[1] / 9
    if num_F > 1:
        F = V[:, -1].reshape(3, 3, num_F)
        for i in range(num_F):
            U, S, VT = np.linalg.svd(F[:, :, i])
    else:
        F = V[:, -1].reshape(3, 3)  # Last row of V is the row of F
        # Enforcing Rank 2
        U, S, VT = np.linalg.svd(F)
        S[-1] = 0
        F = U @ np.diag(S) @ VT
        F = F.reshape(3, 3, 1)
    return F

def estimate_epipole(F):
    """
    Estimate the epipole from the Fundamental Matrix.

    Parameters
    ----------
    F : numpy.ndarray
        The Fundamental Matrix.

    Returns
    -------
    numpy.ndarray
        The epipole coordinates.
    """
    _, _, V = np.linalg.svd(F)
    e = V[-1, :]
    e /= e[-1]
    return e

def plot_epipolar_lines(F, points1, points2, img1, img2):
    """
    Plot the epipolar lines on the images.

    Parameters
    ----------
    F : numpy.ndarray
        The Fundamental Matrix.
    points1 : numpy.ndarray
        The points from the first image in the shape of (n, 2).
    points2 : numpy.ndarray
        The points from the second image in the shape of (n, 2).
    img1 : numpy.ndarray
        The first image.
    img2 : numpy.ndarray
        The second image.
    """
    e1 = estimate_epipole(F)
    e2 = estimate_epipole(F.T)
    for pt in points1:
        i1 = cv2.line(img1, pt.astype(np.int32), e1[:-1].astype(np.int32), (0, 0, 0), 2)
    plt.imshow(i1)
    plt.axis('off')
    plt.title('Epipolar Lines')
    plt.show()
    for pt in points2:
        i2 = cv2.line(img2, pt.astype(np.int32), e2[:-1].astype(np.int32), (0, 0, 0), 2)
    plt.imshow(i2)
    plt.axis('off')
    plt.title('Epipolar Lines')
    plt.show()
    output_dir = '/home/wpi/RBE549-S25-S01/P2-BuildingBuiltInMinutes/Phase1/Outputs/'
    plt.savefig(output_dir + 'epipolar_lines_img1.png')
    plt.imshow(i2)
    plt.axis('off')
    plt.title('Epipolar Lines')
    plt.savefig(output_dir + 'epipolar_lines_img2.png')
