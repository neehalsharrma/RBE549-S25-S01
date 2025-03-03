"""
This module provides functions to estimate the Fundamental Matrix and plot epipolar lines.

Functions
---------
get_equation(Point1: np.ndarray, Point2: np.ndarray) -> np.ndarray
    Compute the equation for the fundamental matrix estimation.

estimateF(points1: np.ndarray, points2: np.ndarray) -> np.ndarray
    Estimate the Fundamental Matrix from the given points.

estimateF_7pt(points1: np.ndarray, points2: np.ndarray) -> np.ndarray
    Estimate the Fundamental Matrix from the given points using the 7-point algorithm.

estimate_epipole(F: np.ndarray) -> np.ndarray
    Estimate the epipole from the Fundamental Matrix.

plot_epipolar_lines(F: np.ndarray, points1: np.ndarray, points2: np.ndarray, img1: np.ndarray, img2: np.ndarray) -> None
    Plot the epipolar lines on the images.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


def get_equation(Point1: np.ndarray, Point2: np.ndarray) -> np.ndarray:
    """
    Compute the equation for the fundamental matrix estimation.

    Parameters
    ----------
    Point1 : np.ndarray
        The point from the first image as (x1, y1).
    Point2 : np.ndarray
        The point from the second image as (x2, y2).

    Returns
    -------
    np.ndarray
        The equation as a 1x9 array.
    """
    x1, y1 = Point1
    x2, y2 = Point2
    return np.array([x1 * x2, y1 * x2, x2, x1 * y2, y1 * y2, y2, x1, y1, 1])


def estimateF(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """
    Estimate the Fundamental Matrix from the given points.

    Parameters
    ----------
    points1 : np.ndarray
        The points from the first image in the shape of (n, 2).
    points2 : np.ndarray
        The points from the second image in the shape of (n, 2).

    Returns
    -------
    np.ndarray
        The estimated Fundamental Matrix.
    """
    # Number of point correspondences
    n = points1.shape[0]

    # Initialize matrix A with zeros
    A = np.zeros((n, 9))

    # Construct the matrix A using the point correspondences
    for i in range(n):
        A[i] = get_equation(points1[i, :2], points2[i, :2])

    # Perform Singular Value Decomposition (SVD) on matrix A
    U, S, VT = np.linalg.svd(A)
    V = VT.T

    # The last column of V (corresponding to the smallest singular value) is the solution for F
    F = V[:, -1].reshape(3, 3)

    # Enforce rank 2 constraint on the Fundamental Matrix
    U, S, VT = np.linalg.svd(F)
    S[-1] = 0  # Set the smallest singular value to 0
    F = U @ np.diag(S) @ VT  # Recompute F with the rank 2 constraint

    return F


def estimateF_7pt(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """
    Estimate the Fundamental Matrix from the given points using the 7-point algorithm.

    Parameters
    ----------
    points1 : np.ndarray
        The points from the first image in the shape of (n, 2).
    points2 : np.ndarray
        The points from the second image in the shape of (n, 2).

    Returns
    -------
    np.ndarray
        The estimated Fundamental Matrix. Either 3x3 or 3x3xN.
    """
    n = points1.shape[0]
    A = np.zeros((n, 9))

    # Append a column of ones to the points to convert them to homogeneous coordinates
    points1 = np.append(points1, np.ones((n, 1)), axis=1)  # n x 3
    points2 = np.append(points2, np.ones((n, 1)), axis=1)  # n x 3

    # Construct the matrix A using the point correspondences
    for i in range(n):
        A[i] = get_equation(points1[i, :2], points2[i, :2])

    # Perform Singular Value Decomposition (SVD) on matrix A
    U, S, VT = np.linalg.svd(A)
    V = VT.T

    # Determine the number of possible Fundamental Matrices
    num_F = V.shape[1] // 9

    if num_F > 1:
        # Reshape the last columns of V into multiple 3x3 matrices
        F = V[:, -1].reshape(3, 3, num_F)
        for i in range(num_F):
            U, S, VT = np.linalg.svd(F[:, :, i])
    else:
        # Reshape the last column of V into a 3x3 matrix
        F = V[:, -1].reshape(3, 3)
        # Enforce rank 2 constraint on the Fundamental Matrix
        U, S, VT = np.linalg.svd(F)
        S[-1] = 0
        F = U @ np.diag(S) @ VT
        F = F.reshape(3, 3, 1)

    return F


def estimate_epipole(F: np.ndarray) -> np.ndarray:
    """
    Estimate the epipole from the Fundamental Matrix.

    Parameters
    ----------
    F : np.ndarray
        The Fundamental Matrix.

    Returns
    -------
    np.ndarray
        The epipole.
    """
    # Perform Singular Value Decomposition (SVD) on matrix F
    _, _, V = np.linalg.svd(F)

    # Extract the last row of V (corresponding to the smallest singular value)
    e = V[-1, :]

    # Normalize the vector e by dividing by its last element
    e /= e[-1]

    # Return the normalized vector e
    return e


def plot_epipolar_lines(
    F: np.ndarray,
    points1: np.ndarray,
    points2: np.ndarray,
    img1: np.ndarray,
    img2: np.ndarray,
) -> None:
    """
    Plot the epipolar lines on the images.

    Parameters
    ----------
    F : np.ndarray
        The Fundamental Matrix.
    points1 : np.ndarray
        The points from the first image in the shape of (n, 2).
    points2 : np.ndarray
        The points from the second image in the shape of (n, 2).
    img1 : np.ndarray
        The first image.
    img2 : np.ndarray
        The second image.
    """
    # Estimate the epipoles for both images
    e1 = estimate_epipole(F)
    e2 = estimate_epipole(F.T)

    # Draw epipolar lines on the first image
    for pt in points1:
        # Draw a line from the point to the epipole
        cv2.line(img1, pt.astype(np.int32), e1[:-1].astype(np.int32), (0, 0, 0), 2)

    # Display the first image with epipolar lines
    plt.imshow(img1)
    plt.axis("off")
    plt.title("Epipolar Lines on Image 1")
    plt.show()

    # Draw epipolar lines on the second image
    for pt in points2:
        # Draw a line from the point to the epipole
        cv2.line(img2, pt.astype(np.int32), e2[:-1].astype(np.int32), (0, 0, 0), 2)

    # Display the second image with epipolar lines
    plt.imshow(img2)
    plt.axis("off")
    plt.title("Epipolar Lines on Image 2")
    plt.show()
