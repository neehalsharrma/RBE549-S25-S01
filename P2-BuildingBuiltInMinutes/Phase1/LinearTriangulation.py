"""
Module for performing linear triangulation to estimate 3D points from 2D correspondences
in two images taken from different camera positions.

Functions
---------
linearTriangulation(K, R1, C1, R2, C2, points1, points2)
    Estimates 3D points using linear triangulation.
see_triangulation(Points)
    Visualizes the triangulated 3D points by plotting the X and Z coordinates.
linearTriangulation2(K, R1, C1, R2, C2, points1, points2)
    Estimates 3D points using linear triangulation with skew-symmetric matrices.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.dont_write_bytecode = True


def linearTriangulation(K, R1, C1, R2, C2, points1, points2):
    """
    Linear Triangulation to estimate the 3D points.

    Parameters
    ----------
    K : array_like
        The intrinsic camera matrix in the shape of (3, 3).
    R1 : array_like
        The rotation matrix of the first camera in the shape of (3, 3).
    C1 : array_like
        The center of the first camera in the shape of (3, 1).
    R2 : array_like
        The rotation matrix of the second camera in the shape of (3, 3).
    C2 : array_like
        The center of the second camera in the shape of (3, 1).
    points1 : array_like
        The 2D points from the first image in the shape of (n, 2).
    points2 : array_like
        The 2D points from the second image in the shape of (n, 2).

    Returns
    -------
    array_like
        The estimated 3D points in the shape of (n, 4).
    """
    # Create the pose matrices for the cameras
    P1 = K @ R1 @ np.hstack((np.eye(3), -C1))
    P2 = K @ R2 @ np.hstack((np.eye(3), -C2))

    # Extract rows of the projection matrices
    p1_1 = P1[0, :].reshape(1, 4)
    p1_2 = P1[1, :].reshape(1, 4)
    p1_3 = P1[2, :].reshape(1, 4)

    p2_1 = P2[0, :].reshape(1, 4)
    p2_2 = P2[1, :].reshape(1, 4)
    p2_3 = P2[2, :].reshape(1, 4)

    points3D = []
    for i in range(points1.shape[0]):
        x1 = points1[i, 0]
        y1 = points1[i, 1]
        x2 = points2[i, 0]
        y2 = points2[i, 1]
        
        # Form the matrix A using the equations from Hartley and Zisserman
        A = np.array(
            [
                [x1 * p1_3 - p1_1],
                [y1 * p1_3 - p1_2],
                [x2 * p2_3 - p2_1],
                [y2 * p2_3 - p2_2],
            ]
        )
        A = np.array(A).reshape(4, 4)

        # Perform SVD and extract the 3D point
        _, _, VT = np.linalg.svd(A)
        V = VT.T
        X = V[:, -1]
        X = X / X[3]  # Normalize the point
        points3D.append(X)

    return np.array(points3D).reshape(-1, 4)


def see_triangulation(Points):
    """
    Visualize the triangulated 3D points by plotting the X and Z coordinates.

    Parameters
    ----------
    Points : array_like
        The 3D points in the shape of (n, 4).
    """
    # Extract the X and Z coordinates from the 3D points
    X = Points[:, 0].reshape(-1, 1)
    Z = Points[:, 2]

    # Print the 3D points and their shapes for debugging purposes
    print(Points)
    print(Points.shape)
    print(X.shape)

    # Create a new figure for the plot
    plt.figure(figsize=(8, 6))

    # Scatter plot of the X and Z coordinates
    plt.scatter(X, Z, color="b", label="3D Points (Projected X-Z)")

    # Set the labels and title of the plot
    plt.xlabel("X (Horizontal)")
    plt.ylabel("Z (Depth)")
    plt.title("Plot of Points by Z (Depth) and X (Horizontal)")

    # Add horizontal and vertical lines at the origin
    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)

    # Add a grid to the plot
    plt.grid(True, linestyle="--", alpha=0.6)

    # Add a legend to the plot
    plt.legend()

    # Display the plot
    plt.show()

    # Save the plot to a file
    plt.savefig('Data/IntermediateOutputImages/triangulation_plot.png')


def linearTriangulation2(K, R1, C1, R2, C2, points1, points2):
    """
    Linear Triangulation to estimate the 3D points using skew-symmetric matrices.

    Parameters
    ----------
    K : array_like
        The intrinsic camera matrix in the shape of (3, 3).
    R1 : array_like
        The rotation matrix of the first camera in the shape of (3, 3).
    C1 : array_like
        The center of the first camera in the shape of (3, 1).
    R2 : array_like
        The rotation matrix of the second camera in the shape of (3, 3).
    C2 : array_like
        The center of the second camera in the shape of (3, 1).
    points1 : array_like
        The 2D points from the first image in the shape of (n, 2).
    points2 : array_like
        The 2D points from the second image in the shape of (n, 2).

    Returns
    -------
    array_like
        The estimated 3D points in the shape of (n, 4).
    """

    def skew_matrix(x):
        """
        Create a skew-symmetric matrix from a 3-element vector.

        Parameters
        ----------
        x : array_like
            A 3-element vector.

        Returns
        -------
        array_like
            A 3x3 skew-symmetric matrix.
        """
        return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])

    # Create the pose matrices for the cameras
    P1 = K @ R1 @ np.hstack((np.eye(3), -C1))
    P2 = K @ R2 @ np.hstack((np.eye(3), -C2))

    # Add a column of ones to the 2D points to convert them to homogeneous coordinates
    points1 = np.hstack((points1, np.ones((points1.shape[0], 1))))
    points2 = np.hstack((points2, np.ones((points2.shape[0], 1))))

    points3D = []
    for i in range(points1.shape[0]):
        # Create skew-symmetric matrices for the points
        p1 = skew_matrix(points1[i])
        p2 = skew_matrix(points2[i])

        # Form the matrix A using the skew-symmetric matrices and pose matrices
        A = np.vstack((p1 @ P1, p2 @ P2))

        # Perform SVD and extract the 3D point
        _, D, V = np.linalg.svd(A)
        X = V[-1, :]
        X = X / X[3]  # Normalize the point
        points3D.append(X)

    return np.array(points3D).reshape(-1, 4)
