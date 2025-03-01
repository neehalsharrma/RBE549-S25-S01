import numpy as np


def linearTriangulation(K, R1, C1, R2, C2, points1, points2):
    """
    Linear Triangulation to estimate the 3D points.
    @ K: The intrinsic camera matrix in the shape of (3, 3)
    @ R1: The rotation matrix of the first camera in the shape of (3, 3)
    @ C1: The center of the first camera in the shape of (3, 1)
    @ R2: The rotation matrix of the second camera in the shape of (3, 3)
    @ C2: The center of the second camera in the shape of (3, 1)
    @ points1: The 2D points from the first image in the shape of (n, 2)
    @ points2: The 2D points from the second image in the shape of (n, 2)

    @ return: The estimated 3D points in the shape of (n, 4)
    """
    # Create the pose matrices for the cameras
    P1 = K @ R1 @ np.hstack((np.eye(3), -C1))
    P2 = K @ R2 @ np.hstack((np.eye(3), -C2))

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
        # From Page 312 of Hartley and Zisserman
        A = np.array([[x1 * p1_3 - p1_1],
                      [y1 * p1_3 - p1_2],
                      [x2 * p2_3 - p2_1],
                      [y2 * p2_3 - p2_2]]).reshape(4, 4)
        _, _, VT = np.linalg.svd(A)
        V = VT.T
        X = V[:, -1]
        X = X / X[3]
        points3D.append(X)

    return np.array(points3D).reshape(-1, 4)


def linearTriangulation2(K, R1, C1, R2, C2, points1, points2):
    """
    Linear Triangulation to estimate the 3D points.
    @ K: The intrinsic camera matrix in the shape of (3, 3)
    @ R1: The rotation matrix of the first camera in the shape of (3, 3)
    @ C1: The center of the first camera in the shape of (3, 1)
    @ R2: The rotation matrix of the second camera in the shape of (3, 3)
    @ C2: The center of the second camera in the shape of (3, 1)
    @ points1: The 2D points from the first image in the shape of (n, 2)
    @ points2: The 2D points from the second image in the shape of (n, 2)

    @ return: The estimated 3D points in the shape of (n, 4)
    """

    def skew_matrix(x):
        """
        takes a (3, 1) vector and return the skew-symmetric 3 x 3 matrix
        """
        return np.array([[0, -x[2], x[1]],
                         [x[2], 0, -x[0]],
                         [-x[1], x[0], 0]])

    # Create the pose matrices for the cameras
    P1 = K @ R1 @ np.hstack((np.eye(3), -C1))
    P2 = K @ R2 @ np.hstack((np.eye(3), -C2))
    points1 = np.hstack((points1, np.ones((points1.shape[0], 1))))
    points2 = np.hstack((points2, np.ones((points2.shape[0], 1))))

    points3D = []
    for i in range(points1.shape[0]):
        p1 = skew_matrix(points1[i])
        p2 = skew_matrix(points2[i])
        A = np.vstack((p1 @ P1, p2 @ P2))
        _, D, V = np.linalg.svd(A)
        X = V[-1, :]
        X = X / X[3]
        points3D.append(X)

    return np.array(points3D).reshape(-1,4)
