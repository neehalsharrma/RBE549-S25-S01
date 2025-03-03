import numpy as np
import scipy


def calc_loss(x: np.array, P: np.array, X: np.array) -> float:
    """
    Calculate the loss for the non-linear triangulation.
    @ x: The 2D points from the image in the shape of (n, 3) since it's homogenized
    @ P: The projection matrix in the shape of (3, 4)
    @ X: The 3D points in the shape of (n, 3)
    @ return The loss for the non-linear triangulation, the shape is a (1, 3) vector.
    """
    x_hat = P @ X.T
    x_hat = x_hat / x_hat[2]  # divide by the last row of P.T @ X
    error = x - x_hat
    return np.power(error, 2)
    # return np.linalg.norm(error)


def loss_func(linear_X: np.array, x1: np.array, x2: np.array, P1: np.array, P2: np.array):
    """
    Perform non-linear triangulation to estimate the 3D points.
    @ linear_X: The linear estimate of the 3D points in the shape of (1, 4)
    @ x1: The 2D points from the first image in the shape of (1, 3)
    @ x2: The 2D points from the second image in the shape of (1, 3)
    @ P1: The projection matrix for the first camera. This is assumed to be P = [I | 0] --> 3x4 matrix.
    @ P2: The projection matrix for the second camera calculated from the Essential Matrix. --> 3x4 matrix.
    """
    error1 = calc_loss(x1, P1, linear_X)
    error2 = calc_loss(x2, P2, linear_X)
    return np.concatenate((error1, error2)).flatten()

# Returns 3d Points homogenized points
def non_linear_triangulation(K: np.array, R1: np.array, C1: np.array, R2: np.array, C2: np.array, x1: np.array,
                             x2: np.array, linear_X: np.array) -> tuple[np.array, list[float]]:
    """
    Perform non-linear triangulation to estimate the 3D points.
    We assume that the pose of camera one is [I | 0] as a 3x4 matrix.
    The pose of camera two is [R | t] as a 3x4 matrix.

    @ K: The intrinsic camera matrix in the shape of (3, 3)
    @ C1: The center of the first camera in the shape of (3, 1)
    @ R1: The rotation matrix of the first camera in the shape of (3, 3)
    @ C2: The center of the second camera in the shape of (3, 1)
    @ R2: The rotation matrix of the second camera in the shape of (3, 3)
    @ x1: The 2D points from the first image in the shape of (n, 2)
    @ x2: The 2D points from the second image in the shape of (n, 2)
    @ linear_X: The linear estimate of the 3D points in the shape of (n, 4)
    @ return: The refined estimated 3D points in the shape of (n, 4)
    """
    # Create the pose matrices for the cameras
    P1 = K @ R1 @ np.hstack((np.eye(3), -C1))  # 3x4 matrix for camera 1 pose
    P2 = K @ R2 @ np.hstack((np.eye(3), -C2))  # 3x4 matrix for camera 2 pose

    num_features = x1.shape[0]
    refined_X = []
    costs = []

    for i in range(num_features):
        point1 = x1[i, :]
        point2 = x2[i, :]
        x0 = linear_X[i, :]
        optimized = scipy.optimize.least_squares(loss_func, x0, args=(point1, point2, P1, P2), method='lm')

        refined_X.append(optimized.x)
        costs.append(optimized.cost)
        if i % 100 == 0:
            print(f"Processed {i} points")
    refined_X = np.array(refined_X).reshape(num_features, 4)
    refined_X = refined_X / refined_X[:, 3].reshape(num_features, 1)
    refined_X = refined_X.reshape(num_features, 4)
    return refined_X, costs
