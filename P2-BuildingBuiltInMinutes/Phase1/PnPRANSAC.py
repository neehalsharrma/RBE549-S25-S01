import numpy as np


def linear_PnP(X, x, K):
    pass


def error_func(P: np.array, X: np.array, x: np.array, threshold: float) -> float:
    """
    Calculate the error function for the PnP problem.
    @ P: The projection matrix in the shape of (3, 4)
    @ X: The homogenized 3D points in the shape of (n, 4)
    @ x: The homogenized 2D points in the shape of (n, 3)
    @ return: The number of inlier
    """
    x_hat = P @ X.T
    x_hat = x_hat / x_hat[2]  # divide by the last row of P.T @ X
    diffs = x - x_hat  # Shape: (n, 3)
    # Calculate the Euclidean distance of the reprojection error
    errors = np.linalg.norm(diffs, axis=1)  # Shape: (n,)
    return np.sum(errors < threshold)


def get_inliers(P, X, x, threshold: float) -> tuple[np.array, np.array]:
    """
    Get the inliers given the threshold.
    @ P: The projection matrix in the shape of (3, 4)
    @ X: The homogenized 3D points in the shape of (n, 4)
    @ x: The homogenized 2D points in the shape of (n, 3)
    @ threshold: The threshold to be used.
    @ return: The inlier and outlier indices
    """
    x_hat = P @ X.T
    x_hat = x_hat / x_hat[2]  # divide by the last row of P.T @ X
    diffs = x - x_hat  # Shape: (n, 3)
    # Calculate the Euclidean distance of the reprojection error
    errors = np.linalg.norm(diffs, axis=1)  # Shape: (n,)
    inliers = np.argwhere(errors < threshold).squeeze()
    outliers = np.argwhere(errors >= threshold).squeeze()
    return inliers, outliers


def PNP_RANSAC(world_X, img_x, K, threshold=0.1, acc_thresh=0.85, max_iters=1000)->tuple[np.array, np.array, np.array, np.array]:
    """
    Perform PnP with RANSAC to estimate the camera pose.
    @ world_X: The homogenized  3D points in the shape of (n, 4)
    @ img_x: The homogenized 2D points in the shape of (n, 3)
    @ K: The intrinsic camera matrix in the shape of (3, 3)
    @ threshold: The threshold to determine if a point is an inlier or outlier
    @ acc_thresh: The threshold to determine if the current model is the best model
    @ max_iters: The maximum number of iterations to perform
    @ return: The best camera pose and rotation based on PnP estimation,
              and return the indices of the inliers and outliers
    """
    num_features = world_X.shape[0]
    best_acc = 0
    inliers = None
    outliers = None

    best_C = np.zeros((3, 1))
    best_R = np.eye(3)

    while best_acc < acc_thresh and max_iters > 0:
        # Randomly select 6 points
        idx = np.random.choice(num_features, 6, replace=False)
        x_sample = img_x[idx]
        X_sample = world_X[idx]
        C, R = linear_PnP(X_sample, x_sample, K)
        P = K @ np.hstack((R, -R @ C))
        num_inliers = error_func(P, world_X, img_x, threshold)
        acc = num_inliers / num_features
        if acc > best_acc:
            best_acc = acc
            best_C = C
            best_R = R
            inliers, outliers = get_inliers(P, world_X, img_x, threshold)
        max_iters -= 1
    return best_C, best_R, inliers, outliers
