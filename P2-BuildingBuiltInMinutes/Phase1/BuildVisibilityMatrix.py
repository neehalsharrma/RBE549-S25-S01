import numpy as np

def buildVisibilityMatrix(C_matrices, R_matrices, K, world_X_points, img_x_points) ->np.array:
    """
    @ C_matrices: The camera centers in the shape of list[(n, 3)] of length num_imgs
    @ R_matrices: The rotation matrices in the shape of list[(3, 3)] of length num_imgs
    @ K: The intrinsic camera matrix in the shape of (3, 3)
    @ world_X_points: The homogenized 3D points for the world coordinate system in the shape of (n, 4)
    @ img_x_points: The homogenized 2D points from each image in the shape of list[(n, 3)] of length num_imgs

    @ return: The binary mask visibility matrix in the shape of (num_imgs, n) where Vij
               is one if the jth point is visible from the ith camera and zero otherwise
    """
    num_imgs = len(C_matrices)
    n = world_X_points.shape[0]
    visibility_matrix = np.zeros((num_imgs, n))
    for i in range(num_imgs):
        C = C_matrices[i]
        R = R_matrices[i]
        P = K @ np.hstack((R, -R @ C))
        # Calculate the reprojection error for each point
        x_hat = P @ world_X_points.T
        x_hat = x_hat / x_hat[2]  # divide by the last row of P.T @ X
        diffs = img_x_points[i] - x_hat.T  # Shape: (n, 3)
        # Calculate the Euclidean distance of the reprojection error
        errors = np.linalg.norm(diffs, axis=1)  # Shape: (n,)
        visibility_matrix[i] = (errors < 0.1).astype(int)
    return visibility_matrix