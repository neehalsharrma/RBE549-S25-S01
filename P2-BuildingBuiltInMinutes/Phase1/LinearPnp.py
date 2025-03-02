import numpy as np

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
    return np.linalg.norm(error)

def get_inliers(x, X, P, threshold):
    points2d=[]
    points3d=[]
    for i in range(x.shape[0]):
        if calc_loss(x[i], P, X[i]) < threshold :   
            points2d.append(x[i])
            points3d.append(X[i])
    return points2d, points3d

def calc_inliers(x, X, P, threshold):
    tot=0
    for i in range(x.shape[0]):
        if calc_loss(x[i], P, X[i]) < threshold :   
            tot+=1
    return tot

def get_equation(point3D, point2D):
    X, Y, Z, _= point3D
    x,y, _ = point2D
    return np.array([[X, Y, Z, 1, 0, 0, 0, 0, -x*X, -x*Y, -x*Z, -x],
                     [0, 0, 0, 0, X, Y, Z, 1, -y*X, -y*Y, -y*Z, -y]])


def linear_PnP(K, points2D, points3D):
    """
    Linear PnP to estimate the 3D points.
    @ K: The intrinsic camera matrix in the shape of (3, 3)

    @ R: The rotation matrix of the camera in the shape of (3, 3)
    @ C: The center of the camera in the shape of (3, 1)
    @ points2D: The 2D points from the image in the shape of (n, 2)
    @ points3D: The 3D points in the shape of (n, 3)


    """
    # Solve linear least squares
    # AP=0

    for i in range(points3D.shape[0]):
        a= get_equation(points3D[i], points2D[i])
        if i>0:
            A= np.vstack((A, a))
        else:
            A=a
    
    U, S, VT= np.linalg.svd(A)
    P=VT.T[-1].reshape((3,4))
    R=P[:,:3]
    C=P[:,-1]

    # Reset R
    U, D, VT= np.linalg.svd(R)
    R= U @ VT
    if np.linalg.det(R)==-1:
        R=-R
    
    return R,C


def PnPRansac(K,points2D, points3D, threshold= float(5), acc_thresh=0.85):
    rng = np.random.default_rng()
    best_percent=0

    tot_size= points2D.shape[0]
    while best_percent < acc_thresh:
        random_samples= rng.integers(0, tot_size, size= 6, replace=False)
        point2D= points2D[random_samples]
        point3D = points3D[random_samples]
        R,C= linear_PnP(K, point2D, point3D)
        P= np.hstack(R,C)
        num_inliers= calc_inliers(points2D, points3D, P, threshold)

        percent_match = num_inliers / tot_size
        # if a better match is found, update the best match
        if percent_match > best_percent:
            best_percent = percent_match
            best_points2D, best_points3D = get_inliers(points2D, points3D, P, threshold)
            print(f"Best Percent: {best_percent}")

    print(f'Original No. of features: {tot_size}')
    print(f"No. of inliers: {best_points2D.shape[0]}")
    best_R,best_C= linear_PnP(K, point2D, point3D)

    return best_R, best_C



