import numpy as np
import cv2
import matplotlib.pyplot as plt


def normalization_matrix(points):
    """
    Compute the similarity transformation (normalization matrix) that
    translates the centroid of the points to the origin and scales the RMS distance to sqrt(2).
    @ points: (n x 2) array of points.
    @ return: 3x3 normalization matrix.
    """
    centroid = np.mean(points, axis=0)  # Compute centroid (x̄, ȳ)

    # Compute the RMS distance from the centroid
    rms_dist = np.sqrt(np.mean(np.sum((points - centroid) ** 2, axis=1)))
    scale = np.sqrt(2) / rms_dist
    # Normalization matrix
    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1]
    ])

    return T


def getEquation(Point1, Point2):
    x1, y1 = Point1
    x2, y2 = Point2
    return np.array([x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1])


# Assuming Points 1 to be np array of shape (n, 2) and Points 2 to be np array of shape (n, 2)
# Returns the estimated Fundamental Matrix
def estimateF(points1, points2):
    """
    Estimate the Fundamental Matrix from the given points.
    @ Points1: The points from the first image in the shape of (n, 2)
    @ Points2: The points from the second image.in the shape of (n, 2)
    @ return: The estimated Fundamental Matrix.
    """
    n = points1.shape[0]
    A = np.zeros((n, 9))
    T1 = normalization_matrix(points1)
    T2 = normalization_matrix(points2)
    points1 = np.append(points1, np.ones((n, 1)), axis=1)  # n x 3
    points2 = np.append(points2, np.ones((n, 1)), axis=1)  # n x 3
    points1 = (T1 @ points1.T).T  # n x 3
    points2 = (T2 @ points2.T).T  # n x 3
    for i in range(n):
        A[i] = getEquation(points1[i, :2], points2[i, :2])
    U, S, VT = np.linalg.svd(A)
    V = VT.T
    F = V[:, -1].reshape(3, 3)  # Last row of V is the row of F

    # Enforcing Rank 2
    U, S, VT = np.linalg.svd(F)
    S[-1] = 0
    F = U @ np.diag(S) @ VT
    F = T2.T @ F @ T1
    return F

def estimate_epipole(F):
    '''
       Inputs:
           Fundamental Matrix F
       Outputs:
           Epipole

       Since epilines should pass through the epipole estimate epipole using the formula: F @ e = 0
       '''
    _, _, V = np.linalg.svd(F)
    e = V[-1, :]
    e /= e[-1]
    return e

def plot_epipolar_lines(F, points1, points2, img1, img2):
    """
    Plot the epipolar lines on the images.
    @ F: The Fundamental Matrix.
    @ points1: The points from the first image in the shape of (n, 2)
    @ points2: The points from the second image in the shape of (n, 2)
    @ img1: The first image.
    @ img2: The second image.
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
