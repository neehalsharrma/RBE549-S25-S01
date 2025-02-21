import numpy as np

def getEquation(Point1, Point2):
    x1, y1 = Point1
    x2, y2 = Point2
    return np.array([x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1])

# Assuming Points 1 to be np array of shape (n, 2) and Points 2 to be np array of shape (n, 2)
# Returns the estimated Fundamental Matrix
def estimateF(Points1, Points2):
    """
    Estimate the Fundamental Matrix from the given points.
    @ Points1: The points from the first image in the shape of (n, 2)
    @ Points2: The points from the second image.in the shape of (n, 2)
    @ return: The estimated Fundamental Matrix.
    """
    n = Points1.shape[0]
    A = np.zeros((n, 9))
    for i in range(n):
        A[i] = getEquation(Points1[i], Points2[i])
    U, S, VT = np.linalg.svd(A)
    # V= VT.T

    # Check this, I am not sure about this (If it should be this or V[-1, :])
    F = VT[:,-1].reshape(3, 3) # Last row of V is the row of F
    F[2,2] = 0
    # Enforcing Rank 2
    # U, S, VT = np.linalg.svd(F)
    # S[-1, -1] = 0
    # F = U @ np.diag(S) @ VT
    return F