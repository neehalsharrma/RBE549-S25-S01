import numpy as np


# Was going to call LinearTriangualtion multiple times to create x_set
# Linear Triangulation with (C_out[i], [0,0,0], R_out[i], np.eye(3)) sort of thing
# But we can also directly start over and call linear triangulation functions here instead like in this method
def getCorrectPose(C_out, R_out, X_set):
    """
    Get the correct pose from the set of poses.
    @ C_out: The camera centers in the shape of (4, 3)
    @ R_out: The rotation matrices in the shape of (4, 3, 3)
    @ X_set: The set of 3D points in the shape of (n, 3)
    @ return: The correct camera center and rotation matrix based on the number of points in front of the camera.
            --> Shape of C_out[i] = (1, 3) and R_out[i] = (3, 3)
    """
    best_count = 0
    index = 0
    for i in range(len(C_out)):
        count = 0
        C = C_out[i, :].T  # shape (3, 1)
        r3 = R_out[i, -1, :].reshape(1, 3)  # shape (1, 3)
        for j in range(X_set.shape[0]):
            x = X_set[j, :].T  # shape (3, 1)
            if np.dot(r3.T, (x - C)) > 0:
                count += 1

        if count > best_count:
            best_count = count
            index = i

    return C_out[index, :], R_out[index, :, :]
