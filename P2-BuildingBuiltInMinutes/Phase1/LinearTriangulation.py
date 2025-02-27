import numpy as np

def linearTriangulation(K1, K2, R1, R2, C1, C2, points1, points2):
    R1= np.concatenate((R1,C1), axis=1)
    R1= np.concatenate((R2,C2), axis=1)
    P1= K1 @ R1
    P2= K2 @ R2

    p11= P1[1:].reshape(1, 4)
    p12= P1[2:].reshape(1, 4)
    p13= P1[3:].reshape(1, 4)

    p21= P2[1:].reshape(1, 4)
    p22= P2[2:].reshape(1, 4)
    p23= P2[3:].reshape(1, 4)

    points3D= []
    for i in range(points1.shape[0]):
        x1= points1[i, 0]
        y1= points1[i, 1]
        x2= points2[i, 0]
        y2= points2[i, 1]

        A= np.array([[x1*p13 - p11], [y1*p13 - p12], [x2*p23 - p21], [y2*p23 - p22]])
        _, _, V_T= np.linalg.svd(A)
        V= V_T.T
        X= V[:, -1]
        X= X/X[3]
        points3D.append(X)

    return np.array(points3D)

