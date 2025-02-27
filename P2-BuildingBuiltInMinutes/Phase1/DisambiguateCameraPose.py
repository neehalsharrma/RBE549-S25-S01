import numpy as np


# Was going to call LinearTriangualtion multiple times to create x_set
# Linear Triangulation with (C_out[i], [0,0,0], R_out[i], np.eye(3)) sort of thing
# But we can also directly start over and call linear triangulation functions here instead like in this method
def getCorrectPose(C_out, R_out, X_set):
    count_points=[]
    for i in range(len(C_out)):
        count=0
        C= C_out[i]
        r3= R_out[i, 2: ].reshape(1,3)
        

        for j in range(X_set[i].shape[0]):
            x= X_set[i][j, :]
            if np.dot(r3.T, (x-C))>0:
                count+=1
