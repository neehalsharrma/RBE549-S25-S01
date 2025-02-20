import numpy as np



def estimateE(F, K):
    E= K.T @ F @ K

    # Enforcing Rank 2
    U, S, VT = np.linalg.svd(E)
    S[-1,-1] = 0
    E= U @ S @ VT
    return E