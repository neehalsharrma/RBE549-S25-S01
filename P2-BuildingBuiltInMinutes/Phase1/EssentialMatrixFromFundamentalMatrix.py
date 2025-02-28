import numpy as np
from LoadData import loadCalibrationMatrix
import os


def estimateE(F, K) -> np.ndarray:
    E = K.T @ F @ K

    # Enforcing Rank 2
    U, S, VT = np.linalg.svd(E)
    S[-1] = 0
    S = np.diag(S)
    E = U @ S @ VT
    return E
