import numpy as np
from LoadData import loadCalibrationMatrix


def estimateE(F, K, calibration_path:str = '../P2Data/') -> np.ndarray:
    E= K.T @ F @ K

    # Enforcing Rank 2
    U, S, VT = np.linalg.svd(E)
    S[-1,-1] = 0
    E= U @ S @ VT
    return E