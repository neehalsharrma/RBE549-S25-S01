import numpy as np
from LoadData import loadCalibrationMatrix
import os


def estimateE(F, K, calibration_path: str = '../P2Data/') -> np.ndarray:
    K = loadCalibrationMatrix(os.path.join(calibration_path, 'calibration.txt'))
    E = K.T @ F @ K

    # Enforcing Rank 2
    U, S, VT = np.linalg.svd(E)
    S[-1, -1] = 0
    E = U @ S @ VT
    return E
