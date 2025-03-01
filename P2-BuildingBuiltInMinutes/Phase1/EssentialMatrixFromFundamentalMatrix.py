import numpy as np

def estimateE(F, K) -> np.ndarray:
    """
    Estimate the Essential Matrix from the Fundamental Matrix.
    @ F: The Fundamental Matrix as 3x3.
    @ K: The Calibration Matrix as 3x3.
    @ return: The Essential Matrix as 3x3.
    """
    E = K.T @ F @ K

    # Enforcing Rank 2
    U, S, VT = np.linalg.svd(E)
    S=[1,1,0]
    E = U @ np.diag(S) @ VT
    return E

