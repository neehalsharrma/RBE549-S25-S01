"""
This module provides a function to estimate the Essential Matrix from the Fundamental Matrix.

Functions
---------
estimateE(F: np.ndarray, K: np.ndarray) -> np.ndarray
    Estimate the Essential Matrix from the Fundamental Matrix.
"""

import numpy as np


def estimateE(F: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Estimate the Essential Matrix from the Fundamental Matrix.

    Parameters
    ----------
    F : np.ndarray
        The Fundamental Matrix as 3x3.
    K : np.ndarray
        The Calibration Matrix as 3x3.

    Returns
    -------
    np.ndarray
        The Essential Matrix as 3x3.
    """
    # Compute the Essential Matrix
    E = K.T @ F @ K

    # Enforcing Rank 2
    U, S, VT = np.linalg.svd(E)
    S=[1,1,0]
    E = U @ np.diag(S) @ VT
    return E
