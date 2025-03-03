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
    singular_values = [1, 1, 0]  # Set the singular values to [1, 1, 0]
    singular_values = np.diag(singular_values)
    E = U @ singular_values @ VT  # Recompute the Essential Matrix with enforced rank 2
    return E
