"""
Module for estimating the Essential Matrix from the Fundamental Matrix.

This module provides a function to estimate the Essential Matrix given the 
Fundamental Matrix and the Calibration Matrix using the relationship between 
them. The Essential Matrix is then enforced to have rank 2.
"""

import numpy as np


def estimate_E(F, K) -> np.ndarray:
    """
    Estimate the Essential Matrix from the Fundamental Matrix.

    Parameters
    ----------
    F : ndarray
        The Fundamental Matrix as a 3x3 numpy array.
    K : ndarray
        The Calibration Matrix as a 3x3 numpy array.

    Returns
    -------
    E : ndarray
        The Essential Matrix as a 3x3 numpy array.

    Notes
    -----
    The Essential Matrix is computed using the formula E = K.T @ F @ K.
    After computation, the matrix is enforced to have rank 2 by setting the
    smallest singular value to zero.
    """
    # Compute the initial Essential Matrix
    E = K.T @ F @ K

    # Enforcing Rank 2 by setting the smallest singular value to zero
    U, S, VT = np.linalg.svd(E)
    S = [1, 1, 0]
    E = U @ np.diag(S) @ VT

    return E
