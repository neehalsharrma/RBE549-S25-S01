"""
Utilities Module
================

This module provides utility functions for various tasks such as timing,
value remapping, file handling, and data processing. These functions are
designed to be general-purpose and reusable across different parts of the
project.

Functions
---------
- tic() : Start a timer and return the current time.
- toc(start_time) : Calculate elapsed time since the timer started.
- remap(value, output_min, output_max, input_min, input_max) : Remap a value
  from one range to another while maintaining ratios.
- find_latest_model(checkpoint_path) : Find the latest model checkpoint file
  in a directory.
- convert_to_one_hot(vector, num_labels) : Convert a vector of integers to a
  one-hot encoded matrix.
"""

import glob
import os
import sys
import time

import numpy as np
from typing import Union, Optional

# Prevent Python from generating .pyc files
sys.dont_write_bytecode = True


def tic() -> float:
    """
    Start a timer.

    Returns
    -------
    float
        The current time in seconds since the epoch.
    """
    start_time = time.time()
    return start_time


def toc(start_time: float) -> float:
    """
    Stop the timer and calculate elapsed time.

    Parameters
    ----------
    start_time : float
        The start time returned by `tic`.

    Returns
    -------
    float
        The elapsed time in seconds.
    """
    return time.time() - start_time


def remap(
    value: Union[float, np.ndarray],
    output_min: float,
    output_max: float,
    input_min: float,
    input_max: float,
) -> Optional[Union[float, np.ndarray]]:
    """
    Remap a value from one range to another while maintaining ratios.

    Parameters
    ----------
    value : float or array-like
        The input value(s) to be remapped.
    output_min : float
        The minimum value of the output range.
    output_max : float
        The maximum value of the output range.
    input_min : float
        The minimum value of the input range.
    input_max : float
        The maximum value of the input range.

    Returns
    -------
    float or array-like
        The remapped value(s). Returns None if either input or output range has zero length.

    Notes
    -----
    Uses numpy's interp function for the remapping.
    """
    # Check for zero range in input or output
    if output_min == output_max or input_min == input_max:
        print("Warning: Zero range in input or output")
        return None

    # Use numpy's interp function
    return np.interp(value, [input_min, input_max], [output_min, output_max])


def find_latest_model(checkpoint_path: str) -> Optional[str]:
    """
    Find the latest model checkpoint file in a directory.

    Parameters
    ----------
    checkpoint_path : str
        The directory path containing checkpoint files.

    Returns
    -------
    str
        The name of the latest checkpoint file (without path and extension).
    """
    # Get a list of all checkpoint files
    file_list = glob.glob(os.path.join(checkpoint_path, "*.ckpt"))
    if not file_list:
        print("Warning: No checkpoint files found")
        return None

    # Find the most recently created file
    latest_file = max(file_list, key=os.path.getctime)

    # Extract the file name without path and extension
    latest_file = os.path.basename(latest_file).replace(".ckpt", "")
    return latest_file


def convert_to_one_hot(vector: Union[np.ndarray, list], num_labels: int) -> np.ndarray:
    """
    Convert a vector of integers to a one-hot encoded matrix.

    Parameters
    ----------
    vector : array-like
        The input vector of integers.
    num_labels : int
        The number of labels (columns in the one-hot matrix).

    Returns
    -------
    numpy.ndarray
        A one-hot encoded matrix.
    """
    return np.equal.outer(vector, np.arange(num_labels)).astype(np.float_)
