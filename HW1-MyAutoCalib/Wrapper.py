"""
This module provides functions to perform automatic camera calibration.

Functions:
    find_image_points(images, pattern_size)
    calibrate_camera(images, pattern_size)
    save_calibration_parameters(filename, mtx, dist, rvecs, tvecs)
"""

import argparse
import glob
from typing import List, Tuple

import cv2
import numpy as np


def find_image_points(
    images: List[str], pattern_size: Tuple[int, int]
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Find image points for camera calibration.

    Args:
        images (List[str]): List of image file paths.
        pattern_size (Tuple[int, int]): Size of the chessboard pattern.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]: Object points and image points.
    """
    obj_points = []
    img_points = []
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2)

    for image in images:
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret:
            obj_points.append(objp)
            img_points.append(corners)

    return obj_points, img_points


def calibrate_camera(
    images: List[str], pattern_size: Tuple[int, int]
) -> Tuple[float, np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """
    Calibrate the camera using the provided images and pattern size.

    Args:
        images (List[str]): List of image file paths.
        pattern_size (Tuple[int, int]): Size of the chessboard pattern.

    Returns:
        Tuple[float, np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]: Calibration parameters.
    """
    obj_points, img_points = find_image_points(images, pattern_size)
    if not obj_points or not img_points:
        raise ValueError(
            "No corners found in images. Check your pattern size and images."
        )

    # Read the first image to get the image size
    img = cv2.imread(images[0])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Perform camera calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, gray.shape[::-1], None, None
    )
    return ret, mtx, dist, rvecs, tvecs


def save_calibration_parameters(
    filename: str,
    mtx: np.ndarray,
    dist: np.ndarray,
    rvecs: List[np.ndarray],
    tvecs: List[np.ndarray],
) -> None:
    """
    Save the calibration parameters to a file.

    Args:
        filename (str): The file to save the parameters to.
        mtx (np.ndarray): Camera matrix.
        dist (np.ndarray): Distortion coefficients.
        rvecs (List[np.ndarray]): Rotation vectors.
        tvecs (List[np.ndarray]): Translation vectors.
    """
    np.savez(filename, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automatic Camera Calibration")
    parser.add_argument(
        "image_path", 
        type=str, 
        nargs='?', 
        default="./Data", 
        help="Path to calibration images"
    )
    parser.add_argument(
        "pattern_size", 
        type=str, 
        nargs='?', 
        default="9,6", 
        help="Chessboard pattern size as 'rows,cols'"
    )
    args = parser.parse_args()

    # Get the list of calibration images
    images = glob.glob(f"{args.image_path}/*.jpg")
    
    # Parse the chessboard pattern size
    pattern_size = tuple(map(int, args.pattern_size.split(',')))
    
    # Calibrate the camera
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(images, pattern_size)
    
    # Save the calibration parameters
    save_calibration_parameters("calibration_parameters.npz", mtx, dist, rvecs, tvecs)
    
    print("Calibration successful. Parameters saved to calibration_parameters.npz")
