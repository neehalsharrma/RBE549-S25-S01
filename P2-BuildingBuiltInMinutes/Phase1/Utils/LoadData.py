"""
This module provides utility functions for loading and displaying image data,
calibration matrices, and feature matches for Structure from Motion (SfM) tasks.

Functions
---------
load_image(img, data_path='P2Data/')
    Load an image from the specified path and return it as a NumPy array.
load_calibration_matrix(data_path='P2Data/')
    Load the calibration matrix from the specified path and return it as a NumPy array.
load_data_full(img, data_path='P2Data/', num_images=5)
    Load the full data for an image and return the number of matches and the data as a tuple.
load_data_sparse(img_num, data_path='P2Data/', num_images=5)
    Load the sparse data for an image and return it as a NumPy array.
load_correspondences(image1, image2, data_path='P2Data/', num_images=5)
    Load the correspondences between two images and return them as a NumPy array.
show_features(points, img1)
    Display the features on the specified image.
show_matches(image1, image2, data_path='P2Data/')
    Display the matches between two images.
show_matches2(image1, image2, points1, points2)
    Display the matches between two images using provided points.
"""

import sys
sys.dont_write_bytecode = True

from encodings import utf_8
import numpy as np
import cv2
import matplotlib.pyplot as plt


def load_image(img: int, data_path: str = "P2Data/") -> np.ndarray:
    """
    Load the image from the given path and return the image as a NumPy array.

    Parameters
    ----------
    img : int
        The image number to load.
    data_path : str, optional
        The path to the data directory (default is 'P2Data/').

    Returns
    -------
    np.ndarray
        The loaded image as a NumPy array.
    """
    # Construct the image file path
    img_path = data_path + str(img) + ".png"
    # Read the image using OpenCV
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    # Convert the image from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_calibration_matrix(data_path: str = "P2Data/") -> np.ndarray:
    """
    Load the calibration matrix from the given path and return the calibration matrix as a NumPy array.

    Parameters
    ----------
    data_path : str, optional
        The relative path to the data directory (default is 'P2Data/').

    Returns
    -------
    np.ndarray
        The calibration matrix as a NumPy array.
    """
    # Construct the calibration file path
    calibration_file = data_path + "calibration.txt"
    # Read the calibration file
    with open(calibration_file, "r", encoding="utf-8") as file:
        lines = file.readlines()
        # Initialize the calibration matrix
        K = np.zeros((3, 3), dtype=np.float32)
        # Populate the calibration matrix with values from the file
        for i, line in enumerate(lines):
            values = list(map(float, line.split()))
            K[i] = np.array(values)
    return K


def load_data_full(
    img: int, data_path: str = "P2Data/", num_images: int = 5
) -> tuple[int, np.ndarray]:
    """
    Load the data from the given path and return the data as a tuple.

    Parameters
    ----------
    img : int
        The image number to load.
    data_path : str, optional
        The path to the data directory (default is 'P2Data/').
    num_images : int, optional
        The number of images to be used in the SfM matching (default is 5).

    Returns
    -------
    tuple[int, np.ndarray]
        A tuple containing the number of matches and the data from the file.
    """
    # Construct the matching file path
    matching_file = data_path + "matching" + str(img) + ".txt"
    header_data = 6  # Number of header data columns
    # Read the matching file
    with open(matching_file, "r", encoding="utf-8") as file:
        lines = file.readlines()
        # Extract the number of features
        n_features = int(lines[0].split(":")[1].strip())
        num_lines = len(lines) - 1
        # Initialize the matches array
        matches = np.zeros(
            (num_lines, header_data + (num_images - 1) * 3), dtype=np.float32
        )
        # Populate the matches array with values from the file
        for i, line in enumerate(lines[1:]):
            values = list(map(float, line.split()))
            matches[i, : len(values)] = np.array(values)
    return n_features, matches


def load_data_sparse(
    img_num: int, data_path: str = "P2Data/", num_images: int = 5
) -> np.ndarray:
    """
    Load the data from the given path and return the data as a sparse matrix.

    Parameters
    ----------
    img_num : int
        The image number to load.
    data_path : str, optional
        The path to the data directory (default is 'P2Data/').
    num_images : int, optional
        The number of images to be used in the SfM matching (default is 5).

    Returns
    -------
    np.ndarray
        A sparse matrix of the data from the file.
    """
    # Construct the matching file path
    matching_file = data_path + "matching" + str(img_num) + ".txt"
    header_data = 6  # Number of header data columns
    # Read the matching file
    with open(matching_file, "r", encoding="utf-8") as file:
        lines = file.readlines()
        num_lines = len(lines) - 1
        # Initialize the matches array
        matches = np.zeros(
            (num_lines, header_data - 1 + (num_images - img_num) * 3), dtype=np.float32
        )
        # Populate the matches array with values from the file
        for i, line in enumerate(lines[1:]):
            values = list(map(float, line.split()))
            matches[i, : (header_data - 1)] = np.array(values[1:header_data])
            for j in range((len(values) - header_data) // 3):
                offset = header_data + j * 3
                img_id = int(values[offset]) - img_num - 1
                matches[
                    i, (header_data - 1) + img_id * 2 : (header_data + 1) + img_id * 2
                ] = np.array(values[offset + 1 : offset + 3])
    return matches


def load_correspondences(
    image1: int, image2: int, data_path: str = "P2Data/", num_images: int = 5
) -> np.ndarray:
    """
    Load the correspondences between two images from the given path and return the data as a NumPy array.

    Parameters
    ----------
    image1 : int
        The first image number for correspondences.
    image2 : int
        The second image number for correspondences.
    data_path : str, optional
        The path to the data directory (default is 'P2Data/').
    num_images : int, optional
        The number of images to be used in the SfM matching (default is 5).

    Returns
    -------
    np.ndarray
        The correspondences as a num_correspondences x 4 array.
    """
    # Load the full data for the first image
    _, matches = load_data_full(image1, data_path, num_images)
    header_data = 6  # Number of header data columns
    # Initialize the correspondences array
    correspondences = np.ndarray((0, 4), dtype=np.float32)
    # Populate the correspondences array with values from the matches
    for i in range(len(matches)):
        x1, y1 = matches[i, 4:6]
        num_matches = int(matches[i, 0])
        for j in range(num_matches - 1):
            img_id = int(matches[i, header_data + j * 3])
            if img_id != image2:
                continue
            x2, y2 = matches[i, (header_data + 1) + j * 3 : 9 + j * 3]
            correspondences = np.append(
                correspondences, np.array([[x1, y1, x2, y2]]), axis=0
            )
            break
    return correspondences


def show_features(points, img1):
    """
    Display the features on the image.

    Parameters
    ----------
    points : np.ndarray
        The points to display.
    img1 : np.ndarray
        The image on which to display the points.
    """
    img1_features = img1.copy()
    # Draw circles on the image at the feature points
    for i in range(points.shape[0]):
        y, x = points[i, 4:6]
        cv2.circle(img1_features, (int(x), int(y)), 5, (0, 255, 0), -1)
    # Display the image with features
    plt.figure(figsize=(10, 10))
    plt.imshow(img1_features)
    plt.axis("off")
    plt.show()


def show_matches(image1: int, image2: int, data_path: str = "P2Data/"):
    """
    Display the matches between two images.

    Parameters
    ----------
    image1 : int
        The first image number.
    image2 : int
        The second image number.
    data_path : str, optional
        The path to the data directory (default is 'P2Data/').
    """
    # Load the images
    img1 = load_image(image1)
    img2 = load_image(image2)
    # Concatenate the images side by side
    img = np.concatenate((img1, img2), axis=1)
    # Load the correspondences between the images
    correspondences = load_correspondences(image1, image2)
    # Draw circles and lines for the matches
    for i in range(correspondences.shape[0]):
        x1, y1, x2, y2 = correspondences[i]
        cv2.circle(img, (int(x1), int(y1)), 3, (0, 0, 255), 2)
        cv2.circle(img, (int(x2) + img1.shape[1], int(y2)), 3, (0, 0, 255), 2)
        cv2.line(
            img, (int(x1), int(y1)), (int(x2) + img1.shape[1], int(y2)), (127, 0, 0), 1
        )
    # Display the image with matches
    plt.figure(figsize=(10, 10))
    plt.title("Matches between Image " + str(image1) + " and Image " + str(image2))
    plt.imshow(img)
    plt.axis("off")
    plt.show()
    # Save the image with matches to the Outputs directory
    output_path = "Outputs/matches_" + str(image1) + "_" + str(image2) + ".png"
    plt.savefig(output_path)


def show_matches2(image1: int, image2: int, points1: np.array, points2: np.array):
    """
    Display the matches between two images using provided points.

    Parameters
    ----------
    image1 : int
        The first image number.
    image2 : int
        The second image number.
    points1 : np.ndarray
        The points from the first image.
    points2 : np.ndarray
        The points from the second image.
    """
    # Load the images
    img1 = load_image(image1)
    img2 = load_image(image2)
    # Concatenate the images side by side
    img = np.concatenate((img1, img2), axis=1)
    # Draw circles and lines for the matches
    for i in range(points1.shape[0]):
        x1, y1 = points1[i, :]
        x2, y2 = points2[i, :]
        cv2.circle(img, (int(x1), int(y1)), 3, (0, 0, 255), 2)
        cv2.circle(img, (int(x2) + img1.shape[1], int(y2)), 3, (0, 0, 255), 2)
        cv2.line(
            img, (int(x1), int(y1)), (int(x2) + img1.shape[1], int(y2)), (127, 0, 0), 1
        )
    # Display the image with matches
    plt.figure(figsize=(10, 10))
    plt.title("Matches between Image " + str(image1) + " and Image " + str(image2))
    plt.imshow(img)
    plt.axis("off")
    plt.show()
    # Save the image with matches to the Outputs directory
    output_path = "Outputs/matches_" + str(image1) + "_" + str(image2) + ".png"
    plt.savefig(output_path)
