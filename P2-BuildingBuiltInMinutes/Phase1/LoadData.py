"""
This module provides functions to load images, calibration matrices, and correspondences for Structure from Motion (SfM).

Functions
---------
load_image(img: int, data_path: str = "../P2Data/") -> np.ndarray
    Load the image from the given path and return the image as a NumPy array.

load_calibration_matrix(data_path: str = "../P2Data/") -> np.ndarray
    Load the calibration matrix from the given path and return the calibration matrix as a NumPy array.

load_data_full(img: int, data_path: str = "../P2Data/", num_images: int = 5) -> tuple[int, np.ndarray]
    Load the data from the given path and return the data as a tuple.

load_correspondence(image1: int, image2: int, data_path: str = "../P2Data/", num_images: int = 5) -> np.ndarray
    Load the correspondences between two images from the given path and return the data as a NumPy array.

show_features(points: np.ndarray, img1: np.ndarray) -> None
    Show the feature points on the image.

show_matches(image1: int, image2: int, data_path: str = "../P2Data/") -> None
    Show the matches between two images.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


def load_image(img: int, data_path: str = "../P2Data/") -> np.ndarray:
    """
    Load the image from the given path and return the image as a NumPy array.

    Parameters
    ----------
    img : int
        The image number to load.
    data_path : str, optional
        The path to the data directory (default is "../P2Data/").

    Returns
    -------
    np.ndarray
        The image as a NumPy array with shape (height, width, channels).

    Raises
    ------
    FileNotFoundError
        If the image file is not found at the specified path.
    """
    img_path = data_path + str(img) + ".png"  # Construct the image file path
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # Read the image
    if img is None:
        raise FileNotFoundError(f"Image at path {img_path} not found.")  # Raise error if image not found
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert image from BGR to RGB
    return img


def load_calibration_matrix(data_path: str = "../P2Data/") -> np.ndarray:
    """
    Load the calibration matrix from the given path and return the calibration matrix as a NumPy array.

    Parameters
    ----------
    data_path : str, optional
        The path to the data directory (default is "../P2Data/").

    Returns
    -------
    np.ndarray
        The calibration matrix as a NumPy array.
    """
    calibration_file = data_path + "calibration.txt"  # Construct the calibration file path
    with open(calibration_file, "r") as file:
        lines = file.readlines()  # Read all lines from the file
        K = np.zeros((3, 3), dtype=np.float32)  # Initialize a 3x3 matrix
        for i, line in enumerate(lines):
            values = list(map(float, line.split()))  # Convert line to list of floats
            K[i] = np.array(values)  # Assign values to the matrix
    return K


def load_data_full(
    img: int, data_path: str = "../P2Data/", num_images: int = 5
) -> tuple[int, np.ndarray]:
    """
    Load the data from the given path and return the data as a tuple.

    Parameters
    ----------
    img : int
        The image number to load.
    data_path : str, optional
        The path to the data directory (default is "../P2Data/").
    num_images : int, optional
        The number of images to be used in the SfM matching (default is 5).

    Returns
    -------
    tuple[int, np.ndarray]
        A tuple containing the number of matches and the data from the file.

    Notes
    -----
    Format for matches:
    Each Row: (the number of matches for the jth feature)
              (Red Value) (Green Value) (Blue Value)
              (u_current image) (v_current image)
              (image id) (u_{image_id image}) (v_{image_id_image})
              (image id) (u_{image_id_image}) (v_{image id image}) …
    """
    matching_file = data_path + "matching" + str(img) + ".txt"  # Construct the matching file path
    header_data = 6  # Number of header data columns
    with open(matching_file, "r") as file:
        lines = file.readlines()  # Read all lines from the file
        n_features = int(lines[0].split(":")[1].strip())  # Extract the number of features
        num_lines = len(lines) - 1  # Number of lines containing match data
        matches = np.zeros(
            (num_lines, header_data + (num_images - 1) * 3), dtype=np.float32
        )  # Initialize the matches array
        for i, line in enumerate(lines[1:]):
            values = list(map(float, line.split()))  # Convert line to list of floats
            matches[i, : len(values)] = np.array(values)  # Assign values to the matches array

    return n_features, matches


def load_correspondence(
    image1: int, image2: int, data_path: str = "../P2Data/", num_images: int = 5
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
        The path to the data directory (default is "../P2Data/").
    num_images : int, optional
        The number of images to be used in the SfM matching (default is 5).

    Returns
    -------
    np.ndarray
        The correspondences as a num_correspondences x 4 array.
    """
    matching_file = data_path + "matching" + str(image1) + ".txt"  # Construct the matching file path
    header_data = 6  # Number of header data columns
    _, matches = load_data_full(image1, data_path, num_images)  # Load the full data for the first image

    correspondences = np.ndarray((0, 4), dtype=np.float32)  # Initialize the correspondences array
    for i in range(len(matches)):
        x1, y1 = matches[i, 4:6]  # Extract coordinates for the first image
        num_matches = int(matches[i, 0])  # Number of matches for the current feature
        for j in range(num_matches - 1):
            img_id = int(matches[i, header_data + j * 3])  # Extract the image ID
            if img_id != image2:
                continue  # Skip if the image ID does not match the second image
            x2, y2 = matches[i, (header_data + 1) + j * 3 : 9 + j * 3]  # Extract coordinates for the second image
            correspondences = np.append(
                correspondences, np.array([[x1, y1, x2, y2]]), axis=0
            )  # Append the correspondence to the array
            break
    return correspondences


def show_features(points: np.ndarray, img1: np.ndarray) -> None:
    """
    Show the feature points on the image.

    Parameters
    ----------
    points : np.ndarray
        The feature points to be displayed.
    img1 : np.ndarray
        The image on which the feature points will be displayed.
    """
    img1_features = img1.copy()  # Create a copy of the image
    for i in range(points.shape[0]):
        y, x = points[i, 4:6]  # Extract coordinates of the feature point
        cv2.circle(img1_features, (int(x), int(y)), 5, (0, 255, 0), -1)  # Draw a circle at the feature point
    plt.figure(figsize=(10, 10))
    plt.imshow(img1_features)  # Display the image with feature points
    plt.axis("off")
    plt.show()


def show_matches(image1: int, image2: int, data_path: str = "../P2Data/") -> None:
    """
    Show the matches between two images.

    Parameters
    ----------
    image1 : int
        The first image number for matches.
    image2 : int
        The second image number for matches.
    data_path : str, optional
        The path to the data directory (default is "../P2Data/").
    """
    img1 = load_image(image1)  # Load the first image
    img2 = load_image(image2)  # Load the second image
    img = np.concatenate((img1, img2), axis=1)  # Concatenate the two images side by side

    correspondences = load_correspondence(image1, image2)  # Load the correspondences between the two images

    for i in range(correspondences.shape[0]):
        x1, y1, x2, y2 = correspondences[i]  # Extract coordinates of the correspondence
        cv2.circle(img, (int(x1), int(y1)), 3, (0, 0, 255), 2)  # Draw a circle at the feature point in the first image
        cv2.circle(img, (int(x2) + img1.shape[1], int(y2)), 3, (0, 0, 255), 2)  # Draw a circle at the feature point in the second image
        cv2.line(
            img, (int(x1), int(y1)), (int(x2) + img1.shape[1], int(y2)), (127, 0, 0), 1
        )  # Draw a line connecting the feature points

    plt.figure(figsize=(10, 10))
    plt.title("Matches between Image " + str(image1) + " and Image " + str(image2))
    plt.imshow(img)  # Display the image with matches
    plt.axis("off")
    plt.show()
