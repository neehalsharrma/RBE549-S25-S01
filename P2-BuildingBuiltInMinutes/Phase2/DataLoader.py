#!/usr/bin/evn python3
"""
DataLoader Module

This module provides the DataLoader class, which is used to load and process datasets
for Neural Radiance Fields (NeRF). It reads image data, camera poses, and camera
parameters from a specified dataset directory.

Classes
-------
DataLoader
    A class to load and process datasets for NeRF.
"""

import json
import os
import sys

import cv2
import numpy as np

sys.dont_write_bytecode = True  # Prevent __pycache__ generation


class DataLoader:
    """
    A class to load and process datasets for NeRF (Neural Radiance Fields).

    Attributes
    ----------
    data_path : str
        Path to the dataset directory.

    Methods
    -------
    loadDataset(mode):
        Loads the dataset for the specified mode (train or test).
    """

    def __init__(self, data_path="lego"):
        """
        Initializes the DataLoader with the dataset path.

        Parameters
        ----------
        data_path : str, optional
            Path to the dataset directory (default is "lego").
        """
        self.data_path = data_path

    def load_dataset(self, mode):
        """
        Loads the dataset for the specified mode (train or test).

        Parameters
        ----------
        mode : str
            Mode of the dataset to load, either "train" or "test".

        Returns
        -------
        tuple
            A tuple containing:
            - images (numpy.ndarray): Array of loaded images.
            - poses (numpy.ndarray): Array of corresponding camera poses in the world frame.
            - camera_info (list): List containing image width, height, and focal length.
        """
        # Construct paths for images and JSON metadata
        image_base_path = self.data_path + "/"
        jsonfile_path = self.data_path + "/transforms_" + mode + ".json"

        # Load JSON metadata
        with open(jsonfile_path) as jsonfile:
            data = json.load(jsonfile)

        # Extract camera angle
        camera_angle_x = data["camera_angle_x"]
        images = []  # List to store images
        poses = []  # List to store camera poses

        # Iterate through frames in the JSON file
        for i in range(len(data["frames"])):
            frame = data["frames"][i]
            # Construct the full image path
            image_path = os.path.join(image_base_path, frame["file_path"] + ".png")
            # Read the image using OpenCV
            img = cv2.imread(image_path)
            images.append(img)
            # Extract the transformation matrix (camera pose)
            pose = frame["transform_matrix"]
            poses.append(pose)

        # Convert lists to numpy arrays
        images = np.array(images)
        poses = np.array(poses)

        # Extract image dimensions (height and width)
        H, W = images[0].shape[:2]
        # Compute the focal length using the camera angle
        focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

        # Assuming the same camera is used for all images, H, W, and focal length are constant
        camera_info = [W, H, focal]

        return images, poses, camera_info
