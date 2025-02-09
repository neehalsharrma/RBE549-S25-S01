"""
custom_image_dataset.py

This module defines a custom dataset class for loading image pairs and homography matrices
for training or testing purposes. The dataset reads images from specified directories and
homography matrices from a CSV file. It includes functionality to apply optional transformations
to the images and targets.

Classes:
    CustomImageDataset: A custom dataset class for loading image pairs and homography matrices.

Functions:
    extract_number(filename: str) -> int: Extract the first number found in a filename.

Usage:
    # Example usage
    dataset = CustomImageDataset(
        annotations_file="path/to/annotations.csv",
        img_dir="path/to/images/",
        transform=some_transform_function,
        target_transform=some_target_transform_function
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
"""

import os  # Module for interacting with the operating system
import re  # Module for regular expressions
from typing import Optional, Tuple  # For type hinting

import pandas as pd  # Library for data manipulation and analysis
import torch  # PyTorch library
from torch import Tensor  # Tensor class from PyTorch
from torch.utils.data import Dataset  # Dataset class from PyTorch
from torchvision.io import read_image  # Function to read images from files


def extract_number(filename: str) -> int:
    """Extract the first number found in a filename.

    Args:
        filename (str): The name of the file.

    Returns:
        int: The first number found in the filename, or -1 if no number is found.
    """
    match = re.search(r"\d+", filename)
    return int(match.group()) if match else -1


# Works for structure of type
# Phase2/Data/Train
#     PA
#         1_1A.jpg
#         1_2A.jpg
#         2_1A.jpg
#         2_2A.jpg
#         .
#         .
#     PB
#         1_1B.jpg
#         1_2B.jpg
#         2_1B.jpg
#         2_2B.jpg
#         .
#         .


# img_dir includes ending //
class CustomImageDataset(Dataset):
    """
    Custom dataset for loading image pairs and homography matrices.

    Args:
        annotations_file (str): Path to the CSV file with homography matrices.
        img_dir (str): Directory with all the images.
        transform (Optional[callable], optional): Optional transform to be applied on a sample. Defaults to None.
        target_transform (Optional[callable], optional): Optional transform to be applied on the target. Defaults to None.
    """

    def __init__(
        self,
        annotations_file: str,
        img_dir: str,
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
    ) -> None:
        # Read the homography matrices from the CSV file
        try:
            self.hs = pd.read_csv(annotations_file, header=None)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"File {annotations_file} not found.") from exc
        except pd.errors.EmptyDataError as exc:
            raise pd.errors.EmptyDataError(
                f"File {annotations_file} is empty."
            ) from exc
        except pd.errors.ParserError as exc:
            raise pd.errors.ParserError(
                f"File {annotations_file} could not be parsed."
            ) from exc
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        # Get sorted image paths for PA and PB subdirectories
        self.img_paths1 = self.get_image_paths("PA")
        self.img_paths2 = self.get_image_paths("PB")

    def get_image_paths(self, sub_dir: str) -> list:
        """
        Get sorted list of image paths from a subdirectory.

        Args:
            sub_dir (str): Subdirectory name.

        Returns:
            list: Sorted list of image paths.
        """
        dir_path = os.path.join(self.img_dir, sub_dir)
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Directory {dir_path} not found.")
        image_files = [
            os.path.join(dir_path, f)
            for f in os.listdir(dir_path)
            if f.endswith(".jpg")
        ]
        return sorted(image_files, key=extract_number)

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.hs)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Tuple containing the two images and the homography matrix.

        Raises:
            IndexError: If the index is out of bounds.
        """
        if idx >= len(self.img_paths1) or idx >= len(self.img_paths2):
            raise IndexError("Index out of bounds")

        # Load the images and homography matrix for the given index
        PA = self.load_image(self.img_paths1[idx])
        PB = self.load_image(self.img_paths2[idx])
        H = self.load_homography(idx)

        return PA, PB, H

    def load_image(self, path: str) -> Tensor:
        """
        Load an image from a given path.

        Args:
            path (str): Path to the image file.

        Returns:
            Tensor: Loaded image.
        """
        image = read_image(path)
        if self.transform is not None:
            image = self.transform(image)
        return image

    def load_homography(self, idx: int) -> Tensor:
        """
        Load a homography matrix from the annotations file.

        Args:
            idx (int): Index of the homography matrix.

        Returns:
            Tensor: Homography matrix.
        """
        return torch.tensor(self.hs.iloc[idx].tolist())
