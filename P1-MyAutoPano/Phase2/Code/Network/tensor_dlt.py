"""
This module contains the implementation of the TensorDLT class, which is used to calculate the homography matrix
using Direct Linear Transformation (DLT) in a differentiable manner with PyTorch.

Classes:
    TensorDLT: A PyTorch module for calculating the homography matrix using DLT.
"""

import torch
import torch.nn as nn


class TensorDLT(nn.Module):
    """
    TensorDLT is a module for calculating the homography matrix using Direct Linear Transformation (DLT).

    Methods:
        __init__():
            Initializes the TensorDLT module.
        forward(ca: torch.Tensor, h4pt: torch.Tensor) -> torch.Tensor:
            Performs the forward pass to calculate the homography matrix.
    """

    def __init__(self) -> None:
        """
        Initialize the TensorDLT module.
        """
        super(TensorDLT, self).__init__()

    def forward(self, ca: torch.Tensor, h4pt: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass to calculate the homography matrix.

        Args:
            ca (torch.Tensor): Tensor of corners of patch A with shape (batch_size, 8, 1).
            h4pt (torch.Tensor): The 4-point homography matrix calculated from the corners of patch A and patch B with shape (batch_size, 8, 1).

        Returns:
            torch.Tensor: The 3x3 homography matrix with shape (batch_size, 3, 3).
        """
        batch_size = ca.shape[0]

        # Extract x and y coordinates from ca
        x = ca[:, 0::2]  # Extract x coordinates
        y = ca[:, 1::2]  # Extract y coordinates

        # Extract x' and y' coordinates from h4pt
        x_prime = h4pt[:, 0::2]  # Extract x' coordinates
        y_prime = h4pt[:, 1::2]  # Extract y' coordinates

        # Create tensors of zeros and ones with the same shape as x
        zeros = torch.zeros_like(x)  # Batch x 4
        ones = torch.ones_like(x)  # Batch x 4

        # Create rows for the A matrix
        row1 = torch.stack(
            [zeros, zeros, zeros, -x, -y, -ones, y_prime * x, y_prime * y], dim=2
        )  # Batch x 4 x 8
        row2 = torch.stack(
            [x, y, ones, zeros, zeros, zeros, -x_prime * x, -x_prime * y], dim=2
        )  # Batch x 4 x 8

        # Interleave row1 and row2 to form the A matrix
        a = torch.stack([row1, row2], dim=2).reshape(batch_size, 8, 8)  # Batch x 8 x 8
        b = torch.stack([x_prime, y_prime], dim=2).reshape(
            batch_size, 8, 1
        )  # Batch x 8 x 1

        # Calculate the homography matrix H
        h = torch.linalg.pinv(a) @ b  # Batch x 8 x 1
        h = torch.cat([h, torch.ones_like(h[:, 0:1, :])], dim=1).reshape(
            batch_size, 3, 3
        )  # Batch x 3 x 3

        return h
