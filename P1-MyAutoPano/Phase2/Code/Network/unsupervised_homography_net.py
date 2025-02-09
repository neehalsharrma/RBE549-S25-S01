"""
This module contains the implementation of an unsupervised HomographyNet to estimate the homography between two images.

Classes:
    UnsupervisedHomographyNet: A PyTorch module for estimating homography in an unsupervised manner.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from cnn_network import Net  # Import the core network architecture


class UnsupervisedHomographyNet(pl.LightningModule):
    """
    UnsupervisedHomographyNet is a module for estimating the homography between two images in an unsupervised manner.

    Methods:
        __init__():
            Initializes the UnsupervisedHomographyNet module.
        forward(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
            Performs the forward pass to estimate the homography.
        compute_loss(img1: torch.Tensor, img2: torch.Tensor, h_pred: torch.Tensor) -> torch.Tensor:
            Computes the unsupervised loss for homography estimation.
    """
    def __init__(self, hparams: dict) -> None:
        """
        Initialize the UnsupervisedHomographyNet module.

        Args:
            hparams (dict): Hyperparameters for the model.
        """
        super(UnsupervisedHomographyNet, self).__init__()
        self.save_hyperparameters(hparams)
        input_size = hparams.get('InputSize', 128)  # Default value if not provided
        output_size = hparams.get('OutputSize', 8)  # Default value if not provided
        self.model = Net(input_size, output_size)
        self.lr = hparams.get('lr', 0.001)  # Default learning rate if not provided

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass to estimate the homography.

        Args:
            img1 (torch.Tensor): The first input image tensor with shape (batch_size, channels, height, width).
            img2 (torch.Tensor): The second input image tensor with shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: The estimated homography tensor with shape (batch_size, 8).
        """
        # Pass the input images through the network to get the predicted homography
        h_pred = self.model(img1, img2)
        return h_pred

    def compute_loss(self, img1: torch.Tensor, img2: torch.Tensor, h_pred: torch.Tensor) -> torch.Tensor:
        """
        Compute the unsupervised loss for homography estimation.

        Args:
            img1 (torch.Tensor): The first input image tensor with shape (batch_size, channels, height, width).
            img2 (torch.Tensor): The second input image tensor with shape (batch_size, channels, height, width).
            h_pred (torch.Tensor): The predicted homography tensor with shape (batch_size, 8).

        Returns:
            torch.Tensor: The computed loss tensor.
        """
        # Reshape the predicted homography to a 3x3 matrix
        h_pred = h_pred.view(-1, 3, 3)

        # Warp img2 using the predicted homography
        grid = F.affine_grid(h_pred[:, :2, :], img1.size(), align_corners=False)
        img2_warped = F.grid_sample(img2, grid, align_corners=False)

        # Compute the photometric loss between img1 and the warped img2
        loss = F.l1_loss(img1, img2_warped)
        return loss

    def training_step(self, batch, batch_idx):
        """
        Perform a training step.

        Args:
            batch (tuple): A tuple containing input data and the ground truth tensor.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss value.
        """
        # Unpack the batch
        img1, img2 = batch
        # Perform the forward pass to get the predicted homography
        h_pred = self.forward(img1, img2)
        # Compute the loss
        loss = self.compute_loss(img1, img2, h_pred)
        # Log the training loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Perform a validation step.

        Args:
            batch (tuple): A tuple containing the input data and the ground truth tensor.
            batch_idx (int): The index of the batch.

        Returns:
            dict: A dictionary containing the loss value.
        """
        # Unpack the batch
        img1, img2 = batch
        # Perform the forward pass to get the predicted homography
        h_pred = self.forward(img1, img2)
        # Compute the loss
        loss = self.compute_loss(img1, img2, h_pred)
        # Log the validation loss
        self.log("val_loss", loss)
        return loss

    def validation_epoch_end(self, outputs):
        """
        Aggregate the validation results at the end of an epoch.

        Args:
            outputs (list): A list of dictionaries containing the loss for each batch.

        Returns:
            dict: A dictionary containing the average loss for the epoch.
        """
        # Calculate the average validation loss
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        # Log the average validation loss
        self.log("val_loss", avg_loss)
        return {"avg_val_loss": avg_loss}

    def configure_optimizers(self):
        """
        Configure the optimizer for training.

        Returns:
            torch.optim.Optimizer: The optimizer.
        """
        # Use the Adam optimizer with the specified learning rate
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
