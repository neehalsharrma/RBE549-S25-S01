"""
This module contains the implementation of an unsupervised HomographyNet to estimate the homography between two images.

Classes:
    UnsupervisedHomographyNet: A PyTorch module for estimating homography in an unsupervised manner.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from Network.homography_model import Net  # Import the core network architecture


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
        self.model = Net(hparams)
        self.lr = hparams.get('lr', 0.001)  # Default learning rate if not provided

        # Define the localization network for the Spatial Transformer Network (STN)
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )

        # Define the regressor for the 3x2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2),
        )
        
        # Initialize a list to store validation outputs
        self.validation_outputs = []

    def stn(self, x: torch.Tensor) -> torch.Tensor:
        """
        Spatial transformer network forward function.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed input tensor.
        """
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

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
        img1, img2, batch_pset, batch_img, batch_corners = batch
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
        img1, img2, batch_pset, batch_img, batch_corners = batch
        # Perform the forward pass to get the predicted homography
        h_pred = self.forward(img1, img2)
        # Compute the loss
        loss = self.compute_loss(img1, img2, h_pred)
        # Log the validation loss
        self.log("val_loss", loss)
        # Store the validation loss in the outputs list
        self.validation_outputs.append({"val_loss": loss})
        return loss

    def on_validation_epoch_end(self):
        """
        Aggregate the validation results at the end of an epoch.
        """
        # Calculate the average validation loss
        avg_loss = torch.stack([x["val_loss"] for x in self.validation_outputs]).mean()
        # Log the average validation loss
        self.log("avg_val_loss", avg_loss)
        # Clear the validation outputs list
        self.validation_outputs.clear()

    def configure_optimizers(self):
        """
        Configure the optimizer for training.

        Returns:
            torch.optim.Optimizer: The optimizer.
        """
        # Use the Adam optimizer with the specified learning rate
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
