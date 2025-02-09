import sys  # System-specific parameters and functions
from typing import Dict, List, Tuple  # Type hinting for better code readability and error checking

import torch  # Main PyTorch library
import torch.nn as nn  # PyTorch's neural network library
import pytorch_lightning as pl  # PyTorch Lightning library
import kornia  # You can use this to get the transform and warp in this project

# Don't generate pyc codes
sys.dont_write_bytecode = True


def loss_fn(batch_b_pred: torch.Tensor, batch_patch_b: torch.Tensor) -> torch.Tensor:
    """
    Calculate the loss between the predicted and ground truth values.

    Args:
        batch_b_pred (torch.Tensor): The predicted tensor.
        batch_patch_b (torch.Tensor): The ground truth tensor.

    Returns:
        torch.Tensor: The calculated loss.
    """
    criterion = nn.L1Loss()
    loss = criterion(batch_b_pred, batch_patch_b)
    return {'loss': loss}


def conv_block(in_channels: int, out_channels: int, pool: bool = False) -> nn.Sequential:
    """
    Create a convolutional block with optional pooling.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        pool (bool): If True, apply max pooling. Default is False.

    Returns:
        nn.Sequential: A sequential container of the convolutional block.
    """
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class Net(pl.LightningModule):
    """
    Convolutional Neural Network model for homography estimation.

    This class defines a convolutional neural network (CNN) model using PyTorch's
    neural network module. The network consists of several convolutional blocks,
    dropout layers, and fully connected layers.

    Attributes:
        homography_net (nn.Sequential): The sequential container of the network layers.
    """

    def __init__(self, hparams: dict) -> None:
        """
        Initialize the network.

        Args:
            hparams (dict): Hyperparameters for the model.
        """
        super().__init__()
        self.save_hyperparameters(hparams)

        # Extract input size and output size from hyperparameters
        input_size = hparams.get('InputSize', 128)
        output_size = hparams.get('OutputSize', 8)

        # Regression Network
        self.homography_net = nn.Sequential(
            conv_block(6, 64),  # Update input channels to 6 (concatenated img1 and img2)
            conv_block(64, 64, pool=True),
            conv_block(64, 64),
            conv_block(64, 64, pool=True),
            conv_block(64, 128),
            conv_block(128, 128, pool=True),
            conv_block(128, 128),
            conv_block(128, 128),
            nn.Dropout2d(0.5),
            nn.Flatten(),
            nn.Linear(16 * 16 * 128, 1024),
            nn.Dropout(0.5),
            nn.Linear(1024, output_size)
        )

    def forward(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            img1 (torch.Tensor): The first input image tensor.
            img2 (torch.Tensor): The second input image tensor.

        Returns:
            torch.Tensor: The output of the network.
        """
        # Concatenate img1 and img2 along the channel dimension
        x = torch.cat((img1, img2), dim=1)
        h4pt_pred = self.homography_net(x)
        return h4pt_pred

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
        img1, img2, batch_patch_b = batch
        # Perform the forward pass to get the predicted homography
        patch_b_pred = self.forward(img1, img2)
        # Compute the loss
        loss = loss_fn(patch_b_pred, batch_patch_b.to(self.device))['loss']
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
        img1, img2, batch_patch_b = batch
        # Perform the forward pass to get the predicted homography
        patch_b_pred = self.forward(img1, img2)
        # Compute the loss
        loss = loss_fn(patch_b_pred, batch_patch_b.to(self.device))['loss']
        # Log the validation loss
        self.log("val_loss", loss)
        return {"val_loss": loss}

    def configure_optimizers(self):
        """
        Configure the optimizer for training.

        Returns:
            torch.optim.Optimizer: The optimizer.
        """
        # Use the Adam optimizer with the specified learning rate
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['lr'])
        return optimizer
