"""
Network module for Visual-Inertial Odometry (VIO).

This module implements neural network architectures for odometry estimation
using visual and inertial data. It contains three main networks:

- Visual encoder: Processes image pairs to estimate pose change using a CNN+LSTM architecture
- Inertial encoder: Processes IMU measurements (accelerometer and gyroscope data)
  through Conv1D and LSTM layers to estimate pose change
- Visual-Inertial encoder: Fuses outputs from both encoders to leverage complementary
  strengths of visual and inertial measurements for more robust pose estimation

The networks are designed to predict 6-DoF pose changes (3D position and orientation)
between consecutive frames. The module also provides utility functions for data
preprocessing, normalization, and custom loss calculation that balances position
and orientation errors.
"""

from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

# Set device for computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initialize_network_parameters() -> Dict[str, Union[float, int]]:
    """
    Initialize the parameters of the network.

    This function sets up the initial hyperparameters for training the network,
    such as learning rate, batch size, and the number of epochs.

    Returns
    -------
    Dict[str, Union[float, int]]
        Dictionary containing initialized parameters.
    """
    parameters = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "num_epochs": 100,
    }
    return parameters


def remap(
    x: np.ndarray, out_min: float, out_max: float, in_min: float, in_max: float
) -> Optional[np.ndarray]:
    """
    Remap values from one range to another.

    This function takes an input array and maps its values from a specified input range
    to a specified output range. If the input or output range is invalid (e.g., zero range),
    the function returns None.

    Parameters
    ----------
    x : np.ndarray
        Input values to remap.
    out_min : float
        Output minimum value.
    out_max : float
        Output maximum value.
    in_min : float
        Input minimum value.
    in_max : float
        Input maximum value.

    Returns
    -------
    np.ndarray or None
        Remapped values or None if division by zero would occur.
    """
    if out_min == out_max or in_min == in_max:
        return None
    return np.add(
        np.divide(np.multiply(x - in_min, out_max - out_min), in_max - in_min), out_min
    )


def LossFn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calculate the loss between predicted and target poses.

    This function computes a combined loss for position and orientation errors.
    The position loss is calculated using Mean Squared Error (MSE), and the orientation
    loss is calculated using Cosine Embedding Loss. The final loss is a weighted sum
    of these two components.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted poses with shape [B, 6], where B is the batch size.
    target : torch.Tensor
        Target poses with shape [B, 6].

    Returns
    -------
    torch.Tensor
        Combined position and orientation loss.
    """
    loss_pos = nn.MSELoss()
    loss_poss = torch.sqrt(loss_pos(pred[:, :3], target[:, :3]))
    loss_angle = nn.CosineEmbeddingLoss()
    loss_angles = -loss_angle(
        pred[:, 3:],
        target[:, 3:],
        torch.ones(target[:, 3:].shape[0], device=target[:, 3:].device),
    )
    loss = 0.4 * loss_poss + 0.6 * loss_angles
    return loss


def conv(
    in_planes: int,
    out_planes: int,
    kernel_size: int = 3,
    stride: int = 1,
    dropout: float = 0,
) -> nn.Sequential:
    """
    Create a convolutional block with batch normalization, LeakyReLU activation,
    and dropout.

    This function constructs a sequential module containing a 2D convolutional layer,
    batch normalization, LeakyReLU activation, and an optional dropout layer.

    Parameters
    ----------
    in_planes : int
        Number of input channels.
    out_planes : int
        Number of output channels.
    kernel_size : int, optional
        Size of the convolving kernel, by default 3.
    stride : int, optional
        Stride of the convolution, by default 1.
    dropout : float, optional
        Dropout probability, by default 0.

    Returns
    -------
    nn.Sequential
        Sequential module containing Conv2d, BatchNorm2d, LeakyReLU, and Dropout.
    """
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            bias=False,
        ),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Dropout(dropout),
    )


class Visual_encoder(nn.Module):
    """
    Neural network for visual odometry.

    This network processes image pairs to predict relative pose changes between
    consecutive frames. It uses convolutional layers for feature extraction,
    an LSTM for temporal modeling, and a fully connected layer for 6-DoF pose estimation.

    Attributes
    ----------
    conv1 : nn.Sequential
        First convolutional block with 7x7 kernel.
    conv2 : nn.Sequential
        Second convolutional block with 5x5 kernel.
    conv3 : nn.Sequential
        Third convolutional block with 5x5 kernel.
    conv3_1 : nn.Sequential
        Additional convolutional block with 3x3 kernel.
    conv4 : nn.Sequential
        Fourth convolutional block with 3x3 kernel.
    lstm : nn.LSTM
        LSTM layer for temporal feature processing.
    linear : nn.Linear
        Fully connected layer to output 6-DoF pose.
    """

    def __init__(self) -> None:
        """
        Initialize the visual encoder network architecture.

        This constructor defines the convolutional layers, LSTM, and fully connected
        layer used in the visual encoder.
        """
        super(Visual_encoder, self).__init__()
        self.conv1 = conv(6, 64, kernel_size=7, stride=2, dropout=0.2)
        self.conv2 = conv(64, 128, kernel_size=5, stride=2, dropout=0.2)
        self.conv3 = conv(128, 256, kernel_size=5, stride=2, dropout=0.2)
        self.conv3_1 = conv(256, 256, kernel_size=3, stride=1, dropout=0.2)
        self.conv4 = conv(256, 512, kernel_size=3, stride=2, dropout=0.2)
        self.lstm = nn.LSTM(512, 256, 1, batch_first=True)
        self.linear = nn.Linear(256, 6)

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode image through the convolutional layers.

        This method processes the input image batch through the convolutional layers
        to extract high-level features.

        Parameters
        ----------
        x : torch.Tensor
            Input image batch of shape [B, 6, H, W], where B is the batch size,
            6 represents stacked image pairs, and H, W are height and width.

        Returns
        -------
        torch.Tensor
            Encoded image features.
        """
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv6 = self.conv4(out_conv3)
        return out_conv6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the visual encoder.

        This method processes the input image batch through the convolutional layers,
        LSTM, and fully connected layer to predict the 6-DoF pose.

        Parameters
        ----------
        x : torch.Tensor
            Input image batch.

        Returns
        -------
        torch.Tensor
            Predicted pose (position and orientation) with shape [B, 6].
        """
        x = self.encode_image(x)
        batch_size, channels, seq_len, variable = x.size()
        x = x.view(batch_size, seq_len * variable, channels)
        output, (_, _) = self.lstm(x)
        lstm_out = output[:, -1, :]
        output = self.linear(lstm_out)
        return output

    def validation_step(
        self, Img_test_batch: torch.Tensor, pose_test_batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform a validation step.

        This method computes the validation loss for a batch of test images and
        ground truth poses.

        Parameters
        ----------
        Img_test_batch : torch.Tensor
            Batch of test images.
        pose_test_batch : torch.Tensor
            Batch of ground truth poses.

        Returns
        -------
        torch.Tensor
            Validation loss.
        """
        prediction = self.forward(Img_test_batch)
        loss_val = LossFn(prediction, pose_test_batch)
        return loss_val


class Inertial_encoder(nn.Module):
    """
    Neural network for inertial odometry.

    This network processes IMU data sequences to predict relative pose changes.
    It uses 1D convolutional layers to extract features from the IMU measurements
    (3-axis accelerometer and 3-axis gyroscope data), followed by a multi-layer LSTM
    to model temporal dependencies in the motion sequence.

    The architecture is optimized for processing time-series data with 6 channels
    (3 for acceleration, 3 for angular velocity) and outputs a 6-DoF pose estimation.

    Attributes
    ----------
    encoder_conv : nn.Sequential
        Sequential container of Conv1D layers with BatchNorm, LeakyReLU, and Dropout
        for feature extraction from IMU data.
    lstm : nn.LSTM
        Two-layer LSTM for temporal sequence modeling.
    linear : nn.Linear
        Fully connected layer to output 6-DoF pose.
    """

    def __init__(self) -> None:
        """
        Initialize the inertial encoder network architecture.

        This constructor defines the 1D convolutional layers, LSTM, and fully connected
        layer used in the inertial encoder.
        """
        super(Inertial_encoder, self).__init__()
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.1),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.1),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.1),
        )
        self.lstm = nn.LSTM(256, 64, 2, batch_first=True)
        self.linear = nn.Linear(64, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the inertial encoder.

        This method processes the input IMU batch through the convolutional layers,
        LSTM, and fully connected layer to predict the 6-DoF pose.

        Parameters
        ----------
        x : torch.Tensor
            Input IMU batch with shape [B, seq_len, 6], where B is the batch size,
            seq_len is the sequence length, and 6 represents IMU channels.

        Returns
        -------
        torch.Tensor
            Predicted pose (position and orientation) with shape [B, 6].
        """
        x = self.encoder_conv(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        out = self.linear(h_n[-1])
        return out

    def validation_step(
        self, IMU_test_batch: torch.Tensor, pose_test_batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform a validation step.

        This method computes the validation loss for a batch of test IMU data and
        ground truth poses.

        Parameters
        ----------
        IMU_test_batch : torch.Tensor
            Batch of test IMU data.
        pose_test_batch : torch.Tensor
            Batch of ground truth poses.

        Returns
        -------
        torch.Tensor
            Validation loss.
        """
        prediction = self.forward(IMU_test_batch)
        loss_val = LossFn(prediction, pose_test_batch)
        return loss_val


class Visual_Inertial_encoder(nn.Module):
    """
    Neural network for visual-inertial odometry.

    This class implements a fusion network that combines visual and inertial measurements
    to predict relative pose changes. It leverages both the Visual_encoder and
    Inertial_encoder networks as feature extractors, and then fuses their outputs
    to produce a more accurate 6-DoF pose estimation.

    The network architecture consists of:
    1. A visual encoder branch that processes image pairs.
    2. An inertial encoder branch that processes IMU measurements.
    3. A fusion module that combines the outputs from both branches.
    4. Fully connected layers that produce the final pose prediction.

    Attributes
    ----------
    visual : Visual_encoder
        Visual encoder network for processing image pairs.
    inertial : Inertial_encoder
        Inertial encoder network for processing IMU data.
    linear1 : nn.Conv1d
        1D convolutional layer for initial fusion of modalities.
    relu : nn.LeakyReLU
        Activation function.
    linear2 : nn.Linear
        Final fully connected layer to output 6-DoF pose.
    """

    def __init__(self) -> None:
        """
        Initialize the visual-inertial encoder network architecture.

        This constructor defines the visual encoder, inertial encoder, and fusion layers
        used in the visual-inertial encoder.
        """
        super(Visual_Inertial_encoder, self).__init__()
        self.visual = Visual_encoder().to(device)
        self.inertial = Inertial_encoder().to(device)
        self.linear1 = nn.Conv1d(2, 64, kernel_size=3)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.linear2 = nn.Linear(256, 6)

    def forward(
        self, Img_train_batch: torch.Tensor, IMU_train_batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the visual-inertial encoder.

        This method processes the input image and IMU batches through their respective
        encoders, fuses the outputs, and predicts the 6-DoF pose.

        Parameters
        ----------
        Img_train_batch : torch.Tensor
            Batch of image pairs.
        IMU_train_batch : torch.Tensor
            Batch of IMU data.

        Returns
        -------
        torch.Tensor
            Predicted pose (position and orientation) with shape [B, 6].
        """
        img_pose = self.visual(Img_train_batch)
        inertial_pose = self.inertial(IMU_train_batch)
        x = torch.cat((img_pose.unsqueeze(1), inertial_pose.unsqueeze(1)), dim=1).to(
            device
        )
        x = self.linear1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        out = self.linear2(x)
        return out

    def validation_step(
        self,
        Img_test_batch: torch.Tensor,
        IMU_test_batch: torch.Tensor,
        pose_test_batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform a validation step.

        This method computes the validation loss for a batch of test images, IMU data,
        and ground truth poses.

        Parameters
        ----------
        Img_test_batch : torch.Tensor
            Batch of test images.
        IMU_test_batch : torch.Tensor
            Batch of test IMU data.
        pose_test_batch : torch.Tensor
            Batch of ground truth poses.

        Returns
        -------
        torch.Tensor
            Validation loss.
        """
        prediction = self.forward(Img_test_batch, IMU_test_batch)
        loss_val = LossFn(prediction, pose_test_batch)
        return loss_val
