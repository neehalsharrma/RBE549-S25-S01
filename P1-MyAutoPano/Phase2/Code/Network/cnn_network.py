import sys
from typing import List, Dict, Tuple

import torch
import torch.nn as nn

# Don't generate pyc codes
sys.dont_write_bytecode = True


def loss_fn(
    h_real: torch.Tensor, h_pred: torch.Tensor, criterion: nn.Module = nn.MSELoss()
) -> torch.Tensor:
    """
    Calculate the loss between the real and predicted values.

    Args:
        h_real (torch.Tensor): The ground truth tensor.
        h_pred (torch.Tensor): The predicted tensor.
        criterion (nn.Module): The loss function to use. Default is Mean Squared Error loss.

    Returns:
        torch.Tensor: The calculated loss.
    """
    loss = criterion(h_real, h_pred)
    return loss


class BaseModel(nn.Module):
    """
    BaseModel is a base class for training and validation steps in a neural network.

    This class provides methods for performing training and validation steps, as well as aggregating
    validation results at the end of an epoch and printing epoch end results.

    Attributes:
        model (nn.Module): Placeholder for the model, to be defined in subclasses.

    Methods:
        __init__():
            Initializes the base model.
        training_step(batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, torch.Tensor]:
            Performs a training step.
        validation_step(batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, torch.Tensor]:
            Performs a validation step.
        validation_epoch_end(outputs: List[Dict[str, torch.Tensor]]) -> Dict[str, float]:
            Aggregates the validation results at the end of an epoch.
        epoch_end(epoch: int, result: Dict[str, float]) -> None:
            Prints the epoch end results.
    """

    def __init__(self) -> None:
        """
        Initialize the base model.

        This constructor initializes the base model by calling the superclass
        constructor and setting up a placeholder for the model, which should be
        defined in subclasses.
        """
        super().__init__()
        self.model = None  # Placeholder for the model, to be defined in subclasses

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Perform a training step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A tuple containing input data and the ground truth tensor.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the loss value.
        """
        i, h_real = batch
        h_pred = self.model(i)
        loss = loss_fn(h_real, h_pred)
        return {"loss": loss.detach()}

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Perform a validation step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A tuple containing the input data and the ground truth tensor.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the loss value.
        """
        i, h_real = batch
        h_pred = self.model(i)
        loss = loss_fn(h_real, h_pred)
        return {"loss": loss.detach()}

    def validation_epoch_end(
        self, outputs: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:
        """
        Aggregate the validation results at the end of an epoch.

        Args:
            outputs (List[Dict[str, torch.Tensor]]): A list of dictionaries containing the loss for each batch.

        Returns:
            Dict[str, float]: A dictionary containing the average loss for the epoch.
        """
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        return {"avg_loss": avg_loss.item()}

    def epoch_end(self, epoch: int, result: Dict[str, float]) -> None:
        """
        Print the epoch end results.

        Args:
            epoch (int): The current epoch number.
            result (Dict[str, float]): A dictionary containing the loss values.

        Returns:
            None
        """
        print("Epoch [{}], loss: {:.4f}".format(epoch, result["loss"]))


def conv_block(
    in_channels: int, out_channels: int, pool: bool = False
) -> nn.Sequential:
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


class Net(BaseModel):
    """
    Convolutional Neural Network model.

    This class defines a convolutional neural network (CNN) model using PyTorch's
    neural network module. The network consists of several convolutional blocks,
    dropout layers, and fully connected layers.

    Attributes:
        model (nn.Sequential): The sequential container of the network layers.

    Methods:
        __init__(input_size: int, output_size: int) -> None:
            Initializes the network with the given input and output sizes.
        forward(xa: torch.Tensor) -> torch.Tensor:
            Performs the forward pass of the network on the input tensor.
    """

    def __init__(self, input_size: int, output_size: int) -> None:
        """
        Initialize the network.

        Args:
            input_size (int): Size of the input.
            output_size (int): Size of the output.
        """
        super().__init__()
        self.model = nn.Sequential(
            conv_block(2, 64),
            conv_block(64, 64, pool=True),
            conv_block(64, 64),
            conv_block(64, 64, pool=True),
            conv_block(64, 128),
            conv_block(128, 128, pool=True),
            conv_block(128, 128),
            conv_block(128, 128),
            nn.Dropout2d(0.4),
            nn.Flatten(),
            nn.Linear(16 * 16 * 128, 1024),
            nn.Dropout(0.4),
            nn.Linear(1024, output_size),
        )

    def forward(self, xa: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            xa (torch.Tensor): A mini-batch of input images with shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: The output of the network.
        """
        return self.model(xa)
