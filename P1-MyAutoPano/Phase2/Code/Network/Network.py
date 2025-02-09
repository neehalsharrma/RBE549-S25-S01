import torch.nn as nn
import sys
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

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


class HomographyModel(pl.LightningModule):
    def __init__(self, hparams: dict):
        super(HomographyModel, self).__init__()
        self.save_hyperparameters(hparams)
        input_size = hparams.get('InputSize', 128)  # Default value if not provided
        output_size = hparams.get('OutputSize', 8)  # Default value if not provided
        self.model = Net(input_size, output_size)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.model(a, b)

    def training_step(self, batch, batch_idx):
        img_a: torch.Tensor, patch_a: torch.Tensor, patch_b: torch.Tensor, corners: torch.Tensor, gt: torch.Tensor = batch
        delta = self.model(patch_a, patch_b)
        loss = loss_fn(delta, img_a, patch_b, corners)
        logs = {"loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, batch, batch_idx):
        img_a: torch.Tensor, patch_a: torch.Tensor, patch_b: torch.Tensor, corners: torch.Tensor, gt: torch.Tensor = batch
        delta = self.model(patch_a, patch_b)
        loss = loss_fn(delta, img_a, patch_b, corners)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": logs}

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

class Net(nn.Module):
    def __init__(self, input_size, output_size):
        """
        Inputs:
        input_size - Size of the Input
        output_size - Size of the Output
        """
        super().__init__()
        # Define the main model as a sequential container
        self.model = nn.Sequential(
            # First convolutional block with 2 input channels and 64 output channels
            conv_block(2, 64),
            # Second convolutional block with pooling
            conv_block(64, 64, pool=True),
            # Third convolutional block
            conv_block(64, 64),
            # Fourth convolutional block with pooling
            conv_block(64, 64, pool=True),
            # Fifth convolutional block with 128 output channels
            conv_block(64, 128),
            # Sixth convolutional block with pooling
            conv_block(128, 128, pool=True),
            # Seventh convolutional block
            conv_block(128, 128),
            # Eighth convolutional block
            conv_block(128, 128),
            # Dropout layer to prevent overfitting
            nn.Dropout2d(0.4),
            # Flatten the tensor for the fully connected layer
            nn.Flatten(),
            # Fully connected layer with 1024 output features
            nn.Linear(16 * 16 * 128, 1024),
            # Dropout layer to prevent overfitting
            nn.Dropout(0.4),
            # Final fully connected layer with output size
            nn.Linear(1024, output_size),
        )

        # Define the localization network for the Spatial Transformer Network (STN)
        self.localization = nn.Sequential(
            # First convolutional layer with 1 input channel and 8 output channels
            nn.Conv2d(1, 8, kernel_size=7),
            # Max pooling layer
            nn.MaxPool2d(2, stride=2),
            # ReLU activation function
            nn.ReLU(True),
            # Second convolutional layer with 10 output channels
            nn.Conv2d(8, 10, kernel_size=5),
            # Max pooling layer
            nn.MaxPool2d(2, stride=2),
            # ReLU activation function
            nn.ReLU(True),
        )

        # Define the regressor for the 3x2 affine matrix
        self.fc_loc = nn.Sequential(
            # Fully connected layer with 32 output features
            nn.Linear(10 * 3 * 3, 32),
            # ReLU activation function
            nn.ReLU(True),
            # Final fully connected layer with 6 output features (3x2 affine matrix)
            nn.Linear(32, 3 * 2),
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[1].weight.data.zero_()
        self.fc_loc[1].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    #############################
    # You will need to change the input size and output
    # size for your Spatial transformer network layer!
    #############################
    def stn(self, x):
        "Spatial transformer network forward function"
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, xa: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            xa (torch.Tensor): A mini-batch of input images a with shape (batch_size, channels, height, width).
            xb (torch.Tensor): A mini-batch of input images b with shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: The output of the network.
        """
        # Concatenate xa and xb along the channel dimension
        x = torch.stack((xa, xb), dim=1)
        # Pass the concatenated tensor through the model
        out = self.model(x)
        return out
