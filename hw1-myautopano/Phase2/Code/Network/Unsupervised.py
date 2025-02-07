import torch.nn as nn
import sys
import torch
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
import kornia  # You can use this to get the transform and warp in this project
from TensorDLT import TensorDLT

# Don't generate pyc codes
sys.dont_write_bytecode = True


def LossFn(delta, patch_b):
    """
    Inputs:
    delta - the pb estimate from the network --> (B, 1, 128, 128)
    patch_a - the pa patch --> (B, 1, 128, 128)
    patch_b - the pb patch --> (B, 1, 128, 128)
    corners - the corners of the patch a --> (B, 8, 1)
    Outputs:
    loss - the loss value

    Criterion for the loss function:
    MSE between the warped patch_b and the output of the network
    """
    ###############################################
    # Fill your loss function of choice here!
    ###############################################
    loss = F.mse_loss(delta, patch_b)
    ###############################################
    # You can use kornia to get the transform and warp in this project
    # Bonus if you implement it yourself
    ###############################################

    return loss


class HomographyModel(nn.Module):

    def training_step(self, batch, batch_idx):
        img_a, patch_a, patch_b, corners, gt = batch

        delta = self.model(patch_a, patch_b)
        loss = LossFn(delta, img_a, patch_b, corners)
        logs = {"loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, batch, batch_idx):
        patch_a, patch_b, img_a, corners = batch
        delta = self.model(patch_a, patch_b, corners, img_a)
        loss = LossFn(delta, patch_b)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": logs}


class UnsupervisedNet(nn.Module):

    # Spatial Transformer Network accepts images of size 320, 320
    def __init__(self, model: nn.Module, InputSize=(320, 320), OutputSize=(320, 320), device='cpu'):
        """
        Inputs:
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """
        super().__init__()
        #############################
        # Fill your network initialization of choice here!
        #############################
        h1, w1 = InputSize
        h2, w2 = OutputSize
        self.regression_model = model
        self.regression_model.to(device)
        self.TensorDLT = TensorDLT()

        self.M_matrix = torch.tensor([[w1 / 2, 0, w1 / 2],
                                      [0, h1 / 2, h1 / 2],
                                      [0, 0, 1]], dtype=torch.float32)

        #############################
        # You will need to change the input size and output
        # size for your Spatial transformer network layer!
        #############################
        # Spatial transformer localization-network

        # input image is 1 channel 320x320

        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),  # 314x314x8
            nn.MaxPool2d(2, stride=2),  # 157x157x8
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),  # 153x153x10
            nn.MaxPool2d(2, stride=2),  # 76x76x10
            nn.ReLU(True),
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 76 * 76, 64),
            nn.ReLU(True),
            nn.Linear(64, 2 * 3)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    #############################
    # You will need to change the input size and output
    # size for your Spatial transformer network layer!
    #############################
    def stn(self, x):
        "Spatial transformer network forward function"
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 76 * 76)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, pa, pb, ca, img_a):
        """
        Input:
        pa is a MiniBatch of the image a patches --> (B, 1, 128, 128)
        pb is a MiniBatch of the image b patches --> (B, 1, 128, 128)
        ca is a MiniBatch of the image a patch corners --> (B, 8, 1)
        img_a is the image a --> (B, 1, 320, 320)
        Outputs:
        out - output of the network --> (B, 1, 128, 128)
        h_guess - the homography matrix --> (B, 3, 3)
        """
        #############################
        # Fill your network structure of choice here!
        #############################
        b, c, h, w = pa.shape
        h4pt_guess = self.regression_model(pa, pb)
        h_guess = self.TensorDLT.forward(ca, h4pt_guess)  # B x 3 x 3
        # Spatial transformer network forward function
        out = kornia.geometry.warp_perspective(img_a, h_guess, dsize=(320, 320))  # B x 1 x 320 x 320
        # Extract the patches from the output to be the pb_guesses

        x_start = ca[:, 0, 0]  # Extract x-coordinates
        y_start = ca[:, 1, 0]  # Extract y-coordinates

        out = torch.stack([out[i, :, y_start[i]:y_start[i] + 128, x_start[i]:x_start[i] + 128] # B x 1 x 128 x 128
                           for i in range(b)
                           ])
        return out
