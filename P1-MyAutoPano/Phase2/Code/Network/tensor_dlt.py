import torch
import torchvision
import kornia
import numpy
import torch.nn as nn

class TensorDLT(nn.Module):
    def __init__(self):
        super(TensorDLT, self).__init__()


    def forward(self, ca, H4pt):
        """
        Inputs:
        ca - [8 x 1] tensor of corners of patch A
        H4pt - the 4 point homography matrix calculated from the corners of patch A and patch B

        Format: [x1 y1 x2 y2 x3 y3 x4 y4]
        Size: batch_size x 8 x 1
        Outputs:
        H - 3x3 tensor of homography matrix
        """
        batch_size = ca.shape[0]


        x = ca[:, 0::2]  # Extract x coordinates
        y = ca[:, 1::2]  # Extract y coordinates
        x_prime = H4pt[:, 0::2]  # Extract x' coordinates
        y_prime = H4pt[:, 1::2]  # Extract y' coordinates

        zeros = torch.zeros_like(x)  # Batch x 4
        ones = torch.ones_like(x)  # Batch x 4

        row1 = torch.stack([zeros, zeros, zeros, -x, -y, -ones, y_prime * x, y_prime * y], dim=2)  # Batch x 4 x 8
        row2 = torch.stack([x, y, ones, zeros, zeros, zeros, -x_prime * x, -x_prime * y], dim=2)  # Batch x 4 x 8

        A = torch.stack([row1, row2], dim=2).reshape(batch_size, 8, 8)  # Interleave row1 and row2 to form A matrix # batch_size x 8 x 8
        B = torch.stack([x_prime, y_prime], dim=2).reshape(batch_size, 8, 1) # batch_size x 8 x 1

        H = torch.linalg.pinv(A) @ B  # batch_size x 8 x 1
        H = torch.cat([H, torch.ones_like(H[:, 0:1, :])], dim=1).reshape(batch_size, 3, 3)  # batch_size x 3 x 3

        return H
