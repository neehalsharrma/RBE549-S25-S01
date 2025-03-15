"""
NeRFModel: Neural Radiance Fields (NeRF) implementation.

This module defines a PyTorch implementation of the NeRF model, which is used 
to represent 3D scenes as a continuous volumetric function. The model takes 
spatial coordinates and viewing directions as input and outputs RGB color 
values and density (sigma) values.

Classes
-------
NeRFmodel
    Implements the NeRF model with positional encoding and a multi-layer perceptron (MLP).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NeRFmodel(nn.Module):
    """
    Neural Radiance Fields (NeRF) model.

    This class implements the NeRF model, which uses positional encoding and 
    a multi-layer perceptron (MLP) to map spatial coordinates and viewing 
    directions to RGB color values and density (sigma) values.

    Attributes
    ----------
    fc_input_dim : int
        Input dimension for the spatial coordinate MLP.
    fc_feat_input_dim : int
        Input dimension for the viewing direction MLP.
    layers : nn.ModuleList
        List of fully connected layers for the spatial coordinate MLP.
    feature_layer : nn.Linear
        Fully connected layer for combining spatial and directional features.
    rgb_layer : nn.Linear
        Output layer for predicting RGB color values.
    embed_pos_levels : int
        Number of levels for positional encoding of spatial coordinates.
    embed_dir_levels : int
        Number of levels for positional encoding of viewing directions.
    """
    def __init__(self, embed_pos_levels=10, embed_dir_levels=4, hidden_layer_size=256):
        """
        Neural Radiance Fields (NeRF) model.

        Parameters
        ----------
        embed_pos_levels : int, optional
            Number of levels for positional encoding of spatial coordinates, by default 10.
        embed_dir_levels : int, optional
            Number of levels for positional encoding of viewing directions, by default 4.
        hidden_layer_size : int, optional
            Number of neurons in each hidden layer, by default 256.
        """
        super(NeRFmodel, self).__init__()

        self.fc_input_dim = 3 + 3 * 2 * embed_pos_levels
        self.fc_feat_input_dim = 3 + 3 * 2 * embed_dir_levels

        # Define the MLP
        self.layers = nn.ModuleList()
        for i in range(8):
            in_features = self.fc_input_dim if i == 0 else hidden_layer_size
            if i in [4]:
                in_features += self.fc_input_dim

            if i in [7]:
                out_features = hidden_layer_size + 1
            else:
                out_features = hidden_layer_size
            self.layers.append(nn.Linear(in_features, out_features))

        self.feature_layer = nn.Linear(
            hidden_layer_size + self.fc_feat_input_dim, hidden_layer_size // 2
        )
        # Output layer
        self.rgb_layer = nn.Linear(hidden_layer_size // 2, 3)

        # Store the positional encoding length
        self.embed_pos_levels = embed_pos_levels
        self.embed_dir_levels = embed_dir_levels

    def position_encoding(self, inputs, num_levels):
        """
        Apply positional encoding to the input tensor.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (..., 3).
        num_levels : int
            Number of encoding levels.

        Returns
        -------
        torch.Tensor
            Positional encoded tensor of shape (..., 3 + 3 * 2 * num_levels).
        """
        encoded = [inputs]
        for i in range(num_levels):
            encoded.append(torch.sin(2**i * np.pi * inputs))
            encoded.append(torch.cos(2**i * np.pi * inputs))

        encoded_tensor = torch.cat(encoded, dim=-1)
        return encoded_tensor

    def forward(self, positions, directions):
        """
        Forward pass of the NeRF model.

        Parameters
        ----------
        positions : torch.Tensor
            Input spatial coordinates of shape (..., 3).
        directions : torch.Tensor
            Input viewing directions of shape (..., 3).

        Returns
        -------
        tuple
            A tuple containing:
            - torch.Tensor: RGB color values of shape (..., 3).
            - torch.Tensor: Density (sigma) values of shape (...,).
        """
        # Positional encoding for spatial coordinates
        encoded_positions = self.position_encoding(positions, self.embed_pos_levels)
        x = encoded_positions
        for i, layer in enumerate(self.layers):
            if i == 4:
                x = torch.cat([x, encoded_positions], -1)
            x = F.relu(layer(x))
        
        sigma, x = x[..., -1], x[..., :-1]

        # Positional encoding for viewing directions
        encoded_directions = self.position_encoding(directions, self.embed_dir_levels)
        x = torch.cat([x, encoded_directions], -1)
        x = F.relu(self.feature_layer(x))
        x = self.rgb_layer(x)

        return torch.sigmoid(x), sigma
