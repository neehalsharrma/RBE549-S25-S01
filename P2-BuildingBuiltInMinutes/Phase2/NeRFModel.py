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

import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Prevent __pycache__ generation
sys.dont_write_bytecode = True


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

        Notes
        -----
        - A skip connection is added at layer 4 of the spatial coordinate MLP.
        - The last layer of the spatial coordinate MLP outputs both density (sigma) 
          and features, where the output dimension is `hidden_layer_size + 1`.
        """
        super(NeRFmodel, self).__init__()

        # Calculate input dimensions for positional encoding
        self.fc_input_dim = 3 + 3 * 2 * embed_pos_levels  # For spatial coordinates
        self.fc_feat_input_dim = 3 + 3 * 2 * embed_dir_levels  # For viewing directions

        # Define the MLP for spatial coordinates
        self.layers = nn.ModuleList()
        for i in range(8):
            # First layer takes positional encoding as input
            in_features = self.fc_input_dim if i == 0 else hidden_layer_size
            # Skip connection at layer 4
            if i == 4:
                in_features += self.fc_input_dim
            # Last layer outputs density (sigma) and features
            out_features = hidden_layer_size + 1 if i == 7 else hidden_layer_size
            self.layers.append(nn.Linear(in_features, out_features))

        # Define the feature layer for combining spatial and directional features
        self.feature_layer = nn.Linear(
            hidden_layer_size + self.fc_feat_input_dim, hidden_layer_size // 2
        )
        # Output layer for predicting RGB color values
        self.rgb_layer = nn.Linear(hidden_layer_size // 2, 3)

        # Store the positional encoding levels
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
        # Initialize the encoded tensor with the original inputs
        encoded = [inputs]
        # Append sine and cosine functions of the inputs for each level
        for i in range(num_levels):
            encoded.append(torch.sin(2**i * np.pi * inputs))
            encoded.append(torch.cos(2**i * np.pi * inputs))
        # Concatenate all encoded components along the last dimension
        return torch.cat(encoded, dim=-1)

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

        # Pass through the spatial coordinate MLP
        for i, layer in enumerate(self.layers):
            # Add skip connection at layer 4
            if i == 4:
                encoded_positions = torch.cat([encoded_positions, self.position_encoding(positions, self.embed_pos_levels)], -1)
            encoded_positions = F.relu(layer(encoded_positions))

        # Separate density (sigma) and features
        sigma, features = encoded_positions[..., -1], encoded_positions[..., :-1]

        # Positional encoding for viewing directions
        encoded_directions = self.position_encoding(directions, self.embed_dir_levels)
        # Combine spatial features and directional features
        features = torch.cat([features, encoded_directions], -1)
        features = F.relu(self.feature_layer(features))
        # Predict RGB color values
        rgb = self.rgb_layer(features)

        # Apply sigmoid activation to RGB values and return
        return torch.sigmoid(rgb), sigma
