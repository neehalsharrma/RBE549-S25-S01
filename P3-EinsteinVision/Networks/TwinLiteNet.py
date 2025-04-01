"""
TwinLiteNet Model Loader.

This module provides functionality to load the TwinLiteNetPlus model with
pre-trained weights and set it to evaluation mode.

Functions
---------
load_TwinLiteNet(weights: str)
    Loads the TwinLiteNetPlus model with the specified pre-trained weights.
"""

import sys

import torch
from Networks.TwinLiteNetPlus.model.model import TwinLiteNetPlus

# Disable the creation of __pycache__ directories
sys.dont_write_bytecode = True


def load_TwinLiteNet(weights: str = "Pretrained/tlp_medium.pth") -> TwinLiteNetPlus:
    """
    Load the TwinLiteNet model.

    This function initializes a TwinLiteNetPlus model, loads the specified pre-trained
    weights, and sets the model to evaluation mode.

    Parameters
    ----------
    weights : str, optional
        Path to the pre-trained weights file (default is 'Pretrained/tlp_medium.pth').

    Returns
    -------
    TwinLiteNetPlus
        An instance of the TwinLiteNetPlus model loaded with the specified weights.
    """
    # Initialize the TwinLiteNetPlus model
    tlp_medium = TwinLiteNetPlus()
    # Load the pre-trained weights into the model
    tlp_medium.load_state_dict(torch.load(weights))
    # Set the model to evaluation mode
    tlp_medium.eval()
    return tlp_medium
