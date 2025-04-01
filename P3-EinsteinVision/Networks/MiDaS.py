"""
MiDaS and ZoeDepth Model Utilities.

This module provides functionality to refresh the MiDaS repository and load
the ZoeDepth model with pre-trained weights.

Functions
---------
refresh_MiDaS()
    Forces a fresh download of the MiDaS repository and its associated model weights.

load_ZoeDepth()
    Loads the ZoeDepth model with pre-trained weights from the specified repository.
"""

import torch
import sys

# Disable the creation of __pycache__ directories
sys.dont_write_bytecode = True


def refresh_MiDaS() -> None:
    """
    Refresh the MiDaS repository.

    This function forces a fresh download of the MiDaS repository and its associated
    model weights.

    Returns
    -------
    None
    """
    # Trigger a fresh download of the MiDaS repository
    torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)


def load_ZoeDepth() -> torch.nn.Module:
    """
    Load the ZoeDepth model.

    This function loads the ZoeDepth model with pre-trained weights from the specified
    repository.

    Returns
    -------
    torch.nn.Module
        The ZoeDepth model loaded with pre-trained weights.
    """
    repo = "./Networks/ZoeDepth"  # Local source for ZoeDepth
    # Load the ZoeDepth model with pre-trained weights
    model_zoe_nk = torch.hub.load(
        repo, "ZoeD_NK", pretrained=True, source="local", skip_validation=False
    )
    return model_zoe_nk
