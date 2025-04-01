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
import matplotlib.pyplot
import numpy as np
from Networks.ZoeDepth.zoedepth.models.depth_model import DepthModel
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


def load_ZoeDepth()-> DepthModel:
    """
    Load the ZoeDepth model.

    This function loads the ZoeDepth model with pre-trained weights from the specified
    repository.

    Returns
    -------
    torch.nn.Module
        The ZoeDepth model loaded with pre-trained weights.
    """
    repo = "Networks/ZoeDepth/"  # Local source for ZoeDepth
    # Load the ZoeDepth model with pre-trained weights
    model_zoe_nk = torch.hub.load(repo, "ZoeD_NK", pretrained=True, source="local")
    model_zoe_nk.eval()

    return model_zoe_nk

def colorize(value, vmin=None, vmax=None, cmap='gray_r', invalid_val=-99, invalid_mask=None, background_color=(128, 128, 128, 255), gamma_corrected=False, value_transform=None):
    """Converts a depth map to a color image.

    Args:
        value (torch.Tensor, numpy.ndarry): Input depth map. Shape: (H, W) or (1, H, W) or (1, 1, H, W). All singular dimensions are squeezed
        vmin (float, optional): vmin-valued entries are mapped to start color of cmap. If None, value.min() is used. Defaults to None.
        vmax (float, optional):  vmax-valued entries are mapped to end color of cmap. If None, value.max() is used. Defaults to None.
        cmap (str, optional): matplotlib colormap to use. Defaults to 'magma_r'.
        invalid_val (int, optional): Specifies value of invalid pixels that should be colored as 'background_color'. Defaults to -99.
        invalid_mask (numpy.ndarray, optional): Boolean mask for invalid regions. Defaults to None.
        background_color (tuple[int], optional): 4-tuple RGB color to give to invalid pixels. Defaults to (128, 128, 128, 255).
        gamma_corrected (bool, optional): Apply gamma correction to colored image. Defaults to False.
        value_transform (Callable, optional): Apply transform function to valid pixels before coloring. Defaults to None.

    Returns:
        numpy.ndarray, dtype - uint8: Colored depth map. Shape: (H, W, 4)
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    value = value.squeeze()
    if invalid_mask is None:
        invalid_mask = value == invalid_val
    mask = np.logical_not(invalid_mask)

    # normalize
    vmin = np.percentile(value[mask],2) if vmin is None else vmin
    vmax = np.percentile(value[mask],85) if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.

    # squeeze last dim if it exists
    # grey out the invalid values

    value[invalid_mask] = np.nan
    cmapper = matplotlib.cm.get_cmap(cmap)
    if value_transform:
        value = value_transform(value)
        # value = value / value.max()
    value = cmapper(value, bytes=True)  # (nxmx4)

    # img = value[:, :, :]
    img = value[...]
    img[invalid_mask] = background_color

    #     return img.transpose((2, 0, 1))
    if gamma_corrected:
        # gamma correction
        img = img / 255
        img = np.power(img, 2.2)
        img = img * 255
        img = img.astype(np.uint8)
    return img
