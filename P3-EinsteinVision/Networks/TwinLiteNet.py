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
import argparse
import cv2
import numpy as np

# Disable the creation of __pycache__ directories
sys.dont_write_bytecode = True


def load_TwinLiteNet(weights: str = "Pretrained/tlp_medium.pth", config='medium', dev:torch.device=torch.device('cpu')) -> TwinLiteNetPlus:
    """
    Load the TwinLiteNet model.

    This function initializes a TwinLiteNetPlus model, loads the specified pre-trained
    weights, and sets the model to evaluation mode.

    Parameters
    ----------
    weights : str, optional
        Path to the pre-trained weights file (default is 'Pretrained/tlp_medium.pth').
    config: str, optional
        Configuration of the model (default is 'medium').
    dev: torch.device, optional
        Device to load the model on (default is 'cpu').

    Returns
    -------
    TwinLiteNetPlus
        An instance of the TwinLiteNetPlus model loaded with the specified weights.

    """
    # Initialize the TwinLiteNetPlus model
    parser = argparse.ArgumentParser(description="TwinLiteNetPlus Model Loader.")
    parser.add_argument("--config", default=config)
    args = parser.parse_args()
    tlp_medium = TwinLiteNetPlus(args)
    # Load the pre-trained weights into the model

    print(f'Using device: {dev}')
    tlp_medium.load_state_dict(torch.load(weights, map_location=dev))
    # Set the model to evaluation mode
    tlp_medium.eval().to(dev)
    return tlp_medium


def show_seg_result(img, result, index, epoch, save_dir=None, is_ll=False, palette=None):
    # img = mmcv.imread(img)
    # img = img.copy()
    # seg = result[0]
    if palette is None:
        palette = np.random.randint(
            0, 255, size=(3, 3))
    palette[0] = [0, 0, 0]
    palette[1] = [0, 255, 0]
    palette[2] = [255, 0, 0]
    palette = np.array(palette)
    assert palette.shape[0] == 3  # len(classes)
    assert palette.shape[1] == 3
    assert len(palette.shape) == 2

    color_area = np.zeros((result[0].shape[0], result[0].shape[1], 3), dtype=np.uint8)

    # for label, color in enumerate(palette):
    #     color_area[result[0] == label, :] = color

    color_area[result[0] == 1] = [0, 255, 0]
    color_area[result[1] == 1] = [255, 0, 0]
    color_seg = color_area

    # convert to BGR
    color_seg = color_seg[..., ::-1]
    # print(color_seg.shape)
    color_mask = np.mean(color_seg, 2)
    img[color_mask != 0] = img[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
    # img = img * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)
    img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_LINEAR)

    return img

