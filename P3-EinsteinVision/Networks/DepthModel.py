"""
load_unidepth Utils

This module provides functionality to load the load_unidepth model and its
associated weights.

Functions
---------
load_unidepth()
    Load the UniDepth model with pre-trained weights.

"""

import sys


import requests
from PIL import Image
import torch
from transformers import DepthProImageProcessor, DepthProForDepthEstimation

# Disable the creation of __pycache__ directories
sys.dont_write_bytecode = True


def load_depthmodel():
    checkpoint = "Networks/Pretrained/depth_pro"
    image_processor = DepthProImageProcessor.from_pretrained(checkpoint)
    model = DepthProForDepthEstimation.from_pretrained(
        checkpoint, use_fov_model=False, torch_dtype=torch.float32
    )

    return model, image_processor
