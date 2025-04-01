"""
YOLOv11 Model Loader.

This module provides functionality to load the YOLOv11 model using a specified
YAML configuration file and pre-trained weights.

Functions
---------
load_model()
    Loads the YOLOv11 model with the specified configuration and weights.
"""

import sys
from ultralytics import YOLO

# Disable the creation of __pycache__ directories
sys.dont_write_bytecode = True


def load_model() -> YOLO:
    """
    Load the YOLOv11 model.

    This function initializes a YOLO model using the specified YAML configuration
    file and pre-trained weights.

    Returns
    -------
    YOLO
        An instance of the YOLO model loaded with the specified configuration and weights.
    """
    # Initialize the YOLO model with the YAML configuration and load pre-trained weights
    model = YOLO("yolo11l.yaml").load("yolo11l.pt")
    return model
