from transformers import AutoConfig, AutoModel
import torch
import cv2
import numpy as np

def load_orientation_model():
    """
    Load the orientation model.

    Returns
    -------
    model : transformers.AutoModel
        The loaded orientation model.
    """
    # Load the orientation model
    checkpoint = "fort-cyber/Car-orientation-image"
    config = AutoConfig.from_pretrained(checkpoint, cache_dir='./Pretrained')
    model = AutoModel.from_pretrained(checkpoint, config=config, cache_dir='./Pretrained', torch_dtype="auto", device="auto")

    return model

def predict_orientation(model, image: np.array):
     # Preprocess the image and make predictions


    return orientation