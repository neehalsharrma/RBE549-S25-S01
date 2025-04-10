import numpy as np
from ultralytics import YOLO
import os
import cv2

def load_lights_classifier():
    """
    Load the lights classification model.

    Returns
    -------
    model : ultralytics.YOLO
        The loaded lights classification model.
    """
    # Load the lights classification model
    curr_dir = os.path.dirname(__file__)
    model_path = os.path.join(curr_dir, 'Pretrained/yolo11n-lights-seg.pt')
    model = YOLO(model_path)
    return model
#
# def lights_reshape(img: np.array) -> np.array:
#     """
#     Reshape the image for YOLOv11 model.
#     :param img: input image
#     :return: input image reshaped for YOLOv11
#     """
#     # Resize the image to 640x640
#     img = cv2.resize(img, (64,96))
#     return img
