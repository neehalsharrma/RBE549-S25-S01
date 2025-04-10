"""
This module contains functions to load various YOLO models for different tasks.
It includes functions to load various YOLO models for general autonomous driving object detection, lane detection,
traffic sign classification, and traffic light classification.
The models are loaded using the `ultralytics` library and are configured with
specific parameters.

Functions
---------
load_yoloe()
    Load the YOLOv11 model for general object detection.
load_lane_detector()
    Load the lane detection model.
load_yolo_traffic_classifier()
    Load the traffic sign classification model.
load_yolo_lights_classifier()
    Load the traffic light classification model.

Note that the lane detector, traffic sign detector, and traffic light classifier are custom trained models
"""

import sys
from ultralytics import YOLO
import os
import numpy as np
import cv2

# Disable the creation of __pycache__ directories
sys.dont_write_bytecode = True


def load_yoloe() -> YOLO:
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
    # model = YOLO("yolo11l.yaml").load("yolo11l.pt")
    model = YOLO("yoloe-11m-seg.pt")
    names = ["person", "bicycle", "car", "motorcycle", "bus", "train", "truck", 'traffic light',
             'traffic cone','person', 'pickup truck', 'trash can']
    model.set_classes(names, model.get_text_pe(names))
    model.eval()
    return model


def load_lane_detector():
    """
    Load the lane detection model.

    Returns
    -------
    model : ultralytics.YOLO
        The loaded lane detection model.
    """
    # Load the lane detection model
    curr_dir = os.path.dirname(__file__)
    model_path = os.path.join(curr_dir, 'Pretrained/yolo11s_seg_lines.pt')
    model = YOLO(model_path)
    return model


def load_yolo_traffic_classifier():
    """
    Load the traffic classification model.

    Returns
    -------
    model : ultralytics.YOLO
        The loaded traffic signs classification model.
    """
    # Load the traffic classification model
    curr_dir = os.path.dirname(__file__)
    model_path = os.path.join(curr_dir, 'Pretrained/yolo11s-traffic.pt')
    model = YOLO(model_path)
    return model


def load_yolo_lights_classifier():
    """
    Load the lights classification model:
    The lights classifier module takes in an extracted image of a traffic light, rescales it to 64x96 and then
    classifies it into one of:
        - go
        - stop
        - warning
        - stopLeft
        - goLeft
        - goForward
        - warningLeft
    This model was trained on the following data set:
    https://www.kaggle.com/datasets/chandanakuntala/cropped-lisa-traffic-light-dataset
    Returns
    -------
    model : ultralytics.YOLO
        The loaded lights classification model.
    """
    # Load the lights classification model
    curr_dir = os.path.dirname(__file__)
    model_path = os.path.join(curr_dir, 'Pretrained/yolov8n-lights-cls.pt')
    model = YOLO(model_path)
    return model


def process_traffic_lights(model: YOLO, lights: dict[int, np.array]) -> dict[int, str]:
    """
    Process the traffic lights detected by the YOLO model.
    :param model: the loaded YOLO model for traffic light classification
    :param lights: a dictionary of a light ID and its corresponding image that is output by the YOLOe model
    :return: a dictionary of the light ID and its corresponding classification
    """
    outputs = dict()
    for light_id, light_img in lights.items():
        classif = model.predict(light_img, verbose=False)

        outputs[light_id] = classif[0].names[classif[0].probs.top1]
        print(f"Light ID: {light_id}, Class: {outputs[light_id]}, Probabilities: {classif[0].probs.data * 100}")
    return outputs