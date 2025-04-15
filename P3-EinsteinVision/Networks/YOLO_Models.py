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
from numpy.polynomial.polynomial import polyfit
import cv2
from ultralytics.engine import results as Results
from skimage.transform import hough_line

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
             'traffic cone', 'person', 'pickup truck', 'trash can']
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

def process_lanes(lanes: list[Results] ) -> list[object]:
    """
    Process the lanes detected by the YOLO model. This takes the output segmented masks from the model
    and then processes them into a list of lane defined by a quadratic curve. For road markings, it
    returns a center line and the type of marking

    Lane Classses: divider-line, dotted-line, double-line, random-line, road-sign-line, solid-line

    :param lanes: a list of the results of the lane segmentation model
    :return: a dictionary of lane classification with
    a single element list with a JSON dict with the lane data
    """

    """
    JSON Formats
    Solid lines: a quadratic curve with a starting (x,y) and then the coefficients of the curve
    Dashed lines: a quadratic curve with a starting (x,y) and then the coefficients of the curve
    Random lines: a quadratic curve with a starting (x,y) and then the coefficients of the curve
    Example:
    {
        "lane_type": "solid",
        "curve": {
            "start": (x,y),
            "coeffs": [a,b,c]
        }
    }
    
    {
        "lane_type": "dashed",
        "curve": {
            "start": (x,y),
            "coeffs": [a,b,c]
        }
    }
    """
    jsons = []
    dashed_lines = []
    for lane in lanes[0]:
        # The points making up the lane
        lane_seg = lane.masks.cpu().xy[0]
        # Threshold the mask to remove small areas
        area = cv2.contourArea(lane_seg)
        if area < 250:
            continue
        # get the lane classification name
        cls = int(lane.bbox.data.cpu().numpy())
        cls = lane.names[cls]
        lane_mask = lane.masks.data.cpu().numpy().squeeze()
        if cls in ["solid-line", "divider-line", "double-line"]:
            # Process solid lines
            jsons.append(_identify_solid_lines(cls, lane_seg))
        # elif cls == "dotted-line":
        #     # Process dashed lines
        #     dashed_lines.append(lane_mask)
        else:  # random-line or road-sign-line or dotted
            # Process random lines
            jsons.append(_identify_random_lines(cls,lane))
    # jsons.append(_identify_dashed_lines(dashed_lines))

    return jsons

def _identify_solid_lines(cls:str, lane_seg:np.array) -> object:
    """
    Identify the solid lines in the lane segmentation mask.
    :param type: the lane classification
    :param lane: the np.array of the lane segmentation mask
    - Specfically, this is the output of the results.masks.xy
    :return: a single element list with a JSON dict with the lane data
    """

    quad = polyfit(lane_seg[:, 0], lane_seg[:, 1], 3)
    x = (lane_seg[:, 0].min() + lane_seg[0, 0].max())/2
    y = (lane_seg[:, 1].min() + lane_seg[0, 1].max())/2
    json = [{
            "type": cls,
            "position": {"x": x, "y": y, "z": 0},  # Initialize z as 0
            "coeff": {"a": quad[0], "b": quad[1], "c": quad[2], "d": quad[3]},
            "scale": {"x": 1, "y": 1, "z": 1},
        }]
    return json

#
# def _identify_dashed_lines(img_size: tuple[int,int],lanes: list[np.array]) -> dict[str]:
#     """
#     Identify the dashed lines in the lane segmentation mask.
#     :param lanes: the np.array of the lane segmentation mask
#     :return: a JSON string with the all the lane data
#     """
#



def _identify_random_lines(cls:str, lane: np.array) -> object:
    """
    Identify the random lines in the lane segmentation mask.
    :param type: the lane classification
    :param lane: the np.array of the lane segmentation mask
    :return: a single element list with a JSON dict with the lane data
    Lane data in this case is just an orientation bounding box
    - Specfically, this is the output of the results.masks.xy
    """
    oriented_rect = cv2.minAreaRect(lane)
    points  = cv2.boxPoints(oriented_rect)
    # Create a JSON object with the lane data
    json = [{
        "type": cls,
        "position": {"x": points[0][0], "y": points[0][1], "z": 0},  # Initialize z as 0
        "corners": {"a": points[0], "b": points[1], "c": points[2], "d": points[3]},
        "scale": {"x": 1, "y": 1, "z": 1},
    }]
    return json


def process_traffic_lights(model: YOLO, light: np.array) -> str:
    """
    Process the traffic lights detected by the YOLO model.
    :param model: the loaded YOLO model for traffic light classification
    :param lights: a light detected by the YOLOe model
    :return: a string with the classication of the light
    """
    classif = model.predict(light, verbose=False)
    light_id = classif[0].names[classif[0].probs.top1]
    return light_id