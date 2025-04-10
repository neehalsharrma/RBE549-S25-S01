from ultralytics import YOLO
import os

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