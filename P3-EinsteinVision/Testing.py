"""
Wrapper Script for Video Processing and 3D Scene Generation.

This script integrates multiple modules to process a video, extract object and depth
information, and generate a JSON file for spawning 3D objects in Blender. It performs
the following steps:
1. Loads a video using the DataLoader module.
2. Processes each frame through YOLOv11, MiDaS, and TwinLiteNet models.
3. Generates a JSON file (`spawn.json`) containing object data.
4. Executes a Blender script to render 3D scenes based on the JSON file.

Modules Used
------------
- Networks.DataLoader: For loading video and extracting frames.
- Networks.YOLOv11: For object detection.
- Networks.MiDaS: For depth estimation.
- Networks.TwinLiteNet: For additional frame processing.
- Blender.py: For rendering 3D scenes.

Functions
---------
- process_video(video_path: str, output_json: str) -> None
    Processes a video and generates a JSON file with object data.
- run_blender() -> None
    Executes the Blender script to render 3D scenes.
"""

import json
from json import encoder

import matplotlib.pyplot as plt

encoder.FLOAT_REPR = lambda o: format(o, '.2f')
import os
import subprocess
import sys

sys.path.append('Networks/openpifpaf')
sys.path.append('Networks/TwinLiteNetPlus')
from Networks.YOLO_Models import load_lane_detector
import cv2
import numpy as np
import torch
import matplotlib
from Networks.DataLoader import get_frame, load_video, load_calibration_matrix
from Networks.OpticalFlow import get_optical_flow, visualize

from Networks.YOLO_Models import load_yoloe as load_yolo
from tqdm import tqdm  # For progress bar

# Disable the creation of __pycache__ directories
sys.dont_write_bytecode = True


def process_video(video_path: str, video_num: int, output_json: str,
                  device: torch.device = torch.device('cpu')) -> None:
    """
    Process a video and generate a JSON file with object data.

    This function loads a video, processes each frame through YOLOv11 for object detection,
    MiDaS for depth estimation, and TwinLiteNet for additional processing. The results
    are saved in a JSON file.

    Parameters
    ----------
    video_path : str
        Path to the video file or directory containing video sequences.
    output_json : str
        Path to the output JSON file where object data will be saved.

    Returns
    -------
    None
    """

    # Load the video and get the total number of frames
    cap, num_frames, h, w = load_video(video_path=video_path, video_num=video_num)
    print(f"Loaded video with {num_frames} frames.")

    # Load the models
    yolo_model = load_yolo().cpu()
    # lane_detector = load_lane_detector().to(device=device)

    # Initialize the JSON data structure
    spawn_data = []

    # Create directories for saving outputs of each model
    results_dir = "./Testing"
    jsons_dir = os.path.join(results_dir, "jsons")
    yolo_dir = os.path.join(results_dir, f"YOLO/vid_{video_num}")
    yolo_dir_frames = os.path.join(results_dir, f"YOLO/vid_{video_num}/frames")
    lanes_dir = os.path.join(results_dir, f"Lanes/vid_{video_num}")
    flow_dir = os.path.join(results_dir, f"Flow/vid_{video_num}")
    os.makedirs(os.path.join(flow_dir, 'imgs'), exist_ok=True)
    os.makedirs(os.path.join(flow_dir, 'numpy'), exist_ok=True)
    os.makedirs(flow_dir, exist_ok=True)
    os.makedirs(lanes_dir, exist_ok=True)
    os.makedirs(yolo_dir, exist_ok=True)
    os.makedirs(yolo_dir_frames, exist_ok=True)
    os.makedirs(jsons_dir, exist_ok=True)

    last_frame = None

    # Process each frame in the video
    for frame_idx in tqdm(range(0, num_frames, 5)):
        frame = get_frame(cap, frame_idx)
        if frame is None:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # frame = torch.from_numpy(frame).permute(2, 0, 1).float().
        # YOLOv11 object detection
        detections = yolo_model.predict(frame, verbose=False)
        if len(detections) == 0:
            continue

        if last_frame is not None:
            # Optical flow calculation
            flow = get_optical_flow(last_frame, frame)
            np.save(os.path.join(flow_dir, f"numpy/flow_{frame_idx}.npy"), flow)
            # Visualize the flow
            flow_rgb = visualize(flow)
            cv2.imwrite(os.path.join(flow_dir, f"imgs/flow_{frame_idx}.png"), flow_rgb)

        last_frame = frame
        continue

        # lanes = lane_detector.predict(frame, verbose=False)
        # if len(lanes) == 0:
        #     continue
        # # Process the detected lanes
        # lane_data = []
        # lane_frame = frame.copy()
        # for lane in lanes[0]:
        #
        #     lane_mask = lane.masks.data.cpu().numpy().squeeze()
        #     lane_mask = np.array(lane_mask, dtype=np.uint8)
        #     lane_mask = cv2.resize(lane_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        #     lane_seg = lane.masks.cpu().xy[0]
        #     lane_mask = np.stack([lane_mask*100, lane_mask*0, lane_mask*0], axis=-1)
        #     # plot on the original image
        #
        #     area = cv2.contourArea(lane_seg)
        #     if area < 200:
        #         continue
        #
        #     lane_frame = cv2.add(lane_frame, lane_mask)
        #     quad = np.polynomial.polynomial.polyfit(lane_seg[:,0], lane_seg[:,1], 3)
        #
        #     x_min, y_min = lane_seg[:,0].min(), lane_seg[:,1].min()
        #     x_max, y_max = lane_seg[:,0].max(), lane_seg[:,1].max()
        #     x_space = (x_max - x_min) / 10
        #
        #     points = []
        #     for i in range(10):
        #         x = x_min + i * x_space
        #         y = quad[0] + quad[1] * x + quad[2] * x**2 + quad[3] * x**3
        #         points.append((x, y))
        #
        #     # Draw the lane points on the frame
        #     for point in points:
        #         cv2.circle(lane_frame, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)

        # bbox = lane.boxes.xyxy.cpu().numpy()
        # # Draw bounding boxes on the frame
        # cv2.rectangle(
        #     lane_frame,
        #     (int(bbox[0][0]), int(bbox[0][1])),
        #     (int(bbox[0][2]), int(bbox[0][3])),
        #     (0, 255, 0),
        #     2,
        # )
        # cv2.imwrite(os.path.join(lanes_dir, f"lane_{frame_idx}.png"), lane_frame)

        yolo_frame = frame.copy()

        for i, det in enumerate(detections[0].boxes):
            if yolo_model.names[int(det.cls)] != 'car':
                continue

            bbox = det.xyxy.cpu().numpy()  # Bounding box coordinates
            bbox.astype(float)
            # Draw bounding boxes on the frame
            x1, y1, x2, y2 = bbox[0]
            cv2.rectangle(
                yolo_frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2,
            )
            sub_img = frame[int(y1):int(y2), int(x1):int(x2)]
            # Save the cropped image
            os.makedirs(os.path.join(yolo_dir, f"frame_{frame_idx}"), exist_ok=True)
            cv2.imwrite(os.path.join(yolo_dir, f'frame_{frame_idx}', f"crop_{i}.png"), sub_img)

        # Save YOLO-annotated frame
        cv2.imwrite(os.path.join(yolo_dir, f"frames/annotated_frame_{frame_idx}.png"), yolo_frame)


if __name__ == "__main__":
    # Define the input video path and output JSON file path
    video_path = "./Data/Sequences"  # Adjust as needed
    output_json = "./spawn.json"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Process the video and generate the JSON file
    process_video(video_path, 3, output_json, device=dev)

    # Run the Blender script
    # run_blender()
