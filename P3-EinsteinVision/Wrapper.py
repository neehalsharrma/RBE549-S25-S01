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
import os
import subprocess
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from Networks.DataLoader import get_frame, load_video
from Networks.MiDaS import load_ZoeDepth
from Networks.ZoeDepth.zoedepth.utils.misc import colorize
from Networks.TwinLiteNet import (
    load_TwinLiteNet,
    show_seg_result,
    preprocess_img,
    process_output,
)
from Networks.YOLOv11 import load_model as load_yolo
from tqdm import tqdm  # For progress bar

import ssl
from pyffmpeg import FFmpeg

ssl._create_default_https_context = ssl._create_unverified_context

# Disable the creation of __pycache__ directories
sys.dont_write_bytecode = True

# Define global constants for model usage
USE_YOLO = True
USE_MIDAS = True
USE_TWIN_LITE = False

# Define the video number as a global constant
VIDEO_NUMBER = 5


def load_calibration_matrix(data_path: str = "./Data/Calib/") -> np.ndarray:
    """
    Load the calibration matrix from the given path and return the calibration matrix as a NumPy array.

    Parameters
    ----------
    data_path : str, optional
        The relative path to the data directory (default is './Data/Calib/').

    Returns
    -------
    np.ndarray
        The calibration matrix as a NumPy array.
    """
    # Construct the calibration file path
    calibration_file = data_path + "front_cal.txt"
    # Read the calibration file
    with open(calibration_file, "r", encoding="utf-8") as file:
        lines = file.readlines()
        # Initialize the calibration matrix
        K = np.zeros((3, 3), dtype=np.float32)
        # Populate the calibration matrix with values from the file
        for i, line in enumerate(lines):
            values = list(map(float, line.split()))
            K[i] = np.array(values)
    return K


def process_video(
    video_path: str, output_json: str, device: torch.device = torch.device("cpu")
) -> None:
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
    video = VIDEO_NUMBER  # Use the global constant
    cap, num_frames = load_video(video_path=video_path, video_num=video)
    print(f"Loaded video with {num_frames} frames.")

    # Load the models
    if USE_YOLO:
        yolo_model = load_yolo()
    if USE_MIDAS:
        depth_model = load_ZoeDepth()
    if USE_TWIN_LITE:
        twinlite_model = load_TwinLiteNet(
            weights="Networks/Pretrained/tlp_medium.pth", config="medium"
        )

    # Load the calibration matrix
    K = load_calibration_matrix()
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    print(f"Calibration matrix loaded: fx={fx}, fy={fy}, cx={cx}, cy={cy}")

    # Initialize the JSON data structure
    spawn_data = []

    # Create directories for saving outputs of each model
    results_dir = "./Results"
    yolo_dir = os.path.join(results_dir, f"YOLO/vid_{video}")
    depth_dir = os.path.join(results_dir, f"MiDaS/vid_{video}")
    twinlite_dir = os.path.join(results_dir, f"TwinLiteNet/vid_{video}")
    os.makedirs(yolo_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(twinlite_dir, exist_ok=True)

    # Process each frame in the video
    for frame_idx in tqdm(range(0, 20, 5)):
        frame = get_frame(cap, frame_idx)
        if frame is None:
            continue

        print(f"Processing frame {frame_idx}...")  # Debug: Frame processing start

        objects = []  # Initialize objects for this frame

        if USE_YOLO:
            print(f"Running YOLO on frame {frame_idx}...")  # Debug: YOLO start
            yolo_frame = frame.copy()
            # YOLOv11 object detection
            detections = yolo_model.predict(frame, verbose=False)
            print(f"YOLO completed for frame {frame_idx}.")  # Debug: YOLO end
            for det in detections[0].boxes:
                obj_type = det.cls
                bbox = det.xyxy.cpu().numpy()
                x, y = float(bbox[0][0]), float(bbox[0][1])
                objects.append(
                    {
                        "type": yolo_model.names[int(obj_type)],
                        "position": {"x": x, "y": y, "z": 0},  # Initialize z as 0
                        "rotation": {"x": 0, "y": 0, "z": 0},
                        "scale": {"x": 1, "y": 1, "z": 1},
                    }
                )
                # Draw bounding boxes on the frame
                cv2.rectangle(
                    yolo_frame,
                    (int(bbox[0][0]), int(bbox[0][1])),
                    (int(bbox[0][2]), int(bbox[0][3])),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    yolo_frame,
                    yolo_model.names[obj_type.int().item()],
                    (int(bbox[0][0]), int(bbox[0][1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            # Save YOLO-annotated frame
            cv2.imwrite(
                os.path.join(yolo_dir, f"annotated_frame_{frame_idx}.png"), yolo_frame
            )

        if USE_MIDAS:
            print(f"Running MiDaS on frame {frame_idx}...")  # Debug: MiDaS start
            midas_frame = frame.copy()
            midas_frame = midas_frame.astype(np.float32) / 255
            batched_frame = torch.from_numpy(midas_frame).permute(2, 0, 1).unsqueeze(0)

            def get_depth_from_prediction(pred):
                if isinstance(pred, torch.Tensor):
                    pred = pred
                elif isinstance(pred, (list, tuple)):
                    pred = pred[-1]
                elif isinstance(pred, dict):
                    pred = (
                        pred["metric_depth"] if "metric_depth" in pred else pred["out"]
                    )
                else:
                    raise NotImplementedError(f"Unknown output type {type(pred)}")
                return pred

            depth_map = depth_model.infer(batched_frame)
            depth_map = get_depth_from_prediction(depth_map)
            depth_map = depth_map.detach().cpu().squeeze().numpy()

            # Save depth map as an image
            depth_colored = colorize(
                depth_map, cmap="plasma"
            )  # Apply colormap for visualization
            depth_output_path = os.path.join(depth_dir, f"depth_frame_{frame_idx}.png")
            plt.imsave(depth_output_path, depth_colored)
            print(f"MiDaS completed for frame {frame_idx}.")  # Debug: MiDaS end

            # Update z-coordinate for each object using the depth map
            for obj in objects:
                x, y = int(obj["position"]["x"]), int(obj["position"]["y"])
                if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
                    depth = float(depth_map[y, x])  # Depth value becomes y
                    obj["position"]["y"] = depth  # Assign depth to y
                    obj["position"]["z"] = float((x - cx) * depth / fx)  # x becomes z
                    obj["position"]["x"] = (
                        float((y - cy) * depth / fy) * 10
                    )  # z becomes x

        if USE_TWIN_LITE:
            img, pad_h, pad_w, height, width, ratio = preprocess_img(frame)
            # TwinLiteNet processing (e.g., semantic segmentation or other tasks)
            twinlite_output = twinlite_model(img)
            da_seg_mask, ll_seg_mask = process_output(
                twinlite_output, pad_h, pad_w, height, width, ratio
            )

            img_vis = show_seg_result(frame, (da_seg_mask, ll_seg_mask), 0, 0)

            # Save TwinLiteNet output as an image
            cv2.imwrite(
                os.path.join(twinlite_dir, f"twinlite_frame_{frame_idx}.png"),
                img_vis,
            )

        print(f"Frame {frame_idx} processing completed.")  # Debug: Frame processing end

        spawn_data.append({"frame": frame_idx, "objects": objects})

    # Save the JSON data to the specified file
    with open(output_json, "w") as json_file:
        json.dump(
            spawn_data, json_file, indent=4, default=float
        )  # Ensure serialization
    print(f"Saved spawn data to {output_json}")


def generate_video_from_frames(
    frames_dir: str, output_video: str, fps: int = 30
) -> None:
    """
    Generate a video from a directory of frames using the pyffmpeg library.

    Parameters
    ----------
    frames_dir : str
        Path to the directory containing the frames.
    output_video : str
        Path to the output video file.
    fps : int, optional
        Frames per second for the output video (default is 30).

    Returns
    -------
    None
    """
    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(output_video)
    os.makedirs(output_dir, exist_ok=True)

    # Determine the input pattern based on the type of frames
    if "YOLO" in frames_dir:
        input_pattern = os.path.join(frames_dir, "annotated_frame_%d.png")
    elif "MiDaS" in frames_dir:
        input_pattern = os.path.join(frames_dir, "depth_frame_%d.png")
    elif "TwinLiteNet" in frames_dir:
        input_pattern = os.path.join(frames_dir, "twinlite_frame_%d.png")
    else:
        raise ValueError("Unknown frame type in frames_dir")

    # Initialize pyffmpeg
    ff = FFmpeg()

    # Use pyffmpeg to generate the video
    ff.options(
        f"-y -framerate {fps} -i {input_pattern} -c:v libx264 -pix_fmt yuv420p {output_video}"
    )
    print(f"Video generated at {output_video}")


if __name__ == "__main__":
    # Define the input video path and output JSON file path
    video_path = "Data/Sequences"  # Adjust as needed
    output_json = "./spawn.json"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Assign the video number to process
    VIDEO_NUMBER = 5  # Change this value to process a different video

    # Assign the models to use
    USE_YOLO = True  # YOLO
    USE_MIDAS = True  # MiDaS
    USE_TWIN_LITE = True  # TwinLiteNet

    # Process the video and generate the JSON file
    process_video(video_path, output_json, device=dev)

    # Generate video from YOLO-annotated frames
    if USE_YOLO:
        yolo_frames_dir = f"./Results/YOLO/vid_{VIDEO_NUMBER}"
        yolo_video_path = (
            f"./Results/FFMPEG/vid_{VIDEO_NUMBER}/yolo_annotated_video.mp4"
        )
        generate_video_from_frames(yolo_frames_dir, yolo_video_path)

    # Generate video from MiDaS depth outputs
    if USE_MIDAS:
        midas_frames_dir = f"./Results/MiDaS/vid_{VIDEO_NUMBER}"
        midas_video_path = f"./Results/FFMPEG/vid_{VIDEO_NUMBER}/midas_depth_video.mp4"
        generate_video_from_frames(midas_frames_dir, midas_video_path)

    # Generate video from TwinLiteNet outputs
    if USE_TWIN_LITE:
        twinlite_frames_dir = f"./Results/TwinLiteNet/vid_{VIDEO_NUMBER}"
        twinlite_video_path = (
            f"./Results/FFMPEG/vid_{VIDEO_NUMBER}/twinlite_output_video.mp4"
        )
        generate_video_from_frames(twinlite_frames_dir, twinlite_video_path)

    # Run the Blender script to render 3D scenes
    print("Processing completed. Please run the Blender script to render 3D scenes.")
