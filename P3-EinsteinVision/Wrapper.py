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
import torch
from Networks.DataLoader import get_frame, load_video
from Networks.MiDaS import load_ZoeDepth
from Networks.TwinLiteNet import load_TwinLiteNet
from Networks.YOLOv11 import load_model as load_yolo

# Disable the creation of __pycache__ directories
sys.dont_write_bytecode = True


def process_video(video_path: str, output_json: str) -> None:
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
    cap, num_frames = load_video(video_path=video_path)
    print(f"Loaded video with {num_frames} frames.")

    # Load the models
    yolo_model = load_yolo()
    depth_model = load_ZoeDepth()
    twinlite_model = load_TwinLiteNet(
        weights="Networks/Pretrained/tlp_medium.pth", config="medium"
    )

    # Initialize the JSON data structure
    spawn_data = []

    # Create directories for saving outputs of each model
    results_dir = "./Results"
    yolo_dir = os.path.join(results_dir, "YOLO")
    depth_dir = os.path.join(results_dir, "MiDaS")
    twinlite_dir = os.path.join(results_dir, "TwinLiteNet")
    os.makedirs(yolo_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(twinlite_dir, exist_ok=True)

    # Process each frame in the video
    for frame_idx in range(num_frames):
        frame = get_frame(cap, frame_idx)
        if frame is None:
            continue

        # YOLOv11 object detection
        detections = yolo_model.predict(frame, verbose=False)
        objects = []
        for det in detections[0].boxes:
            obj_type = det.cls  # Object class
            bbox = det.xyxy.cpu().numpy()  # Bounding box coordinates
            objects.append(
                {
                    "type": yolo_model.names[int(obj_type)],
                    "position": {"x": bbox[0][0], "y": bbox[0][1], "z": 0},
                    "rotation": {"x": 0, "y": 0, "z": 0},
                    "scale": {"x": 1, "y": 1, "z": 1},
                }
            )
            # Draw bounding boxes on the frame
            cv2.rectangle(
                frame,
                (int(bbox[0][0]), int(bbox[0][1])),
                (int(bbox[0][2]), int(bbox[0][3])),
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                yolo_model.names[int(obj_type)],
                (int(bbox[0][0]), int(bbox[0][1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        # Save YOLO-annotated frame
        cv2.imwrite(os.path.join(yolo_dir, f"annotated_frame_{frame_idx}.png"), frame)

        # MiDaS depth estimation
        depth_map = depth_model(frame)
        depth_map = depth_map.cpu().numpy()

        # Save depth map as an image
        depth_map_normalized = cv2.normalize(
            depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        cv2.imwrite(
            os.path.join(depth_dir, f"depth_frame_{frame_idx}.png"),
            depth_map_normalized,
        )

        # TwinLiteNet processing (e.g., semantic segmentation or other tasks)
        twinlite_output = twinlite_model(
            torch.tensor(frame).permute(2, 0, 1).unsqueeze(0)
        )
        twinlite_output = twinlite_output.cpu().detach().numpy()

        # Save TwinLiteNet output as an image
        twinlite_output_normalized = cv2.normalize(
            twinlite_output[0, 0], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        cv2.imwrite(
            os.path.join(twinlite_dir, f"twinlite_frame_{frame_idx}.png"),
            twinlite_output_normalized,
        )

        # Append frame data to the JSON structure
        spawn_data.append({"frame": frame_idx, "objects": objects})

    # Save the JSON data to the specified file
    with open(output_json, "w") as json_file:
        json.dump(spawn_data, json_file, indent=4)
    print(f"Saved spawn data to {output_json}")


def run_blender() -> None:
    """
    Execute the Blender script to render 3D scenes.

    This function runs the Blender script (`Blender.py`) in background mode.
    The script uses the `spawn.json` file as its input.

    Returns
    -------
    None
    """
    # Define the path to the Blender script
    blender_script = "./Blender.py"

    # Run the Blender script
    subprocess.run(
        ["blender", "--background", "--python", blender_script],
        check=True,  # Ensure the command raises an error if it fails
    )
    print("Blender script executed.")


if __name__ == "__main__":
    # Define the input video path and output JSON file path
    video_path = "./Data/Sequences"  # Adjust as needed
    output_json = "./spawn.json"

    # Process the video and generate the JSON file
    process_video(video_path, output_json)

    # Run the Blender script
    run_blender()
