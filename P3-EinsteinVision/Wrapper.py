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

encoder.FLOAT_REPR = lambda o: format(o, '.2f')
import os
import subprocess
import sys
sys.path.append('Networks/TwinLiteNetPlus')
import cv2
import numpy as np
import torch
import matplotlib
from Networks.DataLoader import get_frame, load_video, load_calibration_matrix
from Networks.DepthModel import load_depthmodel

from Networks.TwinLiteNet import load_TwinLiteNet, show_seg_result, preprocess_img, process_output
from Networks.YOLOv11 import load_model as load_yolo
from tqdm import tqdm  # For progress bar

# Disable the creation of __pycache__ directories
sys.dont_write_bytecode = True

yolo = True
run_depth = False
twin_lite = True


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
    if yolo:
        yolo_model = load_yolo()
    if run_depth:
        depth_model, imgprocessor = load_depthmodel()
        intrinsics = load_calibration_matrix("Data/Calib/front_cal.txt")
        f_x = intrinsics[0][0]

    if twin_lite:
        twinlite_model = load_TwinLiteNet(
            weights="Networks/Pretrained/tlp_medium.pth",
            config="medium",
            dev=device
        )

    # Initialize the JSON data structure
    spawn_data = []

    cmap = matplotlib.colormaps.get_cmap('viridis')

    # Create directories for saving outputs of each model
    results_dir = "./Results"
    jsons_dir = os.path.join(results_dir, "jsons")
    yolo_dir = os.path.join(results_dir, f"YOLO/vid_{video_num}")
    depth_dir = os.path.join(results_dir, f"DepthPro/vid_{video_num}")
    twinlite_dir_base = os.path.join(results_dir, f"TwinLiteNet/vid_{video_num}")
    twinlite_dir_imgs = os.path.join(twinlite_dir_base, 'imgs')
    twinlite_dir_masks = os.path.join(twinlite_dir_base, 'masks')

    os.makedirs(yolo_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(twinlite_dir_imgs, exist_ok=True)
    os.makedirs(twinlite_dir_masks, exist_ok=True)
    os.makedirs(jsons_dir, exist_ok=True)

    # Process each frame in the video
    for frame_idx in tqdm(range(0, num_frames, 5)):
        frame = get_frame(cap, frame_idx)
        if frame is None:
            continue

        if yolo:
            yolo_frame = frame.copy()
            # YOLOv11 object detection
            detections = yolo_model.predict(frame, verbose=False)
            objects = []
            for det in detections[0].boxes:
                obj_type = det.cls  # Object class

                bbox = det.xyxy.cpu().numpy()  # Bounding box coordinates
                bbox.astype(float)
                objects.append(
                    {
                        "type": yolo_model.names[int(obj_type)],
                        "position": {"x": float(bbox[0][0]), "y": float(bbox[0][1]), "z": 0},
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

            yolo_frame = cv2.cvtColor(yolo_frame, cv2.COLOR_RGB2BGR)
            # Save YOLO-annotated frame
            cv2.imwrite(os.path.join(yolo_dir, f"annotated_frame_{frame_idx}.png"), yolo_frame)

        if run_depth:
            depth_frame = frame.copy()
            inputs = imgprocessor(images=depth_frame, return_tensors="pt")

            with torch.no_grad():
                # depth estimation
                # depth = depth_model.infer_image(depth_frame, p_x) # HxW depth map in meters in numpy
                output = depth_model(**inputs)
                output.field_of_view = torch.tensor([f_x])
            output = imgprocessor.post_process_depth_estimation(output, target_sizes=[(h, w)])

            depth = output[0]["predicted_depth"].detach().cpu().numpy()

            np.save(os.path.join(depth_dir,f"depth_{frame_idx}.npy"), depth)

            # Used to check for differences between the two methods of saving the depth map
            # np.save(os.path.join(depth_dir, f"depth_frame_16_{frame_idx}.npy"), depth.astype(np.uint16))
            # Colorize the depth map
            colorized = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            colorized = colorized.astype(np.uint8)
            colorized = (cmap(colorized)[:, :, :3] * 255).astype(np.uint8)

            depth_map = depth * 255
            depth_map = depth_map.astype(np.uint16)

            # Save depth map as an image
            cv2.imwrite(
                os.path.join(depth_dir, f"depth_frame_16_{frame_idx}.png"),
                depth_map)

            colorized = cv2.cvtColor(colorized, cv2.COLOR_RGB2BGR)
            cv2.imwrite(
                os.path.join(depth_dir, f"depth_frame_color_{frame_idx}.png"),
                colorized)

        if twin_lite:
            # Preprocess the frame for TwinLiteNet
            img, pad_h, pad_w, height, width, ratio = preprocess_img(frame)

            twinlite_output = twinlite_model(img)

            da_seg_mask, ll_seg_mask = process_output(twinlite_output, pad_h, pad_w, height, width, ratio)

            img_vis = show_seg_result(frame, (da_seg_mask, ll_seg_mask), 0, 0)
            img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)

            cv2.imwrite(
                os.path.join(os.path.join(twinlite_dir_imgs, f"twinlite_frame_{frame_idx}.png")),
                img_vis)

            stacked = np.stack((da_seg_mask, ll_seg_mask), axis=0)
            np.save(os.path.join
                    (twinlite_dir_masks, f"twinlite_frame_{frame_idx}.npy"), stacked)

            # continue
            # Append frame data to the JSON structure
            spawn_data.append({"frame": frame_idx, "objects": objects})

            # Save the JSON data to the specified file
            with open(output_json, "w") as json_file:
                json.dump(spawn_data, json_file, indent=4)
            print(f"Saved spawn data to {output_json}")
    shutil.copy(output_json, os.path.join(results_dir, 'jsons', f"vid_{video_num}.json"))


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
    video_path = "Data/Sequences"  # Adjust as needed
    output_json = "./spawn.json"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Process the video and generate the JSON file
    process_video(video_path, 8, output_json, device=dev)

    # Run the Blender script
    # run_blender()
