"""
Test Visual Odometry (VO) models.

This module provides functions and operations for testing visual-only odometry models.
It includes functionalities for:

- Generating test data batches from image pairs and pose data.
- Evaluating the model's performance on the test dataset.
- Visualizing and saving the results, including loss metrics and predicted poses.

The module supports loading pretrained weights and checkpoints for model initialization.

Attributes
----------
device : str
    The computation device to be used ('cuda' for GPU or 'cpu').

Functions
---------
generate_visual_test_batch(base_path: str, batch_size: int, start_idx: int, pose_data: np.ndarray) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int]
    Generate a batch of test data for visual-only odometry.

read_csv(file_path: str) -> np.ndarray
    Read a CSV file and return its contents as a numpy array.

plot_and_save_metrics(loss_vs_iteration: List[float], checkpoint_path: str, title: str, filename: str) -> None
    Plot and save loss metrics over iterations.

test_visual_odometry(batch_size: int, checkpoint_path: str, latest_file: Optional[str], base_path: str) -> None
    Perform testing operation for visual-only odometry.

main() -> None
    Main function to parse arguments and run the testing operation.
"""

# Standard library imports
import argparse
import csv
import os
import time
from typing import List, Optional, Tuple, Union

# Third-party imports
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
import transforms3d.euler as euler
from torch.optim import lr_scheduler

# Local imports
from network import LossFn, Visual_encoder
from utils import find_latest_model, tic, toc, remap, convert_to_one_hot

# Set device for computation
device = "cuda"


def generate_visual_test_batch(
    base_path: str, batch_size: int, start_idx: int, pose_data: np.ndarray
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int]:
    """
    Generate a batch of test data for visual-only odometry.

    This function reads image pairs and corresponding pose data from the dataset,
    preprocesses them, and returns a batch of images and poses.

    Parameters
    ----------
    base_path : str
        Path to the data folder containing images and ground truth.
    batch_size : int
        Number of samples in the mini-batch.
    start_idx : int
        Starting index for data generation.
    pose_data : np.ndarray
        Pose data array.

    Returns
    -------
    Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int]
        A tuple containing image batch, pose batch, and updated start index.
    """
    # Initialize empty lists to collect image and pose samples
    img_batch, pose_batch = [], []
    image_count = 0

    # Continue generating samples until we have enough or reach dataset limit
    while image_count < batch_size and start_idx < 500:
        # Use sequential index for testing (not random)
        rand_idx = start_idx

        # Construct paths to consecutive image frames
        img1_path = os.path.join(base_path, f"{rand_idx}.png")
        img2_path = os.path.join(base_path, f"{rand_idx + 1}.png")

        # Get corresponding pose data (with stride of 10 in pose array)
        pose_sample = torch.from_numpy(pose_data[rand_idx * 10])

        # Load and preprocess first image
        img1 = cv2.imread(img1_path)  # Read image using OpenCV
        img1 = cv2.resize(img1, (180, 320)).astype(
            np.float32
        )  # Resize to standard dimensions

        # Load and preprocess second image
        img2 = cv2.imread(img2_path)  # Read image using OpenCV
        img2 = cv2.resize(img2, (180, 320)).astype(
            np.float32
        )  # Resize to standard dimensions

        # Stack images along channel dimension to create image pair representation
        # Axis 2 corresponds to the channel dimension in the (H,W,C) format
        stacked_img = np.concatenate([img1, img2], axis=2).astype(np.float32)

        # Transpose from HWC to CHW format (PyTorch convention) and normalize to [0,1] range
        stacked_img = np.transpose(stacked_img, (2, 0, 1)) / 255.0

        # Convert numpy arrays to PyTorch tensors and add to batch
        img_batch.append(torch.from_numpy(stacked_img))
        pose_batch.append(pose_sample)

        # Update counters for next iteration
        start_idx += 1
        image_count += 1

    # Convert lists to tensor batches (or None if empty)
    img_batch_tensor = torch.stack(img_batch) if img_batch else None
    pose_batch_tensor = torch.stack(pose_batch) if pose_batch else None

    # Return tensors moved to proper device and updated start index
    return (
        img_batch_tensor.to(device) if img_batch_tensor is not None else None,
        pose_batch_tensor.to(device) if pose_batch_tensor is not None else None,
        start_idx,
    )


def read_csv(file_path: str) -> np.ndarray:
    """
    Read a CSV file and return its contents as a numpy array.

    This function is used to load pose data from a CSV file.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.

    Returns
    -------
    np.ndarray
        Array containing the CSV data.
    """
    with open(file_path, mode="r") as file:
        reader = csv.reader(file)
        file_data = [list(map(float, row)) for row in reader]
    return np.array(file_data)


def plot_and_save_metrics(
    loss_vs_iteration: List[float], checkpoint_path: str, title: str, filename: str
) -> None:
    """
    Plot and save loss metrics over iterations.

    This function generates a plot of loss values against iterations and saves it
    to the specified checkpoint path.

    Parameters
    ----------
    loss_vs_iteration : List[float]
        List of loss values over iterations.
    checkpoint_path : str
        Path to save the plot.
    title : str
        Title of the plot.
    filename : str
        Name of the file to save the plot.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(loss_vs_iteration) + 1), loss_vs_iteration, label="Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(f"{title} Loss vs Iteration")
    plt.legend()

    save_path = os.path.join(checkpoint_path, filename)
    plt.savefig(save_path)
    print(f"Figure saved at: {save_path}")
    plt.show()


def test_visual_odometry(
    batch_size: int, checkpoint_path: str, latest_file: Optional[str], base_path: str
) -> None:
    """
    Perform testing operation for visual-only odometry.

    This function initializes the model, loads pretrained weights and checkpoints,
    and iterates through the dataset to evaluate the model's performance.

    Parameters
    ----------
    batch_size : int
        Size of the mini-batch.
    checkpoint_path : str
        Path to save checkpoints.
    latest_file : Optional[str]
        Path to the latest checkpoint file.
    base_path : str
        Base path of the dataset.
    """
    # Initialize the model and move it to the computation device
    model = Visual_encoder().to(device)

    # Load pretrained weights for the model
    pretrained_weights = torch.load(
        "./flownets_bn_EPE2.459.pth.tar", map_location="cpu"
    )
    model_dict = model.state_dict()
    # Filter and update only matching keys from the pretrained weights
    update_dict = {
        k: v for k, v in pretrained_weights["state_dict"].items() if k in model_dict
    }
    model_dict.update(update_dict)
    model.load_state_dict(model_dict)
    model = model.to(device)

    # Load the latest checkpoint if available
    if latest_file is not None:
        checkpoint = torch.load(latest_file + ".ckpt")
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded latest checkpoint with the name {latest_file}....")
    else:
        print("New model initialized....")

    # Iterate through each dataset in the base path
    dataset_names = os.listdir(base_path)
    for dataset_name in dataset_names:
        # Initialize lists for storing test losses and predicted poses
        loss_pose_test = []
        predicted_pose_list = []

        # Load pose data and ground truth from the dataset
        dataset_path = os.path.join(base_path, dataset_name)
        pose_path = os.path.join(dataset_path, "Pose_data.csv")
        pose_data = read_csv(pose_path)
        states_data = os.path.join(dataset_path, "states.mat")
        mat = scipy.io.loadmat(states_data)
        time_data = mat["time"][0][:-2]  # Extract time data
        time_10 = time_data[::10]  # Downsample time data by a factor of 10
        ground_truth = mat["state"][:-2]  # Extract ground truth states
        ground_truth_10 = ground_truth[::10]  # Downsample ground truth states
        gt_position = ground_truth_10[:, :3]  # Extract position data
        gt_quat = ground_truth_10[:, 6:10]  # Extract quaternion data
        combined_data = np.hstack(
            (gt_position, gt_quat)
        )  # Combine position and quaternion
        gt_file = np.column_stack((time_10, combined_data))  # Combine with time data

        # Save ground truth poses to a file
        output_file_path_gt = f"{dataset_name}_pose_gt_abs_visual.txt"
        with open(output_file_path_gt, "w") as f:
            for gt_pose in gt_file:
                f.write(",".join(map(str, gt_pose.flatten())) + "\n")

        print(f"{dataset_name} Testing Started")
        start_idx = 0

        # Generate test batches and evaluate the model
        for i in range(500):
            # Generate a batch of images and poses
            img_batch, pose_batch, start_idx = generate_visual_test_batch(
                dataset_path, batch_size, start_idx, pose_data
            )

            if img_batch is not None and pose_batch is not None:
                model.eval()  # Set the model to evaluation mode
                with torch.no_grad():
                    # Predict poses using the model
                    predicted_pose = model(img_batch.float()).float()
                    predicted_pose = predicted_pose.to(device)

                    # Convert predicted poses to quaternion format
                    rpy = (
                        predicted_pose[:, 3:].cpu().numpy()
                    )  # Extract roll, pitch, yaw
                    quats = euler.euler2quat(
                        rpy[:, 0], rpy[:, 1], rpy[:, 2]
                    )  # Convert to quaternions
                    combined_data = np.hstack(
                        (
                            time_10[i],  # Add corresponding time
                            predicted_pose[:, :3]
                            .cpu()
                            .numpy()[0],  # Add predicted position
                            quats,  # Add predicted quaternion
                        )
                    )

                    # Compute loss between predicted and ground truth poses
                    loss_pose_train = LossFn(predicted_pose, pose_batch.float()).float()
                    loss_pose_train = loss_pose_train.to(device)
                    loss_pose_test.append(
                        loss_pose_train.detach().cpu().numpy()
                    )  # Store loss
                    predicted_pose_list.append(combined_data)  # Store predicted pose

        print(f"{dataset_name} Testing Ended")

        # Save predicted poses to a file
        output_file_path = f"{dataset_name}_pose_test_predicted_abs_visual.txt"
        with open(output_file_path, "w") as f:
            for predicted_pose in predicted_pose_list:
                f.write(",".join(map(str, predicted_pose.flatten())) + "\n")

        # Plot and save loss metrics
        plot_and_save_metrics(
            loss_pose_test,
            checkpoint_path,
            "Testing",
            f"{dataset_name}_test_metrics.png",
        )


def main() -> None:
    """
    Main function to parse arguments and run the testing operation.

    This function parses command-line arguments, sets up paths, and calls the
    test_visual_odometry function to perform the testing process.
    """
    parser = argparse.ArgumentParser()
    # Argument for specifying the base path of the dataset
    parser.add_argument(
        "--BasePath", default="./Data", help="Base path of images, Default:../Data"
    )
    # Argument for specifying the checkpoint path
    parser.add_argument(
        "--CheckPointPath",
        default="Checkpoints_Visual/",
        help="Path to save Checkpoints, Default: Checkpoints/",
    )
    # Argument for specifying the mini-batch size
    parser.add_argument(
        "--MiniBatchSize",
        type=int,
        default=1,
        help="Size of the MiniBatch to use, Default:1",
    )
    # Argument for loading the latest checkpoint
    parser.add_argument(
        "--LoadCheckPoint",
        type=int,
        default=1,
        help="Load Model from latest Checkpoint from CheckPointsPath?, Default:0",
    )

    args = parser.parse_args()
    base_path = args.BasePath
    batch_size = args.MiniBatchSize
    load_checkpoint = args.LoadCheckPoint
    checkpoint_path = args.CheckPointPath

    # Create the checkpoint directory if it doesn't exist
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # Find the latest checkpoint file if loading is enabled
    latest_file = find_latest_model(checkpoint_path) if load_checkpoint == 1 else None
    # Call the testing function
    test_visual_odometry(batch_size, checkpoint_path, latest_file, base_path)


if __name__ == "__main__":
    main()
