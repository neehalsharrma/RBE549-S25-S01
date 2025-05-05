"""
Test Visual-Inertial Odometry (VIO) models.

This script contains functions and operations for testing deep learning-based
visual-inertial odometry models. It includes data generation, model evaluation,
and result visualization capabilities.

The module provides functionality to:
- Load pretrained VIO models
- Process image and IMU data from test datasets
- Perform inference and evaluate model performance
- Visualize and save trajectory and loss metrics

Functions
---------
generate_test_batch : Generate a batch of test data
read_csv : Read a CSV file into numpy array
plot_and_save_metrics : Plot and save loss metrics
test_operation : Perform testing operation for VIO
main : Main function to parse arguments and run testing

Notes
-----
This module requires pretrained FlowNet weights to initialize the visual encoder part of the model.

See Also
--------
Network.py : Contains the VIO network architecture definition
"""

# Standard library imports
import argparse
import csv
import os
from typing import List, Optional, Tuple

# Third-party imports
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
import transforms3d.euler as euler

# Local imports
from network import LossFn, Visual_Inertial_encoder
from utils import find_latest_model


def generate_test_batch(
    base_path: str,
    batch_size: int,
    start_idx: int,
    imu_data: np.ndarray,
    pose_data: np.ndarray,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], int]:
    """
    Generate a batch of test data for visual-inertial odometry.

    Parameters
    ----------
    base_path : str
        Path to the data folder containing images, IMU data, and ground truth.
    batch_size : int
        Number of samples in the mini-batch.
    start_idx : int
        Starting index for data generation.
    imu_data : numpy.ndarray
        IMU data array.
    pose_data : numpy.ndarray
        Pose data array.

    Returns
    -------
    Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], int]
        A tuple containing image batch, IMU batch, pose batch, and updated start index.
    """
    img_batch, imu_batch, pose_batch = [], [], []
    image_count = 0

    while image_count < batch_size and start_idx < 500:
        # Use sequential index for test data
        rand_idx = start_idx

        # Construct file paths for consecutive images
        img1_path = os.path.join(base_path, f"{rand_idx}.png")
        img2_path = os.path.join(base_path, f"{rand_idx + 1}.png")

        # Extract corresponding IMU data (10 samples per image)
        imu_sample = torch.from_numpy(imu_data[rand_idx * 10 : rand_idx * 10 + 10])

        # Extract corresponding pose data (ground truth)
        pose_sample = torch.from_numpy(pose_data[rand_idx * 10])

        # Load and preprocess first image
        img1 = cv2.imread(img1_path)
        img1 = cv2.resize(img1, (180, 320)).astype(
            np.float32
        )  # Resize to model's expected input dimensions

        # Load and preprocess second image
        img2 = cv2.imread(img2_path)
        img2 = cv2.resize(img2, (180, 320)).astype(
            np.float32
        )  # Resize to model's expected input dimensions

        # Stack images along channel dimension for optical flow processing
        stacked_img = np.concatenate([img1, img2], axis=2).astype(np.float32)
        # Convert to PyTorch expected format (C, H, W) and normalize pixel values
        stacked_img = np.transpose(stacked_img, (2, 0, 1)) / 255.0

        # Add samples to respective batch lists
        img_batch.append(torch.from_numpy(stacked_img))
        imu_batch.append(imu_sample)
        pose_batch.append(pose_sample)
        start_idx += 1
        image_count += 1

    # Convert lists to tensors if not empty
    img_batch_tensor = torch.stack(img_batch) if img_batch else None
    imu_batch_tensor = torch.stack(imu_batch) if imu_batch else None
    pose_batch_tensor = torch.stack(pose_batch) if pose_batch else None

    # Move tensors to appropriate device and return
    return (
        img_batch_tensor.to(device) if img_batch_tensor is not None else None,
        imu_batch_tensor.to(device) if imu_batch_tensor is not None else None,
        pose_batch_tensor.to(device) if pose_batch_tensor is not None else None,
        start_idx,
    )


def read_csv(file_path: str) -> np.ndarray:
    """
    Read a CSV file and return its contents as a numpy array.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.

    Returns
    -------
    numpy.ndarray
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


def test_operation(
    batch_size: int, checkpoint_path: str, latest_file: Optional[str], base_path: str
) -> None:
    """
    Perform testing operation for visual-inertial odometry.

    Parameters
    ----------
    batch_size : int
        Size of the mini-batch.
    checkpoint_path : str
        Path to save checkpoints.
    latest_file : str or None
        Path to the latest checkpoint file.
    base_path : str
        Base path of the dataset.
    """
    # Initialize model
    model = Visual_Inertial_encoder().to(device)

    # Load pretrained weights
    pretrained_w = torch.load("./flownets_bn_EPE2.459.pth.tar", map_location="cpu")
    model_dict = model.visual.state_dict()
    update_dict = {
        k: v for k, v in pretrained_w["state_dict"].items() if k in model_dict
    }
    model_dict.update(update_dict)
    model.visual.load_state_dict(model_dict)
    model = model.to(device)

    # Load checkpoint if available
    if latest_file is not None:
        checkpoint = torch.load(latest_file + ".ckpt")
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded latest checkpoint with the name {latest_file}....")
    else:
        print("New model initialized....")

    # Iterate through dataset
    names = os.listdir(base_path)
    for name in names:
        loss_pose_test = []
        pose_train_predicted_list = []

        # Initialize combined data for ground truth
        initial_combined_data = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Define paths for IMU and pose data
        basepath = os.path.join(base_path, name)
        imu_path = os.path.join(basepath, "IMU_data_file.csv")
        pose_path = os.path.join(basepath, "Pose_data.csv")

        # Read IMU data from CSV file
        imu_data = read_csv(imu_path)

        # Read pose data from CSV file
        pose_data = read_csv(pose_path)

        # Load states data from MATLAB file
        states_data = os.path.join(basepath, "states.mat")
        mat = scipy.io.loadmat(states_data)

        # Extract time data and downsample it to match the IMU sampling rate
        time_data = mat["time"][0][:-2]
        time_10 = time_data[::10]

        # Extract ground truth states and downsample them
        gt = mat["state"][:-2]
        gt_10 = gt[::10]

        # Separate position and quaternion data from ground truth
        gt_position = gt_10[:, :3]
        gt_quat = gt_10[:, 6:10]

        # Combine position and quaternion data for output
        combined_data = np.hstack((gt_position, gt_quat))
        gt_file = np.column_stack((time_10, combined_data))

        # Initialize an entry for the ground truth file
        initial_entry = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Write ground truth data to a text file
        output_file_path_gt = f"{name}_pose_gt_abs_VIO.txt"
        with open(output_file_path_gt, "w") as f:
            for gt_pose in gt_file:
                f.write(",".join(map(str, gt_pose.flatten())) + "\n")

        print(f"{name} Testing Started")

        # Initialize the starting index for batch generation
        start_idx = 0

        # Iterate through the dataset for testing
        for i in range(500):
            # Generate a batch of test data
            img_batch, imu_batch, pose_batch, start_idx = generate_test_batch(
                basepath, batch_size, start_idx, imu_data, pose_data
            )

            # Check if the generated batch is valid
            if (
                img_batch is not None
                and pose_batch is not None
                and imu_batch is not None
            ):
                # Set the model to evaluation mode
                model.eval()

                # Perform inference without gradient computation
                with torch.no_grad():
                    # Predict poses using the model
                    pose_train_predicted = model(
                        img_batch.float(), imu_batch.float()
                    ).float()

                    # Move predictions to the appropriate device
                    pose_train_predicted = pose_train_predicted.to(device)

                    # Convert predicted poses to Euler angles and quaternions
                    rpy = pose_train_predicted[:, 3:].cpu().numpy()
                    quats = euler.euler2quat(rpy[:, 0], rpy[:, 1], rpy[:, 2])

                    # Combine time, position, and quaternion data for output
                    combined_data = np.hstack(
                        (
                            time_10[i],
                            pose_train_predicted[:, :3].cpu().numpy()[0],
                            quats,
                        )
                    )

                    # Compute loss for the predicted poses
                    loss_pose_train = LossFn(
                        pose_train_predicted, pose_batch.float()
                    ).float()
                    loss_pose_train = loss_pose_train.to(device)

                    # Append loss and predicted data to respective lists
                    loss_pose_test.append(loss_pose_train.detach().cpu().numpy())
                    pose_train_predicted_list.append(combined_data)

        print(f"{name} Testing Ended")

        # Write predicted pose data to a text file
        output_file_path = f"{name}_pose_test_predicted_abs_VIO.txt"
        with open(output_file_path, "w") as f:
            for predicted_pose in pose_train_predicted_list:
                f.write(",".join(map(str, predicted_pose.flatten())) + "\n")

        # Plot and save loss metrics for testing
        plot_and_save_metrics(
            loss_pose_test, checkpoint_path, "Testing", f"{name}_test_metrics.png"
        )


def main() -> None:
    """
    Main function to parse arguments and run the testing operation.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--BasePath", default="./Data", help="Base path of images, Default:../Data"
    )
    parser.add_argument(
        "--CheckPointPath",
        default="Checkpoints_Visual_Inertial_SGD/",
        help="Path to save Checkpoints, Default: Checkpoints/",
    )
    parser.add_argument(
        "--MiniBatchSize",
        type=int,
        default=1,
        help="Size of the MiniBatch to use, Default:1",
    )
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

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    latest_file = find_latest_model(checkpoint_path) if load_checkpoint == 1 else None
    test_operation(batch_size, checkpoint_path, latest_file, base_path)


if __name__ == "__main__":
    main()
