"""
Test module for inertial-only odometry using deep learning.

This module provides functionality to test a trained deep learning model
for inertial-only odometry. It processes IMU sensor data (accelerometer
and gyroscope readings) and compares the predicted poses against ground truth.

Functions
---------
generate_inertial_test_batch : Creates batches of IMU and pose data for testing
read_csv : Utility function to read CSV files into numpy arrays
plot_and_save_metrics : Visualizes and saves loss metrics as plots
test_inertial_odometry : Main testing function that evaluates model performance
    on datasets and saves predicted trajectories
main : Entry point that handles command-line arguments and orchestrates testing

The module includes functionality to:
- Generate test batches from IMU and pose data
- Process and transform pose representations
- Evaluate model performance on test datasets
- Visualize and save test results including trajectories and error metrics
"""

# Standard imports
import argparse
import csv
import os
from typing import List, Optional, Tuple

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
import transforms3d.euler as euler

# Local imports
from network import Inertial_encoder, LossFn, device
from utils import find_latest_model


def generate_inertial_test_batch(
    base_path: str,
    batch_size: int,
    start_idx: int,
    imu_data: np.ndarray,
    pose_data: np.ndarray,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int]:
    """Generate a batch of test data for inertial-only odometry.

    Parameters
    ----------
    base_path : str
        Path to the data folder containing IMU data and ground truth.
    batch_size : int
        Number of samples in the mini-batch.
    start_idx : int
        Starting index for data generation.
    imu_data : np.ndarray
        IMU data array containing accelerometer and gyroscope readings.
    pose_data : np.ndarray
        Pose data array containing ground truth pose information.

    Returns
    -------
    imu_batch_tensor : torch.Tensor or None
        Tensor containing batch of IMU data, or None if no data available.
    pose_batch_tensor : torch.Tensor or None
        Tensor containing batch of pose data, or None if no data available.
    start_idx : int
        Updated index for next batch generation.
    """
    # Initialize empty lists to store batch data
    imu_batch, pose_batch = [], []
    # Counter for collected samples
    sample_count = 0

    # Continue collecting samples until we have enough or reach data limit
    while sample_count < batch_size and start_idx < 500:
        # Use sequential indexing for testing
        rand_idx = start_idx

        # Extract 10 consecutive IMU samples starting at the current index
        # This forms a time window of IMU measurements
        imu_sample = torch.from_numpy(imu_data[rand_idx * 10 : rand_idx * 10 + 10])
        # Extract the corresponding pose ground truth
        pose_sample = torch.from_numpy(pose_data[rand_idx * 10])

        # Add samples to their respective batch lists
        imu_batch.append(imu_sample)
        pose_batch.append(pose_sample)
        # Move to next index and increment counter
        start_idx += 1
        sample_count += 1

    # Convert lists to tensors if not empty, otherwise return None
    imu_batch_tensor = torch.stack(imu_batch) if imu_batch else None
    pose_batch_tensor = torch.stack(pose_batch) if pose_batch else None

    # Transfer tensors to the specified device and return
    return (
        imu_batch_tensor.to(device) if imu_batch_tensor is not None else None,
        pose_batch_tensor.to(device) if pose_batch_tensor is not None else None,
        start_idx,
    )


def read_csv(file_path: str) -> np.ndarray:
    """Read a CSV file and return its contents as a numpy array.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.

    Returns
    -------
    np.ndarray
        Array containing the CSV data with all values converted to float.
    """
    # Open and read the CSV file
    with open(file_path, mode="r") as file:
        reader = csv.reader(file)
        # Convert each row to a list of floats
        file_data = [list(map(float, row)) for row in reader]
    # Return as a numpy array for efficient numerical operations
    return np.array(file_data)


def plot_and_save_metrics(
    loss_vs_iteration: List[float], checkpoint_path: str, title: str, filename: str
) -> None:
    """Plot and save loss metrics over iterations.

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

    Returns
    -------
    None
        This function doesn't return any value but saves the plot to disk.
    """
    # Create a new figure with specified size
    plt.figure(figsize=(10, 5))

    # Plot loss values against iteration numbers
    plt.plot(range(1, len(loss_vs_iteration) + 1), loss_vs_iteration, label="Iteration")

    # Add labels and title to make the plot informative
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(f"{title} Loss vs Iteration")
    plt.legend()

    # Construct full path for saving the plot
    save_path = os.path.join(checkpoint_path, filename)

    # Save the figure to disk
    plt.savefig(save_path)
    print(f"Figure saved at: {save_path}")

    # Display the plot
    plt.show()


def test_inertial_odometry(
    batch_size: int, checkpoint_path: str, latest_file: Optional[str], base_path: str
) -> None:
    """Perform testing operation for inertial-only odometry.

    This function loads the model, processes the test datasets, and evaluates
    the model performance. It also saves predicted poses and visualization metrics.

    Parameters
    ----------
    batch_size : int
        Size of the mini-batch for processing.
    checkpoint_path : str
        Path to save checkpoints and result visualizations.
    latest_file : str or None
        Path to the latest checkpoint file, None if no checkpoint to load.
    base_path : str
        Base path of the dataset containing different sequences.

    Returns
    -------
    None
        This function doesn't return any value but saves results to disk.
    """
    # Initialize the neural network model and move to GPU
    model = Inertial_encoder().to(device)

    # Load model weights from checkpoint if available
    if latest_file is not None:
        checkpoint = torch.load(latest_file + ".ckpt")
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded latest checkpoint with the name {latest_file}....")
    else:
        print("New model initialized....")

    # Process each dataset in the base directory
    dataset_names = os.listdir(base_path)
    for dataset_name in dataset_names:
        # Initialize lists to store test results
        loss_pose_test = []
        predicted_pose_list = []

        # Initialize zero values for initial pose
        initial_combined_data = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Construct paths to dataset files
        dataset_path = os.path.join(base_path, dataset_name)
        imu_file_path = os.path.join(dataset_path, "IMU_data_file.csv")
        pose_file_path = os.path.join(dataset_path, "Pose_data.csv")

        # Load IMU and pose data
        imu_data = read_csv(imu_file_path)
        pose_data = read_csv(pose_file_path)

        # Load ground truth data from MATLAB file
        states_file_path = os.path.join(dataset_path, "states.mat")
        mat = scipy.io.loadmat(states_file_path)

        # Extract time data and remove last two entries
        time_data = mat["time"][0][:-2]
        # Sample every 10th time point for efficiency
        time_10 = time_data[::10]

        # Extract state data (ground truth) and remove last two entries
        ground_truth = mat["state"][:-2]
        # Sample every 10th state for efficiency
        ground_truth_10 = ground_truth[::10]

        # Extract position and quaternion data
        gt_position = ground_truth_10[:, :3]
        gt_quat = ground_truth_10[:, 6:10]

        # Combine position and quaternion for complete pose representation
        combined_data = np.hstack((gt_position, gt_quat))

        # Stack time and pose data for ground truth file
        gt_file = np.column_stack((time_10, combined_data))

        # Initialize empty array for initial pose entry
        initial_entry = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Save ground truth poses to a text file
        output_file_path_gt = f"{dataset_name}_pose_gt_abs_inertial.txt"
        with open(output_file_path_gt, "w") as f:
            for gt_pose in gt_file:
                f.write(",".join(map(str, gt_pose.flatten())) + "\n")

        print(f"{dataset_name} Testing Started")

        # Initialize the starting index for batch generation
        start_idx = 0

        # Process 500 batches (test iterations)
        for i in range(500):
            # Generate a batch of test data
            imu_test_batch, pose_test_batch, start_idx = generate_inertial_test_batch(
                dataset_path, batch_size, start_idx, imu_data, pose_data
            )

            # Only process if valid data was returned
            if imu_test_batch is not None and pose_test_batch is not None:
                # Set model to evaluation mode (no gradient computation)
                model.eval()
                # Disable gradient calculation for efficiency
                with torch.no_grad():
                    # Get model predictions
                    predicted_pose = model(imu_test_batch.float()).float()
                    predicted_pose = predicted_pose.to(device)

                    # Extract rotation angles (roll, pitch, yaw)
                    rpy = predicted_pose[:, 3:].cpu().numpy()

                    # Convert Euler angles to quaternions
                    quats = euler.euler2quat(rpy[:, 0], rpy[:, 1], rpy[:, 2])

                    # Combine timestamp, predicted position, and quaternion
                    combined_data = np.hstack(
                        (
                            time_10[i],
                            predicted_pose[:, :3].cpu().numpy()[0],
                            quats,
                        )
                    )

                    # Calculate loss between predicted and ground truth poses
                    loss_pose_train = LossFn(
                        predicted_pose, pose_test_batch.float()
                    ).float()
                    loss_pose_train = loss_pose_train.to(device)

                    # Store loss and predicted pose for later analysis
                    loss_pose_test.append(loss_pose_train.detach().cpu().numpy())
                    predicted_pose_list.append(combined_data)

        print(f"{dataset_name} Testing Ended")

        # Save predicted poses to a text file
        output_file_path = f"{dataset_name}_pose_test_predicted_abs_inertial.txt"
        with open(output_file_path, "w") as f:
            for predicted_pose in predicted_pose_list:
                f.write(",".join(map(str, predicted_pose.flatten())) + "\n")

        # Plot and save test metrics (loss over iterations)
        plot_and_save_metrics(
            loss_pose_test,
            checkpoint_path,
            "Testing",
            f"{dataset_name}_test_metrics.png",
        )


def main() -> None:
    """Run the main testing operation with command line arguments.

    This function parses command line arguments to configure the testing process,
    including data paths, batch size, and checkpoint loading options.

    Returns
    -------
    None
    """
    # Create argument parser for command line options
    parser = argparse.ArgumentParser()

    # Define command line arguments with default values
    parser.add_argument(
        "--BasePath", default="./Data", help="Base path of images, Default:../Data"
    )
    parser.add_argument(
        "--CheckPointPath",
        default="Checkpoints_Inertial/",
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

    # Parse arguments
    args = parser.parse_args()

    # Extract argument values
    base_path = args.BasePath
    batch_size = args.MiniBatchSize
    load_checkpoint = args.LoadCheckPoint
    checkpoint_path = args.CheckPointPath

    # Create checkpoint directory if it doesn't exist
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # Find latest checkpoint file if loading is enabled
    latest_file = find_latest_model(checkpoint_path) if load_checkpoint == 1 else None

    # Call the main testing function
    test_inertial_odometry(batch_size, checkpoint_path, latest_file, base_path)


# Script entry point - execute main() when running the script directly
if __name__ == "__main__":
    main()
