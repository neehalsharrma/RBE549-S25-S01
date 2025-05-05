"""
Module for Training Visual-Inertial Odometry (VIO) Models.

This module provides functionalities for training visual-inertial odometry models.
It includes methods for data generation, model training, loss visualization, and
checkpoint management.

Functions
---------
generate_training_batch(base_path: str, batch_size: int, train_prob: float, start_idx: int, imu_data: np.ndarray, pose_data: np.ndarray) -> Tuple[Optional[torch.Tensor], ...]
    Generates a batch of training and testing data for visual-inertial odometry.

read_csv(file_path: str) -> np.ndarray
    Reads a CSV file and returns its contents as a numpy array.

pretty_print(num_epochs: int, batch_size: int, latest_file: Optional[str]) -> None
    Prints training configuration details.

plot_and_save_metrics(loss_vs_iteration: List[float], loss_vs_epoch: List[float], checkpoint_path: str, title: str, filename: str) -> None
    Plots and saves loss metrics over iterations and epochs.

train_operation(num_epochs: int, batch_size: int, save_checkpoint: int, checkpoint_path: str, latest_file: Optional[str], base_path: str, logs_path: str, train_val_split: float) -> None
    Performs the training operation for visual-inertial odometry.

main() -> None
    Parses command-line arguments and initiates the training process.

"""

# Standard library imports
import argparse
import csv
import os
from typing import List, Optional, Tuple, Union

# Third-party imports
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import lr_scheduler

# Local imports
from network import LossFn, Visual_Inertial_encoder
from utils import find_latest_model

# Set device for computation
device = "cuda"  # Use GPU if available, otherwise fallback to CPU


def generate_training_batch(
    base_path: str,
    batch_size: int,
    train_prob: float,
    start_idx: int,
    imu_data: np.ndarray,
    pose_data: np.ndarray,
) -> Tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    int,
]:
    """
    Generate a batch of training and testing data for visual-inertial odometry.

    This function reads image, IMU, and pose data from the dataset, processes it,
    and splits it into training and testing batches based on the specified probability.

    Parameters
    ----------
    base_path : str
        Path to the data folder containing images, IMU data, and ground truth.
    batch_size : int
        Number of samples in the mini-batch.
    train_prob : float
        Probability of selecting a sample for training.
    start_idx : int
        Starting index for data generation.
    imu_data : np.ndarray
        IMU data array.
    pose_data : np.ndarray
        Pose data array.

    Returns
    -------
    Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor],
          Optional[torch.Tensor], Optional[torch.Tensor], int]
        A tuple containing training and testing batches for images, IMU, and poses,
        along with the updated start index.
    """
    # Initialize empty lists for training and testing batches
    img_train_batch, img_test_batch = [], []
    imu_train_batch, imu_test_batch = [], []
    pose_train_batch, pose_test_batch = [], []

    train_count = int(train_prob * batch_size)  # Calculate number of training samples
    sample_count = 0  # Counter for processed samples

    while sample_count < batch_size and start_idx < 500:  # Limit to 500 samples
        rand_idx = start_idx  # Use start_idx as the current index

        # Construct file paths for consecutive images
        img1_path = os.path.join(base_path, f"{rand_idx}.png")
        img2_path = os.path.join(base_path, f"{rand_idx + 1}.png")

        # Extract IMU and pose data for the current index
        imu_sample = torch.from_numpy(imu_data[rand_idx * 10 : rand_idx * 10 + 10])
        pose_sample = torch.from_numpy(pose_data[rand_idx * 10])

        # Read and preprocess the first image
        img1 = cv2.imread(img1_path)  # Read the image
        img1 = cv2.resize(img1, (180, 320)).astype(
            np.float32
        )  # Resize and convert type

        # Read and preprocess the second image
        img2 = cv2.imread(img2_path)  # Read the image
        img2 = cv2.resize(img2, (180, 320)).astype(
            np.float32
        )  # Resize and convert type

        # Stack the two images along the channel dimension
        stacked_img = np.concatenate([img1, img2], axis=2).astype(np.float32)
        stacked_img = (
            np.transpose(stacked_img, (2, 0, 1)) / 255.0
        )  # Normalize to [0, 1]

        # Split data into training and testing batches
        if sample_count < train_count:
            img_train_batch.append(torch.from_numpy(stacked_img))  # Add to training set
            imu_train_batch.append(imu_sample)
            pose_train_batch.append(pose_sample)
        else:
            img_test_batch.append(torch.from_numpy(stacked_img))  # Add to testing set
            imu_test_batch.append(imu_sample)
            pose_test_batch.append(pose_sample)

        start_idx += 1  # Increment the index
        sample_count += 1  # Increment the sample counter

    # Convert lists to tensors, or None if the list is empty
    img_train_batch_tensor = torch.stack(img_train_batch) if img_train_batch else None
    img_test_batch_tensor = torch.stack(img_test_batch) if img_test_batch else None
    imu_train_batch_tensor = torch.stack(imu_train_batch) if imu_train_batch else None
    imu_test_batch_tensor = torch.stack(imu_test_batch) if imu_test_batch else None
    pose_train_batch_tensor = (
        torch.stack(pose_train_batch) if pose_train_batch else None
    )
    pose_test_batch_tensor = torch.stack(pose_test_batch) if pose_test_batch else None

    # Return the training and testing batches along with the updated index
    return (
        (
            img_train_batch_tensor.to(device)
            if img_train_batch_tensor is not None
            else None
        ),
        img_test_batch_tensor.to(device) if img_test_batch_tensor is not None else None,
        (
            imu_train_batch_tensor.to(device)
            if imu_train_batch_tensor is not None
            else None
        ),
        imu_test_batch_tensor.to(device) if imu_test_batch_tensor is not None else None,
        (
            pose_train_batch_tensor.to(device)
            if pose_train_batch_tensor is not None
            else None
        ),
        (
            pose_test_batch_tensor.to(device)
            if pose_test_batch_tensor is not None
            else None
        ),
        start_idx,
    )


def read_csv(file_path: str) -> np.ndarray:
    """
    Read a CSV file and return its contents as a numpy array.

    This function reads numerical data from a CSV file and converts it into a numpy array.

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
        reader = csv.reader(file)  # Create a CSV reader object
        file_data = [list(map(float, row)) for row in reader]  # Convert rows to floats
    return np.array(file_data)  # Return as a numpy array


def pretty_print(num_epochs: int, batch_size: int, latest_file: Optional[str]) -> None:
    """
    Print training configuration details.

    This function displays the number of epochs, batch size, and the latest checkpoint file (if any).

    Parameters
    ----------
    num_epochs : int
        Number of epochs for training.
    batch_size : int
        Size of the mini-batch.
    latest_file : str or None
        Name of the latest checkpoint file.
    """
    print("Number of Epochs Training will run for " + str(num_epochs))  # Print epochs
    print("Mini Batch Size " + str(batch_size))  # Print batch size
    if latest_file is not None:
        print(
            "Loading latest checkpoint with the name " + latest_file
        )  # Print checkpoint info


def plot_and_save_metrics(
    loss_vs_iteration: List[float],
    loss_vs_epoch: List[float],
    checkpoint_path: str,
    title: str,
    filename: str,
) -> None:
    """
    Plot and save loss metrics over iterations and epochs.

    This function generates plots for loss values over iterations and epochs,
    and saves the plots to the specified checkpoint path.

    Parameters
    ----------
    loss_vs_iteration : List[float]
        List of loss values over iterations.
    loss_vs_epoch : List[float]
        List of loss values over epochs.
    checkpoint_path : str
        Path to save the plot.
    title : str
        Title of the plot.
    filename : str
        Name of the file to save the plot.
    """
    plt.figure(figsize=(10, 5))  # Create a figure with specified size

    # Plot loss vs iteration
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(loss_vs_iteration) + 1), loss_vs_iteration, label="Iteration")
    plt.xlabel("Iteration")  # Label x-axis
    plt.ylabel("Loss")  # Label y-axis
    plt.title(f"{title} Loss vs Iteration")  # Set plot title
    plt.legend()  # Add legend

    # Plot loss vs epoch
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(loss_vs_epoch) + 1), loss_vs_epoch, label="Epoch")
    plt.xlabel("Epochs")  # Label x-axis
    plt.ylabel("Loss")  # Label y-axis
    plt.title(f"{title} Loss vs Epoch")  # Set plot title
    plt.legend()  # Add legend
    plt.tight_layout()  # Adjust layout

    # Save the plot to the specified path
    save_path = os.path.join(checkpoint_path, filename)
    plt.savefig(save_path)
    print(f"Figure saved at: {save_path}")  # Print save location
    plt.show()  # Display the plot


def train_operation(
    num_epochs: int,
    batch_size: int,
    save_checkpoint: int,
    checkpoint_path: str,
    latest_file: Optional[str],
    base_path: str,
    logs_path: str,
    train_val_split: float,
) -> None:
    """
    Perform training operation for visual-inertial odometry.

    This function initializes the model, loads pretrained weights or checkpoints,
    and iteratively trains the model over the specified number of epochs. It also
    saves checkpoints and logs metrics for training and validation.

    Parameters
    ----------
    num_epochs : int
        Number of epochs to train the model.
    batch_size : int
        Size of the mini-batch.
    save_checkpoint : int
        Frequency of saving checkpoints.
    checkpoint_path : str
        Path to save checkpoints.
    latest_file : str or None
        Path to the latest checkpoint file.
    base_path : str
        Base path of the dataset.
    logs_path : str
        Path to save logs.
    train_val_split : float
        Ratio of training to validation data.
    """
    # Initialize the model and move it to the computation device
    model = Visual_Inertial_encoder().to(device)

    # Load pretrained weights for the visual encoder
    pretrained_w = torch.load("./flownets_bn_EPE2.459.pth.tar", map_location="cpu")
    model_dict = model.visual.state_dict()  # Get model's visual encoder state dict
    update_dict = {
        k: v for k, v in pretrained_w["state_dict"].items() if k in model_dict
    }  # Filter matching keys
    model_dict.update(update_dict)  # Update model's state dict
    model.visual.load_state_dict(model_dict)  # Load updated state dict
    model = model.to(device)  # Move model to device

    # Load checkpoint if available
    if latest_file is not None:
        checkpoint = torch.load(latest_file + ".ckpt")  # Load checkpoint
        start_epoch = int("".join(c for c in latest_file.split("a")[0] if c.isdigit()))
        model.load_state_dict(checkpoint["model_state_dict"])  # Load model state
        print(f"Loaded latest checkpoint with the name {latest_file}....")
    else:
        start_epoch = 0  # Start from scratch
        print("New model initialized....")

    # Initialize lists to store loss metrics
    loss_vs_epoch_train: List[float] = []  # Training loss per epoch
    loss_vs_iteration_train: List[float] = []  # Training loss per iteration
    loss_vs_epoch_val: List[float] = []  # Validation loss per epoch
    loss_vs_iteration_val: List[float] = []  # Validation loss per iteration
    names = os.listdir(base_path)  # List all dataset folders

    # Iterate through epochs
    for epoch in range(start_epoch, num_epochs):
        # Adjust learning rate based on the epoch
        optimizer = torch.optim.Adam(
            model.parameters(), lr=5e-5 if epoch < 250 else 1e-6, weight_decay=5e-6
        )

        # Iterate through each dataset folder
        for name in names:
            basepath = os.path.join(base_path, name)  # Path to dataset folder
            imu_path = os.path.join(basepath, "IMU_data_file.csv")  # IMU data path
            pose_path = os.path.join(basepath, "Pose_data.csv")  # Pose data path
            imu_data = read_csv(imu_path)  # Read IMU data
            pose_data = read_csv(pose_path)  # Read pose data

            print(f"{name} Training Started")
            num_train_samples = len(
                [
                    filename
                    for filename in os.listdir(basepath)
                    if filename.endswith(".png")
                ]
            )  # Count the number of image samples
            num_iterations_per_epoch = int(num_train_samples / batch_size)
            start_idx = 0  # Reset start index for each dataset

            # Iterate through mini-batches
            for per_epoch_counter in range(num_iterations_per_epoch):
                # Generate training and testing batches
                (
                    img_train_batch,
                    img_test_batch,
                    imu_train_batch,
                    imu_test_batch,
                    pose_train_batch,
                    pose_test_batch,
                    start_idx,
                ) = generate_training_batch(
                    basepath,
                    batch_size,
                    train_val_split,
                    start_idx,
                    imu_data,
                    pose_data,
                )

                # Training step
                if (
                    img_train_batch is not None
                    and imu_train_batch is not None
                    and pose_train_batch is not None
                ):
                    model.train()  # Set model to training mode
                    pose_train_predicted = model(
                        img_train_batch.float(), imu_train_batch.float()
                    ).float()  # Forward pass
                    pose_train_predicted = pose_train_predicted.to(device)

                    loss_pose_train = LossFn(
                        pose_train_predicted, pose_train_batch.float()
                    ).float()  # Compute loss
                    loss_pose_train = loss_pose_train.to(device)

                    loss_xyz = (
                        loss_pose_train.detach().cpu().numpy()
                    )  # Detach loss for logging
                    loss_vs_iteration_train.append(loss_xyz)

                    optimizer.zero_grad()  # Zero gradients
                    loss_pose_train.mean().backward()  # Backpropagation
                    optimizer.step()  # Update weights

                    # Save checkpoint periodically
                    if per_epoch_counter % save_checkpoint == 0:
                        save_name = (
                            checkpoint_path
                            + str(epoch)
                            + "a"
                            + str(per_epoch_counter)
                            + "model.ckpt"
                        )
                        torch.save(
                            {
                                "epoch": epoch,
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "loss": loss_pose_train,
                            },
                            save_name,
                        )
                        print("\n" + save_name + " Model Saved...")

                # Validation step
                if (
                    img_test_batch is not None
                    and imu_test_batch is not None
                    and pose_test_batch is not None
                ):
                    model.eval()  # Set model to evaluation mode
                    with torch.no_grad():
                        loss_pose_val = model.validation_step(
                            img_test_batch.float(),
                            imu_test_batch.float(),
                            pose_test_batch.float(),
                        ).float()  # Compute validation loss
                    loss_vs_iteration_val.append(loss_pose_val.detach().cpu().numpy())

            # Compute average loss for the epoch
            average_epoch_loss_train = (
                sum(loss_vs_iteration_train[-num_iterations_per_epoch:])
                / num_iterations_per_epoch
            )
            loss_vs_epoch_train.append(average_epoch_loss_train)
            average_epoch_loss_val = (
                sum(loss_vs_iteration_val[-num_iterations_per_epoch:])
                / num_iterations_per_epoch
            )
            loss_vs_epoch_val.append(average_epoch_loss_val)

            # Save model at the end of the epoch
            save_name = checkpoint_path + str(epoch) + "model.ckpt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss_train": loss_pose_train,
                    "loss_val": loss_pose_val,
                },
                save_name,
            )
            print("\n" + save_name + " Model Saved...")

        print(f"{name} Training Ended")

    # Plot and save training and validation metrics
    plot_and_save_metrics(
        loss_vs_iteration_train,
        loss_vs_epoch_train,
        checkpoint_path,
        "Train",
        "train_metrics.png",
    )
    plot_and_save_metrics(
        loss_vs_iteration_val,
        loss_vs_epoch_val,
        checkpoint_path,
        "Validation",
        "val_metrics.png",
    )


def main() -> None:
    """
    Main function to parse arguments and run the training operation.

    This function parses command-line arguments for training configuration,
    sets up paths for checkpoints and logs, and initiates the training process.
    """
    parser = argparse.ArgumentParser()  # Initialize argument parser
    parser.add_argument(
        "--BasePath", default="./Data", help="Base path of images, Default:../Data"
    )
    parser.add_argument(
        "--CheckPointPath",
        default="Checkpoints_Visual_Inertial/",
        help="Path to save Checkpoints, Default: Checkpoints/",
    )
    parser.add_argument(
        "--NumEpochs",
        type=int,
        default=300,
        help="Number of Epochs to Train for, Default:30",
    )
    parser.add_argument(
        "--MiniBatchSize",
        type=int,
        default=15,
        help="Size of the MiniBatch to use, Default:1",
    )
    parser.add_argument(
        "--LoadCheckPoint",
        type=int,
        default=0,
        help="Load Model from latest Checkpoint from CheckPointsPath?, Default:0",
    )
    parser.add_argument(
        "--LogsPath",
        default="Logs_Visual_Inertial/",
        help="Path to save Logs for Tensorboard, Default=Logs/",
    )
    parser.add_argument(
        "--Split", type=float, default=0.8, help="Train Validation Split"
    )

    args = parser.parse_args()  # Parse command-line arguments
    num_epochs = args.NumEpochs  # Number of epochs
    base_path = args.BasePath  # Base path for dataset
    batch_size = args.MiniBatchSize  # Mini-batch size
    load_checkpoint = args.LoadCheckPoint  # Whether to load checkpoint
    checkpoint_path = args.CheckPointPath  # Path to save checkpoints
    logs_path = args.LogsPath  # Path to save logs
    train_val_split = args.Split  # Train-validation split ratio

    # Create directories if they don't exist
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    if not os.path.exists(logs_path):
        os.makedirs(logs_path)

    # Find the latest checkpoint if loading is enabled
    latest_file = find_latest_model(checkpoint_path) if load_checkpoint == 1 else None
    save_checkpoint = 100  # Frequency of saving checkpoints

    pretty_print(num_epochs, batch_size, latest_file)  # Print training configuration

    # Start the training operation
    train_operation(
        num_epochs,
        batch_size,
        save_checkpoint,
        checkpoint_path,
        latest_file,
        base_path,
        logs_path,
        train_val_split,
    )


if __name__ == "__main__":
    main()  # Entry point for the script
