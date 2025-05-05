"""
Train Visual Odometry (VO) models.

This module provides functions and operations for training visual-only odometry models.
It includes functionalities for:
- Data generation
- Model training
- Loss visualization

The module is designed to work with image data and pose data for odometry tasks.

Functions
---------
generate_visual_training_batch(base_path, batch_size, train_prob, start_idx, pose_data)
    Splits image and pose data into training and testing batches based on the specified probability and batch size.

read_csv(file_path)
    Reads a CSV file and returns its contents as a numpy array.

pretty_print(num_epochs, batch_size, latest_file)
    Prints the training configuration details, including the number of epochs, batch size, and the latest checkpoint file.

plot_and_save_metrics(loss_vs_iteration, loss_vs_epoch, checkpoint_path, title, filename)
    Plots and saves loss metrics over iterations and epochs.

train_visual_odometry(num_epochs, batch_size, save_checkpoint, checkpoint_path, latest_file, base_path, logs_path, train_val_split)
    Initializes the model, loads checkpoints (if available), and trains the model over the specified number of epochs.

main()
    Parses command-line arguments, sets up paths for checkpoints and logs, and initiates the training process.
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
from network import LossFn, Visual_encoder
from utils import find_latest_model

# Set device for computation
device = "cuda"  # Use GPU if available


def generate_visual_training_batch(
    base_path: str,
    batch_size: int,
    train_prob: float,
    start_idx: int,
    pose_data: np.ndarray,
) -> Tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    int,
]:
    """
    Generate a batch of training and testing data for visual-only odometry.

    Parameters
    ----------
    base_path : str
        Path to the data folder containing images and ground truth.
    batch_size : int
        Number of samples in the mini-batch.
    train_prob : float
        Probability of selecting a sample for training.
    start_idx : int
        Starting index for data generation.
    pose_data : np.ndarray
        Pose data array.

    Returns
    -------
    Tuple[Optional[torch.Tensor], Optional[torch.Tensor],
          Optional[torch.Tensor], Optional[torch.Tensor], int]
        A tuple containing training and testing batches for images and poses,
        along with the updated start index.
    """
    # Initialize lists to store training and testing data
    img_train_batch, img_test_batch = [], []
    pose_train_batch, pose_test_batch = [], []

    # Calculate the number of training samples in the batch
    train_count = int(train_prob * batch_size)
    sample_count = 0

    # Loop to generate batches until the batch size is met or data limit is reached
    while sample_count < batch_size and start_idx < 500:
        rand_idx = start_idx  # Use the current index as the random index

        # Construct file paths for consecutive image pairs
        img1_path = os.path.join(base_path, f"{rand_idx}.png")
        img2_path = os.path.join(base_path, f"{rand_idx + 1}.png")
        # Extract the corresponding pose sample
        pose_sample = torch.from_numpy(pose_data[rand_idx * 10])

        # Read and preprocess the first image
        img1 = cv2.imread(img1_path)
        img1 = cv2.resize(img1, (180, 320)).astype(np.float32)

        # Read and preprocess the second image
        img2 = cv2.imread(img2_path)
        img2 = cv2.resize(img2, (180, 320)).astype(np.float32)

        # Stack the two images along the channel dimension and normalize
        stacked_img = np.concatenate([img1, img2], axis=2).astype(np.float32)
        stacked_img = np.transpose(stacked_img, (2, 0, 1)) / 255.0

        # Assign the sample to training or testing batch based on the count
        if sample_count < train_count:
            img_train_batch.append(torch.from_numpy(stacked_img))
            pose_train_batch.append(pose_sample)
        else:
            img_test_batch.append(torch.from_numpy(stacked_img))
            pose_test_batch.append(pose_sample)

        # Increment the index and sample count
        start_idx += 1
        sample_count += 1

    # Convert lists to tensors and move them to the computation device
    img_train_batch_tensor = torch.stack(img_train_batch) if img_train_batch else None
    img_test_batch_tensor = torch.stack(img_test_batch) if img_test_batch else None
    pose_train_batch_tensor = (
        torch.stack(pose_train_batch) if pose_train_batch else None
    )
    pose_test_batch_tensor = torch.stack(pose_test_batch) if pose_test_batch else None

    # Return the generated batches and updated start index
    return (
        (
            img_train_batch_tensor.to(device)
            if img_train_batch_tensor is not None
            else None
        ),
        img_test_batch_tensor.to(device) if img_test_batch_tensor is not None else None,
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

    Parameters
    ----------
    file_path : str
        Path to the CSV file.

    Returns
    -------
    np.ndarray
        Array containing the CSV data.
    """
    # Open the CSV file and read its contents
    with open(file_path, mode="r") as file:
        reader = csv.reader(file)
        # Convert each row of the CSV into a list of floats
        file_data = [list(map(float, row)) for row in reader]

    # Convert the data into a numpy array and return it
    return np.array(file_data)


def pretty_print(num_epochs: int, batch_size: int, latest_file: Optional[str]) -> None:
    """
    Print training configuration details.

    Parameters
    ----------
    num_epochs : int
        Number of epochs for training.
    batch_size : int
        Size of the mini-batch.
    latest_file : str or None
        Name of the latest checkpoint file.
    """
    # Print the number of epochs and batch size
    print("Number of Epochs Training will run for " + str(num_epochs))
    print("Mini Batch Size " + str(batch_size))

    # Print the name of the latest checkpoint file if it exists
    if latest_file is not None:
        print("Loading latest checkpoint with the name " + latest_file)


def plot_and_save_metrics(
    loss_vs_iteration: List[float],
    loss_vs_epoch: List[float],
    checkpoint_path: str,
    title: str,
    filename: str,
) -> None:
    """
    Plot and save loss metrics over iterations and epochs.

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
    # Create a figure with two subplots for iteration and epoch losses
    plt.figure(figsize=(10, 5))

    # Plot loss vs iteration
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(loss_vs_iteration) + 1), loss_vs_iteration, label="Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(f"{title} Loss vs Iteration")
    plt.legend()

    # Plot loss vs epoch
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(loss_vs_epoch) + 1), loss_vs_epoch, label="Epoch")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{title} Loss vs Epoch")
    plt.legend()
    plt.tight_layout()

    # Save the plot to the specified path and display it
    save_path = os.path.join(checkpoint_path, filename)
    plt.savefig(save_path)  # Save the plot as an image file
    print(f"Figure saved at: {save_path}")
    plt.show()  # Display the plot


def train_visual_odometry(
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
    Perform training operation for visual-only odometry.

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
    model = Visual_encoder().to(device)

    # Load pretrained weights for the model
    pretrained_weights = torch.load(
        "./flownets_bn_EPE2.459.pth.tar", map_location="cpu"
    )
    model_dict = model.state_dict()
    # Update the model's state dictionary with the pretrained weights
    update_dict = {
        k: v for k, v in pretrained_weights["state_dict"].items() if k in model_dict
    }
    model_dict.update(update_dict)
    model.load_state_dict(model_dict)
    model = model.to(device)

    # Load checkpoint if available
    if latest_file is not None:
        checkpoint = torch.load(latest_file + ".ckpt")
        # Extract the starting epoch from the checkpoint filename
        start_epoch = int("".join(c for c in latest_file.split("a")[0] if c.isdigit()))
        model.load_state_dict(checkpoint["model_state_dict"])  # Load model weights
        print(f"Loaded latest checkpoint with the name {latest_file}....")
    else:
        start_epoch = 0  # Start from scratch if no checkpoint is available
        print("New model initialized....")

    # Initialize lists to store loss metrics
    loss_vs_epoch_train: List[float] = []
    loss_vs_iteration_train: List[float] = []
    loss_vs_epoch_val: List[float] = []
    loss_vs_iteration_val: List[float] = []

    # Iterate through dataset folders
    dataset_names = os.listdir(base_path)  # List all subdirectories in the base path
    for epoch in range(start_epoch, num_epochs):
        # Adjust learning rate based on the epoch
        optimizer = torch.optim.Adam(
            model.parameters(), lr=5e-5 if epoch < 250 else 1e-6, weight_decay=5e-6
        )

        for dataset_name in dataset_names:
            dataset_path = os.path.join(base_path, dataset_name)  # Path to the dataset
            pose_path = os.path.join(dataset_path, "Pose_data.csv")  # Pose data file
            pose_data = read_csv(pose_path)  # Read pose data

            print(f"{dataset_name} Training Started")
            # Count the number of training samples (e.g., images)
            num_train_samples = len(
                [
                    filename
                    for filename in os.listdir(dataset_path)
                    if filename.endswith(".png")
                ]
            )
            # Calculate the number of iterations per epoch
            num_iterations_per_epoch = int(num_train_samples / batch_size)
            start_idx = 0  # Initialize the starting index for data generation

            # Iterate through training iterations for each epoch
            for iteration in range(num_iterations_per_epoch):
                # Generate training and testing batches
                (
                    img_train_batch,
                    img_test_batch,
                    pose_train_batch,
                    pose_test_batch,
                    start_idx,
                ) = generate_visual_training_batch(
                    dataset_path, batch_size, train_val_split, start_idx, pose_data
                )

                # Perform training step
                if img_train_batch is not None and pose_train_batch is not None:
                    model.train()  # Set the model to training mode
                    pose_train_predicted = model(img_train_batch.float()).float()
                    # Compute the training loss
                    loss_pose_train = LossFn(
                        pose_train_predicted, pose_train_batch.float()
                    ).float()
                    # Append the loss to the iteration loss list
                    loss_vs_iteration_train.append(
                        loss_pose_train.detach().cpu().numpy()
                    )

                    optimizer.zero_grad()  # Reset gradients
                    loss_pose_train.mean().backward()  # Backpropagation
                    optimizer.step()  # Update model parameters

                    # Save model checkpoint periodically
                    if iteration % save_checkpoint == 0:
                        save_name = (
                            checkpoint_path
                            + str(epoch)
                            + "a"
                            + str(iteration)
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

                # Perform validation step
                if img_test_batch is not None and pose_test_batch is not None:
                    model.eval()  # Set the model to evaluation mode
                    with torch.no_grad():  # Disable gradient computation
                        loss_pose_val = model.validation_step(
                            img_test_batch, pose_test_batch
                        )
                    # Append the validation loss to the iteration loss list
                    loss_vs_iteration_val.append(loss_pose_val.detach().cpu().numpy())

            # Calculate and log average loss for the epoch
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

            # Save model checkpoint at the end of the epoch
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
        print(f"{dataset_name} Training Ended")

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

    This function parses command-line arguments, sets up paths for checkpoints
    and logs, and initiates the training process.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    # Define arguments for paths, epochs, batch size, etc.
    parser.add_argument(
        "--BasePath", default="./Data", help="Base path of images, Default:../Data"
    )
    parser.add_argument(
        "--CheckPointPath",
        default="Checkpoints_Visual/",
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
        default="Logs_Visual/",
        help="Path to save Logs for Tensorboard, Default=Logs/",
    )
    parser.add_argument(
        "--Split", type=float, default=0.8, help="Train Validation Split"
    )

    args = parser.parse_args()
    num_epochs = args.NumEpochs
    base_path = args.BasePath
    batch_size = args.MiniBatchSize
    load_checkpoint = args.LoadCheckPoint
    checkpoint_path = args.CheckPointPath
    logs_path = args.LogsPath
    train_val_split = args.Split

    # Create directories for checkpoints and logs if they don't exist
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    if not os.path.exists(logs_path):
        os.makedirs(logs_path)

    # Find the latest checkpoint file if loading from a checkpoint
    latest_file = find_latest_model(checkpoint_path) if load_checkpoint == 1 else None
    save_checkpoint = 100

    # Print training configuration
    pretty_print(num_epochs, batch_size, latest_file)

    # Start the training process
    train_visual_odometry(
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
    main()
