#!/usr/bin/env python
"""
Wrapper.py

This script implements NeRF pipeline for rendering 3D scenes from 2D images.
It includes functions for loading datasets, generating rays, sampling points along rays, rendering images,
and training/testing the NeRF model.

Modules:
--------
- load_dataset(data_path, mode): Load the dataset for training or testing.
- ray_pixel_convert(camera_info, pose, pixelPosition, near, far, args): Convert a pixel position to a ray in 3D space.
- ray_sampling(ray_origin, ray_direction, near, far, n_samples): Sample points along a ray using stratified sampling.
- batch_generate(images, poses, camera_info, args): Generate a batch of rays and their corresponding ground truth RGB values.
- render(model, ray_origin, ray_direction, args): Render RGB values for input rays using the NeRF model.
- loss(groundtruth, prediction): Compute the Mean Squared Error (MSE) loss.
- train(images, poses, camera_info, args): Train the NeRF model with mixed precision and save checkpoints.
- render_test_image(model, pose, camera_info, args): Render a test image for visualization during training.
- test(images, poses, camera_info, args): Test the NeRF model on the test dataset, compute PSNR and SSIM, and save results as a GIF.
- main(args): Main function to load data and start training or testing.
- parser_config(): Configure the argument parser for the script.

Attributes:
-----------
- device : torch.device
    Specifies whether to use GPU or CPU for computations.
- np.random.seed(0) : Sets the random seed for reproducibility.

Usage:
------
Run the script with appropriate command-line arguments to train or test the NeRF model.
"""

import argparse
import os
import random
import sys

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from DataLoader import *
from NeRFModel import NeRFmodel
from torch.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import StructuralSimilarityIndexMeasure
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)

# Prevent __pycache__ generation
sys.dont_write_bytecode = True


def load_dataset(data_path, mode):
    """
    Load the dataset for training or testing.

    Parameters
    ----------
    data_path : str
        Path to the dataset.
    mode : str
        Mode of operation, either 'train' or 'test'.

    Returns
    -------
    tuple
        camera_info : tuple
            Image width, height, and camera matrix (focal length).
        images : list
            List of images in the dataset.
        pose : list
            Corresponding camera poses in the world frame.
    """
    # images, poses, camera_info (W, H, focal)
    return DataLoader(data_path).load_dataset(mode)


def ray_pixel_convert(camera_info, pose, pixelPosition, near, far, args):
    """
    Convert a pixel position in the image to a ray in 3D space.

    Parameters
    ----------
    camera_info : tuple
        A tuple containing the image width (W), height (H), and focal length (focal).
    pose : torch.Tensor
        A 4x4 transformation matrix representing the camera's pose in the world frame.
    pixelPosition : tuple
        The (x, y) coordinates of the pixel in the image.
    near : float
        Near clipping plane distance (not used in this function but included for consistency).
    far : float
        Far clipping plane distance (not used in this function but included for consistency).
    args : argparse.Namespace
        Additional arguments for ray sampling (not used in this function but included for consistency).

    Returns
    -------
    tuple
        ray_origin : torch.Tensor
            The origin of the ray in 3D space, which is the camera's position in the world frame.
        ray_direction : torch.Tensor
            The normalized direction of the ray in 3D space.
    """
    # Ensure pose is a PyTorch tensor
    pose = torch.tensor(pose, dtype=torch.float32, device=device)
    pose_tensor = torch.tensor(pose, dtype=torch.float32, device=device)

    # Generate meshgrid for pixel coordinates
    mesh_x, mesh_y = torch.meshgrid(
        torch.linspace(0, W - 1, W).to(device),
        torch.linspace(0, H - 1, H).to(device),
        indexing="ij",
    )

    # Normalize pixel coordinates to camera space
    x = (mesh_x - W / 2) / focal
    y = (mesh_y - H / 2) / focal

    # Create direction vectors in the camera's local coordinate system
    directions = torch.stack((x, -y, -torch.ones_like(x)), dim=-1)

    # Extract rotation (R) and translation (T) components from the camera pose
    rotation = pose[:3, :3]
    rotation = pose_tensor[:3, :3]
    translation = pose_tensor[:3, -1].view(1, 1, 3)
    # Transform the direction vectors from camera space to world space
    ray_direction = torch.sum(directions[..., None, :] * rotation, dim=-1)
    ray_direction = ray_direction / torch.linalg.norm(
        ray_direction, dim=-1, keepdim=True
    )

    # The ray's origin is the camera's position in the world frame
    ray_origin = translation.expand(ray_direction.shape[0], ray_direction.shape[1], -1)

    x, y = pixelPosition
    x = (x - W / 2) / focal
    y = (y - H / 2) / focal

    # Create direction vector for the single pixel in the camera's local coordinate system
    direction = torch.tensor([x, -y, -1.0], dtype=torch.float32, device=device)

    # Transform the direction vector from camera space to world space
    ray_direction = torch.matmul(rotation, direction)
    ray_direction = ray_direction / torch.linalg.norm(ray_direction)

    # The ray's origin is the camera's position in the world frame
    ray_origin = translation.squeeze()

    return ray_origin, ray_direction


def batch_generate(images, poses, camera_info, args):
    """
    Generate a batch of rays and their corresponding ground truth RGB values.

    Parameters
    ----------
    images : list
        All images in the dataset.
    poses : list
        Corresponding camera poses in the world frame.
    camera_info : tuple
        Image width, height, and camera matrix (focal length).
    args : argparse.Namespace
        Arguments containing batch size and other configurations.

    Returns
    -------
    tuple
        rays_o : torch.Tensor
            Origins of the rays.
        rays_d : torch.Tensor
            Directions of the rays.
        rgb_gt : torch.Tensor
            Ground truth RGB values for the rays.
    """
    W, H, focal = camera_info

    # Randomly select an image
    img_idx = random.randint(0, len(images) - 1)
    image = images[img_idx]
    pose = poses[img_idx]

    # Sample random pixels from the image
    n_rays = args.n_rays_batch
    x_coords = torch.randint(0, W, (n_rays,), device=device)
    y_coords = torch.randint(0, H, (n_rays,), device=device)

    # Initialize tensors for rays and ground truth RGB values
    rays_o = torch.zeros((n_rays, 3), dtype=torch.float32, device=device)
    rays_d = torch.zeros((n_rays, 3), dtype=torch.float32, device=device)
    rgb_gt = torch.zeros((n_rays, 3), dtype=torch.float32, device=device)

    # Generate rays for each sampled pixel
    for i in range(n_rays):
        # Get ray origin and direction for this pixel
        ray_o, ray_d = ray_pixel_convert(
            camera_info, pose, (x_coords[i].item(), y_coords[i].item()), 0, 0, args
        )
        rays_o[i] = ray_o
        rays_d[i] = ray_d

        # Get ground truth RGB value for this pixel
        rgb_gt[i] = torch.tensor(
            image[y_coords[i], x_coords[i]], dtype=torch.float32, device=device
        )

    return rays_o, rays_d, rgb_gt


# Run neural network on various points along the ray and find the color of the ray
def render(model, ray_origin, ray_direction, args):
    """
    Render RGB values for input rays using the NeRF model.

    Parameters
    ----------
    model : NeRFmodel
        Neural Radiance Field (NeRF) model.
    ray_origin : torch.Tensor
        Origins of the input rays.
    ray_direction : torch.Tensor
        Directions of the input rays.
    args : argparse.Namespace
        Additional arguments for rendering.

    Returns
    -------
    torch.Tensor
        RGB values of the input rays.
    """
    n_rays = ray_origin.shape[0]
    n_samples = args.n_sample
    near, far = 2.0, 6.0  # Typical values for synthetic NeRF datasets

    # Stratified sampling along the ray
    t_vals = torch.linspace(near, far, n_samples).to(device)
    t_vals = t_vals.expand(n_rays, n_samples)
    random_offsets = torch.rand(n_rays, n_samples).to(device) * (far - near) / n_samples
    t_vals += random_offsets

    # Compute 3D query points along the rays
    query_points = ray_origin.unsqueeze(1) + ray_direction.unsqueeze(
        1
    ) * t_vals.unsqueeze(-1)
    query_points_flat = query_points.view(-1, 3)

    # Expand ray directions for all samples
    view_dirs = ray_direction.unsqueeze(1).expand(-1, n_samples, -1).reshape(-1, 3)

    # Predict RGB and density for each point along the ray
    rgb, sigma = model(query_points_flat, view_dirs)
    rgb = rgb.view(n_rays, n_samples, 3)
    sigma = sigma.view(n_rays, n_samples)

    # Volume rendering
    deltas = t_vals[:, 1:] - t_vals[:, :-1]
    deltas = torch.cat(
        [deltas, torch.tensor([1e10]).expand(n_rays, 1).to(device)], dim=-1
    )

    alpha = 1.0 - torch.exp(-sigma * deltas)
    transmittance = torch.cumprod(
        torch.cat([torch.ones((n_rays, 1)).to(device), 1.0 - alpha + 1e-10], dim=-1),
        dim=-1,
    )[:, :-1]
    weights = alpha * transmittance

    # Compute final RGB values
    rgb_final = torch.sum(weights.unsqueeze(-1) * rgb, dim=1)

    return rgb_final


def loss(groundtruth, prediction):
    """
    Compute the Mean Squared Error (MSE) loss.

    Parameters
    ----------
    groundtruth : torch.Tensor
        Ground truth RGB values.
    prediction : torch.Tensor
        Predicted RGB values.

    Returns
    -------
    torch.Tensor
        MSE loss value.
    """
    # Calculate the mean squared difference between ground truth and predicted values
    return torch.mean((groundtruth - prediction) ** 2)


def train(images, poses, camera_info, args):
    """
    Train the NeRF model.

    Parameters
    ----------
    images : list
        All images in the dataset.
    poses : list
        Corresponding camera poses in the world frame.
    camera_info : tuple
        Image width, height, and camera matrix (focal length).
    args : argparse.Namespace
        Arguments containing training configurations.
    """
    # Create an instance of the NeRF model and move it to the appropriate device
    model = NeRFmodel().to(device)

    # Set up the optimizer with the specified learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lrate))

    # Create a directory for logs if it doesn't exist
    os.makedirs(args.logs_path, exist_ok=True)

    # Initialize a TensorBoard SummaryWriter for logging training metrics
    writer = SummaryWriter(log_dir=args.logs_path)

    # Load checkpoint if specified
    start_iter = 0
    if args.load_checkpoint:
        checkpoint_path = os.path.join(args.checkpoint_path, "model.pth")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_iter = checkpoint["iteration"]
            print(f"Loaded checkpoint from iteration {start_iter}")

    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_path, exist_ok=True)

    scaler = torch.amp.GradScaler("cuda")  # Initialize GradScaler for mixed precision

    # Training loop
    pbar = tqdm(range(start_iter, int(args.max_iters)))
    for i in pbar:
        # Generate batch of rays
        rays_o, rays_d, rgb_gt = batch_generate(images, poses, camera_info, args)

        with torch.amp.autocast("cuda"):  # Enable autocast for mixed precision
            # Forward pass
            rgb_pred = render(model, rays_o, rays_d, args)

            # Calculate loss
            loss_val = loss(rgb_gt, rgb_pred)

        # Backward pass and optimization
        optimizer.zero_grad()
        scaler.scale(loss_val).backward()  # Scale the loss for mixed precision
        scaler.step(optimizer)  # Step the optimizer
        scaler.update()  # Update the scaler

        # Log progress
        pbar.set_description(f"Loss: {loss_val.item():.4f}")
        writer.add_scalar("Loss/train", loss_val.item(), i)

        # Save checkpoint
        if (i + 1) % int(args.save_ckpt_iter) == 0:
            checkpoint = {
                "iteration": i + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss_val.item(),
            }
            torch.save(checkpoint, os.path.join(args.checkpoint_path, "model.pth"))
            print(f"Saved checkpoint at iteration {i + 1}")

            # Generate and save a test image
            if i > 0:
                with torch.no_grad():
                    test_image = render_test_image(model, poses[0], camera_info, args)

                os.makedirs(args.images_path, exist_ok=True)
                plt.figure()
                plt.imshow(test_image.cpu().numpy())
                plt.savefig(os.path.join(args.images_path, f"iter_{i+1}.png"))
                plt.close()

    writer.close()


def render_test_image(model, pose, camera_info, args):
    """
    Render a test image for visualization during training.

    Parameters
    ----------
    model : NeRFmodel
        Neural Radiance Field (NeRF) model used for rendering.
    pose : ndarray
        Camera pose in the world frame, represented as a 4x4 transformation matrix.
    camera_info : tuple
        A tuple containing the image width (W), height (H), and focal length (focal).
    args : argparse.Namespace
        Additional arguments for rendering, such as the number of samples per ray.

    Returns
    -------
    torch.Tensor
        Rendered test image as a tensor of shape (H_test, W_test, 3), where H_test and W_test
        are the downscaled height and width of the image.
    """
    # Unpack camera information
    W, H, _ = camera_info

    # Downscale the resolution for faster rendering during training
    W_test, H_test = W // 4, H // 4

    # Initialize an empty tensor to store the rendered image
    img = torch.zeros((H_test, W_test, 3)).to(device)

    # Loop through each pixel in the downscaled image
    for j in range(H_test):
        for i in range(W_test):
            # Map the pixel coordinates to the original resolution
            x, y = i * 4, j * 4

            # Generate the ray origin and direction for the current pixel
            ray_o, ray_d = ray_pixel_convert(camera_info, pose, (x, y), 0, 0, args)

            # Convert ray origin and direction to tensors and add a batch dimension
            ray_o = torch.tensor(ray_o, dtype=torch.float32).to(device).unsqueeze(0)
            ray_d = torch.tensor(ray_d, dtype=torch.float32).to(device).unsqueeze(0)

            # Render the RGB value for the current ray
            rgb = render(model, ray_o, ray_d, args)

            # Assign the rendered RGB value to the corresponding pixel in the image
            img[j, i] = rgb[0]

    # Return the rendered test image
    return img


def test(images, poses, camera_info, args):
    """
    Test the NeRF model and generate predictions.

    Parameters
    ----------
    images : torch.Tensor
        Test images.
    poses : torch.Tensor
        Corresponding camera poses.
    camera_info : tuple
        Image width, height, and focal length.
    args : argparse.Namespace
        Testing configurations.

    Returns
    -------
    None
    """
    if args.load_checkpoint:
        # Load the NeRF model and its pre-trained weights from the specified checkpoint
        model = NeRFmodel().to(device)
        model.load_state_dict(
            torch.load(
                os.path.join(args.checkpoint_path, "model.pt"), map_location=device
            )
        )

    # Extract camera parameters: height, width, and focal length
    H, W, _ = camera_info
    model.eval()  # Set the model to evaluation mode

    # Initialize lists to store PSNR and SSIM values for evaluation
    PSNRs = []
    SSIMs = []

    # Define functions to calculate PSNR and SSIM
    def PSNR(ground_truth, prediction):
        """
        Compute the Peak Signal-to-Noise Ratio (PSNR).

        Parameters
        ----------
        ground_truth : numpy.ndarray
            Ground truth image.
        prediction : numpy.ndarray
            Predicted image.

        Returns
        -------
        float
            PSNR value.
        """
        ground_truth = torch.tensor(ground_truth)
        prediction = torch.tensor(prediction)
        mse = torch.mean((ground_truth - prediction) ** 2)
        return 10 * torch.log10(1.0 / mse)

    def SSIM(ground_truth, prediction):
        """
        Compute the Structural Similarity Index Measure (SSIM).

        Parameters
        ----------
        ground_truth : numpy.ndarray
            Ground truth image.
        prediction : numpy.ndarray
            Predicted image.

        Returns
        -------
        float
            SSIM value.
        """
        # Convert ground truth and predicted images to tensors and adjust dimensions
        # Permute the dimensions to match the format (C, H, W) and add a batch dimension
        ground_truth = torch.tensor(ground_truth).permute(2, 0, 1).unsqueeze(0)
        prediction = torch.tensor(prediction).permute(2, 0, 1).unsqueeze(0)

        # Initialize the Structural Similarity Index Measure (SSIM) metric
        # The data_range parameter specifies the range of pixel values (0 to 1 in this case)
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

        # Compute and return the SSIM value between the predicted and ground truth images
        return ssim(prediction, ground_truth)

    with torch.no_grad():  # Disable gradient computation for testing
        # Generate rays and ground truth colors for the test dataset
        test_ray_origins, test_ray_directions, test_ground_truth = batch_generate(
            images, poses, camera_info, args
        )
        # Calculate the number of test images and rays per image
        num_images = test_ground_truth.shape[0] // (H * W)
        num_rays_per_image = H * W
        frames = []  # List to store frames for GIF creation

        for index in range(num_images):
            print(f"Testing on image: {index}")

            if index == 60:  # Render and evaluate only the 60th image
                # Render the predicted image using the NeRF model
                predicted_image = (
                    render(
                        model,
                        test_ray_origins[
                            num_rays_per_image
                            * index : num_rays_per_image
                            * (index + 1)
                        ],
                        test_ray_directions[
                            num_rays_per_image
                            * index : num_rays_per_image
                            * (index + 1)
                        ],
                        args,
                    )
                    .view(H, W, 3)
                    .cpu()
                    .detach()
                    .numpy()
                )
                # Extract the ground truth image for comparison
                groundtruth_image = (
                    test_ground_truth[
                        index * num_rays_per_image : (index + 1) * num_rays_per_image
                    ]
                    .view(H, W, 3)
                    .cpu()
                    .detach()
                    .numpy()
                )

                # Append the predicted image to the frames list for GIF creation
                frames.append((255 * predicted_image).astype(np.uint8))

                # Calculate PSNR (Peak Signal-to-Noise Ratio) for the rendered image
                psnr = PSNR(groundtruth_image, predicted_image)
                PSNRs.append(psnr)

                # Calculate SSIM (Structural Similarity Index Measure) for the rendered image
                ssim = SSIM(groundtruth_image, predicted_image)
                SSIMs.append(ssim)

                if args.plot:  # If plotting is enabled, visualize the results
                    # Plot the ground truth and predicted images side by side
                    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

                    ax[0].imshow(groundtruth_image)
                    ax[0].set_title("Original Test Image")
                    ax[0].axis("off")  # Hide axes ticks

                    ax[1].imshow(predicted_image)
                    ax[1].set_title("Predicted Test Image")
                    ax[1].axis("off")  # Hide axes ticks

                    plt.show()

    # Compute and print the average PSNR and SSIM values across all test images
    print(f"Average PSNR: {torch.mean(torch.tensor(PSNRs))}")
    print(f"Average SSIM: {torch.mean(torch.tensor(SSIMs))}")

    # Save the rendered frames as an animated GIF in the same folder as images
    os.makedirs(args.images_path, exist_ok=True)
    gif_filename = os.path.join(args.images_path, "output.gif")
    imageio.mimsave(gif_filename, frames, fps=30)

    print(f"GIF saved as {gif_filename}")


def main(args):
    """
    Main function to load data and start training or testing.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.
    """
    # load data
    print("Loading data...")
    images, poses, camera_info = load_dataset(args.data_path, args.mode)

    if args.mode == "train":
        print("Start training")
        train(images, poses, camera_info, args)
    elif args.mode == "test":
        print("Start testing")
        args.load_checkpoint = True
        test(images, poses, camera_info, args)


def parser_config():
    """
    Configure the argument parser for the script.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="lego",
        help="Path to the dataset. 'lego' is a sample dataset used for demonstration purposes.",
    )
    parser.add_argument("--mode", default="train", help="train/test/val")
    parser.add_argument("--lrate", default=5e-4, help="training learning rate")
    parser.add_argument(
        "--n_pos_freq",
        default=10,
        help="number of positional encoding frequencies for position",
    )
    parser.add_argument(
        "--n_dirc_freq",
        default=4,
        help="number of positional encoding frequencies for viewing direction",
    )
    parser.add_argument(
        "--n_rays_batch", default=32 * 32 * 4, help="number of rays per batch"
    )
    parser.add_argument("--n_sample", default=400, help="number of sample per ray")
    parser.add_argument(
        "--max_iters", default=10000, help="number of max iterations for training"
    )
    parser.add_argument("--logs_path", default="./logs/", help="logs path")
    parser.add_argument(
        "--checkpoint_path",
        default="./Checkpoints/",
        help="checkpoints path",
    )
    parser.add_argument(
        "--load_checkpoint", default=True, help="whether to load checkpoint or not"
    )
    parser.add_argument(
        "--save_ckpt_iter", default=1000, help="num of iteration to save checkpoint"
    )
    parser.add_argument(
        "--images_path", default="./image/", help="folder to store images"
    )
    return parser


if __name__ == "__main__":
    # Parse command-line arguments and start the main function
    parser = parser_config()
    args = parser.parse_args()
    main(args)
