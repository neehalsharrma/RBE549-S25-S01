"""
Wrapper.py

This script implements a Neural Radiance Field (NeRF) pipeline for rendering 3D scenes from 2D images.
It includes functions for loading datasets, generating rays, sampling points along rays, rendering images,
and training/testing the NeRF model.

Modules:
--------
- loadDataset(data_path, mode): Load the dataset for training or testing.
- PixelToRay(camera_info, pose, pixelPosition, near, far, args): Convert a pixel position to a ray in 3D space.
- PointsAlongRay(ray_origin, ray_direction, near, far, n_samples): Sample points along a ray using stratified sampling.
- generateBatch(images, poses, camera_info, args): Generate a batch of rays and their corresponding ground truth RGB values.
- render(model, rays_origin, rays_direction, args): Render RGB values for input rays using the NeRF model.
- loss(groundtruth, prediction): Compute the Mean Squared Error (MSE) loss.
- train(images, poses, camera_info, args): Train the NeRF model.
- render_test_image(model, pose, camera_info, args): Render a test image for visualization during training.
- test(images, poses, camera_info, args): Test the NeRF model on the test dataset.
- main(args): Main function to load data and start training or testing.
- configParser(): Configure the argument parser for the script.

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
import glob
from tqdm import tqdm
import random
from torch.utils.tensorboard import SummaryWriter
import imageio
import torch
import matplotlib.pyplot as plt
import os

from NeRFModel import NeRFmodel
from DataLoader import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)


def loadDataset(data_path, mode):
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
    return DataLoader(data_path).loadDataset(mode)


def PixelToRay(camera_info, pose, pixelPosition, near, far, args):
    """
    Convert a pixel position to a ray in 3D space.

    Parameters
    ----------
    camera_info : tuple
        Image width, height, and camera matrix (focal length).
    pose : torch.Tensor
        Camera pose in the world frame.
    pixelPosition : tuple
        Pixel position in the image (x, y).
    near : float
        Near clipping plane distance.
    far : float
        Far clipping plane distance.
    args : argparse.Namespace
        Additional arguments for ray sampling.

    Returns
    -------
    tuple
        ray_origin : torch.Tensor
            Origin of the ray in 3D space.
        ray_direction : torch.Tensor
            Direction of the ray in 3D space.
    """
    W, H, focal = camera_info
    x, y = pixelPosition

    norm_x = (x - W * 0.5) / focal
    norm_y = (y - H * 0.5) / focal
    direction = torch.tensor([norm_x, -norm_y, 1.0], dtype=torch.float32).to(device)

    rotation = pose[:3, :3]
    translation = pose[:3, 3]

    ray_direction = torch.matmul(rotation, direction)
    ray_direction = ray_direction / torch.norm(ray_direction)
    ray_origin = translation

    return ray_origin, ray_direction


def PointsAlongRay(ray_origin, ray_direction, near, far, n_samples):
    """
    Sample points along a ray using stratified sampling.

    Parameters
    ----------
    ray_origin : torch.Tensor
        Origin of the ray in 3D space.
    ray_direction : torch.Tensor
        Direction of the ray in 3D space.
    near : float
        Near clipping plane distance.
    far : float
        Far clipping plane distance.
    n_samples : int
        Number of samples along the ray.

    Returns
    -------
    tuple
        points : torch.Tensor
            Sampled 3D points along the ray.
        t_samples : torch.Tensor
            Corresponding t-values for the sampled points.
    """
    t_vals = torch.linspace(near, far, n_samples).to(device)
    rand_t = torch.rand(n_samples - 1).to(device)

    # Scale these random values to be within the bin widths
    bin_widths = t_vals[1:] - t_vals[:-1]

    # Get starting points of each bin
    bin_starts = t_vals[:-1]

    # Generate final sample positions: bin_start + rand * bin_width
    t_samples = bin_starts + rand_t * bin_widths

    # Get the actual 3D points along the ray
    points = ray_origin[None, :] + ray_direction[None, :] * t_samples[:, None]

    return points, t_samples


def generateBatch(images, poses, camera_info, args):
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
    rays_o = np.zeros((n_rays, 3))
    rays_d = np.zeros((n_rays, 3))
    rgb_gt = np.zeros((n_rays, 3))

    # Generate random pixel coordinates
    x_coords = np.random.randint(0, W, size=n_rays)
    y_coords = np.random.randint(0, H, size=n_rays)

    # Generate rays for each sampled pixel
    for i in range(n_rays):
        # Get ray origin and direction for this pixel
        ray_o, ray_d = PixelToRay(
            camera_info, pose, (x_coords[i], y_coords[i]), 0, 0, args
        )
        rays_o[i] = ray_o
        rays_d[i] = ray_d

        # Get ground truth RGB value for this pixel
        rgb_gt[i] = image[y_coords[i], x_coords[i]]

    # Convert to torch tensors
    rays_o = torch.tensor(rays_o, dtype=torch.float32).to(device)
    rays_d = torch.tensor(rays_d, dtype=torch.float32).to(device)
    rgb_gt = torch.tensor(rgb_gt, dtype=torch.float32).to(device)

    return rays_o, rays_d, rgb_gt


# Run neural network on various points along the ray and find the color of the ray
def render(model, rays_origin, rays_direction, args):
    """
    Render RGB values for input rays using the NeRF model.

    Parameters
    ----------
    model : NeRFmodel
        Neural Radiance Field (NeRF) model.
    rays_origin : torch.Tensor
        Origins of the input rays.
    rays_direction : torch.Tensor
        Directions of the input rays.
    args : argparse.Namespace
        Additional arguments for rendering.

    Returns
    -------
    torch.Tensor
        RGB values of the input rays.
    """
    n_rays = rays_origin.shape[0]
    n_samples = args.n_sample
    near, far = 2.0, 6.0  # Typical values for synthetic NeRF datasets

    # Storage for final RGB for each ray
    rgb_final = torch.zeros((n_rays, 3)).to(device)

    # Process each ray
    for i in range(n_rays):
        # Sample points along the ray using stratified sampling
        points, t_vals = PointsAlongRay(
            rays_origin[i], rays_direction[i], near, far, n_samples
        )

        # Calculate distances between adjacent samples
        deltas = t_vals[1:] - t_vals[:-1]
        # Add a small value for the last delta
        deltas = torch.cat([deltas, torch.tensor([1e-10]).to(device)])

        # Normalize directions for the NeRF model
        view_dirs = rays_direction[i].expand(points.shape[0], 3)

        # Predict RGB and density for each point along the ray
        rgb, sigma = model(points, view_dirs)

        # Calculate alpha values (opacity)
        alpha = 1.0 - torch.exp(-sigma.squeeze() * deltas)

        # Calculate transmittance (T_i in the paper)
        # T_i = exp(-sum_{j=1}^{i-1} sigma_j * delta_j)
        exp_term = torch.exp(-sigma.squeeze() * deltas)
        transmittance = torch.cumprod(
            torch.cat([torch.ones(1).to(device), exp_term[:-1]]), dim=0
        )

        # Calculate weights as in the paper: T_i * (1 - exp(-sigma_i * delta_i))
        weights = transmittance * alpha

        # Final rendered RGB for this ray
        rgb_ray = torch.sum(weights.unsqueeze(-1) * rgb, dim=0)

        rgb_final[i] = rgb_ray

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
    # Initialize model, optimizer, and summary writer
    n_pos_freqs = args.n_pos_freq
    n_dir_freqs = args.n_dirc_freq

    model = NeRFmodel().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lrate))

    # Create log directory if it doesn't exist
    os.makedirs(args.logs_path, exist_ok=True)
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

    # Training loop
    pbar = tqdm(range(start_iter, int(args.max_iters)))
    for i in pbar:
        # Generate batch of rays
        rays_o, rays_d, rgb_gt = generateBatch(images, poses, camera_info, args)

        # Forward pass
        rgb_pred = render(model, rays_o, rays_d, args)

        # Calculate loss
        loss_val = loss(rgb_gt, rgb_pred)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

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
        Neural Radiance Field (NeRF) model.
    pose : ndarray
        Camera pose in the world frame.
    camera_info : tuple
        Image width, height, and camera matrix (focal length).
    args : argparse.Namespace
        Additional arguments for rendering.

    Returns
    -------
    torch.Tensor
        Rendered test image.
    """
    W, H, focal = camera_info
    # Use a smaller resolution for faster rendering during training
    W_test, H_test = W // 4, H // 4

    img = torch.zeros((H_test, W_test, 3)).to(device)

    for j in range(H_test):
        for i in range(W_test):
            # Map to original resolution
            x, y = i * 4, j * 4
            ray_o, ray_d = PixelToRay(camera_info, pose, (x, y), 0, 0, args)

            ray_o = torch.tensor(ray_o, dtype=torch.float32).to(device).unsqueeze(0)
            ray_d = torch.tensor(ray_d, dtype=torch.float32).to(device).unsqueeze(0)

            rgb = render(model, ray_o, ray_d, args)
            img[j, i] = rgb[0]

    return img


def test(images, poses, camera_info, args):
    """
    Test the NeRF model on the test dataset.

    Parameters
    ----------
    images : list
        All images in the test dataset.
    poses : list
        Corresponding camera poses in the world frame.
    camera_info : tuple
        Image width, height, and camera matrix (focal length).
    args : argparse.Namespace
        Arguments containing testing configurations.
    """
    # Initialize model
    n_pos_freqs = args.n_pos_freq
    n_dir_freqs = args.n_dirc_freq

    model = NeRFmodel().to(device)

    # Load checkpoint
    checkpoint_path = os.path.join(args.checkpoint_path, "model.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from iteration {checkpoint['iteration']}")
    else:
        print("No checkpoint found. Cannot test without a trained model.")
        return

    # Create images directory if it doesn't exist
    os.makedirs(args.images_path, exist_ok=True)

    # Set the model to evaluation mode
    model.eval()
    with torch.no_grad():
        psnrs = []  # List to store PSNR values for each test image

        # Loop through each test image and its corresponding pose
        for idx, (image, pose) in enumerate(zip(images, poses)):
            print(f"Rendering test image {idx+1}/{len(images)}")

            # Get image dimensions and focal length from camera info
            W, H, focal = camera_info

            # Initialize an empty tensor to store the rendered image
            img = torch.zeros((H, W, 3)).to(device)

            # Process the image in chunks to avoid memory issues
            chunk_size = 64  # Number of rows to process at a time
            for j in tqdm(range(0, H, chunk_size)):
                j_end = min(j + chunk_size, H)  # End row for the current chunk
                for i in range(0, W, chunk_size):
                    i_end = min(i + chunk_size, W)  # End column for the current chunk

                    rays_o = []  # List to store ray origins for the current chunk
                    rays_d = []  # List to store ray directions for the current chunk

                    # Generate rays for each pixel in the current chunk
                    for y in range(j, j_end):
                        for x in range(i, i_end):
                            ray_o, ray_d = PixelToRay(
                                camera_info, pose, (x, y), 0, 0, args
                            )
                            rays_o.append(ray_o)
                            rays_d.append(ray_d)

                    # Convert ray origins and directions to tensors
                    rays_o = torch.tensor(np.stack(rays_o), dtype=torch.float32).to(
                        device
                    )
                    rays_d = torch.tensor(np.stack(rays_d), dtype=torch.float32).to(
                        device
                    )

                    # Render the RGB values for the current chunk of rays
                    rgb_chunk = render(model, rays_o, rays_d, args)

                    # Place the rendered RGB values into the corresponding positions in the image
                    pixel_idx = 0
                    for y in range(j, j_end):
                        for x in range(i, i_end):
                            img[y, x] = rgb_chunk[pixel_idx]
                            pixel_idx += 1

            # Convert the rendered image to a numpy array for saving
            img_np = img.cpu().numpy()
            img_np = np.clip(
                img_np, 0, 1
            )  # Ensure pixel values are in the range [0, 1]

            # Save the rendered image as a PNG file
            plt.figure()
            plt.imshow(img_np)
            plt.savefig(os.path.join(args.images_path, f"test_{idx:03d}.png"))
            plt.close()

            # Save the rendered image using imageio for compatibility
            imageio.imwrite(
                os.path.join(args.images_path, f"test_{idx:03d}.png"),
                (img_np * 255).astype(np.uint8),
            )

            # Calculate PSNR (Peak Signal-to-Noise Ratio) if ground truth is available
            if image is not None:
                gt = torch.tensor(image, dtype=torch.float32).to(
                    device
                )  # Ground truth image
                mse = torch.mean((gt - img) ** 2)  # Mean Squared Error
                psnr = -10 * torch.log10(mse)  # PSNR calculation
                psnrs.append(psnr.item())  # Append PSNR value to the list
                print(f"PSNR: {psnr.item():.2f} dB")

        # Report the average PSNR across all test images
        if psnrs:
            print(f"Average PSNR: {np.mean(psnrs):.2f} dB")


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
    images, poses, camera_info = loadDataset(args.data_path, args.mode)

    if args.mode == "train":
        print("Start training")
        train(images, poses, camera_info, args)
    elif args.mode == "test":
        print("Start testing")
        args.load_checkpoint = True
        test(images, poses, camera_info, args)


def configParser():
    """
    Configure the argument parser for the script.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", default="nerf_synthetic/lego", help="dataset path"
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
    parser = configParser()
    args = parser.parse_args()
    main(args)
