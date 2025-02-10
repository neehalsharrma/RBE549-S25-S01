"""
Testing script for the unsupervised HomographyNet to estimate the homography between two images.

This script loads the trained model, performs inference on a test dataset, and prints the results.
"""

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from custom_image_dataset import CustomImageDataset  # Import the custom dataset
from Network.unsupervised_homography_net import (
    UnsupervisedHomographyNet,
)  # Import the unsupervised HomographyNet
import argparse  # Import argparse for command-line arguments


def main():
    # Set random seed for reproducibility
    pl.seed_everything(42)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test the unsupervised HomographyNet.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="Checkpoints/unsupervised-homographynet.ckpt",
        help="Path to the model checkpoint",
    )
    args = parser.parse_args()

    # Define hyperparameters
    hparams = {
        "InputSize": 128,
        "OutputSize": 8,
        "batch_size": args.batch_size,
    }

    # Initialize the model
    model = UnsupervisedHomographyNet.load_from_checkpoint(args.checkpoint_path)

    # Define data transformations
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),  # Convert tensor to PIL Image
            transforms.ToTensor(),  # Convert PIL Image to tensor
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    # Load the test dataset
    test_dataset = CustomImageDataset(
        annotations_file="Code/TxtFiles/TestLabels.csv",
        img_dir="Data/Test/",
        transform=transform,
    )

    # Create data loader
    test_loader = DataLoader(
        test_dataset, batch_size=hparams["batch_size"], shuffle=False
    )

    # Initialize the PyTorch Lightning trainer
    trainer = pl.Trainer()

    # Test the model
    results = trainer.test(model, test_loader)
    print(results)


if __name__ == "__main__":
    main()
