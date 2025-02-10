"""
Training script for the unsupervised HomographyNet to estimate the homography between two images.

This script initializes the model, loads the data, and runs the training loop.
"""

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torchvision import transforms
from custom_image_dataset import CustomImageDataset  # Import the custom dataset
from Network.homography_model import Net  # Import the Net from homography_model
import argparse  # Import argparse for command-line arguments

class Net(pl.LightningModule):
    # ...existing code...

    def validation_step(self, batch, batch_idx):
        # ...existing code...
        self.log('val_loss', val_loss, sync_dist=True)  # Update logging for distributed setting
        # ...existing code...

def main():
    # Set random seed for reproducibility
    pl.seed_everything(42)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train the unsupervised HomographyNet.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    args = parser.parse_args()

    # Define hyperparameters
    hparams = {
        'InputSize': 128,
        'OutputSize': 8,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
    }

    # Initialize the model
    model = Net(hparams)

    # Define data transformations
    transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert tensor to PIL Image
        transforms.ToTensor(),  # Convert PIL Image to tensor
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load the training and validation datasets
    train_dataset = CustomImageDataset(
        annotations_file='Code/TxtFiles/TrainLabels.csv',
        img_dir='Data/Train/',
        transform=transform
    )
    val_dataset = CustomImageDataset(
        annotations_file='Code/TxtFiles/TestLabels.csv',
        img_dir='Data/Val/',
        transform=transform
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=hparams['batch_size'], shuffle=False)

    # Define the checkpoint callback to save the best model
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='Checkpoints/',
        filename='unsupervised-homographynet-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
    )

    # Initialize the PyTorch Lightning trainer
    trainer = pl.Trainer(
        max_epochs=hparams['num_epochs'],
        devices=torch.cuda.device_count() if torch.cuda.is_available() else 0,
        callbacks=[checkpoint_callback]
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
