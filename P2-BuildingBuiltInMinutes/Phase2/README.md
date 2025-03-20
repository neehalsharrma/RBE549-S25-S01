# Neural Radiance Fields (NeRF) Implementation

This project implements a Neural Radiance Fields (NeRF) pipeline for rendering 3D scenes from 2D images. It includes modules for dataset loading, ray generation, sampling, rendering, and training/testing the NeRF model.

## Project Structure

- **`DataLoader.py`**: Handles loading and preprocessing of datasets, including images, camera poses, and camera parameters.
- **`Wrapper.py`**: Implements the NeRF pipeline, including ray generation, sampling, rendering, training, and testing.
- **`NeRFModel.py`**: Defines the NeRF model using PyTorch, including positional encoding and a multi-layer perceptron (MLP).
- **`requirements.txt`**: Lists all Python dependencies required to run the project.

## Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional but recommended for faster training)

## Installation

1. Ensure you are in the Phase 2 folder, as all filepaths are relative.

2. Create a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the NeRF Model

1. Prepare your dataset in the required format (e.g., images and `transforms_train.json` for training).
2. Run the training script:
   ```bash
   python Phase2/Wrapper.py --data_path <path-to-dataset> --mode train
   ```
   Replace `<path-to-dataset>` with the path to your dataset directory. The default folder used is `lego` from the official repository.

### Testing the NeRF Model

1. Ensure a trained model checkpoint is available in the specified checkpoint path.
2. Run the testing script:
   ```bash
   python Phase2/Wrapper.py --data_path <path-to-dataset> --mode test
   ```

### Command-Line Arguments

The script supports several command-line arguments for customization. Some key arguments include:
- `--data_path`: Path to the dataset (default: `lego`).
- `--mode`: Operation mode (`train` or `test`).
- `--lrate`: Learning rate for training (default: `5e-4`).
- `--n_rays_batch`: Number of rays per batch (default: `32*32*4`).
- `--max_iters`: Maximum number of training iterations (default: `10000`).
- `--checkpoint_path`: Path to save/load model checkpoints (default: `./Checkpoints/`).

For a full list of arguments, refer to the `parser_config` function in `Wrapper.py`.

## Results

- During training, rendered test images are saved in the specified `images_path`.
- After testing, a GIF of the rendered results is saved as `output.gif`.