# Project 2: Building Built In Minutes

This folder contains the implementation of Structure from Motion (SfM) and NeRF.

## Setup Instructions

### Step 1: Create a Python Virtual Environment

1. Create a virtual environment named `venv`:

    ```sh
    python3 -m venv venv
    ```

2. Activate the virtual environment:
    - On Windows:

        ```sh
        venv\Scripts\activate
        ```

    - On macOS and Linux:

        ```sh
        source venv/bin/activate
        ```

### Step 2: Install Required Packages

1. Install the required packages using `pip`:

    ```sh
    pip install -r requirements.txt
    ```

## Usage

### SfM Instructions

1. Ensure that the virtual environment is activated.

2. Run the `Wrapper.py` script:

    ```sh
    python Wrapper.py
    ```

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
- After testing, a GIF of the rendered results is saved as `output.gif`.# README