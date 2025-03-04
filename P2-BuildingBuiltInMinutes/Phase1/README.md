# Phase 1: Building Built In Minutes

This folder contains the implementation of various computer vision algorithms for 3D reconstruction, including linear triangulation, PnP, RANSAC, and bundle adjustment.

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

### Step 3: Run the Phase 1 Code

1. Ensure that the virtual environment is activated.

2. Run the `Wrapper.py` script:

    ```sh
    python Wrapper.py
    ```

## Description of Files

- `LinearTriangulation.py`: Module for performing linear triangulation to estimate 3D points from 2D correspondences in two images.
- `LinearPnp.py`: Module for performing Linear Perspective-n-Point (PnP) and PnP RANSAC for estimating the camera pose from 2D-3D point correspondences.
- `GetInliersRANSAC.py`: Module for computing inliers using RANSAC algorithm.
- `ExtractCameraPose.py`: Module for extracting camera pose from an essential matrix.
- `EstimateFundamentalMatrix.py`: Module for estimating the Fundamental Matrix and plotting epipolar lines.
- `EssentialMatrixFromFundamentalMatrix.py`: Module for estimating the Essential Matrix from the Fundamental Matrix.
- `DisambiguateCameraPose.py`: Module for disambiguating the correct camera pose from a set of possible poses.
- `BundleAdjustment.py`: Module for performing bundle adjustment to refine camera poses and 3D points.
- `BuildVisibilityMatrix.py`: Module for building a visibility matrix for 3D points projected onto 2D images.
- `Wrapper.py`: Main script to run the entire Structure from Motion (SfM) pipeline.
- `RANSAC_7PT.py`: Module for RANSAC algorithms to estimate the Fundamental Matrix using Sampson distance and the 7-point algorithm.
- `PnPRANSAC.py`: Module for performing Perspective-n-Point (PnP) with RANSAC.
- `NonLinearTriangulation.py`: Module for performing non-linear triangulation to estimate 3D points from 2D image correspondences.
- `NonlinearPnP.py`: Module for performing Nonlinear Perspective-n-Point (PnP) problem to refine camera center and rotation matrix.

## Notes

- Ensure that all dependencies are installed as specified in the `requirements.txt` file.
- The output files and plots will be saved in the `Outputs` directory within the `Phase1` folder.
