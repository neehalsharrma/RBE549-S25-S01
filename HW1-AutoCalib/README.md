# HW1: MyAutoCalib

## Overview

This module provides functions to perform automatic camera calibration based on the method presented by Zhengyou Zhang. The script detects chessboard corners in a set of images, computes the intrinsic and extrinsic parameters of the camera, and optimizes these parameters to minimize reprojection errors.

## Requirements

- Python 3.x
- OpenCV
- NumPy
- SciPy

## Installation

1. Create a virtual environment:

    ```sh
    python -m venv venv
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

3. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Place your calibration images in the `Data/` directory. The images should contain a chessboard pattern.

2. Run the calibration script:

    ```sh
    python Wrapper.py
    ```

3. The script will process the images, compute the camera parameters, and save the results in the `Output/` directory. The output will include:
    - Rectified images with reprojected points marked.
    - Printed results of the optimized intrinsic matrix and distortion coefficients.

## Example

```sh
python Wrapper.py
```

## Output

The script will print the optimized intrinsic matrix and distortion coefficients, and save the rectified images in the `Output/` directory.

## Notes

- Ensure that the chessboard pattern size and square size are correctly set in the script.
- The default chessboard pattern size is 9x6, and the square size is 21.5 mm. Adjust these values if your pattern is different.
