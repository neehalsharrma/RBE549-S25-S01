# Deep Visual-Inertial Odometry Networks

This README explains the neural network architectures for Visual-Inertial Odometry (VIO) implemented in the `network.py` file.

## Overview

The code implements three complementary neural networks for odometry estimation:

1. **Visual Encoder**: Processes image pairs to estimate pose changes
2. **Inertial Encoder**: Processes IMU data (accelerometer and gyroscope) to estimate pose changes
3. **Visual-Inertial Encoder**: Fuses outputs from both encoders for more robust pose estimation

All networks are designed to predict 6-DoF pose changes between consecutive frames (3D position and orientation).

## Network Architectures

### 1. Visual Encoder

A CNN+LSTM architecture that processes stacked image pairs:

```
Input (stacked image pairs) → CNN layers → LSTM → Linear layer → 6-DoF pose
```

**Network details:**
- Takes 6-channel input (2 stacked RGB images)
- CNN backbone with decreasing kernel sizes (7×7→5×5→3×3)
- Progressive feature maps (64→128→256→512 channels)
- Consistent dropout (0.2) for regularization
- LSTM layer to capture temporal relationships
- Linear layer to output 6-DoF pose (position and orientation)

### 2. Inertial Encoder

Processes IMU time-series data:

```
Input (IMU data) → 1D CNN layers → LSTM → Linear layer → 6-DoF pose
```

**Network details:**
- Takes 6-channel input (3-axis accelerometer + 3-axis gyroscope)
- 1D convolutional layers appropriate for time-series data
- Progressive channel expansion (6→64→128→256)
- Lower dropout (0.1) compared to the visual encoder
- 2-layer LSTM to capture complex temporal dynamics
- Linear layer to output 6-DoF pose (position and orientation)

### 3. Visual-Inertial Encoder (Fusion Network)

Combines visual and inertial data for more robust pose estimation:

```
Visual input → Visual_encoder → ┐
                                 ├→ Concatenate → Conv1D fusion → Linear → 6-DoF pose
IMU input → Inertial_encoder → ┘
```

**Network details:**
- Uses both Visual and Inertial encoders as feature extractors
- Simple but effective fusion via concatenation
- 1D convolution for feature fusion (2→64 channels)
- Final linear layer to output 6-DoF pose

## Design Inspirations

These networks draw inspiration from multiple established architectures:

- **Visual Encoder**: Influenced by FlowNet/DeepVO, VGG-style progression, and ResNet's kernel patterns
- **Inertial Encoder**: Similar to IONet and LSTM-based IMU processing networks
- **Fusion Network**: Resembles VINet and other selective sensor fusion approaches

## Loss Function

The custom loss function balances position and orientation errors:
- Position error uses MSE (weighted 0.4)
- Orientation error uses cosine similarity (weighted 0.6)

This weighting reflects the different scales and importance of position vs. orientation accuracy in odometry tasks.

## Key Features

1. **Complementary modalities**: Visual data provides rich spatial information but struggles with challenging lighting conditions and motion blur; IMU provides direct motion measurements but suffers from drift
   
2. **CNN+LSTM hybrid architecture**: CNNs excel at spatial feature extraction while LSTMs capture temporal dependencies

3. **Modular design**: Independent development and evaluation of each modality before fusion

4. **Progressive feature extraction**: Increasing feature channels while reducing spatial dimensions for hierarchical feature learning