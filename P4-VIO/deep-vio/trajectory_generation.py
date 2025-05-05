"""
Trajectory generation for Visual-Inertial Odometry (VIO).

This module provides functions to generate and visualize trajectories for VIO systems.
It includes methods to create circular and figure-eight trajectories, visualize them in 3D,
and save the trajectory data to a CSV file.

Functions
---------
generate_circular_trajectory(radius: float, center: List[float]) -> np.ndarray
    Generates a circular trajectory based on the given radius and center.

generate_figure_eight_trajectory(amplitude: float, fixed_z: float) -> np.ndarray
    Generates a figure-eight trajectory with a fixed z-coordinate.

visualize_trajectory(trajectory: np.ndarray) -> None
    Visualizes the given trajectory in 3D space.

save_trajectory_to_csv(trajectory: np.ndarray, file_path: str) -> None
    Saves the trajectory data to a CSV file.

main() -> None
    Main function to generate, save, and visualize trajectories.
"""

# Standard library imports
import csv  # For saving trajectory data to CSV files
import math  # For mathematical operations like sine, cosine, etc.
import random  # For generating random yaw, pitch, and roll values
from typing import List, Tuple  # For type annotations

# Third-party imports
import matplotlib.pyplot as plt  # For 3D visualization of trajectories
import numpy as np  # For numerical operations and array handling


def generate_circular_trajectory(radius: float, center: List[float]) -> np.ndarray:
    """
    Generate a circular trajectory.

    Parameters
    ----------
    radius : float
        Radius of the circle.
    center : List[float]
        Coordinates of the circle's center [x, y, z].

    Returns
    -------
    np.ndarray
        Array containing the trajectory points.
        Each point is of format [2, yaw, pitch, roll, delta_x, delta_y, delta_z, 1, 1]
    """
    trajectory_points = []  # List to store trajectory points
    angles = np.linspace(
        0, 2 * np.pi, 10
    )  # Generate 10 evenly spaced angles for the circle
    last_position = [
        0,
        0,
        0,
        0,
        0,
        0,
    ]  # Initialize the last position for delta calculations

    for angle in angles:
        # Calculate x, y, z coordinates for the current angle
        x = radius * math.cos(angle) + center[0]
        y = radius * math.sin(angle) + center[1]
        z = center[2]

        # Generate random yaw, pitch, and roll values
        yaw = random.uniform(0, 2 * math.pi)
        pitch = random.uniform(0, math.pi / 4)
        roll = random.uniform(0, math.pi / 4)

        # Calculate deltas for x, y, z
        current_position = [yaw, pitch, roll, x, y, z]
        delta_x = x - last_position[3]
        delta_y = y - last_position[4]
        delta_z = z - last_position[5]
        last_position = current_position  # Update last position

        # Append the trajectory point in the required format
        trajectory_points.append([2, yaw, pitch, roll, delta_x, delta_y, delta_z, 1, 1])

    return np.array(trajectory_points)  # Convert list to numpy array


def generate_figure_eight_trajectory(amplitude: float, fixed_z: float) -> np.ndarray:
    """
    Generate a figure-eight trajectory.

    Parameters
    ----------
    amplitude : float
        Amplitude of the figure-eight pattern.
    fixed_z : float
        Fixed z-coordinate for the trajectory.

    Returns
    -------
    np.ndarray
        Array containing the trajectory points.
        Each point is of format [2, yaw, pitch, roll, delta_x, delta_y, delta_z, 1, 1]
    """
    time_steps = np.linspace(0, 2 * np.pi, 30)  # Generate 30 evenly spaced time steps
    last_position = [
        0,
        0,
        0,
        0,
        0,
        0,
    ]  # Initialize the last position for delta calculations
    trajectory_points = []  # List to store trajectory points

    for time_step in time_steps:
        # Calculate x, y coordinates for the figure-eight pattern
        x = amplitude * math.sin(time_step)
        y = amplitude * math.sin(time_step) * math.cos(time_step)

        # Generate random yaw, pitch, and roll values
        yaw = random.uniform(0, 2 * math.pi)
        pitch = random.uniform(0, math.pi / 4)
        roll = random.uniform(0, math.pi / 4)

        # Calculate deltas for x, y, z
        current_position = [yaw, pitch, roll, x, y, fixed_z]
        delta_x = x - last_position[3]
        delta_y = y - last_position[4]
        delta_z = fixed_z - last_position[5]
        last_position = current_position  # Update last position

        # Append the trajectory point in the required format
        trajectory_points.append([2, yaw, pitch, roll, delta_x, delta_y, delta_z, 1, 1])

    return np.array(trajectory_points)  # Convert list to numpy array


def visualize_trajectory(trajectory: np.ndarray) -> None:
    """
    Visualize a trajectory in 3D.

    Parameters
    ----------
    trajectory : np.ndarray
        The trajectory points to visualize.
    """
    # Extract x, y, z coordinates from the trajectory
    x = trajectory[:, 4]
    y = trajectory[:, 5]
    z = trajectory[:, 6]

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z, c=z, cmap="viridis")  # Color points based on z-coordinate
    ax.set_xlabel("X")  # Label for X-axis
    ax.set_ylabel("Y")  # Label for Y-axis
    ax.set_zlabel("Z")  # Label for Z-axis
    plt.show()  # Display the plot


def save_trajectory_to_csv(trajectory: np.ndarray, file_path: str) -> None:
    """
    Save trajectory data to a CSV file.

    Parameters
    ----------
    trajectory : np.ndarray
        The trajectory data to save.
    file_path : str
        Path to the output CSV file.
    """
    # Open the file in write mode and create a CSV writer
    with open(file_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(trajectory)  # Write each trajectory point as a row


def main() -> None:
    """
    Main function to generate and visualize trajectories.
    """
    # Generate a figure-eight trajectory with amplitude 5 and fixed z-coordinate 10
    trajectory = generate_figure_eight_trajectory(5, 10)

    # Save the trajectory to a CSV file named "data.csv"
    csv_file = "data.csv"
    save_trajectory_to_csv(trajectory, csv_file)

    # Visualize the generated trajectory in 3D
    visualize_trajectory(trajectory)


if __name__ == "__main__":
    main()  # Execute the main function
