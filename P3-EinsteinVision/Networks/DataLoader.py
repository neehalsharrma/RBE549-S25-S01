"""
This module provides utility functions for loading video data and extracting frames
from video files. It is designed to work with OpenCV and NumPy for video processing.

Functions
---------
- load_video(video_path: str, video_num: int, cam_type: str, distorted: bool) -> tuple[cv2.VideoCapture, int]
    Loads a video file and returns a video capture object and the number of frames.
- get_frame(cap: cv2.VideoCapture, frame_num: int) -> np.ndarray or None
    Retrieves a specific frame from a video capture object.
"""

import cv2
import numpy as np
import os


def load_video(
    video_path: str = "../Data/P3Data/Sequences",
    video_num: int = 1,
    cam_type: str = "front",
    distorted: bool = False,
) -> tuple[cv2.VideoCapture, int]:
    """
    Load a video file and return a video capture object and the number of frames.

    Parameters
    ----------
    video_path : str, optional
        The base directory where video sequences are stored (default is '../Data/P3Data/Sequences').
    video_num : int, optional
        The scene number of the video to load (default is 1).
    cam_type : str, optional
        The camera type to filter videos (e.g., 'front', 'rear') (default is 'front').
    distorted : bool, optional
        Whether to load the distorted ('Raw') or undistorted ('Undist') video (default is False).

    Returns
    -------
    tuple[cv2.VideoCapture, int]
        A tuple containing the video capture object and the total number of frames in the video.

    Raises
    ------
    FileNotFoundError
        If the specified video file is not found.
    """
    # Determine the video type based on distortion flag
    vid_type = "Raw" if not distorted else "Undist"

    # Construct the full path to the video directory
    video_path = os.path.join(video_path, f"scene{video_num}/{vid_type}")

    # List all files in the video directory
    videos = os.listdir(video_path)

    # Filter the video files based on the camera type
    video = [i for i in videos if cam_type in i][0]

    # Create a video capture object
    cap = cv2.VideoCapture(os.path.join(video_path, video))

    # Get the total number of frames in the video
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    return cap, num_frames


def get_frame(cap: cv2.VideoCapture, frame_num: int) -> np.ndarray or None:
    """
    Get a frame from a video capture object.

    Parameters
    ----------
    cap : cv2.VideoCapture
        The video capture object that is being used.
    frame_num : int
        The frame number that is being requested.

    Returns
    -------
    np.ndarray or None
        The frame at the requested frame number as a NumPy array, or None if the frame number is out of bounds.

    Notes
    -----
    The frame is returned in RGB format.

    Raises
    ------
    ValueError
        If the frame number is negative.
    """
    # Check if the frame number is valid
    if frame_num < 0:
        raise ValueError("Frame number must be non-negative.")
    if frame_num >= cap.get(cv2.CAP_PROP_FRAME_COUNT):
        return None

    # Set the video capture object to the desired frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

    # Read the frame from the video capture object
    ret, frame = cap.read()
    if not ret:
        return None

    # Convert the frame from BGR to RGB format
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return frame
