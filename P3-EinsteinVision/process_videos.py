import os
from pyffmpeg import FFmpeg


def generate_video_from_frames(
    frames_dir: str, output_video: str, fps: int = 30
) -> None:
    """
    Generate a video from a directory of frames using the pyffmpeg library.

    Parameters
    ----------
    frames_dir : str
        Path to the directory containing the frames.
    output_video : str
        Path to the output video file.
    fps : int, optional
        Frames per second for the output video (default is 30).

    Returns
    -------
    None
    """
    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(output_video)
    os.makedirs(output_dir, exist_ok=True)

    # Define the input pattern for frame images
    input_pattern = os.path.join(frames_dir, "*.png")

    # Initialize pyffmpeg
    ff = FFmpeg()

    # Use pyffmpeg to generate the video
    ff.options(
        f"-y -framerate {fps} -pattern_type glob -i {input_pattern} -c:v libx264 -pix_fmt yuv420p {output_video}"
    )
    print(f"Video generated at {output_video}")


def process_videos(video_number: int) -> None:
    """
    Process videos based on the given video number and generate output videos.

    Parameters
    ----------
    video_number : int
        The video number to process.

    Returns
    -------
    None
    """
    # Define directories for input frames and output videos
    yolo_frames_dir = f"./Results/YOLO/vid_{video_number}"
    midas_frames_dir = f"./Results/MiDaS/vid_{video_number}"
    twinlite_frames_dir = f"./Results/TwinLiteNet/vid_{video_number}"
    output_dir = f"./Results/FFMPEG/vid_{video_number}"

    # Generate YOLO-annotated video
    yolo_output_video = os.path.join(output_dir, "yolo_annotated_video.mp4")
    generate_video_from_frames(yolo_frames_dir, yolo_output_video)

    # Generate MiDaS depth video
    midas_output_video = os.path.join(output_dir, "midas_depth_video.mp4")
    generate_video_from_frames(midas_frames_dir, midas_output_video)

    # Generate TwinLiteNet output video
    twinlite_output_video = os.path.join(output_dir, "twinlite_output_video.mp4")
    generate_video_from_frames(twinlite_frames_dir, twinlite_output_video)


if __name__ == "__main__":
    # Prompt the user for the video number
    video_number = int(input("Enter the video number to process: "))

    # Process the videos
    process_videos(video_number)
