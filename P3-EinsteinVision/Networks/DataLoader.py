import cv2
import numpy as np
import os

def load_video(video_path:str = '../Data/P3Data/Sequences', video_num:int = 1, cam_type:str='front', distorted:bool = False) -> [cv2.VideoCapture, int]:
    vid_type = 'Raw' if not distorted else 'Undist'
    video_path = os.path.join(video_path, f'scene{video_num}/{vid_type}')
    videos = os.listdir(video_path)
    video = [i for i in videos if cam_type in i][0]
    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, num_frames

def get_frame(cap:cv2.VideoCapture, frame_num:int) -> np.ndarray or None:
    """
    Get a frame from a video capture object
    :param cap: the video capture object that is being used
    :param frame_num: the frame number that is being requested
    :return: the frame at the requested frame number or None if the frame number is out of bounds
    """
    if frame_num >= cap.get(cv2.CAP_PROP_FRAME_COUNT):
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if not ret:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame
