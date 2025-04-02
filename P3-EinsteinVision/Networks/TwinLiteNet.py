"""
TwinLiteNet Model Loader.

This module provides functionality to load the TwinLiteNetPlus model with
pre-trained weights and set it to evaluation mode.

Functions
---------
load_TwinLiteNet(weights: str)
    Loads the TwinLiteNetPlus model with the specified pre-trained weights.
"""

import sys

import torch
from Networks.TwinLiteNetPlus.model.model import TwinLiteNetPlus
import argparse
import cv2
import numpy as np

# Disable the creation of __pycache__ directories
sys.dont_write_bytecode = True


def load_TwinLiteNet(weights: str = "Pretrained/tlp_medium.pth", config='medium',
                     dev: torch.device = torch.device('cpu')) -> TwinLiteNetPlus:
    """
    Load the TwinLiteNet model.

    This function initializes a TwinLiteNetPlus model, loads the specified pre-trained
    weights, and sets the model to evaluation mode.

    Parameters
    ----------
    weights : str, optional
        Path to the pre-trained weights file (default is 'Pretrained/tlp_medium.pth').
    config: str, optional
        Configuration of the model (default is 'medium').
    dev: torch.device, optional
        Device to load the model on (default is 'cpu').

    Returns
    -------
    TwinLiteNetPlus
        An instance of the TwinLiteNetPlus model loaded with the specified weights.

    """
    # Initialize the TwinLiteNetPlus model
    parser = argparse.ArgumentParser(description="TwinLiteNetPlus Model Loader.")
    parser.add_argument("--config", default=config)
    args = parser.parse_args()
    tlp_medium = TwinLiteNetPlus(args)
    # Load the pre-trained weights into the model

    print(f'Using device: {dev}')
    tlp_medium.load_state_dict(torch.load(weights, map_location=dev))
    # Set the model to evaluation mode
    tlp_medium.eval().to(dev)
    return tlp_medium


def preprocess_img(frame, img_size=640):
    img, ratio, pad = letterbox_for_img(frame, new_shape=(img_size, img_size), auto=True)
    h0, w0 = frame.shape[:2]  # orig hw
    h, w = img.shape[:2]
    shapes = (h0, w0), ((h / h0, w / w0), pad)

    # Convert
    # img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.array(img)
    img = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).unsqueeze(0) / 255

    _, _, height, width = img.shape
    pad_w, pad_h = shapes[1][1]
    pad_w = int(pad_w)
    pad_h = int(pad_h)
    ratio = shapes[1][0][1]

    return img, pad_h, pad_w, height, width, ratio


def process_output(output, pad_h, pad_w, height, width, ratio):
    da_seg_out, ll_seg_out = output

    da_predict = da_seg_out[:, :, pad_h:(height - pad_h), pad_w:(width - pad_w)]
    da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=int(1 / ratio), mode='bilinear')
    _, da_seg_mask = torch.max(da_seg_mask, 1)
    da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()

    ll_predict = ll_seg_out[:, :, pad_h:(height - pad_h), pad_w:(width - pad_w)]
    ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=int(1 / ratio), mode='bilinear')
    _, ll_seg_mask = torch.max(ll_seg_mask, 1)
    ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()

    return da_seg_mask, ll_seg_mask


def letterbox_for_img(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding

    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def show_seg_result(img, result, index, epoch, save_dir=None, is_ll=False, palette=None):
    # img = mmcv.imread(img)
    # img = img.copy()
    # seg = result[0]
    if palette is None:
        palette = np.random.randint(
            0, 255, size=(3, 3))
    palette[0] = [0, 0, 0]
    palette[1] = [0, 255, 0]
    palette[2] = [255, 0, 0]
    palette = np.array(palette)
    assert palette.shape[0] == 3  # len(classes)
    assert palette.shape[1] == 3
    assert len(palette.shape) == 2

    color_area = np.zeros((result[0].shape[0], result[0].shape[1], 3), dtype=np.uint8)

    # for label, color in enumerate(palette):
    #     color_area[result[0] == label, :] = color

    color_area[result[0] == 1] = [0, 255, 0]
    color_area[result[1] == 1] = [255, 0, 0]
    color_seg = color_area

    # convert to BGR
    color_seg = color_seg[..., ::-1]
    # print(color_seg.shape)
    color_mask = np.mean(color_seg, 2)
    img[color_mask != 0] = img[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
    # img = img * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)
    img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_LINEAR)
    img[0:30, :, :] = [0, 0, 0]
    img[690:, :, :] = [0, 0, 0]
    return img
