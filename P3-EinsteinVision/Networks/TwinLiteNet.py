import cv2
import numpy as np
import os
import torch
from TwinLiteNetPlus.model.model import TwinLiteNetPlus

def load_TwinLiteNet(weights:str = 'Pretrained/tlp_medium.pth') -> TwinLiteNetPlus:
    tlp_medium = TwinLiteNetPlus()
    tlp_medium.load_state_dict(torch.load(weights))
    tlp_medium.eval()
    return tlp_medium