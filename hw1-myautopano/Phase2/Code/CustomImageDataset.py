import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
import torch
import re

def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')

# Works for strucure of type
# Phase2/Data/Train
        # PA
            # 1_1A.jpg
            # 1_2A.jpg
            # 2_1A.jpg
            # 2_2A.jpg
            # .
            # .
        # PB
            # 1_1B.jpg
            # 1_2B.jpg
            # 2_1B.jpg
            # 2_2B.jpg
            # .
            # .

# img_dir includes ending //
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.Hs= pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        image_files = [str(img_dir+"PA//"+f) for f in os.listdir(img_dir+"PA\\") if f.endswith(".jpg")]
        self.img_paths1= sorted(image_files, key=extract_number)
        image_files = [str(img_dir+"PB//"+f) for f in os.listdir(img_dir+"PB\\") if f.endswith(".jpg")]
        self.img_paths2= sorted(image_files, key=extract_number)

    def __len__(self):
        return len(self.Hs)
    
    def __getitem__(self, idx):
        pa_path= self.img_paths1[idx]
        pb_path= self.img_paths2[idx]

        PA= read_image(pa_path)
        PB= read_image(pb_path)
        H= torch.tensor(self.Hs.iloc[idx].tolist())
        # Add transformation code if required

        return PA, PB, H