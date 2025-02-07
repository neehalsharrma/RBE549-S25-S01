import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

import numpy as np
import CustomImageDataset
from Phase2.Code.Network.TensorDLT import TensorDLT

dataset = CustomImageDataset.UnsupervisedDataSet('C_A.csv', '../Data/', test=False)

for i in range(2):
    pa, pb, img_a, ca = dataset[i]
    print(pa.shape, pb.shape, ca.shape, img_a.shape)

    B = img_a.shape[0]  # Batch size
    x_start = ca[:, 0, 0]  # Extract x-coordinates
    y_start = ca[:, 1, 0]  # Extract y-coordinates

    # Use advanced indexing to extract patches
    out = torch.stack([
        img_a[i, :, y_start[i]:y_start[i] + 128, x_start[i]:x_start[i] + 128]
        for i in range(B)
    ])

    print(out.shape)



    # dlt = TensorDLT()
    #
    # H = dlt(ca[0:2, :, :], torch.tensor([[0, 0, 128, 0, 0, 128, 128, 128],
    #                         [-213, -181, -219, 219, -15, -212, 15, 312]], dtype=torch.float64).unsqueeze(2))
