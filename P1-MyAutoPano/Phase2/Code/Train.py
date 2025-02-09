#!/usr/bin/env python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""


# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)
# termcolor, do (pip install termcolor)

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.optim import AdamW
# from Network.CNN_Network import HomographyModel
import cv2
import sys
import os
import numpy as np
import random
import skimage
import PIL
import os
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import time
from Misc.MiscUtils import *
from Misc.DataUtils import *
from torchvision.transforms import ToTensor
import argparse
import shutil
import string
from termcolor import colored, cprint
import math as m
from tqdm import tqdm

from Network.CNN_Network import Net

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("     ###     DEVICE- ", device)
# Save file names in dictionary and then directly read and populate through setup
# Then just get randx path
def GenerateBatch(BasePath, DirNamesTrainPA, DirNamesTrainPB,TrainCoordinates, ImageSize, MiniBatchSize):
    """
    Inputs:
    BasePath - Path to COCO folder without "/" at the end
    DirNamesTrain - Variable with Subfolder paths to train files
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    TrainCoordinates - Coordinatess corresponding to Train
    NOTE that TrainCoordinates can be replaced by Val/TestCoordinatess for generating batch corresponding to validation (held-out testing in this case)/testing
    ImageSize - Size of the Image
    MiniBatchSize is the size of the MiniBatch
    Outputs:
    I1Batch - Batch of images
    CoordinatesBatch - Batch of coordinates
    """
    Input_Tensors=[]
    CoordinatesBatch = []

    ImageNum = 0
    while ImageNum < MiniBatchSize:
        # Generate random image
        RandIdx = random.randint(0, len(DirNamesTrainPA) - 1)

        RandImageNameA = DirNamesTrainPA[RandIdx] 
        RandImageNameB = DirNamesTrainPB[RandIdx] 
        # print(RandImageNameA,RandImageNameB, RandIdx, len(TrainCoordinates), len(DirNamesTrainPA))
        ImageNum += 1

        ##########################################################
        # Add any standardization or data augmentation here!
        ##########################################################
        PA = cv2.imread(RandImageNameA, cv2.IMREAD_GRAYSCALE)
        PB = cv2.imread(RandImageNameB, cv2.IMREAD_GRAYSCALE)

        PA = torch.tensor(PA, dtype=torch.float32) / 255.0
        PB = torch.tensor(PB, dtype=torch.float32) / 255.0

        PA= PA.unsqueeze(0)
        PB= PB.unsqueeze(0)
        input_tensor= torch.cat((PA, PB), dim=0)

        Coordinates = TrainCoordinates[RandIdx]

        # Append All Images and Mask
        Input_Tensors.append(input_tensor)
        CoordinatesBatch.append(torch.tensor(Coordinates))

    return torch.stack(Input_Tensors).to(device), torch.stack(CoordinatesBatch).to(device)

def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print("Number of Epochs Training will run for " + str(NumEpochs))
    print("Factor of reduction in training data is " + str(DivTrain))
    print("Mini Batch Size " + str(MiniBatchSize))
    print("Number of Training Images " + str(NumTrainSamples))
    if LatestFile is not None:
        print("Loading latest checkpoint with the name " + LatestFile)

def evaluate(model,BasePath, DirNamesTestPA, DirNamesTestPB,TestCoordinates, ImageSize, MiniBatchSize, NumTestRunsPerEpoch):
    model.eval()

    outputs=[]
    for _ in range(NumTestRunsPerEpoch):
        NumIterationsPerEpoch = len(TestCoordinates) // MiniBatchSize
        # NumIterationsPerEpoch=2
        for _ in range(NumIterationsPerEpoch):
            I, CoordinatesBatch = GenerateBatch(BasePath, DirNamesTestPA, DirNamesTestPB, TestCoordinates, ImageSize, MiniBatchSize)
            outputs.append(model.validation_step((I, CoordinatesBatch)))
    
    return model.validation_epoch_end(outputs)

def save_loss_data(data, filename):
    with open(filename, "a") as file:
        file.write(",".join(map(str, data)) + "\n")
    print("Loss data saved to ", filename)
        

def open_new_file(filename):
    with open(filename, "w") as file:
        file.write("train_loss, test_loss, epoch\n")
    print("Created new file ", filename)

def TrainOperation(
    DirNamesTrainPA,
    DirNamesTrainPB,
    TrainCoordinates,
    DirNamesTestPA,
    DirNamesTestPB,
    TestCoordinates,
    NumTrainSamples,
    ImageSize,
    NumEpochs,
    MiniBatchSize,
    SaveCheckPoint,
    CheckPointPath,
    DivTrain,
    LatestFile,
    BasePath,
    LogsPath,
    ModelType,
    NumTestRunsPerEpoch
):
    """
    Inputs:
    ImgPH is the Input Image placeholder
    DirNamesTrain - Variable with Subfolder paths to train files
    TrainCoordinates - Coordinates corresponding to Train/Test
    NumTrainSamples - length(Train)
    ImageSize - Size of the image
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
    LatestFile - Latest checkpointfile to continue training
    BasePath - Path to COCO folder without "/" at the end
    LogsPath - Path to save Tensorboard Logs
        ModelType - Supervised or Unsupervised Model
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath
    """
    # Predict output with forward pass
    model = Net(128, 128)
    model = model.to(device)

    ###############################################
    # Fill your optimizer of choice here!
    ###############################################
    Optimizer = torch.optim.SGD(model.parameters(), lr = 0.0001, momentum = 0.9)

    # Tensorboard
    # Create a summary to monitor loss tensor
    Writer = SummaryWriter(LogsPath)

    if os.path.exists("loss_log.csv"):
        df= pd.read_csv("loss_log.csv")
        loss_log_capture=df.to_dict(orient='records')
    else:
        loss_log_capture=[]

    if LatestFile is not None:
        CheckPoint = torch.load(LatestFile)
        # Extract only numbers from the name
        StartEpoch = int("".join(c for c in LatestFile.split("a")[0] if c.isdigit()))
        model.load_state_dict(CheckPoint["model_state_dict"])
        print("Loaded latest checkpoint with the name " + LatestFile + "....")
    else:
        StartEpoch = 0
        print("New model initialized....")

    for Epochs in tqdm(range(StartEpoch, NumEpochs)):
        NumIterationsPerEpoch = int(NumTrainSamples / MiniBatchSize / DivTrain)
        # NumIterationsPerEpoch=2
        model.train() #Set model to train mode
        outputs=[]
        # for batch 
        for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
            # Generate Mini Batch for training
            I, CoordinatesBatch = GenerateBatch(
                BasePath, DirNamesTrainPA, DirNamesTrainPB, TrainCoordinates, ImageSize, MiniBatchSize
            )


            # Predict output with forward pass
            LossThisBatch = model.training_step((I, CoordinatesBatch))
            Optimizer.zero_grad()
            LossThisBatch.backward()
            Optimizer.step()

            result={}
            result["loss"]= LossThisBatch
            outputs.append(result)

            # Save checkpoint every some SaveCheckPoint's iterations
            if PerEpochCounter % SaveCheckPoint == 0:
                # Save the Model learnt in this epoch
                SaveName = (
                    CheckPointPath
                    + str(Epochs)
                    + "a"
                    + str(PerEpochCounter)
                    + "model.ckpt"
                )

                torch.save(
                    {
                        "epoch": Epochs,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": Optimizer.state_dict(),
                        "loss": LossThisBatch,
                    },
                    SaveName,
                )
                print("\n" + SaveName + " Model Saved...")

            # Tensorboard
            Writer.add_scalar('LossEveryIter', result["loss"], Epochs*NumIterationsPerEpoch + PerEpochCounter)
            # If you don't flush the tensorboard doesn't update until a lot of iterations!
            Writer.flush()

        # Run model evaluation
        test_result = evaluate(model,BasePath, DirNamesTestPA, DirNamesTestPB,TestCoordinates, ImageSize, MiniBatchSize, NumTestRunsPerEpoch)
        train_result= model.validation_epoch_end(outputs)
        # Just to print test loss and train
        model.epoch_end(Epochs, result) 

        # Save results in a single variable for saving
        result={"test_loss": test_result["avg_loss"], "train_loss": train_result["avg_loss"], "epoch": Epochs}
        loss_log_capture.append(result)

        # Tensorboard
        Writer.add_scalar('TestLossEveryEpoch', result["test_loss"], Epochs)
        Writer.add_scalar('TrainLossEveryEpoch', result["train_loss"], Epochs)

        # Save model every epoch
        SaveName = CheckPointPath + str(Epochs) + "model.ckpt"
        torch.save(
            {
                "epoch": Epochs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": Optimizer.state_dict(),
                "loss": LossThisBatch,
            },
            SaveName,
        )
        print("\n" + SaveName + " Model Saved...")

        save_loss_data([result["train_loss"], result["test_loss"], Epochs+1], "loss_log.csv")


def main():
    """
    Inputs:
    # None
    # Outputs:
    # Runs the Training and testing code based on the Flag
    #"""
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--BasePath",
        default="/home/lening/workspace/rbe549/YourDirectoryID_p1/Phase2/Data",
        help="Base path of images, Default:/home/lening/workspace/rbe549/YourDirectoryID_p1/Phase2/Data",
    )
    Parser.add_argument(
        "--CheckPointPath",
        default="../Checkpoints/",
        help="Path to save Checkpoints, Default: ../Checkpoints/",
    )

    Parser.add_argument(
        "--ModelType",
        default="Unsup",
        help="Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup",
    )
    Parser.add_argument(
        "--NumEpochs",
        type=int,
        default=50,
        help="Number of Epochs to Train for, Default:50",
    )
    Parser.add_argument(
        "--DivTrain",
        type=int,
        default=1,
        help="Factor to reduce Train data by per epoch, Default:1",
    )
    Parser.add_argument(
        "--MiniBatchSize",
        type=int,
        default=64,
        help="Size of the MiniBatch to use, Default:1",
    )
    Parser.add_argument(
        "--LoadCheckPoint",
        type=int,
        default=0,
        help="Load Model from latest Checkpoint from CheckPointsPath?, Default:0",
    )
    Parser.add_argument(
        "--LogsPath",
        default="Logs/",
        help="Path to save Logs for Tensorboard, Default=Logs/",
    )

    

    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    BasePath = Args.BasePath
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath
    ModelType = Args.ModelType

    # Setup all needed parameters including file reading
    (
        DirNamesTrainPA,
        DirNamesTrainPB,
        DirNamesTestPA,
        DirNamesTestPB,
        SaveCheckPoint,
        ImageSize,
        NumTrainSamples,
        TrainCoordinates,
        TestCoordinates,
        NumTestRunsPerEpoch
    ) = SetupAll(BasePath, CheckPointPath)

    # Find Latest Checkpoint File
    if LoadCheckPoint == 1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None
        open_new_file("loss_log.csv")

    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

    TrainOperation(
        DirNamesTrainPA,
        DirNamesTrainPB,
        TrainCoordinates,
        DirNamesTestPA,
        DirNamesTestPB,
        TestCoordinates,
        NumTrainSamples,
        ImageSize,
        NumEpochs,
        MiniBatchSize,
        SaveCheckPoint,
        CheckPointPath,
        DivTrain,
        LatestFile,
        BasePath,
        LogsPath,
        ModelType,
        NumTestRunsPerEpoch
    )


if __name__ == "__main__":
    main()