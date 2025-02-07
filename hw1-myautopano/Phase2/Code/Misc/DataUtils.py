"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

import os
import cv2
import numpy as np
import random
import skimage
import PIL
import sys
import pandas as pd

# Don't generate pyc codes
sys.dont_write_bytecode = True


def SetupAll(BasePath, CheckPointPath):
    """
    Inputs:
    BasePath is the base path where Images are saved without "/" at the end
    CheckPointPath - Path to save checkpoints/model
    Outputs:
    DirNamesTrain - Variable with Subfolder paths to train files
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    ImageSize - Size of the image
    NumTrainSamples - length(Train)
    NumTestRunsPerEpoch - Number of passes of Val data with MiniBatchSize
    Trainabels - Labels corresponding to Train
    NumClasses - Number of classes
    """
    # Setup DirNames
    # DirNamesTrain = SetupDirNames(BasePath)

    # Read and Setup Labels
    relative_path = "hw1-myautopano/Phase2/Code"
    LabelsPathTrain = relative_path+'/TxtFiles/TrainLabels.csv'
    DirNamesTrainPA= relative_path+'/TxtFiles/DirNamesTrainPA.txt'
    DirNamesTrainPB= relative_path+'/TxtFiles/DirNamesTrainPB.txt'

    LabelsPathTest = relative_path+'/TxtFiles/TestLabels.csv'
    DirNamesTestPA= relative_path+'/TxtFiles/DirNamesTestPA.txt'
    DirNamesTestPB= relative_path+'/TxtFiles/DirNamesTestPB.txt'

    TrainLabels = ReadLabels(LabelsPathTrain)
    PaFilesTrain= ReadDirNames(DirNamesTrainPA)
    PbFilesTrain=ReadDirNames(DirNamesTrainPB)

    TestLabels = ReadLabels(LabelsPathTest)
    PaFilesTest= ReadDirNames(DirNamesTestPA)
    PbFilesTest=ReadDirNames(DirNamesTestPB)
    # If CheckPointPath doesn't exist make the path
    if not (os.path.isdir(CheckPointPath)):
        os.makedirs(CheckPointPath)

    # Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    SaveCheckPoint = 200
    # Number of passes of Val data with MiniBatchSize
    NumTestRunsPerEpoch = 1

    # Image Input Shape
    ImageSize = [128, 128, 1]
    NumTrainSamples = len(TrainLabels)

    return (
        PaFilesTrain,
        PbFilesTrain,
        PaFilesTest,
        PbFilesTest,
        SaveCheckPoint,
        ImageSize,
        NumTrainSamples,
        TrainLabels,
        TestLabels,
        NumTestRunsPerEpoch
    )


def ReadLabels(LabelsPathTrain):
    print(os.getcwd())
    if not (os.path.isfile(LabelsPathTrain)):
        print("ERROR: Train Labels do not exist in " + LabelsPathTrain)
        sys.exit()
    else:
        TrainLabels = pd.read_csv(LabelsPathTrain, header=None)

    return TrainLabels.values.tolist()

#EDIT# Not used 
# def SetupDirNames(BasePath):
#     """
#     Inputs:
#     BasePath is the base path where Images are saved without "/" at the end
#     Outputs:
#     Writes a file ./TxtFiles/DirNames.txt with full path to all image files without extension
#     """
#     DirNamesTrain = ReadDirNames("./TxtFiles/DirNamesTrain.txt")

#     return DirNamesTrain


def ReadDirNames(ReadPath):
    """
    Inputs:
    ReadPath is the path of the file you want to read
    Outputs:
    DirNames is the data loaded from ./TxtFiles/DirNames.txt which has full path to all image files without extension
    """
    # Read text files
    DirNames = open(ReadPath, "r")
    DirNames = DirNames.read()
    DirNames = DirNames.split()
    return DirNames
