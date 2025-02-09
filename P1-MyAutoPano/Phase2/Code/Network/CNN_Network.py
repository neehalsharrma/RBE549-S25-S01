"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code

Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

import torch.nn as nn
import sys
import torch
import numpy as np
import torch.nn.functional as F
import kornia  # You can use this to get the transform and warp in this project

# Don't generate pyc codes
sys.dont_write_bytecode = True


def LossFn(h_real, h_pred):
    ###############################################
    # Fill your loss function of choice here!
    ###############################################

    ###############################################
    # You can use kornia to get the transform and warp in this project
    # Bonus if you implement it yourself
    ###############################################
    criterion = nn.MSELoss()
    loss=criterion(h_real, h_pred)
    return loss


class BaseModel(nn.Module):
    def training_step(self, batch):
        i, h_real = batch
        h_pred = self.model(i)
        loss = LossFn(h_real, h_pred)
        return loss
    
    def validation_step(self, batch):
        i, h_real = batch
        h_pred = self.model(i)
        loss = LossFn(h_real, h_pred)
        return {"loss": loss.detach()}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        return {"avg_loss": avg_loss.item()}
    
    def epoch_end(self, epoch, result):
        # print("Epoch [{}], val_loss: {:.4f}, train_loss: {:.4f}, ".format(epoch, result['val_loss'], result['train_loss']))
        print("Epoch [{}], loss: {:.4f}".format(epoch, result['loss']))

def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class Net(BaseModel):
    def __init__(self, InputSize, OutputSize):
        """
        Inputs:
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """
        super().__init__()
        #############################
        # Fill your network initialization of choice here!
        #############################

        self.model= nn.Sequential(  conv_block(2, 64),     
                                    conv_block(64, 64, pool = True),
                                    conv_block(64, 64),
                                    conv_block(64, 64, pool = True),
                                    conv_block(64, 128),
                                    conv_block(128, 128, pool = True),
                                    conv_block(128, 128),
                                    conv_block(128, 128),
                                    nn.Dropout2d(0.4),
                                    nn.Flatten(),
                                    nn.Linear(16*16*128, 1024),
                                    nn.Dropout(0.4),
                                    nn.Linear(1024, 8))
        

    def forward(self, xa):
        """
        Input:
        xa is a MiniBatch of the image a
        xb is a MiniBatch of the image b
        Outputs:
        out - output of the network
        """
        #############################
        # Fill your network structure of choice here!
        #############################
        return self.model(xa)
