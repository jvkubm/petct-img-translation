import torch
import torch.nn as nn
from unet_model import UNet3D

class Generator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Generator, self).__init__()
        # Use 3D UNet as the generator model
        self.unet = UNet3D(in_channels, out_channels)

    def forward(self, x):
        return self.unet(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        # Define the discriminator model
        self.model = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True), # check which ReLU is better here acc to literature
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(512, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid() # the question is if sigmoid here is indispensible, otherwise we could use nn.BCEWithLogitsLoss
            # but I think we need here probability in the output to say if the input is real or fake
        )

    def forward(self, x):
        return self.model(x)