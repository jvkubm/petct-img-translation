import torch
import torch.nn as nn
from unet_model import UNet3D, UNet2D

# class Generator(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(Generator, self).__init__()
#         # Use 3D UNet as the generator model
#         self.unet = UNet3D(in_channels, out_channels)

#     def forward(self, x):
#         return self.unet(x)

# class Discriminator(nn.Module):
#     def __init__(self, in_channels):
#         super(Discriminator, self).__init__()
#         # Define the discriminator model
#         self.model = nn.Sequential(
#             nn.Conv3d(in_channels, 64, kernel_size=4, stride=2, padding=1),
#             nn.LeakyReLU(0.2, inplace=True), # check which ReLU is better here acc to literature
#             nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm3d(128),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm3d(256),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv3d(256, 512, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm3d(512),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv3d(512, 1, kernel_size=4, stride=1, padding=0),
#             nn.Sigmoid() # the question is if sigmoid here is indispensible, otherwise we could use nn.BCEWithLogitsLoss
#             # but I think we need here probability in the output to say if the input is real or fake
#         )

#     def forward(self, x):
#         return self.model(x)


class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, is_3d=True):
        super(Generator, self).__init__()
        self.is_3d = is_3d
        if is_3d:
            # Use 3D UNet as the generator model
            self.unet = UNet3D(in_channels, out_channels)
        else:
            # Use 2D UNet as the generator model
            self.unet = UNet2D(in_channels, out_channels)

    def forward(self, x):
        return self.unet(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels, is_3d=True):
        super(Discriminator, self).__init__()
        self.is_3d = is_3d
        if is_3d:
            # Define the 3D discriminator model
            self.model = nn.Sequential(
                nn.Conv3d(in_channels, 64, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
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
                nn.Sigmoid()
            )
        else:
            # Define the 2D discriminator model
            # now I changed to that input is 256x256, depending on input you need to adjust so that output is 1x1
            # solution is patching
            self.model = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1), # output shape: (in_channels, 64, H/2, W/2) assuming 256 -> 128
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # output shape: (1, 128, H/4, W/4), 64
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # output shape: (1, 256, H/8, W/8), 32
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), # output shape: (1, 512, H/16, W/16), 16
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1), # 8
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(1024, 2048, kernel_size=4, stride=2, padding=1), # 4
                nn.BatchNorm2d(2048),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(2048, 1, kernel_size=4, stride=1, padding=0),
                nn.Sigmoid()
            )
            # for input=(1,1,64,64) output=(1,1,1,1)
            # for input=(1,1,256,256) output=(1,1,13,13) why?
    def forward(self, x):
        return self.model(x)
    

# in_channels = 1
# out_channels = 1
# discriminator = Discriminator(in_channels + out_channels, is_3d=False)
# real = torch.ones((1, 1, 1, 1, 1), requires_grad=False)
# pet_images = torch.randn(1, 1, 64, 64)
# gen_ct_images = torch.randn(1, 1, 64, 64)
# input_discriminator = torch.cat((pet_images, gen_ct_images), 1)
# discriminator((input_discriminator), real)