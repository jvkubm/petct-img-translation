import torch
import torch.nn as nn

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, init_features=32):
        super(UNet3D, self).__init__()
        features = init_features
        # Encoder path
        # in the block the feature size stays the same (padding=1)
        # so the reduction in size is done only by maxpooling
        self.encoder1 = UNet3D._block(in_channels, features)
        self.encoder2 = UNet3D._block(features, features * 2)
        self.encoder3 = UNet3D._block(features * 2, features * 4)
        self.encoder4 = UNet3D._block(features * 4, features * 8)
        # Pooling
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2) # reduction by 2 in each dimension
        
        # Bottleneck
        self.bottleneck = UNet3D._block(features * 8, features * 16)
        
        # Decoder path
        self.upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2) # in_channels, out_channels, kernel_size, stride
        # so one deconv layer doubles the size of the input and halves the number of channels
        self.decoder4 = UNet3D._block((features * 8) * 2, features * 8)
        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet3D._block((features * 4) * 2, features * 4)
        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet3D._block((features * 2) * 2, features * 2)
        self.upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = UNet3D._block(features * 2, features)
        
        

        # Final convolution
        self.conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder path
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1) # Skip connection
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        # Final convolution
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name=""):
        # Convolutional block with Conv3D, BatchNorm3D, and ReLU
        return nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=features, kernel_size=3, padding=1, bias=False), 
            # stride default is 1
            # padding is added to all six sides of the input, default=0
            # padding_mode default is zeros
            # so this way in the block the output size is the same as the input size
            # bias adds a learnable bias to the output
            # dilation default is 1, spacing between kernel elements, so its classic, dilation=2 would be Atrous Convolution (dilated convolution)
            # atrous can be used to increase RF size without increasing the number of parameters of the model
            nn.BatchNorm3d(num_features=features),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(num_features=features),
            nn.ReLU(inplace=True),
        )