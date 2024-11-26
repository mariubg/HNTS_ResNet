import torch
import torch.nn as nn
from ResNet.ResBlock3D import ResBlock3D

class ResNetUNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        base_channels = 128
        self.in_channels = base_channels
        stride = 2
        
        
        # Encoder path (similar to original ResNet)
        self.enc_conv1 = nn.Conv3d(in_channels, base_channels, kernel_size=7, stride=2, padding=3)
        self.enc_bn1 = nn.BatchNorm3d(base_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # ResNet encoder ResBlock3Ds
        self.enc_layer1 = self._make_layer(ResBlock3D, base_channels, ResBlock3Ds=2)  # 64 channels
        self.enc_layer2 = self._make_layer(ResBlock3D, 2*base_channels, ResBlock3Ds=2, stride=stride)  # 128 channels
        self.enc_layer3 = self._make_layer(ResBlock3D, 4*base_channels, ResBlock3Ds=2, stride=stride)  # 256 channels
        self.enc_layer4 = self._make_layer(ResBlock3D, 8*base_channels, ResBlock3Ds=2, stride=stride)  # 512 channels
        
        # Decoder path
        self.dec_layer4 = nn.Sequential(
            nn.ConvTranspose3d(8*base_channels, 4*base_channels, kernel_size=2, stride=stride),
            nn.BatchNorm3d(4*base_channels),
            nn.ReLU()
        )
        self.dec_ResBlock3D4 = ResBlock3D(8*base_channels, 4*base_channels)  # 512 = 256 + 256 (skip connection)
        
        self.dec_layer3 = nn.Sequential(
            nn.ConvTranspose3d(4*base_channels, 2*base_channels, kernel_size=2, stride=stride),
            nn.BatchNorm3d(2*base_channels),
            nn.ReLU()
        )
        self.dec_ResBlock3D3 = ResBlock3D(4*base_channels, 2*base_channels)  # 256 = 128 + 128 (skip connection)
        
        self.dec_layer2 = nn.Sequential(
            nn.ConvTranspose3d(2*base_channels, base_channels, kernel_size=2, stride=stride),
            nn.BatchNorm3d(base_channels),
            nn.ReLU()
        )
        self.dec_ResBlock3D2 = ResBlock3D(2*base_channels, base_channels)  # 128 = 64 + 64 (skip connection)
        
        # Final upsampling and output
        self.dec_layer1 = nn.Sequential(
            nn.ConvTranspose3d(base_channels, int(0.5*base_channels), kernel_size=2, stride=stride),
            nn.BatchNorm3d(int(0.5*base_channels)),
            nn.ReLU()
        )
        self.final_upsample = nn.ConvTranspose3d(int(0.5*base_channels), int(0.5*base_channels), kernel_size=2, stride=2)
        self.final_conv = nn.Conv3d(int(0.5*base_channels), out_channels, kernel_size=1)

    def _make_layer(self, ResBlock3D, out_channels, ResBlock3Ds, stride=1):
        layers = []
        layers.append(ResBlock3D(self.in_channels, out_channels, stride=stride))
        self.in_channels = out_channels
        for _ in range(1, ResBlock3Ds):
            layers.append(ResBlock3D(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        print(x.shape)
        # Encoder path with skip connections
        x1 = self.relu(self.enc_bn1(self.enc_conv1(x)))
        x1_pool = self.maxpool(x1)
        
        x2 = self.enc_layer1(x1_pool)
        x3 = self.enc_layer2(x2)
        x4 = self.enc_layer3(x3)
        x5 = self.enc_layer4(x4)

        print("X:", x.shape)
        print("x1", x1.shape)
        print("x2:", x2.shape)
        print("x3", x3.shape)
        print("x4:", x4.shape)
        print("x5", x5.shape)
        
        # Decoder path with skip connections
        d4 = self.dec_layer4(x5)

        print("d4", d4.shape)

        d4_cat = torch.cat([d4, x4], dim=1)
        d4_out = self.dec_ResBlock3D4(d4_cat)
        
        d3 = self.dec_layer3(d4_out)
        d3_cat = torch.cat([d3, x3], dim=1)
        d3_out = self.dec_ResBlock3D3(d3_cat)
        
        d2 = self.dec_layer2(d3_out)
        d2_cat = torch.cat([d2, x2], dim=1)
        d2_out = self.dec_ResBlock3D2(d2_cat)
        
        d1 = self.dec_layer1(d2_out)

        d0 = self.final_upsample(d1)
        
        # Final convolution
        out = self.final_conv(d0)
        
        return out