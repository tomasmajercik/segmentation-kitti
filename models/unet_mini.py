import torch
import torch.nn as nn
import torch.nn.functional as F
"""
reduced UNet model for semantic segmentation
"""
class DoubleConv(nn.Module):# two consecutive convolutional layers with batch normalization and ReLU activation
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):# reduced UNet model for semantic segmentation
    def __init__(self, n_classes: int):
        super().__init__()

        # Encoder
        self.down1 = DoubleConv(3, 32)
        self.down2 = DoubleConv(32, 64)
        self.down3 = DoubleConv(64, 128)
        self.down4 = DoubleConv(128, 256)

        # Bottleneck
        self.bottom = DoubleConv(256, 512)

        # Decoder
        self.up4 = DoubleConv(512 + 256, 256)
        self.up3 = DoubleConv(256 + 128, 128)
        self.up2 = DoubleConv(128 + 64, 64)
        self.up1 = DoubleConv(64 + 32, 32)

        # Final layer
        # outputs shape: [N, n_classes, H, W]
        self.final = nn.Conv2d(32, n_classes, kernel_size=1)

        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        # Encoder
        c1 = self.down1(x)
        c2 = self.down2(self.maxpool(c1))
        c3 = self.down3(self.maxpool(c2))
        c4 = self.down4(self.maxpool(c3))

        # Bottom
        b = self.bottom(self.maxpool(c4))

        # Decoder
        u4 = F.interpolate(b, scale_factor=2, mode='bilinear', align_corners=False)
        u4 = torch.cat([u4, c4], dim=1)
        u4 = self.up4(u4)

        u3 = F.interpolate(u4, scale_factor=2, mode='bilinear', align_corners=False)
        u3 = torch.cat([u3, c3], dim=1)
        u3 = self.up3(u3)

        u2 = F.interpolate(u3, scale_factor=2, mode='bilinear', align_corners=False)
        u2 = torch.cat([u2, c2], dim=1)
        u2 = self.up2(u2)

        u1 = F.interpolate(u2, scale_factor=2, mode='bilinear', align_corners=False)
        u1 = torch.cat([u1, c1], dim=1)
        u1 = self.up1(u1)

        return self.final(u1)
