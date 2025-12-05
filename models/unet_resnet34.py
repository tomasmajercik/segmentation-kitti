import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetResNet34(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        self.layer0 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu
        )   # 1/2
        self.pool0 = backbone.maxpool              # 1/4
        self.layer1 = backbone.layer1              # 1/4
        self.layer2 = backbone.layer2              # 1/8
        self.layer3 = backbone.layer3              # 1/16
        self.layer4 = backbone.layer4              # 1/32

        # bottom
        self.bottom = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # decoder (5 levels)
        self.up4 = UpBlock(512, 256, 256)  # 1/32 → 1/16
        self.up3 = UpBlock(256, 128, 128)  # 1/16 → 1/8
        self.up2 = UpBlock(128, 64, 64)    # 1/8 → 1/4
        self.up1 = UpBlock(64, 64, 64)     # 1/4 → 1/2
        self.up0 = UpBlock(64, 0, 32)      # Fixed: skip_channels=0 since no skip connection

        self.final = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        # encoder
        e1 = self.layer0(x)          # 1/2, 64ch
        e2 = self.layer1(self.pool0(e1))  # 1/4, 64ch
        e3 = self.layer2(e2)         # 1/8, 128ch
        e4 = self.layer3(e3)         # 1/16, 256ch
        e5 = self.layer4(e4)         # 1/32, 512ch

        b = self.bottom(e5)

        # decoder
        d4 = self.up4(b, e4)   # 1/16
        d3 = self.up3(d4, e3)  # 1/8
        d2 = self.up2(d3, e2)  # 1/4
        d1 = self.up1(d2, e1)  # 1/2
        d0 = self.up0(d1, None)  # 1→1 (no skip, just upsample)

        return self.final(d0)