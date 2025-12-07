import torch
import torch.nn as nn
import torch.nn.functional as F
"""
advanced UNet++ model for semantic segmentation
"""
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNetPP(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        filters = [64, 128, 256, 512, 1024] # Number of filters at each level

        self.conv00 = ConvBlock(3, filters[0])
        self.conv10 = ConvBlock(filters[0], filters[1])
        self.conv20 = ConvBlock(filters[1], filters[2])
        self.conv30 = ConvBlock(filters[2], filters[3])
        self.conv40 = ConvBlock(filters[3], filters[4])

        self.conv01 = ConvBlock(filters[0] + filters[1], filters[0])
        self.conv11 = ConvBlock(filters[1] + filters[2], filters[1])
        self.conv21 = ConvBlock(filters[2] + filters[3], filters[2])
        self.conv31 = ConvBlock(filters[3] + filters[4], filters[3])

        self.conv02 = ConvBlock(filters[0] * 2 + filters[1], filters[0])
        self.conv12 = ConvBlock(filters[1] * 2 + filters[2], filters[1])
        self.conv22 = ConvBlock(filters[2] * 2 + filters[3], filters[2])

        self.conv03 = ConvBlock(filters[0] * 3 + filters[1], filters[0])
        self.conv13 = ConvBlock(filters[1] * 3 + filters[2], filters[1])

        self.conv04 = ConvBlock(filters[0] * 4 + filters[1], filters[0])

        self.pool = nn.MaxPool2d(2)
        self.up = lambda x: F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        self.final = nn.Conv2d(filters[0], n_classes, 1)

    def forward(self, x):
        x00 = self.conv00(x)
        x10 = self.conv10(self.pool(x00))
        x01 = self.conv01(torch.cat([x00, self.up(x10)], 1))

        x20 = self.conv20(self.pool(x10))
        x11 = self.conv11(torch.cat([x10, self.up(x20)], 1))
        x02 = self.conv02(torch.cat([x00, x01, self.up(x11)], 1))

        x30 = self.conv30(self.pool(x20))
        x21 = self.conv21(torch.cat([x20, self.up(x30)], 1))
        x12 = self.conv12(torch.cat([x10, x11, self.up(x21)], 1))
        x03 = self.conv03(torch.cat([x00, x01, x02, self.up(x12)], 1))

        x40 = self.conv40(self.pool(x30))
        x31 = self.conv31(torch.cat([x30, self.up(x40)], 1))
        x22 = self.conv22(torch.cat([x20, x21, self.up(x31)], 1))
        x13 = self.conv13(torch.cat([x10, x11, x12, self.up(x22)], 1))
        x04 = self.conv04(torch.cat([x00, x01, x02, x03, self.up(x13)], 1))

        return self.final(x04)
