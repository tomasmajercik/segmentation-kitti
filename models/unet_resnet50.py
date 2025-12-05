import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from models.unet_resnet34 import UpBlock

class UNetResNet50(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        m = resnet50(weights="DEFAULT")
        
        self.in_layer = nn.Sequential(m.conv1, m.bn1, m.relu)  # 1/2, 64ch
        self.pool = m.maxpool                                   # 1/4
        self.enc1 = m.layer1                                    # 1/4, 256ch
        self.enc2 = m.layer2                                    # 1/8, 512ch
        self.enc3 = m.layer3                                    # 1/16, 1024ch
        self.enc4 = m.layer4                                    # 1/32, 2048ch
        
        # Decoder with correct channel counts for ResNet50
        self.up4 = UpBlock(2048, 1024, 512)  # 1/32 → 1/16
        self.up3 = UpBlock(512, 512, 256)    # 1/16 → 1/8
        self.up2 = UpBlock(256, 256, 128)    # 1/8 → 1/4
        self.up1 = UpBlock(128, 64, 64)      # 1/4 → 1/2
        self.up0 = UpBlock(64, 0, 32)        # 1/2 → 1/1 (no skip connection)
        
        self.final = nn.Conv2d(32, n_classes, 1)
    
    def forward(self, x):
        x0 = self.in_layer(x)      # 1/2, 64ch
        x1 = self.enc1(self.pool(x0))  # 1/4, 256ch
        x2 = self.enc2(x1)         # 1/8, 512ch
        x3 = self.enc3(x2)         # 1/16, 1024ch
        x4 = self.enc4(x3)         # 1/32, 2048ch
        
        d4 = self.up4(x4, x3)      # 1/16, 512ch
        d3 = self.up3(d4, x2)      # 1/8, 256ch
        d2 = self.up2(d3, x1)      # 1/4, 128ch
        d1 = self.up1(d2, x0)      # 1/2, 64ch
        d0 = self.up0(d1, None)    # 1/1, 32ch (final upsample to original size)
        
        return self.final(d0)