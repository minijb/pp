from typing import Iterable, Optional
import torch
from torch import nn
from torch.nn.modules.module import Module
from ..mm import ResBlock


# [torch.Size([10, 128, 64, 64]),
#  torch.Size([10, 768, 32, 32]),
#  torch.Size([10, 1024, 16, 16]),
#  torch.Size([10, 1024, 8, 8])]

class UpConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UpConvBlock, self).__init__()
        self.blk = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

    def forward(self, x):
        return self.blk(x)


class Swin_decoder(nn.ModuleList):
    def __init__(self) -> None:
        super().__init__()
        
        self.upconv3 = UpConvBlock(1024, 512)
        self.upconv2 = UpConvBlock(1536, 768)
        self.upconv1 = UpConvBlock(1536, 768)
        self.upconv0 = UpConvBlock(1408, 256)
        self.upconv = UpConvBlock(384, 64)

        self.final_conv = nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1)
        
        
    def forward(self, feature_maps):
        
        out3 = self.upconv3(feature_maps[4]) # 1024 --> 512  8 --> 16
        out3 = torch.cat([out3, feature_maps[3] ], dim=1) # 512 + 1024 = 1536 
        
        out2 = self.upconv2(out3) # 1536 --> 768 ; 16 -> 32
        out2 = torch.cat([out2, feature_maps[2]], dim=1) # 768 * 2 = 1,536 
        
        out1 = self.upconv1(out2) # 1536 -> 768 32 -> 64
        out1 = torch.cat([out1, feature_maps[1]], dim=1) #768 + 128 = 896
        
        out0 = self.upconv0(out1)
        out0 = torch.cat([out0, feature_maps[0]], dim=1) #256 + 128 = 384
        out = self.upconv(out0)
        out = self.final_conv(out)
        
        return out
        
        