import torch
from torch import nn

class UpConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UpConvBlock, self).__init__()
        self.blk = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channel, out_channel,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

    def forward(self, x):
        return self.blk(x)



#[torch.Size([10, 128, 64, 64]), 
# torch.Size([10, 256, 32, 32]), 
# torch.Size([10, 512, 16, 16]), 
# torch.Size([10, 1024, 8, 8])]
class Resnet_decoder(nn.Module):
    def __init__(self) -> None:
        # 768 -> 384 , 384 + 384 = 768   16
        # 768 -> 384 , 384 + 192 = 576   32
        # 575 -> 288 , 288 + 96 = 384    64
        super().__init__()
        self.upconv1 = UpConvBlock(1024, 512)
        self.upconv2 = UpConvBlock(1024, 512)
        self.upconv3 = UpConvBlock(768, 384)
        self.upconv4 = UpConvBlock(512, 256)

        self.upconv5 = UpConvBlock(256, 64)

        self.final_conv = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, features):
        x = self.upconv1(features[3])
        x = torch.cat([x, features[2]], dim=1)

        x = self.upconv2(x)
        x = torch.cat([x, features[1]], dim=1)
        
        x = self.upconv3(x)
        x = torch.cat([x, features[0]], dim=1)
        
        x = self.upconv4(x)
        x = self.upconv5(x)
        
        x = self.final_conv(x)
        return x
