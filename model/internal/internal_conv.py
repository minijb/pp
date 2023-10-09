from torch import nn   
import torch


class MSFF(nn.Module):
    def __init__(self):
        super(MSFF, self).__init__()
        self.blk1 = MSFFBlock(128)
        self.blk2 = MSFFBlock(256)
        self.blk3 = MSFFBlock(512)
        self.blk4 = MSFFBlock(1024)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.upconv43 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        )
        
        self.upconv32 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        )
        self.upconv21 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        )
        
        self.final_conv = nn.Conv2d(64, 256, kernel_size=3, padding=1)


    def forward(self, features):
        # features = [level1, level2, level3]
        f1, f2, f3, f4 = features 
        
        # MSFF Module
        f1_k = self.blk1(f1)
        f2_k = self.blk2(f2)
        f3_k = self.blk3(f3)
        f4_k = self.blk4(f4)

        f3_k = f3_k + self.upconv43(f4_k)
        f2_f = f2_k + self.upconv32(f3_k)
        f1_f = f1_k + self.upconv21(f2_f)

        # spatial attention
        
        # mask 
        m4 = f4[:,512:,...].mean(dim=1, keepdim=True)
        m3 = f3[:,256:,...].mean(dim=1, keepdim=True) * self.upsample(m4)
        m2 = f2[:,128:,...].mean(dim=1, keepdim=True) * self.upsample(m3)
        m1 = f1[:,64:,...].mean(dim=1, keepdim=True) * self.upsample(m2)
        
        f1_out = f1_f * m1

        f1_out = self.final_conv(f1_out)
        # f1_out = temp + f1_out
        
        
        return f1_out



class MSFFBlock(nn.Module):
    def __init__(self, in_channel):
        super(MSFFBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
        self.attn = CoordAtt(in_channel, in_channel)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // 2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channel // 2, in_channel // 2, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x_conv = self.conv1(x)
        x_att = self.attn(x)
        
        x = x_conv * x_att
        x = self.conv2(x)
        return x



class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d(output_size=(None, 1)) # X Avg Pool
        self.pool_w = nn.AdaptiveAvgPool2d(output_size=(1, None)) # Y Avg Pool

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        identity = x
        
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out