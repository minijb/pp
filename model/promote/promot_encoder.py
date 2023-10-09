import torch
from torch import nn
from ..Attention import CoTAttention
from ..mm import LayerNorm2d

# need torch.randn(10,4,256)

# in : [torch.Size([10, 128, 64, 64]), torch.Size([10, 256, 32, 32]), torch.Size([10, 512, 16, 16])]

# use traditional se atttention


def build_conv(channel, size):
    if size == 64:
        return nn.Sequential(
            downsample(channel, channel//2, channel),
            downsample(channel, channel//2, channel)
        )
    elif size == 32:
        return nn.Sequential(
            downsample(channel, channel//2, channel//2)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(channel, channel//2, kernel_size=3, padding=1)
        )


class Promot_encoder(nn.Module):

    def __init__(self, channel_list, size_list) -> None:
        super().__init__()

        self.atten_layers = nn.ModuleList()
        for channel in channel_list:
            layer = CoTAttention(dim=channel)
            self.atten_layers.append(layer)

        self.conv_layers = nn.ModuleList()
        for i, channel in enumerate(channel_list):
            block = build_conv(channel, size_list[i])
            self.conv_layers.append(block)
        
        self.final_conv = nn.Conv1d(512, 128, kernel_size = 3, padding=1)

    def forward(self, feature_list):

        out_list = []
        for index, feature in enumerate(feature_list):
            out_list.append(self.atten_layers[index](feature))

        for index, out in enumerate(out_list):
            out_list[index] = self.conv_layers[index](out)

        output_cat = torch.cat(out_list, dim = 1)
        b, c, h, w = output_cat.shape
        output_cat = output_cat.view(b,c,-1)
        
        output_cat = self.final_conv(output_cat)
        return output_cat

class downsample(nn.ModuleDict):

    def __init__(self, in_channel, interal_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, interal_channel,
                               kernel_size=4, padding=1, stride=2)
        self.LN = LayerNorm2d(interal_channel)
        self.conv2 = nn.Conv2d(
            interal_channel, out_channel, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.LN(x)
        x = self.conv2(x)
        return x

