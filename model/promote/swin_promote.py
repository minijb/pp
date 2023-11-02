from typing import Iterable, Optional
import torch
from torch import nn
from torch.nn.modules.module import Module
from ..mm import Mlp
class Swin_promte(nn.Module):
    
    def __init__(self, channel_list , ss_list) -> None:
        super().__init__()

        out_size = 512
        
        self.promote_build = nn.ModuleList()
        
        for i in range(4):
            self.promote_build.append(
                liner_prom(channel_list[i], out_size, ss_list[i])
            )        
        
        self.fc_list = nn.ModuleList()
        
        for i in range(3):
            self.fc_list.append(
                Mlp(out_size*2, out_size, out_size, drop=0.2)
            )
        
        
    def forward(self, input_list):
        
        output_list1 = []
        
        for index, input in enumerate(input_list):
            out = self.promote_build[index](input)
            output_list1.append(out)
        
        output_list2 = [0,1,2]
        
        for i in range(2,-1,-1):
            line_cat = torch.cat([ output_list1[i+1],   output_list1[i]], dim= 1)
            output_list2[i] = self.fc_list[i](line_cat)
            
        for i, output in enumerate(output_list2):
            output_list2[i] = output_list2[i].unsqueeze(1)
   
        return output_list2
        
        
        
        
        
        


def flatten(input):
    b, c, h, w = input.shape
    return input.view(b, -1)

class liner_prom(nn.ModuleList):
    def __init__(self, in_size, out_size, feature_shape) :
        super().__init__()
        
        internal_size = in_size // 2
        
        if feature_shape == 8:
            self.conv = nn.Sequential(
                nn.Conv2d(in_size, internal_size, kernel_size = 3, stride = 1 , padding = 1, bias = False),
                nn.Conv2d(internal_size, internal_size//4, kernel_size = 3, stride = 1 , padding = 1, bias = False)
            )
            internal_size = internal_size//4
            
            
        elif feature_shape == 16:
            self.conv = nn.Sequential(
                nn.Conv2d(in_size, internal_size, kernel_size = 2 , stride = 2 , bias =  False),
                nn.Conv2d(internal_size, internal_size//2, kernel_size = 3, stride = 1 , padding = 1, bias = False)
            )
            internal_size = internal_size//2
            feature_shape = 8
        elif feature_shape == 32:
            self.conv = nn.Sequential(
                nn.Conv2d(in_size, internal_size, kernel_size = 4 , stride = 4 , bias =  False),
                nn.Conv2d(internal_size, internal_size//2, kernel_size = 3, stride = 1 , padding = 1, bias = False)
            )
            internal_size = internal_size//2
            feature_shape = feature_shape // 4
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_size, internal_size, kernel_size = 8 , stride = 8 , bias =  False),
                nn.Conv2d(internal_size, internal_size//8, kernel_size = 3, stride = 1 , padding = 1, bias = False)
            )
            internal_size = internal_size//8
            feature_shape = feature_shape // 8
        
        self.bn =  nn.BatchNorm2d(internal_size)
        
        fc_size = internal_size * feature_shape * feature_shape
        self.fc1 = nn.Linear(fc_size, fc_size//2)
        self.fc2 = nn.Linear(fc_size//2, out_size)
        
        self.relu = nn.ReLU()
        self.sigmod = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        
        x = flatten(x)
        
        
        x = self.relu(self.fc1(x))
        x = self.sigmod(self.fc2(x))
        return x        
    


    
