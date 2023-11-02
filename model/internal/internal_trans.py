import torch 
from torch import nn
from ..decoder import Transformer
from ..mm import ResBlock
import math


class internal(nn.Module):
    def __init__(self, channel_list) -> None:
        super().__init__()
        
        self.conv_list = nn.ModuleList()
        for  index, channel in enumerate(channel_list):
            self.conv_list.append(conv_block(channel))
            
        self.trans_list = nn.ModuleList()
        self.trans_list.append(Transformer(512, 1, 8))
        self.trans_list.append(Transformer(512, 1, 8))
        self.trans_list.append(Transformer(512, 1, 8))

        
        
            
        
    def forward(self, feature_maps, promote_list):
        assert len(feature_maps) == len(promote_list)
        
        f_map  = []
        for i, feature_map in enumerate(feature_maps):
            temp = self.conv_list[i](feature_map)
            temp = mtt(temp)
            temp = torch.cat([promote_list[i], temp], dim=1)
            temp = temp.permute(1, 0 ,2)
            temp = self.trans_list[i]( temp)
            temp = temp.permute(1, 0, 2)
            temp = temp[:,1:,:]
            bc, HW, ch = temp.shape
            temp = temp.permute(0, 2, 1).view(bc, ch, int(math.sqrt(HW)),int(math.sqrt(HW)))
            temp = torch.cat([feature_map, temp], dim=1)
            
            f_map.append(temp)
            
        return f_map 
            
        
            

class conv_block(nn.Module):
    def __init__(self, channel) -> None:
        super().__init__()
        self.res = ResBlock(channel, 512, 1) 
    
    def forward(self, x):
        x = self.res(x)
        return x
    
def mtt(x):
    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = x.permute(0, 2, 1)
    return x 
