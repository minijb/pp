import torch
from torch import nn

class Pretrained(nn.Module):
    def __init__(self, feature_exe, decoder):
        super().__init__()
        self.feature_exe = feature_exe
        self.decoder = decoder
        
        
    def forward(self, input):
        out = self.feature_exe(input)
        out = self.decoder(out)
        return out