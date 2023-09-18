import torch
from torch import nn


class Main_model(nn.Module):
    
    def __init__(self, encoder, internal,
                 decoder, promote_encoder, memorybank):
        super().__init__()
        self.encoder = encoder
        self.internal_conv = internal
        self.decoder = decoder
        self.promote_encoder = promote_encoder
        self.memorybank = memorybank
        
    def forward(self, inputs):
        features_list = list(self.encoder(inputs))
        encoder_embeding = self.internal_conv(features_list)
        
        
        promote = self.memorybank.select(features_list[0:-1])
        promote = self.promote_encoder(promote)
        
        out = self.decoder(encoder_embeding, promote)
        return out

        