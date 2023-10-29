import torch
from torch import nn



class Main_model_swin_pic(nn.Module):
    def __init__(self, encoder, promote ,memorybank , internal_model, decoder):
        super().__init__()
        self.encoder = encoder
        self.promote = promote
        self.memorybank = memorybank
        self.internal_model = internal_model
        self.decoder = decoder
        
        
    def forward(self, input):
        
                
        features_list = list(self.encoder(input))
        
        dis_list  = self.memorybank.select(features_list[1:-1])
        dis_list.append(features_list[-1])
        
        promote = self.promote(dis_list[1:])
        nn.ModuleList()
        promote = self.internal_model(features_list[2:-1], promote)
        
        to_decoder = []
        to_decoder.append(features_list[0])
        to_decoder.append(features_list[1])
        to_decoder.extend(promote)
        to_decoder.append(features_list[-1])
        
        result = self.decoder(to_decoder)
        
        return result,to_decoder,features_list

        