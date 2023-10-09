import torch
from .swinTransformer import SwinTransformer
from .ConvNext import ConvNeXt
from .memorybank import MemoryBank
from .decoder import TwoWayDecoder,TwoWayTransformer,Resnet_decoder
from .promote import Promot_encoder
from .pretrained_model import Pretrained

from tools import update_stateDict

def build_swin(device , pretrained: str):
    model = SwinTransformer(pretrain_img_size=256, embed_dim=128,
                                        depths=[2, 2, 18, 2],
                                        num_heads=[4, 8, 16, 32],
                                        window_size=7,
                                        ape=False,
                                        drop_path_rate=0.3,
                                        patch_norm=True)
    if device:
        model.to(device)
    if pretrained:
        model.init_weights(pretrained)
    
    return model

def build_convnext(device, pretrained: str):
    model = ConvNeXt(        
        in_chans=3,
        depths=[3, 3, 27, 3], 
        dims=[128, 256, 512, 1024], 
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        out_indices=[0, 1, 2, 3],
        )
    
    if device:
        model.to(device)
    if pretrained:
        model_stateDict = model.state_dict()
        stateDict = update_stateDict(model_stateDict, torch.load(pretrained)['model']) 
        model_stateDict.update(stateDict)
        del model_stateDict['downsample_layers.0.0.weight']
        model.load_state_dict(model_stateDict, strict=False)


    
    return model

def build_memoryBank(device, dataset, nb_memory_sample=30):
    memory_bank = MemoryBank(
        normal_dataset   = dataset,
        nb_memory_sample = nb_memory_sample,
        device           = device
    )
    
    return memory_bank


def build_Decoder(device, pretrained: str):
    model = TwoWayDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=256,
        )
    
    if device:
        model.to(device)
        
        
    if pretrained:
        model_state_dict = model.state_dict()
        save_state_dict = torch.load("./checkpoints/sam_vit_h_4b8939.pth")
        
        
        key_list = list(save_state_dict.keys())
        state_dict = {}
        
        
        for key in key_list:
            temp = ''
            if key.startswith("mask_decoder."):
                temp = key[13:]
                state_dict[temp] = save_state_dict[key]
        state_dict = {k:v for k,v in state_dict.items() if k in model_state_dict.keys()}
        model_state_dict.update(state_dict) 
        model.load_state_dict(model_state_dict) 
        
    return model  

def build_proEncoder(channel_list, size_list, device, pretrain=None):
    model = Promot_encoder(channel_list, size_list)
    
    if device:
        model.to(device)
    # if pretrained:
    
    return model

def build_pretrained(feature_exe, decoder, device):
    model = Pretrained(feature_exe, decoder)
    if device:
        model.to(device)
    return model

def build_resnetDecoder(device):
    decoder = Resnet_decoder()
    if device:
        decoder.to(device)
    return decoder


# build swin + resnetdecoder
def build_pretrained_model(device,swin_pretrained):
    feature_exe = build_swin(device,swin_pretrained)
    decoder = build_Decoder(device)
    
    model = build_pretrained(feature_exe, decoder)
    
    if device:
        model.to(device)
    
    return model