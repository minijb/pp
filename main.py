from tools import wandb_init,setup_default_logging
from dataset import build_dataset, build_dataLoader
from model import build_swin,  build_pretrained, build_resnetDecoder, build_convnext, build_memoryBank, build_Decoder, build_proEncoder, MSFF,Main_model
from model import Swin_promte, internal, Swin_decoder,Main_model_swin
from train import pretrained_train, train_step
import wandb
import os
from config import cfg
import logging
import torch
import datetime

dataset_cfg = cfg['dataset']

train_cfg_main = cfg['train']
pretrained_cfg = cfg['train']['pretrain']
train_cfg = cfg['train']['train']

_logger = logging.getLogger('train')

setup_default_logging()
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda:0")
#device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
_logger.info('Device: {}'.format(device))
use_wandb = train_cfg_main['use_wandb']

def pretrained_step(mode = None, target = None):
    
    
    current_time = datetime.datetime.now()
    tail = str(current_time.day)+"_"+str(current_time.hour)
    
    # logging and init ------------------------------------------

    use_wandb = False
    
    if use_wandb:
        if mode == "test":
            wandb_init("test", "test_pretrained_test"+tail, cfg = train_cfg_main['pretrain'])
        else:
            wandb_init("test", "pretrained"+target+tail, cfg = train_cfg_main['pretrain'])
    # dataset ---------------------------------------------------
    
    
    if target : 
        dataset_cfg['target'] = target
    
    pretrained_dataset = build_dataset(
        datadir = dataset_cfg['datadir'],
        texturedir = dataset_cfg['texturedir'],
        target = dataset_cfg['target'],
        train = True,
        to_memory = True
    )
    
    pretrained_dataloader = build_dataLoader(
        dataset = pretrained_dataset,
        train = True,
        batch_size = 16,
    )
    
    # model ------------------------------------------------
    
    # swin_backbone = build_swin(device, "./checkpoints/upernet_swin_base_patch4_window7_512x512.pth")
    convnext = build_convnext(device, "./checkpoints/convnext_base_1k_224.pth")
    resnet_decoder = build_resnetDecoder(device)
    
    pretrained_model = build_pretrained(convnext, resnet_decoder, device)
    


    if mode == "test":   
        time_str = ""+ str(current_time.month) + "_"+ str(current_time.day)+ "_"+ str(current_time.hour)
        trace_dir = "./tracing/pretrained_"+ time_str + ".csv"
    else:
        trace_dir = None
        mode = None
    
    if mode == "test":
        num_step = pretrained_cfg['test_step']
    else:
        num_step = pretrained_cfg['num_step']
    
    pretrained_train(
        model = pretrained_model,
        trainloader = pretrained_dataloader,
        device=device,
        num_training_steps=num_step,
        mode = mode,
        savedir=trace_dir,
        use_wandb=use_wandb)
    
    wandb.finish()
    
    torch.save(convnext.state_dict(),"./checkpoints/conv_encoder.pt")
    use_wandb = True
def train(feature_exe, target = None):
    
    
    current_time = datetime.datetime.now()
    tail = str(current_time.day)+"_"+str(current_time.hour)
    
    if use_wandb:
        wandb_init("swin_train", "train_"+item+tail, train_cfg_main['train'])
    
    
    # build dataset ------------------------------------------
    
    if target : 
        dataset_cfg['target'] = target
        
    trian_dataset = build_dataset(
        **dataset_cfg,
        train=True
    )
    
    memory_dataset = build_dataset(
        **dataset_cfg,
        train=True,
        to_memory=True
    )
    
    test_dataset = build_dataset(
        **dataset_cfg,
        train=False
    )
    
    trainloader = build_dataLoader(
        dataset=trian_dataset,
        train=True
    )
    
    testloader = build_dataLoader(
        dataset= test_dataset,
        train=False
    )
    
    # build model ------------------------------------------
    encoder_conv = build_convnext(device,  feature_exe)
    
    for name, parameter in encoder_conv.named_parameters():
        parameter.requires_grad = False
    
    memoryBank = build_memoryBank(device, memory_dataset, 30)
    memoryBank.update(encoder_conv)
   
    channel_list = [256, 512, 1024]
    ss_list = [32, 16, 8]
    promte_mode = Swin_promte(channel_list, ss_list)
    promte_mode.to(device)
    
    internal_model = internal(channel_list[0:-1])
    internal_model.to(device)
    
    decoder = Swin_decoder()
    decoder.to(device)
    
    main_model = Main_model_swin(encoder_conv, promte_mode, memoryBank, internal_model, decoder)
    main_model.to(device)
    
    
    train_step(
        model=main_model,
        dataloader=trainloader,
        validloader=testloader,
        num_training_steps=train_cfg['num_step'],
        log_interval=1,
        eval_interval= 50,
        device=device,
        use_wandb=use_wandb,
        savedir="./save/"+dataset_cfg['target']
    )
    wandb.finish()
    

# def delete():
#     promote_encoder = build_proEncoder(device= device, channel_list=[128, 256, 512], size_list=[64, 32, 16])
#     internal_model = MSFF().to(device)
    
#     decoder = build_Decoder(device,"./checkpoints/sam_vit_h_4b8939.pth")
    
#     main_model = Main_model(feature_exe,internal_model, decoder, promote_encoder, memoryBank)


    
if __name__ == "__main__":
    
    # target_list = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
    # for item in target_list:   \
    item = "cable"
    pretrained_step(target=item)
    train(feature_exe="./checkpoints/convnext_base_1k_224.pth", target=item)
