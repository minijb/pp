from tools import wandb_init,setup_default_logging
from dataset import build_dataset, build_dataLoader
from model import build_swin,  build_pretrained, build_resnetDecoder, build_convnext, build_memoryBank, build_Decoder, build_proEncoder, MSFF,Main_model
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

def pretrained_step(mode = None):
    
    # TODO
    current_time = datetime.datetime.now()
    tail = str(current_time.day)+"_"+str(current_time.hour)
    
    # logging and init ------------------------------------------
    
    if use_wandb:
        if mode == "test":
            wandb_init("test", "test_pretrained_test"+tail, cfg = train_cfg_main['pretrain'])
        else:
            wandb_init("test", "test_pretrained"+tail, cfg = train_cfg_main['pretrain'])
    # dataset ---------------------------------------------------
    
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
    
    swin_backbone = build_swin(device, "./checkpoints/upernet_swin_base_patch4_window7_512x512.pth")
    resnet_decoder = build_resnetDecoder(device)
    
    pretrained_model = build_pretrained(swin_backbone, resnet_decoder, device)
    


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
    
    torch.save(swin_backbone.state_dict(),"./checkpoints/swin_encoder.pt")

def train(feature_exe):
    
    
    current_time = datetime.datetime.now()
    tail = str(current_time.day)+"_"+str(current_time.hour)
    
    if use_wandb:
        wandb_init("test", "test_train_"+tail, train_cfg_main['train'])
    
    
    # build dataset ------------------------------------------
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
    encoder_conv = build_convnext(device, "./checkpoints/convnext_base_1k_224.pth")
    
    
    if feature_exe:
        swin_backbone = build_swin(device,None)
        state_dict = torch.load(feature_exe)
        swin_backbone.load(state_dict)
    
    memoryBank = build_memoryBank(device, memory_dataset, 30)
    memoryBank.update(swin_backbone)
    del swin_backbone
    
    promote_encoder = build_proEncoder(device= device, channel_list=[128, 256, 512], size_list=[64, 32, 16])
    internal_model = MSFF().to(device)
    
    decoder = build_Decoder(device,"./checkpoints/sam_vit_h_4b8939.pth")
    
    main_model = Main_model(encoder_conv,internal_model, decoder, promote_encoder, memoryBank)

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
    
    

    
if __name__ == "__main__":
    

    
    pretrained_step()
    train(feature_exe="./checkpoints/swin_encoder.pt")
