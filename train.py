from torch import nn
import torch
from tools import AverageMeter,wandb_log_train
from optimzer import AdamW
from config import cfg
from loss import FocalLoss
from timm.scheduler.cosine_lr import CosineLRScheduler
import wandb


from anomalib.utils.metrics import AUPRO, AUROC
import logging
import torch.nn.functional as F
import numpy as np
import json
import os

_logger = logging.getLogger('train')

main_cfg = cfg['train']
pretrained_cfg = main_cfg['pretrain']
train_cfg = main_cfg['train']

def pretrained_train(model:nn.Module, trainloader,device, num_training_steps:int, 
               log_interval: int = 1, savedir: str = None, use_wandb: bool = False, mode:str = "test"):
    
    # something in train  --------------------------------------------------
    
    # loss
    smooth_loss = nn.SmoothL1Loss()
    
    # optimizer
    optimizer = AdamW(model.parameters(), lr  = pretrained_cfg['lr'])
    
    # tracing ------------------------------------------------------
    
    smooth_loss_trace = AverageMeter()

    # train ----------------------------------------------------------
    
    model.train()
    optimizer.zero_grad()
    
    
    step = 0
    train_model = True
    
    if mode == "test":
        file = open(savedir,"w")
    
    while train_model:
    
        for inputs, _, _ in trainloader:
            
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            sm_loss = smooth_loss(outputs, inputs)
            
            sm_loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
            # ----- update tracing
            
            smooth_loss_trace.update(sm_loss.item())
            
            if use_wandb:
                wandb_log_train({
                    'lr': optimizer.param_groups[0]['lr'],
                    "loss" : smooth_loss_trace.val
                }, step = step)
                
            if (step+1) % log_interval == 0 or step == 0: 
                _logger.info('TRAIN [{:>4d}/{}] '
                        'Loss: {loss.val:>6.4f} ({loss.avg:>6.4f}) '
                        'LR: {lr:.3e} '.format(
                        step+1, num_training_steps, 
                        loss       = smooth_loss_trace, 
                        lr         = optimizer.param_groups[0]['lr'])
                        )
            
            if mode == "test":
                file.write("{step}:{loss}:{loss_avg}\n".format(step = step , 
                                                             loss = smooth_loss_trace.val,
                                                             loss_avg = smooth_loss_trace.avg
                                                             ))
            
            step = step + 1
            
            if step == num_training_steps:
                train_model = False
                break
    
    if mode == "test":
        file.close()
    
    

def train_step(model: nn.Module, dataloader, validloader,num_training_steps,log_interval,eval_interval, device, use_wandb,savedir):
    
    # something in train  --------------------------------------------------
    
    # loss
    smooth_l1_loss = nn.SmoothL1Loss()
    focal_loss = FocalLoss(
        gamma = train_cfg['focal_gamma'],
        alpha = train_cfg['focal_alpha']
    )
    
    criterion = [smooth_l1_loss, focal_loss] 
    
    # optimizer
    optimizer = AdamW(model.parameters(), lr  = train_cfg['lr'], eps=1e-4)

    
    scheduler_cfg = train_cfg['scheduler']
    scheduler = CosineLRScheduler(
        optimizer= optimizer,
        **scheduler_cfg
    )

    
    # tracing
    loss_tracing = AverageMeter()
    
    # metrics
    auroc_image_metric = AUROC(num_classes=1, pos_label=1)
    auroc_pixel_metric = AUROC(num_classes=1, pos_label=1)
    aupro_pixel_metric = AUPRO()
    
    # train ---------------------------------
    model.train()
    optimizer.zero_grad()
    
    best_score = 0
    step = 0
    train_mode = True
    
    while train_mode:
        
        for inputs, masks, targets in dataloader:
            
            inputs, masks, targets = inputs.to(device), masks.to(device), targets.to(device)

            outputs = model(inputs)
            outputs = F.softmax(outputs, dim=1)
            smooth_loss_c = smooth_l1_loss(outputs[:,1,:], masks)
            focal_loss_c = focal_loss(outputs,masks)
            loss = (0.6 * smooth_loss_c) + (0.4 * focal_loss_c)
    
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
            loss_tracing.update(loss.item())
            if np.isnan(loss_tracing.val):
                print("focal_loss : ",focal_loss_c.item())
                print("smooth_loss : ",smooth_loss_c.item())
                return
            
            if use_wandb:
                wandb_log_train({
                    'lr': optimizer.param_groups[0]['lr'],
                    "loss" : loss_tracing.val
                }, step = step)
            
            if (step+1) % log_interval == 0 or step == 0: 
                _logger.info('TRAIN [{:>4d}/{}] '
                        'Loss: {loss.val:>6.4f} ({loss.avg:>6.4f}) '
                        'LR: {lr:.3e} '.format(
                        step+1, num_training_steps, 
                        loss       = loss_tracing, 
                        lr         = optimizer.param_groups[0]['lr'])
                        )
            if ((step+1) % eval_interval == 0 and step != 0) or (step+1) == num_training_steps: 
                eval_metrics = evaluate(
                    model        = model, 
                    dataloader   = validloader, 
                    criterion    = criterion, 
                    log_interval = log_interval,
                    metrics      = [auroc_image_metric, auroc_pixel_metric, aupro_pixel_metric], 
                    device       = device
                )
                model.train()
                eval_log = dict([(f'eval_{k}', v) for k, v in eval_metrics.items()])
                if use_wandb:
                    wandb.log(eval_log, step=step)
                if best_score < np.mean(list(eval_metrics.values())):
                    # save best score
                    state = {'best_step':step}
                    state.update(eval_log)
                    json.dump(state, open(os.path.join(savedir, 'best_score.json'),'w'), indent='\t')

                    # save best model
                    torch.save(model.state_dict(), os.path.join(savedir, f'best_model.pt'))
                    
                    _logger.info('Best Score {0:.3%} to {1:.3%}'.format(best_score, np.mean(list(eval_metrics.values()))))

                    best_score = np.mean(list(eval_metrics.values()))
        
            if scheduler and step <= 10000:
                scheduler.step(step+1)
                
            step += 1
        
            if step == num_training_steps:
                train_mode = False
                break
        
    _logger.info('Best Metric: {0:.3%} (step {1:})'.format(best_score, state['best_step']))
    torch.save(model.state_dict(), os.path.join(savedir, f'latest_model.pt'))
    state = {'latest_step':step}
    state.update(eval_log)
    json.dump(state, open(os.path.join(savedir, 'latest_score.json'),'w'), indent='\t')


def evaluate(model, dataloader, criterion, log_interval, metrics: list, device: str = 'cpu'):
    
    # metrics
    auroc_image_metric, auroc_pixel_metric, aupro_pixel_metric = metrics

    # reset
    auroc_image_metric.reset(); auroc_pixel_metric.reset(); aupro_pixel_metric.reset()

    model.eval()
    with torch.no_grad():
        for idx, (inputs, masks, targets) in enumerate(dataloader):
            inputs, masks, targets = inputs.to(device), masks.to(device), targets.to(device)
            
            # predict
            outputs = model(inputs)
            outputs = F.softmax(outputs, dim=1)
            anomaly_score = torch.topk(torch.flatten(outputs[:,1,:], start_dim=1), 100)[0].mean(dim=1)

            # update metrics
            auroc_image_metric.update(
                preds  = anomaly_score.cpu(), 
                target = targets.cpu()
            )
            auroc_pixel_metric.update(
                preds  = outputs[:,1,:].cpu(),
                target = masks.cpu()
            )
            aupro_pixel_metric.update(
                preds   = outputs[:,1,:].cpu(),
                target  = masks.cpu()
            ) 

    # metrics    
    metrics = {
        'AUROC-image':auroc_image_metric.compute().item(),
        'AUROC-pixel':auroc_pixel_metric.compute().item(),
        'AUPRO-pixel':aupro_pixel_metric.compute().item()

    }

    _logger.info('TEST: AUROC-image: %.3f%% | AUROC-pixel: %.3f%% | AUPRO-pixel: %.3f%%' % 
                (metrics['AUROC-image'], metrics['AUROC-pixel'], metrics['AUPRO-pixel']))


    return metrics
