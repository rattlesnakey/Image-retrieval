import os
import time
import json
import numpy as np

import torch
import torch.nn as nn
from tqdm import tqdm

from torch.cuda.amp import autocast
import torch.distributed as dist
import torch.nn.functional as F
import logging

from AWP import AWP

def is_master(args):
    return (not args.distributed) or args.gpu == 0
#! group contrastive 
def GroupContrastive_loss(text_features, image_features):

    text_embedding_similarity = F.cosine_similarity(text_features.unsqueeze(1), text_features.unsqueeze(0), dim=-1)
    image_embedding_similarity = F.cosine_similarity(image_features.unsqueeze(1), image_features.unsqueeze(0), dim=-1)
    
    n = text_embedding_similarity.size()[0]
    flatten_text_embedding_similarity = text_embedding_similarity.flatten()[:-1].view(n-1, n+1)[:,1:].flatten().view(n, n-1)
    flatten_image_embedding_similarity = image_embedding_similarity.flatten()[:-1].view(n-1, n+1)[:,1:].flatten().view(n, n-1)
    
    kl_text = F.kl_div(F.log_softmax(flatten_text_embedding_similarity, dim=-1), F.softmax(flatten_image_embedding_similarity, dim=-1), reduction='sum')
    kl_image = F.kl_div(F.log_softmax(flatten_image_embedding_similarity, dim=-1), F.softmax(flatten_text_embedding_similarity, dim=-1), reduction='sum')
    loss = (kl_text + kl_image) / 2
    return loss

def get_loss(model, images, texts, loss_img, loss_txt, args, epoch, text_smoothing=False):

    image_features, text_features, logit_scale = model(images, texts, text_smoothing)
    # image_features, text_features, logit_scale = model(images, texts)

    logit_scale = logit_scale.mean()
    if args.distributed and args.aggregate:
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # We gather tensors from all gpus to get more negatives to contrast with.
        gathered_image_features = [
            torch.zeros_like(image_features) for _ in range(world_size)
        ]
        gathered_text_features = [
            torch.zeros_like(text_features) for _ in range(world_size)
        ]
       
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)
        
        
        
        all_image_features = torch.cat(
            [image_features]
            + gathered_image_features[:rank]
            + gathered_image_features[rank + 1 :]
        )
        all_text_features = torch.cat(
            [text_features]
            + gathered_text_features[:rank]
            + gathered_text_features[rank + 1 :]
        )
        
             
            
        # this is needed to send gradients back everywhere.

        logits_per_image = logit_scale * all_image_features @ all_text_features.t()

        logits_per_text = logits_per_image.t()
        
        if args.group_contrastive_loss_ratio > 0:
            if args.group_contrastive_start_epoch <= epoch:
                group_contrastive_loss = GroupContrastive_loss(all_text_features, all_image_features)
            
    else:
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()
                    
        if args.group_contrastive_loss_ratio > 0:
            if args.group_contrastive_start_epoch <= epoch:
                group_contrastive_loss = GroupContrastive_loss(text_features, image_features)

    ground_truth = torch.arange(len(logits_per_image)).long()
    if args.gpu is not None:
        ground_truth = ground_truth.cuda(args.gpu, non_blocking=True)

    total_loss = (
        loss_img(logits_per_image, ground_truth)
        + loss_txt(logits_per_text, ground_truth)
    ) / 2
        
        
    if args.group_contrastive_loss_ratio > 0:
        if args.group_contrastive_start_epoch <= epoch:
            total_loss = total_loss + group_contrastive_loss * args.group_contrastive_loss_ratio
    
    return total_loss

def train(model, data, epoch, optimizer, scaler, scheduler, args):
    # os.environ["WDS_EPOCH"] = str(epoch)
    
    model.train()

    dataloader, sampler = data['train'].dataloader,  data['train'].sampler

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    if args.gpu is not None:
        loss_img = loss_img.cuda(args.gpu)
        loss_txt = loss_txt.cuda(args.gpu)

    if args.distributed and sampler is not None:
        sampler.set_epoch(epoch)

    num_batches_per_epoch = dataloader.num_batches
    end = time.time()
    
    #! AWP Training
    if args.adv_train == 'True':
        logging.info(f'Doing AWP adversial training start from epoch {args.adv_start_epoch}, Now is epoch ${epoch}')
        awp = AWP(model, optimizer, adv_lr=1, adv_eps=0.0001)
    
    if args.group_contrastive_loss_ratio > 0 and args.group_contrastive_start_epoch <= epoch:
        logging.info(f'Doing Group Contrastive training start from epoch {args.group_contrastive_start_epoch}, Now is epoch ${epoch}')
    
    epoch_loss = 0
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        scheduler(step)

        optimizer.zero_grad()

        images, texts, eos_indices = batch
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            texts = texts.cuda(args.gpu, non_blocking=True)
            eos_indices = eos_indices.cuda(args.gpu, non_blocking=True)
       
        data_time = time.time() - end

        m = model.module if args.distributed or args.dp else model

        # with automatic mixed precision.
        if args.precision == "amp":
            with autocast():
                
                if args.adv_train == "True":
                    
                   
                    if args.adv_start_epoch <= epoch:
                        total_loss = awp.attack_backward(get_loss, images, texts, loss_img, loss_txt, args, epoch)
                        scaler.scale(total_loss).backward()
                        scaler.step(optimizer)
                        awp._restore()
         
                else:
                    total_loss = get_loss(model, images, texts, loss_img, loss_txt, args, epoch)
                   
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    
                if args.text_smoothing == 'True':
                    total_loss = get_loss(model, images, texts, loss_img, loss_txt, args, epoch, text_smoothing=True)
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    
            
                
            scaler.update()

        else:
            
            if args.adv_train == "True":
                if args.adv_start_epoch <= epoch:
                    total_loss = awp.attack_backward(get_loss, images, texts, loss_img, loss_txt, args, epoch)
                    total_loss.backward()
                   
                    awp._restore()
            else:
                total_loss = get_loss(model, images, texts, loss_img, loss_txt, args, epoch)
                total_loss.backward()
            
            optimizer.step()
            
            if args.text_smoothing == 'True':
                total_loss = get_loss(model, images, texts, loss_img, loss_txt, args, epoch, text_smoothing=True)
                total_loss.backward()
                optimizer.step()
                
                
                

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        m.logit_scale.data = torch.clamp(m.logit_scale.data, 0, 4.6052)
     
        batch_time = time.time() - end
        end = time.time()

        if is_master(args) and (i % 1) == 0:
            num_samples = i * len(images) * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * i / num_batches_per_epoch
            logging.info(
                f"Train Epoch: {epoch} [{num_samples}/{samples_per_epoch} ({percent_complete:.0f}%)]\t"
                f"Loss: {total_loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}"
                f"\tLR: {optimizer.param_groups[0]['lr']:5f}\tlogit_scale {m.logit_scale.data:.3f}"
            )
            # save train loss / etc.

            timestep = epoch * num_batches_per_epoch + i
            log_data = {
                "loss": total_loss.item(),
                "data_time": data_time,
                "batch_time": batch_time,
                "scale":  m.logit_scale.data.item(),
                "lr": optimizer.param_groups[0]["lr"]
            }
            epoch_loss += log_data['loss']
    return epoch_loss // num_batches_per_epoch


def evaluate(model, data, epoch, args, steps=None):
   
    if not is_master(args):
        return
    
    logging.info(f"Begin to eval epoch: {epoch}...")
    
    model.eval()

    dataloader = data['val'].dataloader

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    if args.gpu is not None:
        loss_img = loss_img.cuda(args.gpu)
        loss_txt = loss_txt.cuda(args.gpu)

    cumulative_loss = 0.0
    num_elements = 0.0
    all_image_features, all_text_features = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images, texts, eos_indices = batch
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                texts = texts.cuda(args.gpu, non_blocking=True)
                eos_indices = eos_indices.cuda(args.gpu, non_blocking=True)
            if args.text_smoothing == 'True':
                image_features, text_features, text_features_enhanced, logit_scale = model(images, texts)
            else:
                image_features, text_features, logit_scale = model(images, texts)
            all_image_features.append(image_features)
            all_text_features.append(text_features)
            logit_scale = logit_scale.mean()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()

            ground_truth = torch.arange(len(images)).long()
            if args.gpu is not None:
                ground_truth = ground_truth.cuda(args.gpu, non_blocking=True)
            total_loss = (
                loss_img(logits_per_image, ground_truth)
                + loss_txt(logits_per_text, ground_truth)
            ) / 2

            batch_size = len(images)
            cumulative_loss += total_loss * batch_size
            num_elements += batch_size

        metrics = {}
        loss = cumulative_loss / num_elements
        metrics.update(
            **{"val_loss": loss.item(), "epoch": epoch, "num_elements": num_elements}
        )

        logging.info(
            f"Eval Epoch: {epoch} "
            + "\t".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        )

    return metrics
