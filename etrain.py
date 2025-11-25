#!/usr/bin/python3
#coding=utf-8

from functools import partial
import sys
import datetime
import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from data import dataset 
import logging as logger
from lib.data_prefetcher import DataPrefetcher
import numpy as np
from train_processes import *
from tools import *

TAG = "scribblecod"
logger.basicConfig(level=logger.INFO, format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', \
                           filename="train_%s.log"%(TAG), filemode="w")


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay*init_lr
        lr=param_group['lr']
    return lr

def validate(model, val_loader, threshold=0.5):
    model.eval()
    total_f1 = 0.0
    cnt = 0
    with torch.no_grad():
        for image, mask, p_fg, p_bg, shape, name in val_loader:
            image, mask = image.cuda().float(), mask.cuda().float()
            out, _, _, _ = model(image, p_fg, p_bg)
            out = F.interpolate(out, size=shape, mode='bilinear', align_corners=False)
            pred = torch.sigmoid(out[0, 0])

     
            pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

       
            pred_binary = (pred > threshold).float()
            mask_binary = (mask[0] > 0.5).float()

    
            TP = (pred_binary * mask_binary).sum().item()
            FP = (pred_binary * (1 - mask_binary)).sum().item()
            FN = ((1 - pred_binary) * mask_binary).sum().item()

            precision = TP / (TP + FP + 1e-8)
            recall = TP / (TP + FN + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            total_f1 += f1
            cnt += 1

    model.train(True)
    return total_f1 / cnt


def validate_multiloader(model, val_loader):
    f1_scores = []
    for v in val_loader:
        st = time.time()
        f1 = validate(model, v)
        f1_scores.append(f1)
        print('Spent %.3fs, %s F1: %s' % (time.time()-st, v.dataset.data_name, f1))
    return sum(f1_scores) / len(f1_scores)

total_epoch = 70
EXP_NAME = '' # change it in main
root = './Dataset'

def train(Dataset, Network, cfg, train_loss, start_from = 0):
    ## dataset
    data = Dataset.Data(cfg)
    loader = DataLoader(data, batch_size=cfg.batch, shuffle=True, num_workers=8)
    val_names = ['CASIAv1', 'Coverage', 'Columbia']
    val_cfg = [Dataset.Config(datapath=f'{root}/test/{i}', mode='test') for i in val_names]
    val_data = [Dataset.Data(v) for v in val_cfg]
    val_loaders = [DataLoader(v, batch_size=1, shuffle=False, num_workers=4) for v in val_data]

    max_f1_list = [0.0 for _ in val_names]
    best_epoch_list = [0 for _ in val_names]
    max_f1_avg = 0.0
    best_epoch_avg = 0

    ## network
    net = Network()
    net = torch.nn.DataParallel(net, device_ids=[0,1,2,3])
    net.train(True)
    net.cuda()

    optimizer = torch.optim.AdamW(net.parameters(), cfg.lr, weight_decay=1e-4)

    ## log
    sw = SummaryWriter(cfg.savepath)
    db_size = len(loader)
    global_step = start_from * db_size
    et = 0

    for epoch in range(start_from, cfg.epoch):
        cur_lr = adjust_lr(optimizer, cfg.lr, epoch, cfg.decay_rate, cfg.decay_epoch)
        print('lr_value: ', cur_lr)
        prefetcher = DataPrefetcher(loader)
        batch_idx = -1
        image, mask, p_fg, p_bg = prefetcher.next()
        while image is not None:
            st = time.time()
            niter = epoch * db_size + batch_idx
            batch_idx += 1
            global_step += 1

            loss2, loss3, loss4 = train_loss(image, mask, p_fg, p_bg, net, dict(epoch=epoch+1, global_step=global_step, sw=sw, t_epo=cfg.epoch))
            loss = loss2 + loss3 + loss4
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sw.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)
            sw.add_scalar('loss', loss.item(), global_step=global_step)

            image, mask, p_fg, p_bg = prefetcher.next()
            ta = time.time() - st
            et = 0.9*et + 0.1*ta if et>0 else ta
            if batch_idx % 10 == 0:
                msg = '%s| %s | eta:%s | step:%d/%d/%d | lr=%.6f | loss=%.6f | loss2=%.6f | loss3=%.6f | loss4=%.6f | loss5=%.6f' % (
                    TAG, datetime.datetime.now(), datetime.timedelta(seconds = int((cfg.epoch*db_size-niter)*et)), global_step, epoch+1, cfg.epoch,
                    optimizer.param_groups[0]['lr'], loss.item(), loss2.item(), loss3.item(), loss4.item(), 0)
                print(msg)
                logger.info(msg)


        if epoch >= 40:
            f1_scores = []
            for idx, v in enumerate(val_loaders):
                f1 = validate(net, v)
                f1_scores.append(f1)
                print(f'VAL {val_names[idx]} F1: {f1}')
                sw.add_scalar(f'val_f1_{val_names[idx]}', f1, global_step=global_step)
        
                if f1 > max_f1_list[idx]:
                    max_f1_list[idx] = f1
                    best_epoch_list[idx] = epoch + 1
                    torch.save(net.state_dict(), f"{cfg.savepath}/model-best-{val_names[idx]}.pth")
                    print(f"[SAVE] Best {val_names[idx]} epoch: {best_epoch_list[idx]}, F1: {max_f1_list[idx]}")

 
            f1_avg = sum(f1_scores) / len(f1_scores)
            sw.add_scalar('val_f1_avg', f1_avg, global_step=global_step)
            print(f'VAL AVG F1: {f1_avg}')
            if f1_avg > max_f1_avg:
                max_f1_avg = f1_avg
                best_epoch_avg = epoch + 1
                torch.save(net.state_dict(), f"{cfg.savepath}/model-best-avg.pth")
                print(f"[SAVE] Best AVG epoch: {best_epoch_avg}, F1: {max_f1_avg}")

        if  epoch % 50 == 0 or epoch == cfg.epoch-2 or epoch == cfg.epoch-1:
            torch.save(net.state_dict(), cfg.savepath + '/model-' + str(epoch + 1) + '.pth')

    print('max val F1(avg) for {} is {}'.format(EXP_NAME, max_f1_avg))
    print('Best epoch per valset:', dict(zip(val_names, best_epoch_list)))


if __name__=='__main__':
    cfg = [.15, 35, 16, 1]
    w_ft, ft_st, topk,w_ftp = cfg
    EXP_NAME = f'trained'
    cfg = dataset.Config(datapath=f'{root}', savepath=f'./weight/{EXP_NAME}/', mode='train', batch=32, lr=1e-4, momen=0.9, decay=5e-4, epoch=total_epoch, label_dir = 'Scribble', decay_rate = 0.1, decay_epoch = 50)
    from net import Net
    tm = partial(train_loss, w_ft=w_ft, ft_st = ft_st, ft_fct=.5, ft_dct = dict(crtl_loss = False, w_ftp=w_ftp, norm=False, topk=topk, step_ratio=2), ft_head=False, mtrsf_prob=1, ops=[0,1,2], w_l2g=0.3, l_me=0.05, me_st=25, multi_sc=0)
    train(dataset, Net, cfg, tm, start_from=0)