
from functools import partial
import os
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
import imageio


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
root = './Dataset'

save_path = './result/'

def test(Dataset, Network, cfg, train_loss, start_from = 0):

    test_sets = ['C1', 'Coverage', 'Columbia','NC16', 'IMD2020','ITW', 'CocoGlide',  'Korus']
 
    val_cfg = [Dataset.Config(datapath=f'{root}/test/{i}', mode='test') for i in test_sets]
    val_data = [Dataset.Data(v) for v in val_cfg]
    val_loaders = [DataLoader(v, batch_size=1, shuffle=False, num_workers=4) for v in val_data]

    
    net = Network()
    net = torch.nn.DataParallel(net, device_ids=[0,1,2,3]) 
    net.load_state_dict(torch.load('./SCAF.pth'))
    net.cuda()
    net.eval()


    for idx, val_loader in enumerate(val_loaders):
        set_name = test_sets[idx]
 
        save_dir = os.path.join(save_path, set_name)
        os.makedirs(save_dir, exist_ok=True)
        for image, mask, p_fg, p_bg, shape, name in val_loader:
            image, mask, p_fg, p_bg = image.cuda().float(), mask.cuda().float(), p_fg.cuda().float(), p_bg.cuda().float()
            with torch.no_grad():
                out, _, _, _ = net(image, p_fg, p_bg)
            res = F.upsample(out, size=mask.shape[2:], mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
  
            save_name = os.path.join(save_dir, name[0] if isinstance(name, (list, tuple)) else name)
            imageio.imwrite(save_name, (res*255).astype(np.uint8))
    

if __name__=='__main__':
    cfg = [.15, 35, 16, 1]
    w_ft, ft_st, topk,w_ftp = cfg
    EXP_NAME = f'trained'
    cfg = dataset.Config(datapath=f'{root}', savepath=f'./result/{EXP_NAME}/', mode='train', batch=16, lr=1e-4, momen=0.9, decay=5e-4, label_dir = 'Scribble', decay_rate = 0.1, decay_epoch = 50)
    from net import Net
    tm = partial(train_loss, w_ft=w_ft, ft_st = ft_st, ft_fct=.5, ft_dct = dict(crtl_loss = False, w_ftp=w_ftp, norm=False, topk=topk, step_ratio=2), ft_head=False, mtrsf_prob=1, ops=[0,1,2], w_l2g=0.3, l_me=0.05, me_st=25, multi_sc=0)
    test(dataset, Net, cfg, tm, start_from=0)

