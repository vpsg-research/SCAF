import torch.nn.functional as F
import torch
from feature_loss import *
from tools import *
from utils import ramps

criterion = torch.nn.CrossEntropyLoss(weight=None, ignore_index=255, reduction='mean').cuda()
loss_lsc = FeatureLoss().cuda()
loss_lsc_kernels_desc_defaults = [{"weight": 1, "xy": 6, "rgb": 0.1}]
loss_lsc_radius = 5
l = 0.3

def get_current_consistency_weight(epoch, consistency=0.1, consistency_rampup=150):
    return consistency * ramps.sigmoid_rampup(epoch, consistency_rampup)
    
def get_transform(ops=[0,1,2]):
    '''One of flip, translate, crop'''
    op = np.random.choice(ops)
    if op==0:
        flip = np.random.randint(0, 2)
        pp = Flip(flip)
    elif op==1:
        # pp = Translate(0.3)
        pp = Translate(0.15)
    elif op==2:
        pp = Crop(0.7, 0.7)
    return pp

def get_featuremap(h, x):
    w = h.weight
    b = h.bias
    c = w.shape[1]
    c1 = F.conv2d(x, w.transpose(0,1), padding=(1,1), groups=c)
    return c1, b


def train_loss(image, mask, p_fg, p_bg, net, ctx, ft_dct, w_ft=.1, ft_st = 60, ft_fct=.5, ft_head=True, mtrsf_prob=1, ops=[0,1,2], w_l2g=0, l_me=0.1, me_st=30, me_all=False, multi_sc=0, l=0.3, sl=1):
    if ctx:
        epoch = ctx['epoch']
        global_step = ctx['global_step']
        sw = ctx['sw']
        t_epo = ctx['t_epo']
    fm = []
    def hook(m, i, o):
        if not ft_head:
            fm.extend(get_featuremap(m, i[0]))
        else:
            fm.append(net.feature_head[0](i[0]))
    hh = net.module.head[0].register_forward_hook(hook)


    do_moretrsf = np.random.uniform() < mtrsf_prob
    if do_moretrsf:
        pre_transform = get_transform(ops)
        image_tr = pre_transform(image)
        p_fg_tr = pre_transform(p_fg)  
        p_bg_tr = pre_transform(p_bg)  
        large_scale = True
    else:
        large_scale = np.random.uniform() < multi_sc
        image_tr = image
        p_fg_tr = p_fg 
        p_bg_tr = p_bg 

    sc_fct = 0.6 if large_scale else 0.3
    image_scale = F.interpolate(image_tr, scale_factor=sc_fct, mode='bilinear', align_corners=True)
    p_fg_scale = F.interpolate(p_fg_tr, scale_factor=sc_fct, mode='bilinear', align_corners=True)  
    p_bg_scale = F.interpolate(p_bg_tr, scale_factor=sc_fct, mode='bilinear', align_corners=True)  
    out2, _, out3, out4 = net(image, p_fg, p_bg)
    # out2_org = out2
    hh.remove()
    out2_s, _, out3_s, out4_s = net(image_scale, p_fg_scale, p_bg_scale)


    loss_intra = []
    if epoch >= me_st:
        def entrp(t, mask, weak_weight=0.1):
            etp = -(F.softmax(t, dim=1) * F.log_softmax(t, dim=1)).sum(dim=1)
            unlabel_mask = (mask == 255)
            labeled_mask = (mask != 255)

            strong_msk = (unlabel_mask & (etp < 0.5)).float()
            weak_msk = labeled_mask.float()

            strong_loss = (etp * strong_msk).sum() / (strong_msk.sum() + 1e-8)
            weak_loss = (etp * weak_msk).sum() / (weak_msk.sum() + 1e-8)

            total_loss = strong_loss + weak_weight * weak_loss
            return total_loss

        consistency_weight = get_current_consistency_weight(
            epoch - me_st, consistency=l_me, consistency_rampup=t_epo - me_st)

        if not me_all:
            e = entrp(out2, mask.squeeze(1))
            loss_intra.append(e * consistency_weight)
            loss_intra += [0, 0]
            sw.add_scalar('intra entropy', e.item(), global_step)
        else:
            for out in [out2, out3, out4]:
                e = entrp(out, mask.squeeze(1))
                loss_intra.append(e * consistency_weight)
            sw.add_scalar('intra entropy', loss_intra[0].item(), global_step)
    else:
        loss_intra.extend([0, 0, 0])

    def out_proc(out2, out3, out4):
        a = [out2, out3, out4]
        a = [i.sigmoid() for i in a]
        a = [torch.cat((1 - i, i), 1) for i in a]
        return a
    out2, out3, out4 = out_proc(out2, out3, out4)
    out2_s, out3_s, out4_s = out_proc(out2_s, out3_s, out4_s)

    if not do_moretrsf:
        out2_scale = F.interpolate(out2[:, 1:2], scale_factor=sc_fct, mode='bilinear', align_corners=True)
        out2_s = out2_s[:, 1:2]

    else:
        out2_ss = pre_transform(out2)
        out2_scale = F.interpolate(out2_ss[:, 1:2], scale_factor=0.3, mode='bilinear', align_corners=True)
        out2_s = F.interpolate(out2_s[:, 1:2], scale_factor=0.3/sc_fct, mode='bilinear', align_corners=True)
    loss_ssc = (SaliencyStructureConsistency(out2_s, out2_scale.detach(), 0.85) * (w_l2g + 1) + SaliencyStructureConsistency(out2_s.detach(), out2_scale, 0.85) * (1 - w_l2g)) if sl else 0
    

    gt = mask.squeeze(1).long()
    bg_label = gt.clone()
    fg_label = gt.clone()
    bg_label[gt != 0] = 255
    fg_label[gt == 0] = 255


    image_ = F.interpolate(image, scale_factor=0.25, mode='bilinear', align_corners=True)
    sample = {'rgb': image_}
    # print('sample :', image_.max(), image_.min(), image_.std())
    out2_ = F.interpolate(out2[:, 1:2], scale_factor=0.25, mode='bilinear', align_corners=True)
    loss2_lsc = loss_lsc(out2_, loss_lsc_kernels_desc_defaults, loss_lsc_radius, sample, image_.shape[2], image_.shape[3])['loss']
    loss2 = loss_ssc + criterion(out2, fg_label) + criterion(out2, bg_label) + l * loss2_lsc + loss_intra[0] ## dominant loss


    out3_ = F.interpolate(out3[:, 1:2], scale_factor=0.25, mode='bilinear', align_corners=True)
    loss3_lsc = loss_lsc(out3_, loss_lsc_kernels_desc_defaults, loss_lsc_radius, sample, image_.shape[2], image_.shape[3])['loss']
    loss3 = criterion(out3, fg_label) + criterion(out3, bg_label) + l * loss3_lsc + loss_intra[1]
    out4_ = F.interpolate(out4[:, 1:2], scale_factor=0.25, mode='bilinear', align_corners=True)
    loss4_lsc = loss_lsc(out4_, loss_lsc_kernels_desc_defaults, loss_lsc_radius, sample, image_.shape[2], image_.shape[3])['loss']
    loss4 = criterion(out4, fg_label) + criterion(out4, bg_label) + l * loss4_lsc + loss_intra[2]
    
    return loss2, loss3, loss4