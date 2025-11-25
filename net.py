#v46
import torch
import torch.nn as nn
import torch.nn.functional as F
# from model.Res2Net import res2net50_v1b_26w_4s
from model.pvtv2 import pvt_v2_b2

def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU) or isinstance(m, nn.GELU) or isinstance(m, nn.LeakyReLU) or isinstance(m, nn.AdaptiveAvgPool2d) or isinstance(m, nn.AdaptiveMaxPool2d) or isinstance(m, nn.ReLU6) or isinstance(m, nn.MaxPool2d) or isinstance(m, nn.Softmax) or isinstance(m, nn.Sigmoid):
            pass
        elif isinstance(m, nn.ModuleList):
            weight_init(m)
        else:
            m.initialize()


class CoordAtt(nn.Module):

    def __init__(self, channels, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))     
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))      
        mid = max(8, channels // reduction)
        self.conv1 = nn.Conv2d(channels, mid, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mid)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mid, channels, kernel_size=1)
        self.conv_w = nn.Conv2d(mid, channels, kernel_size=1)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)                  # [B, C, H, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # [B, C, W, 1]
        y = torch.cat([x_h, x_w], dim=2)      # [B, C, H+W, 1]
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)         # [B, C, 1, W]
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        out = identity * a_h * a_w
        return out

    def initialize(self):
        weight_init(self)


class GFM_CoordAtt(nn.Module):
    def __init__(self, channels, alpha_init=0.1, reduction=32):
        super().__init__()
        self.coord_att = CoordAtt(channels, reduction)
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        self.fg_weight = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))    
        self.bg_weight = nn.Parameter(torch.tensor(0.8, dtype=torch.float32))   
        self.gate_conv = nn.Sequential(
            nn.Conv2d(1, channels, 1),   
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        self.fusion_conv = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, Ft, P_fg, P_bg):
        if P_fg.size()[2:] != Ft.size()[2:]:
            P_fg = F.interpolate(P_fg, size=Ft.size()[2:], mode='bilinear', align_corners=False)
        if P_bg.size()[2:] != Ft.size()[2:]:
            P_bg = F.interpolate(P_bg, size=Ft.size()[2:], mode='bilinear', align_corners=False)

        epsilon = 1e-6
   
        prob_fg = (self.fg_weight * P_fg) / (self.fg_weight * P_fg + self.bg_weight * P_bg + epsilon)  # [B, 1, H, W]

        gate = self.gate_conv(prob_fg)  # [B, C, H, W]
        F_mod = Ft + self.alpha * (Ft * gate)
        F_fuse = self.fusion_conv(torch.cat([Ft, F_mod], dim=1))
        F_ca = self.coord_att(F_fuse)
        return F_ca
    
    def initialize(self):
        weight_init(self)


class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x
    
    def initialize(self):
        weight_init(self)


class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)
    
    def initialize(self):
        weight_init(self) 

class getAlpha(nn.Module):
    def __init__(self, in_channels):
        super(getAlpha, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels, 1, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) 
    
    def initialize(self):
        weight_init(self) 

class ADF(nn.Module):
    def __init__(self, channel):
        super(ADF, self).__init__()
        self.F1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        )
        self.F2 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        )
        self.conv3_1 = ConvBNR(channel, channel)
        self.conv3_2 = ConvBNR(channel, channel)
        self.conv3_3 = ConvBNR(channel, channel)
        self.conv1_1 = Conv1x1(channel, channel)
        self.getalpha = getAlpha(channel)

    def forward(self, feature_map):
        f1 = self.F1(feature_map)
        f2 = self.F2(f1 + feature_map)
        alpha = self.getalpha(torch.cat([f1, f2], dim=1))
        out = feature_map + f1 * alpha + f2 * (1 - alpha)
        
        f1_ef = self.conv3_1(f1*out+f1)
        bk = 1-out
        f2_ef = self.conv3_2(f2*bk)

        fd = self.conv3_3(f1_ef - f2_ef)

        re = self.conv1_1(f1_ef+fd)

        return re
    
    def initialize(self):
        weight_init(self)

class CAM(nn.Module):
    def __init__(self, channel):
        super(CAM, self).__init__()
        self.conv1_1 = Conv1x1(channel, channel)
        self.conv3_1 = ConvBNR(channel // 4, channel // 4, 3)
        self.conv3_2 = ConvBNR(channel // 4, channel // 4, 3)
        self.conv3_3 = ConvBNR(channel // 4, channel // 4, 3)
        self.conv3_4 = ConvBNR(channel // 4, channel // 4, 3)
   
        self.adf0 = ADF(channel // 4)
        self.adf1 = ADF(channel // 4)
        self.adf2 = ADF(channel // 4)
        self.adf3 = ADF(channel // 4)
 

    def forward(self, x):
        
  
        xc = torch.chunk(x, 4, dim=1)
        x0 = self.conv3_1(xc[0] + xc[1] + xc[2] + xc[3])
        ad0 = self.adf0(x0)

        x1 = self.conv3_2(xc[1] + xc[2] + xc[3] + ad0)
        ad1 = self.adf1(x1)

        x2 = self.conv3_3(xc[2] + xc[3] + ad1)
        ad2 = self.adf2(x2)

        x3 = self.conv3_4(xc[3] + ad2)
        ad3 = self.adf3(x3)

        re = self.conv1_1(torch.cat((ad0, ad1, ad2, ad3), dim=1))

        return re 
    
    def initialize(self):
        weight_init(self) 

class contextfeature(nn.Module):
    def __init__(self, channel):
        super(contextfeature, self).__init__()

        self.conv3_1 = ConvBNR(channel, channel)
        self.conv3_2 = ConvBNR(channel*2, channel)
        self.conv3_3 = ConvBNR(channel*2, channel)
        self.ca = CAM(channel)

    def forward(self, low, high):
        if low.size()[2:] != high.size()[2:]:
            high = F.interpolate(high, size=low.size()[2:], mode='bilinear', align_corners=False)
        
        lh = self.conv3_1(low + high)

        hight_ef = self.conv3_2(torch.cat((lh, high), dim=1))

        out_ad = self.ca(hight_ef)

        out = self.conv3_3(torch.cat((lh, out_ad), dim=1))

        return out

        
    def initialize(self):
        weight_init(self) 

 
class EF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(EF, self).__init__()
        self.relu = nn.ReLU(True)

        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            # BasicConv2d(out_channel, out_channel, 3, padding=1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )

        self.branch2Cov = BasicConv2d(in_channel, out_channel, 1)
        self.branch2 = nn.Sequential(
            # BasicConv2d(out_channel, out_channel, 5, padding=2),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )

        self.branch3Cov = BasicConv2d(in_channel, out_channel, 1)
        self.branch3 = nn.Sequential(
            # BasicConv2d(out_channel, out_channel, 7, padding=3),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )

        self.branch3_2 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, 1),
        )

        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2Cov(x) + x1
        x2 = self.branch2(x2)
        x3 = self.branch3Cov(x) + x2
        x3 = self.branch3(x3)
        x3 = self.branch3_2(x3)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), dim=1))
        x = self.relu(x_cat + self.conv_res(x))
        return x
        
    def initialize(self):
        weight_init(self)



class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
    def initialize(self):
        weight_init(self)

class Conv_Block(nn.Module):  # [64, 128, 320, 512]
    def __init__(self, channels):
        super(Conv_Block, self).__init__()
        self.conv1 = nn.Conv2d(channels*3, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels*2, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(channels*2)

        self.conv3 = nn.Conv2d(channels*2, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)

    def forward(self, input1, input2, input3):
        fuse = torch.cat((input1, input2, input3), 1)
        fuse = self.bn1(self.conv1(fuse))
        fuse = self.bn2(self.conv2(fuse))
        fuse = self.bn3(self.conv3(fuse))
        return fuse
    
    def initialize(self):
        weight_init(self)



################################################ Net ###############################################
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.bkbone = pvt_v2_b2()
        path = './model/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.bkbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.bkbone.load_state_dict(model_dict)

        self.conv_block = Conv_Block(64)

        self.ef = nn.ModuleList([
            EF(64, 64),
            EF(128, 64),
            EF(320, 64),
            EF(512, 64)
        ])

        self.cf = nn.ModuleList([
            contextfeature(64),
            contextfeature(64),
            contextfeature(64)
        ])

        self.head = nn.ModuleList([
                nn.Conv2d(64, 1, 1),
                nn.Conv2d(64, 1, 1),
                nn.Conv2d(64, 1, 1)
        ])

        self.gf_ca = nn.ModuleList([
            GFM_CoordAtt(64),
            GFM_CoordAtt(64),
            GFM_CoordAtt(64),
            GFM_CoordAtt(64)
        ])

        self.initialize()


    def forward(self, x, fg, bg, shape=None, epoch=None):
        
        shape = x.size()[2:] if shape is None else shape
        bk_stage2, bk_stage3, bk_stage4, bk_stage5 = self.bkbone(x)
 
        f2 = self.ef[0](bk_stage2)
        f3 = self.ef[1](bk_stage3)
        f4 = self.ef[2](bk_stage4)
        f5 = self.ef[3](bk_stage5)

        f2_ca = self.gf_ca[0](f2, fg, bg)
        f3_ca = self.gf_ca[1](f3, fg, bg)
        f4_ca = self.gf_ca[2](f4, fg, bg)
        f5_ca = self.gf_ca[3](f5, fg, bg)

        fu45 = self.cf[0](f4_ca, f5_ca)
        fu345 = self.cf[1](f3_ca, fu45)
        fu2345 = self.cf[2](f2_ca, fu345)

        
        out0 = F.interpolate(self.head[0](fu2345), size=shape, mode='bilinear', align_corners=False)
        out1 = F.interpolate(self.head[1](fu345), size=shape, mode='bilinear', align_corners=False)
        out2 = F.interpolate(self.head[2](fu45), size=shape, mode='bilinear', align_corners=False)


        return out0, None, out1, out2
 

    def initialize(self):
        print('initialize net')
 
        for name, module in self.named_children():
            if name == 'bkbone':
                print("1")
                continue
            weight_init(module)