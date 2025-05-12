import math
import torch
from torch import cat, nn
import torch.nn.functional as F

class BNRELU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-5)
        self.act = nn.ReLU6()

    def forward(self, x):
        output = self.bn(x)
        output = self.act(output)

        return output
    
    
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, dilation=(1, 1), padding=0, bn_act=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels,  kernel_size, dilation=dilation, padding=kernel_size//2, groups=in_channels,
                                   stride=stride, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn_act = bn_act
        
        if self.bn_act:
            self.bn_gelu = BNGELU(out_channels)
            # self.bn_relu = BNRELU(out_channels)
            

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        if self.bn_act:
            x = self.bn_gelu(x)
            # x = self.bn_relu(x)
        
        return x
    

class CustomGhostModule(nn.Module):
    def __init__(self, exp, in_channels, mid_channels):
        super().__init__()
        self.exp = exp
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        
        self.primary_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_channels, eps=1e-3, momentum=0.999)
        )
        
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels * (self.exp - 1), kernel_size=3, stride=1, padding=1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels * (self.exp - 1), eps=1e-3, momentum=0.999)
        )
        
        
    def forward(self, x):
        x_primary = self.primary_conv(x)
        x_depthwise = self.depthwise_conv(x_primary)
        out = torch.cat([x_primary, x_depthwise], dim=1)
        
        return out
        
        
class InvertedBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=6, kernel_size=3, stride=1, dilation=1, bn_act=False):
        super().__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expansion

        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim)
            # nn.ReLU6(inplace=True)
        ) if expansion != 1 else nn.Identity()

        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=(kernel_size//2)*dilation,
                      dilation=dilation, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim)
            # nn.ReLU6(inplace=True)
        )

        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

        self.bn_act = bn_act
        if self.bn_act:
            self.act = BNRELU(out_channels)

    def forward(self, x):
        residual = x
        out = self.expand(x)
        out = self.depthwise(out)
        out = self.project(out)

        if self.use_residual:
            out = out + residual
            

        if self.bn_act:
            out = self.act(out)

        return out

    





class BNGELU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-5)
        self.act = nn.GELU()

    def forward(self, x):
        output = self.bn(x)
        output = self.act(output)

        return output
    
    
class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding=0, dilation=(1, 1), groups=1, bn_act=False, bias=False):
        super().__init__()
        self.bn_act = bn_act
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_act:
            self.bn_gelu = BNGELU(nOut)

    def forward(self, x):
        output = self.conv(x)

        if self.bn_act:
            output = self.bn_gelu(output)

        return output
    
#add
class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, expansion_ratio=2):
        super().__init__()
        out_channels = in_channels * expansion_ratio

        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.project = nn.Sequential(
            nn.Conv2d(out_channels, in_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x= self.down(x)
        x = self.project(x)
        
        return x
    

"""--------------------Coordinate Attention--------------------------------------"""    
# region coordinate attention 
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        identity = x
        
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out
    
"""----------------------------------------------------------""" 

# region AsymmNet
"""--------------------------AsymmNet------------------------"""



class HardSwish(nn.Module):
    def __init__(self, inplace=False):
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class HardSigmoid(nn.Module):
    def __init__(self, inplace=False):
        super(HardSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class Activation(nn.Module):
    def __init__(self, act_func):
        super(Activation, self).__init__()
        if act_func == "relu":
            self.act = nn.ReLU()
        elif act_func == "relu6":
            self.act = nn.ReLU6()
        elif act_func == "hard_sigmoid":
            self.act = HardSigmoid()
        elif act_func == "hard_swish":
            self.act = HardSwish()
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.act(x)


def make_divisible(x, divisible_by=8):
    return int(math.ceil(x * 1. / divisible_by) * divisible_by)


class _BasicUnit(nn.Module):
    def __init__(self, num_in, num_out, kernel_size=1, strides=1, pad=0, num_groups=1,
                 use_act=True, act_type="relu", norm_layer=nn.BatchNorm2d):
        super(_BasicUnit, self).__init__()
        self.use_act = use_act
        self.conv = nn.Conv2d(in_channels=num_in, out_channels=num_out,
                              kernel_size=kernel_size, stride=strides,
                              padding=pad, groups=num_groups, bias=False,
                              )
        self.bn = norm_layer(num_out)
        if use_act is True:
            self.act = Activation(act_type)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        if self.use_act:
            out = self.act(out)
        return out


class SE_Module(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SE_Module, self).__init__()
        reduction_c = make_divisible(channels // reduction)
        self.out = nn.Sequential(
            nn.Conv2d(channels, reduction_c, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduction_c, channels, 1, bias=True),
            HardSigmoid()
        )

    def forward(self, x):
        y = F.adaptive_avg_pool2d(x, 1)
        y = self.out(y)
        return x * y


class AsymmBottleneck(nn.Module):
    def __init__(self, num_in, num_mid, num_out, kernel_size, asymmrate=1,
                 act_type="relu", use_se=False, strides=1,
                 norm_layer=nn.BatchNorm2d):
        super(AsymmBottleneck, self).__init__()
        assert isinstance(asymmrate, int)
        self.asymmrate = asymmrate
        self.use_se = use_se
        self.use_short_cut_conv = (num_in == num_out and strides == 1)
        self.do_expand = (num_mid > max(num_in, asymmrate * num_in))
        if self.do_expand:
            self.expand = _BasicUnit(num_in, num_mid - asymmrate * num_in,
                                     kernel_size=1,
                                     strides=1, pad=0, act_type=act_type,
                                     norm_layer=norm_layer)
            num_mid += asymmrate * num_in
        self.dw_conv = _BasicUnit(num_mid, num_mid, kernel_size, strides,
                                  pad=self._get_pad(kernel_size), act_type=act_type,
                                  num_groups=num_mid, norm_layer=norm_layer)
        if self.use_se:
            self.se = SE_Module(num_mid)
        self.pw_conv_linear = _BasicUnit(num_mid, num_out, kernel_size=1, strides=1,
                                         pad=0, act_type=act_type, use_act=False,
                                         norm_layer=norm_layer, num_groups=1)

    def forward(self, x):
        if self.do_expand:
            out = self.expand(x)
            feat = []
            for i in range(self.asymmrate):
                feat.append(x)
            feat.append(out)
            for i in range(self.asymmrate):
                feat.append(x)
            if self.asymmrate > 0:
                out = cat(feat, dim=1)
        else:
            out = x
        out = self.dw_conv(out)
        if self.use_se:
            out = self.se(out)
        out = self.pw_conv_linear(out)
        if self.use_short_cut_conv:
            return x + out
        return out

    def _get_pad(self, kernel_size):
        if kernel_size == 1:
            return 0
        elif kernel_size == 3:
            return 1
        elif kernel_size == 5:
            return 2
        elif kernel_size == 7:
            return 3
        else:
            raise NotImplementedError


def get_asymmnet_cfgs(model_name):
    if model_name == 'asymmnet_large':
        inplanes = 16
        cfg = [
            # k, exp, c,  se,     nl,  s,
            # stage1
            [3, 16, 16, False, 'relu', 1],
            # stage2
            [3, 64, 24, False, 'relu', 2],
            [3, 72, 24, False, 'relu', 1],
            # stage3
            [5, 72, 40, True, 'relu', 2],
            [5, 120, 40, True, 'relu', 1],
            [5, 120, 40, True, 'relu', 1],
            # stage4
            [3, 240, 80, False, 'hard_swish', 2],
            [3, 200, 80, False, 'hard_swish', 1],
            [3, 184, 80, False, 'hard_swish', 1],
            [3, 184, 80, False, 'hard_swish', 1],
            [3, 480, 112, True, 'hard_swish', 1],
            [3, 672, 112, True, 'hard_swish', 1],
            # stage5
            [5, 672, 160, True, 'hard_swish', 2],
            [5, 960, 160, True, 'hard_swish', 1],
            [5, 960, 160, True, 'hard_swish', 1],
        ]
        cls_ch_squeeze = 960
        cls_ch_expand = 1280
    elif model_name == 'asymmnet_small':
        inplanes = 16
        cfg = [
            # k, exp, c,  se,     nl,  s,
            [3, 16, 16, True, 'relu', 2],
            [3, 72, 24, False, 'relu', 2],
            [3, 88, 24, False, 'relu', 1],
            [5, 96, 40, True, 'hard_swish', 2],
            [5, 240, 40, True, 'hard_swish', 1],
            [5, 240, 40, True, 'hard_swish', 1],
            [5, 120, 48, True, 'hard_swish', 1],
            [5, 144, 48, True, 'hard_swish', 1],
            [5, 288, 96, True, 'hard_swish', 2],
            [5, 576, 96, True, 'hard_swish', 1],
            [5, 576, 96, True, 'hard_swish', 1],
        ]
        cls_ch_squeeze = 576
        cls_ch_expand = 1280
    else:
        raise ValueError('{} model_name is not supported now!'.format(model_name))

    return inplanes, cfg, cls_ch_squeeze, cls_ch_expand


class AsymmNet(nn.Module):
    def __init__(self, cfgs_name, num_classes=1000, multiplier=1.0, asymmrate=1, dropout_rate=0.2,
                 norm_layer=nn.BatchNorm2d):
        super(AsymmNet, self).__init__()
        inplanes, cfg, cls_ch_squeeze, cls_ch_expand = get_asymmnet_cfgs(cfgs_name)
        k = multiplier
        self.inplanes = make_divisible(inplanes * k)
        self.first_block = nn.Sequential(
            nn.Conv2d(3, self.inplanes, 3, 2, 1, bias=False),
            nn.BatchNorm2d(self.inplanes),
            HardSwish(inplace=True),
        )

        asymm_layers = []
        for layer_cfg in cfg:
            layer = self._make_layer(kernel_size=layer_cfg[0],
                                     exp_ch=make_divisible(k * layer_cfg[1]),
                                     out_channel=make_divisible(k * layer_cfg[2]),
                                     use_se=layer_cfg[3],
                                     act_func=layer_cfg[4],
                                     asymmrate=asymmrate,
                                     stride=layer_cfg[5],
                                     norm_layer=norm_layer,
                                     )
            asymm_layers.append(layer)
        self.asymm_block = nn.Sequential(*asymm_layers)
        self.last_block = nn.Sequential(
            nn.Conv2d(self.inplanes, make_divisible(k * cls_ch_squeeze), 1, bias=False),
            nn.BatchNorm2d(make_divisible(k * cls_ch_squeeze)),
            HardSwish(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(make_divisible(k * cls_ch_squeeze), cls_ch_expand, 1, bias=False),
            HardSwish(),
            nn.Dropout2d(p=dropout_rate, inplace=True),
            nn.Flatten(),
        )
        self.output = nn.Linear(cls_ch_expand, num_classes)

    def _make_layer(self, kernel_size, exp_ch, out_channel, use_se, act_func, asymmrate, stride,
                    norm_layer):
        mid_planes = exp_ch
        out_planes = out_channel
        layer = AsymmBottleneck(self.inplanes, mid_planes,
                                out_planes, kernel_size, asymmrate,
                                act_func, strides=stride, use_se=use_se, norm_layer=norm_layer)
        self.inplanes = out_planes
        return layer

    def forward(self, x):
        x = self.first_block(x)
        x = self.asymm_block(x)
        x = self.last_block(x)
        x = self.output(x)
        return x
"""-----------------------------------------------------------------------------"""