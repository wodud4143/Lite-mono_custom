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
    

# region - DepthwiseSeparable
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, dilation=(1, 1), bn_act=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels,  kernel_size, dilation=dilation, padding=1, 
                                   groups=in_channels,
                                   stride=stride, 
                                   bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn_act = bn_act
        
        if self.bn_act:
            # self.bn_gelu = BNGELU(out_channels)
            self.bn_relu = BNRELU(out_channels)
            

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        if self.bn_act:
            # x = self.bn_gelu(x)
            x = self.bn_relu(x)
        
        return x
    

# region - Ghost
# class CustomGhostModule(nn.Module):
#     def __init__(self, in_channels, out_channels, exp):
#         super().__init__()
#         self.exp = exp
#         self.in_channels = in_channels
#         self.out_channels = out_channels
        
#         self.primary_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
#         self.primary_bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.999)
        
#         self.depthwise_conv = nn.Conv2d(out_channels, out_channels * exp, 
#                                        3, padding=1, groups=out_channels, bias=False)
#         self.depthwise_bn = nn.BatchNorm2d(out_channels * exp, eps=1e-3, momentum=0.999)
                                           
        
#     def forward(self, x):
#         x_primary = self.primary_conv(x)
#         x_primary = self.primary_bn(x_primary)
        
#         x_depthwise = self.depthwise_conv(x_primary)
#         x_depthwise = self.depthwise_bn(x_depthwise)
        
#         return torch.cat([x_primary, x_depthwise], dim=1)
        

class CustomGhostModule(nn.Module):
    def __init__(self, exp, in_channels, out_channels):
        super().__init__()
        self.exp = exp
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.primary_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.999)
        )
        
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * (self.exp - 1), kernel_size=3, stride=1, padding=1, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels * (self.exp - 1), eps=1e-3, momentum=0.999)
        )
        
        
    def forward(self, x):
        x_primary = self.primary_conv(x)
        x_depthwise = self.depthwise_conv(x_primary)
        out = torch.cat([x_primary, x_depthwise], dim=1)
        
        return out
        

        
# region - IB
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
            nn.Conv2d(hidden_dim, hidden_dim, 
                      kernel_size=kernel_size, stride=stride,
                      padding=(kernel_size//2)*dilation,
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

"""-----------------------------------------------------------------------------"""

# region InceptionDWConv2d

class InceptionDWConv2d(nn.Module):
    """ Inception depthweise convolution
    """
    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()
        
        gc = int(in_channels * branch_ratio) # channel numbers of a convolution branch
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size//2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size//2), groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size//2, 0), groups=gc)
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)
        
    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)), 
            dim=1,
        )