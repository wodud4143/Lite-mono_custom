import math

import torch
import torch.nn.functional as F
from torch import cat, nn
from timm.layers import DropPath


# region - [Conv]
class StandardConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, dilation=(1, 1), bn_act=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 
                              kernel_size,
                              stride, 
                              padding,
                              dilation, 
                              bias=True)
        self.bn_act = bn_act
        
        if self.bn_act:
            self.bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.999)
            self.gelu = nn.GELU()


    def forward(self, x):
        x = self.conv(x)
        if self.bn_act:
            x = self.bn(x)
            x = self.gelu(x)
            
        return x
    
    
# region - [DS]
class DepthwiseSeparable(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, dilation=(1, 1), bn_act=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels,  kernel_size, dilation=dilation, padding=1, 
                                   groups=in_channels,
                                   stride=stride, 
                                   bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn_act = bn_act
        
        if self.bn_act:
            self.bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.999)
            self.gelu = nn.GELU()


    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        if self.bn_act:
            x = self.bn(x)
            x = self.gelu(x)
            
        return x



# region - [LayerNorm]
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)


    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x



# region - [AvgPool]
class AvgPool(nn.Module):
    def __init__(self, ratio):
        super().__init__()
    
        assert ratio in [2, 4, 8, 16]
        if ratio == 2:
            self.pool = nn.AvgPool2d(3, stride=2, padding=1)
            
        elif ratio == 4:
            self.pool = nn.Sequential(
                nn.AvgPool2d(3, stride=2, padding=1),
                nn.AvgPool2d(3, stride=2, padding=1)
            )
            
        elif ratio == 8:
            self.pool = nn.Sequential(
                nn.AvgPool2d(3, stride=2, padding=1),
                nn.AvgPool2d(3, stride=2, padding=1),
                nn.AvgPool2d(3, stride=2, padding=1)
            )
            
        elif ratio == 16:
            self.pool = nn.Sequential(
                nn.AvgPool2d(3, stride=2, padding=1),
                nn.AvgPool2d(3, stride=2, padding=1),
                nn.AvgPool2d(3, stride=2, padding=1),
                nn.AvgPool2d(3, stride=2, padding=1)
            )

    def forward(self, x):
        return self.pool(x)