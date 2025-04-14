import torch
from torch import nn

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
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=0, bn_act=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, padding=kernel_size//2, groups=in_channels,
                                   stride=stride, bias=False)
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