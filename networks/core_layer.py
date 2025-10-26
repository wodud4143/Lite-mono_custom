import math
from timm.layers import DropPath

import torch
import torch.nn.functional as F
from torch import cat, nn
from networks import custom_layers as clayers
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from functools import partial


class PositionalEncodingFourier(nn.Module):
    """
    Positional encoding relying on a fourier kernel matching the one used in the
    "Attention is all of Need" paper. The implementation builds on DeTR code
    https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
    """

    def __init__(self, hidden_dim=32, dim=768, temperature=10000):
        super().__init__()
        self.token_projection = nn.Conv2d(hidden_dim * 2, dim, kernel_size=1)
        self.scale = 2 * math.pi
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.dim = dim

    def forward(self, B, H, W):
        mask = torch.zeros(B, H, W).bool().to(self.token_projection.weight.device)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.hidden_dim, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.hidden_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(),
                             pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(),
                             pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = self.token_projection(pos)
        return pos

# region - XCA
class XCA(nn.Module):
    """ Cross-Covariance Attention (XCA) operation where the channels are updated using a weighted
    sum. The weights are obtained from the (softmax normalized) Cross-covariance
    matrix (Q^T K \\in d_h \\times d_h)
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        
        self.last_attn = None


    def forward(self, x):
        # (B, C, H, W) ---> convolution operation
        # ViT : Patch --> position embedding 
        # (B, C, H, W) X --> (B, HxW, C) --> matrix multiplication --> Transpose (4, 3)--> (3, 4)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        # 144 --> 3, 8, 6
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

   
        return x


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature'}





# region - LGFI
class LGFI(nn.Module):
    """
    Local-Global Features Interaction
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, expan_ratio=6,
                 use_pos_emb=True, num_heads=6, qkv_bias=True, attn_drop=0., drop=0.):
        super().__init__()

        self.dim = dim
        self.pos_embd = None
        if use_pos_emb:
            self.pos_embd = PositionalEncodingFourier(dim=self.dim)

        self.norm_xca = clayers.LayerNorm(self.dim, eps=1e-6)

        self.gamma_xca = nn.Parameter(layer_scale_init_value * torch.ones(self.dim),
                                      requires_grad=True) if layer_scale_init_value > 0 else None
        
        self.xca = XCA(self.dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        
        
        self.norm = clayers.LayerNorm(self.dim, eps=1e-6)
        self.pwconv1 = nn.Linear(self.dim, expan_ratio * self.dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expan_ratio * self.dim, self.dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((self.dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


    def forward(self, x):
        input_ = x
        
        B, C, H, W = x.shape
        
        x = x.reshape(B, C, H * W).permute(0, 2, 1)  
        
        if self.pos_embd:
            pos_encoding = self.pos_embd(B, H, W).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
            x = x + pos_encoding
    
            
        x = x + self.gamma_xca * self.xca(self.norm_xca(x))
    
        
        x = x.reshape(B, H, W, C)

        # Inverted Bottleneck
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input_ + self.drop_path(x)
    
        return x


# region - CDilated
class CDilated(nn.Module):
    """
    This class defines the dilated convolution.
    """

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1, bias=False):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        """
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=bias,
                              dilation=d, groups=groups)

    def forward(self, input):
        """
        :param input: input feature map
        :return: transformed feature map
        """

        output = self.conv(input)
        return output
            

# region - Dilated
class DilatedConv(nn.Module):
    """
    A single Dilated Convolution layer in the Consecutive Dilated Convolutions (CDC) module.
    """
    def __init__(self, dim, k, dilation=1, stride=1, drop_path=0.,
                 layer_scale_init_value=1e-6, expan_ratio=6):
        """
        :param dim: input dimension
        :param k: kernel size
        :param dilation: dilation rate
        :param drop_path: drop_path rate
        :param layer_scale_init_value:
        :param expan_ratio: inverted bottelneck residual
        """

        super().__init__()

        self.ddwconv = CDilated(dim, dim, kSize=k, stride=stride, groups=dim, d=dilation)
        self.bn1 = nn.BatchNorm2d(dim)

        self.norm = clayers.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, expan_ratio * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expan_ratio * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x

        x1 = self.ddwconv(x1)
        x1 = self.bn1(x1)
        
        x2 = x2.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x2 = self.pwconv1(x2)
        x2 = self.act(x2)
        x2 = self.pwconv2(x2)
        
        if self.gamma is not None:
            x2 = self.gamma * x2
        x2 = x2.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = torch.cat([x1, x2], dim=1)
        x = input + self.drop_path(x)

        return x
    
    
# region - [AsymDC]
# class AsymDilatedConv(nn.Module):
#     def __init__(self, inc, outc, dilation):
#         super().__init__()
#         self.expansion_conv = nn.Conv2d(inc, outc, kernel_size=1)
        
#         self.conv1x3 = nn.Conv2d(outc, outc, 
#                                  kernel_size=(1, 3),
#                                  padding=(0, 1))
#         self.conv3x1 = nn.Conv2d(outc, outc, 
#                                  kernel_size=(3, 1),
#                                  padding=(1, 0))
#         self.conv3x3 = nn.Conv2d(outc, outc, 
#                                  kernel_size=3,
#                                  padding=dilation,
#                                  dilation=dilation)
#         self.bn1 = nn.BatchNorm2d(outc)
#         self.act = nn.GELU()
        
#         self.reduction_conv = nn.Conv2d(outc, inc, kernel_size=1)
#         self.bn2 = nn.BatchNorm2d(inc)
    
#     def forward(self, x):
#         # 채널 확장 (64 -> 128 -> 256)
#         x = self.expansion_conv(x)
        
#         x = self.conv1x3(x)
#         x = self.conv3x1(x)
        
#         x = self.conv3x3(x)
#         x = self.bn1(x)
#         x = self.act(x)
        
#         x = self.reduction_conv(x)
#         x = self.bn2(x)
        
#         return x
    
class AsymDilatedConv(nn.Module):
    def __init__(self, inc, outc, dilation, drop_path=0.0, residual=False):
        super().__init__()
        self.residual = residual
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        
        self.expansion_conv = nn.Conv2d(inc, outc, kernel_size=1)
        
        self.conv1x5 = nn.Conv2d(outc, outc, 
                                 kernel_size=(1, 5),
                                 padding=(0, 2)
                                 )
        self.conv5x1 = nn.Conv2d(outc, outc, 
                                 kernel_size=(5, 1),
                                 padding=(2, 0) 
                                 ) 
        
        self.dw3x3 = nn.Conv2d(outc*2, outc*2, 3, padding=dilation,
                       dilation=dilation, groups=outc*2, bias=False)
        self.pw1x1 = nn.Conv2d(outc*2, outc, 1, bias=False)
        self.bn_dw  = nn.BatchNorm2d(outc*2)
        self.bn_pw  = nn.BatchNorm2d(outc)
        
        self.bn1 = nn.BatchNorm2d(outc, eps=1e-3, momentum=0.999)
        self.act = nn.ReLU6()
        
        self.reduction_conv = nn.Conv2d(outc, inc, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(inc, eps=1e-3, momentum=0.999)
    
    def forward(self, x):
        # 채널 확장 (64 -> 128 -> 256)
        identity = x
        x = self.expansion_conv(x)
        
        x1 = self.conv1x5(x)
        x2 = self.conv5x1(x)
        
        x = torch.cat([x1,x2],dim=1)

        x = self.dw3x3(x)
        x = self.bn_dw(x)
        x = self.act(x)
        x = self.pw1x1(x)
        x = self.bn_pw(x)
        x = self.act(x)
        
        x = self.act(x1) * x2
        x = self.conv2(self.g(x))
        x = input + self.drop_path(x)
                
        x = self.reduction_conv(x)
        x = self.bn2(x)
        
        if self.residual:
            x = identity + self.drop_path(x)
            return self.act(x)        
        else:
            return self.act(x)  


# class AsymDilatedConv(nn.Module):
#     def __init__(self, inc, outc, dilation, drop_path=0.0, residual=False):
#         super().__init__()
#         self.residual = residual
#         self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        
        
#         self.conv1x5 = nn.Conv2d(inc, inc*2, kernel_size=(1, 5), padding=(0, 2*dilation), dilation=(1, dilation))
#         self.conv5x1 = nn.Conv2d(inc, inc*2, kernel_size=(5, 1), padding=(2*dilation, 0), dilation=(dilation, 1))

        
#         self.bn1x5 = nn.BatchNorm2d(inc*2, eps=1e-3, momentum=0.999)
#         self.bn5x1 = nn.BatchNorm2d(inc*2, eps=1e-3, momentum=0.999)
        
#         self.reduction_conv = nn.Conv2d(inc*2, inc, kernel_size=1)
#         self.reduction_conv_bn = nn.BatchNorm2d(inc, eps=1e-3, momentum=0.999)
        
#         self.conv_3x3 = nn.Conv2d(inc, inc, 
#                                  kernel_size=3,
#                                  stride= 1,
#                                  padding=1
#                                  ) 
#         self.conv_3x3_bn = nn.BatchNorm2d(inc, eps=1e-3, momentum=0.999)
        
#         self.act = nn.ReLU6(inplace=True)
        
        
    
#     def forward(self, x):
#         # 채널 확장 (64 -> 128 -> 256)
#         identity = x
        
#         x1 = self.conv1x5(x)
#         x1 = self.bn1x5(x1)
        
#         x2 = self.conv5x1(x)
#         x2 = self.bn5x1(x2)
        
#         x = torch.add(x1,x2)

#         x = self.act(x) * x
        
#         x = self.reduction_conv_bn(self.reduction_conv(x))
    
#         x = self.conv_3x3_bn(self.conv_3x3(x))
       
#         x = identity + self.drop_path(x)
        
#         return x      

    
# region - [Ghost]

class CustomGhostModule(nn.Module):
    def __init__(self, inc, outc = None, exp=2):
        super().__init__()

        self.midc = inc * 2
        c_half = self.midc // 2
        

        self.expand_pw = nn.Conv2d(inc, inc * exp, kernel_size=1, bias=False)
        self.expand_bn = nn.BatchNorm2d(self.midc,eps=1e-3, momentum=0.999)
        
        self.reduce_pw = nn.Conv2d(c_half, inc, kernel_size=1, bias=False)
        self.reduce_bn = nn.BatchNorm2d(inc ,eps=1e-3, momentum=0.999)
        
        self.conv1x1 = nn.Conv2d(inc * exp, c_half, kernel_size=1, bias=False)

        self.ex_conv = nn.Conv2d(c_half, c_half, kernel_size=3, padding=1, bias=False)
        self.ex_bn = nn.BatchNorm2d(c_half,eps=1e-3, momentum=0.999)
        
        self.ch_conv = nn.Conv2d(c_half, c_half, kernel_size=3, padding=2, dilation=2, bias=False)
        self.ch_bn = nn.BatchNorm2d(c_half,eps=1e-3, momentum=0.999)
        # self.act = nn.GELU()
        self.act = nn.ReLU6()
   
        
    def forward(self, x):
        identity = x

        x = self.expand_pw(x)
        x = self.expand_bn(x)
        x = self.act(x)
        
        # x1, x2 = torch.chunk(x, 2, dim=1)
        
        x1 = self.conv1x1(x)
        x2 = self.conv1x1(x)
        
        x1 = self.ex_conv(x1)
        x1 = self.ex_bn(x1)
        x1 = self.act(x1)
        
        x2 = self.ch_conv(x2)
        x2 = self.ch_bn(x2)
        x2 = self.act(x2)
        
        # x = torch.cat([x1, x2], dim=1)
        
        x = x1 * x2 
        
        x = self.reduce_pw(x)
        x = self.reduce_bn(x)
        
        x = x + identity
        return x


# region [StarModule]
class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)

# class Star(nn.Module):
#     def __init__(self, dim, mlp_ratio=3, drop_path=0.):
#         super().__init__()
#         self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
#         self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
#         self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
#         self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
#         self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
#         self.act = nn.ReLU6()
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
#     def forward(self, x):
#         input = x
#         x = self.dwconv(x)
#         x1, x2 = self.f1(x), self.f2(x)
#         x = self.act(x1) * x2
#         x = self.dwconv2(self.g(x))
#         x = input + self.drop_path(x)
#         return x

class Star(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.conv1 = ConvBN(dim, dim, 5, 1, (5 - 1) // 2, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.conv2 = ConvBN(dim, dim, 5, 1, (5 - 1) // 2, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, x):
        # input = x
        x_init = self.conv1(x)
        x1, x2 = self.f1(x_init), self.f2(x_init)
        x = self.act(x1) * x2
        x = self.conv2(self.g(x))
        x = x_init + x
        # x = input + self.drop_path(x)
        return x