import math
from timm.layers import DropPath

import torch
import torch.nn.functional as F
from torch import cat, nn
from networks import custom_layers as clayers




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
    


# region - [AsymDC]
class AsymDilatedConv(nn.Module):
    def __init__(self, inc, outc, dilation):
        super().__init__()
        self.expansion_conv = nn.Conv2d(inc, outc, kernel_size=1)
        
        self.conv1x3 = nn.Conv2d(outc, outc, 
                                 kernel_size=(1, 3),
                                 padding=(0, 1))
        self.conv3x1 = nn.Conv2d(outc, outc, 
                                 kernel_size=(3, 1),
                                 padding=(1, 0))
        self.conv3x3 = nn.Conv2d(outc, outc, 
                                 kernel_size=3,
                                 padding=dilation,
                                 dilation=dilation)
        self.bn1 = nn.BatchNorm2d(outc)
        self.act = nn.GELU()
        
        self.reduction_conv = nn.Conv2d(outc, inc, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(inc)
    
    def forward(self, x):
        # 채널 확장 (64 -> 128 -> 256)
        x = self.expansion_conv(x)
        
        x = self.conv1x3(x)
        x = self.conv3x1(x)
        
        x = self.conv3x3(x)
        x = self.bn1(x)
        x = self.act(x)
        
        x = self.reduction_conv(x)
        x = self.bn2(x)
        
        return x
    

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
        
        # x1, x2 = torch.chunk(x, 2, dim=1)

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
    
    
# region - [Ghost]
class CustomGhostModule(nn.Module):
    def __init__(self, inc, outc, exp=1):
        super().__init__()
        self.exp = exp
        self.inc = inc
        self.outc = outc
        
        self.conv1 = nn.Conv2d(inc, outc, kernel_size=1, bias=False)
        self.conv1_bn = nn.BatchNorm2d(outc, eps=1e-3, momentum=0.999)
        
        self.conv2 = nn.Conv2d(outc, outc, kernel_size=3, padding=1, bias=False)
        self.conv2_bn = nn.BatchNorm2d(outc, eps=1e-3, momentum=0.999)

        self.conv3 = nn.Conv2d(outc, inc, kernel_size=1, bias=False)
        
        
    def forward(self, x):
        x_1x1 = self.conv1(x)
        x_1x1 = self.conv1_bn(x_1x1)
        
        x_3x3 = self.conv2(x_1x1)
        x_3x3 = self.conv2_bn(x_3x3)
        
        x_3x3 = self.conv3(x_3x3)
        
        return x_3x3
    
# region - IB
# class InvertedBottleneck(nn.Module):
#     def __init__(self, in_channels, out_channels, expansion=6, kernel_size=3, stride=1, dilation=1, bn_act=False):
#         super().__init__()
#         self.use_residual = (stride == 1 and in_channels == out_channels)
#         hidden_dim = in_channels * expansion

#         self.expand = nn.Sequential(
#             nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
#             nn.BatchNorm2d(hidden_dim)
#             # nn.ReLU6(inplace=True)
#         ) if expansion != 1 else nn.Identity()

#         self.depthwise = nn.Sequential(
#             nn.Conv2d(hidden_dim, hidden_dim, 
#                       kernel_size=kernel_size, stride=stride,
#                       padding=(kernel_size//2)*dilation,
#                       dilation=dilation, groups=hidden_dim, bias=False),
#             nn.BatchNorm2d(hidden_dim)
#             # nn.ReLU6(inplace=True)
#         )

#         self.project = nn.Sequential(
#             nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU6(inplace=True)
#         )

#         self.bn_act = bn_act
#         if self.bn_act:
#             self.act = BNRELU(out_channels)

#     def forward(self, x):
#         residual = x
#         out = self.expand(x)
#         out = self.depthwise(out)
#         out = self.project(out)

#         if self.use_residual:
#             out = out + residual
            

#         if self.bn_act:
#             out = self.act(out)

#         return out


# class DownsampleBlock(nn.Module):
#     def __init__(self, in_channels, expansion_ratio=2):
#         super().__init__()
#         out_channels = in_channels * expansion_ratio

#         self.down = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#         self.project = nn.Sequential(
#             nn.Conv2d(out_channels, in_channels, kernel_size=1, stride=1, bias=False),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         x= self.down(x)
#         x = self.project(x)
        
#         return x
    

"""--------------------Coordinate Attention--------------------------------------"""    
# region coordinate attention 
# class h_sigmoid(nn.Module):
#     def __init__(self, inplace=True):
#         super(h_sigmoid, self).__init__()
#         self.relu = nn.ReLU6(inplace=inplace)

#     def forward(self, x):
#         return self.relu(x + 3) / 6

# class h_swish(nn.Module):
#     def __init__(self, inplace=True):
#         super(h_swish, self).__init__()
#         self.sigmoid = h_sigmoid(inplace=inplace)

#     def forward(self, x):
#         return x * self.sigmoid(x)


# class HardSwish(nn.Module):
#     def __init__(self, inplace=False):
#         super(HardSwish, self).__init__()
#         self.inplace = inplace

#     def forward(self, x):
#         return x * F.relu6(x + 3., inplace=self.inplace) / 6.


# class HardSigmoid(nn.Module):
#     def __init__(self, inplace=False):
#         super(HardSigmoid, self).__init__()
#         self.inplace = inplace

#     def forward(self, x):
#         return F.relu6(x + 3., inplace=self.inplace) / 6.


# class Activation(nn.Module):
#     def __init__(self, act_func):
#         super(Activation, self).__init__()
#         if act_func == "relu":
#             self.act = nn.ReLU()
#         elif act_func == "relu6":
#             self.act = nn.ReLU6()
#         elif act_func == "hard_sigmoid":
#             self.act = HardSigmoid()
#         elif act_func == "hard_swish":
#             self.act = HardSwish()
#         else:
#             raise NotImplementedError

#     def forward(self, x):
#         return self.act(x)


# def make_divisible(x, divisible_by=8):
#     return int(math.ceil(x * 1. / divisible_by) * divisible_by)


# class _BasicUnit(nn.Module):
#     def __init__(self, num_in, num_out, kernel_size=1, strides=1, pad=0, num_groups=1,
#                  use_act=True, act_type="relu", norm_layer=nn.BatchNorm2d):
#         super(_BasicUnit, self).__init__()
#         self.use_act = use_act
#         self.conv = nn.Conv2d(in_channels=num_in, out_channels=num_out,
#                               kernel_size=kernel_size, stride=strides,
#                               padding=pad, groups=num_groups, bias=False,
#                               )
#         self.bn = norm_layer(num_out)
#         if use_act is True:
#             self.act = Activation(act_type)

#     def forward(self, x):
#         out = self.conv(x)
#         out = self.bn(out)
#         if self.use_act:
#             out = self.act(out)
#         return out
