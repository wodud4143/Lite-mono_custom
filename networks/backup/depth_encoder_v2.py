import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from timm.layers import DropPath
import math
import torch.cuda
from .model_utils import Conv, CoordAtt, DepthwiseSeparableConv



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


# region - LayerNorm
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

# region - InceptionDWConv2d
class InceptionDWConv2dCDilated(nn.Module):
    """ Inception depthweise convolution
    """
    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125,d=1):
        super().__init__()
        
        padding = int((square_kernel_size - 1) / 2) * d
        gc = int(in_channels * branch_ratio) # channel numbers of a convolution branch
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=padding, dilation=d, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size//2), groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size//2, 0), groups=gc)
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)
        
    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)), 
            dim=1,
        )

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


# region - Half Dims DDW Dilated
# class DilatedConv(nn.Module):
#     """
#     A single Dilated Convolution layer in the Consecutive Dilated Convolutions (CDC) module.
#     """
#     def __init__(self, dim, k, dilation=1, stride=1, drop_path=0.,
#                  layer_scale_init_value=1e-6, expan_ratio=6):
#         """
#         :param dim: input dimension
#         :param k: kernel size
#         :param dilation: dilation rate
#         :param drop_path: drop_path rate
#         :param layer_scale_init_value:
#         :param expan_ratio: inverted bottelneck residual
#         """

#         super().__init__()

        
#         self.ddwconv = CDilated(dim, dim, kSize=k, stride=stride, groups=dim, d=dilation)
#         self.bn1 = nn.BatchNorm2d(dim)

#         self.norm = LayerNorm(dim, eps=1e-6)
#         self.pwconv1 = nn.Linear(dim, expan_ratio * dim)
#         self.act = nn.GELU()
#         self.pwconv2 = nn.Linear(expan_ratio * dim, dim)
#         self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim),
#                                   requires_grad=True) if layer_scale_init_value > 0 else None
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#     def forward(self, x):
#         input = x # torch.Size([8, 48, 48, 160]), torch.Size([8, 80, 24, 80]), torch.Size([8, 128, 12, 40])
        
#         x1, x2 = torch.chunk(x, 2, dim=1)
#         # x1,x2 = input / 2         
#         # 24 + 40 + 64 = 128
#         # 6 + 10 + 16 = 32 * 3 = 64
        
#         x1 = self.ddwconv(x1)
#         x1 = self.bn1(x1)
        
#         x2 = x2.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
#         x2 = self.pwconv1(x2)
#         x2 = self.act(x2)
#         x2 = self.pwconv2(x2)
        
#         if self.gamma is not None:
#             x2 = self.gamma * x2
#         x2 = x2.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

#         x = torch.cat([x1, x2], dim=1)
#         x = input + self.drop_path(x)

#         return x

# region Inception Dilated
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

        
        # self.inceptionddwconv = CDilated(dim, dim, kSize=k, stride=stride, groups=dim, d=dilation)
        self.inceptionddwconv = InceptionDWConv2dCDilated(dim, d=dilation)
        
        self.bn1 = nn.BatchNorm2d(dim)

        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, expan_ratio * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expan_ratio * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x # torch.Size([8, 48, 48, 160]), torch.Size([8, 80, 24, 80]), torch.Size([8, 128, 12, 40])
        x = self.inceptionddwconv(x)
        self.bn1(x)
        x = x.permute(0, 2, 3, 1)
        x = self.act(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
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

        self.norm_xca = LayerNorm(self.dim, eps=1e-6)

        self.gamma_xca = nn.Parameter(layer_scale_init_value * torch.ones(self.dim),
                                      requires_grad=True) if layer_scale_init_value > 0 else None
        
        self.xca = XCA(self.dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        
        
        self.norm = LayerNorm(self.dim, eps=1e-6)
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


# region - AvgPool
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



# region - Main Arch
class LiteMono(nn.Module):
    """
    Lite-Mono
    """
    def __init__(self, in_chans=3, model='lite-mono', height=192, width=640,
                 global_block=[1, 1, 1], global_block_type=['LGFI', 'LGFI', 'LGFI'],
                 drop_path_rate=0.2, layer_scale_init_value=1e-6, expan_ratio=6,
                 heads=[8, 8, 8], use_pos_embd_xca=[True, False, False], **kwargs):
        super().__init__()

        # if model == 'lite-mono':
        self.num_ch_enc = np.array([48, 80, 128])
        self.depth = [4, 4, 10]
        # self.depth = [2, 3, 6]
        self.dims = [48, 80, 128]

        if height == 192 and width == 640:
            self.dilation = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 1, 2, 3, 2, 4, 6]]

        elif height == 320 and width == 1024:
            self.dilation = [[1, 2, 5], [1, 2, 5], [1, 2, 5, 1, 2, 5, 2, 4, 10]]

        for g in global_block_type:
            assert g in ['None', 'LGFI']


        self.avg_pool2 = AvgPool(ratio=2)
        self.avg_pool4 = AvgPool(ratio=4)
        self.avg_pool8 = AvgPool(ratio=8)

        
        self.init_conv = nn.Sequential(
            Conv(nIn=in_chans, nOut=self.dims[0], kSize=3, stride=2, padding=1, bn_act=True),
            # InvertedBottleneck(in_channels=self.dims[0], out_channels=self.dims[0], expansion=2, kernel_size=3, bn_act=True),
            # InvertedBottleneck(in_channels=self.dims[0], out_channels=self.dims[0], expansion=2, kernel_size=3, bn_act=True),
            # CoordAtt(self.dims[0], self.dims[0])
        )
        
        self.depthwise_conv = DepthwiseSeparableConv(self.dims[0]+3, self.dims[0], kernel_size=3)
        # self.ca_layer = CoordAtt(self.dims[0]+3, self.dims[0]+3)
        # self.ghost_layer = CustomGhostModule(in_channels=self.dims[0]+3, 
        #                                     out_channels=self.dims[0]//2, 
        #                                     exp=2)
        
        self.downsample_layer2 = nn.Sequential(
            Conv(self.dims[0]*2+3, self.dims[1], kSize=3, stride=2, padding=1, bn_act=False)
        )
        self.downsample_layer3 = nn.Sequential(
            Conv(self.dims[1]*2+3, self.dims[2], kSize=3, stride=2, padding=1, bn_act=False)
        )
        
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depth))]
        cur = 0

        # add_dims=[24, 0, 0]
        for i in range(3):
            stage_blocks = []
            for j in range(self.depth[i]):
                if j > self.depth[i] - global_block[i] - 1:
                    if global_block_type[i] == 'LGFI':
                        # print('LGFI')
                        stage_blocks.append(LGFI(dim=self.dims[i], drop_path=dp_rates[cur + j],
                                                 expan_ratio=expan_ratio,
                                                 use_pos_emb=use_pos_embd_xca[i], num_heads=heads[i],
                                                 layer_scale_init_value=layer_scale_init_value,
                                                 ))

                    else:
                        raise NotImplementedError
                else:
                    # print('CDC')
                    stage_blocks.append(DilatedConv(dim=self.dims[i], k=3, dilation=self.dilation[i][j], drop_path=dp_rates[cur + j],
                                                    layer_scale_init_value=layer_scale_init_value,
                                                    expan_ratio=expan_ratio))
            print(' ')
            self.stages.append(nn.Sequential(*stage_blocks))
            cur += self.depth[i]

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    # region [forward]
    def forward(self, x):
        x = (x - 0.45) / 0.225 # torch.Size([8, 3, 192, 640])
        
        x_down2 = self.avg_pool2(x) # torch.Size([8, 3, 96, 320])
        x_down4 = self.avg_pool4(x) # torch.Size([8, 3, 48, 160])
        x_down8 = self.avg_pool8(x) # torch.Size([8, 3, 24, 80])
        
        """-------------Stage1-----------------"""
        stage1_ds2 = self.init_conv(x) # torch.Size([8, 48, 96, 320])
        stage1_ds2 = torch.cat((stage1_ds2, x_down2), dim=1) # ch=48+3=51
        stage1_ds4 = self.depthwise_conv(stage1_ds2) # ch=48
        # stage1_ds2 = self.ca_layer(stage1_ds2)
        # stage1_ds4 = self.avg_pool2(stage1_ds2)
        # stage1_ds4 = self.ghost_layer(stage1_ds4) # ch=48+24=72
        """------------------------------------"""

        """------------Stage2-----------------"""
        # for s in range(len(self.stages[0])-1):
        #     stage1_ds4 = self.stages[0][s](stage1_ds4)  # ch=48+24=72
        # stage2_ds4 = self.stages[0][-1](stage1_ds4)     # ch=48+24=72
        
        # CDC
        for s in range(len(self.stages[0])-1): # len(self.stages[0]) = 4, self.stages[0] -> depth[0]
            stage2_ds4 = self.stages[0][s](stage1_ds4)  # ch=48
        # LGFI
        stage2_ds4 = self.stages[0][-1](stage2_ds4)     # ch=48
        """-----------------------------------"""
        
        """------------Stage3-----------------"""
        stage3_ds4 = torch.cat([stage1_ds4, stage2_ds4, x_down4], dim=1)  # channel=99
        stage3_ds8 = self.downsample_layer2(stage3_ds4) # channel=80
        # CDC
        for s in range(len(self.stages[1]) - 1):
            stage3_ds8 = self.stages[1][s](stage3_ds8)
        # LGFI
        stage3_ds8_10 = self.stages[1][-1](stage3_ds8) # channel=80
        """-----------------------------------"""
        
        """------------Stage4-----------------"""
        stage4_ds8 = torch.cat([stage3_ds8, stage3_ds8_10, x_down8], dim=1) # channel=163

        stage4_ds16 = self.downsample_layer3(stage4_ds8) # channel=128
        # CDC
        for s in range(len(self.stages[2]) - 1):
            stage4_ds16 = self.stages[2][s](stage4_ds16)
        # LGFI
        stage4_ds16_10 = self.stages[2][-1](stage4_ds16) # channel=128
        """-----------------------------------"""
        
        
        # return features
        return stage2_ds4, stage3_ds8_10, stage4_ds16_10
