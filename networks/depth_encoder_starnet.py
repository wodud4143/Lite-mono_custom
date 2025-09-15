import math
import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torch.cuda

import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_

from networks import core_layer as core
from networks import custom_layers as clayers


class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)

class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x



# region - Main Arch
class LiteMono(nn.Module):
    """
    Lite-Mono
    """
    def __init__(self, in_chans=3, model='lite-mono', height=192, width=640,
                 global_block=[1, 1, 1], global_block_type=['LGFI', 'LGFI', 'LGFI'],
                 drop_path_rate=0.2, layer_scale_init_value=1e-6, expan_ratio=6,
                 heads=[8, 8, 8], mlp_ratio=3, use_pos_embd_xca=[True, False, False], **kwargs):
        super().__init__()

        self.num_ch_enc = np.array([32, 64, 128])
        self.depth = [4, 6, 4]  # depth 수정 3,5,3
        self.dims = [32, 64, 128]


        if height == 192 and width == 640:
            self.dilation = [[1, 2, 3], [1, 2, 3], [1, 4, 6]] # 다일레이션 수정
            

        for g in global_block_type:
            assert g in ['None', 'LGFI']


        self.avg_pool2 = clayers.AvgPool(ratio=2) # 1/2
        self.avg_pool4 = clayers.AvgPool(ratio=4) # 1/4
        self.avg_pool8 = clayers.AvgPool(ratio=8) # 1/8

        
        # stem layer: 입력 이미지를 처리하고 초기 특성맵 생성
        self.stem = nn.Sequential(
            ConvBN(3, self.dims[0], kernel_size=3, stride=2, padding=1), 
            nn.ReLU6()
        )
        
        # 채널 맞추기용 pointwise conv 레이어들
        self.pw1 = ConvBN(32 + 3, 32, 1)    # stem 출력(32) + x_down2(3) -> 32
        self.pw2 = ConvBN(32 + 3, 32, 1)    # stage1 출력(32) + x_down4(3) -> 64  
        self.pw3 = ConvBN(64 + 3, 64, 1)   # stage2 출력(64) + x_down8(3) -> 128
        
        # stochastic depth를 위한 drop path rate 계산
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depth))]
        self.stages = nn.ModuleList()
        
        self.downsample1 = ConvBN(in_planes=self.dims[0], out_planes=self.dims[0], kernel_size=3, stride=2, padding=1)
        self.downsample2 = ConvBN(in_planes=self.dims[0], out_planes=self.dims[1], kernel_size=3, stride=2, padding=1)
        self.downsample3 = ConvBN(in_planes=self.dims[1], out_planes=self.dims[2], kernel_size=3, stride=2, padding=1)
                
        stage_blocks = [
            Block(dim=self.dims[0], mlp_ratio=mlp_ratio, drop_path=dp_rates[0]),
            Block(dim=self.dims[0], mlp_ratio=mlp_ratio, drop_path=dp_rates[1]),
            Block(dim=self.dims[0], mlp_ratio=mlp_ratio, drop_path=dp_rates[2]),
            core.LGFI(dim=self.dims[0], 
                      drop_path=dp_rates[3], 
                      expan_ratio=expan_ratio,
                      use_pos_emb=use_pos_embd_xca[0], 
                      num_heads=heads[0], 
                      layer_scale_init_value=layer_scale_init_value) 
        ]
        self.stages.append(nn.Sequential(*stage_blocks))
        
        stage_blocks = [
            Block(dim=self.dims[1], mlp_ratio=mlp_ratio, drop_path=dp_rates[4]),
            Block(dim=self.dims[1], mlp_ratio=mlp_ratio, drop_path=dp_rates[5]),
            Block(dim=self.dims[1], mlp_ratio=mlp_ratio, drop_path=dp_rates[6]),
            Block(dim=self.dims[1], mlp_ratio=mlp_ratio, drop_path=dp_rates[7]),
            Block(dim=self.dims[1], mlp_ratio=mlp_ratio, drop_path=dp_rates[8]),
            core.LGFI(dim=self.dims[1], 
                      drop_path=dp_rates[9], 
                      expan_ratio=expan_ratio,
                      use_pos_emb=use_pos_embd_xca[1], 
                      num_heads=heads[1], 
                      layer_scale_init_value=layer_scale_init_value)
        ]
        self.stages.append(nn.Sequential(*stage_blocks))
        
        stage_blocks = [
            Block(dim=self.dims[2], mlp_ratio=mlp_ratio, drop_path=dp_rates[10]),
            Block(dim=self.dims[2], mlp_ratio=mlp_ratio, drop_path=dp_rates[11]),
            Block(dim=self.dims[2], mlp_ratio=mlp_ratio, drop_path=dp_rates[12]),
            core.LGFI(dim=self.dims[2], 
                      drop_path=dp_rates[13], 
                      expan_ratio=expan_ratio,
                      use_pos_emb=use_pos_embd_xca[2], 
                      num_heads=heads[2], 
                      layer_scale_init_value=layer_scale_init_value)
        ]
        self.stages.append(nn.Sequential(*stage_blocks))        

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # region [forward]
    def forward(self, x):
        x = (x - 0.45) / 0.225
        
        x_down2 = self.avg_pool2(x)
        x_down4 = self.avg_pool4(x)
        x_down8 = self.avg_pool8(x)
        
        """(32, 96, 320)"""
        ds2 = self.stem(x)  
        
        """(35, 96, 320)"""
        ds2 = torch.cat([ds2, x_down2], dim=1)  
        
        """(32, 96, 320)"""
        ds2 = self.pw1(ds2)  
        
        """(32, 48, 160)"""
        ds4 = self.downsample1(ds2)
        
        """(32, 48, 160)"""
        for s in range(len(self.stages[0])-1):
            ds4 = self.stages[0][s](ds4) #Asymmblock
        ds4 = self.stages[0][-1](ds4) #LGFI
        
        """(35, 48, 160)"""
        ds4 = torch.cat([ds4, x_down4], dim=1)  
        
        """(32, 48, 160)"""
        ds4 = self.pw2(ds4)  
        
        """(64, 24, 80)"""
        ds8 = self.downsample2(ds4)
        
        """(64, 24, 80)"""
        for s in range(len(self.stages[1])-1):
            ds8 = self.stages[1][s](ds8) #Asymmblock
        ds8 = self.stages[1][-1](ds8) #LGFI
        
        """(67, 24, 80)"""
        ds8 = torch.cat([ds8, x_down8], dim=1)  
        
        """(64, 24, 80)"""
        ds8 = self.pw3(ds8)  
        
        """(128, 12, 40)"""
        ds16 = self.downsample3(ds8)

        """(128, 12, 40)"""
        for s in range(len(self.stages[2]) - 1):
            ds16 = self.stages[2][s](ds16) #Asymmblock
        ds16 = self.stages[2][-1](ds16) #LGFI
        
        return ds4, ds8, ds16
    