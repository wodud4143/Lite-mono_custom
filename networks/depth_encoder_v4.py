import math
import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torch.cuda

from networks import core_layer as core
from networks import custom_layers as clayers




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

        self.num_ch_enc = np.array([32, 64, 128])
        self.depth = [3, 3, 4]  # depth 수정
        self.dims = [32, 64, 128]
        self.asym_dims = [32, 64, 128]

        if height == 192 and width == 640:
            self.dilation = [[1, 2], [2, 1], [2, 3, 5]] # 다일레이션 수정
            
        for g in global_block_type:
            assert g in ['None', 'LGFI']


        # self.avg_pool2 = clayers.AvgPool(ratio=2) # 1/2
        # self.avg_pool4 = clayers.AvgPool(ratio=4) # 1/4
        # self.avg_pool8 = clayers.AvgPool(ratio=8) # 1/8
        
        self.input_conv_s2 = clayers.StandardConv(in_chans, self.dims[0], kernel_size=3, stride=2, padding=1, bn_act=True) # 1/2
        self.input_conv_s4 = clayers.StandardConv(in_chans, self.dims[0], kernel_size=3, stride=4, padding=1, bn_act=True) # 1/4
        self.input_conv_s8 = clayers.StandardConv(in_chans, self.dims[1], kernel_size=3, stride=8, padding=1, bn_act=True) # 1/8

        
        self.init_conv = nn.Sequential(
            clayers.StandardConv(in_chans, self.dims[0],
                              kernel_size=3, 
                              stride=2,
                              padding=1, 
                              bn_act=True)
        )
        # self.ds_conv1 = clayers.StandardConv(self.dims[0]+3, self.dims[0],
        #                                   kernel_size=3,
        #                                   stride=2, 
        #                                   padding=1, 
        #                                   bn_act=True)
        
        self.ds_conv_3x3_32 = clayers.StandardConv(self.dims[0], self.dims[0], kernel_size=3, stride=2, padding=1, bn_act=True)
        
        self.cghost_layer = core.CustomGhostModule(self.dims[0], self.dims[0]//2)

        self.ds_conv2 = clayers.StandardConv(self.dims[0], self.dims[1],
                                          kernel_size=3,
                                          stride=2, 
                                          padding=1, 
                                          bn_act=True)
        self.cghost2_layer = core.CustomGhostModule(self.dims[1], self.dims[1]//2)

        
        # self.downsample_layer2 = nn.Sequential(
        #     clayers.StandardConv(self.dims[0]+3, self.dims[1], 
        #                          kernel_size=3, 
        #                          stride=2,
        #                          padding=1, 
        #                          bn_act=False)
        # )
        
        self.add_ds_conv_64 = clayers.StandardConv(self.dims[0], self.dims[1], kernel_size=3, stride=2, padding=1, bn_act=False)
        
        # self.downsample_layer3 = nn.Sequential(
        #     clayers.StandardConv(self.dims[1]+3, self.dims[2], 
        #                          kernel_size=3, 
        #                          stride=2, 
        #                          padding=1, 
        #                          bn_act=False)
        # )
        
        self.add_ds_conv_128 = clayers.StandardConv(self.dims[1], self.dims[2], kernel_size=3, stride=2, padding=1, bn_act=False)
        
        
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depth))]

        stage_blocks = [
            core.AsymDilatedConv(inc=self.dims[0], outc=self.asym_dims[0], dilation=self.dilation[0][0],drop_path=dp_rates[0],residual=True),
            core.AsymDilatedConv(inc=self.dims[0], outc=self.asym_dims[0], dilation=self.dilation[0][1],drop_path=dp_rates[0 + 1],residual=True),
            # core.AsymDilatedConv(inc=self.dims[0], outc=self.asym_dims[0], dilation=self.dilation[0][2],drop_path=dp_rates[0 + 2],residual=True),
            core.LGFI(dim=self.dims[0], 
                      drop_path=dp_rates[0 + 2], 
                      expan_ratio=expan_ratio,
                      use_pos_emb=use_pos_embd_xca[0], 
                      num_heads=heads[0], 
                      layer_scale_init_value=layer_scale_init_value) 
        ]
        self.stages.append(nn.Sequential(*stage_blocks))
        
        stage_blocks = [
            core.AsymDilatedConv(inc=self.dims[1], outc=self.asym_dims[1], dilation=self.dilation[1][0],drop_path=dp_rates[2 + 1],residual=True),
            core.AsymDilatedConv(inc=self.dims[1], outc=self.asym_dims[1], dilation=self.dilation[1][1],drop_path=dp_rates[2 + 2],residual=True),
            # core.AsymDilatedConv(inc=self.dims[1], outc=self.asym_dims[1], dilation=self.dilation[1][2],drop_path=dp_rates[3 + 3],residual=True),
            core.LGFI(dim=self.dims[1], 
                      drop_path=dp_rates[2 + 3], 
                      expan_ratio=expan_ratio,
                      use_pos_emb=use_pos_embd_xca[1], 
                      num_heads=heads[1], 
                      layer_scale_init_value=layer_scale_init_value)
        ]
        self.stages.append(nn.Sequential(*stage_blocks))
        
        stage_blocks = [
            core.AsymDilatedConv(inc=self.dims[2], outc=self.asym_dims[2], dilation=self.dilation[2][0],drop_path=dp_rates[5 + 1],residual=True),
            core.AsymDilatedConv(inc=self.dims[2], outc=self.asym_dims[2], dilation=self.dilation[2][1],drop_path=dp_rates[5 + 2],residual=True),
            core.AsymDilatedConv(inc=self.dims[2], outc=self.asym_dims[2], dilation=self.dilation[2][2],drop_path=dp_rates[5 + 3],residual=True),
            core.LGFI(dim=self.dims[2], 
                      drop_path=dp_rates[5 + 4], 
                      expan_ratio=expan_ratio,
                      use_pos_emb=use_pos_embd_xca[2], 
                      num_heads=heads[2], 
                      layer_scale_init_value=layer_scale_init_value)
        ]
        self.stages.append(nn.Sequential(*stage_blocks))        

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        elif isinstance(m, (clayers.LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    # region [forward]
    def forward(self, x):
        x = (x - 0.45) / 0.225
        
        # x_down2 = self.avg_pool2(x)
        # x_down4 = self.avg_pool4(x)
        # x_down8 = self.avg_pool8(x)
        
        x_down2 = self.input_conv_s2(x) # 3, 32
        x_down4 = self.input_conv_s4(x) # 3, 32
        x_down8 = self.input_conv_s8(x) # 3, 64
        
        """(32, 96, 320)"""
        ds2 = self.init_conv(x) # 3 ,32
        
        """(32, 96, 320)"""
        ds2 = torch.add(ds2,x_down2)
        
        # """(35, 96, 320)"""
        # ds2 = torch.cat((ds2, x_down2), dim=1)

        """(32, 48, 160)"""
        # ds4 = self.ds_conv1(ds2) # StandardConv(stride = 2)
        ds4 = self.ds_conv_3x3_32(ds2) # 32, 32
        ds4 = self.cghost_layer(ds4) # 32, 32

        """(32, 48, 160)"""
        for s in range(len(self.stages[0])-1):
            ds4 = self.stages[0][s](ds4) #Asymmblock
        ds4 = self.stages[0][-1](ds4) #LGFI
        
        # """(35, 48, 160)"""
        # concat_ds4 = torch.cat([ds4, x_down4], dim=1)
        
        """(32, 48, 160)"""
        add_ds4 = torch.add(ds4,x_down4) 
        
    
        # """(64, 24, 80)"""
        # ds8 = self.downsample_layer2(concat_ds4) # StandardConv(stride = 2)
        
        
        """(64, 24, 80)"""
        ds8 = self.add_ds_conv_64(add_ds4) # StandardConv(stride = 2) 34,64
        
        
        """(64, 24, 80)"""
        for s in range(len(self.stages[1])-1):
            ds8 = self.stages[1][s](ds8) #Asymmblock
        ds8 = self.stages[1][-1](ds8) #LGFI
        
        # """(67, 24, 80)"""
        # concat_ds8 = torch.cat([ds8, x_down8], dim=1)
        
        """(64, 24, 80)"""
        add_ds8 = torch.add(ds8,x_down8)
        
        # """(128, 12, 40)"""
        # ds16 = self.downsample_layer3(concat_ds8) # StandardConv(stride = 2)
        
        """(128, 12, 40)"""
        ds16 = self.add_ds_conv_128(add_ds8) # StandardConv(stride = 2)
        
        for s in range(len(self.stages[2]) - 1):
            ds16 = self.stages[2][s](ds16) #Asymmblock
        ds16 = self.stages[2][-1](ds16) #LGFI

        
        return ds4, ds8, ds16
