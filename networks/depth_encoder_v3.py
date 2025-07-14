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

        # if model == 'lite-mono':
        # self.num_ch_enc = np.array([48, 80, 128])
        self.num_ch_enc = np.array([32, 64, 128])
        self.depth = [4, 4, 10]
        # self.dims = [48, 80, 128]
        self.dims = [32, 64, 128]
        self.asym_dims = [64, 96, 128]

        if height == 192 and width == 640:
            self.dilation = [[1, 2], [1, 3], [1, 4, 6]]
            # self.dilation = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 1, 2, 3, 2, 4, 6]]
            # self.dilation = [[1, 1, 2], [1, 1, 2], [1, 1, 2, 1, 1, 2, 1, 2, 3]]
        elif height == 320 and width == 1024:
            self.dilation = [[1, 2, 5], [1, 2, 5], [1, 2, 5, 1, 2, 5, 2, 4, 10]]

        for g in global_block_type:
            assert g in ['None', 'LGFI']


        self.avg_pool2 = clayers.AvgPool(ratio=2)
        self.avg_pool4 = clayers.AvgPool(ratio=4)
        self.avg_pool8 = clayers.AvgPool(ratio=8)

        
        self.init_conv = nn.Sequential(
            clayers.StandardConv(in_chans, self.dims[0],
                              kernel_size=3, 
                              stride=2,
                              padding=1, 
                              bn_act=True)
        )
        self.ds_conv1 = clayers.StandardConv(self.dims[0]+3, self.dims[0],
                                          kernel_size=3,
                                          stride=2, 
                                          padding=1, 
                                          bn_act=True)
        self.cghost_layer = core.CustomGhostModule(self.dims[0], self.dims[0]//2)

        self.ds_conv2 = clayers.StandardConv(self.dims[0], self.dims[1],
                                          kernel_size=3,
                                          stride=2, 
                                          padding=1, 
                                          bn_act=True)
        self.cghost2_layer = core.CustomGhostModule(self.dims[1], self.dims[1]//2)

        
        self.downsample_layer2 = nn.Sequential(
            clayers.StandardConv(self.dims[0]+3, self.dims[1], 
                                 kernel_size=3, 
                                 stride=2,
                                 padding=1, 
                                 bn_act=False)
        )
        
        # self.exp_conv = clayers.StandardConv(self.dims[1], self.dims[2],
        #                                     kernel_size=1, stride=1, padding=0, bn_act=False)
        
        self.downsample_layer3 = nn.Sequential(
            clayers.StandardConv(self.dims[1]*2+3, self.dims[2], 
                                 kernel_size=3, 
                                 stride=2, 
                                 padding=1, 
                                 bn_act=False)
        )
        
        
        
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depth))]

        stage_blocks = [
            core.AsymDilatedConv(inc=self.dims[1], outc=self.asym_dims[0], dilation=self.dilation[0][0]),
            core.AsymDilatedConv(inc=self.dims[1], outc=self.asym_dims[0], dilation=self.dilation[0][1]),
            core.LGFI(dim=self.dims[1], 
                      drop_path=dp_rates[0 + 2], 
                      expan_ratio=expan_ratio,
                      use_pos_emb=use_pos_embd_xca[0], 
                      num_heads=heads[0], 
                      layer_scale_init_value=layer_scale_init_value)
        ]
        self.stages.append(nn.Sequential(*stage_blocks))
        
        stage_blocks = [
            core.AsymDilatedConv(inc=self.dims[1], outc=self.asym_dims[1], dilation=self.dilation[1][0]),
            core.AsymDilatedConv(inc=self.dims[1], outc=self.asym_dims[1], dilation=self.dilation[1][1]),
            core.LGFI(dim=self.dims[1], 
                      drop_path=dp_rates[3 + 2], 
                      expan_ratio=expan_ratio,
                      use_pos_emb=use_pos_embd_xca[1], 
                      num_heads=heads[1], 
                      layer_scale_init_value=layer_scale_init_value)
        ]
        self.stages.append(nn.Sequential(*stage_blocks))
        
        stage_blocks = [
            core.AsymDilatedConv(inc=self.dims[2], outc=self.asym_dims[2], dilation=self.dilation[2][0]),
            core.AsymDilatedConv(inc=self.dims[2], outc=self.asym_dims[2], dilation=self.dilation[2][1]),
            core.AsymDilatedConv(inc=self.dims[2], outc=self.asym_dims[2], dilation=self.dilation[2][2]),
            core.LGFI(dim=self.dims[2], 
                      drop_path=dp_rates[3 + 3], 
                      expan_ratio=expan_ratio,
                      use_pos_emb=use_pos_embd_xca[2], 
                      num_heads=heads[2], 
                      layer_scale_init_value=layer_scale_init_value)
        ]
        self.stages.append(nn.Sequential(*stage_blocks))
        
        # cur = 0
        # for i in range(3):
        #     stage_blocks = []
        #     for j in range(self.depth[i]):
                
        #         if j > self.depth[i] - global_block[i] - 1:
        #             if global_block_type[i] == 'LGFI':
        #                 print('LGFI')
        #                 stage_blocks.append(core.LGFI(dim=self.dims[i], drop_path=dp_rates[cur + j],
        #                                          expan_ratio=expan_ratio,
        #                                          use_pos_emb=use_pos_embd_xca[i], num_heads=heads[i],
        #                                          layer_scale_init_value=layer_scale_init_value,
        #                                          ))
        #             else:
        #                 raise NotImplementedError
                
        #         else:
        #             print('asym_dc')
        #             stage_blocks.append(
        #                 core.AsymDilatedConv(inc=self.dims[i], 
        #                                     outc=self.asym_dims[i],
        #                                     dilation=self.dilation[i][j]))
                    
        #             # print('CDC')
        #             # stage_blocks.append(core.DilatedConv(dim=self.dims[i]//2, k=3, 
        #             #                                      dilation=self.dilation[i][j], 
        #             #                                      drop_path=dp_rates[cur + j],
        #             #                                      layer_scale_init_value=layer_scale_init_value,
        #             #                                      expan_ratio=expan_ratio))
            
        #     print(' ')
        #     self.stages.append(nn.Sequential(*stage_blocks))
        #     cur += self.depth[i]

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
        
        x_down2 = self.avg_pool2(x)
        x_down4 = self.avg_pool4(x)
        x_down8 = self.avg_pool8(x)
        
        """(32, 96, 320)"""
        ds2 = self.init_conv(x)
        """(35, 96, 320)"""
        ds2 = torch.cat((ds2, x_down2), dim=1)

        """(32, 48, 160)"""
        ds4 = self.ds_conv1(ds2)
        # ds4 = self.dw_conv(ds2)
        ds4 = self.cghost_layer(ds4)

        """(64, 24, 80)"""
        ds8 = self.ds_conv2(ds4)
        ds8_core = self.cghost2_layer(ds8)

        for s in range(len(self.stages[0])-1):
            ds8_core = self.stages[0][s](ds8_core)
        ds8_core = self.stages[0][-1](ds8_core)
        
        """(35, 48, 160)"""
        concat_ds4 = torch.cat([ds4, x_down4], dim=1)
        """(64, 24, 80)"""
        ds8_core2 = self.downsample_layer2(concat_ds4)
        ds8_core2 = torch.add(ds8_core, ds8_core2)
        # """(96, 24, 80)"""
        # ds8_core2 = self.exp_conv(ds8_2)
        
        """(64, 24, 80)"""
        for s in range(len(self.stages[1])-1):
            ds8_core2 = self.stages[1][s](ds8_core2)
        ds8_core2 = self.stages[1][-1](ds8_core2)
        
        """(131, 24, 80)"""
        concat_ds8 = torch.cat([ds8_core, ds8_core2, x_down8], dim=1)
        """(128, 12, 40)"""
        ds16_core = self.downsample_layer3(concat_ds8)
        
        for s in range(len(self.stages[2]) - 1):
            ds16_core = self.stages[2][s](ds16_core)
        ds16_core = self.stages[2][-1](ds16_core)
        
        return ds4, ds8_core2, ds16_core
