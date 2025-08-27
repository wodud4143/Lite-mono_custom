from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from layers import *
from timm.layers import trunc_normal_


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super().__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'bilinear'
        self.scales = scales

        self.num_ch_enc = num_ch_enc # [32, 64, 128]
        self.num_ch_dec = (self.num_ch_enc / 2).astype('int') # [16, 32, 64]

        # decoder
        self.convs = OrderedDict()
        for i in range(2, -1, -1):
        # for i in range(3, -1, -1):

            num_ch_in = self.num_ch_enc[-1] if i == 2 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)


            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += (self.num_ch_enc[i - 1])
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def forward(self, input_features):
        # self.outputs = {}
        # stage2_ds4, stage3_ds8_10, stage4_ds16_10 = input_features
        ds4, ds8_core2, ds16_core = input_features
        
        upstage2 = self.convs[("upconv", 2, 0)](ds16_core) # (64, 12, 40)
        upstage2 = F.interpolate(upstage2, scale_factor=2, mode='bilinear') # (64, 24, 80)
        upstage2 = torch.cat([upstage2, ds8_core2], dim=1) # (128, 24, 80)
        upstage2 = self.convs[("upconv", 2, 1)](upstage2) # (64, 24, 80)
        upstage2_fin = self.convs[("dispconv", 2)](upstage2) # (1, 24, 80)
        upstage2_fin = F.interpolate(upstage2_fin, scale_factor=2, mode='bilinear')
        upstage2_fin = nn.Sigmoid()(upstage2_fin) # (1, 48, 160)
        
        upstage1 = self.convs[("upconv", 1, 0)](upstage2) # (32, 24, 80)
        upstage1 = F.interpolate(upstage1, scale_factor=2, mode='bilinear') # (32, 48, 160)
        upstage1 = torch.cat([upstage1, ds4], dim=1) # (64, 48, 160)
        upstage1 = self.convs[("upconv", 1, 1)](upstage1) # (32, 48, 160)
        upstage1_fin = self.convs[("dispconv", 1)](upstage1) # (1, 48, 160)
        upstage1_fin = F.interpolate(upstage1_fin, scale_factor=2, mode='bilinear')
        upstage1_fin = nn.Sigmoid()(upstage1_fin) # (1, 96, 320)
        
        upstage0 = self.convs[("upconv", 0, 0)](upstage1) # (16, 48, 160)
        upstage0 = F.interpolate(upstage0, scale_factor=2, mode='bilinear') # (16, 96, 320)
        upstage0 = self.convs[("upconv", 0, 1)](upstage0) # (16, 96, 320)
        upstage0_fin = self.convs[("dispconv", 0)](upstage0) # (1, 96, 320)
        upstage0_fin = F.interpolate(upstage0_fin, scale_factor=2, mode='bilinear')
        upstage0_fin = nn.Sigmoid()(upstage0_fin) # (1, 192, 640) 
        
    
        outputs = {
            ('disp', 2): upstage2_fin,
            ('disp', 1): upstage1_fin,
            ('disp', 0): upstage0_fin,
        }
        
        return outputs