from __future__ import absolute_import, division, print_function
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *
from timm.layers import trunc_normal_


# ========== 추가 모듈들 ==========
class AttentionSkipConnection(nn.Module):
    """Channel & Spatial Attention for Skip Connections"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        # Channel Attention
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        
        # Spatial Attention (얇은 구조 강조)
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.BatchNorm2d(channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Channel attention
        ch_att = self.channel_att(x)
        x_ch = x * ch_att
        
        # Spatial attention
        sp_att = self.spatial_att(x_ch)
        x_out = x_ch * sp_att
        
        return x_out


class EdgeAwareModule(nn.Module):
    """Extract and preserve edge information for thin structures"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.main_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Edge detection branch (vertical edges for poles)
        self.edge_conv = nn.Conv2d(in_channels, out_channels, 3, 
                                    padding=1, bias=False)
        
        # Initialize with edge detection weights
        self._init_edge_weights()
        
    def _init_edge_weights(self):
        with torch.no_grad():
            # Vertical edge kernel (Sobel-like)
            kernel = torch.tensor([[-1, 0, 1],
                                  [-2, 0, 2],
                                  [-1, 0, 1]], dtype=torch.float32)
            kernel = kernel.view(1, 1, 3, 3)
            # Repeat for all input-output channel combinations
            kernel = kernel.repeat(self.edge_conv.out_channels, 
                                  self.edge_conv.in_channels, 1, 1)
            self.edge_conv.weight = nn.Parameter(kernel / (self.edge_conv.in_channels))
        self.edge_conv.weight.requires_grad = True
        
    def forward(self, x):
        main_feat = self.main_conv(x)
        edge_feat = self.edge_conv(x)
        # Combine main features with edge-enhanced features
        return main_feat + 0.3 * edge_feat


class SpatialPositionAware(nn.Module):
    """Position-aware module for handling different image regions"""
    def __init__(self, channels):
        super().__init__()
        
        # Region-aware weighting
        self.region_fc = nn.Sequential(
            nn.Conv2d(channels + 2, channels, 1),  # +2 for y,x coordinates
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Create normalized coordinate grids
        y_coords = torch.linspace(-1, 1, H, device=x.device)
        y_coords = y_coords.view(1, 1, H, 1).expand(B, 1, H, W)
        
        x_coords = torch.linspace(-1, 1, W, device=x.device)
        x_coords = x_coords.view(1, 1, 1, W).expand(B, 1, H, W)
        
        # Combine with features
        x_with_pos = torch.cat([x, y_coords, x_coords], dim=1)
        
        # Generate position-aware weights
        pos_weights = self.region_fc(x_with_pos)
        
        # Apply adaptive weighting (bottom region gets more weight)
        return x * (1.0 + pos_weights)


# ========== 메인 Decoder ==========
class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True,
                 use_attention=True, use_edge_aware=True, use_spatial_aware=True):
        """
        Enhanced Depth Decoder with toggleable modules
        
        Args:
            num_ch_enc: Encoder channel sizes [32, 64, 128]
            scales: Output scales
            num_output_channels: Number of output channels
            use_skips: Use skip connections
            use_attention: Toggle attention on skip connections (얇은 구조 강조)
            use_edge_aware: Toggle edge-aware processing (수직 구조 보존)
            use_spatial_aware: Toggle spatial position awareness (하단 영역/가까운 객체 강화)
        """
        super().__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.use_attention = use_attention
        self.use_edge_aware = use_edge_aware
        self.use_spatial_aware = use_spatial_aware
        self.upsample_mode = 'bilinear'
        self.scales = scales

        self.num_ch_enc = num_ch_enc # [32, 64, 128]
        self.num_ch_dec = (self.num_ch_enc / 2).astype('int') # [16, 32, 64]

        # decoder
        self.convs = OrderedDict()
        
        # Attention modules (얇은 구조 강조)
        if self.use_attention:
            self.skip_attentions = nn.ModuleDict()
        
        # Edge-aware modules (수직 구조 보존)
        if self.use_edge_aware:
            self.edge_modules = nn.ModuleDict()
        
        # Spatial position-aware modules (하단 영역 강화)
        if self.use_spatial_aware:
            self.spatial_modules = nn.ModuleDict()
        
        for i in range(2, -1, -1):
            num_ch_in = self.num_ch_enc[-1] if i == 2 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # Attention for skip connections
            if self.use_attention and self.use_skips and i > 0:
                self.skip_attentions[f"att_{i}"] = AttentionSkipConnection(
                    self.num_ch_enc[i - 1], reduction=8
                )
            
            # Edge-aware processing
            if self.use_edge_aware and i < 2:  # Apply to finer scales (stage 0, 1)
                self.edge_modules[f"edge_{i}"] = EdgeAwareModule(
                    self.num_ch_dec[i], self.num_ch_dec[i]
                )

            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += (self.num_ch_enc[i - 1])
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)
            
            # Spatial position-aware
            if self.use_spatial_aware:
                self.spatial_modules[f"spatial_{i}"] = SpatialPositionAware(num_ch_out)

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
        ds4, ds8_core2, ds16_core = input_features
        
        # ========== Stage 2 (1/8 resolution) ==========
        upstage2 = self.convs[("upconv", 2, 0)](ds16_core) # (64, 12, 40)
        upstage2 = F.interpolate(upstage2, scale_factor=2, mode='bilinear') # (64, 24, 80)
        
        # Skip connection with optional attention
        if self.use_attention and self.use_skips:
            ds8_processed = self.skip_attentions["att_2"](ds8_core2)
            upstage2 = torch.cat([upstage2, ds8_processed], dim=1) # (128, 24, 80)
        else:
            upstage2 = torch.cat([upstage2, ds8_core2], dim=1) # (128, 24, 80)
        
        upstage2 = self.convs[("upconv", 2, 1)](upstage2) # (64, 24, 80)
        
        # Spatial awareness (하단 영역 강화)
        if self.use_spatial_aware:
            upstage2 = self.spatial_modules["spatial_2"](upstage2)
        
        upstage2_fin = self.convs[("dispconv", 2)](upstage2) # (1, 24, 80)
        upstage2_fin = F.interpolate(upstage2_fin, scale_factor=2, mode='bilinear')
        upstage2_fin = nn.Sigmoid()(upstage2_fin) # (1, 48, 160)
        
        # ========== Stage 1 (1/4 resolution) ==========
        upstage1 = self.convs[("upconv", 1, 0)](upstage2) # (32, 24, 80)
        
        # Edge-aware processing (수직 구조 보존)
        if self.use_edge_aware:
            upstage1 = self.edge_modules["edge_1"](upstage1)
        
        upstage1 = F.interpolate(upstage1, scale_factor=2, mode='bilinear') # (32, 48, 160)
        
        # Skip connection with optional attention
        if self.use_attention and self.use_skips:
            ds4_processed = self.skip_attentions["att_1"](ds4)
            upstage1 = torch.cat([upstage1, ds4_processed], dim=1) # (64, 48, 160)
        else:
            upstage1 = torch.cat([upstage1, ds4], dim=1) # (64, 48, 160)
        
        upstage1 = self.convs[("upconv", 1, 1)](upstage1) # (32, 48, 160)
        
        # Spatial awareness
        if self.use_spatial_aware:
            upstage1 = self.spatial_modules["spatial_1"](upstage1)
        
        upstage1_fin = self.convs[("dispconv", 1)](upstage1) # (1, 48, 160)
        upstage1_fin = F.interpolate(upstage1_fin, scale_factor=2, mode='bilinear')
        upstage1_fin = nn.Sigmoid()(upstage1_fin) # (1, 96, 320)
        
        # ========== Stage 0 (1/2 resolution) ==========
        upstage0 = self.convs[("upconv", 0, 0)](upstage1) # (16, 48, 160)
        
        # Edge-aware processing (최고 해상도에서 얇은 구조 강화)
        if self.use_edge_aware:
            upstage0 = self.edge_modules["edge_0"](upstage0)
        
        upstage0 = F.interpolate(upstage0, scale_factor=2, mode='bilinear') # (16, 96, 320)
        upstage0 = self.convs[("upconv", 0, 1)](upstage0) # (16, 96, 320)
        
        # Spatial awareness (가까운 객체 강화)
        if self.use_spatial_aware:
            upstage0 = self.spatial_modules["spatial_0"](upstage0)
        
        upstage0_fin = self.convs[("dispconv", 0)](upstage0) # (1, 96, 320)
        upstage0_fin = F.interpolate(upstage0_fin, scale_factor=2, mode='bilinear')
        upstage0_fin = nn.Sigmoid()(upstage0_fin) # (1, 192, 640) 
        
    
        outputs = {
            ('disp', 2): upstage2_fin,
            ('disp', 1): upstage1_fin,
            ('disp', 0): upstage0_fin,
        }
        
        return outputs