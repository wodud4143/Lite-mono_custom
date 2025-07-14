import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath
import math


# ============== Basic Building Blocks ==============

class LayerNorm(nn.Module):
    """Efficient LayerNorm with channels_first support"""
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # More efficient implementation
            mean = x.mean(dim=(2, 3), keepdim=True)
            var = ((x - mean) ** 2).mean(dim=(2, 3), keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            return self.weight[:, None, None] * x + self.bias[:, None, None]


class EfficientDilatedConv(nn.Module):
    """
    Highly optimized Dilated Convolution - minimal BatchNorm and operations
    """
    def __init__(self, dim, k=3, dilation=1, stride=1, drop_path=0.,
                 layer_scale_init_value=1e-6, expand_ratio=2):  # Further reduced
        super().__init__()
        
        # Fuse all operations into one sequential to reduce calls
        hidden_dim = int(dim * expand_ratio)
        self.conv_block = nn.Sequential(
            # Depthwise dilated without BN
            nn.Conv2d(dim, dim, k, stride, padding=dilation, 
                     dilation=dilation, groups=dim, bias=True),  # Add bias
            # Direct expansion + activation (no BN)
            nn.Conv2d(dim, hidden_dim, 1, bias=False),
            nn.GELU(approximate='tanh'),  # Faster GELU
            # Projection
            nn.Conv2d(hidden_dim, dim, 1, bias=True)  # Use bias instead of BN
        )
        
        # Remove layer scale to reduce operations
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        return x + self.drop_path(self.conv_block(x))


class SimplifiedXCA(nn.Module):
    """
    Simplified Cross-Covariance Attention with reduced operations
    Fixed version with correct tensor reshaping
    """
    def __init__(self, dim, num_heads=4, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) * 0.5)  # Lower init
        
        # Single QKV projection
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # QKV projection and reshape
        qkv = self.qkv(x)  # B, 3*C, H, W
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W)  # B, 3, heads, head_dim, HW
        qkv = qkv.permute(1, 0, 2, 3, 4)  # 3, B, heads, head_dim, HW
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each is B, heads, head_dim, HW
        
        # Normalize
        q = F.normalize(q, dim=2)
        k = F.normalize(k, dim=2)
        
        # Attention - transpose for matrix multiplication
        q = q.transpose(-2, -1)  # B, heads, HW, head_dim
        k = k.transpose(-2, -1)  # B, heads, HW, head_dim
        
        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.temperature  # B, heads, HW, HW
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        v = v.transpose(-2, -1)  # B, heads, HW, head_dim
        x = attn @ v  # B, heads, HW, head_dim
        x = x.transpose(-2, -1).reshape(B, C, H, W)  # B, C, H, W
        
        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class EfficientLGFI(nn.Module):
    """
    Efficient Local-Global Feature Interaction
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=0, 
                 expand_ratio=2, use_pos_emb=False, num_heads=4):
        super().__init__()
        
        self.dim = dim
        
        # Simplified attention
        self.norm1 = LayerNorm(dim)
        self.attn = SimplifiedXCA(dim, num_heads=num_heads)
        
        # Efficient FFN
        hidden_dim = int(dim * expand_ratio)
        self.norm2 = LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, bias=True),
            nn.GELU(approximate='tanh'),
            nn.Conv2d(hidden_dim, dim, 1, bias=True)
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # Attention branch
        x = x + self.drop_path(self.attn(self.norm1(x)))
        
        # FFN branch
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        
        return x


# ============== Main Model ==============

class LiteMono(nn.Module):
    """
    Highly Efficient LiteMono - Optimized based on profiling data
    Reduces BatchNorm usage and operation calls
    """
    def __init__(self, in_chans=3, model='lite-mono', height=192, width=640,
                 global_block=[1, 1, 1], global_block_type=['LGFI', 'LGFI', 'LGFI'],
                 drop_path_rate=0.1, layer_scale_init_value=0,  # Disable layer scale
                 heads=[4, 4, 4], use_pos_embd_xca=[False, False, False], **kwargs):
        
        super().__init__()
        
        # Ultra-optimized configurations
        if model == 'lite-mono':
            self.num_ch_enc = np.array([48, 80, 128])  # Keep original for compatibility
            self.depth = [2, 3, 6]  # Significantly reduced depth
            self.dims = [48, 80, 128]
            self.expand_ratios = [2, 2, 2]  # Minimal expansion
            
        elif model == 'lite-mono-small':
            self.num_ch_enc = np.array([40, 72, 120])
            self.depth = [2, 3, 5]
            self.dims = [40, 72, 120]
            self.expand_ratios = [2, 2, 2]
            
        elif model == 'lite-mono-tiny':
            self.num_ch_enc = np.array([32, 64, 96])
            self.depth = [2, 2, 4]
            self.dims = [32, 64, 96]
            self.expand_ratios = [2, 2, 2]
            
        # Simplified dilation - reduce variety
        self.dilation = [[1, 2], [1, 2], [1, 2, 3, 1]]
            
        # Ultra-efficient stem without BatchNorm
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, self.dims[0], 4, 4, bias=True),  # Single stride-4 conv
            nn.GELU(approximate='tanh')
        )
        
        # Direct downsampling layers (no fusion to reduce ops)
        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(nn.Identity())  # Stem already downsamples
        
        for i in range(2):
            # Simple strided conv for downsampling
            in_ch = self.dims[i]
            out_ch = self.dims[i + 1]
            downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, 2, 1, bias=True),
                nn.GELU(approximate='tanh')
            )
            self.downsample_layers.append(downsample)
        
        # Build stages with minimal blocks
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depth))]
        cur = 0
        
        for i in range(3):
            stage_blocks = []
            expand_ratio = self.expand_ratios[i]
            
            for j in range(self.depth[i]):
                # Only use LGFI for very last block to minimize attention ops
                if j == self.depth[i] - 1 and i == 2:  # Only in last stage
                    block = EfficientLGFI(
                        dim=self.dims[i],
                        drop_path=dp_rates[cur + j],
                        expand_ratio=expand_ratio,
                        use_pos_emb=False,  # Never use pos emb
                        num_heads=heads[i],
                        layer_scale_init_value=0  # Disable
                    )
                else:
                    dilation_idx = j % len(self.dilation[i])
                    block = EfficientDilatedConv(
                        dim=self.dims[i],
                        k=3,
                        dilation=self.dilation[i][dilation_idx],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=0,  # Disable
                        expand_ratio=expand_ratio
                    )
                stage_blocks.append(block)
                
            self.stages.append(nn.Sequential(*stage_blocks))
            cur += self.depth[i]
        
        # Proper weight initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            # Kaiming initialization for Conv layers
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, LayerNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward_features(self, x):
        features = []
        
        # Normalize
        x = (x - 0.45) / 0.225
        
        # Stage 1 - 1/4 scale
        x = self.stem(x)
        x = self.stages[0](x)
        features.append(x)
        
        # Stage 2 - 1/8 scale
        x = self.downsample_layers[1](x)
        x = self.stages[1](x)
        features.append(x)
        
        # Stage 3 - 1/16 scale
        x = self.downsample_layers[2](x)
        x = self.stages[2](x)
        features.append(x)
        
        return features
    
    def forward(self, x):
        return self.forward_features(x)

