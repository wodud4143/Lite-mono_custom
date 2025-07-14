from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from layers import *
from timm.models.layers import trunc_normal_



class MultiQueryAttentionLayerV2(nn.Module):
    """Multi Query Attention in PyTorch."""
    
    def __init__(self, num_heads, key_dim, value_dim, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.dropout = dropout
        
        self.query_proj = nn.Parameter(torch.randn(num_heads, key_dim, key_dim))
        self.key_proj = nn.Parameter(torch.randn(key_dim, key_dim))
        self.value_proj = nn.Parameter(torch.randn(key_dim, value_dim))
        self.output_proj = nn.Parameter(torch.randn(key_dim, num_heads, value_dim))
        
        self.dropout_layer = nn.Dropout(p=dropout)
    
    def _reshape_input(self, t):
        """Reshapes a tensor to three dimensions, keeping the first and last."""
        batch_size, *spatial_dims, channels = t.shape
        num = torch.prod(torch.tensor(spatial_dims))
        return t.view(batch_size, num, channels)
    
    def forward(self, x, m):
        """Run layer computation."""
        reshaped_x = self._reshape_input(x)
        reshaped_m = self._reshape_input(m)

        q = torch.einsum('bnd,hkd->bnhk', reshaped_x, self.query_proj)
        k = torch.einsum('bmd,dk->bmk', reshaped_m, self.key_proj)

        logits = torch.einsum('bnhk,bmk->bnhm', q, k)

        logits = logits / (self.key_dim ** 0.5)
        attention_scores = self.dropout_layer(F.softmax(logits, dim=-1))

        v = torch.einsum('bmd,dv->bmv', reshaped_m, self.value_proj)
        o = torch.einsum('bnhm,bmv->bnhv', attention_scores, v)
        result = torch.einsum('bnhv,dhv->bnd', o, self.output_proj)

        return result.view_as(x)
    
    
    
class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super().__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'bilinear'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = (self.num_ch_enc / 2).astype('int')

        num_heads = 8
        self.dim = 128
        self.mqa_layer = MultiQueryAttentionLayerV2(num_heads=num_heads, key_dim=self.dim, value_dim=self.dim ,dropout=0.1)
        self.mqa_layer = self.mqa_layer.cuda()
        
        # decoder
        self.convs = OrderedDict()
        # for i in range(2, -1, -1):
        for i in range(2, -1, -1):
            # upconv_0
            # num_ch_in = self.num_ch_enc[-1] if i == 2 else self.num_ch_dec[i + 1] 무슨 역할인지 모르겠음 
            num_ch_in = self.num_ch_enc[-1] if i == 2 else self.num_ch_dec[i + 1]
            # self.num_ch_enc: np.array([48, 80, 128])
            # self.num_ch_enc: np.array([48, 80, 128, 168])
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)
            # print(i, num_ch_in, num_ch_out)
            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
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
        self.outputs = {}
        x = input_features[-1]
        
        B, C, H, W = x.shape
        reshaped_x = x.reshape(B, C, H*W).permute(0, 2, 1)
        mqa_x = self.mqa_layer(reshaped_x, reshaped_x)
        x = mqa_x.reshape(B, C, H, W)
        
        for i in range(2, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]

            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            
            if i in self.scales:
                f = upsample(self.convs[("dispconv", i)](x), mode='bilinear')
                self.outputs[("disp", i)] = self.sigmoid(f)
                
    
        
        return self.outputs

