from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from layers import *
from timm.models.layers import trunc_normal_


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super().__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'bilinear'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = (self.num_ch_enc / 2).astype('int')

        # decoder
        self.convs = OrderedDict()
        # for i in range(2, -1, -1):
        for i in range(3, -1, -1):
            # upconv_0
            # num_ch_in = self.num_ch_enc[-1] if i == 2 else self.num_ch_dec[i + 1] 무슨 역할인지 모르겠음 
            num_ch_in = self.num_ch_enc[-1] if i == 3 else self.num_ch_dec[i + 1]
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
                
    #nn모듈에서 forward가 기본 실행 함수? 
    def forward(self, input_features):
        self.outputs = {}
        x = input_features[-1]
        # for i in range(2, -1, -1):
        for i in range(3, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]

            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)

            if i in self.scales:
                f = upsample(self.convs[("dispconv", i)](x), mode='bilinear')
                self.outputs[("disp", i)] = self.sigmoid(f)
                
       

        # disp3 = self.outputs[('disp', 3)]
        # disp2 = self.outputs[('disp', 2)]
        # disp1 = self.outputs[('disp', 1)]
        # disp0 = self.outputs[('disp', 0)]
        
        # import matplotlib.pyplot as plt
        # disp0_sample = disp0[0]
        # disp1_sample = disp1[0]
        # disp2_sample = disp2[0]
        # disp3_sample = disp3[0]

        # disp0_sample = disp0_sample.permute(1, 2, 0)
        # disp1_sample = disp1_sample.permute(1, 2, 0)
        # disp2_sample = disp2_sample.permute(1, 2, 0)
        # disp3_sample = disp3_sample.permute(1, 2, 0)

        # disp0_sample = disp0_sample.detach().cpu().numpy()
        # disp1_sample = disp1_sample.detach().cpu().numpy()
        # disp2_sample = disp2_sample.detach().cpu().numpy()
        # disp3_sample = disp3_sample.detach().cpu().numpy()
        
        # plt.imshow(disp0_sample)
        # plt.savefig('./disp0_sample.png')
        # plt.imshow(disp1_sample)
        # plt.savefig('./disp1_sample.png')
        # plt.imshow(disp2_sample)
        # plt.savefig('./disp2_sample.png')
        # plt.imshow(disp3_sample)
        # plt.savefig('./disp3_sample.png')
        
        return self.outputs

