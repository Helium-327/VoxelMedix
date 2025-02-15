# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/01/17 13:25:38
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 
*      VERSION: v1.0
*      FEATURES: None
=================================================
'''

import torch
import torch.nn as nn
import torch
import torch.nn as nn

class CAM(nn.Module):
    def __init__(self, channels):
        super(CAM, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.maxpool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(channels, channels // 16, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // 16, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.fc(self.avgpool(x))
        max_out = self.fc(self.maxpool(x))
        return x * (avg_out + max_out)

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, reduction_ratio):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction_ratio, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        return self.mlp(x)

class CBR_Block_3x3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CBR_Block_3x3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class FusionMagic_v2(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.2):
        super(FusionMagic_v2, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.maxpool = nn.AdaptiveMaxPool3d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.upsample_1 = nn.Upsample(scale_factor=4, mode='trilinear')
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='trilinear')
        self.cam_128 = CAM(in_channels * 4)
        self.cam_64 = CAM(in_channels * 2)
        self.cam_32 = CAM(in_channels)
        self.cam_out = CAM(in_channels)
        self.cam_192 = CAM(in_channels * 6)
        self.mlp_192 = MLP(in_channels=in_channels * 6, out_channels=in_channels * 4, reduction_ratio=in_channels * 6 // 2)
        self.layer_norm1 = nn.LayerNorm(normalized_shape=(in_channels * 4, 1, 1, 1))
        self.mlp_96 = MLP(in_channels=in_channels * 3, out_channels=in_channels * 2, reduction_ratio=in_channels * 3 // 2)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=(in_channels * 2, 1, 1, 1))
        self.cbr = nn.Sequential(
            CBR_Block_3x3(in_channels=in_channels * 6, out_channels=in_channels * 6),
            CBR_Block_3x3(in_channels=in_channels * 6, out_channels=in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: list[torch.Tensor], pooling_type='all'):
        if pooling_type == 'all':
            x1 = self.cam_128(inputs[0])
            x2 = self.cam_64(inputs[1])
            x3 = self.cam_32(inputs[2])
        elif pooling_type == 'max':
            x1 = self.maxpool(inputs[0])
            x2 = self.maxpool(inputs[1])
            x3 = self.maxpool(inputs[2])
        elif pooling_type == 'avg':
            x1 = self.avgpool(inputs[0])
            x2 = self.avgpool(inputs[1])
            x3 = self.avgpool(inputs[2])
        else:
            raise ValueError("Invalid pooling type")

        out1 = torch.cat([x1, x2], dim=1)
        out1 = self.mlp_192(out1)
        out1 = self.layer_norm1(out1)
        out1 = self.sigmoid(out1)
        out1 = out1.expand_as(inputs[0]) * inputs[0]

        out2 = torch.cat([x2, x3], dim=1)
        out2 = self.mlp_96(out2)
        out2 = self.layer_norm2(out2)
        out2 = self.sigmoid(out2)
        out2 = out2.expand_as(inputs[1]) * inputs[1]

        out1 = self.upsample_1(out1)
        out2 = self.upsample_2(out2)
        out = torch.cat([out1, out2], dim=1)
        out = self.cbr(out)
        out = self.cam_out(out)
        out = out.expand_as(inputs[2]) * inputs[2]
        return out
    
class CBR_Block_3x3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CBR_Block_3x3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // reduction_ratio, channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv3d(channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        channel_attention = self.channel_attention(x)
        spatial_attention = self.spatial_attention(x)
        return x * channel_attention * spatial_attention

class DownSample(nn.Module):
    def __init__(self):
        super(DownSample, self).__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(x)

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, trilinear=False):
        super(UpSample, self).__init__()
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.up(x)

class F_CAC_UNET3D(nn.Module):
    def __init__(self, in_channels, out_channels, features=[32, 64, 128, 256]):
        super(F_CAC_UNET3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.encoders_features = features
        self.decoders_features = (features + [features[-1] * 2])[::-1]

        self.encoders = self._make_encoders()
        self.bottom_layer = nn.Sequential(
            CBR_Block_3x3(features[-1], features[-1] * 2),
            CBAM(features[-1] * 2),
            CBR_Block_3x3(features[-1] * 2, features[-1] * 2)
        )
        self.decoders = self._make_decoders()
        self.out_conv = nn.Conv3d(self.decoders_features[-1], self.out_channels, kernel_size=1)
        self.soft_max = nn.Softmax(dim=1)

    def _make_encoders(self):
        encoders = nn.ModuleList()
        for i in range(len(self.encoders_features)):
            if i == 0:
                encoders.append(CBR_Block_3x3(self.in_channels, self.encoders_features[i]))
            else:
                encoders.append(CBR_Block_3x3(self.encoders_features[i - 1], self.encoders_features[i]))
            encoders.append(CBAM(self.encoders_features[i]))
            encoders.append(CBR_Block_3x3(self.encoders_features[i], self.encoders_features[i]))
            encoders.append(DownSample())
        return encoders

    def _make_decoders(self):
        decoders = nn.ModuleList()
        for i in range(len(self.decoders_features)):
            if i == len(self.decoders_features) - 1:
                continue
            else:
                decoders.append(UpSample(self.decoders_features[i], self.decoders_features[i + 1], trilinear=False))
                decoders.append(CBR_Block_3x3(self.decoders_features[i], self.decoders_features[i + 1]))
                decoders.append(CBAM(self.decoders_features[i + 1]))
                decoders.append(CBR_Block_3x3(self.decoders_features[i + 1], self.decoders_features[i + 1]))
        return decoders

    def forward(self, x):
        skip_out = []
        out = x

        for m in self.encoders:
            if isinstance(m, DownSample):
                skip_out.append(out)
            out = m(out)

        out = self.bottom_layer(out)

        for m in self.decoders:
            if isinstance(m, UpSample):
                out = m(out)
                out = torch.cat([out, skip_out.pop()], dim=1)
            else:
                out = m(out)

        out = self.out_conv(out)
        out = self.soft_max(out)
        return out
    
# 实例化模型
if __name__=="__main__":
    model = F_CAC_UNET3D(in_channels=4, out_channels=4)

    from ptflops import get_model_complexity_info
    # 使用Ptflops计算参数量和FLOPs
    macs, params = get_model_complexity_info(model, (4,128, 128, 128), as_strings=True, print_per_layer_stat=True, verbose=True)

    print(f'Computational complexity: {macs}')
    print(f'Number of parameters: {params}')