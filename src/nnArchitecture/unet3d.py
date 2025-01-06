# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/01/06 16:25:05
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 3D U-Net Model
*      VERSION: v1.0
=================================================
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchinfo import summary
import torch

def print_model_summary(model, input_data, device="cuda"):

    # 使用 torchinfo 生成模型摘要
    # batch_size=1, 输入尺寸为 (128, 128, 128)，通道数为 4
    summary(
        model,
        input_data=input_data,
        col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
        col_width=20,
        row_settings=["var_names"],
        device=device
    )

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, channels_list=None):
        """
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            channels_list (list): List of channel numbers for each level [ch1, ch2, ch3, ch4]
                                Default is [64, 128, 256, 512]
        """
        super(UNet3D, self).__init__()
        
        # Default channel configuration if not provided
        if channels_list is None:
            channels_list = [32, 64, 128, 256]
        
        ch1, ch2, ch3, ch4 = channels_list

        # Downsample path
        self.down1 = ResCBRBlock(in_channels, ch1)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.down2 = ResCBRBlock(ch1, ch2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.down3 = ResCBRBlock(ch2, ch3)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = ResCBRBlock(ch3, ch4)

        # Upsample path
        self.up1 = nn.ConvTranspose3d(ch4, ch3, kernel_size=2, stride=2)
        self.up_conv1 = ResCBRBlock(ch3 * 2, ch3)
        self.up2 = nn.ConvTranspose3d(ch3, ch2, kernel_size=2, stride=2)
        self.up_conv2 = ResCBRBlock(ch2 * 2, ch2)
        self.up3 = nn.ConvTranspose3d(ch2, ch1, kernel_size=2, stride=2)
        self.up_conv3 = ResCBRBlock(ch1 * 2, ch1)

        # Final layer
        self.final_conv = nn.Conv3d(ch1, out_channels, kernel_size=1)
        
        self.softmax = nn.Softmax(dim=1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Downsample
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        d4 = self.pool3(d3)

        # Bottleneck
        bottleneck = self.bottleneck(d4)

        # Upsample
        u1 = self.up1(bottleneck)
        u1 = torch.cat([u1, d3], dim=1)
        u1 = self.up_conv1(u1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.up_conv2(u2)

        u3 = self.up3(u2)
        u3 = torch.cat([u3, d1], dim=1)
        u3 = self.up_conv3(u3)

        # Final layer
        out = self.final_conv(u3)
        
        # gain the probability
        out = self.softmax(out)
        return out

class DownSample(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_sample = nn.MaxPool3d(kernel_size=4, stride=2, padding=1)
    def forward(self, x):
        return self.down_sample(x)

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, trilinear=False):
        super().__init__()
        if trilinear:
            self.up_sample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear'),
                nn.Conv3d(in_channels, out_channels, kernel_size=1)
                )
        else:
            self.up_sample = nn.Sequential(
                nn.ConvTranspose3d(in_channels=in_channels, out_channels=in_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(in_channels),
                nn.ReLU(),
                nn.Conv3d(in_channels, out_channels, kernel_size=1)
                )
        
    def forward(self, x):
        return self.up_sample(x)

# class UNet3D(nn.Module):
#     """原UNET3D网络结构:
#     features=[32, 64, 128, 256] 时最合适，
#     当features=[64, 128, 256]时，占用显存会比较大，
#     当features=[16, 32, 64, 128, 256]时，预测可视化效果不好
#     """
#     def __init__(self, 
#                  in_channels, 
#                  out_channels, 
#                  features=[32, 64, 128, 256],
#                  down_att =False,
#                  up_att =False,
#                  bottom_att = False):
        
#         super(UNet3D, self).__init__()
#         self.in_channels  = in_channels
#         self.out_channels = out_channels
#         self.down_att    = down_att
#         self.up_att      = up_att
#         self.bottom_att  = bottom_att
#         self.encoders_features  = features
#         self.decoders_features  = (features + [features[-1]*2])[::-1]

#         self.encoders     = nn.ModuleList()
#         self.decoders     = nn.ModuleList()
#         self._make_encoders()

#         self.bottom_layer = nn.Sequential(
#             CBR_Block_3x3(features[-1], features[-1]*2),
#             CBR_Block_3x3(features[-1]*2, features[-1]*2)
#         )
#         self._make_decoders()
#         self.out_conv = nn.Conv3d(self.decoders_features[-1], self.out_channels, kernel_size=1)
#         self.soft_max = nn.Softmax(dim=1)
#         # self.crf      = CRF(out_channels)

#     def _make_encoders(self):
#         for i in range(len(self.encoders_features)):
#             if i == 0:
#                 self.encoders.append(CBR_Block_3x3(self.in_channels, self.encoders_features[i]))
#                 self.encoders.append(CBR_Block_3x3(self.encoders_features[i], self.encoders_features[i]))
#             else:
#                 self.encoders.append(CBR_Block_3x3(self.encoders_features[i-1], self.encoders_features[i]))
#                 self.encoders.append(CBR_Block_3x3(self.encoders_features[i], self.encoders_features[i]))
#             self.encoders.append(DownSample())
            
#     def _make_decoders(self):
#         for i in range(len(self.decoders_features)):
#             if i == len(self.decoders_features)-1:
#                 continue
#             else:
#                 self.decoders.append(UpSample(self.decoders_features[i], self.decoders_features[i+1], trilinear=False))
#                 self.decoders.append(CBR_Block_3x3(self.decoders_features[i], self.decoders_features[i+1]))
#                 self.decoders.append(CBR_Block_3x3(self.decoders_features[i+1], self.decoders_features[i+1]))
            
#     def forward(self, x):
#         out = x
#         # print(f'Input shape: {out.shape}')
#         skip_out = []
#         for m in self.encoders:
#             if isinstance(m, DownSample):
#                 skip_out.append(out)
#                 # print(f"skip_out: {out.shape}")
#             out = m(out)
#             # print(f'Encoder shape: {out.shape}')
#             # print("-" * 50)

#         # for t in skip_out:
#             # print(f'Skip connection shape: {t.shape}')

#         out = self.bottom_layer(out)

#         for m in self.decoders:
#             if isinstance(m, UpSample):
#                 out = m(out)
#                 # print(f'up shape : {out.shape}')
#                 # print(f'skip shape : {skip_out[-1].shape}')
#                 out = torch.cat([out, skip_out.pop()], dim=1)
#                 # print(f'after cat shape : {out.shape}')
#                 # print("-" * 50)
#             else:
#                 out = m(out)
#                 # print(f'Decoder shape: {out.shape}')
#         out = self.out_conv(out)
#         out = self.soft_max(out)
#         # out = self.crf
#         return out


# Example usage
if __name__ == '__main__':
    from modules.BasicBlock import *
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_data = torch.randn(1, 4, 128, 128, 128).to(device)
    # model = UNet3D(in_channels=4, out_channels=4, channels_list=[32, 64, 128, 256]).to(device)
    model = UNet3D(in_channels=4, out_channels=4).to(device)
    
    output = model(input_data)
    print(output.shape)
    # Print model summary
    print_model_summary(model, input_data=input_data, device=device)
    
    
else:
    from .modules.BasicBlock import *