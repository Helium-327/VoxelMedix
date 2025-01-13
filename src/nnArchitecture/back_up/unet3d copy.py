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




"""---------------------------------------- UNet3D ----------------------------------------"""
#? 原UNet3D 模型
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
        self.down1 = DoubleCBR_Block_3x3(in_channels, ch1, use_dw=False)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.down2 = DoubleCBR_Block_3x3(ch1, ch2, use_dw=False)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.down3 = DoubleCBR_Block_3x3(ch2, ch3, use_dw=False)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = DoubleCBR_Block_3x3(ch3, ch4, use_dw=False)

        # Upsample path
        self.up1 = nn.ConvTranspose3d(ch4, ch3, kernel_size=2, stride=2)
        self.up_conv1 = DoubleCBR_Block_3x3(ch3 * 2, ch3, use_dw=False)
        self.up2 = nn.ConvTranspose3d(ch3, ch2, kernel_size=2, stride=2)
        self.up_conv2 = DoubleCBR_Block_3x3(ch2 * 2, ch2, use_dw=False)
        self.up3 = nn.ConvTranspose3d(ch2, ch1, kernel_size=2, stride=2)
        self.up_conv3 = DoubleCBR_Block_3x3(ch1 * 2, ch1, use_dw=False)

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
    
    
"""---------------------------------------- Res-UNet3D ----------------------------------------"""
class ResUNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, channels_list=None):
        """
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            channels_list (list): List of channel numbers for each level [ch1, ch2, ch3, ch4]
                                Default is [64, 128, 256, 512]
        """
        super(ResUNet3D, self).__init__()
        
        # Default channel configuration if not provided
        if channels_list is None:
            channels_list = [32, 64, 128, 256]
        
        ch1, ch2, ch3, ch4 = channels_list

        # Downsample path
        self.down1 = ResCBRBlock(in_channels, ch1, use_dw=False)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.down2 = ResCBRBlock(ch1, ch2, use_dw=False)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.down3 = ResCBRBlock(ch2, ch3, use_dw=False)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = ResCBRBlock(ch3, ch4, use_dw=False)

        # Upsample path
        self.up1 = nn.ConvTranspose3d(ch4, ch3, kernel_size=2, stride=2)
        self.up_conv1 = ResCBRBlock(ch3 * 2, ch3, use_dw=False)
        self.up2 = nn.ConvTranspose3d(ch3, ch2, kernel_size=2, stride=2)
        self.up_conv2 = ResCBRBlock(ch2 * 2, ch2, use_dw=False)
        self.up3 = nn.ConvTranspose3d(ch2, ch1, kernel_size=2, stride=2)
        self.up_conv3 = ResCBRBlock(ch1 * 2, ch1, use_dw=False)

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


"""---------------------------------------- DW-UNet3D ----------------------------------------"""
#? DW-UNet3D 将 UNet3D 中的 ResCBRBlock 替换为 RDABlock，并在下采样路径中添加了深度可分离卷积。

class DW_UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, channels_list=None):
        """
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            channels_list (list): List of channel numbers for each level [ch1, ch2, ch3, ch4]
                                Default is [32, 64, 128, 256]
        """
        super(DW_UNet3D, self).__init__()
        
        # Default channel configuration if not provided
        if channels_list is None:
            channels_list = [32, 64, 128, 256]
        
        ch1, ch2, ch3, ch4 = channels_list

        # Downsample path
        self.down1 = ResCBRBlock(in_channels, ch1, use_dw=False)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.down2 = RDABlock(ch1, ch2, use_dw=False)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.down3 = RDABlock(ch2, ch3, use_dw=False)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = RDABlock(ch3, ch4, use_dw=False)

        # Upsample path
        self.up1 = nn.ConvTranspose3d(ch4, ch3, kernel_size=2, stride=2)
        self.up_conv1 = RDABlock(ch3 * 2, ch3, use_dw=False)
        self.up2 = nn.ConvTranspose3d(ch3, ch2, kernel_size=2, stride=2)
        self.up_conv2 = RDABlock(ch2 * 2, ch2, use_dw=False)
        self.up3 = nn.ConvTranspose3d(ch2, ch1, kernel_size=2, stride=2)
        self.up_conv3 = ResCBRBlock(ch1 * 2, ch1, use_dw=False)

        # Final layer
        self.final_conv = nn.Conv3d(ch1, out_channels, kernel_size=1)
        
        self.softmax = nn.Softmax(dim=1)
        
        self._init_weights()
    

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

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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