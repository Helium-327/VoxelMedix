# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/02/15 11:21:32
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 
*      VERSION: v1.0
*      FEATURES: UNet3D 改进网络
=================================================
'''
# ✅ DW卷积
# ✅ 实例归一化
# ✅ LeakyReLU激活函数
# ✅ 残差

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchinfo import summary
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.utils.test_unet import test_unet

def init_weights_3d(m):
    """Initialize 3D卷积和BN层的权重"""
    if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm3d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class DepthwiseSeparbleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, groups=in_channels),
            nn.InstanceNorm3d(in_channels),
            nn.LeakyReLU(inplace=True)
        )
        self.pointwise = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),  # 注意: out_channels 应匹配后续输入
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)  # 必须执行逐点卷积
        return x

class ResConv3D(nn.Module):
    """带残差连接的各向异性卷积块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            DepthwiseSeparbleConv3D(out_channels, out_channels),
            nn.InstanceNorm3d(out_channels),
        )
        self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        self.relu = nn.LeakyReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.conv(x)
        if self.shortcut:
            residual = self.shortcut(residual)
        out += residual
        return self.relu(out)
    
class Down3D(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            ResConv3D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up3D(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.Sequential(
                nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2),
                nn.InstanceNorm3d(out_channels//2),
                nn.LeakyReLU(inplace=True)
            )
        
        self.conv = ResConv3D(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, f_list=[32, 64, 128, 256], trilinear=True):
        super(UNet3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.inc = ResConv3D(in_channels, f_list[0])  # 4 --> 32
        self.down1 = Down3D(f_list[0], f_list[1])        # 32 --> 64
        self.down2 = Down3D(f_list[1], f_list[2])        # 64 --> 128
        self.down3 = Down3D(f_list[2], f_list[3])        # 128 --> 256
        self.down4 = Down3D(f_list[3], f_list[3])        # 256 --> 256
        
        self.up1 = Up3D(f_list[3]*2, f_list[2], trilinear)  # 512 --> 128
        self.up2 = Up3D(f_list[3], f_list[1], trilinear)    # 256 --> 64
        self.up3 = Up3D(f_list[2], f_list[0], trilinear)    # 128 --> 32
        self.up4 = Up3D(f_list[1], f_list[0], trilinear)    # 64 --> 32
        self.outc = nn.Conv3d(f_list[0], out_channels, kernel_size=1)  # 32 --> 4
        
        self.apply(init_weights_3d) # 初始化权重

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
if __name__ == "__main__":
    test_unet(model_class=UNet3D)