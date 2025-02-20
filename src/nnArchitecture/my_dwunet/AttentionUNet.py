# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/02/20 16:33:15
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: AttentionUNet DW版本
*      VERSION: v1.0
*      FEATURES: 
=================================================
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
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
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
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

class UpSample(nn.Module):
    """3D Up Convolution"""
    def __init__(self, in_channels, out_channels, trilinear=True):
        super(UpSample, self).__init__()
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.Sequential(
                nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.InstanceNorm3d(out_channels),
                nn.LeakyReLU(inplace=True)
            )
    def forward(self, x):
        return self.up(x)

class AttentionBlock3D(nn.Module):
    """3D Attention Gate"""
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock3D, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.LeakyReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class AttentionUNet3D(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, f_list=[32, 64, 128, 256], trilinear=True):
        super(AttentionUNet3D, self).__init__()
        
        self.MaxPool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.Conv1 = ResConv3D(in_channels, f_list[0])
        self.Conv2 = ResConv3D(f_list[0], f_list[1])
        self.Conv3 = ResConv3D(f_list[1], f_list[2])
        self.Conv4 = ResConv3D(f_list[2], f_list[3])
        
        self.bottleneck = ResConv3D(f_list[3], f_list[3])
        
        self.Up5 = UpSample(f_list[3], f_list[3], trilinear)
        self.Att5 = AttentionBlock3D(F_g=f_list[3], F_l=f_list[3], F_int=f_list[3]//2)
        self.UpConv5 = ResConv3D(f_list[3]*2, f_list[3]//2)
        
        self.Up4 = UpSample(f_list[2], f_list[2], trilinear)
        self.Att4 = AttentionBlock3D(F_g=f_list[2], F_l=f_list[2], F_int=f_list[2]//2)
        self.UpConv4 = ResConv3D(f_list[2]*2, f_list[2]//2)
        
        self.Up3 = UpSample(f_list[1], f_list[1], trilinear)
        self.Att3 = AttentionBlock3D(F_g=f_list[1], F_l=f_list[1], F_int=f_list[1]//2)
        self.UpConv3 = ResConv3D(f_list[1]*2, f_list[1]//2)
        
        self.Up2 = UpSample(f_list[0], f_list[0], trilinear)
        self.Att2 = AttentionBlock3D(F_g=f_list[0], F_l=f_list[0], F_int=f_list[0]//2)
        self.UpConv2 = ResConv3D(f_list[0]*2, f_list[0])
        
        self.outc = nn.Conv3d(f_list[0], out_channels, kernel_size=1)
        
        self.apply(init_weights_3d)  # 初始化权重
        
    def forward(self, x):
        # Encoder
        x1 = self.Conv1(x)       # [B, 32, D, H, W]
        x2 = self.MaxPool(x1)
        x2 = self.Conv2(x2)      # [B, 64, D/2, H/2, W/2]
        x3 = self.MaxPool(x2)
        x3 = self.Conv3(x3)      # [B, 128, D/4, H/4, W/4]
        x4 = self.MaxPool(x3)
        x4 = self.Conv4(x4)      # [B, 256, D/8, H/8, W/8]
        x5 = self.MaxPool(x4)
        
        # Bottleneck
        x5 = self.bottleneck(x5)      # [B, 256, D/16, H/16, W/16]
        
        # Decoder with Attention
        d5 = self.Up5(x5)        # [B, 256, D/8, H/8, W/8]
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.UpConv5(d5)    # [B, 128, D/8, H/8, W/8]
        
        d4 = self.Up4(d5)        # [B, 128, D/4, H/4, W/4]
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.UpConv4(d4)    # [B, 64, D/4, H/4, W/4]
        
        d3 = self.Up3(d4)        # [B, 64, D/2, H/2, W/2]
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.UpConv3(d3)    # [B, 32, D/2, H/2, W/2]
        
        d2 = self.Up2(d3)        # [B, 32, D, H, W]
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.UpConv2(d2)    # [B, 32, D, H, W]
        
        out = self.outc(d2)  # [B, out_channels, D, H, W]
        return out
    
if __name__ == '__main__':
    test_unet(model_class=AttentionBlock3D)