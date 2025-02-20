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

from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
# ✅ DW卷积
# ✅ 实例归一化
# ✅ LeakyReLU激活函数
# ✅ 残差
#    空间相关性注意力

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
            nn.Conv3d(in_channels, in_channels, kernel_size=3, groups=in_channels, padding=1),
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

class ResDConv3D(nn.Module):
    """带残差连接的深度可分离卷积块"""
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


class Hybrid_Pooling(nn.Module):
    def __init__(self, out_size):
        super(Hybrid_Pooling, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(out_size)
        self.max_pool = nn.AdaptiveMaxPool3d(out_size)
    def forward(self, x):
        return 0.5 * self.avg_pool(x) + 0.5 * self.max_pool(x)
    
        
class SE(nn.Module):
    def __init__(self, in_channels, reduction_ratio=2):
        super(SE, self).__init__()

        if in_channels // reduction_ratio <= 0:
                    raise ValueError(f"Reduction ratio {reduction_ratio} is too large for the number of input channels {in_channels}.")
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y.expand_as(x)       

class SCGA(nn.Module):
    """Spatial Correlation Grouped Attention"""
    def __init__(self, channels, group=2):  # 添加group参数
        super(SCGA, self).__init__()
        self.group = group
        assert channels % group == 0, "channels must be divisible by group"
        self.softmax = nn.Softmax(dim=1)
        self.averagePooling = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.maxPooling = nn.AdaptiveMaxPool3d((1, 1, 1))
        self.Pool_h = Hybrid_Pooling((None, 1, 1))
        self.Pool_w = Hybrid_Pooling((1, None, 1))
        self.Pool_d = Hybrid_Pooling((1, 1, None))

        self.groupNorm = nn.InstanceNorm3d(channels // group)
        self.conv1x1x1 = nn.Sequential(
            nn.Conv3d(channels // group, channels // group, kernel_size=1),
            nn.InstanceNorm3d(channels // group),
            nn.LeakyReLU(inplace=True)
        )
        self.conv3x3x3 = nn.Sequential(
            nn.Conv3d(channels // group, channels // group, kernel_size=3, padding=1),
            nn.InstanceNorm3d(channels // group),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        b, c, d, h, w = x.size()
        group = self.group
        # 正确分组处理
        group_x = x.view(b * group, c // group, d, h, w)  # [B*G, C/G, D, H, W]

        # 各方向池化
        x_h = self.Pool_h(group_x)  # [B*G, C/G, D, 1, 1]
        x_w = self.Pool_w(group_x).permute(0, 1, 3, 2, 4)  # [B*G, C/G, H, 1, 1]
        x_d = self.Pool_d(group_x).permute(0, 1, 4, 3, 2)  # [B*G, C/G, W, 1, 1]

        # 拼接+卷积+拆分
        hwd = self.conv1x1x1(torch.cat([x_h, x_w, x_d], dim=2))  # dim2: D+H+W
        x_h, x_w, x_d = torch.split(hwd, [d, h, w], dim=2)

        # 生成注意力权重
        x_h_sigmoid = torch.sigmoid(x_h)  # [B*G, C/G, D, 1, 1]
        x_w_sigmoid = torch.sigmoid(x_w.permute(0, 1, 3, 2, 4))  # [B*G, C/G, 1, H, 1]
        x_d_sigmoid = torch.sigmoid(x_d.permute(0, 1, 4, 3, 2))  # [B*G, C/G, 1, 1, W]
        
        x_attended = x_h_sigmoid * x_w_sigmoid * x_d_sigmoid  # [B*G, C/G, D, H, W]

        # 注意力应用
        x1 = self.groupNorm(group_x * x_attended)
        x11 = self.softmax(self.averagePooling(x1).flatten(2).permute(0, 2, 1))  # [B*G, 1, C/G]
        x12 = x1.view(b * group, -1, d * h * w)  # [B*G, C/G, D*H*W]

        # 3x3x3路径
        x2 = self.conv3x3x3(group_x)
        x21 = self.softmax(self.averagePooling(x2).flatten(2).permute(0, 2, 1))  # [B*G, 1, C/G]
        x22 = x2.view(b * group, -1, d * h * w)  # [B*G, C/G, D*H*W]

        # 权重计算
        weights = (torch.bmm(x11, x22) + torch.bmm(x21, x12))  # [B*G, 1, D*H*W]
        weights = weights.view(b * group, 1, d, h, w).sigmoid()

        # 输出
        output = group_x * weights
        return output.view(b, c, d, h, w)
    
    
class SCA(nn.Module):
    def __init__(self, channels):
        super(SCA, self).__init__()
        
        self.hybridPooling = Hybrid_Pooling(1)          # 3D 全局混合池化
        self.avg_pooling = nn.AdaptiveAvgPool3d(1)
        self.conv1x1x1 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1),
            nn.InstanceNorm3d(channels),
            nn.LeakyReLU(inplace=True)
        )
        
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(channels*4, channels, kernel_size=1),
            nn.InstanceNorm3d(channels),
            nn.LeakyReLU(inplace=True)
        )
        self.Channels_attention = SE(channels)
        
        self.out_conv = nn.Sequential(
            nn.Conv3d(channels*4, channels, kernel_size=1),
            nn.InstanceNorm3d(channels),
            nn.LeakyReLU(inplace=True)
        )
            
    def forward(self, x):
        b, c, d, h, w = x.size()
        x0 =  self.conv1x1x1(x)
        x1 = self.conv1x1x1(x)
        x2 = self.conv1x1x1(x)
        x3 = self.conv1x1x1(x)
        
        hwd = self.hybridPooling(x0)
        cwd = self.hybridPooling(x1.permute(0, 2, 1, 3, 4))
        chd = self.hybridPooling(x2.permute(0, 3, 1, 2, 4))
        cwh = self.hybridPooling(x3.permute(0, 4, 1, 2, 3))
        
        hwd_map = F.sigmoid(self.conv1x1x1(hwd)).expand_as(x) * x 
        cwd_map = F.sigmoid(self.conv1x1x1(cwd)).expand_as(x) * x
        chd_map = F.sigmoid(self.conv1x1x1(chd)).expand_as(x) * x
        cwh_map = F.sigmoid(self.conv1x1x1(cwh)).expand_as(x) * x
        
        fusion_map =  self.fusion_conv(torch.cat([hwd_map, cwd_map, chd_map, cwh_map], dim=1)) # [B, C, H, W, D]
        
        return_map = x*F.sigmoid(self.avg_pooling(self.conv1x1x1(fusion_map))) # [B, C, H, W, D]
        
        
        x_out = self.Channels_attention(return_map)
        
        return x_out + x
    
# class SCA(nn.Module):
#     """Spatial Correlation Grouped Attention (空间相关性注意力机制)"""
#     def __init__(self, channels):
#         super(SCA, self).__init__()
#         assert channels   > 0
#         self.softmax = nn.Softmax(dim=-1)
#         self.hybridPooling = Hybrid_Pooling((1, 1, 1))          # 3D 全局混合池化
#         self.Pool_h = Hybrid_Pooling((None, 1, 1))       # 高度方向池化
#         self.Pool_w = Hybrid_Pooling((1, None, 1))       # 宽度方向池化
#         self.Pool_d = Hybrid_Pooling((1, 1, None))       # 深度方向池化

#         self.conv1x1x1 = nn.Sequential(
#             nn.Conv3d(channels, channels, kernel_size=1),
#             nn.InstanceNorm3d(channels),
#             nn.LeakyReLU(inplace=True)
#         )
#         self.conv3x3x3 = nn.Sequential(
#             nn.Conv3d(channels ,channels, kernel_size=3, stride=1, padding=1),
#             nn.InstanceNorm3d(channels  ),
#             nn.LeakyReLU(inplace=True)
#         )
    
#     def forward(self, x):
#         b, c, d, h, w = x.size()
#         # x = x.reshape(b , -1, d, h, w)  # 分组处理

#         # 高度、宽度、深度方向池化
#         x_h = self.Pool_h(x)  # [B*G, C/G, D, 1, 1]
#         x_w = self.Pool_w(x).permute(0, 1, 3, 2, 4)  # [B*G, C/G, 1, H, 1]
#         x_d = self.Pool_d(x).permute(0, 1, 4, 3, 2)  # [B*G, C/G, 1, 1, W]

#         # 拼接并卷积
#         hwd = self.conv1x1x1(torch.cat([x_h, x_w, x_d], dim=2))  # 拼接后卷积
#         x_h, x_w, x_d = torch.split(hwd, [d, h, w], dim=2)       # 拆分

#         # Apply sigmoid activation
#         x_h_sigmoid = F.sigmoid(x_h)
#         x_w_sigmoid = F.sigmoid(x_w)
#         x_d_sigmoid = F.sigmoid(x_d)
        
#         # Apply attention maps using broadcasting
#         x_attended = x_h_sigmoid * x_w_sigmoid * x_d_sigmoid

#         x1 = x_attended * x
#         x11 = self.softmax(self.hybridPooling(x1).reshape(b, -1, 1).permute(0, 2, 1))  # 全局平均池化 + softmax
#         x12 = x1.reshape(b, c, -1).permute(2, 1, 0)

#         # 3x3x3 路径
#         x2 = self.conv3x3x3(x)  # 通过 3x3x3 卷积层
#         x21 = self.softmax(self.hybridPooling(x2).reshape(b, -1, 1).permute(0, 2, 1))  # 全局平均池化 + softmax
#         x22 = x2.reshape(b, c, -1).permute(2, 1, 0)

#         # 计算权重
#         weights = (torch.matmul(x11, x22) + torch.matmul(x21, x12))
#         return (x * weights.sigmoid()).reshape(b, c, d, h, w)


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

class SCGA_UNet3D(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, f_list=[32, 64, 128, 256], trilinear=True):
        super(SCGA_UNet3D, self).__init__()
        
        self.MaxPool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.Conv1 = ResDConv3D(in_channels, f_list[0])
        self.Conv2 = ResDConv3D(f_list[0], f_list[1])
        self.Conv3 = ResDConv3D(f_list[1], f_list[2])
        self.Conv4 = ResDConv3D(f_list[2], f_list[3])
        
        self.bottleneck = nn.Sequential(
            SCGA(f_list[3]),
            ResDConv3D(f_list[3], f_list[3])
        )
        
        self.Up5 = UpSample(f_list[3], f_list[3], trilinear)
        self.Att5 = AttentionBlock3D(F_g=f_list[3], F_l=f_list[3], F_int=f_list[3]//2)
        self.UpConv5 = ResDConv3D(f_list[3]*2, f_list[3]//2)
        
        self.Up4 = UpSample(f_list[2], f_list[2], trilinear)
        self.Att4 = AttentionBlock3D(F_g=f_list[2], F_l=f_list[2], F_int=f_list[2]//2)
        self.UpConv4 = ResDConv3D(f_list[2]*2, f_list[2]//2)
        
        self.Up3 = UpSample(f_list[1], f_list[1], trilinear)
        self.Att3 = AttentionBlock3D(F_g=f_list[1], F_l=f_list[1], F_int=f_list[1]//2)
        self.UpConv3 = ResDConv3D(f_list[1]*2, f_list[1]//2)
        
        self.Up2 = UpSample(f_list[0], f_list[0], trilinear)
        self.Att2 = AttentionBlock3D(F_g=f_list[0], F_l=f_list[0], F_int=f_list[0]//2)
        self.UpConv2 = ResDConv3D(f_list[0]*2, f_list[0])
        
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
    test_unet(model_class=SCGA_UNet3D)