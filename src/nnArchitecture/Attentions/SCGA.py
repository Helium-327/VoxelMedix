# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/02/20 21:08:28
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: SCGA (Spatial Correlation Grouped Attention)模块 实现
*      VERSION: v1.0
*      FEATURES: 
=================================================
'''


import torch
import torch.nn as nn
import torch.nn.functional as F

class Hybrid_Pooling(nn.Module):
    def __init__(self, out_size):
        super(Hybrid_Pooling, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(out_size)
        self.max_pool = nn.AdaptiveMaxPool3d(out_size)
    def forward(self, x):
        return self.avg_pool(x) + self.max_pool(x)
    
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