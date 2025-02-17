# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/02/16 12:22:25
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 
*      VERSION: v1.0
*      FEATURES: SKConv 可选择卷积 2D
=================================================
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchinfo import summary
import time

# class SKConv(nn.Module):
#     def __init__(self, features, WH, M, G, r, stride=1 ,L=32):
#         """ Constructor
#         Args:
#             features: input channel dimensionality.
#             WH: input spatial dimensionality, used for GAP kernel size.
#             M: the number of branchs.
#             G: num of convolution groups.
#             r: the radio for compute d, the length of z.
#             stride: stride, default 1.
#             L: the minimum dim of the vector z in paper, default 32.
#         """
#         super(SKConv, self).__init__()
#         d = max(int(features/r), L)
#         self.M = M
#         self.features = features
#         self.convs = nn.ModuleList([])
#         for i in range(M):
#             self.convs.append(nn.Sequential(
#                 nn.Conv2d(features, features, kernel_size=3+i*2, stride=stride, padding=1+i, groups=G),
#                 nn.BatchNorm2d(features),
#                 nn.ReLU(inplace=False)
#             ))
#         # self.gap = nn.AvgPool2d(int(WH/stride))
#         self.fc = nn.Linear(features, d)
#         self.fcs = nn.ModuleList([])
#         for i in range(M):
#             self.fcs.append(
#                 nn.Linear(d, features)
#             )
#         self.softmax = nn.Softmax(dim=1)
        
#     def forward(self, x):
#         for i, conv in enumerate(self.convs):
#             fea = conv(x).unsqueeze_(dim=1)
#             if i == 0:
#                 feas = fea
#             else:
#                 feas = torch.cat([feas, fea], dim=1)
#         fea_U = torch.sum(feas, dim=1)
#         # fea_s = self.gap(fea_U).squeeze_()
#         fea_s = fea_U.mean(-1).mean(-1)
#         fea_z = self.fc(fea_s)
#         for i, fc in enumerate(self.fcs):
#             vector = fc(fea_z).unsqueeze_(dim=1)
#             if i == 0:
#                 attention_vectors = vector
#             else:
#                 attention_vectors = torch.cat([attention_vectors, vector], dim=1)
#         attention_vectors = self.softmax(attention_vectors)
#         attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
#         fea_v = (feas * attention_vectors).sum(dim=1)
#         return fea_v
   

class SKConv(nn.Module):
    def __init__(self, in_channels, out_channels, M=2, G=32, r=16):
        super(SKConv, self).__init__()
        self.M = M  # 路径数量，决定不同卷积核的选择数量
        self.G = G  # 分组卷积的组数，控制每个路径的基数
        self.r = r  # 缩减比例，控制融合操作中的参数数量
        self.out_channels = out_channels

        # Split: 使用不同大小的卷积核
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=G)
        self.conv5x5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, groups=G)

        # Fuse: 全局平均池化和全连接层
        self.gap = nn.AdaptiveAvgPool2d(1)  # 全局平均池化，生成通道统计信息
        self.fc = nn.Sequential(
            nn.Linear(out_channels, out_channels // r),  # 全连接层，缩减通道数
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Linear(out_channels // r, out_channels * M),  # 全连接层，扩展通道数
            nn.Softmax(dim=1)  # Softmax函数，生成选择权重
        )

    def forward(self, x):
        # Split
        U1 = self.conv3x3(x)  # 3x3卷积
        U2 = self.conv5x5(x)  # 5x5卷积
        U = torch.stack([U1, U2], dim=1)  # 将两个卷积结果堆叠，shape: [batch, M, C, H, W]

        # Fuse
        S = self.gap(U1 + U2).squeeze(-1).squeeze(-1)  # 全局平均池化并压缩维度，shape: [batch, C]
        Z = self.fc(S).view(-1, self.M, self.out_channels, 1, 1)  # 全连接层并调整维度，shape: [batch, M, C, 1, 1]

        # Select
        V = (U * Z).sum(dim=1)  # 根据选择权重聚合特征图，shape: [batch, C, H, W]
        return V
     
class SKUnit(nn.Module):
    def __init__(self, in_features, out_features, WH, M, G, r, mid_features=None, stride=1, L=32):
        """ Constructor
        Args:
            in_features: input channel dimensionality.
            out_features: output channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            mid_features: the channle dim of the middle conv with stride not 1, default out_features/2.
            stride: stride.
            L: the minimum dim of the vector z in paper.
        """
        super(SKUnit, self).__init__()
        if mid_features is None:
            mid_features = int(out_features/2)
        self.feas = nn.Sequential(
            nn.Conv2d(in_features, mid_features, 1, stride=1),
            nn.BatchNorm2d(mid_features),
            SKConv(mid_features, WH, M, G, r, stride=stride, L=L),
            nn.BatchNorm2d(mid_features),
            nn.Conv2d(mid_features, out_features, 1, stride=1),
            nn.BatchNorm2d(out_features)
        )
        if in_features == out_features: # when dim not change, in could be added diectly to out
            self.shortcut = nn.Sequential()
        else: # when dim not change, in should also change dim to be added to out
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_features, out_features, 1, stride=stride),
                nn.BatchNorm2d(out_features)
            )
    
    def forward(self, x):
        fea = self.feas(x)
        return fea + self.shortcut(x)
    
if __name__=='__main__':
    x = torch.rand(8, 64, 32, 32)
    conv = SKConv(64, 32, 3, 8, 2)
    out = conv(x)
    # criterion = nn.L1Loss()
    # loss = criterion(out, x)
    # loss.backward()
    print('out shape : {}'.format(out.shape))
    # print('loss value : {}'.format(loss))