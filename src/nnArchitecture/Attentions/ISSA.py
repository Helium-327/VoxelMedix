# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/01/08 19:33:16
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: ISSA: Improve Self-Attention
*      VERSION: v1.0
=================================================
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class ISSA(nn.Module):
    def __init__(self, channels, reduction_ratio=8):
        super(ISSA, self).__init__()
        
        self.channels = channels
        self.reduction = reduction_ratio
        
        # Pointwise convolutions to generate Q, K, V
        self.query_conv = nn.Conv3d(channels, channels // reduction_ratio, kernel_size=1)
        self.key_conv = nn.Conv3d(channels, channels // reduction_ratio, kernel_size=1)
        self.value_conv = nn.Conv3d(channels, channels, kernel_size=1)
        
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        B, C, L1, L2, L3 = x.size()
        
        Q = self.query_conv(x)
        K = self.key_conv(x)
        V = self.value_conv(x)
        
        Q_reshaped = Q.permute(0, 2, 3, 1, 4).contiguous().view(B, L1*L2, L3, -1)
        K_reshaped = K.permute(0, 2, 3, 1, 4).contiguous().view(B, L1*L2, -1, L3)
        V_reshaped = V.permute(0, 2, 3, 1, 4).contiguous().view(B, L1*L2, L3, -1)
        
        scale_factor = (self.channels // self.reduction) ** 0.5
        
        attention_scores = torch.matmul(Q_reshaped, K_reshaped) / scale_factor
        attention_weights = self.softmax(attention_scores)
        
        attention_out = torch.matmul(attention_weights, V_reshaped)
        
        
        