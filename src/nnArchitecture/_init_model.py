# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/01/15 14:36:46
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 模型初始化
*      VERSION: v1.0
*      FEATURES: 
=================================================
'''
import torch.nn as nn

def init_all_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        if m.weight is not None:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)