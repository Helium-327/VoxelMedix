
# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2024/12/12 16:47:55
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 二维EMA注意力
=================================================
'''

import torch

from torch import nn

class EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        super(EMA, self).__init__()
        self.group = factor
        assert channels // self.group > 0
        self.softmax = nn.Softmax(dim=-1)
        self.averagePooling = nn.AdaptiveAvgPool2d((1,1))
        self.Pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.Pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.groupNorm = nn.GroupNorm(channels // self.group, channels//self.group)
        self.conv1x1 = nn.Conv2d(channels // self.group, channels // self.group, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.group, channels // self.group, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b*self.group, -1, h, w)
        x_h = self.Pool_h(group_x)  # 高度方向池化
        x_w = self.Pool_w(group_x).permute(0, 1, 3, 2)  # 宽度方向池化

        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2)) # 拼接之后卷积
        x_h, x_w = torch.split(hw, [h, w], dim=2)       # 拆分

        # 1x1路径
        x1 = self.groupNorm(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())          # 高度的注意力
        x11 = self.softmax(self.averagePooling(x1).reshape(b*self.group, -1, 1).permute(0, 2, 1)) # 对 x1 进行平均池化，然后进行 softmax 操作
        x12 = x1.reshape(b*self.group, c//self.group, -1)

        # 3x3路径
        x2 = self.conv3x3(group_x) # 通过 3x3卷积层
        x21 = self.softmax(self.averagePooling(x2).reshape(b*self.group, -1, 1).permute(0, 2, 1)) # 对 x2 进行平均池化，然后进行 softmax 操作
        x22 = x2.reshape(b*self.group, c//self.group, -1)

        weights = (torch.matmul(x11, x22) + torch.matmul(x21, x12)).reshape(b * self.group, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)
        

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ema = EMA(128).to(device)
    input_data = torch.rand(1, 128, 128, 128).to(device)
    output_data = ema(input_data)


    print(ema)

    print(output_data.shape)