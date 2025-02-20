import torch
from torch import nn
import torch.nn.functional as F

import torch
from torch import nn

# class EMA3D(nn.Module):
#     def __init__(self, channels, factor=8):
#         super(EMA3D, self).__init__()
#         self.group = factor
#         assert channels % self.group == 0
#         self.softmax = nn.Softmax(dim=1)
#         self.averagePooling = nn.AdaptiveAvgPool3d((1, 1, 1))
#         self.Pool_h = nn.AdaptiveAvgPool3d((None, 1, 1))
#         self.Pool_w = nn.AdaptiveAvgPool3d((1, None, 1))
#         self.Pool_d = nn.AdaptiveAvgPool3d((1, 1, None))
        
#         self.groupNorm = nn.GroupNorm(channels // self.group, channels // self.group)
#         self.conv3x3x3 = nn.Conv3d(channels // self.group, channels // self.group, kernel_size=3, stride=1, padding=1)

#     def forward(self, x):
#         b, c, d, h, w = x.size()
#         group_x = x.reshape(b * self.group, c // self.group, d, h, w)
        
#         # Pooling along different dimensions
#         x_h = self.Pool_h(group_x)  # [B*G, C/G, D, 1, 1]
#         x_w = self.Pool_w(group_x)  # [B*G, C/G, 1, H, 1]
#         x_d = self.Pool_d(group_x)  # [B*G, C/G, 1, 1, W]
        
#         # Apply sigmoid activation
#         x_h_sigmoid = x_h.sigmoid()
#         x_w_sigmoid = x_w.sigmoid()
#         x_d_sigmoid = x_d.sigmoid()
        
#         # Apply attention maps using broadcasting
#         x_attended = group_x * x_h_sigmoid * x_w_sigmoid * x_d_sigmoid
        
#         # Group Normalization
#         x1 = self.groupNorm(x_attended)
        
#         # 3x3x3 convolution path
#         x2 = self.conv3x3x3(group_x)
        
#         # Global average pooling
#         x1_pool = self.averagePooling(x1).view(b * self.group, c // self.group)
#         x2_pool = self.averagePooling(x2).view(b * self.group, c // self.group)
        
#         # Softmax for attention weights
#         x1_soft = self.softmax(x1_pool.unsqueeze(1))  # [B*G, 1, C/G]
#         x2_soft = self.softmax(x2_pool.unsqueeze(1))  # [B*G, 1, C/G]
        
#         # Matrix multiplication to get weights
#         weights = torch.matmul(x1_soft, x2_pool.unsqueeze(2))  # [B*G, 1, 1]
#         weights = weights.view(b * self.group, 1, d, h, w)
        
#         # Apply sigmoid to weights and multiply with group_x
#         output = group_x * weights.sigmoid()
#         output = output.view(b, c, d, h, w)
        
#         return output

class EMA3D(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA3D, self).__init__()
        self.group = factor
        assert channels // self.group > 0
        self.softmax = nn.Softmax(dim=-1)
        self.averagePooling = nn.AdaptiveAvgPool3d((1, 1, 1))  # 3D 全局平均池化
        self.maxPooling = nn.AdaptiveMaxPool3d((1, 1, 1))      # 3D 全局最大池化
        self.Pool_h = nn.AdaptiveAvgPool3d((None, 1, 1))       # 高度方向池化
        self.Pool_w = nn.AdaptiveAvgPool3d((1, None, 1))       # 宽度方向池化
        self.Pool_d = nn.AdaptiveAvgPool3d((1, 1, None))       # 深度方向池化

        self.groupNorm = nn.GroupNorm(channels // self.group, channels // self.group)
        self.conv1x1x1 = nn.Sequential(
            nn.Conv3d(channels // self.group, channels // self.group, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(channels // self.group),
            nn.ReLU(inplace=True)
        )
        self.conv3x3x3 = nn.Sequential(
            nn.Conv3d(channels // self.group, channels // self.group, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(channels // self.group),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b, c, d, h, w = x.size()
        group_x = x.reshape(b * self.group, -1, d, h, w)  # 分组处理

        # 高度、宽度、深度方向池化
        x_c = self.maxPooling(group_x)  # [B*G, C/G, 1, 1, 1]
        x_h = self.Pool_h(group_x)  # [B*G, C/G, D, 1, 1]
        x_w = self.Pool_w(group_x).permute(0, 1, 3, 2, 4)  # [B*G, C/G, 1, H, 1]
        x_d = self.Pool_d(group_x).permute(0, 1, 4, 3, 2)  # [B*G, C/G, 1, 1, W]

        # 拼接并卷积
        hwd = self.conv1x1x1(torch.cat([x_h, x_w, x_d], dim=2))  # 拼接后卷积
        x_h, x_w, x_d = torch.split(hwd, [d, h, w], dim=2)       # 拆分

        # Apply sigmoid activation
        x_h_sigmoid = x_h.sigmoid().view(b*self.group, c // self.group, d, 1, 1)
        x_w_sigmoid = x_w.sigmoid().view(b*self.group, c // self.group, 1, h, 1)
        x_d_sigmoid = x_d.sigmoid().view(b*self.group, c // self.group, 1, 1, w)
        
        # Apply attention maps using broadcasting
        x_attended =x_h_sigmoid * x_w_sigmoid * x_d_sigmoid
        
        x1 = self.groupNorm(group_x * x_attended)  # 高度、宽度、深度注意力
        x11 = self.softmax(self.averagePooling(x1).reshape(b * self.group, -1, 1).permute(0, 2, 1))  # 全局平均池化 + softmax
        x12 = x1.reshape(b * self.group, c // self.group, -1)

        # 3x3x3 路径
        x2 = self.conv3x3x3(group_x)  # 通过 3x3x3 卷积层
        x21 = self.softmax(self.averagePooling(x2).reshape(b * self.group, -1, 1).permute(0, 2, 1))  # 全局平均池化 + softmax
        x22 = x2.reshape(b * self.group, c // self.group, -1)

        # 计算权重
        weights = (torch.matmul(x11, x22) + torch.matmul(x21, x12)).reshape(b * self.group, -1, d, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, d, h, w)


# class EMA3D(nn.Module):
#     def __init__(self, channels, factor=32):
#         super(EMA3D, self).__init__()
#         self.group = factor
#         assert channels % self.group == 0
#         self.softmax = nn.Softmax(dim=1)
#         self.averagePooling = nn.AdaptiveAvgPool3d((1, 1, 1))
#         self.Pool_h = nn.AdaptiveAvgPool3d((None, 1, 1))
#         self.Pool_w = nn.AdaptiveAvgPool3d((1, None, 1))
#         self.Pool_d = nn.AdaptiveAvgPool3d((1, 1, None))
        
#         self.conv_h = nn.Conv3d(channels // self.group, channels // self.group, kernel_size=1)
#         self.conv_w = nn.Conv3d(channels // self.group, channels // self.group, kernel_size=1)
#         self.conv_d = nn.Conv3d(channels // self.group, channels // self.group, kernel_size=1)
#         self.conv_comb = nn.Conv3d(3 * (channels // self.group), channels // self.group, kernel_size=1)
        
#         self.groupNorm = nn.GroupNorm(channels // self.group, channels // self.group)
#         self.conv3x3x3 = nn.Conv3d(channels // self.group, channels // self.group, kernel_size=3, stride=1, padding=1)

#     def forward(self, x):
#         b, c, d, h, w = x.size()
#         group_x = x.reshape(b * self.group, c // self.group, d, h, w)
        
#         # Pooling along different dimensions
#         x_h = self.Pool_h(group_x)        # [B*G, C/G, D, 1, 1]
#         x_w = self.Pool_w(group_x)        # [B*G, C/G, 1, H, 1]
#         x_d = self.Pool_d(group_x)        # [B*G, C/G, 1, 1, W]
        
#         # Apply 1x1 convolutions
#         conv_h_xh = self.conv_h(x_h)      # [B*G, C/G, D, 1, 1]
#         conv_w_xw = self.conv_w(x_w)      # [B*G, C/G, 1, H, 1]
#         conv_d_xd = self.conv_d(x_d)      # [B*G, C/G, 1, 1, W]
        
#         # Upsample to original spatial dimensions
#         upsample_h = F.interpolate(conv_h_xh, size=(d, h, w), mode='trilinear', align_corners=True)
#         upsample_w = F.interpolate(conv_w_xw, size=(d, h, w), mode='trilinear', align_corners=True)
#         upsample_d = F.interpolate(conv_d_xd, size=(d, h, w), mode='trilinear', align_corners=True)
        
#         # Concatenate along the channel dimension
#         concat = torch.cat([upsample_h, upsample_w, upsample_d], dim=1)  # [B*G, 3*(C/G), D, H, W]
        
#         # Combine the features
#         combined = self.conv_comb(concat)  # [B*G, C/G, D, H, W]
        
#         # Generate attention weights
#         weights = torch.sigmoid(combined)
        
#         # Apply attention weights
#         attended_x = group_x * weights
        
#         # Group normalization
#         x1 = self.groupNorm(attended_x)
        
#         # 3x3x3 convolution path
#         x2 = self.conv3x3x3(group_x)
        
#         # Global average pooling for x1 and x2
#         x1_pool = self.averagePooling(x1).view(b * self.group, c // self.group)
#         x2_pool = self.averagePooling(x2).view(b * self.group, c // self.group)
        
#         # Softmax for attention weights
#         x1_soft = self.softmax(x1_pool.unsqueeze(1))  # [B*G, 1, C/G]
#         x2_soft = self.softmax(x2_pool.unsqueeze(1))  # [B*G, 1, C/G]
        
#         # Matrix multiplication to get final weights
#         final_weights = torch.matmul(x1_soft, x2_pool.unsqueeze(2))  # [B*G, 1, 1]
#         final_weights = final_weights.view(b * self.group, 1, d, h, w)
        
#         # Apply final weights to group_x
#         output = group_x * final_weights.sigmoid()
#         output = output.view(b, c, d, h, w)
        
#         return output



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ema3d = EMA3D(128).to(device)
    input_data = torch.rand(1, 128, 32, 32, 32).to(device)  # 3D 输入
    output_data = ema3d(input_data)

    print(ema3d)
    print(output_data.shape)  # 输出形状应与输入形状一致