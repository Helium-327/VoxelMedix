# 可变形3D卷积核心组件示例
import torch
import torch.nn as nn
# from mmcv.ops import ModulatedDeformConv3d

import torch
import torch.nn as nn
import torch.nn.functional as F

class DeformConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DeformConv3D, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        
        # 预测偏移量（offset）的卷积层
        # 输出通道数为 3 * kernel_size^3 （每个位置x,y,z三个方向的偏移）
        self.offset_conv = nn.Conv3d(
            in_channels, 
            3 * self.kernel_size**3, 
            kernel_size=self.kernel_size, 
            padding=self.padding
        )
        
        # 常规卷积层（用于实际的特征提取）
        self.conv = nn.Conv3d(
            in_channels, 
            out_channels, 
            kernel_size=self.kernel_size, 
            padding=self.padding
        )

    def forward(self, x):
        # 1. 预测偏移量 [B, 3*K^3, D, H, W]
        offset = self.offset_conv(x)
        
        # 2. 生成可变形采样网格
        B, _, D, H, W = x.size()
        grid = self._generate_grid(B, D, H, W, x.device)  # 初始网格
        deformed_grid = grid + offset.permute(0, 2, 3, 4, 1)  # 应用偏移量
        
        # 3. 双线性插值采样
        deformed_x = F.grid_sample(x, deformed_grid, padding_mode='border', align_corners=False)
        
        # 4. 常规卷积操作
        output = self.conv(deformed_x)
        return output

    def _generate_grid(self, batch_size, D, H, W, device):
        # 生成归一化的3D网格坐标 [-1, 1]
        z, y, x = torch.meshgrid(
            torch.linspace(-1, 1, D, device=device),
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device)
        )
        grid = torch.stack((x, y, z), dim=-1)  # [D, H, W, 3]
        grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)  # [B, D, H, W, 3]
        return grid
    
    
if __name__ == '__main__':
    x = torch.randn(1, 4, 128, 128, 128)
    model = DeformConv3D(4, 4, kernel_size=3)
    output = model(x)