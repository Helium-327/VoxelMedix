# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/02/15 12:28:22
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 
*      VERSION: v1.0
*      FEATURES: 魔改AttentionUNet3D
=================================================
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchinfo import summary
import time

class AnisotropicConv3D(nn.Module):
    """各向异性3D卷积 (kernel_size分解为空间和深度卷积)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.spatial_conv = nn.Conv3d(in_channels, out_channels, kernel_size=(1,3,3), padding=(0,1,1))
        self.depth_conv = nn.Conv3d(out_channels, out_channels, kernel_size=(3,1,1), padding=(1,0,0))
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.spatial_conv(x)
        x = self.depth_conv(x)
        return self.relu(self.bn(x))

class ResConv3D(nn.Module):
    """带残差连接的各向异性卷积块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            AnisotropicConv3D(in_channels, out_channels),
            AnisotropicConv3D(out_channels, out_channels)
        )
        self.downsample = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.conv(x)
        if self.downsample:
            residual = self.downsample(residual)
        out += residual
        return self.relu(out)

class DynamicConvAttention3D(nn.Module):
    """动态卷积注意力机制"""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        # 动态权重生成网络
        self.W = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(F_g + F_l, F_int, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(F_int, F_g * F_l, kernel_size=1),
            nn.Sigmoid()
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_l, 1, kernel_size=1),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        self.F_g = F_g
        self.F_l = F_l
    
    def forward(self, g, x):
        # 生成动态卷积权重
        combined = torch.cat([g, x], dim=1)
        dynamic_weights = self.W(combined).view(-1, self.F_g, self.F_l, 1, 1, 1)
        
        # 应用动态卷积
        g = g.unsqueeze(2)
        attended = (g * dynamic_weights).sum(dim=1)
        
        # 生成注意力图
        return x * self.psi(attended)

class DenseASPP3D(nn.Module):
    """3D DenseASPP模块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        dilations = [1, 2, 4, 8]
        self.aspp_blocks = nn.ModuleList()
        current_channels = in_channels
        
        for d in dilations:
            block = nn.Sequential(
                nn.Conv3d(current_channels, out_channels, kernel_size=3, 
                         padding=d, dilation=d),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.aspp_blocks.append(block)
            current_channels += out_channels
        
        self.final_conv = nn.Conv3d(current_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        features = [x]
        for block in self.aspp_blocks:
            new_feature = block(torch.cat(features, dim=1))
            features.append(new_feature)
        return self.final_conv(torch.cat(features[1:], dim=1))

class UpSample(nn.Module):
    """各向异性上采样"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(1,2,2), stride=(1,2,2)),
            AnisotropicConv3D(out_channels, out_channels)
        )
    
    def forward(self, x):
        return self.up(x)

class ModifiedAttentionUNet3D(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, f_list=[32, 64, 128, 256]):
        super().__init__()
        # Encoder
        self.encoder = nn.ModuleList([
            ResConv3D(in_channels, f_list[0]),
            ResConv3D(f_list[0], f_list[1]),
            ResConv3D(f_list[1], f_list[2]),
            ResConv3D(f_list[2], f_list[3])
        ])
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = DenseASPP3D(f_list[3], f_list[3])
        
        # Decoder
        self.upconvs = nn.ModuleList([
            UpSample(f_list[3], f_list[2]),
            UpSample(f_list[2], f_list[1]),
            UpSample(f_list[1], f_list[0]),
            UpSample(f_list[0], f_list[0])
        ])
        
        # Attention gates
        self.attentions = nn.ModuleList([
            DynamicConvAttention3D(f_list[3], f_list[3], f_list[3]//2),
            DynamicConvAttention3D(f_list[2], f_list[2], f_list[2]//2),
            DynamicConvAttention3D(f_list[1], f_list[1], f_list[1]//2),
            DynamicConvAttention3D(f_list[0], f_list[0], f_list[0]//2)
        ])
        
        # Final convolution
        self.outc = nn.Conv3d(f_list[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        # Encoder
        for i, down in enumerate(self.encoder):
            x = down(x)
            if i != len(self.encoder)-1:
                skips.append(x)
                x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        for i, (up, att) in enumerate(zip(self.upconvs, self.attentions)):
            x = up(x)
            skip = skips[-(i+1)]
            x = att(x, skip)
            x = torch.cat([x, skip], dim=1)
            if i != len(self.upconvs)-1:
                x = self.encoder[-(i+2)](x)
        
        return self.outc(x)
    
def test_unet():
    # 配置参数
    batch_size = 1
    in_channels = 4
    spatial_size = 128
    num_classes = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 生成测试数据
    input_tensor = torch.randn(
        batch_size,
        in_channels,
        spatial_size,
        spatial_size,
        spatial_size
    ).to(device)
    
    # 初始化模型
    model = ModifiedAttentionUNet3D(
        in_channels=in_channels,
        out_channels=num_classes,
        # channel_list=[32, 64, 128, 256, 512]
    ).to(device)
    
    # 打印模型结构
    # 使用torchinfo生成模型摘要
    summary(model, input_size=(batch_size, in_channels, spatial_size, spatial_size, spatial_size), device=device)
    
    # 前向传播并测量时间
    start_time = time.time()
    with torch.no_grad():
        output = model(input_tensor)
    elapsed_time = time.time() - start_time
    print(f"前向传播时间: {elapsed_time:.6f}秒")
    
    # 验证输出尺寸
    assert output.shape == (batch_size, num_classes, spatial_size, spatial_size, spatial_size), \
        f"输出尺寸错误，期望: {(batch_size, num_classes, spatial_size, spatial_size, spatial_size)}, 实际: {output.shape}"
    
    # 打印测试结果
    print("\n测试通过！")
    print("输入尺寸:", input_tensor.shape)
    print("输出尺寸:", output.shape)
    print("设备信息:", device)
    print("最后一层权重范数:", torch.norm(model.outc.weight).item())
    
if __name__ == '__main__':
    test_unet()