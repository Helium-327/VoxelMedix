# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/02/15 15:28:45
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 
*      VERSION: v1.0
*      FEATURES: Attention Unet 3D + ResConv + DenseASPP(BottleNeck) + AnisotropicConv3D + OptimizedAttentionGate
=================================================
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchinfo import summary
import time

def init_weights_3d(m):
    """Initialize 3D卷积和BN层的权重"""
    if isinstance(m, nn.Conv3d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm3d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.ConvTranspose3d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

"""================================== 改进模块 =================================================="""
# 改进1：DoubleConv --> ResConv3D
# 改进2：BottleNeck --> DenseASPP
# 改进3：Conv3D --> AnisotropicConv3D
# 改进4：AttentionGate --> DynamicConvAttention3D

class AnisotropicConv3D(nn.Module):
    """各向异性3D卷积 (kernel_size分解为空间和深度卷积)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.spatial_conv = nn.Conv3d(in_channels, out_channels, kernel_size=(1,3,3), padding=(0,1,1))
        self.depth_conv = nn.Conv3d(out_channels, out_channels, kernel_size=(3,1,1), padding=(1,0,0))

    def forward(self, x):
        x = self.spatial_conv(x)
        out = self.depth_conv(x)
        return out

class ResConv3D(nn.Module):
    """带残差连接的各向异性卷积块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            AnisotropicConv3D(in_channels, out_channels),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            AnisotropicConv3D(out_channels, out_channels),
            nn.BatchNorm3d(out_channels)
        )
        self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.conv(x)
        if self.shortcut:
            residual = self.shortcut(residual)
        out += residual
        return self.relu(out)

class DenseASPP3D(nn.Module):
    def __init__(self, in_channels, out_channels, reduce_rate=2, dilations=[1, 2, 4, 8]):
        super(DenseASPP3D, self).__init__()
        self.aspp1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels//reduce_rate, kernel_size=3, padding=dilations[0], dilation=dilations[0]),
            nn.BatchNorm3d(out_channels//reduce_rate),
            nn.ReLU(inplace=True)
        )
        self.aspp2 = nn.Sequential(
            nn.Conv3d(in_channels + out_channels//reduce_rate, out_channels//reduce_rate, kernel_size=3, padding=dilations[1], dilation=dilations[1]),
            nn.BatchNorm3d(out_channels//reduce_rate),
            nn.ReLU(inplace=True)
        )
        self.aspp3 = nn.Sequential(
            nn.Conv3d(in_channels + 2*(out_channels//reduce_rate), out_channels//reduce_rate, kernel_size=3, padding=dilations[2], dilation=dilations[2]),
            nn.BatchNorm3d(out_channels//reduce_rate),
            nn.ReLU(inplace=True)
        )
        self.aspp4 = nn.Sequential(
            nn.Conv3d(in_channels + 3*(out_channels//reduce_rate), out_channels//reduce_rate, kernel_size=3, padding=dilations[3], dilation=dilations[3]),
            nn.BatchNorm3d(out_channels//reduce_rate),
            nn.ReLU(inplace=True)
        )
        self.global_avg = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, out_channels//reduce_rate, kernel_size=1),
            nn.ReLU(inplace=True)
        ) 
        self.fusion = nn.Conv3d(5*(out_channels//reduce_rate), out_channels, 1)
        
    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(torch.cat([x, x1], 1))
        x3 = self.aspp3(torch.cat([x, x1, x2], 1))
        x4 = self.aspp4(torch.cat([x, x1, x2, x3], 1))
        x5 = self.global_avg(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='trilinear', align_corners=True)
        x = torch.cat([x1, x2, x3, x4, x5], 1)
        x = self.fusion(x)
        return x

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
"""====================================== 原网络 ================================================= """

# class AttentionBlock3D(nn.Module):
#     """3D Attention Gate"""
#     def __init__(self, F_g, F_l, F_int):
#         super(AttentionBlock3D, self).__init__()
#         self.W_g = nn.Sequential(
#             nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm3d(F_int)
#         )
        
#         self.W_x = nn.Sequential(
#             nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm3d(F_int)
#         )
        
#         self.psi = nn.Sequential(
#             nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm3d(1),
#             nn.Sigmoid()
#         )
        
#         self.relu = nn.ReLU(inplace=True)
        
#     def forward(self, g, x):
#         g1 = self.W_g(g)
#         x1 = self.W_x(x)
#         psi = self.relu(g1 + x1)
#         psi = self.psi(psi)
#         return x * psi

class UpSample(nn.Module):
    """3D Up Convolution"""
    def __init__(self, in_channels, out_channels, trilinear=True):
        super(UpSample, self).__init__()
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        
        # self.conv = ResConv3D(in_channels, out_channels)

    def forward(self, x):
        return self.up(x)

class DyAnisoDasppResAtteUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, f_list=[32, 64, 128, 256], trilinear=True):
        super(DyAnisoDasppResAtteUNet, self).__init__()
        
        self.MaxPool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.Conv1 = ResConv3D(in_channels, f_list[0])
        self.Conv2 = ResConv3D(f_list[0], f_list[1])
        self.Conv3 = ResConv3D(f_list[1], f_list[2])
        self.Conv4 = ResConv3D(f_list[2], f_list[3])
        
        self.bottleneck = DenseASPP3D(f_list[3], f_list[3])
        
        self.Up5 = UpSample(f_list[3], f_list[3], trilinear)
        self.Att5 = DynamicConvAttention3D(F_g=f_list[3], F_l=f_list[3], F_int=f_list[3]//2)
        self.UpConv5 = ResConv3D(f_list[3]*2, f_list[3]//2)
        
        self.Up4 = UpSample(f_list[2], f_list[2], trilinear)
        self.Att4 = DynamicConvAttention3D(F_g=f_list[2], F_l=f_list[2], F_int=f_list[2]//2)
        self.UpConv4 = ResConv3D(f_list[2]*2, f_list[2]//2)
        
        self.Up3 = UpSample(f_list[1], f_list[1], trilinear)
        self.Att3 = DynamicConvAttention3D(F_g=f_list[1], F_l=f_list[1], F_int=f_list[1]//2)
        self.UpConv3 = ResConv3D(f_list[1]*2, f_list[1]//2)
        
        self.Up2 = UpSample(f_list[0], f_list[0], trilinear)
        self.Att2 = DynamicConvAttention3D(F_g=f_list[0], F_l=f_list[0], F_int=f_list[0]//2)
        self.UpConv2 = ResConv3D(f_list[0]*2, f_list[0])
        
        self.outc = nn.Conv3d(f_list[0], out_channels, kernel_size=1)

        self.apply(init_weights_3d)
        
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
    model = DyAnisoDasppResAtteUNet(
        in_channels=in_channels,
        out_channels=num_classes,
        # channel_list=[32, 64, 128, 256, 512]
    ).to(device)
    
    # 检查权重初始化是否正确
    for name, param in model.named_parameters():
        if 'weight' in name and 'conv' in name:
            print(f"{name}: mean={param.data.mean():.4f}, std={param.data.std():.4f}")
        elif 'bias' in name:
            print(f"{name}: value={param.data[:2]} (should be 0)")
        
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
    # dynamicConvAttention3D = DynamicConvAttention3D(64, 64, 32)
    # print(dynamicConvAttention3D)