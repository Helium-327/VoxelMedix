# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/02/15 11:21:32
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 
*      VERSION: v1.0
*      FEATURES: UNet3D 复现网络
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

class DoubleConv3D(nn.Module):
    """(conv3D -> BN -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down3D(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up3D(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        
        self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, f_list=[32, 64, 128, 256], trilinear=True):
        super(UNet3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.inc = DoubleConv3D(in_channels, f_list[0])  # 4 --> 32
        self.down1 = Down3D(f_list[0], f_list[1])        # 32 --> 64
        self.down2 = Down3D(f_list[1], f_list[2])        # 64 --> 128
        self.down3 = Down3D(f_list[2], f_list[3])        # 128 --> 256
        self.down4 = Down3D(f_list[3], f_list[3])        # 256 --> 256
        
        self.up1 = Up3D(f_list[3]*2, f_list[2], trilinear)  # 512 --> 128
        self.up2 = Up3D(f_list[3], f_list[1], trilinear)    # 256 --> 64
        self.up3 = Up3D(f_list[2], f_list[0], trilinear)    # 128 --> 32
        self.up4 = Up3D(f_list[1], f_list[0], trilinear)    # 64 --> 32
        self.outc = nn.Conv3d(f_list[0], out_channels, kernel_size=1)  # 32 --> 4
        
        self.apply(init_weights_3d) # 初始化权重

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
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
    model = UNet3D(
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
    
if __name__ == "__main__":
    test_unet()