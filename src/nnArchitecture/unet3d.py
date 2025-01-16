# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/01/06 16:25:05
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 3D U-Net Model
*      VERSION: v1.0
=================================================
'''
import sys
sys.path.append('/root/workspace/VoxelMedix/src/nnArchitecture')
from ast import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from _init_model import init_all_weights

"""----------------------- 参数打印函数 -----------------------"""
def print_model_summary(model, input_data, device="cuda"):

    # 使用 torchinfo 生成模型摘要
    # batch_size=1, 输入尺寸为 (128, 128, 128)，通道数为 4
    summary(
        model,
        input_data=input_data,
        col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
        col_width=20,
        row_settings=["var_names"],
        device=device
    )


"""---------------------------------------- UNet 基础块 ----------------------------------------"""
class ResCBRBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_dw=False):
        super(ResCBRBlock, self).__init__()
        self.use_dw = use_dw

        if use_dw:
            # Depthwise Separable Convolution
            self.conv1 = nn.Sequential(
                nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),  # Depthwise Convolution
                nn.Conv3d(in_channels, out_channels, kernel_size=1)  # Pointwise Convolution
            )
            self.conv2 = nn.Sequential(
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels),  # Depthwise Convolution
                nn.Conv3d(out_channels, out_channels, kernel_size=1)  # Pointwise Convolution
            )
        else:
            # Standard Convolution
            self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
            self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class ConvAttBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_dw=False):
        super(ConvAttBlock, self).__init__()
        self.use_dw = use_dw

        if use_dw:
            # Depthwise Separable Convolution
            self.conv1 = nn.Sequential(
                nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),  # Depthwise Convolution
                nn.Conv3d(in_channels, out_channels, kernel_size=1)  # Pointwise Convolution
            )
            self.conv2 = nn.Sequential(
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels),  # Depthwise Convolution
                nn.Conv3d(out_channels, out_channels, kernel_size=1)  # Pointwise Convolution
            )
        else:
            # Standard Convolution
            self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
            self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.atten = SE(out_channels)
        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.atten(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class CBR_Block_3x3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(CBR_Block_3x3, self).__init__()
        self.cbr_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.cbr_conv(x)
    
class CBR_Block_5x5(CBR_Block_3x3):
    def __init__(self, in_channels:int, out_channels:int):
        super(CBR_Block_5x5, self).__init__(in_channels, out_channels)
        self.conv[0] = nn.Conv3d(in_channels, out_channels, kernel_size=5, padding=2, dilation=1, bias=False)


class DoubleCBR_Block_3x3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_cbr = nn.Sequential(
            CBR_Block_3x3(in_channels, out_channels),
            CBR_Block_3x3(out_channels, out_channels),
        )

    def forward(self, x):
        return self.double_cbr(x)
    
class SE(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super(SE, self).__init__()

        if in_channels // reduction_ratio <= 0:
                    raise ValueError(f"Reduction ratio {reduction_ratio} is too large for the number of input channels {in_channels}.")
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y.expand_as(x)
    
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
        self.SE = SE(channels // self.group)

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
        x_attended = group_x * x_h_sigmoid * x_w_sigmoid * x_d_sigmoid
        
        x1 = self.groupNorm(group_x * x_attended)  # 高度、宽度、深度注意力
        x11 = self.softmax(self.averagePooling(x1).reshape(b * self.group, -1, 1).permute(0, 2, 1))  # 全局平均池化 + softmax
        x12 = x1.reshape(b * self.group, c // self.group, -1)

        # 3x3x3 路径
        x2 = self.SE(self.conv3x3x3(group_x))  # 通过 3x3x3 卷积层
        x21 = self.softmax(self.averagePooling(x2).reshape(b * self.group, -1, 1).permute(0, 2, 1))  # 全局平均池化 + softmax
        x22 = x2.reshape(b * self.group, c // self.group, -1)

        # 计算权重
        weights = (torch.matmul(x11, x22) + torch.matmul(x21, x12)).reshape(b * self.group, -1, d, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, d, h, w)
    
"""---------------------------------------- UNet3D 基座类 ----------------------------------------"""
class BaseUNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, channels_list, soft=False):
        super(BaseUNet3D, self).__init__()
        
        # Placeholder for specific block types
        # self.encoder = None
        # self.bottleneck_block = None
        # self.decoder = None
        if soft:
            self.Pool = SoftPool3D
        else:
            self.Pool = nn.MaxPool3d
            
        ch1, ch2, ch3, ch4 = channels_list

        
        # encodersample path
        self.encoder1 = self.encoder(in_channels, ch1)
        self.pool1 = self.Pool(kernel_size=2, stride=2)
        self.encoder2 = self.encoder(ch1, ch2)
        self.pool2 = self.Pool(kernel_size=2, stride=2)
        self.encoder3 = self.encoder(ch2, ch3)
        self.pool3 = self.Pool(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = self.bottleneck_block(ch3, ch4)

        # Upsample path
        self.up1 = nn.ConvTranspose3d(ch4, ch3, kernel_size=2, stride=2)
        self.decoder1 = self.decoder(ch3 * 2, ch3)
        self.up2 = nn.ConvTranspose3d(ch3, ch2, kernel_size=2, stride=2)
        self.decoder2 = self.decoder(ch2 * 2, ch2)
        self.up3 = nn.ConvTranspose3d(ch2, ch1, kernel_size=2, stride=2)
        self.decoder3 = self.decoder(ch1 * 2, ch1)

        # Final layer
        self.final_conv = nn.Conv3d(ch1, out_channels, kernel_size=1)
        
        self.softmax = nn.Softmax(dim=1)
        
        self.apply(init_all_weights)



    def forward(self, x):
        # encodersample
        d1 = self.encoder1(x)
        d2 = self.encoder2(self.pool1(d1))
        d3 = self.encoder3(self.pool2(d2))
        d4 = self.pool3(d3)

        # Bottleneck
        bottleneck = self.bottleneck(d4)

        # Upsample
        u1 = self.up1(bottleneck)
        u1 = torch.cat([u1, d3], dim=1)
        u1 = self.decoder1(u1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.decoder2(u2)

        u3 = self.up3(u2)
        u3 = torch.cat([u3, d1], dim=1)
        u3 = self.decoder3(u3)

        # Final layer
        out = self.final_conv(u3)
        
        # Gain the probability
        out = self.softmax(out)
        return out
    

class BaseDWUNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, channels_list, soft=False):
        super(BaseDWUNet3D, self).__init__()
        
        # Placeholder for specific block types
        # self.encoder = None
        # self.bottleneck_block = None
        # self.decoder = None
        ch1, ch2, ch3, ch4 = channels_list
        if soft:
            self.Pool = SoftPool3D
        else:
            self.Pool = nn.MaxPool3d

        # encodersample path
        self.encoder1 = self.encoder(in_channels, ch1, use_dw=True)
        self.pool1 = self.Pool(kernel_size=2, stride=2)
        self.encoder2 = self.encoder(ch1, ch2, use_dw=True)
        self.pool2 = self.Pool(kernel_size=2, stride=2)
        self.encoder3 = self.encoder(ch2, ch3, use_dw=True)
        self.pool3 = self.Pool(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = self.bottleneck_block(ch3, ch4)

        # Upsample path
        self.up1 = nn.ConvTranspose3d(ch4, ch3, kernel_size=2, stride=2)
        self.decoder1 = self.decoder(ch3 * 2, ch3, use_dw=True)
        self.up2 = nn.ConvTranspose3d(ch3, ch2, kernel_size=2, stride=2)
        self.decoder2 = self.decoder(ch2 * 2, ch2, use_dw=True)
        self.up3 = nn.ConvTranspose3d(ch2, ch1, kernel_size=2, stride=2)
        self.decoder3 = self.decoder(ch1 * 2, ch1, use_dw=True)

        # Final layer
        self.final_conv = nn.Conv3d(ch1, out_channels, kernel_size=1)
        
        self.softmax = nn.Softmax(dim=1)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # encodersample
        d1 = self.encoder1(x)
        d2 = self.encoder2(self.pool1(d1))
        d3 = self.encoder3(self.pool2(d2))
        d4 = self.pool3(d3)

        # Bottleneck
        bottleneck = self.bottleneck(d4)

        # Upsample
        u1 = self.up1(bottleneck)
        u1 = torch.cat([u1, d3], dim=1)
        u1 = self.decoder1(u1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.decoder2(u2)

        u3 = self.up3(u2)
        u3 = torch.cat([u3, d1], dim=1)
        u3 = self.decoder3(u3)

        # Final layer
        out = self.final_conv(u3)
        
        # Gain the probability
        out = self.softmax(out)
        return out

"""---------------------------------------- UNet3D 模型 ----------------------------------------"""
# Specific UNet3D implementations
class UNet3D(BaseUNet3D):
    def __init__(
        self, 
        in_channels:int, 
        out_channels:int, 
        channels_list:List=None
        ):
        if channels_list is None:
            channels_list = [32, 64, 128, 256]
        self.encoder = DoubleCBR_Block_3x3
        self.bottleneck_block = DoubleCBR_Block_3x3
        self.decoder = DoubleCBR_Block_3x3
        super(UNet3D, self).__init__(in_channels, out_channels, channels_list)

class soft_UNet3D(BaseUNet3D):
    def __init__(
        self, 
        in_channels:int, 
        out_channels:int, 
        channels_list:List=None, 
        soft:bool=True):
        if channels_list is None:
            channels_list = [32, 64, 128, 256]
        self.encoder = DoubleCBR_Block_3x3
        self.bottleneck_block = DoubleCBR_Block_3x3
        self.decoder = DoubleCBR_Block_3x3
        super(soft_UNet3D, self).__init__(in_channels, out_channels, channels_list, soft=True)
        
"""---------------------------------------- DCAI_UNet3D 模型 ----------------------------------------"""
#? A: Attention Layer
#? DC: Dilated Convolution Layer
#? I: Inception Layer
class CAD_UNet3D(BaseUNet3D):
    def __init__(
        self, 
        in_channels:int, 
        out_channels:int, 
        channels_list:List=None
        ):
        if channels_list is None:
            channels_list = [32, 64, 128, 256]
        self.encoder = ConvAttBlock
        self.bottleneck_block = DoubleCBR_Block_3x3
        self.decoder = ConvAttBlock
        super(CAD_UNet3D, self).__init__(in_channels, out_channels, channels_list)

class soft_CAD_UNet3D(BaseUNet3D):
    def __init__(self, in_channels, out_channels, channels_list=None, soft=True):
        if channels_list is None:
            channels_list = [32, 64, 128, 256]
        self.encoder = ConvAttBlock
        self.bottleneck_block = DoubleCBR_Block_3x3
        self.decoder = ConvAttBlock
        super(soft_CAD_UNet3D, self).__init__(in_channels, out_channels, channels_list, soft=True)
                
"""---------------------------------------- DCAI_UNet3D 模型 ----------------------------------------"""
#? A: Attention Layer
#? DC: Dilated Convolution Layer
#? I: Inception Layer
class CADI_UNet3D(BaseUNet3D):
    def __init__(self, in_channels, out_channels, channels_list=None):
        if channels_list is None:
            channels_list = [32, 64, 128, 256]
        self.encoder = ConvAttBlock
        self.bottleneck_block = D_Inception_Block
        self.decoder = ConvAttBlock
        super(CADI_UNet3D, self).__init__(in_channels, out_channels, channels_list)

class soft_CADI_UNet3D(BaseUNet3D):
    def __init__(self, in_channels, out_channels, channels_list=None, soft=True):
        if channels_list is None:
            channels_list = [32, 64, 128, 256]
        self.encoder = ConvAttBlock
        self.bottleneck_block = D_Inception_Block
        self.decoder = ConvAttBlock
        super(soft_CADI_UNet3D, self).__init__(in_channels, out_channels, channels_list, soft=True)

"""---------------------------------------- DW_UNet3D 模型 ----------------------------------------"""
#! 其效果较慢，前几个epoch，会出现梯度不稳定的情况
class DW_UNet3D(BaseDWUNet3D):
    def __init__(self, in_channels, out_channels, channels_list=None):
        if channels_list is None:
            channels_list = [32, 64, 128, 256]
        self.encoder = ConvAttBlock
        self.bottleneck_block = D_Inception_Block
        self.decoder = ConvAttBlock
        super(DW_UNet3D, self).__init__(in_channels, out_channels, channels_list)

class soft_DW_UNet3D(BaseDWUNet3D):
    def __init__(self, in_channels, out_channels, channels_list=None, soft=True):
        if channels_list is None:
            channels_list = [32, 64, 128, 256]

        self.encoder = ConvAttBlock
        self.bottleneck_block = D_Inception_Block
        self.decoder = ConvAttBlock
        super(soft_DW_UNet3D, self).__init__(in_channels, out_channels, channels_list, soft=True)

# Example usage
if __name__ == '__main__':
    from modules.BasicBlock import *
    from modules.SoftPooling import SoftPool3D
    from ptflops import get_model_complexity_info
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # input_data = torch.randn(1, 4, 128, 128, 128).to(device)
    # # model = UNet3D(in_channels=4, out_channels=4, channels_list=[32, 64, 128, 256]).to(device)
    # model = soft_UNet3D(in_channels=4, out_channels=4).to(device)
    
    # output = model(input_data)
    # print(output.shape)
    # # Print model summary
    # print_model_summary(model, input_data=input_data, device=device)
    
    
    # 假设你的 UNet 模型类名为 UNet3D

    # 定义输入张量的形状 (batch_size, channels, depth, height, width)
    input_shape = (4, 128, 128, 128)  # 替换为你的输入形状

    # 创建模型实例
    model = UNet3D(in_channels=4, out_channels=2, channels_list=[32, 64, 128, 256])

    # 使用 ptflops 计算 FLOPs 和参数量
    macs, params = get_model_complexity_info(
        model, 
        input_shape, 
        as_strings=True, 
        print_per_layer_stat=True, 
        verbose=True
    )

    print(f"Computational complexity: {macs}")
    print(f"Number of parameters: {params}")
    
else:
    from .modules.BasicBlock import *
    from .modules.SoftPooling import SoftPool3D