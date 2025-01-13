# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/01/06 16:25:05
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 3D U-Net Model
*      VERSION: v1.0
=================================================
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, in_channels, out_channels, channels_list=None):
        if channels_list is None:
            channels_list = [32, 64, 128, 256]
        self.encoder = DoubleCBR_Block_3x3
        self.bottleneck_block = DoubleCBR_Block_3x3
        self.decoder = DoubleCBR_Block_3x3
        super(UNet3D, self).__init__(in_channels, out_channels, channels_list)

class soft_UNet3D(BaseUNet3D):
    def __init__(self, in_channels, out_channels, channels_list=None, soft=True):
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
    def __init__(self, in_channels, out_channels, channels_list=None):
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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_data = torch.randn(1, 4, 128, 128, 128).to(device)
    # model = UNet3D(in_channels=4, out_channels=4, channels_list=[32, 64, 128, 256]).to(device)
    model = soft_UNet3D(in_channels=4, out_channels=4).to(device)
    
    output = model(input_data)
    print(output.shape)
    # Print model summary
    print_model_summary(model, input_data=input_data, device=device)
    
    
else:
    from .modules.BasicBlock import *
    from .modules.SoftPooling import SoftPool3D