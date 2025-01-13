# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/01/06 17:13:04
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 基础卷积模块
*      VERSION: v1.0
=================================================
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class RDABlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_dw=False):
        super(RDABlock, self).__init__()
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
        
        self.atten = EMA3D(out_channels)
        
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
    
class DownSample(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_sample = nn.MaxPool3d(kernel_size=4, stride=2, padding=1)
    def forward(self, x):
        return self.down_sample(x)

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, trilinear=False):
        super().__init__()
        if trilinear:
            self.up_sample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear'),
                nn.Conv3d(in_channels, out_channels, kernel_size=1)
                )
        else:
            self.up_sample = nn.Sequential(
                nn.ConvTranspose3d(in_channels=in_channels, out_channels=in_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(in_channels),
                nn.ReLU(),
                nn.Conv3d(in_channels, out_channels, kernel_size=1)
                )
    def forward(self, x):
        return self.up_sample(x)
    
class ResAttCBR_3x3(nn.Module):
    def __init__(self, in_channels, out_channels, att=False):
        super().__init__()
        block_list = []
        block_list.append(CBR_Block_3x3(in_channels, out_channels))
        if att:
            block_list.append(CBAM(out_channels))
        block_list.append(CBR_Block_3x3(out_channels, out_channels))
        self.double_cbr = nn.Sequential(*block_list)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return F.relu(self.conv(x) + self.double_cbr(x))
    
class EncoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, dilation=False, cbam=False, residual=False):
        super().__init__()
        self.use_cbam = cbam
        self.residual = residual
        if dilation:
            self.residual = nn.Sequential(
                # nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1),
                # nn.BatchNorm3d(out_channels),
                # nn.ReLU(),
                nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2),
                nn.BatchNorm3d(in_channels),
                nn.ReLU(),
                nn.Conv3d(in_channels, out_channels, kernel_size=1) 
            )
        else:
            self.residual = nn.Sequential( 
                nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(),
                nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
                nn.BatchNorm3d(out_channels),
            )
        self.shortcut = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
        )
        self.relu = nn.ReLU()

        self.cbam = CBAM(in_channels, ratio=in_channels//2, kernel_size=3)

    def forward(self, x):
        if self.use_cbam:
            x = self.cbam(x)
        residual_out = self.residual(x)

        if self.residual:     
            shortcut_out = self.shortcut(x)
            out = self.relu(residual_out + shortcut_out)
        else:
            out = self.relu(residual_out)
        return out

class DecoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, dilation=False, cbam=False, residual=False):
        super().__init__()
        self.use_cbam = cbam
        self.use_residual = residual
        if dilation:
            self.residual = nn.Sequential(
                nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, dilation=1),
                nn.BatchNorm3d(in_channels),
                nn.ReLU(),
                nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2),
                nn.BatchNorm3d(in_channels),
                nn.ReLU(),
                # nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=3, dilation=3),
                # nn.BatchNorm3d(in_channels),
                nn.ReLU(),
                nn.Conv3d(in_channels, out_channels, kernel_size=1)
            )
        else:
            self.residual = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
                # nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(),
                nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
                nn.BatchNorm3d(out_channels),
            )
        
        self.shortcut = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
        )
        self.cbam = CBAM(in_channels, ratio=in_channels//2, kernel_size=3)

        self.relu = nn.ReLU()

    def forward(self, x, skipped=None):
        if skipped is not None:
            x = torch.cat([x, skipped], dim=1)
            if self.use_cbam:
                x = self.cbam(x)
        residual_out = self.residual(x)

        if self.use_residual:
            shortcut_out = self.shortcut(x)
            out = self.relu(residual_out + shortcut_out)
        else:
            out = self.relu(residual_out)

        return out

class Encoder(nn.Module):
    def __init__(self, in_channels=4, dilation_flags=[False, False, False, False]):
        super().__init__()
        self.DownSample = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encoder1 = EncoderBottleneck(in_channels, 32, 5, 2, 1, dilation_flags[0])     
        self.encoder2 = EncoderBottleneck(32, 64, 5, 2, 1, dilation_flags[1])
        self.encoder3 = EncoderBottleneck(64, 128, 3, 1, 1, dilation_flags[2])
        self.encoder4 = EncoderBottleneck(128, 256, 3, 1, 1, dilation_flags[3])
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out1 = self.encoder1(x)
        out = self.DownSample(out1)

        out2 = self.encoder2(out)
        out = self.DownSample(out2)

        out3 = self.encoder3(out)
        out = self.DownSample(out3)

        out4 = self.encoder4(out)
        out = self.DownSample(out4)
        
        out = self.dropout(out)

        skip_connections = [out1, out2, out3, out4] # 32x128 64x64 128x32 256x16

        return out, skip_connections

class Decoder(nn.Module):
    def __init__(self, out_channels, dilation_flags=[False, False, False, False], fusion=False):
        super().__init__()
        self.fusion = fusion

        self.upsample1 = nn.Sequential(
            nn.ConvTranspose3d(256, 256, kernel_size=4, stride=2, padding=1),
            CBR_Block_3x3(256, 256)
        )
        self.decoder1 = DecoderBottleneck(512, 128, 3, 1, 1, dilation_flags[0])

        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(128, 128, kernel_size=4, stride=2, padding=1),
            CBR_Block_3x3(128, 128)
            )
        self.decoder2 = DecoderBottleneck(256, 64, 3, 1, 1, dilation_flags[1])

        self.upsample3 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=4, stride=2, padding=1),
            CBR_Block_5x5(64, 64)
            )
        self.decoder3 = DecoderBottleneck(128, 32, 3, 1, 1, dilation_flags[2])

        self.upsample4 = nn.Sequential(
            nn.ConvTranspose3d(32, 32, kernel_size=4, stride=2, padding=1),
            CBR_Block_5x5(32, 32)
            )

        self.decoder4 = DecoderBottleneck(64, 32, 5, 2, 1, dilation_flags[3])
        
        self.FusionMagic = FusionMagic_v2(32, 32)
        self.dropout = nn.Dropout(p=0.2)
        self.out_conv = nn.Sequential(
            CBR_Block_3x3(32, 32),
            nn.Conv3d(32, out_channels, kernel_size=1)
            )

    def forward(self, x, skip_connections):
        
        skip_connections = skip_connections[::-1]

        out = self.upsample1(x) # 256x8 --> 256 x16
        out1 = self.decoder1(out, skip_connections[0]) # (256 + 256)x16 --> 128x32

        out = self.upsample2(out1) 
        out2 = self.decoder2(out, skip_connections[1]) # (128 + 128)x32 --> 64x64

        out = self.upsample3(out2)
        out3 = self.decoder3(out, skip_connections[2]) # (64 + 64)x64 --> 32x128

        if self.fusion:
            out3 = self.FusionMagic([out1, out2, out3])

        out = self.upsample4(out3)
        out4 = self.decoder4(out, skip_connections[3]) # (32 + 32)x128 --> 32x256
        
        out = self.out_conv(out4)

        return out

class Inception_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
            # nn.ReLU(),
        )
        self.branch2 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
        )
        self.branch3 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
            # nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, dilation=2, padding=2), # 膨胀卷积，膨胀率为2
            nn.BatchNorm3d(out_channels),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
        )
        self.branch5 = nn.Sequential(
            nn.AvgPool3d(kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
        )
        self.out_conv = nn.Conv3d(out_channels * 5, out_channels, kernel_size=1)

    def forward(self, x):
        out = torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x),
            self.branch5(x),
        ], dim=1)
        out = F.relu(self.out_conv(out))
        return out

class D_Inception_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )
        self.branch2 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
        )
        self.branch3 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, dilation=1, padding=1), # 膨胀卷积，膨胀率为2
            nn.BatchNorm3d(out_channels),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, dilation=2, padding=2), # 膨胀卷积，膨胀率为2
            nn.BatchNorm3d(out_channels),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, dilation=3, padding=3), # 膨胀卷积，膨胀率为2
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )
        self.branch4 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
        )
        self.branch5 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.AvgPool3d(kernel_size=3, stride=1, padding=1),
        )
        self.out_conv = nn.Conv3d(out_channels * 5, out_channels, kernel_size=1)

    def forward(self, x):
        out = torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x),
            self.branch5(x),
        ], dim=1)
        out = F.relu(self.out_conv(out))
        return out

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, reduction_ratio=4, num_hidden_layers=1):
        super(MLP, self).__init__()
        hidden_dim = in_channels // reduction_ratio
        hidden_layers_list = [nn.Conv3d(hidden_dim, hidden_dim, kernel_size=1), nn.ReLU()] * num_hidden_layers
        self.input_layer = nn.Conv3d(in_channels, hidden_dim, kernel_size=1)
        self.relu = nn.ReLU()
        self.hidden_layers = nn.Sequential(*hidden_layers_list)
        self.output_layer = nn.Conv3d(hidden_dim, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        x = self.relu(x)
        return x
    
class CAM(nn.Module):
    def __init__(self, in_dim, ratio=16):
        super(CAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = MLP(in_dim, in_dim, ratio)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avgout = self.fc(self.avg_pool(x))
        maxout = self.fc(self.max_pool(x))
        out = avgout + maxout
        return self.sigmoid(out)


class FusionMagic(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.2): # 32, 128
        super().__init__()
        # 分别对后两层的输入进行平均池化操作，得到每个通道的平均值
        self.avgpooloing = nn.AdaptiveAvgPool3d(1)
        self.maxpooling = nn.AdaptiveMaxPool3d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm1 = nn.LayerNorm([in_channels, 1, 1, 1])
        self.layer_norm2 = nn.LayerNorm([in_channels*2, 1, 1, 1])

        # 使用SE_Block进行将cat之后的特征进行压缩激发
        # self.SE_layer1 = SE_Block(in_channels*) # in_channels*6 = in)_channels*2 + in_channels*2*2
        self.layer_norm3 = nn.LayerNorm([in_channels*4, 1, 1, 1])

        self.layer_norm4 = nn.LayerNorm([in_channels*6, 1, 1, 1])

        self.layer_norm5 = nn.LayerNorm([in_channels*7, 1, 1, 1])

        self.MLP = nn.Sequential(
            nn.Conv3d(in_channels=in_channels*7, out_channels=in_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(in_channels=in_channels, out_channels=in_channels*7, kernel_size=1),
            nn.ReLU()
        )
        self.Conv1 = nn.Conv3d(in_channels*7, out_channels, kernel_size=1)

        self.sigmoid = nn.Sigmoid()
        
    def forward(self, inputs:list[torch.tensor]):
        # 对后两层的输入进行平均池化操作，得到每个通道的平均值
        x1 = self.avgpooloing(inputs[-1])
        x1 = self.maxpooling(x1)
        
        x1 = self.layer_norm1(x1)

        x2 = self.avgpooloing(inputs[-2])
        x2 = self.maxpooling(x2)
        x2 = self.layer_norm2(x2)

        x3 = self.avgpooloing(inputs[0])
        x3 = self.maxpooling(x3)
        x3 = self.layer_norm3(x3)

        out = torch.cat([x2, x3], dim=1)
        out = self.avgpooloing(out)
        out = self.maxpooling(out)
        out = self.layer_norm4(out)

        out = torch.cat([x1, out], dim=1)
        out = self.avgpooloing(out)
        out = self.maxpooling(out)
        out = self.layer_norm5(out)

        # out = self.dropout(out)

        out = self.MLP(out)
        out = self.Conv1(out)
        # out = self.SE_layer1(out)

        out = self.sigmoid(out)

        return out

class FusionMagic_v2(nn.Module):
    
    def __init__(self, in_channels, out_channels, dropout=0.2): # 输入128， 输出32
        super(FusionMagic_v2, self).__init__()
        # 分别对后两层的输入进行平均池化操作，得到每个通道的平均值
        self.avgpooloing = nn.AdaptiveAvgPool3d(1)
        self.maxpooling = nn.AdaptiveMaxPool3d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.upsample_1 = nn.Upsample(scale_factor=4, mode='trilinear')
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='trilinear')
        # 使用SE_Block进行将cat之后的特征进行压缩激发
        # self.SE_layer1 = SE_Block(in_channels*) # in_channels*6 = in)_channels*2 + in_channels*2*2

        self.cam_128 = CAM(in_channels*4)

        self.cam_64 = CAM(in_channels*2)

        self.cam_32 = CAM(in_channels)

        self.cam_out = CAM(in_channels)

        self.cam_192 = CAM(in_channels*6)

        self.mlp_192 = MLP(in_channels=in_channels*6, out_channels=in_channels*4, reduction_ratio=in_channels*6//2)
        self.layer_norm1 = nn.LayerNorm(normalized_shape=(in_channels*4, 1, 1, 1))

        self.mlp_96 = MLP(in_channels=in_channels*3, out_channels=in_channels*2, reduction_ratio=in_channels*3//2)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=(in_channels*2, 1, 1, 1))

        # self.mlp_288 = MLP(in_channels=in_channels*9, out_channels=in_channels*9)
        # self.layer_norm3 = nn.LayerNorm(normalized_shape=(in_channels*9, 1, 1, 1))

        self.cbr = nn.Sequential(
            CBR_Block_3x3(in_channels=in_channels*6, out_channels=in_channels*6),
            CBR_Block_3x3(in_channels=in_channels*6, out_channels=in_channels)
        )
        self.out_conv = nn.Conv3d(in_channels*2, in_channels, kernel_size=1)

        self.sigmoid = nn.Sigmoid()
        
    def forward(self, inputs:list[torch.tensor], pooling_type='all'):
        # 对后两层的输入进行平均池化操作，得到每个通道的平均值
        if pooling_type == 'all':
            x1 = self.cam_128(inputs[0])
            x2 = self.cam_64(inputs[1])
            x3 = self.cam_32(inputs[2])
        elif pooling_type == 'max':
            x1 = self.maxpooling(inputs[0])
            x2 = self.maxpooling(inputs[1])
            x3 = self.maxpooling(inputs[2])
        elif pooling_type == 'avg':
            x1 = self.avgpooloing(inputs[0])
            x2 = self.avgpooloing(inputs[1])
            x3 = self.avgpooloing(inputs[2])
        else:
            raise ValueError("Invalid pooling type")
        
        out1 = torch.cat([x1, x2], dim=1) # 128 + 64 = 192  # TODO: 下一步可以在cat之后, 进行切分成三个特征层，然后仿照qkv的方式，设计级联结构，然后再进行融合
        out1 = self.mlp_192(out1)
        out1 = self.layer_norm1(out1)
        out1 = self.sigmoid(out1)
        out1 = out1.expand_as(inputs[0]) * inputs[0]

        out2 = torch.cat([x2, x3], dim=1) # 64 + 32 = 96
        out2 = self.mlp_96(out2)
        out2 = self.layer_norm2(out2)
        out2 = self.sigmoid(out2)
        out2 = out2.expand_as(inputs[1]) * inputs[1]

        out1 = self.upsample_1(out1) # 128 x 64
        out2 = self.upsample_2(out2) # 64 x 64

        out = torch.cat([out1, out2], dim=1) # 196
        out = self.cbr(out)
        # out = self.cam_out(out)
        out = out.expand_as(inputs[2]) * inputs[2] + inputs[2]
        # out = self.out_conv(F.relu(out))
        out = F.relu(out) 

        return out
    

class FusionMagic_v3(nn.Module):
    
    def __init__(self, in_channels, out_channels, dropout=0.2): # 输入128， 输出32
        super(FusionMagic_v2, self).__init__()
        # 分别对后两层的输入进行平均池化操作，得到每个通道的平均值
        self.avgpooloing = nn.AdaptiveAvgPool3d(1)
        self.maxpooling = nn.AdaptiveMaxPool3d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.upsample_1 = nn.Upsample(scale_factor=4, mode='trilinear')
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='trilinear')
        # 使用SE_Block进行将cat之后的特征进行压缩激发
        # self.SE_layer1 = SE_Block(in_channels*) # in_channels*6 = in)_channels*2 + in_channels*2*2

        self.cam_128 = CBAM(in_channels*4, reduction_ratio=in_channels*2, kernel_size=3)

        self.cam_64 = CBAM(in_channels*2, reducetion_ratio=in_channels, kernel_size=3)

        self.cam_32 = CBAM(in_channels, reduction_ratio=in_channels//2, kernel_size=3)

        self.cam_out = CBAM(in_channels, reduction_ratio=in_channels//2, kernel_size=3)

        self.cam_192 = CBAM(in_channels*6, reduction_ratio=in_channels*3, kernel_size=3)

        self.mlp_192 = MLP(in_channels=in_channels*6, out_channels=in_channels*4, reduction_ratio=in_channels*6//2)
        self.layer_norm1 = nn.LayerNorm(normalized_shape=(in_channels*4, 1, 1, 1))

        self.mlp_96 = MLP(in_channels=in_channels*3, out_channels=in_channels*2, reduction_ratio=in_channels*3//2)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=(in_channels*2, 1, 1, 1))

        # self.mlp_288 = MLP(in_channels=in_channels*9, out_channels=in_channels*9)
        # self.layer_norm3 = nn.LayerNorm(normalized_shape=(in_channels*9, 1, 1, 1))

        self.cbr = nn.Sequential(
            CBR_Block_3x3(in_channels=in_channels*6, out_channels=in_channels*6),
            CBR_Block_3x3(in_channels=in_channels*6, out_channels=in_channels)
        )
            

        self.sigmoid = nn.Sigmoid()
        
    def forward(self, inputs:list[torch.tensor], pooling_type='all'):
        # 对后两层的输入进行平均池化操作，得到每个通道的平均值
        if pooling_type == 'all':
            x1 = self.cam_128(inputs[0])
            x2 = self.cam_64(inputs[1])
            x3 = self.cam_32(inputs[2])
        elif pooling_type == 'max':
            x1 = self.maxpooling(inputs[0])
            x2 = self.maxpooling(inputs[1])
            x3 = self.maxpooling(inputs[2])
        elif pooling_type == 'avg':
            x1 = self.avgpooloing(inputs[0])
            x2 = self.avgpooloing(inputs[1])
            x3 = self.avgpooloing(inputs[2])
        else:
            raise ValueError("Invalid pooling type")
        
        out1 = torch.cat([x1, x2], dim=1) # 128 + 64 = 192  # TODO: 下一步可以在cat之后, 进行切分成三个特征层，然后仿照qkv的方式，设计级联结构，然后再进行融合
        out1 = self.mlp_192(out1)
        out1 = self.layer_norm1(out1)
        out1 = self.sigmoid(out1)
        out1 = out1.expand_as(inputs[0]) * inputs[0]

        out2 = torch.cat([x2, x3], dim=1) # 64 + 32 = 96
        out2 = self.mlp_96(out2)
        out2 = self.layer_norm2(out2)
        out2 = self.sigmoid(out2)
        out2 = out2.expand_as(inputs[1]) * inputs[1]

        out1 = self.upsample_1(out1) # 128 x 64
        out2 = self.upsample_2(out2) # 64 x 64

        out = torch.cat([out1, out2], dim=1) # 196
        out = self.cbr(out)
        out = self.cam_out(out)
        out = out.expand_as(inputs[2]) * inputs[2]
        return out
    
# class Adaptive_FusionMagic_v2(nn.Module):
#     def __init__(self, features, dropout=0.2): # 三输入为[128, 64, 32]; 四输入为 [256, 128, 64, 32]
#         super(FusionMagic_v2, self).__init__()
#         # 分别对后两层的输入进行平均池化操作，得到每个通道的平均值
#         self.avgpooloing = nn.AdaptiveAvgPool3d(1)
#         self.maxpooling = nn.AdaptiveMaxPool3d(1)
#         self.dropout = nn.Dropout(p=dropout)
#         self.upsample_1 = nn.Upsample(scale_factor=4, mode='trilinear')
#         self.upsample_2 = nn.Upsample(scale_factor=2, mode='trilinear')
#         # 使用SE_Block进行将cat之后的特征进行压缩激发
#         # self.SE_layer1 = SE_Block(in_channels*) # in_channels*6 = in)_channels*2 + in_channels*2*2


#         self.cams = [CAM(feature) for feature in features]
#         if len(features) == 3:
#             self.cams.append(CAM(features[-1]))
#             self.cams.append(CAM(features[-1] + features[0]))

#         self.mlp_192 = MLP(in_channels=in_channels*6, out_channels=in_channels*4)
#         self.layer_norm1 = nn.LayerNorm(normalized_shape=(in_channels*4, 1, 1, 1))

#         self.mlp_96 = MLP(in_channels=in_channels*3, out_channels=in_channels*2)
#         self.layer_norm2 = nn.LayerNorm(normalized_shape=(in_channels*2, 1, 1, 1))

#         # self.mlp_288 = MLP(in_channels=in_channels*9, out_channels=in_channels*9)
#         # self.layer_norm3 = nn.LayerNorm(normalized_shape=(in_channels*9, 1, 1, 1))

#         self.cbr = nn.Sequential(
#             CBR_Block_3x3(in_channels=in_channels*6, out_channels=in_channels*6),
#             CBR_Block_3x3(in_channels=in_channels*6, out_channels=in_channels)
#         )
            

#         self.sigmoid = nn.Sigmoid()
        
#     def forward(self, inputs:list[torch.tensor], pooling_type='all'):
#         # 对后两层的输入进行平均池化操作，得到每个通道的平均值
#         if pooling_type == 'all':
#             x1 = self.cam_128(inputs[0])
#             x2 = self.cam_64(inputs[1])
#             x3 = self.cam_32(inputs[2])
#         elif pooling_type == 'max':
#             x1 = self.maxpooling(inputs[0])
#             x2 = self.maxpooling(inputs[1])
#             x3 = self.maxpooling(inputs[2])
#         elif pooling_type == 'avg':
#             x1 = self.avgpooloing(inputs[0])
#             x2 = self.avgpooloing(inputs[1])
#             x3 = self.avgpooloing(inputs[2])
#         else:
#             raise ValueError("Invalid pooling type")
        
#         out1 = torch.cat([x1, x2], dim=1) # 128 + 64 = 192
#         out1 = self.mlp_192(out1)
#         out1 = self.layer_norm1(out1)
#         out1 = self.sigmoid(out1)
#         out1 = out1.expand_as(inputs[0]) * inputs[0]

#         out2 = torch.cat([x2, x3], dim=1) # 64 + 32 = 96
#         out2 = self.mlp_96(out2)
#         out2 = self.layer_norm2(out2)
#         out2 = self.sigmoid(out2)
#         out2 = out2.expand_as(inputs[1]) * inputs[1]

#         out1 = self.upsample_1(out1) # 128 x 64
#         out2 = self.upsample_2(out2) # 64 x 64

#         out = torch.cat([out1, out2], dim=1) # 196
#         out = self.cbr(out)
#         out = self.cam_out(out)
#         out = out.expand_as(inputs[2]) * inputs[2]
#         return out

if __name__ == '__main__':
    from Attentions import *
    
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FusionMagic(32, 128).to(device)
    inputs = [torch.randn(1, 32, 128, 128, 128).to(device), torch.randn(1, 64, 64, 64, 64).to(device), torch.randn(1, 128, 32, 32, 32).to(device)]
    # print(model)
    output = model(inputs)
    print(output.shape)

else:
    from .Attentions import *


        


        
        
        