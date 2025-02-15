# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/02/13 20:26:08
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 
*      VERSION: v1.0
*      FEATURES: 改进的UNet3D结构
=================================================
'''
# v1:计划基于Attention UNet3D改进，增加inception模块，改进注意力机制

from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchinfo import summary
import time


from collections.abc import Sequence
from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Norm


"""=================================== 初始化模块 ==========================================="""
def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1
                                     or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' %
                    init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm3d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
 
    print('initialize network with %s' % init_type)
    net.apply(init_func)
 
"""======================================= 注意力模块 ========================================"""
class CGGA(nn.Module):
    """Cross Dimension Attention(跨维度空间注意力机制)"""
    def __init__(self, channels, factor=8):
        super(CGGA, self).__init__()
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
        x_c_sigmoid = x_c.sigmoid().view(b*self.group, c // self.group, 1, 1, 1)
        x_h_sigmoid = x_h.sigmoid().view(b*self.group, c // self.group, d, 1, 1)
        x_w_sigmoid = x_w.sigmoid().view(b*self.group, c // self.group, 1, h, 1)
        x_d_sigmoid = x_d.sigmoid().view(b*self.group, c // self.group, 1, 1, w)
        
        # Apply attention maps using broadcasting
        x_attended = group_x * x_h_sigmoid * x_w_sigmoid * x_d_sigmoid * x_c_sigmoid
        
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
    

"""========================================== Inception Block ================================"""
class Inception_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Inception_Block, self).__init__()
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
        super(D_Inception_Block, self).__init__()
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

"""=========================================== Encoder Block =========================================="""
class ResBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock3D, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3,padding=1, bias=False),
            nn.BatchNorm3d(out_channels))
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
            
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        return F.relu(out)

"""============================================== BottleNeckBlock ============================================="""
class DenseASPP3D(nn.Module):
    def __init__(self, in_channels, out_channels, rates=[1, 2, 4, 8]):
        super(DenseASPP3D, self).__init__()
        self.aspp1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels//4, kernel_size=3, padding=rates[0], dilation=rates[0]),
            nn.BatchNorm3d(out_channels//4),
            nn.ReLU(inplace=True)
        )
        self.aspp2 = nn.Sequential(
            nn.Conv3d(in_channels+out_channels//4, out_channels//4, kernel_size=3, padding=rates[1], dilation=rates[1]),
            nn.BatchNorm3d(out_channels//4),
            nn.ReLU(inplace=True)
        )
        self.aspp3 = nn.Sequential(
            nn.Conv3d(in_channels + 2*(out_channels//4), out_channels//4, kernel_size=3, padding=rates[2], dilation=rates[2]),
            nn.BatchNorm3d(out_channels//4),
            nn.ReLU(inplace=True)
        )
        self.aspp4 = nn.Sequential(
            nn.Conv3d(in_channels + 3*(out_channels//4), out_channels//4, kernel_size=3, padding=rates[3], dilation=rates[3]),
            nn.BatchNorm3d(out_channels//4),
            nn.ReLU(inplace=True)
        )
        self.global_avg = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, out_channels//4, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.fusion = nn.Conv3d(5*(out_channels//4), out_channels, 1)
        
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
"""============================================ Decoder Block ============================================="""

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, convTranspose=True):
        super(DecoderBlock, self).__init__()
        if convTranspose:
            self.up = nn.ConvTranspose3d(in_channels=in_channels, out_channels=in_channels,kernel_size=2,stride=2)
        else:
            self.up = nn.Upsample(scale_factor=2)
 
        self.Conv = ResBlock3D(in_channels, out_channels)
 
    def forward(self, x):
        x = self.up(x)
        x = self.Conv(x)
        return x
 
 
# class single_conv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(single_conv, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv3d(in_channels,
#                       out_channels,
#                       kernel_size=3,
#                       stride=1,
#                       padding=1,
#                       bias=True), 
#             nn.BatchNorm3d(out_channels),
#             nn.ReLU(inplace=True))
 
#     def forward(self, x):
#         x = self.conv(x)
#         return x
 
""" ============================================== Attention Gate =========================================="""
class Attention_block(nn.Module):
 
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g,
                      F_int,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True), 
            nn.BatchNorm3d(F_int))
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l,
                      F_int,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True), 
            nn.BatchNorm3d(F_int))
 
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1), nn.Sigmoid())
 
        self.relu = nn.ReLU(inplace=True)
 
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class Proposed_UNet(nn.Module):
    """
    in_channel: input image channels
    num_classes: output class number 
    channel_list: a channel list for adjust the model size
    checkpoint: 是否有checkpoint  if False： call normal init
    convTranspose: 是否使用反卷积上采样。True: use nn.convTranspose  Flase: use nn.Upsample
    """
    def __init__(self,
                 in_channel=4,
                 num_classes=4,
                 channel_list=[64, 128, 256, 512, 1024],
                 checkpoint=False,
                 convTranspose=True):
        super(Proposed_UNet, self).__init__()
 
        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
 
        self.decoder1 = ResBlock3D(in_channels=in_channel, out_channels=channel_list[0])
        self.decoder2 = ResBlock3D(in_channels=channel_list[0], out_channels=channel_list[1])
        self.decoder3 = ResBlock3D(in_channels=channel_list[1], out_channels=channel_list[2])
        self.decoder4 = ResBlock3D(in_channels=channel_list[2], out_channels=channel_list[3])
        
        self.BottleNeck = DenseASPP3D(in_channels=channel_list[3], out_channels=channel_list[4])

        # self.bottle_neck = Inception_Block(channel_list[4], channel_list[4])  # TODO: 改进2：BottleNeck使用Inception Block
        self.Up5 = DecoderBlock(in_channels=channel_list[4], out_channels=channel_list[3], convTranspose=convTranspose)
        # self.skip4 = CGGA(channel_list[3])                                      # TODO: 改进1：跳跃连接使用Cross Dimension Attention
        self.Att5 = Attention_block(F_g=channel_list[3],
                                    F_l=channel_list[3],
                                    F_int=channel_list[2])
        self.DecoderBlock5 = ResBlock3D(in_channels=channel_list[4],
                                   out_channels=channel_list[3])
 
        self.Up4 = DecoderBlock(in_channels=channel_list[3], out_channels=channel_list[2], convTranspose=convTranspose)
        # self.skip3 = CGGA(channel_list[2])
        self.Att4 = Attention_block(F_g=channel_list[2],
                                    F_l=channel_list[2],
                                    F_int=channel_list[1])
        self.DecoderBlock4 = ResBlock3D(in_channels=channel_list[3],
                                   out_channels=channel_list[2])
 
        self.Up3 = DecoderBlock(in_channels=channel_list[2], out_channels=channel_list[1], convTranspose=convTranspose)
        # self.skip2 = CGGA(channel_list[1])
        self.Att3 = Attention_block(F_g=channel_list[1],
                                    F_l=channel_list[1],
                                    F_int=64)
        self.DecoderBlock3 = ResBlock3D(in_channels=channel_list[2],
                                   out_channels=channel_list[1])
 
        self.Up2 = DecoderBlock(in_channels=channel_list[1], out_channels=channel_list[0], convTranspose=convTranspose)
        # self.skip1 = CGGA(channel_list[0])
        self.Att2 = Attention_block(F_g=channel_list[0],
                                    F_l=channel_list[0],
                                    F_int=channel_list[0] // 2)
        self.DecoderBlock2 = ResBlock3D(in_channels=channel_list[1],
                                   out_channels=channel_list[0])
 
        self.Conv_1x1 = nn.Conv3d(channel_list[0],
                                  num_classes,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)
 
        if not checkpoint:
            init_weights(self)
 
    def forward(self, x):
        # encoder
        x1 = self.decoder1(x)
 
        x2 = self.Maxpool(x1)
        x2 = self.decoder2(x2)
 
        x3 = self.Maxpool(x2)
        x3 = self.decoder3(x3)
 
        x4 = self.Maxpool(x3)
        x4 = self.decoder4(x4)
 
        x5 = self.Maxpool(x4)
        
        x5 = self.BottleNeck(x5)
        # decoder
        d5 = self.Up5(x5)
        # x4 = self.skip4(x4)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.DecoderBlock5(d5)
 
        d4 = self.Up4(d5)
        # x3 = self.skip3(x3)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.DecoderBlock4(d4)
 
        d3 = self.Up3(d4)
        # x2 = self.skip2(x2)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.DecoderBlock3(d3)
 
        d2 = self.Up2(d3)
        # x1 = self.skip1(x1)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.DecoderBlock2(d2)
 
        d1 = self.Conv_1x1(d2)

        return d1

    
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
    model = Proposed_UNet(
        in_channel=in_channels,
        num_classes=num_classes,
        channel_list=[32, 64, 128, 256, 512]
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
    print("最后一层权重范数:", torch.norm(model.Conv_1x1.weight).item())
        
if __name__ == "__main__":
    test_unet()