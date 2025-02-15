import torch
import torch.nn as nn
import torch.nn.functional as F

class DeformableConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # 主卷积权重
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size, kernel_size)
        )
        
        # 偏移量生成网络
        self.offset_conv = nn.Conv3d(in_channels,
                                    3 * kernel_size**3,  # 每个采样点xyz三个方向的偏移
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding)
        
        # 初始化参数
        nn.init.kaiming_uniform_(self.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)

    def forward(self, x):
        # 生成偏移量 [B, 3*K^3, D, H, W]
        offset = self.offset_conv(x)
        
        # 应用可变形卷积
        return deform_conv3d(x, offset, self.weight, 
                            stride=self.stride,
                            padding=self.padding)

def deform_conv3d(input, offset, weight, stride=1, padding=1):
    # 自定义3D可变形卷积实现
    N, C, D, H, W = input.size()
    K = weight.size(2)
    
    # 生成采样网格
    grid = F.affine_grid(torch.eye(3,4).unsqueeze(0).to(input.device), 
                        input.size(), align_corners=False)
    
    # 计算偏移后的采样位置
    offset = offset.view(N, 3, K, K, K, D, H, W).permute(0,1,5,6,7,2,3,4)
    offset = offset.reshape(N, 3, D*K, H*K, W*K)
    grid = grid + 2 * offset.permute(0,2,3,4,1) / torch.tensor([D, H, W], device=input.device)
    
    # 三线性插值采样
    sampled = F.grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    
    # 滑动窗口提取
    unfolded = F.unfold(sampled, kernel_size=K, dilation=stride, padding=padding)
    unfolded = unfolded.view(N, C, K*K*K, -1)
    weight = weight.view(weight.size(0), -1)
    
    # 矩阵乘法实现卷积
    output = torch.einsum('bckn,ok->bcon', unfolded, weight)
    output = output.view(N, -1, D, H, W)
    return output

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None, deformable=False):
        super().__init__()
        mid_channels = mid_channels or out_channels
        
        if deformable:
            self.conv1 = DeformableConv3d(in_channels, mid_channels, 3)
        else:
            self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1)
        
        self.block = nn.Sequential(
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        return self.block(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, init_features=32, deform_layers=[3,4]):
        super().__init__()
        features = init_features
        
        # 编码器
        self.encoder1 = UNet3D._block(in_channels, features, "enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = UNet3D._block(features, features*2, "enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = UNet3D._block(features*2, features*4, "enc3", deformable=3 in deform_layers)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = UNet3D._block(features*4, features*8, "enc4", deformable=4 in deform_layers)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        # 瓶颈层
        self.bottleneck = UNet3D._block(features*8, features*16, "bottleneck")

        # 解码器
        self.upconv4 = nn.ConvTranspose3d(features*16, features*8, kernel_size=2, stride=2)
        self.decoder4 = UNet3D._block(features*16, features*8, "dec4")
        self.upconv3 = nn.ConvTranspose3d(features*8, features*4, kernel_size=2, stride=2)
        self.decoder3 = UNet3D._block(features*8, features*4, "dec3")
        self.upconv2 = nn.ConvTranspose3d(features*4, features*2, kernel_size=2, stride=2)
        self.decoder2 = UNet3D._block(features*4, features*2, "dec2")
        self.upconv1 = nn.ConvTranspose3d(features*2, features, kernel_size=2, stride=2)
        self.decoder1 = UNet3D._block(features*2, features, "dec1")

        # 最终卷积
        self.conv = nn.Conv3d(features, out_channels, kernel_size=1)

    @staticmethod
    def _block(in_channels, features, name=None, deformable=False):
        return DoubleConv(in_channels, features, deformable=deformable)

    def forward(self, x):
        # 编码器
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        # 瓶颈层
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # 解码器
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        return self.conv(dec1)



class AdaptiveDeformConv3d(DeformableConv3d):
    """自适应调节偏移量强度"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = nn.Parameter(torch.tensor(0.1))  # 可学习缩放系数

    def forward(self, x):
        offset = super().offset_conv(x)
        return deform_conv3d(x, self.alpha * offset, self.weight, 
                            self.stride, self.padding)


class MultiScaleDeformBlock(nn.Module):
    """多尺度可变形卷积融合"""
    def __init__(self, channels):
        super().__init__()
        self.conv3 = DeformableConv3d(channels, channels, kernel_size=3)
        self.conv5 = DeformableConv3d(channels, channels, kernel_size=5)
        self.conv7 = DeformableConv3d(channels, channels, kernel_size=7)
        self.fusion = nn.Conv3d(3*channels, channels, kernel_size=1)

    def forward(self, x):
        return self.fusion(torch.cat([
            self.conv3(x),
            self.conv5(x), 
            self.conv7(x)
        ], dim=1))

def test_3d_unet_with_deform():
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
        deform_layers=[3,4]  # 在最后两个下采样阶段使用可变形卷积
    ).to(device)

    # 打印模型结构
    print(model)
    
    # 前向传播
    with torch.no_grad():
        output = model(input_tensor)
    
    # 验证输出尺寸
    assert output.shape == (batch_size, num_classes, spatial_size, spatial_size, spatial_size), \
        f"输出尺寸错误，期望: {(batch_size, num_classes, spatial_size, spatial_size, spatial_size)}, 实际: {output.shape}"
    
    # 打印测试结果
    print("\n测试通过！")
    print("输入尺寸:", input_tensor.shape)
    print("输出尺寸:", output.shape)
    print("设备信息:", device)
    print("最后一层权重范数:", torch.norm(model.conv.weight).item())

if __name__ == "__main__":
    # 执行测试
    test_3d_unet_with_deform()
