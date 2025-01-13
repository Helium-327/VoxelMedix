'''
================================================
*      CREATE ON: 2024/10/11 21:46:07
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 注意力模块
=================================================
'''
'''
 Finished:
    - ✅ SE_Block    | DDL: 2024//
    - ✅ CBAM_Block  | DDL: 2024//
    - ✅ Self Attention       | DDL: 2024//

'''



''''================================================== SE Attention ====================================================='''
import torch
import torch.nn as nn

from torch.nn import functional as F

from torchinfo import summary

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
    

''''======================= CBAM(Convolutional Block Attention Module) Attention ====================================='''
class ChannelAttention(nn.Module):
    def __init__(self, in_dim, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(in_dim, in_dim // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(in_dim // ratio, in_dim, 1, bias=False))
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avgout = self.fc(self.avg_pool(x))
        maxout = self.fc(self.max_pool(x))
        out = avgout + maxout
        return self.sigmoid(out)
    

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x) 
    

class CBAM(nn.Module):
    def __init__(self, in_dim, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_dim, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        b, c, d, h, w = x.shape
        y = self.ca(x) * x
        z = self.sa(y) * y
        return z
    
'''''========================================== Self Attention ================================================='''
class SelfAttention3D(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention3D, self).__init__()
        self.query_conv =   nn.Conv3d(in_channels, in_channels // 8, kernel_size=1) # 全连接
        self.key_conv   =   nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv =   nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, D, H, W = x.size()
        
        # Permute and reshape the input to separate batch and spatial dimensions
        query = self.query_conv(x).view(batch_size, -1, D, H, W).permute(0, 2, 3, 4, 1).contiguous()
        key = self.key_conv(x).view(batch_size, -1, D, H, W).permute(0, 2, 3, 4, 1).contiguous()
        value = self.value_conv(x).view(batch_size, -1, D, H, W).permute(0, 2, 3, 4, 1).contiguous()
        
        # Calculate the attention scores
        energy = torch.matmul(query, key.transpose(-1, -2))
        attention = F.softmax(energy, dim=-1)
        
        # Apply the attention to the values
        out = torch.matmul(attention, value)
        
        # Reshape and permute the output to match the original input shape
        out = out.permute(0, 4, 1, 2, 3).contiguous().view(batch_size, C, D, H, W)
        
        # Scale and add a residual connection
        out = self.gamma * out + x
        
        return out
    
'''''========================================== MHA Attention(太吃显存) ================================================='''
class MultiHeadAttention3D(nn.Module):
    def __init__(self, in_channels, num_heads):
        super(MultiHeadAttention3D, self).__init__()
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        
        self.query_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, D, H, W = x.size()
        
        # Split the input into multiple heads
        query = self.query_conv(x).view(batch_size, self.num_heads, self.head_dim, D, H, W)
        key = self.key_conv(x).view(batch_size, self.num_heads, self.head_dim, D, H, W)
        value = self.value_conv(x).view(batch_size, self.num_heads, self.head_dim, D, H, W)
        
        # Permute and reshape the input to separate batch and spatial dimensions
        query = query.permute(0, 1, 3, 4, 5, 2).contiguous().view(batch_size * self.num_heads, D * H * W, self.head_dim)
        key = key.permute(0, 1, 3, 4, 5, 2).contiguous().view(batch_size * self.num_heads, D * H * W, self.head_dim)
        value = value.permute(0, 1, 3, 4, 5, 2).contiguous().view(batch_size * self.num_heads, D * H * W, self.head_dim)
        
        # Calculate the attention scores
        energy = torch.matmul(query, key.transpose(-1, -2))
        attention = F.softmax(energy, dim=-1)
        
        # Apply the attention to the values
        out = torch.matmul(attention, value)
        
        # Reshape and permute the output to match the original input shape
        out = out.view(batch_size, self.num_heads, D * H * W, self.head_dim)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, C, D, H, W)
        
        # Scale and add a residual connection
        out = self.gamma * out + x
        
        return out
    


'''''========================================== CPCA Attention(太吃显存) ================================================='''

class CPCA_ChannelAttention(nn.Module):
    def __init__(self, input_channels, internal_neurons):
        super(CPCA_ChannelAttention, self).__init__()
        self.fc1 = nn.Conv3d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.fc2 = nn.Conv3d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x1 = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x1 = self.fc1(x1)
        x1 = F.relu(x1, inplace=True)
        x1 = self.fc2(x1)
        x1 = torch.sigmoid(x1)
        x2 = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
        x2 = self.fc1(x2)
        x2 = F.relu(x2, inplace=True)
        x2 = self.fc2(x2)
        x2 = torch.sigmoid(x2)
        x = x1 + x2
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x

class CPCA(nn.Module):
    def __init__(self, channels, channelAttention_reduce=4):
        super().__init__()
        self.ca = CPCA_ChannelAttention(input_channels=channels, internal_neurons=channels // channelAttention_reduce)
        self.dconv5_5 = nn.Conv3d(channels, channels, kernel_size=5, padding=2, groups=channels)
        self.dconv1_7 = nn.Conv3d(channels, channels, kernel_size=(1, 7), padding=(0, 3), groups=channels)
        self.dconv7_1 = nn.Conv3d(channels, channels, kernel_size=(7, 1), padding=(3, 0), groups=channels)
        self.dconv1_11 = nn.Conv3d(channels, channels, kernel_size=(1, 11), padding=(0, 5), groups=channels)
        self.dconv11_1 = nn.Conv3d(channels, channels, kernel_size=(11, 1), padding=(5, 0), groups=channels)
        self.dconv1_21 = nn.Conv3d(channels, channels, kernel_size=(1, 21), padding=(0, 10), groups=channels)
        self.dconv21_1 = nn.Conv3d(channels, channels, kernel_size=(21, 1), padding=(10, 0), groups=channels)
        self.conv = nn.Conv3d(channels, channels, kernel_size=1, padding=0)
        self.act = nn.GELU()

    def forward(self, inputs):
        inputs = self.conv(inputs)
        inputs = self.act(inputs)
        inputs = self.ca(inputs)
        x_init = self.dconv5_5(inputs)
        x_1 = self.dconv1_7(x_init)
        x_1 = self.dconv7_1(x_1)
        x_2 = self.dconv1_11(x_init)
        x_2 = self.dconv11_1(x_2)
        x_3 = self.dconv1_21(x_init)
        x_3 = self.dconv21_1(x_3)
        x = x_1 + x_2 + x_3 + x_init
        spatial_att = self.conv(x)
        out = spatial_att * inputs
        out = self.conv(out)
        return out
    
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

    
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_data = torch.randn(1, 128, 32, 32, 32)
    input_data = input_data.to(device)
    model = EMA3D(128)
    model = model.to(device)
    print(model)
    output = model(input_data)
    print(output.shape)
    summary(model, (1, 128, 32, 32, 32))