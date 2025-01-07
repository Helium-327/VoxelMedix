# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2024/10/09 12:14:52
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: self-attention #TODO: 待修改
=================================================
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

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

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Example usage:
    in_channels = 64  # This should match the number of channels in your 3D input data
    attention_layer = SelfAttention3D(in_channels=in_channels).to(device=device)
    input_data = torch.randn(1, 64, 64, 64, 64).to(device)
    print(attention_layer(input_data).shape)

    # Assuming 'input_3d' is your 3D input tensor with shape (batch_size, channels, depth, height, width)
    # input_3d = torch.randn(batch_size, in_channels, depth, height, width)
    # output_3d = attention_layer(input_3d)