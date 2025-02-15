# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/01/13 10:29:34
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: UXNet (Fork From leeh43/nnUNet)
*      VERSION: v1.0
=================================================
'''
import sys
sys.path.append('/root/workspace/VoxelMedix/src/nnArchitecture')

from typing import Tuple, Union
import torch.nn as nn
import torch.nn.functional as F
import torch
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from timm.models.layers import trunc_normal_, DropPath
from functools import partial

import torch.nn as nn
import torch.nn.init as init

# from _init_model import init_all_weights

class LayerNorm(nn.Module):
    """LayerNorm 支持两种数据格式：channels_last（默认）和 channels_first。
    输入维度的顺序。channels_last 对应于形状为 (batch_size, height, width, channels) 的输入，
    而 channels_first 对应于形状为 (batch_size, channels, height, width) 的输入。
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x

class ux_block(nn.Module):
    """ConvNeXt 块。有两种等效的实现方式：
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; 全部在 (N, C, H, W) 中
    (2) DwConv -> 置换到 (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; 置换回来
    我们使用 (2)，因为在 PyTorch 中它稍微快一些。

    参数:
        dim (int): 输入通道数。
        drop_path (float): 随机深度率。默认: 0.0
        layer_scale_init_value (float): Layer Scale 的初始值。默认: 1e-6。
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Conv3d(dim, 4 * dim, kernel_size=1, groups=dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv3d(4 * dim, dim, kernel_size=1, groups=dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1)  # (N, C, H, W, D) -> (N, H, W, D, C)
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 2, 3, 4, 1)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 4, 1, 2, 3)
        x = input + self.drop_path(x)
        return x

class uxnet_conv(nn.Module):
    """UXNet 的卷积部分。

    参数:
        in_chans (int): 输入图像的通道数。默认: 3
        depths (tuple(int)): 每个阶段的块数。默认: [3, 3, 9, 3]
        dims (int): 每个阶段的特征维度。默认: [96, 192, 384, 768]
        drop_path_rate (float): 随机深度率。默认: 0.
        layer_scale_init_value (float): Layer Scale 的初始值。默认: 1e-6.
        out_indices (list): 输出的索引。默认: [0, 1, 2, 3]
    """

    def __init__(
        self,
        in_chans=1,
        depths=[2, 2, 2, 2],
        dims=[48, 96, 192, 384],
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        out_indices=[0, 1, 2, 3],
    ):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem 和 3 个中间下采样卷积层
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv3d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[
                    ux_block(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.out_indices = out_indices

        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(4):
            layer = norm_layer(dims[i_layer])
            layer_name = f"norm{i_layer}"
            self.add_module(layer_name, layer)

    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                x_out = norm_layer(x)
                outs.append(x_out)
        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x

class ProjectionHead(nn.Module):
    """投影头，用于将特征映射到低维空间。

    参数:
        dim_in (int): 输入维度。
        proj_dim (int): 投影维度。默认: 256
        proj (str): 投影类型。默认: "convmlp"
        bn_type (str): 归一化类型。默认: "torchbn"
    """

    def __init__(self, dim_in, proj_dim=256, proj="convmlp", bn_type="torchbn"):
        super(ProjectionHead, self).__init__()
        if proj == "linear":
            self.proj = nn.Conv2d(dim_in, proj_dim, kernel_size=1)
        elif proj == "convmlp":
            self.proj = nn.Sequential(
                nn.Conv3d(dim_in, dim_in, kernel_size=1),
                nn.Sequential(nn.BatchNorm3d(dim_in), nn.ReLU()),
                nn.Conv3d(dim_in, proj_dim, kernel_size=1),
            )

    def forward(self, x):
        return F.normalize(self.proj(x), p=2, dim=1)

class UXNET(nn.Module):
    """UXNet 模型。

    参数:
        in_chans (int): 输入通道数。默认: 1
        out_chans (int): 输出通道数。默认: 13
        depths (list): 每个阶段的块数。默认: [1, 1, 1, 1]
        feat_size (list): 每个阶段的特征大小。默认: [32, 64, 128, 256]
        drop_path_rate (float): 随机深度率。默认: 0
        layer_scale_init_value (float): Layer Scale 的初始值。默认: 1e-6
        hidden_size (int): 隐藏层大小。默认: 768
        norm_name (Union[Tuple, str]): 归一化类型。默认: "instance"
        conv_block (bool): 是否使用卷积块。默认: True
        res_block (bool): 是否使用残差块。默认: True
        spatial_dims (int): 空间维度数。默认: 3
        predict_mode (bool): 是否处于预测模式。默认: True
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=13,
        depths=[1, 1, 1, 1],
        feat_size=[32, 64, 128, 256],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        hidden_size: int = 768,
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        spatial_dims=3,
        predict_mode=True
    ) -> None:
        super().__init__()
        self.predict_mode = predict_mode
        self.hidden_size = hidden_size
        self.in_chans = in_channels
        self.out_chans = out_channels
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value
        self.out_indice = list(range(len(self.feat_size)))
        self.spatial_dims = spatial_dims

        self.uxnet_3d = uxnet_conv(
            in_chans=self.in_chans,
            depths=self.depths,
            dims=self.feat_size,
            drop_path_rate=self.drop_path_rate,
            layer_scale_init_value=1e-6,
            out_indices=self.out_indice,
        )

        # 编码器部分
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.in_chans,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        # 解码器部分
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.hidden_size,
            out_channels=self.feat_size[3],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(
            spatial_dims=spatial_dims, in_channels=32, out_channels=self.out_chans
        )
        
        # self.apply(init_all_weights)

    def proj_feat(self, x, hidden_size, feat_size):
        """将特征投影到指定维度。"""
        new_view = (x.size(0), *feat_size, hidden_size)
        x = x.view(new_view)
        new_axes = (0, len(x.shape) - 1) + tuple(d + 1 for d in range(len(feat_size)))
        x = x.permute(new_axes).contiguous()
        return x

    def forward(self, x_in):
        """前向传播。"""
        outs = self.uxnet_3d(x_in)
        enc1 = self.encoder1(x_in)
        x2 = outs[0]
        enc2 = self.encoder2(x2)
        x3 = outs[1]
        enc3 = self.encoder3(x3)
        x4 = outs[2]
        enc4 = self.encoder4(x4)
        enc_hidden = self.encoder5(outs[3])
        dec3 = self.decoder5(enc_hidden, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0)
        if self.predict_mode:
            return self.out(out)
        else:
            # return self.softmax(out), self.softmax(self.out(out)) #FIXME:两个输出分别是什么
            return self.out(out)

if __name__ == "__main__":
    model = UXNET(in_channels=4, out_channels=4)
    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(model, (4, 128, 128, 128), as_strings=True, print_per_layer_stat=True, verbose=True)
    print(f'Computational complexity: {macs}')
    print(f'Number of parameters: {params}')