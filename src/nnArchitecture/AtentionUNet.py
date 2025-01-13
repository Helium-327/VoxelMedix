# -*- coding: UTF-8 -*-
'''
================================================
*      创建时间: 2025/01/13 10:44:22
*      作者: @Junyin Xiong
*      描述: 
*      版本: v1.0
=================================================
'''
# TODO: 待测试
# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Norm

__all__ = ["AttentionUnet"]


class ConvBlock(nn.Module):
    """
    卷积块，包含两个卷积层，每个卷积层后接激活函数和批归一化。
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int = 3,
        strides: int = 1,
        dropout=0.0,
    ):
        super().__init__()
        layers = [
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=None,
                adn_ordering="NDA",  # Normalization -> Dropout -> Activation
                act="relu",
                norm=Norm.BATCH,
                dropout=dropout,
            ),
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                strides=1,
                padding=None,
                adn_ordering="NDA",
                act="relu",
                norm=Norm.BATCH,
                dropout=dropout,
            ),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_c: torch.Tensor = self.conv(x)
        return x_c


class UpConv(nn.Module):
    """
    上采样卷积层，用于将特征图的分辨率提高。
    """

    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int, kernel_size=3, strides=2, dropout=0.0):
        super().__init__()
        self.up = Convolution(
            spatial_dims,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=kernel_size,
            act="relu",
            adn_ordering="NDA",
            norm=Norm.BATCH,
            dropout=dropout,
            is_transposed=True,  # 使用转置卷积进行上采样
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_u: torch.Tensor = self.up(x)
        return x_u


class AttentionBlock(nn.Module):
    """
    注意力机制模块，用于增强特征图中的重要区域。
    """

    def __init__(self, spatial_dims: int, f_int: int, f_g: int, f_l: int, dropout=0.0):
        super().__init__()
        self.W_g = nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=f_g,
                out_channels=f_int,
                kernel_size=1,
                strides=1,
                padding=0,
                dropout=dropout,
                conv_only=True,  # 仅卷积，不进行归一化和激活
            ),
            Norm[Norm.BATCH, spatial_dims](f_int),
        )

        self.W_x = nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=f_l,
                out_channels=f_int,
                kernel_size=1,
                strides=1,
                padding=0,
                dropout=dropout,
                conv_only=True,
            ),
            Norm[Norm.BATCH, spatial_dims](f_int),
        )

        self.psi = nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=f_int,
                out_channels=1,
                kernel_size=1,
                strides=1,
                padding=0,
                dropout=dropout,
                conv_only=True,
            ),
            Norm[Norm.BATCH, spatial_dims](1),
            nn.Sigmoid(),  # 使用Sigmoid激活函数生成注意力权重
        )

        self.relu = nn.ReLU()

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi: torch.Tensor = self.relu(g1 + x1)  # 相加后通过ReLU激活
        psi = self.psi(psi)

        return x * psi  # 将注意力权重应用于输入特征图


class AttentionLayer(nn.Module):
    """
    注意力层，包含上采样、注意力机制和特征融合。
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        submodule: nn.Module,
        up_kernel_size=3,
        strides=2,
        dropout=0.0,
    ):
        super().__init__()
        self.attention = AttentionBlock(
            spatial_dims=spatial_dims, f_g=in_channels, f_l=in_channels, f_int=in_channels // 2
        )
        self.upconv = UpConv(
            spatial_dims=spatial_dims,
            in_channels=out_channels,
            out_channels=in_channels,
            strides=strides,
            kernel_size=up_kernel_size,
        )
        self.merge = Convolution(
            spatial_dims=spatial_dims, in_channels=2 * in_channels, out_channels=in_channels, dropout=dropout
        )
        self.submodule = submodule

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fromlower = self.upconv(self.submodule(x))  # 从下层上采样
        att = self.attention(g=fromlower, x=x)  # 应用注意力机制
        att_m: torch.Tensor = self.merge(torch.cat((att, fromlower), dim=1))  # 融合特征
        return att_m


class AttentionUnet(nn.Module):
    """
    Attention U-Net 网络结构，基于 Otkay et al. "Attention U-Net: Learning Where to Look for the Pancreas"
    https://arxiv.org/abs/1804.03999

    参数:
        spatial_dims: 输入图像的空间维度数。
        in_channels: 输入通道数。
        out_channels: 输出类别数。
        channels (Sequence[int]): 通道数序列，顶层块在前。`channels` 的长度应不小于2。
        strides (Sequence[int]): 卷积的步幅。
        kernel_size: 卷积核大小。
        up_kernel_size: 转置卷积层的卷积核大小。
        dropout: dropout比例，默认为0，即不使用dropout。
        predict_mode: 是否处于预测模式，默认为False。
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Sequence[int] | int = 3,
        up_kernel_size: Sequence[int] | int = 3,
        dropout: float = 0.0,
        predict_mode=False,
    ):
        super().__init__()
        self.predict_mode = predict_mode
        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.dropout = dropout

        head = ConvBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=channels[0],
            dropout=dropout,
            kernel_size=self.kernel_size,
        )
        reduce_channels = Convolution(
            spatial_dims=spatial_dims,
            in_channels=channels[0],
            out_channels=out_channels,
            kernel_size=1,
            strides=1,
            padding=0,
            conv_only=True,
        )
        self.up_kernel_size = up_kernel_size

        def _create_block(channels: Sequence[int], strides: Sequence[int]) -> nn.Module:
            if len(channels) > 2:
                subblock = _create_block(channels[1:], strides[1:])
                return AttentionLayer(
                    spatial_dims=spatial_dims,
                    in_channels=channels[0],
                    out_channels=channels[1],
                    submodule=nn.Sequential(
                        ConvBlock(
                            spatial_dims=spatial_dims,
                            in_channels=channels[0],
                            out_channels=channels[1],
                            strides=strides[0],
                            dropout=self.dropout,
                            kernel_size=self.kernel_size,
                        ),
                        subblock,
                    ),
                    up_kernel_size=self.up_kernel_size,
                    strides=strides[0],
                    dropout=dropout,
                )
            else:
                # 下一层是底层，停止递归，创建底层作为该层的子模块
                return self._get_bottom_layer(channels[0], channels[1], strides[0])

        encdec = _create_block(self.channels, self.strides)
        self.model = nn.Sequential(head, encdec, reduce_channels)

    def _get_bottom_layer(self, in_channels: int, out_channels: int, strides: int) -> nn.Module:
        return AttentionLayer(
            spatial_dims=self.dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            submodule=ConvBlock(
                spatial_dims=self.dimensions,
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                dropout=self.dropout,
                kernel_size=self.kernel_size,
            ),
            up_kernel_size=self.up_kernel_size,
            strides=strides,
            dropout=self.dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.predict_mode:
            output = self.model(x)
            return output
        else:
            features = self.model[:-1](x)  # 获取到最后一层之前的特征
            output = self.model[-1](features)  # 使用最后一层进行预测
            return features, output


if __name__ == "__main__":
    model = AttentionUnet(
        3,
        in_channels=4,
        out_channels=4,
        channels=[32, 64, 128, 256, 320],
        strides=[2, 2, 2, 2],
    )
    from ptflops import get_model_complexity_info

    # 使用Ptflops计算参数量和FLOPs
    macs, params = get_model_complexity_info(
        model, (4, 128, 128, 128), as_strings=True, print_per_layer_stat=True, verbose=True
    )

    print(f"Computational complexity: {macs}")
    print(f"Number of parameters: {params}")