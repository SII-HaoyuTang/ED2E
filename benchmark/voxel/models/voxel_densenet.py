from __future__ import annotations

import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp


def _bn_function_factory(norm: nn.Module, relu: nn.Module, conv: nn.Module):
    def bn_function(*inputs: torch.Tensor) -> torch.Tensor:
        concated_features = torch.cat(inputs, 1)
        return conv(relu(norm(concated_features)))

    return bn_function


class _DenseLayer(nn.Module):
    def __init__(
        self,
        num_input_features: int,
        growth_rate: int,
        bn_size: int,
        drop_rate: float,
        efficient: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm3d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(
            num_input_features,
            bn_size * growth_rate,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.norm2 = nn.BatchNorm3d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            bn_size * growth_rate,
            growth_rate,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.drop_rate = drop_rate
        self.efficient = efficient

    def forward(self, *prev_features: torch.Tensor) -> torch.Tensor:
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.efficient and any(x.requires_grad for x in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _DenseBlock(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        drop_rate: float,
        efficient: bool = False,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _DenseLayer(
                    num_input_features + i * growth_rate,
                    growth_rate=growth_rate,
                    bn_size=bn_size,
                    drop_rate=drop_rate,
                    efficient=efficient,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, init_features: torch.Tensor) -> torch.Tensor:
        features = [init_features]
        for layer in self.layers:
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super().__init__(
            OrderedDict(
                [
                    ("norm", nn.BatchNorm3d(num_input_features)),
                    ("relu", nn.ReLU(inplace=True)),
                    (
                        "conv",
                        nn.Conv3d(
                            num_input_features,
                            num_output_features,
                            kernel_size=1,
                            stride=1,
                            bias=False,
                        ),
                    ),
                    ("pool", nn.AvgPool3d(kernel_size=2, stride=2)),
                ]
            )
        )


class VoxelDenseNetRegressor(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_dim: int = 6,
        growth_rate: int = 12,
        block_config: tuple[int, int] = (16, 16),
        compression: float = 0.5,
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0.0,
        small_inputs: bool = True,
        efficient: bool = False,
    ) -> None:
        super().__init__()
        assert 0 < compression <= 1

        if small_inputs:
            self.features = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "conv0",
                            nn.Conv3d(
                                in_channels,
                                num_init_features,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False,
                            ),
                        )
                    ]
                )
            )
        else:
            self.features = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "conv0",
                            nn.Conv3d(
                                in_channels,
                                num_init_features,
                                kernel_size=7,
                                stride=2,
                                padding=3,
                                bias=False,
                            ),
                        ),
                        ("norm0", nn.BatchNorm3d(num_init_features)),
                        ("relu0", nn.ReLU(inplace=True)),
                        ("pool0", nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
                    ]
                )
            )

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.features.add_module(f"denseblock{i + 1}", block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=int(num_features * compression),
                )
                self.features.add_module(f"transition{i + 1}", trans)
                num_features = int(num_features * compression)

        self.features.add_module("norm_final", nn.BatchNorm3d(num_features))
        self.classifier = nn.Linear(num_features, out_dim)

        for name, param in self.named_parameters():
            if "conv" in name and "weight" in name:
                n = param.size(0) * param.size(2) * param.size(3)
                param.data.normal_().mul_(math.sqrt(2.0 / n))
            elif "norm" in name and "weight" in name:
                param.data.fill_(1)
            elif "norm" in name and "bias" in name:
                param.data.zero_()
            elif "classifier" in name and "bias" in name:
                param.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool3d(out, (1, 1, 1))
        out = torch.flatten(out, 1)
        return self.classifier(out)
