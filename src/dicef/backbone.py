# Copyright (c) 2026 Liangliang Han
# SPDX-License-Identifier: MIT

"""3D ResNet-18, including inert token parameters for checkpoint compatibility."""
from __future__ import annotations
import torch
from torch import nn


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)
        self.downsample = (
            nn.Identity()
            if stride == 1 and in_channels == channels
            else nn.Sequential(
                nn.Conv3d(in_channels, channels, 1, stride=stride, bias=False),
                nn.BatchNorm3d(channels),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + identity)


class StandardResNetDualBranch(nn.Module):
    """Standard 3D ResNet with separate classification and token paths."""

    def __init__(
        self,
        block: type[nn.Module],
        layers: tuple[int, int, int, int],
        in_channels: int = 1,
        num_classes: int = 3,
        token_dim: int = 256,
        token_grid_size: int = 3,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.token_grid_size = token_grid_size
        self.in_channels = 64
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(3, stride=2, padding=1),
        )
        self.block = block
        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        final_channels = 512 * block.expansion

        # The standard classification path never passes through token_projection.
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Linear(final_channels, num_classes)

        # The token branch is independent but remains differentiable to the shared backbone.
        self.token_pool = nn.AdaptiveAvgPool3d(token_grid_size)
        self.token_projection = nn.Conv3d(final_channels, token_dim, 1, bias=False)
        self.token_norm = nn.LayerNorm(token_dim)
        self.apply(self._init_weights)

    def _make_layer(self, channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [self.block(self.in_channels, channels, stride)]
        self.in_channels = channels * self.block.expansion
        layers.extend(self.block(self.in_channels, channels) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv3d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.BatchNorm3d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0, 0.01)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)

    def global_feature(self, feature_map: torch.Tensor) -> torch.Tensor:
        return self.global_pool(feature_map).flatten(1)

    def make_tokens(self, feature_map: torch.Tensor) -> torch.Tensor:
        tokens = self.token_projection(self.token_pool(feature_map)).flatten(2).transpose(1, 2)
        return self.token_norm(tokens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_map = self.forward_features(x)
        return self.classifier(self.global_feature(feature_map))

    def forward_with_tokens(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feature_map = self.forward_features(x)
        global_feature = self.global_feature(feature_map)
        return {
            "feature_map": feature_map,
            "global_feature": global_feature,
            "tokens": self.make_tokens(feature_map),
            "class_logits": self.classifier(global_feature),
        }


def standard_resnet18_dual_branch(
    in_channels: int = 1,
    num_classes: int = 3,
    token_dim: int = 256,
    token_grid_size: int = 3,
) -> StandardResNetDualBranch:
    return StandardResNetDualBranch(
        BasicBlock3D, (2, 2, 2, 2), in_channels, num_classes, token_dim, token_grid_size
    )
