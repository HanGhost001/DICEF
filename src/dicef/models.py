# Copyright (c) 2026 Liangliang Han
# SPDX-License-Identifier: MIT

"""Shared-head encoders and frozen-feature fusion."""
from __future__ import annotations
import torch
from torch import nn
from .backbone import standard_resnet18_dual_branch


def standard_resnet_dual_branch(backbone, in_channels, num_classes):
    if backbone != "resnet18":
        raise ValueError("Only the formal ResNet-18 backbone is included")
    return standard_resnet18_dual_branch(in_channels, num_classes)


class T1PetDtiThreeEncoderSharedHead(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.t1_encoder = standard_resnet_dual_branch("resnet18", in_channels=1, num_classes=3)
        self.pet_encoder = standard_resnet_dual_branch("resnet18", in_channels=1, num_classes=3)
        self.dti_encoder = standard_resnet_dual_branch("resnet18", in_channels=3, num_classes=3)
        feature_dim = self.t1_encoder.classifier.in_features
        self.shared_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 3),
        )

    def forward(self, t1: torch.Tensor, pet: torch.Tensor, dti: torch.Tensor) -> dict[str, torch.Tensor]:
        t1_pack = self.t1_encoder.forward_with_tokens(t1)
        pet_pack = self.pet_encoder.forward_with_tokens(pet)
        dti_pack = self.dti_encoder.forward_with_tokens(dti)
        t1_feature = t1_pack["global_feature"]
        pet_feature = pet_pack["global_feature"]
        dti_feature = dti_pack["global_feature"]
        return {
            "t1_feature": t1_feature,
            "pet_feature": pet_feature,
            "dti_feature": dti_feature,
            "t1_logits": t1_pack["class_logits"],
            "pet_logits": pet_pack["class_logits"],
            "dti_logits": dti_pack["class_logits"],
            "shared_t1_logits": self.shared_head(t1_feature),
            "shared_pet_logits": self.shared_head(pet_feature),
            "shared_dti_logits": self.shared_head(dti_feature),
        }


class FusionMlp(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.2, linear: bool = False):
        super().__init__()
        if linear:
            self.net = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, 3))
        else:
            self.net = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Dropout(dropout),
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 3),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
