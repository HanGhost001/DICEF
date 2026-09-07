# Copyright (c) 2026 Liangliang Han
# SPDX-License-Identifier: MIT

"""Archived per-modality augmentation; shared teacher/student views."""
from __future__ import annotations
import math
from typing import Any
import torch
import torch.nn.functional as F


def augmentation_config(mode: str) -> dict[str, Any]:
    if mode == "none":
        return {
            "affine_prob": 0.0,
            "rotation_deg": 0.0,
            "translate_vox": 0.0,
            "scale_range": [1.0, 1.0],
            "intensity_prob": 0.0,
            "noise_prob": 0.0,
            "noise_std": 0.0,
        }
    config = {
        "affine_prob": 0.8,
        "rotation_deg": 7.0,
        "translate_vox": 5.0,
        "scale_range": [0.95, 1.05],
        "intensity_prob": 0.5,
        "noise_prob": 0.25,
        "noise_std": 0.02,
    }
    if mode == "spatial":
        config.update({"intensity_prob": 0.0, "noise_prob": 0.0, "noise_std": 0.0})
    return config


def random_affine_3d(
    x: torch.Tensor,
    max_rotation_deg: float,
    max_translate_vox: float,
    scale_range: tuple[float, float],
    probability: float,
) -> torch.Tensor:
    if probability <= 0 or torch.rand(1, device=x.device).item() > probability:
        return x
    n, _, d, h, w = x.shape
    theta = torch.zeros((n, 3, 4), dtype=x.dtype, device=x.device)
    max_rot = math.radians(max_rotation_deg)
    for i in range(n):
        angles = (torch.rand(3, device=x.device, dtype=x.dtype) * 2 - 1) * max_rot
        ax, ay, az = angles
        cx, sx = torch.cos(ax), torch.sin(ax)
        cy, sy = torch.cos(ay), torch.sin(ay)
        cz, sz = torch.cos(az), torch.sin(az)
        rx = torch.stack(
            [
                torch.stack([torch.ones_like(cx), torch.zeros_like(cx), torch.zeros_like(cx)]),
                torch.stack([torch.zeros_like(cx), cx, -sx]),
                torch.stack([torch.zeros_like(cx), sx, cx]),
            ]
        )
        ry = torch.stack(
            [
                torch.stack([cy, torch.zeros_like(cy), sy]),
                torch.stack([torch.zeros_like(cy), torch.ones_like(cy), torch.zeros_like(cy)]),
                torch.stack([-sy, torch.zeros_like(cy), cy]),
            ]
        )
        rz = torch.stack(
            [
                torch.stack([cz, -sz, torch.zeros_like(cz)]),
                torch.stack([sz, cz, torch.zeros_like(cz)]),
                torch.stack([torch.zeros_like(cz), torch.zeros_like(cz), torch.ones_like(cz)]),
            ]
        )
        scale = torch.empty(1, device=x.device, dtype=x.dtype).uniform_(scale_range[0], scale_range[1]).squeeze()
        matrix = (rz @ ry @ rx) / scale
        trans_vox = (torch.rand(3, device=x.device, dtype=x.dtype) * 2 - 1) * max_translate_vox
        trans_norm = torch.stack([2 * trans_vox[0] / max(w, 1), 2 * trans_vox[1] / max(h, 1), 2 * trans_vox[2] / max(d, 1)])
        theta[i, :, :3] = matrix
        theta[i, :, 3] = trans_norm
    grid = F.affine_grid(theta, x.size(), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def augment_batch(x: torch.Tensor, cfg: dict) -> torch.Tensor:
    x = random_affine_3d(
        x,
        max_rotation_deg=cfg["rotation_deg"],
        max_translate_vox=cfg["translate_vox"],
        scale_range=tuple(cfg["scale_range"]),
        probability=cfg["affine_prob"],
    )
    if cfg["intensity_prob"] > 0:
        mask = torch.rand((x.shape[0], 1, 1, 1, 1), device=x.device) < cfg["intensity_prob"]
        scale = torch.empty((x.shape[0], 1, 1, 1, 1), device=x.device).uniform_(0.9, 1.1)
        shift = torch.empty((x.shape[0], 1, 1, 1, 1), device=x.device).uniform_(-0.05, 0.05)
        x = torch.where(mask, x * scale + shift, x)
    if cfg["noise_std"] > 0 and torch.rand(1, device=x.device).item() < cfg["noise_prob"]:
        x = x + torch.randn_like(x) * cfg["noise_std"]
    return x
