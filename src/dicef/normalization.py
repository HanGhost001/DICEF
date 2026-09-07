# Copyright (c) 2026 Liangliang Han
# SPDX-License-Identifier: MIT

"""Numeric cache transforms; registration and anatomical preprocessing are external."""
import numpy as np


def normalize_t1(array: np.ndarray) -> np.ndarray:
    return np.clip(array, -3.0, 3.0) / 3.0


def normalize_pet(array: np.ndarray) -> np.ndarray:
    mask = array != 0
    result = np.zeros_like(array, dtype=np.float32)
    result[mask] = (np.clip(array[mask], 0.0, 2.5) - 1.0) / 0.5
    return result


def normalize_dti(fa: np.ndarray, md: np.ndarray, rd: np.ndarray) -> np.ndarray:
    fa = np.clip(fa, 0.0, 1.0)
    md = np.clip(md, 0.0, 0.004) / 0.004
    rd = np.clip(rd, 0.0, 0.004) / 0.004
    return np.stack([fa, md, rd], axis=0).astype(np.float32)
