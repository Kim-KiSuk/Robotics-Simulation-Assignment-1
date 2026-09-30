# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Deterministic ground-material sampling, independent of simulation and global RNGs."""

from __future__ import annotations

import numpy as np


def sample_ground_friction(
    count: int,
    seed: int,
    static_range: tuple[float, float],
    dynamic_ratio_range: tuple[float, float],
) -> np.ndarray:
    """Return [static, dynamic] pairs with dynamic <= static.

    Stratify static coefficients over the requested interval, then shuffle the
    assignment to spatial regions. Dynamic/static ratios are uniform samples.
    These are ground material coefficients, not combined foot-ground friction.
    """
    if count < 1:
        raise ValueError("At least one collision region is required")
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("friction_seed must be a nonnegative integer")
    if len(static_range) != 2 or len(dynamic_ratio_range) != 2:
        raise ValueError("Friction ranges must contain exactly two values")
    low, high = static_range
    ratio_low, ratio_high = dynamic_ratio_range
    if not np.isfinite([low, high, ratio_low, ratio_high]).all():
        raise ValueError("Friction ranges must be finite")
    if not 0.0 <= low <= high:
        raise ValueError("static_friction_range must satisfy 0 <= low <= high")
    if not 0.0 <= ratio_low <= ratio_high <= 1.0:
        raise ValueError("dynamic_friction_ratio_range must satisfy 0 <= low <= high <= 1")
    rng = np.random.default_rng(seed)
    static = low + (high - low) * (np.arange(count) + rng.random(count)) / count
    rng.shuffle(static)
    dynamic = static * rng.uniform(ratio_low, ratio_high, count)
    return np.column_stack((static, dynamic))
