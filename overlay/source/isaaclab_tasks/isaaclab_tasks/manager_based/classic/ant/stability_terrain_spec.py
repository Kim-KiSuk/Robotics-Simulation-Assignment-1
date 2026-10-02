# Copyright (c) 2026, Ant generalization assignment contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Further training: retain familiar terrain ranges and add a modest harder mixture."""

from copy import deepcopy
from math import radians, tan

TASK = "Isaac-Ant-Six-TrainMix-HeightScan-Stability-v0"
TERRAIN_SEED = 1200
ORIGINAL_PROPORTION = 0.7


def extend_training_terrains(original):
    """Fresh configs, 70% original ranges + 30% expanded ranges; no eval map import.

    The prefix T1..T5 is retained so the existing importer binds the original
    profile's friction. Proportions describe tile draws, not episode time.
    """
    result = {}
    for name, cfg in original.items():
        familiar, expanded = deepcopy(cfg), deepcopy(cfg)
        familiar.proportion *= ORIGINAL_PROPORTION
        expanded.proportion *= 1.0 - ORIGINAL_PROPORTION
        if hasattr(expanded, "noise_range"):
            amplitude = 0.04 if "small" in name else 0.08
            expanded.noise_range = (-amplitude, amplitude)
            expanded.downsampled_scale = 0.6
        elif hasattr(expanded, "num_waves"):
            expanded.amplitude_range = (0.04, 0.16)
            expanded.num_waves = 3 if expanded.num_waves == 2 else 5
        elif hasattr(expanded, "slope_range"):
            expanded.slope_range = tuple(tan(radians(deg)) for deg in (4.0, 16.0))
        elif hasattr(expanded, "grid_height_range"):
            expanded.grid_height_range = (0.03, 0.12)
            # The stock generator needs a nonzero remainder for its border.
            expanded.grid_width = 0.85 if name.startswith("T4__") else 0.55
        elif hasattr(expanded, "step_height_range"):
            expanded.step_height_range = (0.04, 0.10)
        else:
            raise TypeError(f"Unsupported training terrain: {name}")
        result[name] = familiar
        result[name + "__expanded"] = expanded
    return result
