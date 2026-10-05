# Copyright (c) 2026, Ant generalization assignment contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Training-only width variation and a repeated three-map schedule."""

from copy import deepcopy

TERRAIN_SEEDS = (1201, 1301, 1401)
STAGE_SEEDS = TERRAIN_SEEDS * 2
ITERATIONS_PER_STAGE = 200


def diversify_training_widths(original):
    """Keep family proportions and height ranges; split selected widths three ways.

    Regular grids have contiguous cells, so grid width changes edge spacing,
    not empty gaps. No holes or new evaluation terrain families are introduced.
    """
    result = {}
    changed = 0
    for name, cfg in original.items():
        field, widths = None, None
        if name.startswith("T4__") and hasattr(cfg, "grid_width"):
            field, widths = "grid_width", (0.55, 0.70, 0.90)
        elif name.startswith("T5__") and hasattr(cfg, "grid_width"):
            # 0.50 divides the 12 m tile exactly; this generator requires a
            # positive leftover border, so use 0.55 for the narrow variant.
            field, widths = "grid_width", (0.55, 0.65, 0.85)
        elif name.startswith("T5__") and hasattr(cfg, "step_width"):
            field, widths = "step_width", (0.45, 0.60, 0.80)
        if widths is None:
            result[name] = deepcopy(cfg)
            continue
        changed += 1
        for index, width in enumerate(widths):
            variant = deepcopy(cfg)
            variant.proportion /= len(widths)
            setattr(variant, field, width)
            result[f"{name}__width{index}"] = variant
    if changed != 4:
        raise ValueError(f"Expected four grid/stair components, found {changed}")
    return result
