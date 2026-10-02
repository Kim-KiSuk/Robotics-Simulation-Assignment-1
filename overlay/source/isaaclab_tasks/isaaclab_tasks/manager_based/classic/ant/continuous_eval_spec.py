# Copyright (c) 2026, Ant generalization assignment contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""E2: a separate continuous-terrain evaluation scenario, with no flat tiles.

These values are fixed before policy scoring. E1 and training remain unchanged.
Slope values are rise/run, not radians. The stock height-field discretization
can produce small horizontal faces; no flat tile or central platform is added.
"""

from math import radians, tan

TASK = "Isaac-Ant-Six-Eval-Continuous-v0"
TERRAIN_SEED = 9217
TILE_SIZE = (8.0, 8.0)
NUM_ROWS = 40
NUM_COLS = 40
HORIZONTAL_SCALE = 0.25
VERTICAL_SCALE = 0.005
FRICTION = (0.9, 0.75)  # Same as E1 to isolate the change in geometry.
SLOPE_DEGREES = (8.0, 16.0)


def make_sub_terrains(terrain, surface_function):
    """Reuse IsaacLab height-field generators, with fresh configurations."""
    slopes = tuple(tan(radians(angle)) for angle in SLOPE_DEGREES)
    configs = {
        "rough": terrain.HfRandomUniformTerrainCfg(
            proportion=0.4, noise_range=(-0.06, 0.06), noise_step=0.01,
            downsampled_scale=0.5, border_width=0.0,
        ),
        "wave_3": terrain.HfWaveTerrainCfg(
            proportion=0.15, amplitude_range=(0.08, 0.18), num_waves=3, border_width=0.0,
        ),
        "wave_5": terrain.HfWaveTerrainCfg(
            proportion=0.15, amplitude_range=(0.08, 0.18), num_waves=5, border_width=0.0,
        ),
        "slope": terrain.HfPyramidSlopedTerrainCfg(
            proportion=0.15, slope_range=slopes, platform_width=0.0, border_width=0.0,
        ),
        "slope_inv": terrain.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.15, slope_range=slopes, platform_width=0.0, border_width=0.0,
        ),
    }
    for cfg in configs.values():
        cfg.function = surface_function
    return configs
