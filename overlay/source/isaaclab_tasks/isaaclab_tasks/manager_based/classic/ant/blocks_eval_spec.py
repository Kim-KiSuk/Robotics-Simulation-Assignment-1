# Copyright (c) 2026, Ant generalization assignment contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""E3: continuous terrain plus square blocks, shared by 60D and height-scan policies."""

from math import radians, tan

TASK = "Isaac-Ant-Six-Eval-Blocks-v0"
SCAN_TASK = "Isaac-Ant-Six-Eval-Blocks-HeightScan-v0"
TRAIN_TASK = "Isaac-Ant-Six-TrainMix-HeightScan-v0"
TERRAIN_SEED = 9317
TILE_SIZE = (8.0, 8.0)
NUM_ROWS = NUM_COLS = 40
HORIZONTAL_SCALE = 0.25
VERTICAL_SCALE = 0.005
FRICTION = (0.9, 0.75)


def make_sub_terrains(terrain, continuous_surface, block_surface):
    """New evaluation configuration; never imported by the training terrain factory."""
    slopes = tuple(tan(radians(deg)) for deg in (8.0, 16.0))
    result = {
        "rough": terrain.HfRandomUniformTerrainCfg(
            proportion=0.25, noise_range=(-0.06, 0.06), noise_step=0.01,
            downsampled_scale=0.5, border_width=0.0,
        ),
        "wave_3": terrain.HfWaveTerrainCfg(
            proportion=0.10, amplitude_range=(0.08, 0.18), num_waves=3, border_width=0.0,
        ),
        "wave_5": terrain.HfWaveTerrainCfg(
            proportion=0.10, amplitude_range=(0.08, 0.18), num_waves=5, border_width=0.0,
        ),
        "slope": terrain.HfPyramidSlopedTerrainCfg(
            proportion=0.10, slope_range=slopes, platform_width=0.0, border_width=0.0,
        ),
        "slope_inv": terrain.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.10, slope_range=slopes, platform_width=0.0, border_width=0.0,
        ),
    }
    for cfg in result.values():
        cfg.function = continuous_surface
    # The stock grid generator requires a flat border and always adds a central
    # platform, even when its width is zero. Use a full-coverage box grid instead.
    result["blocks"] = terrain.MeshRandomGridTerrainCfg(
        proportion=0.35, grid_width=0.8, grid_height_range=(0.03, 0.12),
        platform_width=0.0, holes=False,
    )
    result["blocks"].function = block_surface
    return result
