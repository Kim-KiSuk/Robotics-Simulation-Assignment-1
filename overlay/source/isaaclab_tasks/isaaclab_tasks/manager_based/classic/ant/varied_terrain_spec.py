# Copyright (c) 2026, Ant generalization assignment contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain v2 parameters. No Isaac Sim imports, also used by the CPU mesh audit."""

from math import radians, tan


def make_varied_sub_terrains(terrain):
    """Fresh configurations: noise 40%, waves 30%, slopes 30%, flat tiles 0%.

    Noise has three nominal sampling ranges. Spline interpolation may overshoot
    those bounds. Wave amplitude is the IsaacLab parameter, not peak-to-peak
    height. Slopes use rise/run, with angle bounds converted by tan().
    """
    slopes = (tan(radians(6.0)), tan(radians(18.0)))
    return {
        "rough_low": terrain.HfRandomUniformTerrainCfg(
            proportion=0.15, noise_range=(-0.04, 0.04), noise_step=0.01, downsampled_scale=0.8,
        ),
        "rough_medium": terrain.HfRandomUniformTerrainCfg(
            proportion=0.15, noise_range=(-0.08, 0.08), noise_step=0.01, downsampled_scale=0.8,
        ),
        "rough_high": terrain.HfRandomUniformTerrainCfg(
            proportion=0.10, noise_range=(-0.12, 0.12), noise_step=0.01, downsampled_scale=1.2,
        ),
        "waves_long": terrain.HfWaveTerrainCfg(
            proportion=0.15, amplitude_range=(0.06, 0.18), num_waves=2,
        ),
        "waves_short": terrain.HfWaveTerrainCfg(
            proportion=0.15, amplitude_range=(0.06, 0.18), num_waves=4,
        ),
        "slope": terrain.HfPyramidSlopedTerrainCfg(
            proportion=0.15, slope_range=slopes, platform_width=0.0,
        ),
        "inverted_slope": terrain.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.15, slope_range=slopes, platform_width=0.0,
        ),
    }
