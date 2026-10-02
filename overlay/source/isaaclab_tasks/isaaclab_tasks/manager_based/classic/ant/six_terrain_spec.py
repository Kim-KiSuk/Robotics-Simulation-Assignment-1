# Copyright (c) 2026, Ant generalization assignment contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Six named environment profiles, independent of simulator imports.

T1..T5 are training scenarios. E1 is a held-out draw of their geometry mixture
with a separate fixed ground material. Mix is an execution mode, not a seventh
scenario. Factories return fresh configs so existing experiments are unchanged.
"""

from math import radians, tan

TRAIN_PROFILES = ("T1", "T2", "T3", "T4", "T5")
PROFILES = {
    "T1": dict(title="Small roughness", seed=1101, friction=(1.0, 1.0)),
    "T2": dict(title="Waves", seed=1102, friction=(0.8, 0.6)),
    "T3": dict(title="Slopes", seed=1103, friction=(1.2, 1.0)),
    "T4": dict(title="Uneven square blocks", seed=1104, friction=(0.6, 0.45)),
    "T5": dict(title="IsaacLab terrain mixture", seed=1105, friction=(1.4, 1.1)),
    # Selected using spawn-geometry coverage only, without running a learned policy.
    "E1": dict(title="Held-out mixture", seed=9117, friction=(0.9, 0.75)),
}
MIX_SEED = 1100
TILE_SIZE = (12.0, 12.0)
TASKS = {p: f"Isaac-Ant-Six-Train{i}-v0" for i, p in enumerate(TRAIN_PROFILES, 1)}
TASKS.update(E1="Isaac-Ant-Six-Eval-v0", Mix="Isaac-Ant-Six-TrainMix-v0")


def seed_for(profile):
    return MIX_SEED if profile == "Mix" else PROFILES[profile]["seed"]


def make_sub_terrains(profile, terrain):
    """Use actual IsaacLab generators, keeping five profile weights equal in Mix/E1."""
    if profile in ("Mix", "E1"):
        result = {}
        for name in TRAIN_PROFILES:
            for key, cfg in make_sub_terrains(name, terrain).items():
                cfg.proportion /= len(TRAIN_PROFILES)
                result[key] = cfg
        return result
    slopes = (tan(radians(3.0)), tan(radians(12.0)))
    if profile == "T1":
        entries = {
            "small": terrain.HfRandomUniformTerrainCfg(
                proportion=0.5, noise_range=(-0.02, 0.02), noise_step=0.01, downsampled_scale=0.8),
            "medium": terrain.HfRandomUniformTerrainCfg(
                proportion=0.5, noise_range=(-0.06, 0.06), noise_step=0.01, downsampled_scale=0.8),
        }
    elif profile == "T2":
        entries = {
            "long": terrain.HfWaveTerrainCfg(proportion=0.5, amplitude_range=(0.02, 0.10), num_waves=2),
            "short": terrain.HfWaveTerrainCfg(proportion=0.5, amplitude_range=(0.02, 0.10), num_waves=4),
        }
    elif profile == "T3":
        entries = {
            "slope": terrain.HfPyramidSlopedTerrainCfg(
                proportion=0.5, slope_range=slopes, platform_width=1.0),
            "slope_inv": terrain.HfInvertedPyramidSlopedTerrainCfg(
                proportion=0.5, slope_range=slopes, platform_width=1.0),
        }
    elif profile == "T4":
        # Based on the user's description, not a pixel/scale reconstruction of the inaccessible image.
        entries = {
            "blocks": terrain.MeshRandomGridTerrainCfg(
                proportion=1.0, grid_width=0.7, grid_height_range=(0.02, 0.10),
                platform_width=0.8, holes=False),
        }
    elif profile == "T5":
        entries = {
            "stairs": terrain.MeshPyramidStairsTerrainCfg(
                proportion=0.2, step_height_range=(0.03, 0.08), step_width=0.6,
                platform_width=1.2, border_width=0.2, holes=False),
            "stairs_inv": terrain.MeshInvertedPyramidStairsTerrainCfg(
                proportion=0.2, step_height_range=(0.03, 0.08), step_width=0.6,
                platform_width=1.2, border_width=0.2, holes=False),
            "grid": terrain.MeshRandomGridTerrainCfg(
                proportion=0.2, grid_width=0.65, grid_height_range=(0.02, 0.08),
                platform_width=0.8, holes=False),
            "slope": terrain.HfPyramidSlopedTerrainCfg(
                proportion=0.2, slope_range=slopes, platform_width=1.0),
            "slope_inv": terrain.HfInvertedPyramidSlopedTerrainCfg(
                proportion=0.2, slope_range=slopes, platform_width=1.0),
        }
    else:
        raise ValueError(f"Unknown profile: {profile}")
    return {f"{profile}__{key}": cfg for key, cfg in entries.items()}
