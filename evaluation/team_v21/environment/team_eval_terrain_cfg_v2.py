# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Frozen v2 terrain definitions for the team's Ant comparison.

Version 2 replaces every v1 seed.  The geometry families remain disjoint from
the three members' declared training sets: rails, narrow gaps, and repeated
cylinders were not used for training by any member.
"""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg


def _generator(*, seed: int, difficulty: float, sub_terrains: dict) -> TerrainGeneratorCfg:
    return TerrainGeneratorCfg(
        seed=seed,
        curriculum=True,
        size=(8.0, 8.0),
        border_width=20.0,
        # Forty rows leave at least 304 m of continuous generated terrain in
        # +x when wrappers restrict starts to rows 0..4.
        num_rows=40,
        num_cols=20,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        difficulty_range=(difficulty, difficulty),
        use_cache=False,
        sub_terrains=sub_terrains,
    )


TEAM_V2_UNSEEN_EASY_CFG = _generator(
    seed=53001,
    difficulty=0.5,
    sub_terrains={
        "rails": terrain_gen.MeshRailsTerrainCfg(
            proportion=1.0,
            rail_thickness_range=(0.08, 0.12),
            rail_height_range=(0.04, 0.04),
            platform_width=2.0,
        )
    },
)


TEAM_V2_UNSEEN_MEDIUM_CFG = _generator(
    seed=53012,
    difficulty=0.5,
    sub_terrains={
        "narrow_gap": terrain_gen.MeshGapTerrainCfg(
            proportion=1.0,
            gap_width_range=(0.05, 0.12),
            platform_width=2.0,
        )
    },
)


TEAM_V2_UNSEEN_HARD_CFG = _generator(
    seed=53003,
    difficulty=0.55,
    sub_terrains={
        "repeated_cylinders": terrain_gen.MeshRepeatedCylindersTerrainCfg(
            proportion=1.0,
            platform_width=2.0,
            platform_height=0.02,
            abs_height_noise=(-0.005, 0.005),
            object_params_start=terrain_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(
                num_objects=30,
                height=0.03,
                radius=0.16,
                max_yx_angle=0.0,
                degrees=True,
            ),
            object_params_end=terrain_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(
                num_objects=60,
                height=0.09,
                radius=0.22,
                max_yx_angle=8.0,
                degrees=True,
            ),
        )
    },
)


TEAM_V2_SEEN_CONTROL_CFG = _generator(
    seed=51004,
    difficulty=0.45,
    sub_terrains={
        "plane": terrain_gen.MeshPlaneTerrainCfg(proportion=0.25),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.25,
            noise_range=(-0.05, 0.05),
            noise_step=0.01,
            border_width=0.25,
        ),
        "random_grid": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.25,
            grid_width=0.45,
            grid_height_range=(0.02, 0.08),
            platform_width=2.0,
            holes=False,
        ),
        "up_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.25,
            slope_range=(0.07, 0.18),
            platform_width=2.5,
            border_width=0.25,
        ),
    },
)


TEAM_V2_EVAL_TERRAINS = {
    "unseen_easy": TEAM_V2_UNSEEN_EASY_CFG,
    "unseen_medium": TEAM_V2_UNSEEN_MEDIUM_CFG,
    "unseen_hard": TEAM_V2_UNSEEN_HARD_CFG,
    "seen_control": TEAM_V2_SEEN_CONTROL_CFG,
}

TEAM_V2_FRICTION_RANGES = {
    "unseen_easy": {
        "static_friction_range": (1.0, 1.0),
        "dynamic_friction_range": (1.0, 1.0),
    },
    "unseen_medium": {
        "static_friction_range": (0.65, 1.35),
        "dynamic_friction_range": (0.55, 1.15),
    },
    "unseen_hard": {
        "static_friction_range": (0.35, 1.60),
        "dynamic_friction_range": (0.25, 1.30),
    },
    "seen_control": {
        "static_friction_range": (0.50, 1.50),
        "dynamic_friction_range": (0.40, 1.20),
    },
}
