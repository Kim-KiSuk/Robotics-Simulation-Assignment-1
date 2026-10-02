# Copyright (c) 2026, Ant generalization assignment contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Continuous E2 evaluation terrain, compatible with the 60D Six policy."""

import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass

from . import continuous_eval_spec as spec
from .ant_rough_env_cfg import AntRoughEnvCfg, AntRoughSceneCfg
from .continuous_eval_surface import continuous_surface
from .six_terrain import SixTerrainGenerator
from .terrain import AntTerrainImporter


def make_eval_scene():
    scene = AntRoughSceneCfg(num_envs=100, env_spacing=5.0, clone_in_fabric=False)
    # A single common friction material allows the existing spatial importer to
    # combine all terrain types. The full ray mesh is not a PhysX collider.
    scene.terrain = scene.terrain.replace(
        class_type=AntTerrainImporter,
        terrain_generator=scene.terrain.terrain_generator.replace(
            class_type=SixTerrainGenerator,
            seed=spec.TERRAIN_SEED,
            size=spec.TILE_SIZE,
            num_rows=spec.NUM_ROWS,
            num_cols=spec.NUM_COLS,
            horizontal_scale=spec.HORIZONTAL_SCALE,
            vertical_scale=spec.VERTICAL_SCALE,
            slope_threshold=None,
            border_width=0.0,
            curriculum=False,
            use_cache=False,
            sub_terrains=spec.make_sub_terrains(terrain_gen, continuous_surface),
        ),
        physics_material=scene.terrain.physics_material.replace(
            static_friction=spec.FRICTION[0], dynamic_friction=spec.FRICTION[1],
        ),
    )
    return scene


@configclass
class AntContinuousEvalEnvCfg(AntRoughEnvCfg):
    # Keep the Six observation/action interface, original seven rewards,
    # ground-relative termination, reset distribution and 16 second limit.
    scene: AntRoughSceneCfg = make_eval_scene()
