# Copyright (c) 2026, Ant generalization assignment contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""E3 evaluation map. The 60D task also accepts the existing baseline/TrainMix policies."""

import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass

from . import blocks_eval_spec as spec
from .ant_continuous_eval_env_cfg import make_eval_scene as make_continuous_scene
from .ant_rough_env_cfg import AntRoughEnvCfg, AntRoughSceneCfg
from .block_surface import square_block_surface
from .continuous_eval_surface import continuous_surface


def make_blocks_scene():
    scene = make_continuous_scene()
    scene.terrain.terrain_generator = scene.terrain.terrain_generator.replace(
        seed=spec.TERRAIN_SEED, size=spec.TILE_SIZE,
        num_rows=spec.NUM_ROWS, num_cols=spec.NUM_COLS,
        horizontal_scale=spec.HORIZONTAL_SCALE, vertical_scale=spec.VERTICAL_SCALE,
        sub_terrains=spec.make_sub_terrains(terrain_gen, continuous_surface, square_block_surface),
    )
    scene.terrain.physics_material = scene.terrain.physics_material.replace(
        static_friction=spec.FRICTION[0], dynamic_friction=spec.FRICTION[1],
    )
    return scene


@configclass
class AntBlocksEvalEnvCfg(AntRoughEnvCfg):
    scene: AntRoughSceneCfg = make_blocks_scene()
