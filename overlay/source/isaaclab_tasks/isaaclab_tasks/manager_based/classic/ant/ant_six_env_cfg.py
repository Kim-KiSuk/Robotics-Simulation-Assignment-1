# Copyright (c) 2026, Ant generalization assignment contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Five training scenarios, one held-out scenario, and one joint training mode."""

from copy import deepcopy
from dataclasses import fields

import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass

from .ant_rough_env_cfg import AntRoughEnvCfg, AntRoughSceneCfg
from .six_terrain import SixTerrainGenerator, SixTerrainImporterCfg
from .six_terrain_spec import PROFILES, make_sub_terrains, seed_for


def make_scene(profile):
    # The mixed mesh/material scene exceeds the available 8 GiB VRAM at 4096.
    # CLI --num_envs still overrides this; keep all six scenarios on one default.
    scene = AntRoughSceneCfg(num_envs=1024, env_spacing=5.0, clone_in_fabric=False)
    base = scene.terrain
    values = {field.name: deepcopy(getattr(base, field.name)) for field in fields(base)}
    values.pop("class_type")
    scene.terrain = SixTerrainImporterCfg(**values, suite_profile=profile)
    scene.terrain.terrain_generator = base.terrain_generator.replace(
        class_type=SixTerrainGenerator, seed=seed_for(profile), sub_terrains=make_sub_terrains(profile, terrain_gen),
    )
    if profile != "Mix":
        static, dynamic = PROFILES[profile]["friction"]
        scene.terrain.physics_material = base.physics_material.replace(static_friction=static, dynamic_friction=dynamic)
    scene.terrain.visual_material = base.visual_material.replace(diffuse_color=(0.45, 0.45, 0.45))
    return scene


@configclass
class AntSixTrain1EnvCfg(AntRoughEnvCfg):
    scene: AntRoughSceneCfg = make_scene("T1")


@configclass
class AntSixTrain2EnvCfg(AntRoughEnvCfg):
    scene: AntRoughSceneCfg = make_scene("T2")


@configclass
class AntSixTrain3EnvCfg(AntRoughEnvCfg):
    scene: AntRoughSceneCfg = make_scene("T3")


@configclass
class AntSixTrain4EnvCfg(AntRoughEnvCfg):
    scene: AntRoughSceneCfg = make_scene("T4")


@configclass
class AntSixTrain5EnvCfg(AntRoughEnvCfg):
    scene: AntRoughSceneCfg = make_scene("T5")


@configclass
class AntSixEvalEnvCfg(AntRoughEnvCfg):
    scene: AntRoughSceneCfg = make_scene("E1")


@configclass
class AntSixTrainMixEnvCfg(AntRoughEnvCfg):
    scene: AntRoughSceneCfg = make_scene("Mix")
