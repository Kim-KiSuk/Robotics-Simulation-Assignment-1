# Copyright (c) 2026, Ant generalization assignment contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain v2: no flat tiles, stronger varied relief, isolated from A/B/C/R/H/HR."""

import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass

from .ant_rough_env_cfg import AntRoughEnvCfg, AntRoughSceneCfg
from .ant_robust_env_cfg import AntHistoryObservationsCfg, AntRobustRewardsCfg
from .varied_terrain_spec import make_varied_sub_terrains


@configclass
class AntVariedSceneCfg(AntRoughSceneCfg):
    # Retain map size, resolution, placement, collider partitioning and friction.
    terrain = AntRoughSceneCfg().terrain.replace(
        terrain_generator=AntRoughSceneCfg().terrain.terrain_generator.replace(
            sub_terrains=make_varied_sub_terrains(terrain_gen),
        ),
        # Ant observations contain no images; this only improves GUI visibility.
        visual_material=AntRoughSceneCfg().terrain.visual_material.replace(
            diffuse_color=(0.45, 0.45, 0.45),
        ),
    )


@configclass
class AntVariedEnvCfg(AntRoughEnvCfg):
    scene: AntVariedSceneCfg = AntVariedSceneCfg(num_envs=4096, env_spacing=5.0, clone_in_fabric=False)


@configclass
class AntVariedRewardEnvCfg(AntVariedEnvCfg):
    rewards: AntRobustRewardsCfg = AntRobustRewardsCfg()


@configclass
class AntVariedHistoryEnvCfg(AntVariedEnvCfg):
    observations: AntHistoryObservationsCfg = AntHistoryObservationsCfg()


@configclass
class AntVariedHistoryRewardEnvCfg(AntVariedHistoryEnvCfg):
    rewards: AntRobustRewardsCfg = AntRobustRewardsCfg()
