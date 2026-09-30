# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Experiment C: B's rough terrain plus spatial ground-friction variation."""

from copy import deepcopy
from dataclasses import fields

from isaaclab.utils import configclass

from .ant_rough_env_cfg import AntRoughEnvCfg, AntRoughSceneCfg
from .terrain_dr import AntFrictionTerrainCfg, AntFrictionTerrainImporter


def _friction_terrain_cfg() -> AntFrictionTerrainCfg:
    """Copy all of B's terrain fields without sharing mutable configurations."""
    base = AntRoughSceneCfg().terrain
    values = {field.name: deepcopy(getattr(base, field.name)) for field in fields(base)}
    values["class_type"] = AntFrictionTerrainImporter
    return AntFrictionTerrainCfg(**values)


@configclass
class AntRoughDRSceneCfg(AntRoughSceneCfg):
    terrain: AntFrictionTerrainCfg = _friction_terrain_cfg()


@configclass
class AntRoughDREnvCfg(AntRoughEnvCfg):
    # Inherit observations, actions, rewards, events, terminations and timing.
    scene: AntRoughDRSceneCfg = AntRoughDRSceneCfg(num_envs=4096, env_spacing=5.0, clone_in_fabric=False)
