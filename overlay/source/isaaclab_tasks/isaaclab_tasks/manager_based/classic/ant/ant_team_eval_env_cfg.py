# Copyright (c) 2026, Ant generalization assignment contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluation-only adapters: frozen team terrain/material, unchanged 123D policy contract."""

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from .ant_env_cfg import EventCfg, RewardsCfg
from .ant_wave_range_env_cfg import AntWaveRangeEnvCfg
from .team_eval_distribution import load_frozen_module
from .terrain import AntTerrainImporter

_terrain = load_frozen_module("team_eval_terrain_cfg.py")
_material = load_frozen_module("team_eval_material_event.py")


@configclass
class AntTeamEvalEventsCfg(EventCfg):
    robot_material = EventTermCfg(
        func=_material.randomize_rigid_body_material_per_env,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )


@configclass
class AntTeamEvalBaseEnvCfg(AntWaveRangeEnvCfg):
    events: AntTeamEvalEventsCfg = AntTeamEvalEventsCfg()
    rewards: RewardsCfg = RewardsCfg()

    def set_team_level(self, level):
        previous = self.scene.terrain
        # Keep the existing XY grid/spawn-height handling and sensor mesh path.
        # SixTerrainImporter cannot be reused: the frozen generator has no Six
        # material bookkeeping, and seen_control legitimately uses curriculum
        # to generate its terrain columns. Do not change the generator itself.
        self.scene.terrain = TerrainImporterCfg(
            class_type=AntTerrainImporter,
            prim_path=previous.prim_path,
            terrain_type="generator",
            collision_group=previous.collision_group,
            terrain_generator=_terrain.TEAM_EVAL_TERRAINS[level].copy(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0,
                friction_combine_mode="multiply", restitution_combine_mode="multiply",
                restitution=0.0,
            ),
            visual_material=previous.visual_material.copy(),
            debug_vis=False,
        )
        self.events.robot_material.params.update(_terrain.TEAM_EVAL_FRICTION_RANGES[level])
        self.episode_length_s = 16.0
        self.scene.num_envs = 100
        self.curriculum = None
        # Preserve the trained ground-relative 0.31 m termination. The distributed
        # prose does not name a height coordinate frame; document this explicitly.
        assert self.terminations.torso_height.params["minimum_height"] == 0.31


@configclass
class AntTeamSeenControlEnvCfg(AntTeamEvalBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.set_team_level("seen_control")


@configclass
class AntTeamUnseenEasyEnvCfg(AntTeamEvalBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.set_team_level("unseen_easy")


@configclass
class AntTeamUnseenMediumEnvCfg(AntTeamEvalBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.set_team_level("unseen_medium")


@configclass
class AntTeamUnseenHardEnvCfg(AntTeamEvalBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.set_team_level("unseen_hard")
