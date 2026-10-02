# Copyright (c) 2026, Ant generalization assignment contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Controlled observation experiment: 60 existing states + 63 local heights."""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

from .ant_blocks_eval_env_cfg import make_blocks_scene
from .ant_rough_env_cfg import AntRoughEnvCfg, AntRoughObservationsCfg, AntRoughSceneCfg
from .ant_six_env_cfg import make_scene
from .height_scan_mdp import relative_ground_scan
from .height_scan_sensor import IdealTerrainRayCaster


@configclass
class AntHeightScanSceneCfg(AntRoughSceneCfg):
    height_scanner = RayCasterCfg(
        class_type=IdealTerrainRayCaster,
        prim_path="{ENV_REGEX_NS}/Robot/torso",
        offset=RayCasterCfg.OffsetCfg(pos=(0.6, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.3, size=(2.4, 1.8), ordering="xy"),
        mesh_prim_paths=["/World/ground/terrain/mesh"],
        debug_vis=False,
    )


def with_height_scanner(scene):
    # Use a declared configclass field so Hydra/copy/replace retain the sensor.
    from copy import deepcopy
    from dataclasses import fields

    return AntHeightScanSceneCfg(**{field.name: deepcopy(getattr(scene, field.name)) for field in fields(scene)})


@configclass
class AntHeightScanObservationsCfg(AntRoughObservationsCfg):
    @configclass
    class PolicyCfg(AntRoughObservationsCfg.PolicyCfg):
        terrain_heights = ObsTerm(func=relative_ground_scan)

    policy: PolicyCfg = PolicyCfg()


@configclass
class AntHeightScanTrainEnvCfg(AntRoughEnvCfg):
    # Exact same five training terrain profiles/materials/seed as the previous run.
    scene: AntHeightScanSceneCfg = with_height_scanner(make_scene("Mix"))
    observations: AntHeightScanObservationsCfg = AntHeightScanObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt


@configclass
class AntHeightScanEvalEnvCfg(AntHeightScanTrainEnvCfg):
    # E3 geometry, friction, reset, termination, reward, time limit match the 60D task.
    scene: AntHeightScanSceneCfg = with_height_scanner(make_blocks_scene())
