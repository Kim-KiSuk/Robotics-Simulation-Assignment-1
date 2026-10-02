# Copyright (c) 2026, Ant generalization assignment contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""123D checkpoint-compatible stability fine-tuning. Evaluation stays on original rewards."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from . import robust_mdp
from .ant_env_cfg import RewardsCfg
from .ant_height_scan_env_cfg import AntHeightScanSceneCfg, AntHeightScanTrainEnvCfg, with_height_scanner
from .ant_six_env_cfg import make_scene
from .stability_terrain_spec import TERRAIN_SEED, extend_training_terrains


@configclass
class AntHeightStabilityRewardsCfg(RewardsCfg):
    # Keep all seven original terms, including uncapped progress. RewardManager
    # multiplies rates by dt; failure_impulse cancels dt for -5 per failure event.
    failure = RewTerm(func=robust_mdp.failure_impulse, weight=-5.0)
    clearance_risk = RewTerm(
        func=robust_mdp.low_clearance_risk, weight=-0.5,
        params={"safe_height": 0.40, "fall_height": 0.31},
    )


def make_stability_scene():
    scene = with_height_scanner(make_scene("Mix"))
    generator = scene.terrain.terrain_generator
    scene.terrain.terrain_generator = generator.replace(
        seed=TERRAIN_SEED, sub_terrains=extend_training_terrains(generator.sub_terrains),
    )
    return scene


@configclass
class AntHeightStabilityEnvCfg(AntHeightScanTrainEnvCfg):
    scene: AntHeightScanSceneCfg = make_stability_scene()
    rewards: AntHeightStabilityRewardsCfg = AntHeightStabilityRewardsCfg()
