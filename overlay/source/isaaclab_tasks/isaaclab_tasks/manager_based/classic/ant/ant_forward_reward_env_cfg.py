# Copyright (c) 2026, Ant generalization assignment contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Reward-only training experiment on the unchanged WaveRange terrain.

The team evaluation tasks still inherit AntWaveRangeEnvCfg directly and use
the original RewardsCfg. Do not substitute this config into evaluation.
"""

from isaaclab.managers import RewardTermCfg
from isaaclab.utils import configclass

from .ant_env_cfg import RewardsCfg
from .ant_wave_range_env_cfg import AntWaveRangeEnvCfg
from .robust_mdp import failure_impulse


@configclass
class AntForwardRewardsCfg(RewardsCfg):
    # Potential-based progress is uncapped; do not reward only a target speed.
    progress = RewardsCfg().progress.replace(weight=1.5)
    # Reduce bonuses that can be earned without advancing toward the target.
    alive = RewardsCfg().alive.replace(weight=0.1)
    move_to_target = RewardsCfg().move_to_target.replace(weight=0.25)
    # Keep energy, joint limits, action magnitude and upright terms unchanged.
    # RewardManager integrates by dt; failure_impulse cancels dt, giving exactly
    # -3 per true termination. A pure timeout is not penalized.
    failure = RewardTermCfg(func=failure_impulse, weight=-3.0)


@configclass
class AntForwardRewardEnvCfg(AntWaveRangeEnvCfg):
    """Preserve terrain difficulty, sensors, action, events and termination."""

    rewards: AntForwardRewardsCfg = AntForwardRewardsCfg()
