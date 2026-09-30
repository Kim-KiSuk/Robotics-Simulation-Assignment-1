# Copyright (c) 2026, Ant generalization assignment contributors.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .ant_rough_ppo_cfg import AntRoughPPORunnerCfg


@configclass
class AntRoughRewardPPORunnerCfg(AntRoughPPORunnerCfg):
    experiment_name = "ant_rough_reward"


@configclass
class AntRoughHistoryPPORunnerCfg(AntRoughPPORunnerCfg):
    experiment_name = "ant_rough_history"


@configclass
class AntRoughHistoryRewardPPORunnerCfg(AntRoughPPORunnerCfg):
    experiment_name = "ant_rough_history_reward"
