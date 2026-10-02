# Copyright (c) 2026, Ant generalization assignment contributors.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from .ant_rough_ppo_cfg import AntRoughPPORunnerCfg


@configclass
class AntVariedPPORunnerCfg(AntRoughPPORunnerCfg):
    experiment_name = "ant_varied"


@configclass
class AntVariedRewardPPORunnerCfg(AntRoughPPORunnerCfg):
    experiment_name = "ant_varied_reward"


@configclass
class AntVariedHistoryPPORunnerCfg(AntRoughPPORunnerCfg):
    experiment_name = "ant_varied_history"


@configclass
class AntVariedHistoryRewardPPORunnerCfg(AntRoughPPORunnerCfg):
    experiment_name = "ant_varied_history_reward"
