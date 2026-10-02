# Copyright (c) 2026, Ant generalization assignment contributors.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from .rsl_rl_ppo_cfg import AntPPORunnerCfg


@configclass
class AntHeightScanPPORunnerCfg(AntPPORunnerCfg):
    # PPO/hidden layers unchanged. Input layer automatically becomes 123D.
    experiment_name = "ant_six_height"
    max_iterations = 4000
