# Copyright (c) 2026, Ant generalization assignment contributors.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from .rsl_rl_ppo_cfg import AntPPORunnerCfg


@configclass
class AntSixPPORunnerCfg(AntPPORunnerCfg):
    # Exact baseline PPO/network settings, only a separate output folder.
    experiment_name = "ant_six"
