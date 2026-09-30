# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Baseline PPO with a separate rough-terrain output directory."""

from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import AntPPORunnerCfg


@configclass
class AntRoughPPORunnerCfg(AntPPORunnerCfg):
    experiment_name = "ant_rough"
