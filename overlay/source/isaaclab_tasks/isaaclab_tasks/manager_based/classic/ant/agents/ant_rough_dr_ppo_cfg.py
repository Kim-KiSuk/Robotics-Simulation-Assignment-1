# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Identical PPO budget/architecture; C checkpoints use a separate directory."""

from isaaclab.utils import configclass

from .ant_rough_ppo_cfg import AntRoughPPORunnerCfg


@configclass
class AntRoughDRPPORunnerCfg(AntRoughPPORunnerCfg):
    experiment_name = "ant_rough_dr"
