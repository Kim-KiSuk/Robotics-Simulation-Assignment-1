# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ant locomotion environment (similar to OpenAI Gym Ant-v2).
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Ant-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_env_cfg:AntEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Ant-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_rough_env_cfg:AntRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ant_rough_ppo_cfg:AntRoughPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Ant-Rough-DR-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_rough_dr_env_cfg:AntRoughDREnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ant_rough_dr_ppo_cfg:AntRoughDRPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Ant-Rough-Reward-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_robust_env_cfg:AntRoughRewardEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ant_robust_ppo_cfg:AntRoughRewardPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Ant-Rough-History-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_robust_env_cfg:AntRoughHistoryEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ant_robust_ppo_cfg:AntRoughHistoryPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Ant-Rough-History-Reward-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_robust_env_cfg:AntRoughHistoryRewardEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ant_robust_ppo_cfg:AntRoughHistoryRewardPPORunnerCfg",
    },
)
