# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ant locomotion environment (similar to OpenAI Gym Ant-v2).
"""

import gymnasium as gym

from . import agents

from .six_terrain_spec import TASKS as _SIX_TASKS
from .continuous_eval_spec import TASK as _CONTINUOUS_EVAL_TASK
from . import blocks_eval_spec as _BLOCKS_SPEC

for _task, _module, _cfg, _agent_module, _agent_cfg in (
    (_BLOCKS_SPEC.TASK, "ant_blocks_eval_env_cfg", "AntBlocksEvalEnvCfg", "ant_six_ppo_cfg", "AntSixPPORunnerCfg"),
    (_BLOCKS_SPEC.SCAN_TASK, "ant_height_scan_env_cfg", "AntHeightScanEvalEnvCfg", "ant_height_scan_ppo_cfg", "AntHeightScanPPORunnerCfg"),
    (_BLOCKS_SPEC.TRAIN_TASK, "ant_height_scan_env_cfg", "AntHeightScanTrainEnvCfg", "ant_height_scan_ppo_cfg", "AntHeightScanPPORunnerCfg"),
):
    gym.register(
        id=_task,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.{_module}:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.{_agent_module}:{_agent_cfg}",
        },
    )

# Five scenarios + one held-out scenario. Mix reuses the five training profiles.
for _profile, _task in _SIX_TASKS.items():
    _cfg = {"E1": "AntSixEvalEnvCfg", "Mix": "AntSixTrainMixEnvCfg"}.get(
        _profile, f"AntSixTrain{_profile[1:]}EnvCfg"
    )
    gym.register(
        id=_task,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.ant_six_env_cfg:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.ant_six_ppo_cfg:AntSixPPORunnerCfg",
        },
    )

##
# Register Gym environments.
##

gym.register(
    id=_CONTINUOUS_EVAL_TASK,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_continuous_eval_env_cfg:AntContinuousEvalEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ant_six_ppo_cfg:AntSixPPORunnerCfg",
    },
)

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


gym.register(
    id="Isaac-Ant-Varied-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_varied_env_cfg:AntVariedEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ant_varied_ppo_cfg:AntVariedPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Ant-Varied-Reward-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_varied_env_cfg:AntVariedRewardEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ant_varied_ppo_cfg:AntVariedRewardPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Ant-Varied-History-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_varied_env_cfg:AntVariedHistoryEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ant_varied_ppo_cfg:AntVariedHistoryPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Ant-Varied-History-Reward-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_varied_env_cfg:AntVariedHistoryRewardEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ant_varied_ppo_cfg:AntVariedHistoryRewardPPORunnerCfg",
    },
)
