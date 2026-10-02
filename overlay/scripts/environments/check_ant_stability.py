# Copyright (c) 2026, Ant generalization assignment contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Check stability fine-tuning physics, reward timing and old-task preservation."""

import argparse
import json
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=16)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args).app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg
from isaaclab_tasks.manager_based.classic.ant.ant_height_scan_env_cfg import AntHeightScanTrainEnvCfg, AntHeightScanEvalEnvCfg
from isaaclab_tasks.manager_based.classic.ant.ant_height_stability_env_cfg import AntHeightStabilityEnvCfg
from isaaclab_tasks.manager_based.classic.ant.stability_terrain_spec import TASK
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(TASK, "rsl_rl_cfg_entry_point")
def main(cfg, agent):
    saved = [c().to_dict() for c in (AntEnvCfg, AntHeightScanTrainEnvCfg, AntHeightScanEvalEnvCfg)]
    old = AntHeightScanTrainEnvCfg()
    for name in ("observations", "actions", "events", "terminations", "sim"):
        assert getattr(cfg,name).to_dict() == getattr(old,name).to_dict(), name
    assert cfg.scene.robot.to_dict() == old.scene.robot.to_dict()
    for key, value in old.rewards.to_dict().items():
        assert cfg.rewards.to_dict()[key] == value, key
    assert len(cfg.rewards.to_dict()) == 9
    assert len(AntHeightScanEvalEnvCfg().rewards.to_dict()) == 7
    assert cfg.scene.terrain.terrain_generator.seed == 1200
    assert cfg.scene.num_envs == 1024
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = 42
    env = gym.make(TASK, cfg=cfg)
    try:
        base = env.unwrapped
        with torch.inference_mode():
            obs,_ = env.reset(seed=42)
            assert obs["policy"].shape == (args.num_envs,123)
            action = torch.zeros(args.num_envs,8,device=base.device)
            for _ in range(120):
                obs,reward,_,_,_ = env.step(action)
                assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
                assert torch.isfinite(base.scene["height_scanner"].data.ray_hits_w).all()
            # Force a failure and a pure timeout on the SAME next simulation step.
            env.reset(seed=42)
            robot = base.scene["robot"]
            pose = robot.data.root_state_w[:, :7].clone()
            ground = base.scene["ground_height"].data.ray_hits_w[:,0,2]
            pose[0,2] = ground[0] - 0.5
            robot.write_root_pose_to_sim(pose)
            base.episode_length_buf[1] = base.max_episode_length - 1
            _, _, terminated, truncated, _ = env.step(action)
            assert terminated[0] and truncated[1] and not terminated[1]
            failure = dict(base.reward_manager.get_active_iterable_terms(0))["failure"][0]
            timeout = dict(base.reward_manager.get_active_iterable_terms(1))["failure"][0]
            # get_active_iterable_terms reports rates, so integrate by step_dt.
            assert abs(failure * base.step_dt + 5.0) < 1e-5, failure
            assert timeout == 0.0, timeout
        assert [c().to_dict() for c in (AntEnvCfg, AntHeightScanTrainEnvCfg, AntHeightScanEvalEnvCfg)] == saved
        result = dict(passed=True,num_envs=args.num_envs,observation_dim=123,training_reward_terms=9,
                      evaluation_reward_terms=7,forced_failure_penalty=-5,timeout_penalty=0,
                      old_configs_unchanged=True,policy_performance_evaluation=False)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+"\n")
        print(f"[CHECK] PASS: {result}",flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc();sys.stdout.flush();sys.stderr.flush()
        raise
    else:
        sys.stdout.flush();app.close()
