# Copyright (c) 2026, Ant generalization assignment contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Physics/configuration smoke check for terrain v2; run after training has stopped."""

import argparse
import sys
import traceback
from isaaclab.app import AppLauncher

TASKS = (
    "Isaac-Ant-Varied-v0", "Isaac-Ant-Varied-Reward-v0",
    "Isaac-Ant-Varied-History-v0", "Isaac-Ant-Varied-History-Reward-v0",
)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default=TASKS[0], choices=TASKS)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
app = launcher.app

import gymnasium as gym
import torch
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.classic.ant.ant_rough_env_cfg import AntRoughEnvCfg
from isaaclab_tasks.manager_based.classic.ant.terrain_mdp import base_height_above_ground
from isaaclab_tasks.utils import load_cfg_from_registry


def main():
    old = AntRoughEnvCfg().to_dict()
    for task in TASKS:
        cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
        tiles = cfg.scene.terrain.terrain_generator.sub_terrains
        assert "flat" not in tiles and len(tiles) == 7
        assert abs(sum(tile.proportion for tile in tiles.values()) - 1.0) < 1e-12
        for key in ("actions", "events", "terminations", "sim"):
            assert getattr(cfg, key).to_dict() == old[key], key
        agent = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
        old_agent = load_cfg_from_registry("Isaac-Ant-Rough-v0", "rsl_rl_cfg_entry_point")
        assert agent.algorithm.to_dict() == old_agent.algorithm.to_dict()
        assert agent.policy.to_dict() == old_agent.policy.to_dict()
    assert AntRoughEnvCfg().to_dict() == old
    cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    cfg.scene.num_envs = 16
    cfg.seed = 24
    cfg.sim.device = args.device
    cfg.scene.terrain.terrain_generator.seed = 4001
    env = gym.make(args.task, cfg=cfg)
    try:
        with torch.inference_mode():
            obs, _ = env.reset()
            base = env.unwrapped
            size = 240 if "History" in args.task else 60
            assert obs["policy"].shape == (16, size)
            assert base.action_manager.total_action_dim == 8
            assert torch.isfinite(obs["policy"]).all()
            assert (base_height_above_ground(base) >= 0.49).all()
            base._reset_idx(torch.tensor([0, 15], device=base.device))
            for _ in range(120):
                assert app.is_running(), "Simulator closed during check"
                obs, reward, _, _, _ = env.step(torch.zeros(16, 8, device=base.device))
                assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
                assert torch.isfinite(base.scene["ground_height"].data.ray_hits_w).all()
        print(f"[CHECK] PASS: {args.task}, original config preserved, {size} observations, spawn, rays, resets, 120 steps", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    else:
        sys.stdout.flush()
        app.close()
