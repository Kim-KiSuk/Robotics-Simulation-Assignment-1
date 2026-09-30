# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Check registration, baseline isolation, terrain spawning and batched resets.

Run: ./isaaclab.sh -p scripts/environments/check_ant_rough.py --headless
This is a physics/configuration check, not a learned-policy evaluation.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=120)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs < 2 or args.steps < 1:
    parser.error("Use at least two environments and one step")
launcher = AppLauncher(args)
app = launcher.app

import gymnasium as gym
import numpy as np
import sys
import traceback
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg
from isaaclab_tasks.manager_based.classic.ant.ant_rough_env_cfg import AntRoughEnvCfg
from isaaclab_tasks.manager_based.classic.ant.terrain_mdp import base_height_above_ground
from isaaclab_tasks.utils import load_cfg_from_registry


def main():
    flat = AntEnvCfg()
    cfg = AntRoughEnvCfg()
    assert flat.scene.terrain.terrain_type == "plane"
    assert cfg.actions.to_dict() == flat.actions.to_dict()
    assert cfg.rewards.to_dict() == flat.rewards.to_dict()
    assert cfg.events.to_dict() == flat.events.to_dict()
    assert cfg.sim.dt == flat.sim.dt and cfg.decimation == flat.decimation
    assert cfg.episode_length_s == flat.episode_length_s == 16.0
    assert list(cfg.observations.policy.__dict__) == list(flat.observations.policy.__dict__)
    assert cfg.observations.policy.base_height.func is base_height_above_ground
    # Instantiating/changing the rough config must not mutate a new flat config.
    cfg.scene.terrain.physics_material.static_friction = 0.4
    assert AntEnvCfg().scene.terrain.physics_material.static_friction == 1.0
    cfg.scene.terrain.physics_material.static_friction = 1.0
    flat_agent = load_cfg_from_registry("Isaac-Ant-v0", "rsl_rl_cfg_entry_point")
    rough_agent = load_cfg_from_registry("Isaac-Ant-Rough-v0", "rsl_rl_cfg_entry_point")
    assert flat_agent.experiment_name == "ant" and rough_agent.experiment_name == "ant_rough"
    assert flat_agent.policy.to_dict() == rough_agent.policy.to_dict()
    assert flat_agent.algorithm.to_dict() == rough_agent.algorithm.to_dict()
    # Geometry must depend on terrain seed, independently of the reset RNG.
    small = cfg.scene.terrain.terrain_generator.copy()
    small.num_rows, small.num_cols = 2, 3
    np.random.seed(10)
    first = small.class_type(small, device="cpu").terrain_mesh
    np.random.seed(20)
    second = small.class_type(small, device="cpu").terrain_mesh
    assert np.array_equal(first.vertices, second.vertices)
    assert np.array_equal(first.faces, second.faces)
    small.seed = 2001
    third = small.class_type(small, device="cpu").terrain_mesh
    assert not np.array_equal(first.vertices, third.vertices)
    print("[CHECK] Terrain seed independent of reset seed: PASS", flush=True)
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = 42
    env = gym.make("Isaac-Ant-Rough-v0", cfg=cfg)
    try:
        with torch.inference_mode():
            obs, _ = env.reset()
            policy_obs = obs["policy"]
            print(f"[CHECK] Observation shape: {tuple(policy_obs.shape)}; action size: {env.unwrapped.action_manager.total_action_dim}")
            assert env.unwrapped.action_manager.total_action_dim == 8
            assert torch.isfinite(policy_obs).all()
            ground = env.unwrapped.scene["ground_height"].data.ray_hits_w[:, 0, 2]
            assert ground.shape == (args.num_envs,) and torch.isfinite(ground).all()
            heights = base_height_above_ground(env.unwrapped).squeeze(-1)
            assert (heights >= 0.49).all(), f"Invalid initial clearance: {heights}"
            print(f"[CHECK] Spawn clearance: {heights.min().item():.4f} .. {heights.max().item():.4f} m")
            # Exercise last sensor index: catches Fabric/clone sensor count errors.
            env.unwrapped._reset_idx(torch.tensor([0, args.num_envs - 1], device=env.unwrapped.device))
            actions = torch.zeros(args.num_envs, 8, device=env.unwrapped.device)
            endings = 0
            for _ in range(args.steps):
                if not app.is_running():
                    raise RuntimeError("Simulation closed before check finished")
                obs, reward, terminated, truncated, _ = env.step(actions)
                assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
                hits = env.unwrapped.scene["ground_height"].data.ray_hits_w
                assert torch.isfinite(hits).all(), "Ground ray missed during the stationary smoke test"
                endings += (terminated | truncated).sum().item()
            print(f"[CHECK] PASS: baseline isolation, registered PPO, ground rays, resets, {args.steps} physics steps ({endings} episode endings).")
    finally:
        env.close()


try:
    main()
except BaseException:
    # Print before Kit shutdown, which may terminate the interpreter directly.
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    raise
else:
    sys.stdout.flush()
    sys.stderr.flush()
    app.close()
