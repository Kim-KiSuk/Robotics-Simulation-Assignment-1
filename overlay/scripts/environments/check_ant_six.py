# Copyright (c) 2026, Ant generalization assignment contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Short simulator check: configs, collisions/materials, ground rays, resets and observations."""

import argparse
import json
from pathlib import Path
import sys
import traceback
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--profile", choices=("T1", "T2", "T3", "T4", "T5", "E1", "Mix"), default="Mix")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--check-map-rng", action="store_true")
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg
from isaaclab_tasks.manager_based.classic.ant.ant_rough_env_cfg import AntRoughEnvCfg
from isaaclab_tasks.manager_based.classic.ant.terrain_mdp import base_height_above_ground
from isaaclab_tasks.manager_based.classic.ant.six_terrain_spec import TASKS, PROFILES, TRAIN_PROFILES
from isaaclab_tasks.manager_based.classic.ant.six_terrain import SixTerrainGenerator


def main():
    old = AntRoughEnvCfg().to_dict()
    baseline = AntEnvCfg().to_dict()
    for profile, task in TASKS.items():
        cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
        for key in ("actions", "rewards", "events"):
            assert getattr(cfg, key).to_dict() == baseline[key], (profile, key)
        assert cfg.sim.to_dict() == baseline["sim"], (profile, "physical simulation settings")
        assert cfg.scene.num_envs == 1024
        for key in ("observations", "terminations"):
            assert getattr(cfg, key).to_dict() == old[key], (profile, key)
        assert cfg.episode_length_s == 16.0 and cfg.decimation == 2
        keys = cfg.scene.terrain.terrain_generator.sub_terrains
        assert all(k.split("__")[0] in TRAIN_PROFILES for k in keys)
    assert AntRoughEnvCfg().to_dict() == old, "Shared configuration was mutated"
    assert AntEnvCfg().to_dict() == baseline
    if args.check_map_rng:
        grid_cfg = load_cfg_from_registry(TASKS["T4"], "env_cfg_entry_point").scene.terrain.terrain_generator.copy()
        grid_cfg.num_rows = grid_cfg.num_cols = 2
        np_before = np.random.get_state()
        torch_before = torch.get_rng_state().clone()
        cuda_before = torch.cuda.get_rng_state_all()
        first = SixTerrainGenerator(grid_cfg).terrain_mesh.vertices.copy()
        np_after = np.random.get_state()
        assert np_before[0] == np_after[0] and np_before[2:] == np_after[2:]
        np.testing.assert_array_equal(np_before[1], np_after[1])
        assert torch.equal(torch_before, torch.get_rng_state())
        assert all(torch.equal(a, b) for a, b in zip(cuda_before, torch.cuda.get_rng_state_all()))
        np.random.seed(987)
        torch.manual_seed(987)
        second = SixTerrainGenerator(grid_cfg).terrain_mesh.vertices
        np.testing.assert_array_equal(first, second)
        grid_cfg.seed += 1
        third = SixTerrainGenerator(grid_cfg).terrain_mesh.vertices
        assert not np.array_equal(first, third)
        np.random.set_state(np_before)
        torch.set_rng_state(torch_before)
        torch.cuda.set_rng_state_all(cuda_before)
        print("[CHECK] PASS: map seed controls torch grid; robot RNGs preserved", flush=True)
    cfg = load_cfg_from_registry(TASKS[args.profile], "env_cfg_entry_point")
    assert args.num_envs >= 1
    cfg.scene.num_envs = args.num_envs
    cfg.seed = 24
    cfg.sim.device = args.device
    env = gym.make(TASKS[args.profile], cfg=cfg)
    try:
        with torch.inference_mode():
            obs, _ = env.reset()
            base = env.unwrapped
            assert obs["policy"].shape == (args.num_envs, 60)
            assert base.action_manager.total_action_dim == 8
            assert (base_height_above_ground(base) >= 0.49).all()
            base._reset_idx(torch.tensor(sorted({0, args.num_envs - 1}), device=base.device))
            for _ in range(120):
                assert app.is_running()
                obs, reward, _, _, _ = env.step(torch.zeros(args.num_envs, 8, device=base.device))
                assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
                assert torch.isfinite(base.scene["ground_height"].data.ray_hits_w).all()
        terrain = base.scene.terrain
        for name, coeff in terrain.six_materials.items():
            expected = PROFILES[name if args.profile == "Mix" else args.profile]["friction"]
            assert tuple(coeff) == expected
        result = dict(profile=args.profile, task=TASKS[args.profile], passed=True,
                      scope=f"{args.num_envs} robots, full map, 120 zero-action steps; no training or policy scoring",
                      terrain_seed=cfg.scene.terrain.terrain_generator.seed, rng_check=args.check_map_rng,
                      materials=terrain.six_materials, tiles=len(terrain.six_tile_records),
                      triangles=sum(r["faces"] for r in terrain.six_tile_records), observation_dim=60)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"[CHECK] PASS: {result}", flush=True)
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
