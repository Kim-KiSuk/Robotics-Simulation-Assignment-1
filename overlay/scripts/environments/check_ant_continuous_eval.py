# Copyright (c) 2026, Ant generalization assignment contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""E2 configuration/physics checks, including a no-reset ground support probe.

No policy is loaded. Check the console/Kit log for PhysX errors as well as this
script's assertions. The support probe disables height termination temporarily;
that diagnostic change is NEVER applied to the registered evaluation task.
"""

import argparse
import json
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=100)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
from pxr import UsdGeom, UsdPhysics, UsdShade

import isaaclab.sim as sim_utils
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.classic.ant import continuous_eval_spec as spec
from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg
from isaaclab_tasks.manager_based.classic.ant.ant_six_env_cfg import AntSixEvalEnvCfg, AntSixTrainMixEnvCfg
from isaaclab_tasks.manager_based.classic.ant.terrain_mdp import base_height_above_ground
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(spec.TASK, "rsl_rl_cfg_entry_point")
def main(cfg, agent_cfg):
    saved = [AntEnvCfg().to_dict(), AntSixEvalEnvCfg().to_dict(), AntSixTrainMixEnvCfg().to_dict()]
    for name in ("actions", "rewards", "events", "observations", "terminations", "sim"):
        assert getattr(cfg, name).to_dict() == saved[1][name], name
    assert cfg.scene.robot.to_dict() == saved[1]["scene"]["robot"]
    assert cfg.scene.num_envs == 100 and cfg.scene.env_spacing == 5.0
    assert cfg.scene.clone_in_fabric is False
    assert cfg.episode_length_s == 16.0 and cfg.decimation == 2
    cfg.scene.num_envs = args.num_envs
    cfg.seed = 24
    cfg.sim.device = args.device
    env = gym.make(spec.TASK, cfg=cfg)
    try:
        base = env.unwrapped
        stage = sim_utils.get_current_stage()
        parent = stage.GetPrimAtPath("/World/ground/terrain/colliders")
        colliders = list(parent.GetChildren())
        assert colliders
        triangle_count = 0
        for prim in colliders:
            assert prim.HasAPI(UsdPhysics.CollisionAPI)
            assert UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == "none"
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose="physics")
            api = UsdPhysics.MaterialAPI(material.GetPrim())
            assert np.isclose(api.GetStaticFrictionAttr().Get(), spec.FRICTION[0])
            assert np.isclose(api.GetDynamicFrictionAttr().Get(), spec.FRICTION[1])
            triangle_count += len(UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get())
        assert triangle_count == 3_276_800
        ray_mesh = stage.GetPrimAtPath("/World/ground/terrain/mesh")
        assert not ray_mesh.HasAPI(UsdPhysics.CollisionAPI)

        # Check the actual seeded generator's reproducibility independently of
        # robot RNGs, on a smaller map so this check does not duplicate the atlas.
        small = cfg.scene.terrain.terrain_generator.copy()
        small.num_rows = small.num_cols = 2
        np_before, torch_before = np.random.get_state(), torch.get_rng_state().clone()
        cuda_before = torch.cuda.get_rng_state_all()
        first = small.class_type(small).terrain_mesh.vertices.copy()
        np_after = np.random.get_state()
        assert np_before[0] == np_after[0] and np_before[2:] == np_after[2:]
        np.testing.assert_array_equal(np_before[1], np_after[1])
        assert torch.equal(torch_before, torch.get_rng_state())
        assert all(torch.equal(a, b) for a, b in zip(cuda_before, torch.cuda.get_rng_state_all()))
        np.random.seed(999)
        torch.manual_seed(999)
        np.testing.assert_array_equal(first, small.class_type(small).terrain_mesh.vertices)
        small.seed += 1
        assert not np.array_equal(first, small.class_type(small).terrain_mesh.vertices)
        np.random.set_state(np_before)
        torch.set_rng_state(torch_before)
        torch.cuda.set_rng_state_all(cuda_before)

        with torch.inference_mode():
            obs, _ = env.reset()
            assert obs["policy"].shape == (args.num_envs, 60)
            assert base.action_manager.total_action_dim == 8
            assert (base_height_above_ground(base) >= 0.49).all()
            base._reset_idx(torch.tensor(sorted({0, args.num_envs - 1}), device=base.device))
            action = torch.zeros(args.num_envs, 8, device=base.device)
            for _ in range(120):
                obs, rew, _, _, _ = env.step(action)
                assert torch.isfinite(obs["policy"]).all() and torch.isfinite(rew).all()
                assert torch.isfinite(base.scene["ground_height"].data.ray_hits_w).all()

            # A working ray cast alone cannot prove that collision cooking worked.
            # Let passive robots settle for 3s without fall resets, and verify that
            # they are still supported by the actual surface instead of falling through.
            term = base.termination_manager.get_term_cfg("torso_height")
            saved_minimum = term.params["minimum_height"]
            term.params["minimum_height"] = -100.0
            base.termination_manager.set_term_cfg("torso_height", term)
            env.reset()
            minimum_clearance = float("inf")
            for _ in range(180):
                obs, rew, terminated, truncated, _ = env.step(action)
                assert not terminated.any() and not truncated.any(), "Support probe must not auto-reset"
                assert torch.isfinite(obs["policy"]).all() and torch.isfinite(rew).all()
                assert torch.isfinite(base.scene["ground_height"].data.ray_hits_w).all()
                minimum_clearance = min(minimum_clearance, base_height_above_ground(base).min().item())
            assert minimum_clearance > -0.05, f"Possible missing ground collision: {minimum_clearance}"
            term.params["minimum_height"] = saved_minimum
            base.termination_manager.set_term_cfg("torso_height", term)
        assert [AntEnvCfg().to_dict(), AntSixEvalEnvCfg().to_dict(), AntSixTrainMixEnvCfg().to_dict()] == saved
        assert load_cfg_from_registry(spec.TASK, "env_cfg_entry_point").terminations.torso_height.params["minimum_height"] == 0.31
        result = dict(task=spec.TASK, passed=True, num_envs=args.num_envs, robot_seed=24,
                      terrain_seed=spec.TERRAIN_SEED, observation_dim=60, action_dim=8,
                      collision_meshes=len(colliders), collision_triangles=triangle_count,
                      scope="120 normal zero-action steps + 180-step no-reset support probe; no policy scoring",
                      minimum_support_probe_clearance_m=minimum_clearance,
                      friction=spec.FRICTION, original_reward=True, old_configs_unchanged=True,
                      terrain_rng_isolated=True)
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
