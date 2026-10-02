# Copyright (c) 2026, Ant generalization assignment contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Actual Hydra/PhysX checks for E3 and its height-scan interface; no policy scoring."""

import argparse
import json
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-Ant-Six-Eval-Blocks-HeightScan-v0")
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
from isaaclab_tasks.manager_based.classic.ant import blocks_eval_spec as spec
from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg
from isaaclab_tasks.manager_based.classic.ant.ant_six_env_cfg import AntSixEvalEnvCfg, AntSixTrainMixEnvCfg
from isaaclab_tasks.manager_based.classic.ant.ant_continuous_eval_env_cfg import AntContinuousEvalEnvCfg
from isaaclab_tasks.manager_based.classic.ant.ant_blocks_eval_env_cfg import AntBlocksEvalEnvCfg
from isaaclab_tasks.manager_based.classic.ant.ant_height_scan_env_cfg import AntHeightScanTrainEnvCfg, AntHeightScanEvalEnvCfg
from isaaclab_tasks.manager_based.classic.ant.terrain_mdp import base_height_above_ground
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args.task, "rsl_rl_cfg_entry_point")
def main(cfg, agent_cfg):
    factories = (AntEnvCfg, AntSixEvalEnvCfg, AntSixTrainMixEnvCfg, AntContinuousEvalEnvCfg)
    saved = [factory().to_dict() for factory in factories]
    old_train, scan_train = AntSixTrainMixEnvCfg(), AntHeightScanTrainEnvCfg()
    e3, scan_e3 = AntBlocksEvalEnvCfg(), AntHeightScanEvalEnvCfg()
    for old, new in ((old_train, scan_train), (e3, scan_e3)):
        for name in ("actions", "rewards", "events", "terminations", "sim"):
            assert getattr(old, name).to_dict() == getattr(new, name).to_dict(), name
        assert old.scene.terrain.to_dict() == new.scene.terrain.to_dict()
        assert old.scene.robot.to_dict() == new.scene.robot.to_dict()
        for name, term in old.observations.policy.to_dict().items():
            assert term == new.observations.policy.to_dict()[name], name
    scan = "HeightScan" in args.task
    cfg.scene.num_envs = args.num_envs
    cfg.seed = 24
    cfg.sim.device = args.device
    env = gym.make(args.task, cfg=cfg)
    try:
        base = env.unwrapped
        stage = sim_utils.get_current_stage()
        colliders = list(stage.GetPrimAtPath("/World/ground/terrain/colliders").GetChildren())
        count = 0
        for prim in colliders:
            assert prim.HasAPI(UsdPhysics.CollisionAPI)
            assert UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == "none"
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose="physics")
            material = UsdPhysics.MaterialAPI(material.GetPrim())
            assert np.isclose(material.GetStaticFrictionAttr().Get(), 0.9)
            assert np.isclose(material.GetDynamicFrictionAttr().Get(), 0.75)
            count += len(UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get())
        assert count == 2_812_944
        assert not stage.GetPrimAtPath("/World/ground/terrain/mesh").HasAPI(UsdPhysics.CollisionAPI)
        small = cfg.scene.terrain.terrain_generator.copy()
        small.num_rows = small.num_cols = 2
        before_np, before_torch = np.random.get_state(), torch.get_rng_state().clone()
        before_cuda = torch.cuda.get_rng_state_all()
        first = small.class_type(small).terrain_mesh.vertices.copy()
        np.testing.assert_array_equal(first, small.class_type(small).terrain_mesh.vertices)
        small.seed += 1
        assert not np.array_equal(first, small.class_type(small).terrain_mesh.vertices)
        after_np = np.random.get_state()
        assert before_np[0] == after_np[0] and before_np[2:] == after_np[2:]
        np.testing.assert_array_equal(before_np[1], after_np[1])
        assert torch.equal(before_torch, torch.get_rng_state())
        assert all(torch.equal(a, b) for a, b in zip(before_cuda, torch.cuda.get_rng_state_all()))
        with torch.inference_mode():
            obs, _ = env.reset(seed=24)
            expected_dim = 123 if scan else 60
            assert obs["policy"].shape == (args.num_envs, expected_dim)
            assert torch.isfinite(obs["policy"]).all()
            assert base.action_manager.total_action_dim == 8
            assert (base_height_above_ground(base) >= 0.49).all()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            np.savez(args.output.with_suffix(".initial.npz"),
                     obs=obs["policy"][:, :60].cpu().numpy(),
                     root=base.scene["robot"].data.root_state_w.cpu().numpy(),
                     joints=base.scene["robot"].data.joint_pos.cpu().numpy())
            if scan:
                sensor = base.scene["height_scanner"]
                assert sensor.num_instances == args.num_envs and sensor.num_rays == 63
                assert torch.isfinite(sensor.data.ray_hits_w).all()
                local = sensor.ray_starts[0, :, :2]
                torch.testing.assert_close(local.min(dim=0).values, torch.tensor([-0.6, -0.9], device=base.device))
                torch.testing.assert_close(local.max(dim=0).values, torch.tensor([1.8, 0.9], device=base.device))
                before = torch.cuda.get_rng_state_all()
                sensor.reset(torch.tensor([0, args.num_envs - 1], device=base.device))
                assert all(torch.equal(a, b) for a, b in zip(before, torch.cuda.get_rng_state_all()))
                # Check real ray XY locations after a 90 degree yaw change.
                robot = base.scene["robot"]
                pose = robot.data.root_state_w[:, :7].clone()
                turned = pose.clone()
                turned[0, 3:] = torch.tensor([2**-0.5, 0.0, 0.0, 2**-0.5], device=base.device)
                robot.write_root_pose_to_sim(turned)
                sensor.update(0.0, force_recompute=True)
                expected = torch.stack((-local[:, 1], local[:, 0]), dim=-1) + turned[0, :2]
                torch.testing.assert_close(sensor.data.ray_hits_w[0, :, :2], expected, atol=2e-5, rtol=1e-5)
                robot.write_root_pose_to_sim(pose)
                sensor.update(0.0, force_recompute=True)
            base._reset_idx(torch.tensor([0, args.num_envs - 1], device=base.device))
            action = torch.zeros(args.num_envs, 8, device=base.device)
            for _ in range(120):
                obs, reward, _, _, _ = env.step(action)
                assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
                assert torch.isfinite(base.scene["ground_height"].data.ray_hits_w).all()
                if scan:
                    assert torch.isfinite(base.scene["height_scanner"].data.ray_hits_w).all()
                    assert obs["policy"][:, 60:].abs().max() <= 1
            # Confirm the actual collision surface supports bodies, not only rays.
            term = base.termination_manager.get_term_cfg("torso_height")
            term.params["minimum_height"] = -100.0
            base.termination_manager.set_term_cfg("torso_height", term)
            env.reset(seed=24)
            minimum = float("inf")
            for _ in range(180):
                obs, reward, terminated, truncated, _ = env.step(action)
                assert not terminated.any() and not truncated.any()
                assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
                minimum = min(minimum, base_height_above_ground(base).min().item())
            assert minimum > -0.05, f"Possible collision failure: {minimum}"
            term.params["minimum_height"] = 0.31
            base.termination_manager.set_term_cfg("torso_height", term)
        assert [factory().to_dict() for factory in factories] == saved
        result = dict(task=args.task, passed=True, num_envs=args.num_envs, observation_dim=expected_dim,
                      action_dim=8, triangles=count, colliders=len(colliders), original_rewards=True,
                      minimum_support_clearance_m=minimum, terrain_rng_isolated=True,
                      scan_reset_rng_neutral=scan, yaw_ray_alignment_checked=scan,
                      old_configs_unchanged=True, steps=300, policy_evaluation_performed=False)
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
