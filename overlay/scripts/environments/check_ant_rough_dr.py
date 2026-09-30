# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Verify C's physics material bindings, batched stepping and isolation from A/B."""

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
from pxr import PhysxSchema, UsdGeom, UsdPhysics, UsdShade

import isaaclab.sim as sim_utils
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg
from isaaclab_tasks.manager_based.classic.ant.ant_rough_env_cfg import AntRoughEnvCfg
from isaaclab_tasks.manager_based.classic.ant.ant_rough_dr_env_cfg import AntRoughDREnvCfg
from isaaclab_tasks.manager_based.classic.ant.friction_utils import sample_ground_friction
from isaaclab_tasks.utils import load_cfg_from_registry


def main():
    flat, rough, cfg = AntEnvCfg(), AntRoughEnvCfg(), AntRoughDREnvCfg()
    assert flat.scene.terrain.terrain_type == "plane"
    assert rough.scene.terrain.physics_material.static_friction == 1.0
    for field in ("observations", "actions", "rewards", "events", "terminations", "sim"):
        assert getattr(cfg, field).to_dict() == getattr(rough, field).to_dict(), field
    assert cfg.scene.terrain.terrain_generator.to_dict() == rough.scene.terrain.terrain_generator.to_dict()
    assert cfg.scene.robot.to_dict() == rough.scene.robot.to_dict()
    assert cfg.scene.ground_height.to_dict() == rough.scene.ground_height.to_dict()
    assert cfg.decimation == rough.decimation and cfg.episode_length_s == rough.episode_length_s
    cfg.scene.terrain.physics_material.static_friction = 0.2
    assert AntRoughEnvCfg().scene.terrain.physics_material.static_friction == 1.0
    assert AntEnvCfg().scene.terrain.physics_material.static_friction == 1.0
    cfg.scene.terrain.physics_material.static_friction = 1.0
    b_agent = load_cfg_from_registry("Isaac-Ant-Rough-v0", "rsl_rl_cfg_entry_point")
    c_agent = load_cfg_from_registry("Isaac-Ant-Rough-DR-v0", "rsl_rl_cfg_entry_point")
    before, after = b_agent.to_dict(), c_agent.to_dict()
    assert after.pop("experiment_name") == "ant_rough_dr"
    assert before.pop("experiment_name") == "ant_rough"
    assert before == after
    print("[CHECK] A/B preserved; C changes only terrain importer/material configuration and log name", flush=True)
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = 42
    env = gym.make("Isaac-Ant-Rough-DR-v0", cfg=cfg)
    try:
        base = env.unwrapped
        terrain = base.scene.terrain
        assignments = terrain.friction_assignments
        assert len(assignments) == 16, "Default B map should still have 16 identical collision regions"
        expected = sample_ground_friction(16, cfg.scene.terrain.friction_seed, (0.4, 1.5), (0.75, 1.0))
        stage = sim_utils.get_current_stage()
        # The full render/query mesh must NOT accidentally become a collider.
        query = stage.GetPrimAtPath("/World/ground/terrain/mesh")
        assert not query.HasAPI(UsdPhysics.CollisionAPI)
        assert len(UsdGeom.Mesh(query).GetFaceVertexCountsAttr().Get()) == 3_484_800

        def verify_materials():
            for row, pair in zip(assignments, expected):
                collider = stage.GetPrimAtPath(row["collider"])
                bound, _ = UsdShade.MaterialBindingAPI(collider).ComputeBoundMaterial(materialPurpose="physics")
                assert str(bound.GetPath()) == row["material"]
                physics = UsdPhysics.MaterialAPI(bound.GetPrim())
                actual = [physics.GetStaticFrictionAttr().Get(), physics.GetDynamicFrictionAttr().Get()]
                np.testing.assert_allclose(actual, pair, rtol=1e-6)
                assert physics.GetRestitutionAttr().Get() == 0.0
                assert PhysxSchema.PhysxMaterialAPI(bound.GetPrim()).GetFrictionCombineModeAttr().Get() == "average"

        verify_materials()
        with torch.inference_mode():
            obs, _ = env.reset()
            assert obs["policy"].shape == (args.num_envs, 60)
            assert base.action_manager.total_action_dim == 8
            assert torch.isfinite(obs["policy"]).all()
            ground = base.scene["ground_height"].data.ray_hits_w[:, 0, 2]
            assert torch.isfinite(ground).all()
            clearance = base.scene["robot"].data.root_pos_w[:, 2] - ground
            assert (clearance >= 0.49).all()
            base._reset_idx(torch.tensor([0, args.num_envs - 1], device=base.device))
            actions = torch.zeros(args.num_envs, 8, device=base.device)
            for _ in range(args.steps):
                if not app.is_running():
                    raise RuntimeError("Simulator closed before verification completed")
                obs, rewards, _, _, _ = env.step(actions)
                assert torch.isfinite(obs["policy"]).all() and torch.isfinite(rewards).all()
                assert torch.isfinite(base.scene["ground_height"].data.ray_hits_w).all()
            verify_materials()
        print(f"[CHECK] PASS: {len(assignments)} bound ground materials, 60D/8D, reset and {args.steps} physics steps", flush=True)
    finally:
        env.close()


try:
    main()
except BaseException:
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    raise
else:
    sys.stdout.flush()
    sys.stderr.flush()
    app.close()
