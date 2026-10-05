# Copyright (c) 2026, Ant generalization assignment contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""User-run smoke test: two robots, strict checkpoint loading, one control step.

No training or episode-level policy scoring. Does not alter the distributed files.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--seed", type=int, default=24)
AppLauncher.add_app_launcher_args(parser)
args, overrides = parser.parse_known_args()
if overrides or args.num_envs != 2 or args.seed != 24:
    parser.error("Team smoke test requires seed 24, num_envs 2 and no configuration overrides")
if not args.checkpoint.is_file():
    parser.error(f"Checkpoint does not exist: {args.checkpoint}")
if args.output.exists():
    parser.error("Choose a new output path; existing reports are not overwritten")
sys.argv = [sys.argv[0]]
app = AppLauncher(args).app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner
from pxr import UsdGeom, UsdPhysics, UsdShade

import isaaclab.sim as sim_utils
import isaaclab_tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import EventCfg, RewardsCfg, TerminationsCfg
from isaaclab_tasks.manager_based.classic.ant.ant_diverse_train_env_cfg import AntDiverseBalanceEnvCfg
from isaaclab_tasks.manager_based.classic.ant.team_eval_v21_distribution import TASKS, load_frozen_module


@hydra_task_config(args.task, "rsl_rl_cfg_entry_point")
def main(cfg, agent):
    if args.task not in TASKS.values():
        raise ValueError("Expected one of the four TeamEvalV21-Balance tasks")
    level = next(k for k, v in TASKS.items() if v == args.task)
    frozen = load_frozen_module("team_eval_terrain_cfg_v2.py")
    original = AntDiverseBalanceEnvCfg()
    for field in ("observations", "actions"):
        assert getattr(cfg, field).to_dict() == getattr(original, field).to_dict(), field
    for field in ("robot", "ground_height", "height_scanner"):
        assert getattr(cfg.scene, field).to_dict() == getattr(original.scene, field).to_dict(), field
    # v2.1 uses the common original reset, not the training-only Z lift.
    original_events = EventCfg()
    for field in ("reset_base", "reset_robot_joints"):
        assert getattr(cfg.events, field).to_dict() == getattr(original_events, field).to_dict(), field
    assert cfg.rewards.to_dict() == RewardsCfg().to_dict()
    assert cfg.terminations.to_dict() == TerminationsCfg().to_dict()
    assert cfg.scene.terrain.max_init_terrain_level == 4
    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    manifest_path = Path(__file__).resolve().parents[2] / "docs/ant_team_eval_v21/checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if digest != manifest["checkpoint_sha256"] or args.checkpoint.resolve() != Path(manifest["checkpoint"]).resolve():
        raise ValueError("Checkpoint differs from the user-frozen final model")
    assert cfg.scene.terrain.terrain_generator.to_dict() == frozen.TEAM_V2_EVAL_TERRAINS[level].to_dict()
    assert cfg.curriculum is None and cfg.episode_length_s == 16.0
    assert cfg.decimation == 2 and cfg.sim.dt == 1 / 120
    cfg.scene.num_envs = 2
    cfg.seed = agent.seed = 24
    cfg.sim.device = agent.device = args.device
    wrapped = RslRlVecEnvWrapper(gym.make(args.task, cfg=cfg), clip_actions=agent.clip_actions)
    try:
        base = wrapped.unwrapped
        assert base.max_episode_length == 960
        assert base.scene.env_origins[:, 0].max().item() <= -124.0 + 1e-4
        assert base.scene.terrain.terrain_levels.max().item() <= 4
        runner = OnPolicyRunner(wrapped, agent.to_dict(), log_dir=None, device=agent.device)
        runner.load(str(args.checkpoint.resolve()), load_optimizer=False)
        policy = runner.get_inference_policy(device=base.device)
        with torch.inference_mode():
            observations, _ = wrapped.reset()
            assert observations["policy"].shape == (2, 123)
            assert base.action_manager.total_action_dim == 8
            materials = base.scene["robot"].root_physx_view.get_material_properties().clone()
            assert materials.shape[0] == 2 and materials.shape[1] > 0
            torch.testing.assert_close(materials, materials[:, :1].expand_as(materials), rtol=0, atol=0)
            ranges = frozen.TEAM_V2_FRICTION_RANGES[level]
            for column, name in ((0, "static_friction_range"), (1, "dynamic_friction_range")):
                lo, hi = ranges[name]
                assert ((materials[..., column] >= lo - 1e-6) & (materials[..., column] <= hi + 1e-6)).all()
            assert (materials[..., 1] <= materials[..., 0] + 1e-6).all()
            assert (materials[..., 2] == 0).all()
            stage = sim_utils.get_current_stage()
            colliders = list(stage.GetPrimAtPath("/World/ground/terrain/colliders").GetChildren())
            assert colliders
            for prim in colliders:
                assert prim.HasAPI(UsdPhysics.CollisionAPI)
                assert len(UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get()) > 0
                material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose="physics")
                api = UsdPhysics.MaterialAPI(material.GetPrim())
                assert api.GetStaticFrictionAttr().Get() == api.GetDynamicFrictionAttr().Get() == 1.0
                assert api.GetRestitutionAttr().Get() == 0.0
                assert material.GetPrim().GetAttribute("physxMaterial:frictionCombineMode").Get() == "multiply"
            actions = policy(observations)
            assert actions.shape == (2, 8) and torch.isfinite(actions).all()
            observations, rewards, _, _ = wrapped.step(actions)
            assert torch.isfinite(observations["policy"]).all() and torch.isfinite(rewards).all()
            for sensor in ("ground_height", "height_scanner"):
                assert torch.isfinite(base.scene[sensor].data.ray_hits_w).all()
            torch.testing.assert_close(materials, base.scene["robot"].root_physx_view.get_material_properties())
        result = dict(passed=True, scope="reset and one control step; not a full evaluation",
                      task=args.task, terrain_seed=cfg.scene.terrain.terrain_generator.seed,
                      checkpoint=str(args.checkpoint.resolve()),
                      checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                      seed=24, num_envs=2, observation_dimension=123, action_dimension=8,
                      world_z_termination_m=0.31,
                      origin_x_max=float(base.scene.env_origins[:, 0].max().item()),
                      max_init_terrain_level=4,
                      observation_order=base.observation_manager.active_terms["policy"],
                      reward_terms=base.reward_manager.active_terms,
                      robot_material_pairs=materials[:, 0, :].tolist(),
                      uniform_material_within_environment=True, original_rewards=True,
                      frozen_generator_config_unchanged=True, preserved_robot_actions_sensors=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"[CHECK] PASS: {result}", flush=True)
    finally:
        wrapped.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
