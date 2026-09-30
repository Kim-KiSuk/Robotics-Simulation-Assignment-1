# Copyright (c) 2026, Ant generalization assignment contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Validate experiment isolation, history resets, reward signs and terminal snapshots."""

import argparse
import sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
app = launcher.app

import gymnasium as gym
import torch
from types import SimpleNamespace
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg
from isaaclab_tasks.manager_based.classic.ant.ant_rough_env_cfg import AntRoughEnvCfg
from isaaclab_tasks.manager_based.classic.ant.ant_rough_dr_env_cfg import AntRoughDREnvCfg
from isaaclab_tasks.manager_based.classic.ant.evaluation_mdp import AntEvaluationRecorderCfg
from isaaclab_tasks.manager_based.classic.ant import robust_mdp


def main():
    baseline = AntRoughEnvCfg()
    original = baseline.to_dict()
    for name, history, shaped in (("Reward", False, True), ("History", True, False), ("History-Reward", True, True)):
        task = f"Isaac-Ant-Rough-{name}-v0"
        cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
        for field in ("scene", "actions", "events", "terminations", "sim"):
            assert getattr(cfg, field).to_dict() == getattr(baseline, field).to_dict(), field
        assert cfg.episode_length_s == baseline.episode_length_s
        assert (cfg.observations.policy.history_length == 4) == history
        assert (len(cfg.rewards.to_dict()) == len(baseline.rewards.to_dict()) + 3) == shaped
        agent = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
        old_agent = load_cfg_from_registry("Isaac-Ant-Rough-v0", "rsl_rl_cfg_entry_point")
        assert agent.algorithm.to_dict() == old_agent.algorithm.to_dict()
        assert agent.policy.to_dict() == old_agent.policy.to_dict()
    assert AntEnvCfg().scene.terrain.terrain_type == "plane"
    assert AntRoughEnvCfg().to_dict() == original
    assert AntRoughDREnvCfg().rewards.to_dict() == baseline.rewards.to_dict()
    print("[CHECK] A/B/C isolation and three experiment registrations: PASS", flush=True)

    # A small map is sufficient for reset/reward unit integration, not performance evaluation.
    cfg.scene.num_envs = 4
    cfg.scene.terrain.terrain_generator.num_rows = 4
    cfg.scene.terrain.terrain_generator.num_cols = 4
    cfg.sim.device = args.device
    cfg.seed = 42
    cfg.recorders = AntEvaluationRecorderCfg()
    env = gym.make("Isaac-Ant-Rough-History-Reward-v0", cfg=cfg)
    try:
        base = env.unwrapped
        with torch.inference_mode():
            obs, _ = env.reset()
            assert obs["policy"].shape == (4, 240)
            assert base.action_manager.total_action_dim == 8
            # B/C reward dictionaries must remain unchanged even after manager resolution.
            assert AntRoughEnvCfg().rewards.to_dict() == baseline.rewards.to_dict()
            risk = robust_mdp.low_clearance_risk(base)
            assert ((risk >= 0) & (risk <= 1)).all()
            seen_terminal = False
            for step in range(90):
                if step == 10:
                    # Force a low torso so the reset test does not depend on passive gait stability.
                    ids = torch.tensor([0], device=base.device)
                    pose = base.scene["robot"].data.root_state_w[ids, :7].clone()
                    pose[:, 2] = base.scene["ground_height"].data.ray_hits_w[ids, 0, 2] - 0.5
                    base.scene["robot"].write_root_pose_to_sim(pose, env_ids=ids)
                    base.scene["robot"].write_root_velocity_to_sim(torch.zeros(1, 6, device=base.device), env_ids=ids)
                action = torch.zeros(4, 8, device=base.device)
                obs, rewards, terminated, truncated, _ = env.step(action)
                assert torch.isfinite(obs["policy"]).all() and torch.isfinite(rewards).all()
                assert torch.allclose(robust_mdp.failure_impulse(base) * base.step_dt, terminated.float())
                assert torch.all(robust_mdp.action_change_l2(base) >= 0)
                if terminated.any():
                    seen_terminal = True
                    # Snapshot contains the fallen state, while the environment already reset.
                    ids = terminated.nonzero().flatten()
                    assert (base.ant_evaluation_snapshot.position[ids, 2] < base.scene["robot"].data.root_pos_w[ids, 2]).all()
                    # All four history samples of a reset term start at the NEW initial state.
                    history = base.observation_manager._group_obs_term_history_buffer["policy"]["base_height"].buffer
                    assert torch.allclose(history[ids], history[ids, :1].expand_as(history[ids]))
                    break
            assert seen_terminal, "Forced low torso must exercise auto-reset"
            # Time-limit-only steps carry zero failure impulse; a coincident fall remains penalized.
            fake = SimpleNamespace(
                termination_manager=SimpleNamespace(terminated=torch.tensor([False, True]), time_outs=torch.tensor([True, True])),
                step_dt=1 / 60,
            )
            assert torch.allclose(robust_mdp.failure_impulse(fake) * fake.step_dt, torch.tensor([0.0, 1.0]))
        print("[CHECK] 240 observations, finite rewards, terminal snapshots and history reset: PASS", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    else:
        sys.stdout.flush()
        sys.stderr.flush()
        app.close()
