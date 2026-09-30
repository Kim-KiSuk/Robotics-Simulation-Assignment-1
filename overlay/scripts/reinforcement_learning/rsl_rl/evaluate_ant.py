# Copyright (c) 2026, Ant generalization assignment contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate one complete episode per Ant using the ORIGINAL seven reward terms.

No learning or policy export. Saves per-episode JSON, resolved configs and a
checkpoint hash. Use the task matching the checkpoint's observation dimension.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

from isaaclab.app import AppLauncher
import cli_args

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-Ant-Rough-v0")
parser.add_argument("--num_envs", type=int, default=100)
parser.add_argument("--seed", type=int, default=24)
parser.add_argument("--output", required=True, help="New JSON output path; existing results are never overwritten.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
if args.num_envs < 1 or not args.checkpoint:
    parser.error("--checkpoint and a positive --num_envs are required")
output = Path(args.output).expanduser().resolve()
if output.exists() or output.with_suffix(".env.yaml").exists() or output.with_suffix(".agent.yaml").exists():
    parser.error(f"Output already exists: {output}")
sys.argv = [sys.argv[0]] + hydra_args
launcher = AppLauncher(args)
app = launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import RewardsCfg
from isaaclab_tasks.manager_based.classic.ant.evaluation_mdp import AntEvaluationRecorderCfg
from isaaclab_tasks.utils.hydra import hydra_task_config


def summary(values):
    values = values.to(torch.float64)
    return {"mean": values.mean().item(), "std": values.std(unbiased=False).item()}


@hydra_task_config(args.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    if args.task not in {
        "Isaac-Ant-v0", "Isaac-Ant-Rough-v0", "Isaac-Ant-Rough-DR-v0",
        "Isaac-Ant-Rough-Reward-v0", "Isaac-Ant-Rough-History-v0", "Isaac-Ant-Rough-History-Reward-v0",
    }:
        raise ValueError("This evaluator supports only this project's Ant tasks")
    if getattr(env_cfg.scene.terrain, "randomize_friction", False):
        raise ValueError("Use Isaac-Ant-Rough-v0 to evaluate C under a fixed, common friction condition")
    checkpoint = Path(args.checkpoint).expanduser().resolve(strict=True)
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args)
    env_cfg.seed = agent_cfg.seed
    env_cfg.scene.num_envs = args.num_envs
    if args.device is not None:
        env_cfg.sim.device = args.device
    # Deliberately override training reward shaping for a common comparison.
    env_cfg.rewards = RewardsCfg()
    env_cfg.recorders = AntEvaluationRecorderCfg()
    output.parent.mkdir(parents=True, exist_ok=True)
    dump_yaml(str(output.with_suffix(".env.yaml")), env_cfg)
    dump_yaml(str(output.with_suffix(".agent.yaml")), agent_cfg)
    env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    try:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(checkpoint), load_optimizer=False)
        policy = runner.get_inference_policy(device=env.device)
        base = env.unwrapped
        snapshot = base.ant_evaluation_snapshot
        with torch.inference_mode():
            obs = env.get_observations()
            start = base.scene["robot"].data.root_pos_w.clone()
            target_xy = torch.tensor([1000.0, 0.0], device=env.device)
            initial_distance = torch.linalg.vector_norm(start[:, :2] - target_xy, dim=1)
            reward_total = torch.zeros(env.num_envs, dtype=torch.float64, device=env.device)
            steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
            finished = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            failed = torch.zeros_like(finished)
            survived = torch.zeros_like(finished)
            missing_ground = torch.zeros_like(finished)
            endpoint = start.clone()
            for _ in range(env.max_episode_length):
                if not app.is_running():
                    break
                active = ~finished
                obs, reward, done, _ = env.step(policy(obs))
                reward_total[active] += reward[active]
                steps[active] += 1
                endpoint[active] = snapshot.position[active]
                missing_ground[active] |= snapshot.missing_ground[active]
                ended = active & done.bool()
                failed[ended] = base.reset_terminated[ended]
                survived[ended] = base.reset_time_outs[ended] & ~base.reset_terminated[ended]
                finished |= ended
                if finished.all():
                    break
            completed = int(finished.sum().item())
            if completed != env.num_envs:
                raise RuntimeError(f"Incomplete evaluation: {completed}/{env.num_envs}; no result JSON saved")
            delta_x = endpoint[:, 0] - start[:, 0]
            target_progress = initial_distance - torch.linalg.vector_norm(endpoint[:, :2] - target_xy, dim=1)
            duration = steps * base.step_dt
            speed = target_progress / duration
            result = {
                "schema_version": 1,
                "task": args.task,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "seed": env_cfg.seed,
                "terrain_seed": getattr(env_cfg.scene.terrain.terrain_generator, "seed", None),
                "static_friction": env_cfg.scene.terrain.physics_material.static_friction,
                "dynamic_friction": env_cfg.scene.terrain.physics_material.dynamic_friction,
                "reward_protocol": "original_Ant_7_terms_v1",
                "observation_dimension": int(obs["policy"].shape[-1]),
                "completed": completed,
                "step_dt": base.step_dt,
                "max_episode_steps": env.max_episode_length,
                "summary": {
                    "reward": summary(reward_total), "steps": summary(steps),
                    "forward_displacement_m": summary(delta_x),
                    "target_progress_m": summary(target_progress),
                    "target_speed_m_s": summary(speed),
                    "survival_rate": survived.float().mean().item(),
                    "failure_rate": failed.float().mean().item(),
                    "missing_ground_episodes": int(missing_ground.sum().item()),
                },
                "episodes": [
                    {"env_id": i, "reward": reward_total[i].item(), "steps": steps[i].item(),
                     "survived": survived[i].item(), "failed": failed[i].item(),
                     "missing_ground": missing_ground[i].item(),
                     "forward_displacement_m": delta_x[i].item(),
                     "target_progress_m": target_progress[i].item(), "target_speed_m_s": speed[i].item()}
                    for i in range(env.num_envs)
                ],
            }
        with output.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
        print(f"[INFO] Completed first episodes: {completed}/{env.num_envs}")
        for key in ("reward", "steps"):
            label = "Episode reward total" if key == "reward" else "Episode steps"
            stat = result["summary"][key]
            print(f"[RESULT] {label}: mean={stat['mean']:.6f}, std={stat['std']:.6f}")
        print(f"[RESULT] Survival rate: {result['summary']['survival_rate']:.6f}")
        print(f"[RESULT] JSON: {output}")
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
