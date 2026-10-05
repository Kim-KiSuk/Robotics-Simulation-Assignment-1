# Copyright (c) 2026, Ant generalization assignment contributors.
# SPDX-License-Identifier: BSD-3-Clause
"""Hold the first reset for inspection, then record a passive first episode.

No checkpoint, training, or official evaluation. Body centers and ground rays
are diagnostic evidence, NOT an exact collision-shape penetration test.
"""

import argparse
import json
import math
from pathlib import Path
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-Ant-Six-Eval-Blocks-HeightScan-v0")
parser.add_argument("--num_envs", type=int, default=100)
parser.add_argument("--seed", type=int, default=24)
parser.add_argument("--env_index", type=int, default=39)
parser.add_argument("--hold_seconds", type=float, default=30.0)
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--slowdown", type=float, default=4.0)
parser.add_argument(
    "--spawn_z_offset", type=float, default=0.0,
    help="Diagnostic-only fixed reset Z offset in metres; not a policy evaluation or permanent fix.",
)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not 0 <= args.env_index < args.num_envs or args.steps < 1:
    parser.error("Require 0 <= env_index < num_envs and steps >= 1")
if not math.isfinite(args.hold_seconds) or args.hold_seconds < 0:
    parser.error("hold_seconds must be finite and nonnegative")
if not math.isfinite(args.slowdown) or args.slowdown < 1:
    parser.error("slowdown must be finite and >= 1")
if not math.isfinite(args.spawn_z_offset) or not 0 <= args.spawn_z_offset <= 0.5:
    parser.error("spawn_z_offset must be finite and in [0, 0.5] metres")
args.output = args.output.expanduser().resolve()
if args.output.exists():
    parser.error(f"Output already exists: {args.output}")
app = AppLauncher(args).app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.managers import RecorderTerm, RecorderTermCfg
from isaaclab.managers.recorder_manager import DatasetExportMode, RecorderManagerBaseCfg
from isaaclab.utils import configclass
from isaaclab_tasks.utils import load_cfg_from_registry


def capture(env):
    i = args.env_index
    data = env.scene["robot"].data
    ground = env.scene["ground_height"].data.ray_hits_w[i, 0, 2]
    return dict(
        root_position=data.root_pos_w[i].detach().cpu().tolist(),
        root_quaternion_wxyz=data.root_quat_w[i].detach().cpu().tolist(),
        root_velocity=data.root_lin_vel_w[i].detach().cpu().tolist(),
        ground_z=float(ground), clearance_m=float(data.root_pos_w[i, 2] - ground),
        up=float(-data.projected_gravity_b[i, 2]),
        joint_positions=data.joint_pos[i].detach().cpu().tolist(),
        body_centers=data.body_pos_w[i].detach().cpu().tolist(),
        scan_hits=env.scene["height_scanner"].data.ray_hits_w[i].detach().cpu().tolist(),
    )


frames = []


class InitialStateRecorder(RecorderTerm):
    def record_post_step(self):
        env = self._env
        frame = capture(env)
        frame.update(step=int(env.episode_length_buf[args.env_index]),
                     terminated=bool(env.termination_manager.terminated[args.env_index]),
                     timed_out=bool(env.termination_manager.time_outs[args.env_index]))
        frames.append(frame)
        return None, None


@configclass
class InspectRecorderCfg(RecorderManagerBaseCfg):
    dataset_export_mode = DatasetExportMode.EXPORT_NONE
    export_in_record_pre_reset = False
    snapshot = RecorderTermCfg(class_type=InitialStateRecorder)


def finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, list):
        return [finite_json(v) for v in value]
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    return value


def main():
    cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    cfg.seed = args.seed
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    # Direct assignment avoids Hydra's None-to-string asset_name conversion.
    cfg.viewer.origin_type = "env"
    cfg.viewer.env_index = args.env_index
    cfg.viewer.eye = (-2.0, 2.0, 1.2)
    cfg.viewer.lookat = (0.0, 0.0, 0.3)
    if args.spawn_z_offset:
        # Change only this diagnostic instance's ordinary root-reset pose.
        # The existing reset draws the same number of random samples even for
        # a zero-width range. Do not move XY, change joint samples or terrain.
        from isaaclab.envs.mdp import reset_root_state_uniform

        if cfg.events.reset_base.func is not reset_root_state_uniform:
            raise ValueError("Z-offset diagnosis requires the stock reset_root_state_uniform event")
        pose_range = dict(cfg.events.reset_base.params["pose_range"])
        if pose_range.get("z", (0.0, 0.0)) != (0.0, 0.0):
            raise ValueError("Existing reset has a Z range; refusing to replace it")
        pose_range["z"] = (args.spawn_z_offset, args.spawn_z_offset)
        cfg.events.reset_base.params["pose_range"] = pose_range
    cfg.recorders = InspectRecorderCfg()
    env = gym.make(args.task, cfg=cfg)
    try:
        base = env.unwrapped
        with torch.inference_mode():
            # Like RslRlVecEnvWrapper: seed through cfg before construction,
            # then reset once without reseeding the initialization RNG stream.
            env.reset()
            initial = capture(base)
            print(f"[INSPECT] env={args.env_index}, initial clearance={initial['clearance_m']:.6f} m", flush=True)
            print(f"[INSPECT] Diagnostic-only reset Z offset: {args.spawn_z_offset:g} m", flush=True)
            print(f"[INSPECT] Holding for {args.hold_seconds:g}s without env.step; inspect legs and terrain now.", flush=True)
            end = time.monotonic() + args.hold_seconds
            while app.is_running() and time.monotonic() < end:
                base.sim.render()
                time.sleep(1 / 60)
            print("[INSPECT] Zero effort phase. Passive falling alone does NOT prove initial overlap.", flush=True)
            actions = torch.zeros(base.num_envs, base.action_manager.total_action_dim, device=base.device)
            for _ in range(args.steps):
                if not app.is_running():
                    break
                start = time.monotonic()
                env.step(actions)
                if frames[-1]["terminated"] or frames[-1]["timed_out"]:
                    break  # Frames were captured before this step's automatic reset.
                until = start + base.step_dt * args.slowdown
                while app.is_running() and time.monotonic() < until:
                    base.sim.render()
                    time.sleep(0.005)
            report = dict(
                task=args.task, seed=args.seed, num_envs=args.num_envs, env_id=args.env_index,
                terrain_seed=cfg.scene.terrain.terrain_generator.seed, step_dt=base.step_dt,
                diagnostic_spawn_z_offset_m=args.spawn_z_offset,
                body_names=base.scene["robot"].body_names, joint_names=base.scene["robot"].joint_names,
                initial=initial, frames=frames,
                protocol="one reset, initial hold without env.step, zero efforts, selected first episode only; post-physics/pre-reset frames",
                limitations="Not policy evaluation or an exact collision-shape overlap test. Zero-effort Ant can fall normally. Nonfinite values are null. Initial state is after simulator reset/forward, not before simulator construction.",
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x") as stream:
                json.dump(finite_json(report), stream, indent=2, allow_nan=False)
                stream.write("\n")
            print(f"[RESULT] Initial-state diagnostics: {args.output}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
