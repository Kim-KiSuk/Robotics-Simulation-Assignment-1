# Copyright (c) 2026, Ant generalization assignment contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Read-only evaluation snapshot, taken after physics and BEFORE auto-reset."""

import json
import os
from pathlib import Path

import torch

from isaaclab.managers import RecorderTerm, RecorderTermCfg
from isaaclab.managers.recorder_manager import DatasetExportMode, RecorderManagerBaseCfg
from isaaclab.utils import configclass

from .robust_mdp import terrain_relative_vertical_motion


class FirstEpisodeMotionAudit:
    """Optional read-only motion statistics, stopped at each FIRST termination.

    No contact sensor is added. Relative-height variation also contains terrain
    steps, so it must not be described as a pure jumping-height measurement.
    The existing evaluator's official reward/episode results are untouched.
    """

    def __init__(self, env, output):
        self.output = Path(output).expanduser().resolve()
        if self.output.exists():
            raise FileExistsError(f"Motion audit already exists: {self.output}")
        if "height_scanner" not in env.scene.sensors or "ground_height" not in env.scene.sensors:
            raise ValueError("Motion audit requires the existing height_scanner and ground_height sensors")
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.finished = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.failed = torch.zeros_like(self.finished)
        self.sums = {
            name: torch.zeros(env.num_envs, dtype=torch.float64, device=env.device)
            for name in ("steps", "finite_steps", "smooth_steps", "vz2", "relative_vz2",
                         "height", "height2", "action_delta2", "predicted_cost", "excess_steps")
        }
        self.written = False

    def update(self, env):
        if self.written:
            return
        active = ~self.finished
        data = env.scene["robot"].data
        height = data.root_pos_w[:, 2] - env.scene["ground_height"].data.ray_hits_w[:, 0, 2]
        vz = data.root_lin_vel_w[:, 2]
        finite = torch.isfinite(height) & torch.isfinite(vz)
        residual, supported = terrain_relative_vertical_motion(env)
        valid = finite & supported & torch.isfinite(residual)
        delta2 = (env.action_manager.action - env.action_manager.prev_action).square().sum(dim=-1)
        excess = (residual.abs() - 0.5).clamp(0.0, 2.0).square()
        values = {
            "steps": torch.ones_like(height), "finite_steps": finite,
            "smooth_steps": valid, "vz2": torch.where(finite, vz.square(), 0.0),
            "relative_vz2": torch.where(valid, residual.square(), 0.0),
            "height": torch.where(finite, height, 0.0),
            "height2": torch.where(finite, height.square(), 0.0),
            "action_delta2": delta2,
            "predicted_cost": torch.where(valid, -0.05 * excess * env.step_dt, 0.0),
            "excess_steps": valid & (residual.abs() > 0.5),
        }
        for name, value in values.items():
            self.sums[name] += torch.where(active, value.to(torch.float64), 0.0)
        ended = active & (env.termination_manager.terminated | env.termination_manager.time_outs)
        self.failed[ended] = env.termination_manager.terminated[ended]
        self.finished |= ended
        if bool(self.finished.all()):
            self.write(env)

    def write(self, env):
        s = {key: value.cpu().tolist() for key, value in self.sums.items()}
        failed = self.failed.cpu().tolist()
        episodes = []
        for i in range(len(failed)):
            steps, finite, smooth = s["steps"][i], s["finite_steps"][i], s["smooth_steps"][i]
            height_mean = s["height"][i] / finite if finite else None
            episodes.append(dict(
                env_id=i, steps=int(steps), terminated=failed[i],
                finite_height_velocity_fraction=finite / steps,
                planar_support_fraction=smooth / steps,
                vertical_speed_rms_m_s=(s["vz2"][i] / finite)**0.5 if finite else None,
                compensated_vertical_speed_rms_m_s=(s["relative_vz2"][i] / smooth)**0.5 if smooth else None,
                clearance_std_m=max(0.0, s["height2"][i] / finite - height_mean**2)**0.5 if finite else None,
                action_delta_l2_rms=(s["action_delta2"][i] / steps)**0.5,
                excess_vertical_motion_fraction=s["excess_steps"][i] / steps,
                hypothetical_vertical_penalty=s["predicted_cost"][i],
            ))
        report = dict(
            schema_version=1, completed=len(episodes), step_dt=env.step_dt,
            protocol="first episodes, post-physics/pre-reset; read-only, no changes to official reward",
            parameters=dict(radius=0.46, max_fit_error=0.025, deadband=0.5, max_excess=2.0, weight=-0.05),
            limitations="No foot-contact measurement. Clearance variation includes terrain edges; compensated RMS uses planar support only. Null means no valid samples.",
            episodes=episodes,
        )
        with self.output.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        self.written = True
        print(f"[RESULT] Motion diagnostics: {self.output}")


class AntStepSnapshot(RecorderTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        env.ant_evaluation_snapshot = self
        self.position = torch.zeros(env.num_envs, 3, device=env.device)
        self.missing_ground = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.missing_scan = torch.zeros_like(self.missing_ground)
        # Opt-in output path, passed by the user's evaluation command. The
        # normal evaluation path does no extra work and retains its interface.
        output = os.environ.get("ANT_MOTION_DIAGNOSTICS_FILE")
        self.motion_audit = FirstEpisodeMotionAudit(env, output) if output else None

    def record_post_step(self):
        self.position.copy_(self._env.scene["robot"].data.root_pos_w)
        if "ground_height" in self._env.scene.sensors:
            hits = self._env.scene["ground_height"].data.ray_hits_w[:, 0, 2]
            self.missing_ground.copy_(~torch.isfinite(hits))
        if "height_scanner" in self._env.scene.sensors:
            hits = self._env.scene["height_scanner"].data.ray_hits_w
            self.missing_scan.copy_(~torch.isfinite(hits).all(dim=(1, 2)))
        if self.motion_audit is not None:
            self.motion_audit.update(self._env)
        return None, None


@configclass
class AntEvaluationRecorderCfg(RecorderManagerBaseCfg):
    dataset_export_mode = DatasetExportMode.EXPORT_NONE
    export_in_record_pre_reset = False
    snapshot = RecorderTermCfg(class_type=AntStepSnapshot)
