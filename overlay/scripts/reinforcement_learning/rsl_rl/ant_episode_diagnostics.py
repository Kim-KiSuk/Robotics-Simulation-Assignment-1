# Copyright (c) 2026, Ant generalization assignment contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""First-episode motion metrics; consumes PRE-reset snapshots, never reset poses."""

import torch


def summary(values):
    values = values.to(torch.float64)
    return {"mean": values.mean().item(), "std": values.std(unbiased=False).item()}


class FirstEpisodeDiagnostics:
    def __init__(self, start_position, target_xy=(1000.0, 0.0)):
        self.start = start_position.to(torch.float64).clone()
        self.endpoint = self.start.clone()
        self.target = torch.tensor(target_xy, dtype=torch.float64, device=self.start.device)
        self.finished = torch.zeros(len(self.start), dtype=torch.bool, device=self.start.device)
        self.failed = torch.zeros_like(self.finished)
        self.timed_out = torch.zeros_like(self.finished)
        self.missing_ground = torch.zeros_like(self.finished)
        self.missing_scan = torch.zeros_like(self.finished)

    def update(self, position_before_reset, terminated, time_out, missing_ground, missing_scan):
        active = ~self.finished
        ended = active & (terminated | time_out)
        self.endpoint[active] = position_before_reset[active].to(torch.float64)
        self.missing_ground[active] |= missing_ground[active]
        self.missing_scan[active] |= missing_scan[active]
        self.failed[ended] = terminated[ended]
        # A fall on the final step is a failure, even if the timeout also fires.
        self.timed_out[ended] = time_out[ended] & ~terminated[ended]
        self.finished |= ended

    def result(self, reward, steps, dt):
        if not self.finished.all():
            raise RuntimeError("Incomplete first episodes: diagnostic result will not be saved")
        forward = self.endpoint[:, 0] - self.start[:, 0]
        progress = (torch.linalg.vector_norm(self.start[:, :2] - self.target, dim=1)
                    - torch.linalg.vector_norm(self.endpoint[:, :2] - self.target, dim=1))
        duration = steps.to(torch.float64) * dt
        speed = progress / duration
        return {
            "summary": {
                "reward": summary(reward), "steps": summary(steps),
                "forward_displacement_m": summary(forward), "target_progress_m": summary(progress),
                "episode_duration_s": summary(duration), "target_speed_m_s": summary(speed),
                "timeout_without_failure_rate": self.timed_out.double().mean().item(),
                "termination_rate": self.failed.double().mean().item(),
                "missing_ground_episodes": int(self.missing_ground.sum().item()),
                "missing_scan_episodes": int(self.missing_scan.sum().item()),
            },
            "episodes": [
                dict(env_id=i, reward=reward[i].item(), steps=steps[i].item(),
                     forward_displacement_m=forward[i].item(), target_progress_m=progress[i].item(),
                     target_speed_m_s=speed[i].item(), timed_out=self.timed_out[i].item(),
                     terminated=self.failed[i].item(), missing_ground=self.missing_ground[i].item(),
                     missing_scan=self.missing_scan[i].item())
                for i in range(len(self.start))
            ],
        }
