# Copyright (c) 2026, Ant generalization assignment contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Small, optional reward terms; no changes to Ant physics or termination."""

import torch


def failure_impulse(env) -> torch.Tensor:
    """RewardManager multiplies by dt: weight=-5 gives -5 per failure.

    A pure time limit is not a failure. A fall on the last step still is.
    """
    return env.termination_manager.terminated.float() / env.step_dt


def low_clearance_risk(env, safe_height: float = 0.40, fall_height: float = 0.31) -> torch.Tensor:
    """Bounded quadratic penalty only when the torso approaches the ground."""
    if safe_height <= fall_height:
        raise ValueError("safe_height must exceed fall_height")
    ground_z = env.scene["ground_height"].data.ray_hits_w[:, 0, 2]
    height = env.scene["robot"].data.root_pos_w[:, 2] - ground_z
    height = torch.nan_to_num(height, nan=0.0, posinf=0.0, neginf=0.0)
    return ((safe_height - height) / (safe_height - fall_height)).clamp(0.0, 1.0).square()


def action_change_l2(env) -> torch.Tensor:
    """Weak penalty on changes in the eight effort commands."""
    return (env.action_manager.action - env.action_manager.prev_action).square().sum(dim=-1)


def excessive_tilt_risk(env, safe_up: float = 0.85, severe_up: float = 0.5) -> torch.Tensor:
    """Bounded risk from torso tilt, including a tilted but stationary torso.

    ``up`` is cos(tilt from world vertical), already available in Ant's state.
    No cost below about 31.8 degrees; quadratic growth to one at 60 degrees.
    Yaw, translation and vertical velocity are not penalized by this term.
    This is a rate: RewardManager applies weight * step_dt, not an event cost.
    Non-finite physics state is deliberately not hidden by a nan replacement.
    """
    if not -1.0 <= severe_up < safe_up <= 1.0:
        raise ValueError("Require -1 <= severe_up < safe_up <= 1")
    up = -env.scene["robot"].data.projected_gravity_b[:, 2]
    return ((safe_up - up) / (safe_up - severe_up)).clamp(0.0, 1.0).square()


def terrain_relative_vertical_motion(
    env, radius: float = 0.46, max_fit_error: float = 0.025,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vertical speed residual over a locally planar, finite height scan.

    Fit z = a*x + b*y + c to existing rays within radius of the torso.
    Return vz - (a*vx + b*vy), and a validity mask. Velocities/points are world
    frame quantities. No differencing across terrain edges or episode resets.
    Reject nonplanar support, sparse/degenerate samples and any missing scan.
    Plane-following ascent/descent has zero residual; this is NOT foot contact.
    """
    if not 0.0 < radius < float("inf") or not 0.0 < max_fit_error < float("inf"):
        raise ValueError("radius and max_fit_error must be finite and positive")
    robot = env.scene["robot"].data
    hits = env.scene["height_scanner"].data.ray_hits_w
    finite = torch.isfinite(hits).all(dim=(1, 2))
    # Zero replacement is for safe arithmetic only; the mask rejects such scans.
    points = torch.nan_to_num(hits, nan=0.0, posinf=0.0, neginf=0.0) - robot.root_pos_w[:, None, :]
    selected = points[:, :, :2].square().sum(dim=-1) <= radius**2
    weights = selected.to(points.dtype)
    count = weights.sum(dim=1).clamp_min(1.0)
    center = (points * weights[:, :, None]).sum(dim=1) / count[:, None]
    delta = points - center[:, None, :]
    x, y, z = delta.unbind(dim=-1)
    xx = (weights * x * x).sum(dim=1)
    yy = (weights * y * y).sum(dim=1)
    xy = (weights * x * y).sum(dim=1)
    xz = (weights * x * z).sum(dim=1)
    yz = (weights * y * z).sum(dim=1)
    det = xx * yy - xy.square()
    denominator = det.clamp_min(1.0e-8)
    slope_x = (xz * yy - yz * xy) / denominator
    slope_y = (yz * xx - xz * xy) / denominator
    error = (z - slope_x[:, None] * x - slope_y[:, None] * y).abs()
    error = torch.where(selected, error, torch.zeros_like(error)).amax(dim=1)
    valid = finite & (count >= 6) & (det > 1.0e-8) & (error <= max_fit_error)
    velocity = robot.root_lin_vel_w
    residual = velocity[:, 2] - slope_x * velocity[:, 0] - slope_y * velocity[:, 1]
    return residual, valid


def excess_vertical_motion(
    env, deadband: float = 0.5, max_excess: float = 2.0,
    radius: float = 0.46, max_fit_error: float = 0.025,
) -> torch.Tensor:
    """Bounded, training-only rate; do not penalize all uphill world-Z speed.

    With weight -0.05, cost lies in [-0.2, 0] per second (at most -3.2/16s).
    The deadband and coefficient are experimental, not learned optimum values.
    Unknown/nonplanar terrain gets no extra cost, not an artificial flat surface.
    """
    if not 0.0 <= deadband < float("inf") or not 0.0 < max_excess < float("inf"):
        raise ValueError("deadband must be finite/nonnegative and max_excess finite/positive")
    residual, valid = terrain_relative_vertical_motion(env, radius, max_fit_error)
    cost = (residual.abs() - deadband).clamp(0.0, max_excess).square()
    return torch.where(valid, cost, torch.zeros_like(cost))
