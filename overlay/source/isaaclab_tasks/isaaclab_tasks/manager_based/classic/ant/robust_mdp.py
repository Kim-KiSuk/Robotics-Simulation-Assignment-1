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
