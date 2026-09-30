# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Ground-relative observations and termination for the experimental Ant task.

Design reference: https://github.com/Stick-0/isaac-ant-rough-terrain
The baseline task does not import these terms.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def base_height_above_ground(env: ManagerBasedRLEnv) -> torch.Tensor:
    """One scalar clearance; missing terrain produces zero, never NaN/Inf."""
    ground_z = env.scene["ground_height"].data.ray_hits_w[:, 0, 2]
    clearance = env.scene["robot"].data.root_pos_w[:, 2] - ground_z
    return torch.nan_to_num(clearance, nan=0.0, posinf=0.0, neginf=0.0).unsqueeze(-1)


def root_height_below_ground_minimum(env: ManagerBasedRLEnv, minimum_height: float) -> torch.Tensor:
    """End a fall relative to local terrain; a missing ray also ends the episode.

    Terrain is finite. Going outside it is counted as a failure, so evaluations
    must also inspect missing-ground counts rather than interpreting all failures
    as physical falls.
    """
    return base_height_above_ground(env).squeeze(-1) < minimum_height
