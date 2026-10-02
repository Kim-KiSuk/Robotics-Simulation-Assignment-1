# Copyright (c) 2026, Ant generalization assignment contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Local terrain shape in metres, independent of world elevation and torso bounce."""

import torch


def relative_ground_scan(env):
    """63 terrain heights relative to the ground directly below the torso.

    Positive means higher terrain. Unknown rays use -1 m, the lower clipping
    limit, rather than pretending to be flat. This is an ideal simulated terrain
    query, not an RGB camera or a hardware-realistic depth sensor.
    """
    hits = env.scene["height_scanner"].data.ray_hits_w[:, :, 2]
    under_body = env.scene["ground_height"].data.ray_hits_w[:, :1, 2]
    heights = hits - under_body
    return torch.nan_to_num(heights, nan=-1.0, posinf=-1.0, neginf=-1.0).clamp(-1.0, 1.0)
