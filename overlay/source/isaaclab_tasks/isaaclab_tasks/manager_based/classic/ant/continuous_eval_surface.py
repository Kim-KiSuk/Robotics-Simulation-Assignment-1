# Copyright (c) 2026, Ant generalization assignment contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Reuse stock height functions without adding a flat padding strip to E2.

Adjacent tiles share a zero-height edge (one vertex line, not a platform).
The stock wrapper's extra zero padding is omitted. Small horizontal faces from
height quantization can remain; this is not a claim of nonzero slope everywhere.
"""

from copy import copy

import numpy as np
import trimesh


def _raw_height_function(cfg):
    # Lazy import keeps the geometry module usable by the CPU inspection tool.
    from isaaclab.terrains.height_field import hf_terrains

    if hasattr(cfg, "noise_range"):
        return hf_terrains.random_uniform_terrain.__wrapped__
    if hasattr(cfg, "num_waves"):
        return hf_terrains.wave_terrain.__wrapped__
    if hasattr(cfg, "slope_range"):
        return hf_terrains.pyramid_sloped_terrain.__wrapped__
    raise TypeError(f"Unsupported continuous height-field configuration: {type(cfg)}")


def continuous_surface(difficulty, cfg):
    size_x, size_y = cfg.size
    nx = int(round(size_x / cfg.horizontal_scale)) + 1
    ny = int(round(size_y / cfg.horizontal_scale)) + 1
    if not np.allclose(((nx - 1) * cfg.horizontal_scale, (ny - 1) * cfg.horizontal_scale), cfg.size):
        raise ValueError("E2 tile dimensions must be multiples of horizontal_scale")
    raw_cfg = copy(cfg)
    # Stock raw functions use int(size / resolution) samples; request both end
    # vertices of our physical tile, instead of padding the result with zeros.
    raw_cfg.size = (nx * cfg.horizontal_scale, ny * cfg.horizontal_scale)
    if hasattr(cfg, "slope_range"):
        # Preserve the specified slope-based peak height despite the extra sample.
        raw_cfg.slope_range = tuple(s * size_x / raw_cfg.size[0] for s in cfg.slope_range)
    heights = _raw_height_function(cfg)(difficulty, raw_cfg).copy()
    if heights.shape != (nx, ny):
        raise ValueError(f"Unexpected raw height field shape: {heights.shape}")
    heights[[0, -1], :] = 0
    heights[:, [0, -1]] = 0
    xx, yy = np.meshgrid(np.linspace(0, size_x, nx), np.linspace(0, size_y, ny), indexing="ij")
    vertices = np.column_stack((xx.ravel(), yy.ravel(), heights.ravel() * cfg.vertical_scale))
    ii, jj = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing="ij")
    k = (ii * ny + jj).ravel()
    faces = np.empty((2 * len(k), 3), dtype=np.int64)
    faces[0::2] = np.column_stack((k, k + ny + 1, k + 1))
    faces[1::2] = np.column_stack((k, k + ny, k + ny + 1))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    center = (np.abs(xx - size_x / 2) < 1.0) & (np.abs(yy - size_y / 2) < 1.0)
    origin = np.array([size_x / 2, size_y / 2, heights[center].max() * cfg.vertical_scale])
    return [mesh], origin
