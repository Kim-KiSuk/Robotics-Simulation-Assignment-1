# Copyright (c) 2026, Ant generalization assignment contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Closed square blocks covering the whole tile without a flat border/platform."""

import numpy as np
import trimesh


def square_block_surface(difficulty, cfg):
    size = np.asarray(cfg.size, dtype=float)
    counts = np.rint(size / cfg.grid_width).astype(int)
    if np.any(counts < 2) or not np.allclose(counts * cfg.grid_width, size):
        raise ValueError("Block tile size must be an integer multiple of grid_width")
    if cfg.holes or cfg.platform_width != 0:
        raise ValueError("This block surface requires no holes and no central platform")
    amplitude = cfg.grid_height_range[0] + difficulty * (cfg.grid_height_range[1] - cfg.grid_height_range[0])
    if not 0 < amplitude < 1:
        raise ValueError("Block height amplitude must be between 0 and 1 metre")
    # NumPy RNG is isolated by AntTerrainGenerator. Geometry does not depend on
    # whether CUDA is visible, unlike the stock torch-based grid generator.
    heights = np.random.uniform(-amplitude, amplitude, tuple(counts))
    template = trimesh.creation.box(extents=(cfg.grid_width, cfg.grid_width, 1.0))
    vertices = np.tile(template.vertices, (counts.prod(), 1, 1))
    ii, jj = np.meshgrid(np.arange(counts[0]), np.arange(counts[1]), indexing="ij")
    centers = np.column_stack(((ii.ravel() + 0.5) * cfg.grid_width,
                               (jj.ravel() + 0.5) * cfg.grid_width))
    vertices[:, :, :2] += centers[:, None, :]
    top = template.vertices[:, 2] > 0
    vertices[:, top, 2] = heights.ravel()[:, None]
    vertices[:, ~top, 2] = -1.0
    faces = template.faces[None, :, :] + 8 * np.arange(counts.prod())[:, None, None]
    mesh = trimesh.Trimesh(vertices.reshape(-1, 3), faces.reshape(-1, 3), process=False)
    # The importer obtains actual spawn height by ray casting, not this origin.
    origin = np.array([size[0] / 2, size[1] / 2, heights.max()])
    return [mesh], origin
