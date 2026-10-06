# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Ant-only terrain helpers; shared Isaac Lab terrain code is unchanged.

Preserves the Ant XY grid, adjusts spawn heights, and separates
the ray query surface from spatially partitioned triangle colliders.
"""

from __future__ import annotations

import numpy as np
import torch
import trimesh

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainGenerator, TerrainImporter
from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh


class AntTerrainGenerator(TerrainGenerator):
    """Skip zero-area border faces without patching the shared generator."""

    def __init__(self, cfg, device="cpu"):
        # Stock height-field functions use legacy np.random, while tile choices
        # use Generator. Seed both without consuming the reset/policy RNG state.
        state = np.random.get_state()
        try:
            if cfg.seed is not None:
                np.random.seed(cfg.seed)
            super().__init__(cfg, device=device)
        finally:
            np.random.set_state(state)

    def _add_terrain_border(self):
        if self.cfg.border_width > 0.0:
            super()._add_terrain_border()


def collision_chunks(mesh: trimesh.Trimesh, cell_size: float = 132.0, max_faces: int = 500_000):
    """Group nearby faces to keep each static collider's bounds local."""
    if cell_size <= 0.0 or max_faces <= 0:
        raise ValueError("cell_size and max_faces must be positive")
    if len(mesh.faces) == 0:
        return
    origin = mesh.bounds[0, :2]
    columns = max(1, int(np.ceil(mesh.extents[1] / cell_size)))
    cells = np.empty(len(mesh.faces), dtype=np.int64)
    for begin in range(0, len(mesh.faces), max_faces):
        end = min(begin + max_faces, len(mesh.faces))
        centers = mesh.vertices[mesh.faces[begin:end], :2].mean(axis=1)
        xy = np.floor((centers - origin) / cell_size).astype(np.int64)
        cells[begin:end] = xy[:, 0] * columns + np.minimum(xy[:, 1], columns - 1)
    order = np.argsort(cells, kind="stable")
    boundaries = np.r_[0, np.flatnonzero(np.diff(cells[order])) + 1, len(order)]
    for begin, end in zip(boundaries[:-1], boundaries[1:]):
        for start in range(begin, end, max_faces):
            faces = mesh.faces[order[start : min(start + max_faces, end)]]
            vertices, indices = np.unique(faces.reshape(-1), return_inverse=True)
            yield trimesh.Trimesh(vertices=mesh.vertices[vertices], faces=indices.reshape(-1, 3), process=False)


def _define_mesh(stage, path: str, mesh: trimesh.Trimesh):
    """Create geometry without implicitly making the full ray mesh a collider."""
    from pxr import UsdGeom, Vt

    prim = UsdGeom.Mesh.Define(stage, path)
    prim.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(mesh.vertices, dtype=np.float32)))
    prim.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(mesh.faces, dtype=np.int32).reshape(-1)))
    prim.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(mesh.faces), 3, dtype=np.int32)))
    prim.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.asarray(mesh.bounds, dtype=np.float32)))
    prim.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    return prim


class AntTerrainImporter(TerrainImporter):
    """Preserve the baseline XY grid and place each Ant above its footprint."""

    def import_mesh(self, name: str, mesh: trimesh.Trimesh):
        mesh.merge_vertices()
        origins = self._compute_env_origins_grid(self.cfg.num_envs, self.cfg.env_spacing)
        axis = torch.linspace(-0.5, 0.5, 5, device=self.device)
        x, y = torch.meshgrid(axis, axis, indexing="ij")
        offsets = torch.stack((x.flatten(), y.flatten(), torch.zeros_like(x.flatten())), dim=-1)
        starts = origins[:, None, :] + offsets[None, :, :]
        starts[:, :, 2] = float(mesh.bounds[1, 2]) + 10.0
        directions = torch.zeros_like(starts)
        directions[:, :, 2] = -1.0
        query_mesh = convert_to_warp_mesh(mesh.vertices, mesh.faces, device=self.device)
        hits = raycast_mesh(starts, directions, query_mesh)[0]
        if not torch.isfinite(hits).all():
            raise ValueError("Terrain misses Ant starting footprints. Increase rows/columns or reduce num_envs.")
        origins[:, 2] = hits[:, :, 2].max(dim=1).values
        self._grid_origins = origins
        del query_mesh
        self._import_collision_meshes(name, mesh)

    def _import_collision_meshes(self, name: str, mesh: trimesh.Trimesh):
        from pxr import Gf, UsdGeom, UsdPhysics

        stage = sim_utils.get_current_stage()
        path = f"{self.cfg.prim_path}/{name}"
        if path in self.terrain_prim_paths:
            raise ValueError(f"Terrain already exists: {path}")
        UsdGeom.Xform.Define(stage, path)
        _define_mesh(stage, f"{path}/mesh", mesh)
        if self.cfg.visual_material is not None:
            visual_path = f"{path}/visualMaterial"
            self.cfg.visual_material.func(visual_path, self.cfg.visual_material)
            sim_utils.bind_visual_material(f"{path}/mesh", visual_path)
        material_path = f"{path}/physicsMaterial"
        self.cfg.physics_material.func(material_path, self.cfg.physics_material)
        UsdGeom.Scope.Define(stage, f"{path}/colliders")
        count = 0
        for count, chunk in enumerate(collision_chunks(mesh), start=1):
            chunk_path = f"{path}/colliders/chunk_{count:04d}"
            center = chunk.bounds.mean(axis=0)
            chunk.apply_translation(-center)
            prim = _define_mesh(stage, chunk_path, chunk)
            UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(*center))
            prim.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
            sim_utils.define_collision_properties(chunk_path, sim_utils.CollisionPropertiesCfg(collision_enabled=True))
            UsdPhysics.MeshCollisionAPI.Apply(prim.GetPrim()).CreateApproximationAttr("none")
            sim_utils.bind_physics_material(chunk_path, material_path)
        self.terrain_prim_paths.append(path)
        print(f"[INFO] Ant rough terrain: {len(mesh.faces):,} triangles, {count} collision meshes.")

    def configure_env_origins(self, origins=None):
        if self.cfg.terrain_type == "generator":
            self.terrain_origins = None
            self.env_origins = self._grid_origins
        else:
            super().configure_env_origins(origins)
