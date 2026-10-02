# Copyright (c) 2026, Ant generalization assignment contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Seed torch-based grid generation and bind profile materials to actual faces."""

import numpy as np
import torch
import trimesh

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from .six_terrain_spec import PROFILES, TRAIN_PROFILES
from .terrain import AntTerrainGenerator, AntTerrainImporter, _define_mesh, collision_chunks


class SixTerrainGenerator(AntTerrainGenerator):
    def __init__(self, cfg, device="cpu"):
        if cfg.border_width != 0 or cfg.curriculum or cfg.use_cache or cfg.seed is None:
            raise ValueError("Six-map suite requires an explicit seed, no outer border/curriculum/cache")
        self.tile_records = []
        self._profile_by_config = {id(c): key.split("__", 1)[0] for key, c in cfg.sub_terrains.items()}
        # MeshRandomGridTerrainCfg uses torch RNG, unlike the height-field generators.
        # Keep map generation independent of the robot reset / policy RNGs.
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(cfg.seed)
            super().__init__(cfg, device=device)
        if sum(r["faces"] for r in self.tile_records) != len(self.terrain_mesh.faces):
            raise RuntimeError("Terrain face/material bookkeeping mismatch")
        self.terrain_mesh.metadata["six_tiles"] = self.tile_records

    def _add_sub_terrain(self, mesh, origin, row, col, sub_terrain_cfg):
        self.tile_records.append(dict(profile=self._profile_by_config[id(sub_terrain_cfg)],
                                      row=int(row), col=int(col), faces=len(mesh.faces)))
        super()._add_sub_terrain(mesh, origin, row, col, sub_terrain_cfg)


class SixTerrainImporter(AntTerrainImporter):
    """Same safe spawn-height handling as B; each collision face has one material."""

    def _import_collision_meshes(self, name, mesh):
        from pxr import Gf, UsdGeom, UsdPhysics, UsdShade

        records = mesh.metadata["six_tiles"]
        self.six_tile_records = records
        tile_groups = {p: [] for p in TRAIN_PROFILES}
        start = 0
        for record in records:
            end = start + record["faces"]
            tile_groups[record["profile"]].append(mesh.faces[start:end])
            start = end
        if start != len(mesh.faces):
            raise RuntimeError("Material groups do not cover mesh")
        stage = sim_utils.get_current_stage()
        path = f"{self.cfg.prim_path}/{name}"
        UsdGeom.Xform.Define(stage, path)
        _define_mesh(stage, f"{path}/mesh", mesh)
        if self.cfg.visual_material is not None:
            visual_path = f"{path}/visualMaterial"
            self.cfg.visual_material.func(visual_path, self.cfg.visual_material)
            sim_utils.bind_visual_material(f"{path}/mesh", visual_path)
        UsdGeom.Scope.Define(stage, f"{path}/colliders")
        UsdGeom.Scope.Define(stage, f"{path}/materials")
        self.six_materials = {}
        total = 0
        for profile, faces in tile_groups.items():
            if not faces:
                continue
            material = self.cfg.physics_material.copy()
            if self.cfg.suite_profile == "Mix":
                material.static_friction, material.dynamic_friction = PROFILES[profile]["friction"]
            mat_path = f"{path}/materials/{profile}"
            material.func(mat_path, material)
            self.six_materials[profile] = (material.static_friction, material.dynamic_friction)
            indices, remap = np.unique(np.concatenate(faces).ravel(), return_inverse=True)
            group_mesh = trimesh.Trimesh(mesh.vertices[indices], remap.reshape(-1, 3), process=False)
            # Keep the collider count bounded: splitting into tiny cells also
            # increases PhysX's aggregate-pair and scratch-memory requirements.
            for index, chunk in enumerate(
                collision_chunks(group_mesh, cell_size=self.cfg.collision_cell_size)
            ):
                chunk_path = f"{path}/colliders/{profile}_{index:04d}"
                center = chunk.bounds.mean(axis=0)
                chunk.apply_translation(-center)
                prim = _define_mesh(stage, chunk_path, chunk)
                UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(*center))
                prim.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
                sim_utils.define_collision_properties(
                    chunk_path, sim_utils.CollisionPropertiesCfg(collision_enabled=True)
                )
                UsdPhysics.MeshCollisionAPI.Apply(prim.GetPrim()).CreateApproximationAttr("none")
                sim_utils.bind_physics_material(chunk_path, mat_path)
                bound, _ = UsdShade.MaterialBindingAPI(prim.GetPrim()).ComputeBoundMaterial(materialPurpose="physics")
                if str(bound.GetPath()) != mat_path:
                    raise RuntimeError(f"Incorrect terrain material on {chunk_path}")
                total += 1
        self.terrain_prim_paths.append(path)
        print(f"[SIX] {self.cfg.suite_profile}: {len(mesh.faces):,} triangles, {total} colliders, "
              f"ground materials={self.six_materials}", flush=True)


@configclass
class SixTerrainImporterCfg(TerrainImporterCfg):
    class_type: type = SixTerrainImporter
    suite_profile: str = "T1"
    collision_cell_size: float = 132.0
    """XY partition size in metres; affects collision grouping, not terrain shape."""
