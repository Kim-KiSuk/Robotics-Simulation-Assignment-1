# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Spatial ground-friction randomization without changing B's geometry/colliders."""

from __future__ import annotations

import json

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from .friction_utils import sample_ground_friction
from .terrain import AntTerrainImporter


class AntFrictionTerrainImporter(AntTerrainImporter):
    """Assign one ground material to each existing spatial collision chunk.

    The default B map has 16 regions of 132 x 132 m. Their material map is fixed
    within a run (including resets), and changes with friction_seed. Robots can
    cross regions. This is not per-robot or per-episode material randomization.
    """

    def __init__(self, cfg):
        if cfg.randomize_friction:
            if cfg.terrain_type != "generator":
                raise ValueError("Ant spatial friction DR requires generated terrain")
            # Validate settings before expensive terrain generation/physics setup.
            sample_ground_friction(1, cfg.friction_seed, cfg.static_friction_range, cfg.dynamic_friction_ratio_range)
        super().__init__(cfg)

    def _import_collision_meshes(self, name, mesh):
        from pxr import UsdGeom, UsdPhysics, UsdShade

        # Reuse B's EXACT mesh creation, partitioning and spawn-height calculation.
        super()._import_collision_meshes(name, mesh)
        self.friction_assignments = []
        if not self.cfg.randomize_friction:
            return
        stage = sim_utils.get_current_stage()
        root = f"{self.cfg.prim_path}/{name}"
        colliders = sorted(
            stage.GetPrimAtPath(f"{root}/colliders").GetChildren(), key=lambda prim: str(prim.GetPath())
        )
        coefficients = sample_ground_friction(
            len(colliders), self.cfg.friction_seed,
            self.cfg.static_friction_range, self.cfg.dynamic_friction_ratio_range,
        )
        UsdGeom.Scope.Define(stage, f"{root}/frictionMaterials")
        for index, (collider, (static, dynamic)) in enumerate(zip(colliders, coefficients)):
            if not collider.HasAPI(UsdPhysics.CollisionAPI):
                raise RuntimeError(f"Missing collision API: {collider.GetPath()}")
            material_path = f"{root}/frictionMaterials/region_{index:04d}"
            # Restitution and combination rules remain identical to B.
            material = self.cfg.physics_material.replace(
                static_friction=float(static), dynamic_friction=float(dynamic)
            )
            material.func(material_path, material)
            sim_utils.bind_physics_material(str(collider.GetPath()), material_path)
            bound, _ = UsdShade.MaterialBindingAPI(collider).ComputeBoundMaterial(materialPurpose="physics")
            if str(bound.GetPath()) != material_path:
                raise RuntimeError(f"Failed to bind ground material: {collider.GetPath()}")
            self.friction_assignments.append({
                "collider": str(collider.GetPath()),
                "material": material_path,
                "static_friction": float(static),
                "dynamic_friction": float(dynamic),
            })
        print(
            f"[INFO] Ant spatial friction DR: seed={self.cfg.friction_seed}, regions={len(colliders)}, "
            f"static={coefficients[:, 0].min():.3f}..{coefficients[:, 0].max():.3f}, "
            f"dynamic={coefficients[:, 1].min():.3f}..{coefficients[:, 1].max():.3f}",
            flush=True,
        )
        print(f"[FRICTION_MAP] {json.dumps(self.friction_assignments)}", flush=True)


@configclass
class AntFrictionTerrainCfg(TerrainImporterCfg):
    class_type: type = AntFrictionTerrainImporter
    randomize_friction: bool = True
    friction_seed: int = 43
    static_friction_range: tuple[float, float] = (0.4, 1.5)
    dynamic_friction_ratio_range: tuple[float, float] = (0.75, 1.0)
