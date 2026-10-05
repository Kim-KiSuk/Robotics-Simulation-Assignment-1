# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Task-specific event terms for reproducible Ant evaluation."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs.mdp.events import randomize_rigid_body_material
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class randomize_rigid_body_material_per_env(randomize_rigid_body_material):
    """Assign one material sample to every collision shape in an environment."""

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        static_friction_range: tuple[float, float],
        dynamic_friction_range: tuple[float, float],
        restitution_range: tuple[float, float],
        num_buckets: int,
        asset_cfg: SceneEntityCfg,
        make_consistent: bool = True,
    ):
        del static_friction_range, dynamic_friction_range, restitution_range, asset_cfg, make_consistent

        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device="cpu")
        else:
            env_ids = env_ids.cpu()

        total_num_shapes = self.asset.root_physx_view.max_shapes
        bucket_ids = torch.randint(0, num_buckets, (len(env_ids), 1), device="cpu")
        material_samples = self.material_buckets[bucket_ids].expand(-1, total_num_shapes, -1)
        materials = self.asset.root_physx_view.get_material_properties()

        if self.num_shapes_per_body is not None:
            for body_id in self.asset_cfg.body_ids:
                start_idx = sum(self.num_shapes_per_body[:body_id])
                end_idx = start_idx + self.num_shapes_per_body[body_id]
                materials[env_ids, start_idx:end_idx] = material_samples[:, start_idx:end_idx]
        else:
            materials[env_ids] = material_samples

        self.asset.root_physx_view.set_material_properties(materials, env_ids)
