# Copyright (c) 2026, Ant generalization assignment contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Training-only friction variation; no dependency on the team evaluation bundle."""

import torch

from isaaclab.envs.mdp.events import randomize_rigid_body_material


class randomize_training_material_per_env(randomize_rigid_body_material):
    """One pair for all shapes of an Ant, sampled once at startup.

    A private CPU generator keeps material sampling from consuming the policy or
    reset RNG stream. The inherited initializer supplies the bounded material
    buckets and enforces dynamic <= static friction.
    """

    def __init__(self, cfg, env):
        self.generator = torch.Generator(device="cpu").manual_seed(cfg.params["material_seed"])
        with torch.random.fork_rng(devices=[]):
            # Set only CPU state; torch.manual_seed would also touch CUDA state.
            torch.random.set_rng_state(self.generator.get_state())
            super().__init__(cfg, env)
            self.generator.set_state(torch.random.get_rng_state())
        if self.asset_cfg.body_ids != slice(None):
            raise ValueError("Training friction event must select all robot bodies")

    def __call__(self, env, env_ids, static_friction_range, dynamic_friction_range,
                 restitution_range, num_buckets, asset_cfg, make_consistent, material_seed):
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device="cpu")
        else:
            env_ids = env_ids.cpu()
        view = self.asset.root_physx_view
        bucket_ids = torch.randint(len(self.material_buckets), (len(env_ids), 1),
                                   generator=self.generator, device="cpu")
        samples = self.material_buckets[bucket_ids].expand(-1, view.max_shapes, -1)
        materials = view.get_material_properties()
        materials[env_ids] = samples
        view.set_material_properties(materials, env_ids)
