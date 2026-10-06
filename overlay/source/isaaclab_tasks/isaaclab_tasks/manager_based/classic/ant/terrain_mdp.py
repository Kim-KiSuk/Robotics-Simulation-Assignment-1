# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Ground-relative observations and termination for the experimental Ant task.

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


class AntSpawnBank:
    """Training-only local spawn positions on the existing static ray mesh.

    Each environment keeps its own 5 m grid cell. Slot zero is the old XY;
    other slots are uniform XY offsets. Reject missing/out-of-map footprints,
    never a terrain type or steepness. Height uses the same 5x5 maximum as the
    original importer: this is sampled clearance, NOT full-body collision proof.
    A private CPU RNG leaves policy/joint/material random streams untouched.
    """

    def __init__(self, env, xy_range: float, bank_size: int, reset_seed: int):
        import hashlib
        import json
        from pathlib import Path

        import warp as wp
        from isaaclab.utils.warp import raycast_mesh

        if not 0.0 < xy_range < float("inf"):
            raise ValueError("xy_range must be finite and positive")
        if not isinstance(bank_size, int) or bank_size < 2:
            raise ValueError("bank_size must be an integer >= 2")
        if xy_range + 0.5 >= env.cfg.scene.env_spacing / 2:
            raise ValueError("Spawn footprint must stay inside its original grid cell")
        self.params = (xy_range, bank_size, reset_seed)
        self.rng = torch.Generator(device="cpu").manual_seed(reset_seed)
        self.last_slot = torch.full((env.num_envs,), -1, dtype=torch.long, device="cpu")
        sensor = env.scene["ground_height"]
        # Events are constructed before simulation startup; this constructor is
        # called lazily on the first reset, after RayCaster has initialized.
        _ = sensor.data
        if len(sensor.cfg.mesh_prim_paths) != 1:
            raise ValueError("Ant spawn bank requires the existing single static ground mesh")
        mesh = sensor.meshes[sensor.cfg.mesh_prim_paths[0]]
        vertices = wp.to_torch(mesh.points)  # view, no second Warp/BVH mesh allocation
        lower, upper = vertices.amin(dim=0), vertices.amax(dim=0)
        defaults = env.scene["robot"].data.default_root_state
        anchors = env.scene.env_origins[:, :2] + defaults[:, :2]
        offsets = (torch.rand((env.num_envs, bank_size, 2), generator=self.rng, device="cpu") * 2 - 1) * xy_range
        offsets[:, 0] = 0.0
        self.positions = torch.zeros(env.num_envs, bank_size, 3, device=env.device)
        self.positions[:, :, :2] = anchors[:, None, :] + offsets.to(env.device)
        valid = torch.zeros(env.num_envs, bank_size, dtype=torch.bool, device=env.device)
        axis = torch.linspace(-0.5, 0.5, 5, device=env.device)
        x, y = torch.meshgrid(axis, axis, indexing="ij")
        footprint = torch.stack((x.flatten(), y.flatten()), dim=-1)
        # Bound temporary query memory independently of the number of robots.
        for start in range(0, env.num_envs, 32):
            stop = min(start + 32, env.num_envs)
            xy = self.positions[start:stop, :, None, :2] + footprint[None, None, :, :]
            inside = ((xy >= lower[:2]) & (xy <= upper[:2])).all(dim=-1).all(dim=-1)
            rays = torch.zeros(stop - start, bank_size, 25, 3, device=env.device)
            rays[..., :2] = xy
            rays[..., 2] = upper[2] + 10.0
            rays = rays.reshape(-1, 25, 3)
            directions = torch.zeros_like(rays); directions[..., 2] = -1.0
            hits = raycast_mesh(rays, directions, mesh)[0].reshape(stop - start, bank_size, 25, 3)
            finite = torch.isfinite(hits).all(dim=-1).all(dim=-1)
            valid[start:stop] = inside & finite
            top = hits[..., 2].amax(dim=-1)
            self.positions[start:stop, :, 2] = top + defaults[start:stop, None, 2]
        self.valid = valid.cpu()
        counts = self.valid.sum(dim=1)
        if bool((counts < 2).any()):
            raise ValueError("Fewer than two valid spawn candidates for some robots; inspect map coverage")
        # Invalid slots cannot be selected. Keep all stored coordinates finite.
        self.positions[~valid] = 0.0
        report = dict(
            reset_seed=reset_seed, xy_range=xy_range, bank_size=bank_size, num_envs=env.num_envs,
            terrain_seed=env.scene.terrain.cfg.terrain_generator.seed,
            valid_candidates=int(counts.sum()), min_candidates=int(counts.min()), max_candidates=int(counts.max()),
            footprint="5x5, +/-0.5m; initial torso Z = sampled max ground + default root Z",
            rejected_criteria="missing ground or map bounds only; no terrain-type/slope filter",
            positions_sha256=hashlib.sha256(self.positions.cpu().numpy().tobytes()).hexdigest(),
        )
        self.report = report
        if getattr(env.cfg, "log_dir", None):
            path = Path(env.cfg.log_dir) / "spawn_bank_audit.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x") as stream:
                json.dump(report, stream, indent=2); stream.write("\n")
        print(f"[AntSpawn] {int(counts.sum())}/{env.num_envs * bank_size} valid candidates; "
              f"per-robot min={int(counts.min())}; private seed={reset_seed}")

    def sample(self, env_ids: torch.Tensor) -> torch.Tensor:
        ids = env_ids.to(device="cpu", dtype=torch.long)
        weights = self.valid[ids].float()
        previous = self.last_slot[ids]
        rows = torch.nonzero(previous >= 0, as_tuple=False).flatten()
        weights[rows, previous[rows]] = 0.0  # do not immediately repeat a slot
        slots = torch.multinomial(weights, 1, generator=self.rng).squeeze(-1)
        self.last_slot[ids] = slots
        device = self.positions.device
        return self.positions[ids.to(device), slots.to(device)].clone()


def reset_root_from_spawn_bank(
    env, env_ids: torch.Tensor | None, xy_range: float = 1.5,
    bank_size: int = 16, reset_seed: int = 52017,
):
    """Replace only training root position; retain stock orientation/velocity reset.

    Calling the original reset also preserves its RNG consumption before the
    unchanged joint reset. Both root writes occur before any physics step.
    Environment origins, default root state and all evaluation tasks stay fixed.
    """
    from isaaclab.envs.mdp import reset_root_state_uniform

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    else:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
    if env_ids.numel() == 0:
        return
    bank = getattr(env, "ant_training_spawn_bank", None)
    if bank is None:
        bank = AntSpawnBank(env, xy_range, bank_size, reset_seed)
        env.ant_training_spawn_bank = bank
    if bank.params != (xy_range, bank_size, reset_seed):
        raise ValueError("Spawn bank parameters cannot change within a run")
    positions = bank.sample(env_ids)
    reset_root_state_uniform(env, env_ids, pose_range={}, velocity_range={})
    asset = env.scene["robot"]
    pose = asset.data.default_root_state[env_ids, :7].clone()
    pose[:, :3] = positions
    asset.write_root_pose_to_sim(pose, env_ids=env_ids)
