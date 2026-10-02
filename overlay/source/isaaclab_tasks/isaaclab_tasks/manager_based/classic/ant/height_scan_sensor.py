# Copyright (c) 2026, Ant generalization assignment contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Ideal height sensor whose addition does not consume the robot-reset RNG."""

from isaaclab.sensors import RayCaster, SensorBase


class IdealTerrainRayCaster(RayCaster):
    def __init__(self, cfg):
        if any(cfg.drift_range) or any(any(bounds) for bounds in cfg.ray_cast_drift_range.values()):
            raise ValueError("IdealTerrainRayCaster requires zero drift; use RayCaster for randomized drift")
        super().__init__(cfg)

    def reset(self, env_ids=None):
        # The stock reset samples even zero-width drift ranges, advancing the
        # global RNG before joint reset. This deterministic sensor needs no draws.
        SensorBase.reset(self, env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self.drift[ids] = 0.0
        self.ray_cast_drift[ids] = 0.0
