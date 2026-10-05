# Copyright (c) 2026, Ant generalization assignment contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Matched continuation tasks; USD, robot, observations and rewards stay fixed."""

from isaaclab.utils import configclass

from .ant_height_scan_env_cfg import AntHeightScanTrainEnvCfg
from .wave_range_spec import broaden_wave_counts


@configclass
class AntWaveControlEnvCfg(AntHeightScanTrainEnvCfg):
    """Original HeightScan training environment with a separate run name."""


@configclass
class AntWaveRangeEnvCfg(AntHeightScanTrainEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        generator = self.scene.terrain.terrain_generator
        self.scene.terrain.terrain_generator = generator.replace(
            sub_terrains=broaden_wave_counts(generator.sub_terrains),
        )
