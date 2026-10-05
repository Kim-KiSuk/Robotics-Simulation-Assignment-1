# Copyright (c) 2026, Ant generalization assignment contributors.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from .ant_height_scan_ppo_cfg import AntHeightScanPPORunnerCfg


@configclass
class AntWaveRangePPORunnerCfg(AntHeightScanPPORunnerCfg):
    experiment_name = "ant_six_wave_range"
    max_iterations = 500
    algorithm = AntHeightScanPPORunnerCfg().algorithm.replace(
        learning_rate=1.0e-5, schedule="fixed", entropy_coef=0.001,
    )


@configclass
class AntWaveControlPPORunnerCfg(AntWaveRangePPORunnerCfg):
    experiment_name = "ant_six_wave_control"
