# Copyright (c) 2026, Ant generalization assignment contributors.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .ant_wave_range_ppo_cfg import AntWaveRangePPORunnerCfg


@configclass
class AntFrictionTrainPPORunnerCfg(AntWaveRangePPORunnerCfg):
    experiment_name = "ant_wave_friction"
    max_iterations = 1000
    # Both arms use the same conservative continuation optimizer settings.
    algorithm = AntWaveRangePPORunnerCfg().algorithm.replace(
        learning_rate=1.0e-5, schedule="fixed", entropy_coef=0.001,
    )


@configclass
class AntFrictionControlPPORunnerCfg(AntFrictionTrainPPORunnerCfg):
    experiment_name = "ant_wave_friction_control"
