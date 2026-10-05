# Copyright (c) 2026, Ant generalization assignment contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Matched training experiment: WaveRange terrain with/without robot friction DR."""

from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from .ant_env_cfg import EventCfg
from .ant_wave_range_env_cfg import AntWaveRangeEnvCfg
from .friction_train_mdp import randomize_training_material_per_env


@configclass
class AntFrictionTrainEventsCfg(EventCfg):
    robot_material = EventTermCfg(
        func=randomize_training_material_per_env,
        mode="startup",
        params={
            # Omit body_names: SceneEntityCfg's slice(None) selects all shapes.
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.2, 1.8),
            "dynamic_friction_range": (0.15, 1.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,
            "material_seed": 42017,
        },
    )


@configclass
class AntFrictionControlEnvCfg(AntWaveRangeEnvCfg):
    """Same original WaveRange conditions, with a matched continuation budget."""


@configclass
class AntFrictionTrainEnvCfg(AntWaveRangeEnvCfg):
    """Only add independently sampled robot friction on the training map.

    Ground materials and their average combine rule remain unchanged. These
    ranges are robot coefficients, not the resulting contact coefficients.
    No team evaluation map, seed or material event is imported into training.
    """

    events: AntFrictionTrainEventsCfg = AntFrictionTrainEventsCfg()
