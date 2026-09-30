# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Mild rough-terrain Ant, separate from the original flat environment.

Terrain is fixed within a run. Friction remains 1.0 in this first experiment.
Ground clearance replaces world Z without increasing observation dimensions.
"""

import isaaclab.terrains as terrain_gen
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

from . import terrain_mdp
from .ant_env_cfg import AntEnvCfg, MySceneCfg, ObservationsCfg, TerminationsCfg
from .terrain import AntTerrainGenerator, AntTerrainImporter


@configclass
class AntRoughSceneCfg(MySceneCfg):
    terrain = MySceneCfg().terrain.replace(
        class_type=AntTerrainImporter,
        terrain_type="generator",
        max_init_terrain_level=None,
        terrain_generator=terrain_gen.TerrainGeneratorCfg(
            class_type=AntTerrainGenerator,
            # Override this independently of policy/reset seed for held-out maps.
            seed=42,
            size=(12.0, 12.0),
            num_rows=44,
            num_cols=44,
            border_width=0.0,
            horizontal_scale=0.4,
            vertical_scale=0.005,
            slope_threshold=None,
            curriculum=False,
            use_cache=False,
            sub_terrains={
                "flat": terrain_gen.HfPyramidSlopedTerrainCfg(
                    proportion=0.2, slope_range=(0.0, 0.0), platform_width=2.0
                ),
                "rough": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=0.4, noise_range=(-0.03, 0.03), noise_step=0.01, downsampled_scale=0.8
                ),
                "waves": terrain_gen.HfWaveTerrainCfg(
                    proportion=0.2, amplitude_range=(0.02, 0.08), num_waves=3
                ),
                # Slope is rise/run (0.10 is about 5.7 degrees), not degrees.
                "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                    proportion=0.1, slope_range=(0.02, 0.10), platform_width=2.0
                ),
                "inverted_slope": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                    proportion=0.1, slope_range=(0.02, 0.10), platform_width=2.0
                ),
            },
        ),
    )
    ground_height = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(0.0, 0.0)),
        mesh_prim_paths=["/World/ground/terrain/mesh"],
        debug_vis=False,
    )


@configclass
class AntRoughObservationsCfg(ObservationsCfg):
    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        base_height = ObsTerm(func=terrain_mdp.base_height_above_ground)

    policy: PolicyCfg = PolicyCfg()


@configclass
class AntRoughTerminationsCfg(TerminationsCfg):
    torso_height = DoneTerm(func=terrain_mdp.root_height_below_ground_minimum, params={"minimum_height": 0.31})


@configclass
class AntRoughEnvCfg(AntEnvCfg):
    # Keep the 5 m XY grid; RayCaster reset buffers need USD-visible clones.
    scene: AntRoughSceneCfg = AntRoughSceneCfg(num_envs=4096, env_spacing=5.0, clone_in_fabric=False)
    observations: AntRoughObservationsCfg = AntRoughObservationsCfg()
    terminations: AntRoughTerminationsCfg = AntRoughTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.ground_height.update_period = self.decimation * self.sim.dt
