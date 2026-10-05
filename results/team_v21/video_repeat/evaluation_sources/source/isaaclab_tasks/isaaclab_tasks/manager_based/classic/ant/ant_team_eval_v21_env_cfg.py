"""v2.1-only evaluation wrappers for the frozen 123D Balance scratch policy."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.utils import configclass

from .ant_diverse_train_env_cfg import AntDiverseBalanceEnvCfg
from .ant_env_cfg import EventCfg, RewardsCfg, TerminationsCfg
from .team_eval_v21_distribution import load_frozen_module
from .terrain import AntTerrainImporter

_terrain = load_frozen_module("team_eval_terrain_cfg_v2.py")
_material = load_frozen_module("team_eval_material_event.py")


class TeamV21TerrainImporter(TerrainImporter):
    """Native tile origins and RNG; retain the trained sensors' mesh path.

    Reuse only the collider/ray-mesh construction helper, NOT AntTerrainImporter's
    grid origins, footprint height adjustment or custom generator.
    """

    def import_mesh(self, name, mesh):
        AntTerrainImporter._import_collision_meshes(self, name, mesh)

    def configure_env_origins(self, origins=None):
        if self.cfg.max_init_terrain_level != 4:
            raise ValueError("v2.1 requires max_init_terrain_level=4")
        super().configure_env_origins(origins)
        if not torch.isfinite(self.env_origins).all():
            raise ValueError("v2.1 has non-finite terrain origins")
        if not ((self.terrain_levels >= 0) & (self.terrain_levels <= 4)).all():
            raise ValueError("v2.1 spawned outside terrain levels 0..4")
        if self.env_origins[:, 0].max().item() > -124.0 + 1e-4:
            raise ValueError("v2.1 origin x exceeds -124 m; evaluation is invalid")
        print(f"[INFO] v2.1 origins x: {self.env_origins[:, 0].min().item():.3f}"
              f" .. {self.env_origins[:, 0].max().item():.3f} m; levels 0..4")


@configclass
class TeamV21EventsCfg(EventCfg):
    robot_material = EventTermCfg(
        func=_material.randomize_rigid_body_material_per_env,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )


@configclass
class AntTeamV21BaseEnvCfg(AntDiverseBalanceEnvCfg):
    rewards: RewardsCfg = RewardsCfg()
    # v2.1 explicitly requires original world-Z termination. Keep the learned
    # ground-relative observation unchanged; observation and termination differ.
    terminations: TerminationsCfg = TerminationsCfg()
    events: TeamV21EventsCfg = TeamV21EventsCfg()

    def set_v21_level(self, level):
        previous = self.scene.terrain
        self.scene.terrain = TerrainImporterCfg(
            class_type=TeamV21TerrainImporter,
            prim_path=previous.prim_path,
            terrain_type="generator",
            collision_group=previous.collision_group,
            terrain_generator=_terrain.TEAM_V2_EVAL_TERRAINS[level].copy(),
            max_init_terrain_level=4,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0,
                friction_combine_mode="multiply", restitution_combine_mode="multiply",
                restitution=0.0,
            ),
            visual_material=previous.visual_material.copy(),
            debug_vis=False,
        )
        self.events.robot_material.params.update(_terrain.TEAM_V2_FRICTION_RANGES[level])
        self.seed = 24  # zero_agent.py has no --seed option; policy eval also passes --seed 24.
        self.scene.num_envs = 100
        self.episode_length_s = 16.0
        self.decimation = 2
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.curriculum = None  # Keep generator.curriculum=True: it defines the frozen geometry.
        # Fixed preview camera near the first spawn; does not track a falling robot.
        self.viewer.origin_type = "env"
        self.viewer.env_index = 0
        self.viewer.eye = (10.0, 10.0, 7.0)
        self.viewer.lookat = (3.0, 0.0, 0.0)


@configclass
class AntTeamV21SeenControlEnvCfg(AntTeamV21BaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.set_v21_level("seen_control")


@configclass
class AntTeamV21UnseenEasyEnvCfg(AntTeamV21BaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.set_v21_level("unseen_easy")


@configclass
class AntTeamV21UnseenMediumEnvCfg(AntTeamV21BaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.set_v21_level("unseen_medium")


@configclass
class AntTeamV21UnseenHardEnvCfg(AntTeamV21BaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.set_v21_level("unseen_hard")
