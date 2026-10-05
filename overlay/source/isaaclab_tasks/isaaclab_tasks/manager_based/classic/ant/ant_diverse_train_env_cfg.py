# Copyright (c) 2026, Ant generalization assignment contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Training geometry/friction diversity with shaped or original rewards."""

from isaaclab.envs.mdp import ang_vel_xy_l2, push_by_setting_velocity
from isaaclab.managers import EventTermCfg, RewardTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from .ant_env_cfg import RewardsCfg
from .ant_forward_reward_env_cfg import AntForwardRewardEnvCfg
from .ant_friction_train_env_cfg import AntFrictionTrainEventsCfg
from .diverse_training_spec import TERRAIN_SEEDS, diversify_training_widths
from .robust_mdp import action_change_l2, excess_vertical_motion, excessive_tilt_risk, failure_impulse
from .terrain_mdp import reset_root_from_spawn_bank


@configclass
class AntDiverseTrainEnvCfg(AntForwardRewardEnvCfg):
    events: AntFrictionTrainEventsCfg = AntFrictionTrainEventsCfg()

    def __post_init__(self):
        super().__post_init__()
        generator = self.scene.terrain.terrain_generator
        self.scene.terrain.terrain_generator = generator.replace(
            seed=TERRAIN_SEEDS[0],
            sub_terrains=diversify_training_widths(generator.sub_terrains),
        )


@configclass
class AntDiversePushEventsCfg(AntFrictionTrainEventsCfg):
    # The local IsaacLab function adds sampled velocity to the current root
    # velocity. Apply only a small lateral delta; no forward-speed replacement.
    recovery_push = EventTermCfg(
        func=push_by_setting_velocity,
        mode="interval",
        interval_range_s=(6.0, 10.0),
        params={"asset_cfg": SceneEntityCfg("robot"), "velocity_range": {"y": (-0.15, 0.15)}},
    )


@configclass
class AntDiversePushTrainEnvCfg(AntDiverseTrainEnvCfg):
    events: AntDiversePushEventsCfg = AntDiversePushEventsCfg()


@configclass
class AntDiverseOriginalRewardEnvCfg(AntDiverseTrainEnvCfg):
    """Restore exactly the evaluation reward while retaining training diversity.

    A separate config preserves the prior ForwardReward/Diverse experiments.
    This has no extra failure or clearance penalty, and no recovery pushes.
    """

    rewards: RewardsCfg = RewardsCfg()


@configclass
class AntDiverseActionSmoothRewardsCfg(RewardsCfg):
    # Penalize abrupt changes in policy commands, without filtering or rescaling
    # the actual effort actions. RewardManager integrates this rate by step_dt.
    # Initial experimental coefficient; not a validated optimum.
    action_change = RewardTermCfg(func=action_change_l2, weight=-0.002)


@configclass
class AntDiverseActionSmoothEnvCfg(AntDiverseOriginalRewardEnvCfg):
    """Single-term reward experiment; terrain and the policy interface stay fixed."""

    rewards: AntDiverseActionSmoothRewardsCfg = AntDiverseActionSmoothRewardsCfg()


@configclass
class AntDiverseBalanceRewardsCfg(AntDiverseActionSmoothRewardsCfg):
    # A training-only rate penalty on body-frame roll/pitch angular speed.
    # Does not penalize vertical velocity, yaw or holding a static slope angle.
    # Initial experimental coefficient, not a validated optimum.
    body_rotation = RewardTermCfg(func=ang_vel_xy_l2, weight=-0.01)

    # Opt-in failure-only experiment on the existing Balance task. Zero keeps
    # the historical Balance reward unchanged; CLI weight=-2 charges exactly
    # -2 once on a true termination (including a fall at the time limit), not
    # on a normal timeout. failure_impulse cancels RewardManager's step_dt.
    # Do not increase progress weight or penalize successful terrain clearance.
    failure = RewardTermCfg(func=failure_impulse, weight=0.0)


@configclass
class AntDiverseBalanceEventsCfg(AntFrictionTrainEventsCfg):
    # Expose a typed Z range for Hydra without changing the default reset.
    # The opt-in lift experiment uses [0.15, 0.15]; historical Balance uses 0.
    # Keep the same stock reset function and its RNG draw count, XY, orientation,
    # velocities, joint reset and startup material randomization.
    reset_base = AntFrictionTrainEventsCfg().reset_base.replace(
        params={"pose_range": {"z": (0.0, 0.0)}, "velocity_range": {}},
    )


@configclass
class AntDiverseBalanceEnvCfg(AntDiverseActionSmoothEnvCfg):
    """Add one reward term; preserve geometry, observations, actions and resets."""

    rewards: AntDiverseBalanceRewardsCfg = AntDiverseBalanceRewardsCfg()
    events: AntDiverseBalanceEventsCfg = AntDiverseBalanceEventsCfg()


@configclass
class AntDiverseBalanceSpawnEventsCfg(AntFrictionTrainEventsCfg):
    reset_base = EventTermCfg(
        func=reset_root_from_spawn_bank, mode="reset",
        params={"xy_range": 1.5, "bank_size": 16, "reset_seed": 52017},
    )


@configclass
class AntDiverseBalanceSpawnEnvCfg(AntDiverseBalanceEnvCfg):
    """Change only training starts, preserving Balance reward and policy ABI."""

    events: AntDiverseBalanceSpawnEventsCfg = AntDiverseBalanceSpawnEventsCfg()


@configclass
class AntDiverseBalanceVerticalRewardsCfg(AntDiverseBalanceRewardsCfg):
    # Only change from the successful Balance: a small, bounded cost for
    # excess vertical speed on locally planar terrain. No posture/failure cost.
    vertical_motion = RewardTermCfg(
        func=excess_vertical_motion, weight=-0.05,
        params={"deadband": 0.5, "max_excess": 2.0, "radius": 0.46, "max_fit_error": 0.025},
    )


@configclass
class AntDiverseBalanceVerticalEnvCfg(AntDiverseBalanceEnvCfg):
    """Same 123D policy interface and unchanged terrain; a reward-only candidate."""

    rewards: AntDiverseBalanceVerticalRewardsCfg = AntDiverseBalanceVerticalRewardsCfg()


@configclass
class AntDiverseBalancePostureRewardsCfg(AntDiverseBalanceRewardsCfg):
    # One additional training-only term. Preserve uncapped original progress
    # and Balance's two smoothing terms. Do not add a terminal failure cost.
    # Risk is zero up to ~31.8 deg, then rises continuously to 60 deg.
    # At 45 deg the integrated cost is about -0.33 per second, max -2/s.
    # Initial experimental coefficient, not a demonstrated improvement.
    tilt_risk = RewardTermCfg(
        func=excessive_tilt_risk, weight=-2.0,
        params={"safe_up": 0.85, "severe_up": 0.5},
    )


@configclass
class AntDiverseBalancePostureEnvCfg(AntDiverseBalanceEnvCfg):
    """Reward-only scratch candidate; keep Balance's terrain and policy ABI."""

    rewards: AntDiverseBalancePostureRewardsCfg = AntDiverseBalancePostureRewardsCfg()


@configclass
class AntDiverseProgressSafeRewardsCfg(AntDiverseBalanceRewardsCfg):
    # Keep alive, posture, energy, joint limits and both smoothing terms intact.
    # Training-only hypothesis: modestly prefer progress, but charge true failure
    # once. failure_impulse cancels RewardManager's dt; a pure timeout costs zero.
    progress = RewardsCfg().progress.replace(weight=1.2)
    failure = RewardTermCfg(func=failure_impulse, weight=-5.0)


@configclass
class AntDiverseProgressSafeEnvCfg(AntDiverseBalanceEnvCfg):
    """Reward-only candidate for scratch training; evaluation stays unchanged."""

    rewards: AntDiverseProgressSafeRewardsCfg = AntDiverseProgressSafeRewardsCfg()
