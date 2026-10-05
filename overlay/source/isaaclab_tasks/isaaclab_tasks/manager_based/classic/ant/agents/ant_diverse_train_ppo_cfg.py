# Copyright (c) 2026, Ant generalization assignment contributors.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .ant_forward_reward_ppo_cfg import AntForwardRewardPPORunnerCfg


@configclass
class AntDiverseTrainPPORunnerCfg(AntForwardRewardPPORunnerCfg):
    experiment_name = "ant_wave_diverse"
    max_iterations = 200


@configclass
class AntDiversePushTrainPPORunnerCfg(AntDiverseTrainPPORunnerCfg):
    experiment_name = "ant_wave_diverse_push"


@configclass
class AntDiverseOriginalRewardPPORunnerCfg(AntDiverseTrainPPORunnerCfg):
    experiment_name = "ant_wave_diverse_original"
    # Candidate for retaining future progress after obstacles. This is a
    # training-only discount, not an evaluation metric or policy input change.
    algorithm = AntDiverseTrainPPORunnerCfg().algorithm.replace(gamma=0.995)


@configclass
class AntDiverseActionSmoothPPORunnerCfg(AntDiverseOriginalRewardPPORunnerCfg):
    # Only the logging destination changes relative to the control runner.
    experiment_name = "ant_wave_diverse_action_smooth"


@configclass
class AntDiverseBalancePPORunnerCfg(AntDiverseActionSmoothPPORunnerCfg):
    experiment_name = "ant_wave_diverse_balance"


@configclass
class AntDiverseBalanceSpawnPPORunnerCfg(AntDiverseBalancePPORunnerCfg):
    """Same PPO and sample budget as the successful Balance scratch run."""

    experiment_name = "ant_wave_diverse_balance_spawn_scratch"
    max_iterations = 4000
    num_steps_per_env = 32
    resume = False
    algorithm = AntDiverseBalancePPORunnerCfg().algorithm.replace(
        learning_rate=5.0e-4, schedule="adaptive", num_mini_batches=4, lam=0.95,
    )


@configclass
class AntDiverseBalanceVerticalPPORunnerCfg(AntDiverseBalancePPORunnerCfg):
    """Match Balance scratch sample budget and PPO; never warm-start by default."""

    experiment_name = "ant_wave_diverse_balance_vertical_scratch"
    max_iterations = 4000
    num_steps_per_env = 32
    resume = False
    algorithm = AntDiverseBalancePPORunnerCfg().algorithm.replace(
        learning_rate=5.0e-4, schedule="adaptive", num_mini_batches=4, lam=0.95,
    )


@configclass
class AntDiverseBalancePosturePPORunnerCfg(AntDiverseBalancePPORunnerCfg):
    """Match the successful Balance scratch run, not continuation defaults."""

    experiment_name = "ant_wave_diverse_balance_posture_scratch"
    max_iterations = 4000
    num_steps_per_env = 32
    resume = False
    algorithm = AntDiverseBalancePPORunnerCfg().algorithm.replace(
        learning_rate=5.0e-4, schedule="adaptive", num_mini_batches=4, lam=0.95,
    )


@configclass
class AntDiverseProgressSafePPORunnerCfg(AntDiverseBalancePPORunnerCfg):
    # Match the executed Balance scratch run, not its continuation defaults.
    experiment_name = "ant_wave_diverse_progress_safe_scratch"
    max_iterations = 4000
    algorithm = AntDiverseBalancePPORunnerCfg().algorithm.replace(
        learning_rate=5.0e-4, schedule="adaptive",
    )


@configclass
class AntDiverseBalanceLongPPORunnerCfg(AntDiverseBalancePPORunnerCfg):
    """Scratch PPO candidate using exactly the existing Balance environment.

    1024*64*2000 matches the previous 1024*32*4000 sample budget.
    Eight minibatches preserve 8192 samples per optimizer step and 80000
    optimizer steps in total. Returns/advantages span a longer rollout; this
    does not change the actor input, action frequency, or episode time limit.
    """

    experiment_name = "ant_wave_diverse_balance_long_scratch"
    num_steps_per_env = 64
    max_iterations = 2000
    algorithm = AntDiverseBalancePPORunnerCfg().algorithm.replace(
        learning_rate=5.0e-4, schedule="adaptive", num_mini_batches=8, lam=0.97,
    )
