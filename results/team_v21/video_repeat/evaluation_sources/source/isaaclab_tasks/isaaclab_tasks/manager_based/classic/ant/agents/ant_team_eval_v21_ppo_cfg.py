"""Match the final Failure2Lift scratch policy/algorithm; evaluation only."""

from isaaclab.utils import configclass

from .ant_diverse_train_ppo_cfg import AntDiverseBalancePPORunnerCfg


@configclass
class AntTeamV21BalancePPORunnerCfg(AntDiverseBalancePPORunnerCfg):
    experiment_name = "ant_wave_diverse_balance_failure2_lift_scratch"
    max_iterations = 6000
    algorithm = AntDiverseBalancePPORunnerCfg().algorithm.replace(
        learning_rate=5.0e-4, schedule="adaptive",
    )
