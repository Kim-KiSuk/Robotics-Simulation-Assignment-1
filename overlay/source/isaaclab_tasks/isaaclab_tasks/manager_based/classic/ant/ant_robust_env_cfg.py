# Copyright (c) 2026, Ant generalization assignment contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Three controlled experiments on B: reward, history, and their combination."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from . import robust_mdp
from .ant_env_cfg import RewardsCfg
from .ant_rough_env_cfg import AntRoughEnvCfg, AntRoughObservationsCfg


@configclass
class AntRobustRewardsCfg(RewardsCfg):
    # Preserve the seven baseline terms, including uncapped progress.
    failure = RewTerm(func=robust_mdp.failure_impulse, weight=-5.0)
    clearance_risk = RewTerm(
        func=robust_mdp.low_clearance_risk, weight=-0.5,
        params={"safe_height": 0.40, "fall_height": 0.31},
    )
    action_change = RewTerm(func=robust_mdp.action_change_l2, weight=-0.002)


@configclass
class AntHistoryObservationsCfg(AntRoughObservationsCfg):
    @configclass
    class PolicyCfg(AntRoughObservationsCfg.PolicyCfg):
        # Four samples INCLUDING the current sample: 60 -> 240 dimensions.
        # IsaacLab stores history per term, then concatenates the terms.
        history_length = 4
        flatten_history_dim = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class AntRoughRewardEnvCfg(AntRoughEnvCfg):
    rewards: AntRobustRewardsCfg = AntRobustRewardsCfg()


@configclass
class AntRoughHistoryEnvCfg(AntRoughEnvCfg):
    observations: AntHistoryObservationsCfg = AntHistoryObservationsCfg()


@configclass
class AntRoughHistoryRewardEnvCfg(AntRoughHistoryEnvCfg):
    rewards: AntRobustRewardsCfg = AntRobustRewardsCfg()
