# Copyright (c) 2026, Ant generalization assignment contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Read-only evaluation snapshot, taken after physics and BEFORE auto-reset."""

import torch

from isaaclab.managers import RecorderTerm, RecorderTermCfg
from isaaclab.managers.recorder_manager import DatasetExportMode, RecorderManagerBaseCfg
from isaaclab.utils import configclass


class AntStepSnapshot(RecorderTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        env.ant_evaluation_snapshot = self
        self.position = torch.zeros(env.num_envs, 3, device=env.device)
        self.missing_ground = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def record_post_step(self):
        self.position.copy_(self._env.scene["robot"].data.root_pos_w)
        if "ground_height" in self._env.scene.sensors:
            hits = self._env.scene["ground_height"].data.ray_hits_w[:, 0, 2]
            self.missing_ground.copy_(~torch.isfinite(hits))
        return None, None


@configclass
class AntEvaluationRecorderCfg(RecorderManagerBaseCfg):
    dataset_export_mode = DatasetExportMode.EXPORT_NONE
    export_in_record_pre_reset = False
    snapshot = RecorderTermCfg(class_type=AntStepSnapshot)
