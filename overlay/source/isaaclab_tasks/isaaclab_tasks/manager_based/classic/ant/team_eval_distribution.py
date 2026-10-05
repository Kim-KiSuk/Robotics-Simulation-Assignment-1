# Copyright (c) 2026, Ant generalization assignment contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Read the team's frozen distribution without copying or editing its files."""

import hashlib
import importlib.util
from pathlib import Path
import sys

DISTRIBUTION_ROOT = Path("/home/kisuk/team_eval_distribution_v1")
FROZEN_ENVIRONMENT_SHA256 = {
    "team_eval_terrain_cfg.py": "34a62242b2d5e7b07183c0d50938c931ddd3d4fdab3ef0eb3d286b9b070dd4b5",
    "team_eval_material_event.py": "a3d064aaa6730543aa6d87d04c873c5ed241605aaa4b0d66aacbef9ebb66e686",
}
TASKS = {
    "seen_control": "Isaac-Ant-TeamEvalV1-WaveRange-SeenControl-v0",
    "unseen_easy": "Isaac-Ant-TeamEvalV1-WaveRange-UnseenEasy-v0",
    "unseen_medium": "Isaac-Ant-TeamEvalV1-WaveRange-UnseenMedium-v0",
    "unseen_hard": "Isaac-Ant-TeamEvalV1-WaveRange-UnseenHard-v0",
}


def load_frozen_module(filename):
    path = DISTRIBUTION_ROOT / "environment" / filename
    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if digest != FROZEN_ENVIRONMENT_SHA256[filename]:
        raise RuntimeError(f"Frozen team evaluation file changed: {path}")
    name = f"{__package__}._distributed_{path.stem}"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            # Execute the verified bytes without writing __pycache__ into the
            # shared distribution, even when that directory happens to be writable.
            exec(compile(source, str(path), "exec"), module.__dict__)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name]
