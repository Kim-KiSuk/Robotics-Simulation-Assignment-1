"""Load only verified v2.1 distribution bytes; never modify the shared bundle."""

import hashlib
import os
import importlib.util
from pathlib import Path
import sys

DISTRIBUTION_ROOT = Path(os.environ.get(
    "TEAM_ANT_EVAL_V21_ROOT", "/home/kisuk/team_ant_eval_environment_only_v2_1"
)).expanduser().resolve()
FROZEN_SHA256 = {
    "team_eval_terrain_cfg_v2.py": "f514cdfeb99eb5e113bea223517bb1b5bdbf325ee6da9b8d1cdaefc2754b1716",
    "team_eval_material_event.py": "720d66fa582ba9a2c15fb82a4c9a234e47d8fd99532f369e407fd67ce65ffc5a",
}
TASKS = {
    "seen_control": "Isaac-Ant-TeamEvalV21-Balance-SeenControl-v0",
    "unseen_easy": "Isaac-Ant-TeamEvalV21-Balance-UnseenEasy-v0",
    "unseen_medium": "Isaac-Ant-TeamEvalV21-Balance-UnseenMedium-v0",
    "unseen_hard": "Isaac-Ant-TeamEvalV21-Balance-UnseenHard-v0",
}


def load_frozen_module(filename):
    path = DISTRIBUTION_ROOT / "environment" / filename
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != FROZEN_SHA256[filename]:
        raise RuntimeError(f"Frozen v2.1 distribution changed: {path}")
    name = f"{__package__}._v21_{path.stem}"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            exec(compile(source, str(path), "exec"), module.__dict__)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name]
