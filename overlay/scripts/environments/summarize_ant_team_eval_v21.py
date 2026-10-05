"""Validate completed v2.1 files and fill the distributed CSV; never run policies."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re

import yaml

ROOT = Path(__file__).resolve().parents[2]
LEVELS = {
    "SeenControl": ("seen_control", 51004),
    "UnseenEasy": ("unseen_easy", 53001),
    "UnseenMedium": ("unseen_medium", 53012),
    "UnseenHard": ("unseen_hard", 53003),
}


def collect(output):
    manifest = json.loads((output / "checkpoint_manifest.json").read_text())
    checkpoint = Path(manifest["checkpoint"])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != manifest["checkpoint_sha256"]:
        raise ValueError("Frozen checkpoint changed")
    distribution = Path(manifest["distribution"])
    for name, digest in manifest["distribution_files"].items():
        if hashlib.sha256((distribution / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Distribution changed: {name}")
    with (distribution / "result_template.csv").open(newline="") as stream:
        fields = next(csv.reader(stream))
    rows = []
    for label, (level, terrain_seed) in LEVELS.items():
        task = f"Isaac-Ant-TeamEvalV21-Balance-{label}-v0"
        result = json.loads((output / f"{label}.json").read_text())
        smoke = json.loads((output / "smoke" / f"{label}.json").read_text())
        expected = dict(task=task, seed=24, num_envs=2, passed=True,
                        checkpoint_sha256=manifest["checkpoint_sha256"],
                        observation_dimension=123, action_dimension=8,
                        max_init_terrain_level=4)
        for key, value in expected.items():
            if smoke.get(key) != value:
                raise ValueError(f"{label}: smoke mismatch for {key}")
        expected = dict(task=task, seed=24, terrain_seed=terrain_seed, completed=100,
                        max_episode_steps=960, observation_dimension=123,
                        reward_protocol="original_Ant_7_terms_v1",
                        checkpoint_sha256=manifest["checkpoint_sha256"])
        for key, value in expected.items():
            if result.get(key) != value:
                raise ValueError(f"{label}: result mismatch for {key}")
        if Path(result["checkpoint"]).resolve() != checkpoint.resolve():
            raise ValueError(f"{label}: wrong checkpoint path")
        cfg = yaml.load((output / f"{label}.env.yaml").read_text(), Loader=yaml.BaseLoader)
        terrain = cfg["scene"]["terrain"]
        if not math.isclose(result["step_dt"], 1 / 60) or float(cfg["episode_length_s"]) != 16.:
            raise ValueError(f"{label}: expected 16 seconds at 60 policy steps/s")
        if int(terrain["max_init_terrain_level"]) != 4 or cfg["curriculum"] != "null":
            raise ValueError(f"{label}: invalid terrain level or runtime curriculum")
        if cfg["events"]["reset_base"]["params"]["pose_range"] != {}:
            raise ValueError(f"{label}: unexpected reset pose override")
        term = cfg["terminations"]["torso_height"]
        if not term["func"].endswith(":root_height_below_minimum") or float(term["params"]["minimum_height"]) != .31:
            raise ValueError(f"{label}: expected original world-Z termination")
        weights = {k: float(v["weight"]) for k, v in cfg["rewards"].items()}
        if weights != dict(progress=1., alive=.5, upright=.1, move_to_target=.5,
                           action_l2=-.005, energy=-.05, joint_pos_limits=-.1):
            raise ValueError(f"{label}: expected seven original rewards")
        log = (output / f"{label}.log").read_text(errors="replace")
        origins = re.findall(r"v2\.1 origins x:\s*([-\d.]+)\s*\.\.\s*([-\d.]+) m", log)
        if not origins or any(float(high) > -124. + 1e-4 for low, high in origins):
            raise ValueError(f"{label}: origin boundary validation missing/failed")
        if "Completed first episodes: 100/100" not in log:
            raise ValueError(f"{label}: incomplete first episodes")
        episodes = result["episodes"]
        if len(episodes) != 100 or {e["env_id"] for e in episodes} != set(range(100)):
            raise ValueError(f"{label}: invalid episode IDs")
        if any(not 1 <= e["steps"] <= 960 for e in episodes):
            raise ValueError(f"{label}: invalid episode length")
        survival = sum(e["steps"] == 960 and e["timed_out"] and not e["terminated"] for e in episodes) / 100
        s = result["summary"]
        if not math.isclose(survival, s["timeout_without_failure_rate"], abs_tol=1e-9):
            raise ValueError(f"{label}: inconsistent 960-step survival")
        values = [s[name][stat] for name in ("reward", "steps", "forward_displacement_m") for stat in ("mean", "std")]
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"{label}: nonfinite statistics")
        rows.append(dict(
            member=manifest["member"], checkpoint_name=checkpoint.name,
            checkpoint_sha256=manifest["checkpoint_sha256"], task_id=task,
            environment=level, seed=24, num_envs=100, obs_dim=123, action_dim=8,
            reward_mean=s["reward"]["mean"], reward_std=s["reward"]["std"],
            steps_mean=s["steps"]["mean"], steps_std=s["steps"]["std"],
            survival_rate=survival, forward_mean_m=s["forward_displacement_m"]["mean"],
            forward_std_m=s["forward_displacement_m"]["std"], completed_episodes=100,
            notes=f"v2.1; original 7 rewards/world-Z termination; original reset; origin_x_max={origins[-1][1]}; missing_ground={s['missing_ground_episodes']}; missing_scan={s['missing_scan_episodes']}",
        ))
    with (output / "result_template.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[RESULT] CSV: {output / 'result_template.csv'}")
    for row in rows:
        print(f"{row['environment']}: reward={row['reward_mean']:.6f} ± {row['reward_std']:.6f}, survival={row['survival_rate']:.0%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    collect(parser.parse_args().output.resolve())
