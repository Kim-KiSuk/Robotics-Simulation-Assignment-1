# Copyright (c) 2026, Ant generalization assignment contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Run a fixed, sequential control/treatment experiment and save original-reward evaluations.

Run inside the IsaacLab conda environment. No parallel GPU jobs, USD edits,
checkpoint selection on test scores, or automatic changes to the protocol.
"""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[3]
ANT = ROOT / "source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant"
DEFAULT_PARENT = ROOT / "logs/rsl_rl/ant_six_height/2026-10-02_05-21-39_six_mix_height_seed42/model_3999.pt"
EVAL_TASK = "Isaac-Ant-Six-Eval-Blocks-HeightScan-v0"
ARMS = {
    "control": ("Isaac-Ant-Six-TrainMix-HeightScan-Control-v0", "ant_six_wave_control"),
    "wave_range": ("Isaac-Ant-Six-TrainMix-HeightScan-WaveRange-v0", "ant_six_wave_range"),
}
FATAL = re.compile(
    r"Scene state is corrupted|failed to allocate memory|simulation will miss interactions|"
    r"CUDA error:|CUDA out of memory|Traceback \(most recent call last\)|[Bb]uffer overflow|"
    r"PhysX error:"
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def source_manifest():
    files = list(ANT.rglob("*.py"))
    files += list(Path(__file__).parent.glob("*.py"))
    files += [ROOT / "source/isaaclab_assets/isaaclab_assets/robots/ant.py"]
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(files)}


def run_stage(name, command, output, state, sources):
    if source_manifest() != sources:
        raise RuntimeError("Experiment source changed; refusing to mix code versions across stages")
    log = output / (name + ".log")
    state.update(stage=name, state="running", command=command, log=str(log))
    write_json(output / "status.json", state)
    print(f"[STAGE] {name}", flush=True)
    with log.open("x") as stream:
        proc = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                start_new_session=True, env={**os.environ, "TERM": "xterm"})
        state["child_pid"] = proc.pid
        write_json(output / "status.json", state)
        try:
            # Stop this experiment's process group if a physics/CUDA error appears.
            with log.open() as reader:
                while proc.poll() is None:
                    if FATAL.search(reader.read()):
                        raise RuntimeError(f"Physics/runtime error in {log}; see log before using the run")
                    time.sleep(1)
            code = proc.wait()
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
    text = log.read_text(errors="replace")
    if code or FATAL.search(text):
        raise RuntimeError(f"Stage {name} failed (exit={code}); see {log}")
    # Kit logs can contain simulation errors omitted from the console.
    kit_paths = [p.strip().strip("\"'") for p in re.findall(r"Logging to file:\s*([^\r\n]+)", text)]
    for raw in kit_paths:
        path = Path(raw)
        if path.is_file() and FATAL.search(path.read_text(errors="replace")):
            raise RuntimeError(f"Physics/runtime error in Kit log: {path}")
    state.setdefault("completed_stages", []).append(name)
    state.pop("child_pid", None)
    write_json(output / "status.json", state)


def audit_training(folder, iterations):
    import math
    import torch
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    checkpoint = folder / f"model_{iterations - 1}.pt"
    model = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert model["iter"] == iterations - 1
    assert model["model_state_dict"]["actor.0.weight"].shape[1] == 123
    assert all(torch.isfinite(t).all() for t in model["model_state_dict"].values())
    events = EventAccumulator(str(folder), size_guidance={"scalars": 0}).Reload()
    assert "Train/mean_reward" in events.Tags()["scalars"]
    for tag in events.Tags()["scalars"]:
        samples = events.Scalars(tag)
        assert len(samples) == iterations, (tag, len(samples))
        assert all(math.isfinite(s.value) for s in samples), tag
        if not tag.endswith("/time"):
            assert samples[-1].step == iterations - 1
    return dict(checkpoint=str(checkpoint), sha256=sha256(checkpoint), iterations=iterations,
                policy_tensors_finite=True, scalar_tags=len(events.Tags()["scalars"]))


def start_terrain_summary(data):
    """Classify the fixed 100-position E3 starting grid, not the failure position."""
    import numpy as np

    if data["task"] != EVAL_TASK or data["completed"] != 100:
        raise ValueError("Terrain grouping supports the E3 100-environment protocol only")
    keys = ["rough", "wave_3", "wave_5", "slope", "slope_inv", "blocks"]
    rng = np.random.default_rng(data["terrain_seed"])
    tiles = []
    for _ in range(40 * 40):
        tiles.append(keys[rng.choice(6, p=[0.25, 0.1, 0.1, 0.1, 0.1, 0.35])])
        rng.uniform(0, 1)
    groups = {k: [] for k in keys}
    for episode in data["episodes"]:
        i, j = divmod(episode["env_id"], 10)
        x, y = -(i - 4.5) * 5, (j - 4.5) * 5
        row, col = int((x + 160) // 8), int((y + 160) // 8)
        groups[tiles[row * 40 + col]].append(episode)
    return {kind: dict(count=len(es),
                       early_termination_2s=sum(e["terminated"] and e["steps"] <= 120 for e in es),
                       terminations=sum(e["terminated"] for e in es),
                       progress_mean_m=float(np.mean([e["target_progress_m"] for e in es])),
                       timeout_rate=float(np.mean([e["timed_out"] for e in es])))
            for kind, es in groups.items() if es}


def summarize(output, results):
    lines = ["# WaveRange 실험 결과", "", "원본 Ant 보상, seed 24, 환경 100개, 첫 에피소드.", "",
             "| 조건 | 모델 | 보상 mean ± std | 전진량 mean ± std (m) | 실패 없이 16초 | 2초 이내 종료 |",
             "|---|---|---:|---:|---:|---:|"]
    for name, model, path in results:
        data = json.loads(path.read_text())
        assert data["completed"] == 100 and data["reward_protocol"] == "original_Ant_7_terms_v1"
        s = data["summary"]
        reward, progress = s["reward"], s["target_progress_m"]
        early = sum(e["terminated"] and e["steps"] <= 120 for e in data["episodes"])
        lines.append(f"| {name} | {model} | {reward['mean']:.6f} ± {reward['std']:.6f} | "
                     f"{progress['mean']:.6f} ± {progress['std']:.6f} | "
                     f"{100*s['timeout_without_failure_rate']:.0f}% | {early}/100 |")
        write_json(path.with_name(path.stem + "_start_terrain.json"), start_terrain_summary(data))
    lines += ["", "대조군과 WaveRange는 같은 부모·학습량·PPO 설정을 사용한다. 학습 seed는 42 한 개다.",
              "이 표만으로 통계적 유의성이나 모든 미사용 지형에서의 개선을 주장하지 않는다.",
              "seed 9317은 개발용 검증, 9417은 본 실험 설정 고정 후 확인한 별도 평가 배치다.",
              "`*_start_terrain.json`은 시작 지형별 분류다. 종료 순간의 지형 분류가 아니며 작은 집단은 해석에 주의한다.", ""]
    (output / "RESULTS.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=500)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    parent = args.parent.resolve(strict=True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    state = dict(state="preparing", pid=os.getpid(), started=datetime.now(ZoneInfo("Asia/Seoul")).isoformat())
    sources = source_manifest()
    write_json(output / "source_manifest.json", sources)
    conditions = [("validation_9317", 9317, 0.9, 0.75), ("holdout_9417", 9417, 0.9, 0.75),
                  ("holdout_9417_low", 9417, 0.5, 0.35), ("holdout_9417_high", 9417, 1.4, 1.1)]
    protocol = dict(hypothesis="Broader training wave counts reduce early termination on short-wavelength terrain",
                    parent=str(parent), parent_sha256=sha256(parent), training_seed=42, evaluation_seed=24,
                    training_envs=1024, evaluation_envs=100, iterations_per_arm=args.iterations,
                    rollout_steps=32, training_terrain_seed=1100, warm_start_min_std=0.02,
                    robot_hardware_modified=False, usd_modified=False, reward="original Ant 7 terms",
                    sole_environment_change="T2 wave counts 2/4 -> 2/4/6/8; amplitude and T2 proportion unchanged",
                    conditions=conditions, checkpoint_selection="last iteration, fixed before evaluation",
                    training_seeds_repeated=False, stages=[])
    write_json(output / "protocol.json", protocol)
    models = {"parent": parent}
    try:
        for arm, (task, experiment) in ARMS.items():
            run_name = output.name + "_" + arm
            command = [str(ROOT / "isaaclab.sh"), "-p", "scripts/reinforcement_learning/rsl_rl/train.py",
                       "--task", task, "--headless", "--num_envs", "1024", "--max_iterations", str(args.iterations),
                       "--seed", "42", "--run_name", run_name, "--warm_start", str(parent),
                       "--warm_start_min_std", "0.02"]
            protocol["stages"].append(dict(name="train_" + arm, command=command))
            write_json(output / "protocol.json", protocol)
            run_stage("train_" + arm, command, output, state, sources)
            folders = list((ROOT / "logs/rsl_rl" / experiment).glob("*_" + run_name))
            if len(folders) != 1:
                raise RuntimeError(f"Expected one run for {run_name}, found {folders}")
            audit = audit_training(folders[0], args.iterations)
            write_json(output / (arm + "_training_audit.json"), audit)
            models[arm] = Path(audit["checkpoint"])
        results = []
        for name, seed, static, dynamic in conditions:
            for model, checkpoint in models.items():
                result = output / f"{model}_{name}.json"
                command = [str(ROOT / "isaaclab.sh"), "-p", "scripts/reinforcement_learning/rsl_rl/play_one_episode.py",
                           "--task", EVAL_TASK, "--checkpoint", str(checkpoint), "--headless", "--seed", "24",
                           "--num_envs", "100", "--results_file", str(result),
                           f"env.scene.terrain.terrain_generator.seed={seed}",
                           f"env.scene.terrain.physics_material.static_friction={static}",
                           f"env.scene.terrain.physics_material.dynamic_friction={dynamic}"]
                stage = "eval_" + model + "_" + name
                protocol["stages"].append(dict(name=stage, command=command))
                write_json(output / "protocol.json", protocol)
                run_stage(stage, command, output, state, sources)
                data = json.loads(result.read_text())
                assert data["completed"] == 100 and data["checkpoint_sha256"] == sha256(checkpoint)
                assert data["terrain_seed"] == seed
                assert data["static_friction"] == static and data["dynamic_friction"] == dynamic
                results.append((name, model, result))
                summarize(output, results)
        state.update(state="completed", finished=datetime.now(ZoneInfo("Asia/Seoul")).isoformat())
        write_json(output / "status.json", state)
    except BaseException as exc:
        state.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        state.pop("child_pid", None)
        write_json(output / "status.json", state)
        raise


if __name__ == "__main__":
    main()
