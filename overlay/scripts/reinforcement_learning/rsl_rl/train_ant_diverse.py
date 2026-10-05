# Copyright (c) 2026, Ant generalization assignment contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""User-run, foreground multi-map training. Preview only unless --execute is set."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[3]
ANT = ROOT / "source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant"
_spec = importlib.util.spec_from_file_location("diverse_training_spec", ANT / "diverse_training_spec.py")
SPEC = importlib.util.module_from_spec(_spec)
_spec_path = ANT / "diverse_training_spec.py"
# Planning must not create bytecode files as a side effect.
exec(compile(_spec_path.read_bytes(), str(_spec_path), "exec"), SPEC.__dict__)
TASKS = {
    "diverse": "Isaac-Ant-WaveRange-Diverse-v0",
    "pushes": "Isaac-Ant-WaveRange-Diverse-Push-v0",
    "control": "Isaac-Ant-WaveRange-ForwardReward-v0",
}
EXPERIMENTS = {
    "diverse": "ant_wave_diverse",
    "pushes": "ant_wave_diverse_push",
    "control": "ant_wave_diverse_control",
}
FATAL = re.compile(r"\[Error\]|Traceback|Scene state is corrupted|failed to allocate memory|"
                   r"simulation will miss interactions|CUDA.*(?:error|out of memory)|\bnan\b", re.I)


def stage_command(args, stage, terrain_seed, checkpoint, run_name):
    experiment = EXPERIMENTS[args.variant] + ("_validation" if args.smoke else "")
    iterations = 2 if args.smoke else args.iterations_per_stage
    envs = 16 if args.smoke else args.num_envs
    cmd = [str(ROOT / "isaaclab.sh"), "-p", "scripts/reinforcement_learning/rsl_rl/train.py",
           "--task", TASKS[args.variant], "--headless", "--num_envs", str(envs),
           "--max_iterations", str(iterations), "--seed", "42", "--warm_start", str(checkpoint),
           "--run_name", run_name, f"agent.experiment_name={experiment}",
           f"env.scene.terrain.terrain_generator.seed={terrain_seed}"]
    if stage == 0:
        cmd += ["--warm_start_min_std", "0.05"]
    if args.variant != "control":
        cmd += [f"env.events.robot_material.params.material_seed={42017 + stage}"]
    return cmd, experiment, iterations


def checkpoint_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_checkpoint(path):
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    for key in ("actor.0.weight", "critic.0.weight"):
        if tuple(state[key].shape) != (400, 123):
            raise ValueError(f"Not a compatible 123D checkpoint: {key}")
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError(f"Non-finite model state: {path}")
    return checkpoint_hash(path)


def stop_child(process):
    # Kill only the process group created by this launcher, never other jobs.
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def run_stage(cmd, log_path):
    with log_path.open("x") as log:
        process = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1, start_new_session=True)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
                if FATAL.search(line):
                    raise RuntimeError(f"Training error detected; inspect {log_path}: {line.strip()}")
            if process.wait() != 0:
                raise RuntimeError(f"Training returned {process.returncode}; inspect {log_path}")
        finally:
            stop_child(process)
            process.stdout.close()


def write_manifest(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--variant", choices=TASKS, default="diverse")
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--iterations_per_stage", type=int, default=SPEC.ITERATIONS_PER_STAGE)
    parser.add_argument("--smoke", action="store_true", help="Only first map, 16 environments, two training iterations")
    parser.add_argument("--execute", action="store_true", help="Start training in this terminal; otherwise print a plan")
    args = parser.parse_args()
    if args.num_envs < 1 or args.iterations_per_stage < 1:
        parser.error("Environment count and iteration count must be positive")
    checkpoint = args.checkpoint.expanduser().resolve()
    seeds = (1100,) * len(SPEC.STAGE_SEEDS) if args.variant == "control" else SPEC.STAGE_SEEDS
    if args.smoke:
        seeds = seeds[:1]
    if not args.execute:
        print("PREVIEW ONLY: no process launched and no files written.")
        for stage, seed in enumerate(seeds):
            cmd, _, _ = stage_command(args, stage, seed, checkpoint, f"preview_s{stage + 1}_t{seed}")
            print(shlex.join(cmd))
            checkpoint = Path(f"<checkpoint_saved_by_stage_{stage + 1}>")
        return
    if not checkpoint.is_file():
        parser.error(f"Checkpoint not yet available: {checkpoint}. Wait for the current training to finish.")
    source_hash = validate_checkpoint(checkpoint)
    run_id = f"{args.variant}_{uuid.uuid4().hex[:12]}"
    folder = ROOT / "logs/ant_diverse_pipeline" / run_id
    folder.mkdir(parents=True, exist_ok=False)
    manifest_path = folder / "manifest.json"
    manifest = dict(status="running", variant=args.variant, smoke=args.smoke,
                    source_checkpoint=str(checkpoint), source_sha256=source_hash,
                    terrain_seeds=list(seeds), policy_seed=42,
                    observation_dimension=123, action_dimension=8,
                    optimizer_reset_each_stage=True, stages=[])
    write_manifest(manifest_path, manifest)
    print(f"[PIPELINE] Records: {folder}", flush=True)
    try:
        for stage, seed in enumerate(seeds):
            run_name = f"{run_id}_s{stage + 1}_t{seed}"
            cmd, experiment, iterations = stage_command(args, stage, seed, checkpoint, run_name)
            item = dict(stage=stage + 1, terrain_seed=seed, input_checkpoint=str(checkpoint),
                        material_seed=None if args.variant == "control" else 42017 + stage,
                        command=cmd, status="running")
            manifest["stages"].append(item)
            write_manifest(manifest_path, manifest)
            print(f"[PIPELINE] Stage {stage + 1}/{len(seeds)}: {shlex.join(cmd)}", flush=True)
            run_stage(cmd, folder / f"stage_{stage + 1}.log")
            candidates = list((ROOT / "logs/rsl_rl" / experiment).glob(
                f"*_{run_name}/model_{iterations - 1}.pt"))
            if len(candidates) != 1:
                raise RuntimeError(f"Expected one new final checkpoint, got {candidates}")
            checkpoint = candidates[0].resolve()
            digest = validate_checkpoint(checkpoint)
            item.update(status="complete", output_checkpoint=str(checkpoint), output_sha256=digest)
            write_manifest(manifest_path, manifest)
        manifest.update(status="complete", final_checkpoint=str(checkpoint), final_sha256=digest)
        label = "SMOKE CHECKPOINT (not a final policy)" if args.smoke else "FINAL CHECKPOINT"
        print(f"[PIPELINE] {label}: {checkpoint}", flush=True)
        print(f"[PIPELINE] SHA-256: {digest}", flush=True)
    except BaseException as error:
        manifest.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        raise
    finally:
        write_manifest(manifest_path, manifest)


if __name__ == "__main__":
    main()
