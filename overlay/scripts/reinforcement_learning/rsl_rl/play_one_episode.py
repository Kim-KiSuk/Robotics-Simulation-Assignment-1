# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play one episode from a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play one episode with an RL agent from RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--diagnostics", action="store_true", help="Ant: original rewards, first-episode distance and terminations.")
parser.add_argument("--results_file", type=Path, help="Ant: new result JSON path; implies --diagnostics; never overwrites.")
parser.add_argument("--video_folder", type=str, help="Optional recording directory, e.g. a separate folder for each map.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
if args_cli.results_file:
    args_cli.diagnostics = True
    args_cli.results_file = args_cli.results_file.expanduser().resolve()
    for path in (args_cli.results_file, args_cli.results_file.with_suffix(".env.yaml"),
                 args_cli.results_file.with_suffix(".agent.yaml")):
        if path.exists():
            parser.error(f"Result/config already exists: {path}; choose a new --results_file")
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play one episode with an RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # configure the viewer to track the Ant root
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (-4.0, 4.0, 2.5)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.5)

    if args_cli.diagnostics:
        from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg, RewardsCfg
        from isaaclab_tasks.manager_based.classic.ant.evaluation_mdp import AntEvaluationRecorderCfg
        if not isinstance(env_cfg, AntEnvCfg):
            raise ValueError("--diagnostics/--results_file currently support this project's Ant tasks only")
        env_cfg.rewards = RewardsCfg()
        env_cfg.recorders = AntEvaluationRecorderCfg()
        if args_cli.results_file:
            from isaaclab.utils.io import dump_yaml
            args_cli.results_file.parent.mkdir(parents=True, exist_ok=True)
            dump_yaml(str(args_cli.results_file.with_suffix(".env.yaml")), env_cfg)
            dump_yaml(str(args_cli.results_file.with_suffix(".agent.yaml")), agent_cfg)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": args_cli.video_folder or os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    episode_rewards = torch.zeros(env.num_envs, dtype=torch.float64, device=env.device)
    episode_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    finished = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    diagnostics = None
    if args_cli.diagnostics:
        from ant_episode_diagnostics import FirstEpisodeDiagnostics
        diagnostics = FirstEpisodeDiagnostics(env.unwrapped.scene["robot"].data.root_pos_w)
        snapshot = env.unwrapped.ant_evaluation_snapshot
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, rewards, dones, extras = env.step(actions)
            # Include the terminal step, then ignore auto-reset episodes for finished environments.
            active = ~finished
            episode_rewards[active] += rewards[active]
            episode_steps[active] += 1
            if diagnostics is not None:
                diagnostics.update(snapshot.position, env.unwrapped.reset_terminated,
                                   env.unwrapped.reset_time_outs, snapshot.missing_ground, snapshot.missing_scan)
            finished |= dones.bool()
        timestep += 1

        # Wait for the first episode of every environment to finish.
        if finished.all().item():
            print(f"[INFO] All {env.num_envs} environments finished their first episode.")
            break

        # Recording length must not truncate episode statistics.
        if timestep >= env.max_episode_length:
            print(f"[INFO] Reached maximum episode length: {env.max_episode_length}")
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    completed = int(finished.sum().item())
    print(f"[INFO] Completed first episodes: {completed}/{env.num_envs}")
    if completed != env.num_envs:
        print("[INFO] Statistics include partial episodes for unfinished environments.")
    if env.num_envs == 1:
        print(f"[RESULT] Episode reward total: {episode_rewards[0].item():.6f}")
        print(f"[RESULT] Episode steps: {episode_steps[0].item()}")
    else:
        # Population standard deviation across the evaluated environments.
        steps = episode_steps.to(dtype=torch.float64)
        print(
            f"[RESULT] Episode reward total: mean={episode_rewards.mean().item():.6f}, "
            f"std={episode_rewards.std(unbiased=False).item():.6f}"
        )
        print(
            f"[RESULT] Episode steps: mean={steps.mean().item():.6f}, "
            f"std={steps.std(unbiased=False).item():.6f}"
        )

    if diagnostics is not None and completed == env.num_envs:
        import hashlib
        import json
        result = diagnostics.result(episode_rewards, episode_steps, dt)
        terrain = env_cfg.scene.terrain
        result.update(schema_version=1, task=args_cli.task, checkpoint=os.path.abspath(resume_path),
                      checkpoint_sha256=hashlib.sha256(Path(resume_path).read_bytes()).hexdigest(),
                      seed=env_cfg.seed, terrain_seed=getattr(terrain.terrain_generator, "seed", None),
                      static_friction=terrain.physics_material.static_friction,
                      dynamic_friction=terrain.physics_material.dynamic_friction,
                      material_note="Mix uses per-profile friction; see the resolved env config",
                      reward_protocol="original_Ant_7_terms_v1", completed=completed,
                      step_dt=dt, max_episode_steps=env.max_episode_length,
                      observation_dimension=env.unwrapped.observation_manager.group_obs_dim["policy"][0])
        for key, label in (("target_progress_m", "Target progress (m)"),
                           ("forward_displacement_m", "Forward displacement (m)")):
            values = result["summary"][key]
            print(f"[RESULT] {label}: mean={values['mean']:.6f}, std={values['std']:.6f}")
        print(f"[RESULT] Timeout without failure rate: {result['summary']['timeout_without_failure_rate']:.6f}")
        print(f"[RESULT] Termination rate: {result['summary']['termination_rate']:.6f}")
        print(f"[RESULT] Missing ground/scan episodes: {result['summary']['missing_ground_episodes']}/"
              f"{result['summary']['missing_scan_episodes']}")
        if args_cli.results_file:
            with args_cli.results_file.open("x") as stream:
                json.dump(result, stream, indent=2, allow_nan=False)
                stream.write("\n")
            print(f"[RESULT] JSON: {args_cli.results_file}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
