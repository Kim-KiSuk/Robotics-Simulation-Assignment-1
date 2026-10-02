# Copyright (c) 2026, Ant generalization assignment contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""CPU checks: expanded terrain, reward units, and actual RSL-RL warm-start semantics."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import trimesh
import yaml
from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner

from inspect_ant_six import ROOT, load_file, provider


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("/tmp/ant_stability_cpu.json"))
    args = parser.parse_args()
    folder = ROOT / "source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant"
    old = load_file("_old_stability_spec", folder / "six_terrain_spec.py")
    new = load_file("_new_stability_spec", folder / "stability_terrain_spec.py")
    rewards = load_file("_stability_rewards", folder / "robust_mdp.py")
    warm = load_file("_stability_warm_start", ROOT / "scripts/reinforcement_learning/rsl_rl/warm_start.py")
    original = old.make_sub_terrains("Mix", provider())
    before = deepcopy(original)
    expanded = new.extend_training_terrains(original)
    assert vars(original[next(iter(original))]) == vars(before[next(iter(before))])
    assert len(expanded) == 24
    assert np.isclose(sum(c.proportion for c in expanded.values()), 1.0)
    assert np.isclose(sum(c.proportion for k, c in expanded.items() if k.endswith("__expanded")), 0.3)
    assert all(cfg == original[name] for name, cfg in before.items())
    cases = []
    with patch("torch.cuda.is_available", return_value=False):
        for name, cfg in expanded.items():
            for difficulty in (0., 0.5, 1.):
                np.random.seed(101)
                torch.manual_seed(101)
                meshes, origin = cfg.function(difficulty, cfg)
                mesh = trimesh.util.concatenate(meshes)
                assert np.isfinite(mesh.vertices).all() and np.isfinite(origin).all()
                assert (mesh.area_faces > 1e-12).all(), name
                top = mesh.triangles[mesh.face_normals[:, 2] > 0.1, :, 2]
                assert np.ptp(top) > 0.005
                cases.append(dict(name=name, difficulty=difficulty, triangles=len(mesh.faces)))
    # Match generator draws and the actual 1024 robot XY grid without simulating.
    rng = np.random.default_rng(new.TERRAIN_SEED)
    keys = list(expanded)
    weights = [cfg.proportion for cfg in expanded.values()]
    tiles = []
    for _ in range(44 * 44):
        tiles.append(keys[rng.choice(len(keys), p=weights)])
        rng.uniform(0, 1)
    ii, jj = np.meshgrid(np.arange(32), np.arange(32), indexing="ij")
    x, y = -(ii.ravel()-15.5)*5, (jj.ravel()-15.5)*5
    row, col = np.floor((x+264)/12).astype(int), np.floor((y+264)/12).astype(int)
    starts = [tiles[r*44+c] for r,c in zip(row,col)]
    assert set(k.split("__")[0] for k in starts) == set(old.TRAIN_PROFILES)
    assert any(k.endswith("__expanded") for k in starts)
    # RewardManager integrates rates over dt: failure must remain -5 per event.
    for dt in (1/60, 1/120):
        env = SimpleNamespace(step_dt=dt, termination_manager=SimpleNamespace(terminated=torch.tensor([False, True])))
        torch.testing.assert_close(-5 * rewards.failure_impulse(env) * dt, torch.tensor([0., -5.]))
    height = torch.tensor([0.5, 0.4, 0.355, 0.31, 0.2])
    xyz = torch.zeros(5, 3);xyz[:,2] = height
    env = SimpleNamespace(scene={"robot":SimpleNamespace(data=SimpleNamespace(root_pos_w=xyz)),
                                "ground_height":SimpleNamespace(data=SimpleNamespace(ray_hits_w=torch.zeros(5,1,3)))})
    torch.testing.assert_close(rewards.low_clearance_risk(env), torch.tensor([0.,0.,0.25,1.,1.]))

    # Use real installed RSL-RL and the user's trained checkpoint, not a fake loader.
    cfg = yaml.safe_load((args.checkpoint.parent / "params/agent.yaml").read_text())
    cfg["algorithm"].update(learning_rate=1e-5, schedule="fixed", entropy_coef=0.001)
    obs = TensorDict({"policy":torch.linspace(-0.5,0.5,4*123).reshape(4,123)}, batch_size=[4])
    env = SimpleNamespace(num_envs=4, num_actions=8, device="cpu", get_observations=lambda:obs)
    runner = OnPolicyRunner(env, deepcopy(cfg), device="cpu")
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    assert source["optimizer_state_dict"]["state"], "Source should contain optimizer momentum"
    report = warm.load_warm_start(runner, args.checkpoint, 0.02)
    for name, value in runner.alg.policy.state_dict().items():
        if name != "std":
            torch.testing.assert_close(value, source["model_state_dict"][name], rtol=0, atol=0)
    action = runner.alg.policy.act_inference(obs).detach().clone()
    runner.alg.policy.load_state_dict(source["model_state_dict"])
    torch.testing.assert_close(action, runner.alg.policy.act_inference(obs), rtol=0, atol=0)
    warm.load_warm_start(runner, args.checkpoint, 0.02)
    assert torch.all(runner.alg.policy.std >= 0.02)
    assert not runner.alg.optimizer.state and runner.current_learning_iteration == 0
    assert all(group["lr"] == 1e-5 for group in runner.alg.optimizer.param_groups)
    result = dict(passed=True, terrain_cases=len(cases), checkpoint_transfer=report,
                  original_config_unchanged=True, deterministic_actor_unchanged=True,
                  initial_expanded_positions=sum(k.endswith("__expanded") for k in starts),
                  initial_positions=1024, failure_penalty_dt_invariant=True,
                  clearance_risk_boundary_checks=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(f"[CHECK] PASS: {result}")


if __name__ == "__main__":
    main()
