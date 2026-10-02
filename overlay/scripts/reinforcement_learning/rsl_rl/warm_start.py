# Copyright (c) 2026, Ant generalization assignment contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Transfer a compatible PPO policy into a fresh optimizer and iteration schedule."""

import hashlib
import math
from pathlib import Path

import torch


def load_warm_start(runner, checkpoint, min_std=None):
    """Load actor/critic/normalizers/noise; preserve the new optimizer/config.

    min_std, if provided, raises small exploration standard deviations ONCE at
    initialization. It is not a permanent bound during PPO updates, and does not
    change deterministic actor actions. The checkpoint itself is never modified.
    """
    checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
    if runner.alg.optimizer.state:
        raise ValueError("Warm start requires a fresh runner/optimizer, not an already trained one")
    if min_std is not None and (not math.isfinite(min_std) or min_std <= 0):
        raise ValueError("warm_start_min_std must be finite and positive")
    if getattr(runner.alg, "rnd", None) is not None:
        raise ValueError("Warm start currently supports PPO without RND")
    runner.load(str(checkpoint), load_optimizer=False, map_location=runner.device)
    source_iteration = runner.current_learning_iteration
    runner.current_learning_iteration = 0
    runner.tot_timesteps = 0
    runner.tot_time = 0
    policy = runner.alg.policy
    before, after = None, None
    if min_std is not None:
        with torch.no_grad():
            if hasattr(policy, "std"):
                before = policy.std.detach().cpu().tolist()
                policy.std.clamp_(min=min_std)
                after = policy.std.detach().cpu().tolist()
            elif hasattr(policy, "log_std"):
                before = policy.log_std.detach().exp().cpu().tolist()
                policy.log_std.clamp_(min=math.log(min_std))
                after = policy.log_std.detach().exp().cpu().tolist()
            else:
                raise ValueError("Policy has no supported std/log_std exploration parameter")
    if runner.alg.optimizer.state:
        raise RuntimeError("Checkpoint optimizer state unexpectedly restored")
    if any(not torch.isfinite(value).all() for value in policy.state_dict().values()):
        raise ValueError("Warm-start model contains non-finite values")
    return dict(checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                source_iteration=source_iteration, new_iteration=0, optimizer_state_restored=False,
                optimizer_learning_rates=[group["lr"] for group in runner.alg.optimizer.param_groups],
                min_std_at_initialization=min_std, std_before=before, std_after=after)
