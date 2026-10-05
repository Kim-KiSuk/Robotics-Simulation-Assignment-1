# Copyright (c) 2026, Ant generalization assignment contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Single-factor experiment: broaden only the T2 training wave counts."""

from copy import deepcopy

CONTROL_TASK = "Isaac-Ant-Six-TrainMix-HeightScan-Control-v0"
TREATMENT_TASK = "Isaac-Ant-Six-TrainMix-HeightScan-WaveRange-v0"


def broaden_wave_counts(original):
    """Keep half of each T2 component and use the other half for shorter waves.

    T2 remains 20% of TrainMix: counts 2, 4, 6, 8 each get 5% overall.
    Other profiles, amplitude, resolution, tile size and friction are unchanged.
    Inserting each new entry next to its parent preserves the probability
    intervals of all non-wave profiles under the same generator seed.
    """
    result = {}
    changed = 0
    for name, cfg in original.items():
        familiar = deepcopy(cfg)
        if name.startswith("T2__"):
            if cfg.num_waves not in (2, 4):
                raise ValueError(f"Unexpected source wave count: {cfg.num_waves}")
            familiar.proportion *= 0.5
            shorter = deepcopy(familiar)
            shorter.num_waves = 6 if cfg.num_waves == 2 else 8
            result[name] = familiar
            result[name + "__shorter"] = shorter
            changed += 1
        else:
            result[name] = familiar
    if changed != 2:
        raise ValueError("Expected exactly the two original T2 wave configurations")
    return result
