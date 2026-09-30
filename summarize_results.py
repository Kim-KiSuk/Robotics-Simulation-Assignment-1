#!/usr/bin/env python3
"""Reproduce per-run and pooled population statistics from complete legacy logs."""

import csv
import math
from pathlib import Path
import re


def main():
    root = Path(__file__).resolve().parent
    runs = []
    for path in sorted((root / "artifacts" / "evaluation_logs").rglob("*.log")):
        text = path.read_text()
        completion = re.search(r"Completed first episodes: (\d+)/(\d+)", text)
        if not completion or completion[1] != completion[2]:
            raise ValueError(f"Incomplete log: {path}")
        match = re.fullmatch(r"([BC])_(default|low|high)_terrain(\d+)_seed(\d+)", path.stem)
        if match:
            model, friction, terrain, seed = match.groups()
            domain = "rough"
        else:
            match = re.fullmatch(r"(default|low_friction|high_friction)_seed(\d+)", path.stem)
            if not match:
                raise ValueError(f"Unknown log: {path}")
            model, domain, terrain = "A", "flat", ""
            friction, seed = match.groups()
            friction = friction.removesuffix("_friction")
        row = dict(model=model, domain=domain, friction=friction, terrain_seed=terrain, reset_seed=int(seed), n=int(completion[1]))
        for key, label in (("reward", "Episode reward total"), ("steps", "Episode steps")):
            stat = re.search(rf"{label}: mean=([-\d.]+), std=([-\d.]+)", text)
            if stat is None:
                raise ValueError(f"Missing {key}: {path}")
            row[key + "_mean"], row[key + "_std"] = map(float, stat.groups())
        row["source"] = str(path.relative_to(root))
        runs.append(row)
    pooled = []
    for key in sorted({(r["model"], r["domain"], r["friction"]) for r in runs}):
        group = [r for r in runs if (r["model"], r["domain"], r["friction"]) == key]
        n = sum(r["n"] for r in group)
        row = dict(zip(("model", "domain", "friction"), key), n=n, maps=len(group))
        for metric in ("reward", "steps"):
            mean = sum(r["n"] * r[metric + "_mean"] for r in group) / n
            variance = sum(r["n"] * (r[metric + "_std"] ** 2 + (r[metric + "_mean"] - mean) ** 2) for r in group) / n
            row[metric + "_mean"] = mean
            row[metric + "_std"] = math.sqrt(variance)
        pooled.append(row)
    output = root / "results"
    output.mkdir(exist_ok=True)
    for name, rows in (("runs", runs), ("pooled", pooled)):
        with (output / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Summarized {len(runs)} runs into {len(pooled)} groups; std uses population variance, not standard error")


if __name__ == "__main__":
    main()
