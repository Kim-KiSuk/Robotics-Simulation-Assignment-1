# Copyright (c) 2026, Ant generalization assignment contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Audit actual v2 height-field meshes on CPU without starting Isaac Sim or CUDA.

Uses the installed checkout's pure NumPy/SciPy/trimesh terrain functions and the
same parameter factory as AntVariedEnvCfg. Does not validate PhysX/task loading.
"""

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def terrain_provider():
    # Load the actual pure terrain modules under a private package name so their
    # relative imports work without importing IsaacLab's simulator-dependent __init__.
    folder = ROOT / "source/isaaclab/isaaclab/terrains/height_field"
    package = ModuleType("_ant_cpu_height_field")
    package.__path__ = [str(folder)]
    sys.modules[package.__name__] = package
    load_file(package.__name__ + ".utils", folder / "utils.py")
    hf = load_file(package.__name__ + ".hf_terrains", folder / "hf_terrains.py")

    def config(function, inverted=False):
        def make(**kwargs):
            return SimpleNamespace(
                function=function, size=(12.0, 12.0), horizontal_scale=0.4,
                vertical_scale=0.005, border_width=0.0, slope_threshold=None,
                inverted=inverted, **kwargs,
            )
        return make

    return SimpleNamespace(
        HfRandomUniformTerrainCfg=config(hf.random_uniform_terrain),
        HfWaveTerrainCfg=config(hf.wave_terrain),
        HfPyramidSlopedTerrainCfg=config(hf.pyramid_sloped_terrain),
        HfInvertedPyramidSlopedTerrainCfg=config(hf.pyramid_sloped_terrain, inverted=True),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("logs/ant_varied_inspection"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = ROOT / "source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/varied_terrain_spec.py"
    profile = load_file("_ant_varied_spec", profile_path)
    configs = profile.make_varied_sub_terrains(terrain_provider())
    assert "flat" not in configs
    assert abs(sum(c.proportion for c in configs.values()) - 1.0) < 1e-12
    records, plotted = [], {}
    for name, cfg in configs.items():
        for difficulty in (0.0, 0.5, 1.0):
            np.random.seed(42)
            meshes, origin = cfg.function(difficulty, cfg)
            mesh = meshes[0]
            assert np.isfinite(mesh.vertices).all() and np.isfinite(origin).all()
            assert np.ptp(mesh.vertices[:, 2]) > 0.01, (name, difficulty)
            assert len(mesh.faces) == 1800
            assert (mesh.area_faces > 0).all(), "Degenerate collision triangles"
            assert cfg.size == (12.0, 12.0), "Height-field wrapper failed to restore tile size"
            z = mesh.vertices[:, 2]
            record = dict(terrain=name, difficulty=difficulty, z_min_m=float(z.min()), z_max_m=float(z.max()),
                          peak_to_peak_m=float(np.ptp(z)), triangles=len(mesh.faces))
            records.append(record)
            if difficulty == 0.5:
                plotted[name] = mesh
            np.random.seed(42)
            repeat, _ = cfg.function(difficulty, cfg)
            np.testing.assert_array_equal(mesh.vertices, repeat[0].vertices)
    # Reproduce the exact tile-type/difficulty RNG sequence of AntTerrainGenerator.
    names = list(configs)
    probabilities = np.array([configs[name].proportion for name in names])
    probabilities /= probabilities.sum()
    tile_counts = {}
    for seed in (42, 4001, 4002):
        rng = np.random.default_rng(seed)
        counts = Counter()
        for _ in range(44 * 44):
            counts[names[rng.choice(len(names), p=probabilities)]] += 1
            rng.uniform(0.0, 1.0)
        assert sum(counts.values()) == 1936 and "flat" not in counts
        tile_counts[str(seed)] = dict(counts)
    result = dict(
        validation_scope="CPU mesh geometry only; not a policy or PhysX evaluation",
        terrain_spec_sha256=hashlib.sha256(profile_path.read_bytes()).hexdigest(),
        geometry=records, tile_counts=tile_counts, full_map_triangles=1936 * 1800,
    )
    (args.output_dir / "mesh_audit.json").write_text(json.dumps(result, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
    for ax, (name, mesh) in zip(axes.flat, plotted.items()):
        norm = Normalize(-1.5, 1.5)
        artist = ax.tripcolor(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.faces,
                              mesh.vertices[:, 2], cmap="terrain", norm=norm, shading="gouraud")
        ax.set_title(name + f"\nrelief {np.ptp(mesh.vertices[:, 2]):.3f} m")
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
    axes.flat[-1].axis("off")
    fig.colorbar(artist, ax=list(axes.flat[:-1]), label="Surface height (m), shared scale")
    fig.suptitle("Ant terrain v2: real generated tile meshes, difficulty 0.5 (CPU inspection)")
    fig.savefig(args.output_dir / "terrain_tiles.png", dpi=150)
    plt.close(fig)
    print("[CHECK] PASS: 21 meshes, non-flat relief, finite/non-degenerate geometry, reproducible heights")
    print("[CHECK] PASS: flat tile probability 0; 3 map seeds; full map has 3,484,800 triangles")
    for record in records:
        if record["difficulty"] == 0.5:
            print(f"[TERRAIN] {record['terrain']}: height {record['z_min_m']:.3f}..{record['z_max_m']:.3f} m")
    print(f"[OUTPUT] {args.output_dir / 'terrain_tiles.png'}")


if __name__ == "__main__":
    main()
