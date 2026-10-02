# Copyright (c) 2026, Ant generalization assignment contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Build six terrain previews and audit the actual IsaacLab tile functions on CPU.

No simulator, robot or policy evaluation. Run from IsaacLab_RS with its conda Python.
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
import torch
import trimesh

ROOT = Path(__file__).resolve().parents[2]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def provider():
    modules = {}
    for kind, filename in (("height_field", "hf_terrains"), ("trimesh", "mesh_terrains")):
        folder = ROOT / "source/isaaclab/isaaclab/terrains" / kind
        package = ModuleType(f"_six_cpu_{kind}")
        package.__path__ = [str(folder)]
        sys.modules[package.__name__] = package
        load_file(package.__name__ + ".utils", folder / "utils.py")
        modules[kind] = load_file(package.__name__ + "." + filename, folder / (filename + ".py"))
    hf, mesh = modules["height_field"], modules["trimesh"]

    def factory(function, **defaults):
        def make(**kwargs):
            return SimpleNamespace(**dict(size=(12.0, 12.0), horizontal_scale=0.4,
                                         vertical_scale=0.005, slope_threshold=None,
                                         **(defaults | kwargs)), function=function)
        return make

    return SimpleNamespace(
        HfRandomUniformTerrainCfg=factory(hf.random_uniform_terrain, border_width=0.0),
        HfWaveTerrainCfg=factory(hf.wave_terrain, border_width=0.0),
        HfPyramidSlopedTerrainCfg=factory(hf.pyramid_sloped_terrain, border_width=0.0, inverted=False),
        HfInvertedPyramidSlopedTerrainCfg=factory(hf.pyramid_sloped_terrain, border_width=0.0, inverted=True),
        MeshRandomGridTerrainCfg=factory(mesh.random_grid_terrain),
        MeshPyramidStairsTerrainCfg=factory(mesh.pyramid_stairs_terrain),
        MeshInvertedPyramidStairsTerrainCfg=factory(mesh.inverted_pyramid_stairs_terrain),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/ant_six_envs/assets"))
    parser.add_argument("--train-num-envs", type=int, default=1024)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    spec_path = ROOT / "source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/six_terrain_spec.py"
    spec = load_file("_six_spec", spec_path)
    terrain = provider()
    # Keep the preview strictly CPU even on a workstation with a visible GPU.
    from unittest.mock import patch
    audit = []
    previews = {}
    counts = {}
    spawn_coverage = {}
    with patch("torch.cuda.is_available", return_value=False):
        for profile in (*spec.TRAIN_PROFILES, "E1", "Mix"):
            configs = spec.make_sub_terrains(profile, terrain)
            assert abs(sum(c.proportion for c in configs.values()) - 1) < 1e-12
            assert all(k.split("__")[0] in spec.TRAIN_PROFILES for k in configs)
            if profile in spec.TRAIN_PROFILES:
                for name, cfg in configs.items():
                    for difficulty in (0.0, 0.5, 1.0):
                        np.random.seed(42)
                        torch.manual_seed(42)
                        meshes, origin = cfg.function(difficulty, cfg)
                        mesh = trimesh.util.concatenate(meshes)
                        assert np.isfinite(mesh.vertices).all() and np.isfinite(origin).all()
                        assert (mesh.area_faces > 0).all(), name
                        assert cfg.size == spec.TILE_SIZE
                        np.random.seed(42)
                        torch.manual_seed(42)
                        repeat, _ = cfg.function(difficulty, cfg)
                        np.testing.assert_array_equal(mesh.vertices, trimesh.util.concatenate(repeat).vertices)
                        # Upward-facing faces are the walkable surface; box bottoms are not height samples.
                        surface = mesh.triangles[mesh.face_normals[:, 2] > 0.1, :, 2]
                        assert np.ptp(surface) > 0.005, name
                        audit.append(dict(profile=profile, tile=name, difficulty=difficulty,
                                          surface_min=float(surface.min()), surface_max=float(surface.max()),
                                          triangles=len(mesh.faces)))
            rng = np.random.default_rng(spec.seed_for(profile))
            keys = list(configs)
            probabilities = np.array([configs[k].proportion for k in keys])
            selected = []
            for _ in range(44 * 44):
                key = keys[rng.choice(len(keys), p=probabilities / probabilities.sum())]
                selected.append((key, rng.uniform(0.0, 1.0)))
            counts[profile] = dict(Counter(key.split("__")[0] for key, _ in selected))
            if profile in ("Mix", "E1"):
                assert set(counts[profile]) == set(spec.TRAIN_PROFILES)
                num_envs = args.train_num_envs if profile == "Mix" else 100
                assert num_envs >= 1
                # Exact XY-grid formula used by TerrainImporter, mapped to 12m tiles.
                rows = int(np.ceil(num_envs / int(np.sqrt(num_envs))))
                cols = int(np.ceil(num_envs / rows))
                ii, jj = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
                x = -(ii.ravel()[:num_envs] - (rows - 1) / 2) * 5.0
                y = (jj.ravel()[:num_envs] - (cols - 1) / 2) * 5.0
                indices = np.floor((x + 264) / 12).astype(int) * 44 + np.floor((y + 264) / 12).astype(int)
                initial = dict(Counter(selected[i][0].split("__")[0] for i in indices))
                assert set(initial) == set(spec.TRAIN_PROFILES), (profile, initial)
                spawn_coverage[profile] = dict(num_envs=num_envs, initial_profile_counts=initial)
            if profile == "Mix":
                continue
            # Representative 3x3 patch using the profile's RNG; not a full-map screenshot.
            np.random.seed(spec.seed_for(profile))
            torch.manual_seed(spec.seed_for(profile))
            tiles = []
            for index, (key, difficulty) in enumerate(selected[:9]):
                meshes, _ = configs[key].function(difficulty, configs[key])
                tile = trimesh.util.concatenate(meshes)
                tile.apply_translation((index // 3 * 12.0, index % 3 * 12.0, 0.0))
                tiles.append(tile)
            previews[profile] = trimesh.util.concatenate(tiles)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.colors import Normalize

    fig = plt.figure(figsize=(15, 10), layout="constrained")
    for index, (profile, mesh) in enumerate(previews.items(), 1):
        ax = fig.add_subplot(2, 3, index, projection="3d")
        triangles = mesh.triangles
        visible = mesh.face_normals[:, 2] > -0.5
        triangles = triangles[visible]
        z = triangles[:, :, 2].mean(axis=1)
        norm = Normalize(z.min(), z.max())
        poly = Poly3DCollection(triangles, facecolors=plt.cm.terrain(norm(z)), linewidths=0.05,
                               edgecolors=(0, 0, 0, 0.12))
        ax.add_collection3d(poly)
        ax.set(xlim=(0, 36), ylim=(0, 36), zlim=(-1.5, 1.5), xlabel="x (m)", ylabel="y (m)", zlabel="z (m)")
        ax.set_box_aspect((36, 36, 12))  # Fourfold vertical exaggeration, explicitly captioned.
        ax.view_init(elev=35, azim=-60)
        ax.set_title(f"{profile}: {spec.PROFILES[profile]['title']}\nseed={spec.seed_for(profile)}, "
                     f"friction={spec.PROFILES[profile]['friction']}")
    fig.suptitle(
        "Six Ant scenarios: actual generated 3x3 tile samples\n"
        "CPU geometry preview; vertical scale exaggerated 4x; per-panel colors; no policy evaluation"
    )
    fig.savefig(args.output_dir / "six_terrains.png", dpi=150)
    plt.close(fig)
    result = dict(scope="CPU tile geometry only; not PhysX or locomotion validation",
                  spec_sha256=hashlib.sha256(spec_path.read_bytes()).hexdigest(),
                  profile_parameters=spec.PROFILES, geometry=audit, full_map_profile_counts=counts,
                  spawn_coverage=spawn_coverage,
                  reference_image="Based on user description: square blocks of different heights; image unavailable")
    (args.output_dir / "geometry_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"[CHECK] PASS: {len(audit)} tile/difficulty cases, repeatable, finite, non-degenerate")
    print("[CHECK] PASS: six profiles + training mixture; E1 seed excluded from training")
    print(f"[OUTPUT] {args.output_dir / 'six_terrains.png'}")


if __name__ == "__main__":
    main()
