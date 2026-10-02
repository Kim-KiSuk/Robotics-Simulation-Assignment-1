# Copyright (c) 2026, Ant generalization assignment contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Inspect actual E3 geometry, block coverage and the 100 official spawn positions on CPU."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import trimesh

from inspect_ant_six import ROOT, load_file, provider


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/ant_six_envs/assets/blocks_eval"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    folder = ROOT / "source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant"
    spec = load_file("_blocks_spec_cpu", folder / "blocks_eval_spec.py")
    surface = load_file("_blocks_continuous_cpu", folder / "continuous_eval_surface.py")
    blocks = load_file("_blocks_surface_cpu", folder / "block_surface.py")
    terrain = provider()

    def raw_function(cfg):
        if hasattr(cfg, "noise_range"):
            return terrain.HfRandomUniformTerrainCfg().function.__wrapped__
        if hasattr(cfg, "num_waves"):
            return terrain.HfWaveTerrainCfg().function.__wrapped__
        return terrain.HfPyramidSlopedTerrainCfg().function.__wrapped__

    surface._raw_height_function = raw_function
    configs = spec.make_sub_terrains(terrain, surface.continuous_surface, blocks.square_block_surface)
    keys = list(configs)
    probabilities = np.array([cfg.proportion for cfg in configs.values()])
    assert np.isclose(probabilities.sum(), 1)
    for cfg in configs.values():
        cfg.size = spec.TILE_SIZE
        cfg.horizontal_scale = spec.HORIZONTAL_SCALE
        cfg.vertical_scale = spec.VERTICAL_SCALE
        cfg.slope_threshold = None

    def build(key, difficulty):
        meshes, origin = configs[key].function(difficulty, configs[key])
        mesh = trimesh.util.concatenate(meshes)
        assert np.isfinite(mesh.vertices).all() and np.isfinite(origin).all()
        assert (mesh.area_faces > 1e-10).all()
        upward = mesh.face_normals[:, 2] > 0.1
        assert np.ptp(mesh.triangles[upward, :, 2]) > 0.01
        np.testing.assert_allclose(mesh.bounds[:, :2], [[0, 0], spec.TILE_SIZE])
        if key == "blocks":
            assert len(mesh.faces) == 1200
            assert mesh.is_watertight
            # All 100 square tops cover the entire 8 x 8 m tile once.
            assert np.isclose(mesh.area_faces[upward].sum(), 64.0)
            assert (np.abs(mesh.face_normals[:, 2]) < 0.1).any(), "Need vertical block walls"
            assert mesh.triangles[upward, :, 2].min() >= -0.12
            assert mesh.triangles[upward, :, 2].max() <= 0.12
        else:
            assert len(mesh.faces) == 2048
            assert (mesh.face_normals[:, 2] > 0).all()
            horizontal = np.ptp(mesh.triangles[:, :, 2], axis=1) < 1e-10
            assert horizontal.mean() < 0.05
        return mesh

    cases, previews = [], {}
    for key in keys:
        for difficulty in (0.0, 0.5, 1.0):
            np.random.seed(123)
            mesh = build(key, difficulty)
            np.random.seed(123)
            repeat = build(key, difficulty)
            np.testing.assert_array_equal(mesh.vertices, repeat.vertices)
            top = mesh.triangles[mesh.face_normals[:, 2] > 0.1, :, 2]
            cases.append(dict(kind=key, difficulty=difficulty, triangles=len(mesh.faces),
                              surface_min_m=float(top.min()), surface_max_m=float(top.max())))
            if difficulty == 0.5:
                previews[key] = mesh

    np.random.seed(spec.TERRAIN_SEED)
    rng = np.random.default_rng(spec.TERRAIN_SEED)
    selected, total_faces = [], 0
    for _ in range(spec.NUM_ROWS * spec.NUM_COLS):
        key = keys[rng.choice(len(keys), p=probabilities)]
        mesh = build(key, rng.uniform(0, 1))
        total_faces += len(mesh.faces)
        selected.append(key)
    ii, jj = np.meshgrid(np.arange(10), np.arange(10), indexing="ij")
    x, y = -(ii.ravel() - 4.5) * 5, (jj.ravel() - 4.5) * 5
    row, col = np.floor((x + 160) / 8).astype(int), np.floor((y + 160) / 8).astype(int)
    initial = dict(Counter(selected[i * spec.NUM_COLS + j] for i, j in zip(row, col)))
    assert set(initial) == set(keys), initial

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, Normalize
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(14, 9), layout="constrained")
    for i, (key, mesh) in enumerate(previews.items(), 1):
        ax = fig.add_subplot(2, 3, i, projection="3d")
        triangles = mesh.triangles[mesh.face_normals[:, 2] > -0.1].copy()
        top = mesh.triangles[mesh.face_normals[:, 2] > 0.1, :, 2]
        zmin, zmax = top.min(), top.max()
        # Hide the deep support base in the preview; collision geometry is intact.
        triangles[:, :, 2] = np.maximum(triangles[:, :, 2], zmin - 0.025)
        colors = plt.cm.terrain(Normalize(zmin, zmax)(triangles[:, :, 2].mean(axis=1)))
        ax.add_collection3d(Poly3DCollection(triangles, facecolors=colors,
                                            edgecolors=(0, 0, 0, 0.15), linewidths=0.15))
        ax.set(xlim=(0, 8), ylim=(0, 8), zlim=(zmin - 0.03, zmax + 0.03),
               xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", title=key)
        ax.set_xticks([0, 2, 4, 6, 8])
        ax.set_yticks([0, 2, 4, 6, 8])
        ax.set_zticks([round(zmin, 2), round(zmax, 2)])
        ax.tick_params(labelsize=8)
        ax.set_box_aspect((8, 8, 3 * (zmax - zmin + 0.06)))
        ax.view_init(elev=32, azim=-60)
    fig.suptitle("E3: actual CPU tile geometry, not a simulation image\n"
                 "Difficulty 0.5; heights exaggerated 3x; per-panel colors; block support bases hidden")
    fig.savefig(args.output_dir / "preview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")
    labels = np.array([keys.index(k) for k in selected]).reshape(40, 40)
    im = ax.imshow(labels, origin="lower", extent=(-160, 160, -160, 160), interpolation="nearest",
                   cmap=ListedColormap(["#968c3e", "#3e92cc", "#70c5b3", "#c47335", "#9865b5", "#e06c75"]),
                   vmin=-0.5, vmax=5.5)
    ax.scatter(y, x, c="black", s=15)
    ax.set(xlim=(-40, 40), ylim=(-40, 40), xlabel="world y (m)", ylabel="world x (m)",
           title=f"E3 seed {spec.TERRAIN_SEED}: actual central tiles and 100 spawn positions")
    fig.colorbar(im, ax=ax, ticks=np.arange(6)).ax.set_yticklabels(keys)
    fig.savefig(args.output_dir / "spawn_map.png", dpi=150)
    plt.close(fig)
    result = dict(scope="CPU geometry only, no policy evaluation", terrain_seed=spec.TERRAIN_SEED,
                  triangles=total_faces, flat_tiles=0, tiles=len(selected),
                  actual_tile_counts=dict(Counter(selected)), initial_100_counts=initial,
                  proportions=dict(zip(keys, probabilities.tolist())), tile_checks=cases,
                  note="Block tops are horizontal; there are no flat-only tiles, central platforms or flat borders.")
    (args.output_dir / "geometry_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"[CHECK] PASS: {len(cases)} geometry cases and {len(selected)} actual tiles; {total_faces:,} triangles")
    print(f"[CHECK] Initial 100 coverage: {initial}")


if __name__ == "__main__":
    main()
