# Copyright (c) 2026, Ant generalization assignment contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""CPU audit and preview of the actual E2 height-field generators; no policy scoring."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh

from inspect_ant_six import ROOT, load_file, provider


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/ant_six_envs/assets/continuous_eval"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    spec_path = ROOT / "source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/continuous_eval_spec.py"
    spec = load_file("_continuous_eval_spec_cpu", spec_path)
    surface_path = spec_path.with_name("continuous_eval_surface.py")
    surface = load_file("_continuous_eval_surface_cpu", surface_path)
    terrain = provider()
    # Same source functions loaded without launching the simulator; only replace
    # the lazy module import, leaving all geometry calculations unchanged.
    def raw_height_function(cfg):
        if hasattr(cfg, "noise_range"):
            return terrain.HfRandomUniformTerrainCfg().function.__wrapped__
        if hasattr(cfg, "num_waves"):
            return terrain.HfWaveTerrainCfg().function.__wrapped__
        return terrain.HfPyramidSlopedTerrainCfg().function.__wrapped__
    surface._raw_height_function = raw_height_function
    configs = spec.make_sub_terrains(terrain, surface.continuous_surface)
    keys = list(configs)
    probabilities = np.array([c.proportion for c in configs.values()])
    assert abs(probabilities.sum() - 1.0) < 1e-12
    for cfg in configs.values():
        cfg.size = spec.TILE_SIZE
        cfg.horizontal_scale = spec.HORIZONTAL_SCALE
        cfg.vertical_scale = spec.VERTICAL_SCALE
        cfg.slope_threshold = None
        assert cfg.border_width == 0.0 and getattr(cfg, "platform_width", 0.0) == 0.0

    def build(key, difficulty):
        meshes, origin = configs[key].function(difficulty, configs[key])
        mesh = trimesh.util.concatenate(meshes)
        assert np.isfinite(mesh.vertices).all() and np.isfinite(origin).all()
        assert (mesh.area_faces > 0).all()
        assert np.ptp(mesh.vertices[:, 2]) > 0.05, f"Flat/near-flat tile: {key}"
        assert len(mesh.faces) == 2048
        assert (mesh.face_normals[:, 2] > 0).all(), "Mesh winding must face upward"
        border = np.any(np.isclose(mesh.vertices[:, :2], 0) | np.isclose(mesh.vertices[:, :2], 8), axis=1)
        assert np.allclose(mesh.vertices[border, 2], 0), "Neighboring tile edges must meet"
        assert np.mean(np.ptp(mesh.vertices[mesh.faces, 2], axis=1) < 1e-10) < 0.05
        assert configs[key].size == spec.TILE_SIZE, "Generator did not restore config size"
        return mesh

    # Test all endpoints as well as a middle difficulty. Record actual geometry,
    # including the small horizontal faces caused by sampling/quantization.
    cases = []
    previews = {}
    for key in keys:
        for difficulty in (0.0, 0.5, 1.0):
            np.random.seed(spec.TERRAIN_SEED)
            mesh = build(key, difficulty)
            np.random.seed(spec.TERRAIN_SEED)
            repeated = build(key, difficulty)
            np.testing.assert_array_equal(mesh.vertices, repeated.vertices)
            dz = np.ptp(mesh.vertices[mesh.faces, 2], axis=1)
            cases.append(dict(kind=key, difficulty=difficulty, triangles=len(mesh.faces),
                              z_min=float(mesh.vertices[:, 2].min()), z_max=float(mesh.vertices[:, 2].max()),
                              horizontal_face_fraction=float(np.mean(dz < 1e-10))))
            if difficulty == 0.5:
                previews[key] = mesh

    # Reproduce the stock generator's independent type/difficulty RNG and legacy
    # NumPy geometry RNG. Check every tile, not only the five preview samples.
    np.random.seed(spec.TERRAIN_SEED)
    rng = np.random.default_rng(spec.TERRAIN_SEED)
    selected = []
    min_relief = float("inf")
    total_faces = 0
    for _ in range(spec.NUM_ROWS * spec.NUM_COLS):
        key = keys[rng.choice(len(keys), p=probabilities)]
        difficulty = rng.uniform(0.0, 1.0)
        mesh = build(key, difficulty)
        selected.append(key)
        min_relief = min(min_relief, float(np.ptp(mesh.vertices[:, 2])))
        total_faces += len(mesh.faces)
    assert total_faces == 3_276_800

    # The official 100-environment grid spans -22.5..22.5m on both axes.
    ii, jj = np.meshgrid(np.arange(10), np.arange(10), indexing="ij")
    x, y = -(ii.ravel() - 4.5) * 5.0, (jj.ravel() - 4.5) * 5.0
    extent_x = spec.NUM_ROWS * spec.TILE_SIZE[0]
    extent_y = spec.NUM_COLS * spec.TILE_SIZE[1]
    row = np.floor((x + extent_x / 2) / spec.TILE_SIZE[0]).astype(int)
    col = np.floor((y + extent_y / 2) / spec.TILE_SIZE[1]).astype(int)
    initial = dict(Counter(selected[i * spec.NUM_COLS + j] for i, j in zip(row, col)))
    assert set(initial) == set(keys), initial

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, Normalize
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(14, 9), layout="constrained")
    for index, (key, mesh) in enumerate(previews.items(), 1):
        ax = fig.add_subplot(2, 3, index, projection="3d")
        triangles = mesh.triangles
        z = triangles[:, :, 2].mean(axis=1)
        colors = plt.cm.terrain(Normalize(z.min(), z.max())(z))
        ax.add_collection3d(Poly3DCollection(triangles, facecolors=colors, edgecolors=(0, 0, 0, 0.1), linewidths=0.1))
        zmin, zmax = mesh.bounds[:, 2]
        ax.set(xlim=(0, 8), ylim=(0, 8), zlim=(zmin - 0.02, zmax + 0.02),
               xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", title=key)
        ax.set_zticks([round(zmin, 2), round(zmax, 2)])
        ax.set_box_aspect((8, 8, 3 * (zmax - zmin + 0.04)))
        ax.view_init(elev=32, azim=-60)
    ax = fig.add_subplot(2, 3, 6)
    labels = np.array([keys.index(k) for k in selected]).reshape(spec.NUM_ROWS, spec.NUM_COLS)
    colors = ["#968c3e", "#3e92cc", "#70c5b3", "#c47335", "#9865b5"]
    im = ax.imshow(labels, origin="lower", extent=(-extent_y / 2, extent_y / 2, -extent_x / 2, extent_x / 2),
                   cmap=ListedColormap(colors), vmin=-0.5, vmax=len(keys)-0.5, interpolation="nearest")
    ax.scatter(y, x, s=9, c="black")
    ax.set(xlim=(-40, 40), ylim=(-40, 40), xlabel="world y (m)", ylabel="world x (m)",
           title="Actual E2 center: 100 initial positions")
    cb = fig.colorbar(im, ax=ax, ticks=np.arange(len(keys)), shrink=0.75)
    cb.ax.set_yticklabels(keys)
    fig.suptitle("E2 continuous terrain: actual CPU geometry, not a simulation image\n"
                 "3D tiles: difficulty 0.5, vertical scale exaggerated 3x, per-panel colors; seed 9217")
    fig.savefig(args.output_dir / "preview.png", dpi=150)
    plt.close(fig)
    result = dict(scope="CPU actual tile geometry and spawn coverage; no policy evaluation", task=spec.TASK,
                  terrain_seed=spec.TERRAIN_SEED, spec_sha256=hashlib.sha256(spec_path.read_bytes()).hexdigest(),
                  surface_sha256=hashlib.sha256(surface_path.read_bytes()).hexdigest(),
                  configured_proportions=dict(zip(keys, probabilities.tolist())),
                  tiles=len(selected), actual_tile_counts=dict(Counter(selected)),
                  triangles=total_faces, flat_tiles=0, minimum_tile_relief_m=min_relief,
                  map_size_m=[extent_x, extent_y], initial_100_counts=initial,
                  tile_checks=cases,
                  flat_surface_note="No flat tile/platform; local horizontal triangle faces still exist due to discretization.")
    (args.output_dir / "geometry_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"[CHECK] PASS: {len(cases)} endpoint/midpoint cases; all {len(selected)} tiles non-flat")
    print(f"[CHECK] PASS: initial 100 robots cover all five generators: {initial}")
    print(f"[OUTPUT] {args.output_dir}")


if __name__ == "__main__":
    main()
