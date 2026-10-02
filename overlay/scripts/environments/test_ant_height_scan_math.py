# Copyright (c) 2026, Ant generalization assignment contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""CPU regression checks for height semantics and terminal-step diagnostics."""

from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from inspect_ant_six import ROOT, load_file

scan = load_file("_height_math_test", ROOT / "source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/height_scan_mdp.py")
metrics = load_file("_metrics_math_test", ROOT / "scripts/reinforcement_learning/rsl_rl/ant_episode_diagnostics.py")


class HeightScanTest(unittest.TestCase):
    def query(self, heights, ground):
        def sensor(z):
            hits = torch.zeros(*z.shape, 3)
            hits[:, :, 2] = z
            return SimpleNamespace(data=SimpleNamespace(ray_hits_w=hits))
        env = SimpleNamespace(scene={"height_scanner": sensor(heights), "ground_height": sensor(ground)})
        return scan.relative_ground_scan(env)

    def test_relative_height_sign_and_world_translation(self):
        heights = torch.tensor([[0.1, 0.4, -0.1]])
        ground = torch.tensor([[0.1]])
        result = self.query(heights, ground)
        torch.testing.assert_close(result, torch.tensor([[0.0, 0.3, -0.2]]))
        torch.testing.assert_close(result, self.query(heights + 10, ground + 10))

    def test_clipping_and_unknown_height(self):
        values = self.query(torch.tensor([[2.0, -2.0, float("inf"), float("nan")]]), torch.zeros(1, 1))
        torch.testing.assert_close(values, torch.tensor([[1.0, -1.0, -1.0, -1.0]]))


class FirstEpisodeTest(unittest.TestCase):
    def test_terminal_endpoint_is_frozen_and_timeout_failure_precedence(self):
        m = metrics.FirstEpisodeDiagnostics(torch.zeros(3, 3))
        no = torch.zeros(3, dtype=torch.bool)
        m.update(torch.tensor([[2., 0, 0], [1., 0, 0], [1., 0, 0]]),
                 torch.tensor([True, False, False]), no, no, no)
        # The already-finished robot has auto-reset and moved again. Ignore it.
        m.update(torch.tensor([[-100., 0, 0], [3., 0, 0], [4., 0, 0]]),
                 torch.tensor([False, False, True]), torch.tensor([False, True, True]), no, no)
        result = m.result(torch.tensor([10., 20., 30.]), torch.tensor([1, 2, 2]), 1.0)
        self.assertEqual([e["forward_displacement_m"] for e in result["episodes"]], [2., 3., 4.])
        self.assertEqual([e["timed_out"] for e in result["episodes"]], [False, True, False])
        self.assertAlmostEqual(result["summary"]["timeout_without_failure_rate"], 1 / 3)
        self.assertAlmostEqual(result["summary"]["termination_rate"], 2 / 3)
        self.assertAlmostEqual(result["summary"]["reward"]["std"], (200 / 3)**0.5)

    def test_incomplete_evaluation_is_not_a_final_result(self):
        m = metrics.FirstEpisodeDiagnostics(torch.zeros(1, 3))
        with self.assertRaisesRegex(RuntimeError, "Incomplete"):
            m.result(torch.zeros(1), torch.ones(1), 1.0)


if __name__ == "__main__":
    unittest.main()
