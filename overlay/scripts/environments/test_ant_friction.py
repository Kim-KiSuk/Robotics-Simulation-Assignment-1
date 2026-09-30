# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""CPU checks for ground-friction sampling; does not launch Isaac Sim."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/friction_utils.py"
spec = importlib.util.spec_from_file_location("ant_friction_utils", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
sample = module.sample_ground_friction


class FrictionTests(unittest.TestCase):
    def test_reproducibility_and_rng_isolation(self):
        np.random.seed(123)
        state = np.random.get_state()
        first = sample(16, 43, (0.4, 1.5), (0.75, 1.0))
        second = sample(16, 43, (0.4, 1.5), (0.75, 1.0))
        np.testing.assert_array_equal(first, second)
        state_after = np.random.get_state()
        self.assertEqual(state[0], state_after[0])
        np.testing.assert_array_equal(state[1], state_after[1])
        self.assertEqual(state[2:], state_after[2:])
        self.assertFalse(np.array_equal(first, sample(16, 44, (0.4, 1.5), (0.75, 1.0))))

    def test_bounds_and_stratification(self):
        for seed in (0, 43, 99, 1001):
            pairs = sample(16, seed, (0.4, 1.5), (0.75, 1.0))
            static, dynamic = pairs.T
            self.assertTrue(np.isfinite(pairs).all())
            self.assertTrue(((static >= 0.4) & (static <= 1.5)).all())
            self.assertTrue(((dynamic >= 0.75 * static) & (dynamic <= static)).all())
            bins = np.floor((static - 0.4) / 1.1 * 16).astype(int)
            np.testing.assert_array_equal(np.sort(bins), np.arange(16))

    def test_fixed_range_and_zero_friction(self):
        np.testing.assert_allclose(sample(16, 43, (1.0, 1.0), (1.0, 1.0)), 1.0)
        np.testing.assert_allclose(sample(1, 43, (0.0, 0.0), (0.0, 1.0)), 0.0)

    def test_invalid_inputs(self):
        cases = [
            (0, 43, (0.4, 1.5), (0.75, 1.0)),
            (16, -1, (0.4, 1.5), (0.75, 1.0)),
            (16, 43, (-0.1, 1.5), (0.75, 1.0)),
            (16, 43, (1.5, 0.4), (0.75, 1.0)),
            (16, 43, (0.4, float("nan")), (0.75, 1.0)),
            (16, 43, (0.4, 1.5), (0.75, 1.1)),
            (16, 43, (0.4, 1.5), (0.9, 0.8)),
        ]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                sample(*args)


if __name__ == "__main__":
    unittest.main()
