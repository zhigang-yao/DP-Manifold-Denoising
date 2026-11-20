"""Regression checks for the frozen, single-runner paper simulation workflow."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments._shared import pca_protocol, simulation_units
from experiments._shared.artifact_io import atomic_json
from experiments._shared.fourier_torus import sample_unit

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "experiments" / "_shared"


class SharedRefactorTests(unittest.TestCase):
    def test_frozen_pca_settings_and_method_labels(self):
        expected = [
            dict(mode=mode, quantile=q,
                 role="main" if (mode, q) == ("split", 0.5) else "sensitivity",
                 nu_draws=10, alpha=0.1)
            for mode, q in [
                ("split", 0.5), ("split", 0.75),
                ("compose", 0.5), ("compose", 0.75),
            ]
        ]
        self.assertEqual(pca_protocol.PCA_CONFIGS, expected)
        self.assertEqual(
            pca_protocol.pca_label(expected[0], "Gap"),
            "DP-PCA-split-q0.5-Gap",
        )

    def test_seeds_and_public_unit_names(self):
        cell = dict(
            geometry="trefoil", D=5, n=2000, epsilon=1.0,
            a=0.08, sigma=0.04, noise="uniform_ball", Q=20, replicate=0,
        )
        self.assertEqual(
            simulation_units.name_of(cell),
            "trefoil_D5_n2000_e1_a0.08_s0.04_uniform_ball_r0",
        )
        seeds = simulation_units.seed_schedule(cell)
        self.assertEqual(seeds["private_field"], [20261105, 0, 1, 1000, 81])
        self.assertEqual(
            seeds["auxiliary_key"], "trefoil_D5_n2000_eps1_r0_structured_shared",
        )
        self.assertEqual(seeds["pca"]["split-q0.5"]["clip"], [20261105, 0, 1, 1000, 951, 500])
        self.assertEqual(
            seeds["pca"]["compose-q0.75"]["aux"], [20261105, 0, 1, 1000, 993],
        )
        unit = dict(data=dict(cell, seed=20261105), epsilon=1.0, pca=pca_protocol.PCA_CONFIGS)
        self.assertEqual(pca_protocol.seed_schedule(unit), seeds)
        self.assertEqual(
            {k: simulation_units.DEFAULT_N[k] for k in
             ("trefoil", "torus", "swiss_roll_candidate", "double_graph", "bowl4", "linked_tori")},
            dict(trefoil=2000, torus=8000, swiss_roll_candidate=8000,
                 double_graph=16000, bowl4=16000, linked_tori=16000),
        )

    def test_json_receipt_writer_preserves_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "units" / "test.json"
            atomic_json(path, dict(status="COMPLETE", score=np.float64(0.25)))
            self.assertEqual(
                path.read_text(encoding="utf-8"),
                json.dumps(dict(status="COMPLETE", score=np.float64(0.25)),
                           indent=2, sort_keys=True, default=str) + "\n",
            )

    def test_fourier_geometry_seed_is_deterministic(self):
        a = sample_unit(24, 3, 0)
        b = sample_unit(24, 3, 0)
        for key in ("reference", "queries", "latent", "normals", "tangent"):
            self.assertTrue(np.array_equal(a[key], b[key]), key)
        self.assertEqual(a["reference"].shape, (24, 10))
        self.assertEqual(a["queries"].shape, (3, 10))
        self.assertLess(np.max(np.abs(a["normal_dot_tangent"])), 1e-12)

    def test_current_code_has_no_historical_campaign_imports(self):
        for name in ("simulation_units.py", "case_cells.py", "lowd_comparators.py", "vis100.py"):
            source = (SHARED / name).read_text(encoding="utf-8")
            with self.subTest(name=name):
                for historical in ("campaign_2040", "b10_campaign", "paper_campaign"):
                    self.assertNotIn(f"from experiments._shared.{historical} import ", source)


if __name__ == "__main__":
    unittest.main()
