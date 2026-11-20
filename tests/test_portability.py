"""Portability and replay-audit smoke checks."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments._shared.runtime_metrics import peak_rss_bytes
from tests.audit.compare import compare

ROOT = Path(__file__).resolve().parents[1]


class PortabilityTests(unittest.TestCase):
    def test_peak_memory_is_positive_on_this_platform(self):
        self.assertGreater(peak_rss_bytes(), 0)

    def test_shell_scripts_have_lf_endings(self):
        for path in ROOT.rglob("*.sh"):
            with self.subTest(path=str(path.relative_to(ROOT))):
                self.assertNotIn(b"\r", path.read_bytes())

    def test_replay_comparator_rejects_empty_directory(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            with self.assertRaisesRegex(ValueError, "no replay units"):
                compare(Path(a), Path(b))

    def test_replay_comparator_tolerance_and_strict_arrays(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            a, b = Path(a), Path(b)
            for root, rmse in [(a, 0.1000000000001), (b, 0.1)]:
                path = root / "main" / "units" / "one.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({"rmse": {"RAW": rmse, "DP-MD-S": 0.05}}))
            self.assertTrue(compare(a, b))
            self.assertFalse(compare(a, b, rtol=0, atol=0))
            np.savez(a / "main" / "units" / "one.npz", latent=np.array([0.01]))
            self.assertTrue(compare(a, b))  # reduced archive does not contain every array bundle
            self.assertFalse(compare(a, b, require_arrays=True))
            np.savez(b / "main" / "units" / "one.npz", latent=np.array([0.0100000000001]))
            self.assertTrue(compare(a, b, require_arrays=True))

    def test_replay_comparator_detects_missing_method(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            a, b = Path(a), Path(b)
            (a / "one.json").write_text(json.dumps({"rmse": {"RAW": 0.1, "DP-MD-S": 0.05}}))
            (b / "one.json").write_text(json.dumps({"rmse": {"RAW": 0.1}}))
            with self.assertRaisesRegex(ValueError, "no replay RMSE"):
                compare(a, b)


if __name__ == "__main__":
    unittest.main()
