"""Lightweight package and final-artifact layout smoke checks."""
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class LayoutTest(unittest.TestCase):
    def test_core_paths(self):
        for p in ("modules/md/estimator.py", "modules/baselines/dp_pca/mechanism.py",
                  "modules/baselines/dp_pca/estimator.py", "geometry/synthetic.py",
                  "experiments/_shared/case_cells.py", "results/_shared/plotting.py"):
            with self.subTest(path=p):
                self.assertTrue((ROOT / p).is_file())

if __name__ == "__main__":
    unittest.main()
