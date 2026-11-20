# Shared simulation components

This directory contains the reusable implementation of the **simulation experiments**. Estimator algorithms belong in `modules/`. The supported paper workflows start at `reproduce.py` and the corresponding `experiments/<case>/` folders; files here are internal components rather than additional experiments.

## Used by the current paper experiments

| File | Role |
| --- | --- |
| `simulation_units.py` | Canonical per-reference runner; computes MD estimates and private baselines, writes result units |
| `case_cells.py` | Frozen condition grids, reference repetitions, and case-to-unit mapping |
| `pca_protocol.py` | Shared DP-PCA configurations, seed rules, labels, and baseline runner |
| `artifact_io.py` | Atomic JSON receipt writer |
| `final_geometries.py` | Geometries and noise distributions not supplied directly by `geometry/synthetic.py` |
| `fourier_torus.py` | Frozen D=10 Fourier-torus generator (`sample_unit`) |
| `case4_release.py` | Case 4: joint vs per-scale vs per-query releases |
| `lowd_comparators.py` | Baseline comparisons, including the paper's Case 7 |
| `vis100.py` | Supplementary geometry-visualization query batches |
| `env.sh`, `run_cells.sh` | Shared environment and case execution orchestration |

The main case execution path is:

`reproduce.py` → `experiments/<case>/run.py` → case driver → `case_cells.py` → `simulation_units.py`.

The Case 4 release comparison and alternative-baseline experiments use their specialized drivers listed above. The `case7_fourier/` directory retains its original path for compatibility, but represents **supplementary Fourier-torus scans**, not the paper's benchmark Case 7.

## Reproducibility notes

The current paper cases no longer depend on historical campaign managers. Obsolete pilot runners and compatibility shims have been removed from the working tree; their source remains available in the Git history for archival reference.

The frozen simulation settings, per-unit random seeds, output paths, and MD/DP-PCA estimation logic are retained. Targeted regression checks are in `tests/test_shared_refactor.py` and `tests/test_portability.py`. The GitHub Actions workflow also reruns one complete paper unit on Linux. A full numerical rerun of every paper unit is not part of these checks.
