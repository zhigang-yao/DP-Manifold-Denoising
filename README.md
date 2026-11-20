# Differentially Private Manifold Denoising

This repository contains the simulation code and committed results for *Differentially Private Manifold Denoising*. The main method, **DP-MD-S**, uses a private reference dataset to correct public query points toward an unknown manifold. For the statistical methodology and privacy guarantees, see the [paper on arXiv](https://arxiv.org/abs/2604.00942).

## Paper

Jiaqi Wu, Yiqing Sun, and Zhigang Yao. *Differentially Private Manifold Denoising*. arXiv:2604.00942, 2026. [Read the paper](https://arxiv.org/abs/2604.00942).

## Method overview

Let the private reference sample be

$$
\mathcal{D}=\{\mathbf{y}_i\}_{i=1}^{n},\qquad
\mathbf{y}_i=\mathbf{x}_i+\boldsymbol{\epsilon}_i,\qquad
\mathbf{x}_i\in\mathcal{M}\subset\mathbb{R}^{D},
$$

where the unknown manifold $\mathcal{M}$ has intrinsic dimension $d$. Given $m$ public queries $\mathbf{z}_1,\ldots,\mathbf{z}_m$, the objective is to estimate their manifold projections,

$$
\mathbf{z}_j^\star=\pi_{\mathcal{M}}(\mathbf{z}_j),
\qquad j=1,\ldots,m,
$$

while protecting the private reference sample under $(\varepsilon,\delta)$-differential privacy.

The implementation includes private soft correction (**DP-MD-S**), its non-private counterpart (**NP-MD-S**), and hard-projection versions (**DP-MD-H** and **NP-MD-H**). Private corrections use Gaussian-perturbed local moments and a shared Gaussian field across queries and bandwidths. **RAW** denotes the uncorrected queries. The paper describes the estimators, adaptive bandwidth selection, and privacy analysis in detail.

## Code guide

The main estimator interface is `modules.md.estimator.run`. It accepts a reference array of shape `(n, D)` and a public query array of shape `(m, D)`, and returns the four MD estimates, selected bandwidths, and diagnostics. The paired interface requests `intrinsic_dimension` for the hard estimators; the soft correction does not use the true intrinsic dimension.

| Module | Main function or class | Purpose |
| --- | --- | --- |
| `modules/config.py` | `MethodConfig` | Public algorithm settings, including bandwidths and Gaussian-field backend |
| `modules/md/estimator.py` | `run`, `Decoder` | Run the four MD estimators and compute hard/soft corrections |
| `modules/md/moments.py` | `empirical`, `unpack` | Evaluate Gaussian-weighted local moments and decode their coordinate representation |
| `modules/md/backends.py` | `Field` | Manage joint, per-scale, and per-query Gaussian-field releases |
| `modules/md/structured_shared_field.py` | `SharedStructuredField` | Sample the shared field and conditionally evaluate it at corrected query locations |
| `modules/md/projection.py` | `WeightedMomentProjector` | Stabilize private moments with the weighted PSD-plus-box projection |
| `modules/md/inner_cv.py` | `FrobeniusMoment`, `operator_second_moment_cv` | Estimate the covariance perturbation scale using auxiliary draws and a control variate |
| `modules/md/bandwidth.py` | `evaluate`, `select_estimate` | Score candidate bandwidths and select the final correction |
| `modules/privacy.py` | `analytic_gaussian_sigma` | Calibrate Gaussian noise for the privacy budget |
| `modules/baselines/dp_pca/estimator.py` | `release_and_compare` | Global DP-PCA comparison |
| `geometry/` | `synthetic.py` and geometry-specific generators | Generate reference samples, observation noise, and public queries |
| `modules/metrics.py` | `latent_rmse` | Evaluate error against synthetic latent targets |

`modules/baselines/` additionally contains the subsample-and-aggregate manifold-fitting comparators (SA-MF2023 and SA-MF2025). Their pinned upstream sources are installed separately.

## Quick start

Run a small, self-contained trefoil example using the synthetic data generator (no external dataset required):

```bash
uv run python -m examples.trefoil_quickstart
```

The example generates a noisy reference sample and public queries, runs the paired estimators, and reports latent RMSE for RAW, NP-MD-S, and DP-MD-S.

## Simulation experiments

Each experiment folder corresponds to the following figures or tables in the paper and supplement. Final outputs are available in `results/`.

| Experiment | Result | Output file |
| --- | --- | --- |
| `reconstruction/` | Trefoil reconstruction | [`sim_reconstruction.pdf`](results/reconstruction/sim_reconstruction.pdf) |
| `case1_epsilon/` | Case 1: privacy budget | [`sim_rmse_eps_half.pdf`](results/case1_epsilon/sim_rmse_eps_half.pdf) |
| `case2_n/` | Case 2: reference sample size | [`sim_rmse_n_half.pdf`](results/case2_n/sim_rmse_n_half.pdf) |
| `case3_offset/` | Case 3: hard versus soft correction | [`tab_sim_hard_soft_a.tex`](results/case3_offset/tab_sim_hard_soft_a.tex) |
| `case4_release/` | Case 4: shared versus independent releases | [`sim_release_m_grid.pdf`](results/case4_release/sim_release_m_grid.pdf) |
| `case5_disconnected/` | Case 5: linked tori | [`sim_disconnected.pdf`](results/case5_disconnected/sim_disconnected.pdf) |
| `case6_highd/` | Case 6: ambient dimension | [`sim_highd.pdf`](results/case6_highd/sim_highd.pdf) |
| `app_baselines/` | Case 7: benchmarks against global DP-PCA and SA-MF methods | [`tab_app_baselines_m20.tex`](results/app_baselines/tab_app_baselines_m20.tex) |
| `app_noise_sigma/` | Supplementary measurement-noise experiments | [`app_sim_sigma.pdf`](results/app_noise_sigma/app_sim_sigma.pdf) |
| `app_geometries/` | Supplementary geometries | [`app_sim_geometries.pdf`](results/app_geometries/app_sim_geometries.pdf) |
| `case7_fourier/` | Supplementary Fourier-torus sample-size and privacy-budget scans | [`sim_fourier.pdf`](results/case7_fourier/sim_fourier.pdf) |


## Installation and reproduction

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/). The project specifies Python `>=3.11,<3.12` in `pyproject.toml`; `uv.lock` records the resolved dependency versions. Run `uv sync --locked` to install the locked environment. The simulation and plot reproduction commands also require a Bash environment (e.g., Git Bash on Windows).

```bash
uv sync --locked

# Recompute one simulation study
uv run python reproduce.py --case case1_epsilon

# Regenerate paper figures and tables from saved results
uv run python reproduce.py --plots
```

For figure/table regeneration from saved runs, place the plotting data in `data/runs/`. Those run data are not included in the tracked source files; the committed figures and tables are in `results/`. To reproduce the SA-MF benchmarks, first install the verified upstream sources:

```bash
uv run python -m modules.baselines.external_mf_sources --install
```

Restricted UK Biobank data are not included in this simulation code repository.

## Citation

If you use this repository, please cite the paper:

```bibtex
@misc{wu2026differentially,
  title={Differentially Private Manifold Denoising},
  author={Wu, Jiaqi and Sun, Yiqing and Yao, Zhigang},
  year={2026},
  eprint={2604.00942},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2604.00942}
}
```

## License

Copyright (c) 2026 Yaogroup. This repository is licensed under the [MIT License](LICENSE).
