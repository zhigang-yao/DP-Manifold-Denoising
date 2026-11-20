"""Small, self-contained trefoil example (not the frozen paper simulation)."""

import numpy as np

from geometry.synthetic import SyntheticConfig, generate
from modules.config import MethodConfig
from modules.md.estimator import run
from modules.metrics import latent_rmse


def main():
    # Synthetic reference observations, public queries, and evaluation-only truth.
    data = generate(
        SyntheticConfig(
            geometry="trefoil",
            n=800,
            Q=5,
            D=5,
            sigma=0.04,
            a=0.08,
            seed=20261105,
        )
    )

    # Fewer auxiliary draws than in the paper's full simulation for a quick demo.
    cfg = MethodConfig(
        backend="structured_shared",
        inner_B=10,
        inner_estimator="frobenius_cv",
        outer_R=20,
    )
    result = run(
        data["reference"],
        data["queries"],
        epsilon=2.0,
        intrinsic_dimension=data["d"],
        rng=np.random.default_rng(2026),
        config=cfg,
        unit_id="trefoil_quickstart",
    )

    # Synthetic truth is used only for evaluation, never passed to run().
    for method in ("RAW", "NP-MD-S", "DP-MD-S"):
        error = latent_rmse(result.estimates[method], data["latent"])
        print(f"{method:8s}  latent RMSE = {error:.4f}")

    print("DP-MD-S selected bandwidth:", result.selected_bandwidths["DP-MD-S"])
    print("Denoised query shape:", result.estimates["DP-MD-S"].shape)

    # result also includes NP estimates and diagnostics; do not publish it as a DP release.


if __name__ == "__main__":
    main()
