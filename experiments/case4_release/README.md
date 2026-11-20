# Case 4: joint / per-scale / per-query field release

This case uses the frozen experiment drivers in `experiments/_shared/` and the algorithm in `modules/`. The final paper figure/table assets are under `results/case4_release/`. Numerical settings and seeds are inherited from the frozen campaign (2026-11-18, revision c).

Run: `uv run python reproduce.py --case case4_release`. To regenerate plots from the archived unit store, use `uv run python reproduce.py --plots`.
