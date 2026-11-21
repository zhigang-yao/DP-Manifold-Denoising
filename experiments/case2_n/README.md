# Case 2: reference sample size

This case uses the frozen experiment drivers in `experiments/_shared/` and the algorithm in `modules/`. The final paper figure/table assets are under `results/case2_n/`. Numerical settings and seeds are those of the frozen campaign (2026-11-20), in which the public queries are redrawn for every reference repetition.

Run: `uv run python reproduce.py --case case2_n`. To regenerate plots from the archived unit store, use `uv run python reproduce.py --plots`.
