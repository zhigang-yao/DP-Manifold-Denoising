#!/bin/bash
# Appendix: noise radius and noise law: every unit of this case, into $RUNS/main (resumable; finished units are skipped).
source "$(dirname "$0")/../_shared/env.sh"
"$PY" -B -m experiments._shared.case_cells app_noise_sigma > "$T/experiments/app_noise_sigma/cells.txt"
bash "$T/experiments/_shared/run_cells.sh" 12 1 < "$T/experiments/app_noise_sigma/cells.txt"
