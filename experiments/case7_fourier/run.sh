#!/bin/bash
# Case 7: Fourier torus: every unit of this case, into $RUNS/main (resumable; finished units are skipped).
source "$(dirname "$0")/../_shared/env.sh"
"$PY" -B -m experiments._shared.case_cells case7_fourier > "$T/experiments/case7_fourier/cells.txt"
bash "$T/experiments/_shared/run_cells.sh" 12 1 < "$T/experiments/case7_fourier/cells.txt"
