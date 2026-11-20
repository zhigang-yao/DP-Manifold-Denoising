#!/bin/bash
# Case 2: sample size: every unit of this case, into $RUNS/main (resumable; finished units are skipped).
source "$(dirname "$0")/../_shared/env.sh"
"$PY" -B -m experiments._shared.case_cells case2_n > "$T/experiments/case2_n/cells.txt"
bash "$T/experiments/_shared/run_cells.sh" 12 1 < "$T/experiments/case2_n/cells.txt"
