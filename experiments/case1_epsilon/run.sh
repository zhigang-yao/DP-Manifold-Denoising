#!/bin/bash
# Case 1: privacy budget: every unit of this case, into $RUNS/main (resumable; finished units are skipped).
source "$(dirname "$0")/../_shared/env.sh"
"$PY" -B -m experiments._shared.case_cells case1_epsilon > "$T/experiments/case1_epsilon/cells.txt"
bash "$T/experiments/_shared/run_cells.sh" 12 1 < "$T/experiments/case1_epsilon/cells.txt"
