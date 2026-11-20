#!/bin/bash
# Case 5: disconnected manifold (two linked tori): every unit of this case, into $RUNS/main (resumable; finished units are skipped).
source "$(dirname "$0")/../_shared/env.sh"
"$PY" -B -m experiments._shared.case_cells case5_disconnected > "$T/experiments/case5_disconnected/cells.txt"
bash "$T/experiments/_shared/run_cells.sh" 12 1 < "$T/experiments/case5_disconnected/cells.txt"
