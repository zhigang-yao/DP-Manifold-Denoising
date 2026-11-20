#!/bin/bash
# Case 3: query offset, hard vs soft correction: every unit of this case, into $RUNS/main (resumable; finished units are skipped).
source "$(dirname "$0")/../_shared/env.sh"
"$PY" -B -m experiments._shared.case_cells case3_offset > "$T/experiments/case3_offset/cells.txt"
bash "$T/experiments/_shared/run_cells.sh" 12 1 < "$T/experiments/case3_offset/cells.txt"
