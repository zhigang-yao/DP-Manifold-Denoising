#!/bin/bash
# Case 6: ambient dimension: every unit of this case, into $RUNS/main (resumable; finished units are skipped).
source "$(dirname "$0")/../_shared/env.sh"
"$PY" -B -m experiments._shared.case_cells case6_highd > "$T/experiments/case6_highd/cells.txt"
bash "$T/experiments/_shared/run_cells.sh" 8 10 < "$T/experiments/case6_highd/cells.txt"
