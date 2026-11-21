#!/bin/bash
# Case 4: joint / per-scale / per-query release vs m at ε=1. D=5 trefoil and torus, D=10 bowl; 100 references each.
# One process per (geometry, release, m, 10 references); resumable.
set -euo pipefail
source "$(dirname "$0")/../_shared/env.sh"
mkdir -p "$RUNS/case4/logs" "$RUNS/case4_d10/units/logs"
c4() { [ $5 = 10 ] && o="$RUNS/case4_d10/units" || o="$RUNS/case4"
  "$PY" -B -W ignore -m experiments._shared.case4_release "$o" $1 $2 $3 $4 $5 64 1 > "$o/logs/$1_$3_Q$4_$2.log" 2>&1 || { echo "FAILED case4 $*" >&2; return 1; }; }
export -f c4
for Q in 10 20 50 100; do for b in structured_shared structured_independent structured_per_query; do
  for r in 0-9 10-19 20-29 30-39 40-49 50-59 60-69 70-79 80-89 90-99; do echo "trefoil $r $b $Q 5"; echo "torus $r $b $Q 5"; echo "bowl4 $r $b $Q 10"; done
done; done | xargs -P 8 -L 1 bash -c 'c4 $0 $1 $2 $3 $4'
"$PY" -B -m results._shared.case4_d10_csv "$RUNS/case4_d10/units" "$RUNS/case4_d10/per_dataset_rmse.csv"
