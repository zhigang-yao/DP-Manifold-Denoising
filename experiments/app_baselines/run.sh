#!/bin/bash
# Alternative private estimators: SA (MF2023/MF2025, B=50 blocks) at m=20 and m=1, and MD / DP-PCA at m=1,
# for eps in {1,2,4,8}; the m=20 MD / DP-PCA numbers come from the campaign units (Cases 1-3). Resumable.
set -euo pipefail
source "$(dirname "$0")/../_shared/env.sh"
mkdir -p "$RUNS/comp/logs" "$RUNS/comp_m1_eps2/logs"
cmp() { "$PY" -B -W ignore -m experiments._shared.lowd_comparators "$1" $2 $3 $4 $5 $6 > "$1/logs/$2_m$4_$5_e$6_$3.log" 2>&1 || { echo "FAILED comp $*" >&2; return 1; }; }
export -f cmp
for e in 2 1 4 8; do for g in trefoil torus swiss_roll_candidate double_graph bowl4; do
  for c in 0-19 20-99; do echo "$RUNS/comp $g $c 20 sa $e"; done
  for c in 0-24 25-49 50-74 75-99; do echo "$RUNS/comp $g $c 1 sa $e"; done
  # m=1 MD / DP-PCA: eps 2 of the d<=2 geometries and the bowl lives in comp_m1_eps2 (run first, 2026-11-07)
  if [ $e = 2 ] && [ $g != double_graph ]; then o="$RUNS/comp_m1_eps2"; else o="$RUNS/comp"; fi
  for c in 0-49 50-99; do echo "$o $g $c 1 md $e"; echo "$o $g $c 1 pca $e"; done
done; done | xargs -P 4 -L 1 bash -c 'cmp $0 $1 $2 $3 $4 $5'
