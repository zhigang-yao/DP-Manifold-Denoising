#!/bin/bash
# Appendix: further geometries: every unit of this case, into $RUNS/main (resumable; finished units are skipped).
set -euo pipefail
source "$(dirname "$0")/../_shared/env.sh"
"$PY" -B -m experiments._shared.case_cells app_extra_geometries > "$T/experiments/app_geometries/cells.txt"
bash "$T/experiments/_shared/run_cells.sh" 12 1 < "$T/experiments/app_geometries/cells.txt"
# the figure: 100 public queries of reference dataset 0, five releases of 20 per geometry (resumable), then merged
pids=()
for lane in $(seq 0 7); do "$PY" -B -W ignore -m experiments._shared.vis100 run "$RUNS/vis100" $lane 8 & pids+=("$!"); done
for pid in "${pids[@]}"; do wait "$pid"; done
"$PY" -B -m experiments._shared.vis100 merge "$RUNS/vis100"
