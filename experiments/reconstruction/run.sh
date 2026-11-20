#!/bin/bash
# Figure 1: trefoil, D=5, n=2000, eps=1, m=100 public queries, reference dataset 0 (campaign estimator and seeds).
source "$(dirname "$0")/../_shared/env.sh"
O="$RUNS/fig1"; mkdir -p "$O/units"
[ -f "$O/units/trefoil_D5_n2000_e1_m100_r0.json" ] || "$PY" -B -W ignore -c "
from experiments._shared.simulation_units import run_unit
cell = dict(geometry='trefoil', D=5, n=2000, epsilon=1., a=.08, sigma=.04, noise='uniform_ball', Q=100, replicate=0)
run_unit('$(cygpath -m "$O" 2>/dev/null || echo "$O")', 'trefoil_D5_n2000_e1_m100_r0', cell)" > "$O/run.log" 2>&1
