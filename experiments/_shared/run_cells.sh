#!/bin/bash
# Run campaign units (names on stdin, from experiments._shared.case_cells) into $RUNS/main, skipping units whose JSON exists.
# Any failed subprocess must propagate to the calling paper case.
# usage: ... | bash experiments/_shared/run_cells.sh [parallel=12] [chunk=1]
#   chunk>1 groups up to <chunk> replicates of one (geometry, D, n, eps, a, sigma, noise) cell per process, so the
#   public field factor is computed once per process (used for Case 6); results do not depend on the grouping.
set -euo pipefail
source "$(dirname "$0")/env.sh"
P="${1:-12}"; CHUNK="${2:-1}"
O="$RUNS/main"; mkdir -p "$O/units" "$O/logs"
one() { "$PY" -B -W ignore -m experiments._shared.case_cells run "$O" "$@" > "$O/logs/$(echo $1 | sed -E 's/_r[0-9]+$//')_$.log" 2>&1 || { echo "FAILED $1" >&2; return 1; }; }
export -f one; export O
while read n; do [ -f "$O/units/$n.json" ] || echo "$n"; done |
  awk -F'_r' -v k="$CHUNK" '{g=$1; c[g]++; print g"#"int((c[g]-1)/k)" "$0}' | sort -k1,1 -s |
  awk '{if($1!=p){if(p!="")print line; line=$2; p=$1} else line=line" "$2} END{if(p!="")print line}' |
  xargs -r -P "$P" -L 1 bash -c 'one "$0" "$@"'
