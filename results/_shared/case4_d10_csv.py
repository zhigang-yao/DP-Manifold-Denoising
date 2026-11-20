"""Per-dataset Case 4 RMSE of the double graph at D=10 (one row per unit) for results._shared.plotting.

The original CSV (repo results/case4_m_scan_20261107/per_dataset_rmse.csv) has no saved producer; this rebuilds its
double-graph rows from the units. Units without 'Q' are the first m=20 batch.

usage: python -m results._shared.case4_d10_csv <units_dir> <out_csv>
"""
import csv
import json
import sys
from pathlib import Path

COLS = ('DP-MD-S', 'DP-MD-H', 'NP-MD-S', 'NP-MD-H')


def main():
    units, out = map(Path, sys.argv[1:3])
    rows = []
    for p in units.glob('case4b10_double_graph_*.json'):
        r = json.loads(p.read_text())
        rows.append([r['geometry'], r.get('Q', 20), r['backend'], r['replicate'], *(r['rmse'][k] for k in COLS)])
    rows.sort(key=lambda x: (x[1], x[2], x[3]))
    with open(out, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['geometry', 'm', 'backend', 'replicate', *COLS]); w.writerows(rows)


if __name__ == '__main__':
    main()
