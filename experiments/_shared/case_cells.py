"""Campaign units of each simulation case of the paper, in paper order, at 100 reference datasets (2026-11-16).

The cells are those of simulation_units.build_manifest (unchanged); this module sets the privacy budget of each case
and extends the replicates to 0..99. Seeds and unit names follow simulation_units (name_of, seed_schedule), so a unit
is the same computation wherever it is listed. A unit can belong to several cases (the ε=1 default cell of a
geometry is in Cases 1-3); units are stored once in $RUNS/main and every case reads the same unit.

ε per case:
  Case 1 (epsilon_scan)                 the grid {1, 2, 4, 8}
  Case 5, linked tori                   n scan at ε=2, and the grid; the a scan is not reported and not listed
  Case 6 (case6)                        ε=2
  Supplementary Fourier torus          n scan at ε=1, and the grid
  every other scan (n, a, σ, central noise, further geometries)   ε=1

Further geometries (2026-11-18): circle (D=5, n=2000), flat 2-plane (D=5, n=8000), three-dimensional Swiss roll
(D=10, n=16000); the affine segment, figure-eight and helix of the manifest are no longer listed.

usage: python -m experiments._shared.case_cells <case>               print the unit names of one case
       python -m experiments._shared.case_cells --summary            units per case
       python -m experiments._shared.case_cells run <out> <name>...  run units into <out>/units (finished units are skipped)
"""
import sys

from experiments._shared.simulation_units import CONNECTED, build_manifest, name_of, run_unit

EXTRA_CELLS = (('circle', 5, 2000), ('flat2', 5, 8000), ('catenoid', 5, 8000))

REPS = 100

# case -> (roles, geometries); the roles are those of campaign_final.build_manifest
CASES = {
    'case1_epsilon': ({'epsilon_scan'}, CONNECTED),
    'case2_n': ({'n_scan'}, CONNECTED),
    'case3_offset': ({'a_scan'}, CONNECTED),
    'case5_disconnected': ({'epsilon_scan', 'n_scan'}, ('linked_tori',)),
    'case6_highd': ({'case6'}, ('trefoil', 'double_graph')),
    'case7_fourier': ({'epsilon_scan', 'n_scan'}, ('fourier_torus',)),
    'app_noise_sigma': ({'sigma_scan', 'central_noise'}, CONNECTED),
    'app_extra_geometries': (set(), ()),  # EXTRA_CELLS
}


def epsilon_of(role, cell):
    if role in ('epsilon_scan', 'case6'): return cell['epsilon']
    return 2. if cell['geometry'] == 'linked_tori' else 1.


def units(case):
    roles, geometries = CASES[case]
    out = {}
    if case == 'app_extra_geometries':
        for g, D, n in EXTRA_CELLS:
            for r in range(REPS):
                cell = dict(geometry=g, D=D, n=n, epsilon=1., a=.08, sigma=.04, noise='uniform_ball', Q=20, replicate=r)
                out[name_of(cell)] = cell
        return out
    for c in build_manifest()['cells'].values():
        if c['replicate'] or c['geometry'] not in geometries: continue
        base = {k: v for k, v in c.items() if k != 'roles'}
        for role in roles & set(c['roles']):
            for r in range(REPS):
                cell = dict(base, epsilon=float(epsilon_of(role, c)), replicate=r)
                out[name_of(cell)] = cell
    return out


def all_units():
    out = {}
    for case in CASES: out.update(units(case))
    return out


def main():
    if sys.argv[1] == '--summary':
        for case in CASES: print(f'{case:22s} {len(units(case))}')
        print(f'{"all (distinct)":22s} {len(all_units())}')
    elif sys.argv[1] == 'run':
        # as simulation_units.main: units of one (geometry, D, a) group share a PublicFieldCache within this process
        from modules.md.public_field_cache import PublicFieldCache
        out, names = sys.argv[2], sys.argv[3:]
        cells = all_units(); cache = group = None
        for name in names:
            cell = cells[name]
            if (cell['geometry'], cell['D'], cell['a']) != group:
                cache = PublicFieldCache(); group = (cell['geometry'], cell['D'], cell['a'])
            run_unit(out, name, cell, cache)
    else:
        print('\n'.join(units(sys.argv[1])))


if __name__ == '__main__':
    main()
