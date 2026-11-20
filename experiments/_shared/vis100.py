"""Further-geometry view with 100 public queries (user 2026-11-18; reps100/vis500.py with M and CELLS changed;
the circle batches were run there, the plane and catenoid batches in reps_ext/vis100), reference dataset 0, fixed-query design.

One joint release of 500 queries does not fit in memory (the structured field grows like m^2), so the 500 queries
are released in batches of the size used in the paper: Figure 1 (trefoil) 5 x 100, the further-geometry and
linked-tori views 25 x 20. Each batch is a separate release at the cell's ε. The queries are the first 500 of the
cell's query stream (nested in the paper's m except for linked tori, whose balanced draw depends on m), so batch 0 is
exactly the paper's unit except for linked tori; batch b>0 appends b
to the private seed and to the auxiliary key.

usage: python vis500.py list | run <out> <lane> <nlanes> | merge <out>
"""
import sys
from pathlib import Path

import numpy as np

from experiments._shared.simulation_units import METHOD, seed_schedule
from experiments._shared.final_geometries import make_data
from modules.config import MethodConfig

M = 100
CELLS = [('circle', 5, 2000, 1., 20), ('flat2', 5, 8000, 1., 20), ('catenoid', 5, 8000, 1., 20)]


def cell_of(g, D, n, e, Q):
    return dict(geometry=g, D=D, n=n, epsilon=e, a=.08, sigma=.04, noise='uniform_ball', Q=Q, replicate=0)


def jobs():
    return [(g, D, n, e, size, b) for g, D, n, e, size in CELLS for b in range(M // size)]


def run_batch(out, g, D, n, e, size, b):
    from modules.md.estimator import run
    target = Path(out, f'{g}_b{size}_{b:02d}.npz')
    if target.exists(): return
    data = make_data(cell_of(g, D, n, e, M)); seeds = seed_schedule(cell_of(g, D, n, e, size))
    sl = slice(b * size, (b + 1) * size)
    method = dict(METHOD, memory_limit_gib=64.) if size > 20 else METHOD
    md = run(data['reference'], data['queries'][sl], epsilon=e, intrinsic_dimension=data['d'],
             rng=np.random.default_rng(np.random.SeedSequence(seeds['private_field'] + ([b] if b else []))),
             config=MethodConfig(**method), unit_id=seeds['auxiliary_key'] + (f'_b{b}' if b else ''))
    tmp = target.with_suffix('.tmp.npz')
    estimates = dict(md.estimates, RAW=data['queries'][sl])  # as campaign_final.run_unit
    np.savez_compressed(tmp, latent=data['latent'][sl], queries=data['queries'][sl],
                        **{'estimate_' + k: np.asarray(v, float) for k, v in estimates.items()})
    tmp.replace(target)
    print(target.name, flush=True)


def merge(out):
    for g, D, n, e, size in CELLS:
        parts = [np.load(Path(out, f'{g}_b{size}_{b:02d}.npz')) for b in range(M // size)]
        keys = parts[0].files
        np.savez_compressed(Path(out, f'{g}_m{M}_r0.npz'), **{k: np.concatenate([p[k] for p in parts]) for k in keys})
        print(g, 'merged', {k: parts[0][k].shape for k in keys})


if __name__ == '__main__':
    if sys.argv[1] == 'list':
        print(len(jobs()), jobs()[:3])
    elif sys.argv[1] == 'merge':
        merge(sys.argv[2])
    else:
        out, lane, nlanes = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
        Path(out).mkdir(parents=True, exist_ok=True)
        for job in jobs()[lane::nlanes]:
            run_batch(out, *job)
