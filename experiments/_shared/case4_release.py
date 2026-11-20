"""Case 4: joint (structured_shared), per-scale (structured_independent) and per-query (structured_per_query)
releases as the number m=Q of public queries varies; one geometry/release/m and a range of reference datasets per
process.

usage: python -m experiments._shared.case4_release <out_dir> <geometry> <first>-<last> <backend> <Q> <D> [memory_gib=64] [epsilon=2]

Merges case4_batch.py (D=10, double graph) and case4_batch_d.py (D=5, trefoil and torus) of 2026-11-06/08; same
settings, seeds and unit keys. Case 4 has its own data seeds (case4_pilot), not the campaign's. The double-graph
units were drawn with the rejection bound 1.22 hard-coded at the time; the campaign modules use the exact bound
sqrt((1+4a^2)(1+a^2)) = 1.2175 at a=.3. Both give the same distribution, but the realized samples differ, so the
bound is pinned to 1.22 here to reproduce the paper's units bit for bit.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import psutil

import geometry.synthetic as synthetic_module
from modules.config import MethodConfig
from modules.methods import run_all
from geometry.synthetic import SyntheticConfig, generate

GEOMETRY_ID = {"trefoil": 1, "torus": 2, "double_graph": 6}
BASE_N = dict(trefoil=2000, torus=8000, double_graph=16000)
CASE4_DOUBLE_GRAPH_BOUND = 1.22
EPSILON = 2.


def main():
    out, geometry, span, backend, Q, D, *rest = sys.argv[1:]
    first, last = map(int, span.split('-'))
    Q, D, memory = int(Q), int(D), float(rest[0]) if rest else 64.
    epsilon = float(rest[1]) if len(rest) > 1 else EPSILON
    synthetic_module.DOUBLE_GRAPH_VOLUME_BOUND = CASE4_DOUBLE_GRAPH_BOUND
    method = MethodConfig(backend=backend, inner_B=10, inner_estimator="frobenius_cv", outer_R=100,
                          pointwise_fallback=False, memory_limit_gib=memory)
    Path(out).mkdir(parents=True, exist_ok=True)
    for replicate in range(first, last + 1):
        synthetic = SyntheticConfig(geometry=geometry, n=BASE_N[geometry], Q=Q, D=D, sigma=.04, a=.08,
                                    replicate=replicate)
        key = f"case4b10_{geometry}_D{D}_Q{Q}_n{synthetic.n}_eps{epsilon:g}_r{replicate}_{backend}"
        if (Path(out) / f'{key}.json').exists():
            continue
        method.validate(synthetic.D)
        data = generate(synthetic)
        seed = np.random.SeedSequence([synthetic.seed, synthetic.replicate, GEOMETRY_ID[geometry],
                                       int(epsilon * 1000), 81])
        started = time.perf_counter()
        result = run_all(data["reference"], data["queries"], epsilon=epsilon, d=data["d"],
                         rng=np.random.default_rng(seed), config=method, unit_id=key, cache_dir=None)
        se = {k: np.sum((v - data['latent']) ** 2, axis=1).tolist() for k, v in result.estimates.items()}
        record = dict(key=key, geometry=geometry, replicate=replicate, backend=backend, Q=Q,
                      seconds=time.perf_counter() - started,
                      sd_multiplier=result.diagnostics['sd'] / result.diagnostics['base_sd'],
                      stage_timings=result.diagnostics['stage_timings'],
                      rmse={k: float(np.sqrt(np.mean(v))) for k, v in se.items()}, squared_error=se,
                      selected_h=result.selected_bandwidths, threads=int(os.environ.get('DPMD_THREADS', '1')),
                      peak_gib=getattr(psutil.Process().memory_info(), 'peak_wset', 0) / 2**30)
        (Path(out) / f'{key}.json').write_text(json.dumps(record, indent=1))
        print(key, round(record['seconds'], 1), 's', {k: round(v, 4) for k, v in record['rmse'].items()}, flush=True)


if __name__ == '__main__':
    main()
