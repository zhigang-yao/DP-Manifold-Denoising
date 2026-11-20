"""Frozen baseline comparisons for the manuscript (MD, DP-PCA, SA-MF).

Uses paired public queries and the paper's original random-seed schedule.
The m=1 comparison evaluates one distinct query from a fixed public bank
per independent reference dataset.

usage: python -m experiments._shared.lowd_comparators <out> <geometry> <first>-<last> <m> <group> [epsilon]
"""
import os
for _k in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ.setdefault(_k, '1')
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import psutil
from experiments._shared.runtime_metrics import peak_rss_bytes

from experiments._shared.simulation_units import DEFAULT_N
from experiments._shared.pca_protocol import PCA_CONFIGS, jsonable, run_private_pca, seed_schedule
from experiments._shared.artifact_io import atomic_json
from modules.config import MethodConfig
from modules.baselines.manifold_fitting import METHODS as MF_METHODS
from modules.metrics import latent_rmse
from modules.baselines.private_wrappers import run_method
from geometry.synthetic import SyntheticConfig, generate

AMBIENT = dict(trefoil=5, torus=5, swiss_roll_candidate=5, double_graph=10, bowl4=10)
EPSILON = 2.
METHOD = MethodConfig(backend='structured_shared', inner_B=10, inner_estimator='frobenius_cv', outer_R=100,
                      pointwise_fallback=False, gate_threshold=None, memory_limit_gib=6.).as_dict()


QBANK = 100


def unit_for(geometry, replicate, m=20):
    data = asdict(SyntheticConfig(geometry=geometry, n=DEFAULT_N[geometry], Q=QBANK if m == 1 else 20,
                                  D=AMBIENT[geometry],
                                  sigma=.04, a=.08, replicate=replicate, seed=20261105))
    return dict(data=data, method=METHOD, epsilon=EPSILON, pca=PCA_CONFIGS)


def mf_seed(unit, m, mode):
    value = json.dumps(dict(data=unit['data'], m=m, mode=mode, epsilon=EPSILON), sort_keys=True,
                       separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:16], 'big')


def run_group(unit, m, group):
    data = generate(SyntheticConfig(**unit['data']))
    first = unit['data']['replicate'] if m == 1 else 0
    data = dict(data, queries=data['queries'][first:first + m], latent=data['latent'][first:first + m])
    n = unit['data']['n']; delta = n ** -1.1
    extra = {}
    if group == 'md':
        from modules.md.estimator import run
        seeds = seed_schedule(unit)
        md = run(data['reference'], data['queries'], epsilon=EPSILON, intrinsic_dimension=data['d'],
                 rng=np.random.default_rng(np.random.SeedSequence(seeds['private_field'])),
                 config=MethodConfig(**unit['method']), unit_id=seeds['auxiliary_key'] + f'_m{m}')
        predictions = dict(md.estimates)
        extra = dict(selected_h=md.selected_bandwidths, scores=md.scores)
    elif group == 'pca':
        predictions, diagnostics, _ = run_private_pca(data, unit)
        extra = dict(pca={k: {x: v[x] for x in ('radius', 'nu', 'noise_sd', 'private_spectrum') if x in v}
                          for k, v in diagnostics.items()})
    elif group == 'mfo':
        # MF2025 radius over the MD bandwidth grid H9 (oracle choice made later in the report, as in the SIMREV
        # appendix); the wrappers do not expose the radius, so their module-level fit is bound to each r in turn
        import functools
        import modules.baselines.private_wrappers as pw
        from modules.baselines.manifold_fitting import fit
        predictions = {}; mechanisms = {}
        try:
            for radius in METHOD['bandwidths']:
                pw.fit = functools.partial(fit, radius=radius)
                # user 2026-11-07: no LDP (never moves at any r), SA at fixed B=50, NP reference on r0-r19 only
                modes = (('NP', 10), ('SA', 50)) if unit['data']['replicate'] < 20 else (('SA', 50),)
                for mode, blocks in modes:
                    label = f"{mode}-MF2025{'-B50' if blocks == 50 else ''}-r{radius:.4f}"
                    fitted = run_method('MF2025', mode, data['reference'], data['queries'], d=data['d'], sigma=.04,
                                        epsilon=EPSILON, delta=delta,
                                        rng=np.random.default_rng(mf_seed(unit, m, mode)), blocks=blocks)
                    predictions[label] = fitted.estimate
                    mechanisms[label] = fitted.diagnostics.get('mechanism')
        finally:
            pw.fit = fit
        extra = dict(mechanisms=mechanisms)
    elif group == 'sa':
        # final design (user 2026-11-08): SA only, MF's own radius, B=50 blocks fixed a priori (the wrapper default
        # B=10 is far above RAW and is not reported)
        predictions = {}; mechanisms = {}
        for blocks in (50,):
            for method in MF_METHODS:
                label = f'SA-{method}' + ('' if blocks == 10 else f'-B{blocks}')
                fitted = run_method(method, 'SA', data['reference'], data['queries'], d=data['d'], sigma=.04,
                                    epsilon=EPSILON, delta=delta, rng=np.random.default_rng(mf_seed(unit, m, 'SA')),
                                    blocks=blocks)
                predictions[label] = fitted.estimate
                mechanisms[label] = fitted.diagnostics.get('mechanism')
        extra = dict(mechanisms=mechanisms)
    elif group == 'sab':
        # SA block-count scan (oracle-tuned reference only): B=10 is the 'mf' group; same seed per B
        predictions = {}; mechanisms = {}
        for blocks in (20, 50, 100):
            for method in MF_METHODS:
                fitted = run_method(method, 'SA', data['reference'], data['queries'], d=data['d'], sigma=.04,
                                    epsilon=EPSILON, delta=delta, rng=np.random.default_rng(mf_seed(unit, m, 'SA')),
                                    blocks=blocks)
                predictions[f'SA-{method}-B{blocks}'] = fitted.estimate
                mechanisms[f'SA-{method}-B{blocks}'] = fitted.diagnostics.get('mechanism')
        extra = dict(mechanisms=mechanisms)
    else:
        predictions = {}; mechanisms = {}
        for mode in ('NP', 'LDP', 'SA'):
            for method in MF_METHODS:
                # one rng per (mode, method) so MF2023 and MF2025 see the same partition and noise draws
                fitted = run_method(method, mode, data['reference'], data['queries'], d=data['d'], sigma=.04,
                                    epsilon=EPSILON, delta=delta, rng=np.random.default_rng(mf_seed(unit, m, mode)))
                predictions[f'{mode}-{method}'] = fitted.estimate
                mechanisms[f'{mode}-{method}'] = fitted.diagnostics.get('mechanism')
        extra = dict(mechanisms=mechanisms)
    predictions['RAW'] = data['queries']
    errors = {k: np.sum((np.asarray(v, float) - data['latent']) ** 2, axis=1) for k, v in predictions.items()}
    rmse = {k: latent_rmse(np.asarray(v, float), data['latent']) for k, v in predictions.items()}
    return data, errors, rmse, extra


def main():
    global EPSILON
    out, geometry, span, m, group = sys.argv[1:6]
    if len(sys.argv) > 6: EPSILON = float(sys.argv[6])
    first, last = map(int, span.split('-')); m = int(m)
    folder = Path(out) / 'units'; folder.mkdir(parents=True, exist_ok=True)
    for r in range(first, last + 1):
        unit = unit_for(geometry, r, m)
        key = f"{geometry}_D{unit['data']['D']}_m{m}_r{r}_{group}" + ('' if EPSILON == 2. else f'_e{EPSILON:g}')
        target = folder / (key + '.json')
        if target.exists():
            continue
        begin = time.perf_counter()
        data, errors, rmse, extra = run_group(unit, m, group)
        seconds = time.perf_counter() - begin
        atomic_json(target, jsonable(dict(status='COMPLETE', key=key, geometry=geometry, D=unit['data']['D'],
            d=data['d'], n=unit['data']['n'], m=m, replicate=r, group=group, epsilon=EPSILON, unit=unit,
            rmse=rmse, squared_error=errors, seconds=seconds,
            peak_gib=peak_rss_bytes() / 2 ** 30, **extra)))
        print(key, f'{seconds:.1f} s', {k: round(v, 4) for k, v in sorted(rmse.items())}, flush=True)


if __name__ == '__main__':
    main()
