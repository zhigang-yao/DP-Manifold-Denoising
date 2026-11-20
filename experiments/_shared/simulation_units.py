"""Frozen per-reference simulation runner for the paper's current cases.

Executes one scientific unit at a time with the original seeds and outputs.
The public case grid selects 100 independent reference repetitions. Historical
campaign orchestration is intentionally not imported.
"""
import os
for _k in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ.setdefault(_k, '1')
import json
from pathlib import Path
import time

import numpy as np
import psutil
from experiments._shared.runtime_metrics import peak_rss_bytes

from experiments._shared.pca_protocol import PCA_CONFIGS, jsonable, pca_label
from experiments._shared.artifact_io import atomic_json
from experiments._shared.final_geometries import EXTRA, SEED, SEED_ORDER, make_data
from modules.config import MethodConfig
from modules.metrics import latent_rmse
from geometry.synthetic import GEOMETRIES

METHOD = MethodConfig(backend='structured_shared', inner_B=10, inner_estimator='frobenius_cv', outer_R=100,
                      pointwise_fallback=False, gate_threshold=None, memory_limit_gib=6.).as_dict()
AMBIENT = dict(trefoil=5, torus=5, swiss_roll_candidate=5, linked_tori=5, double_graph=10, bowl4=10,
               fourier_torus=10, circle=5, affine=5, figure_eight=5, helix=5, flat2=5, swiss3=10, catenoid=5)
NGRID = dict(trefoil=(1000, 2000, 4000, 8000), torus=(4000, 8000, 16000, 32000),
             swiss_roll_candidate=(4000, 8000, 16000, 32000), linked_tori=(4000, 8000, 16000, 32000),
             double_graph=(8000, 16000, 32000, 64000), bowl4=(8000, 16000, 32000, 64000),
             fourier_torus=(8000, 16000, 32000, 64000))
DEFAULT_N = {g: grid[1] for g, grid in NGRID.items()}
DEFAULT_N['linked_tori'] = 16000
CONNECTED = ('trefoil', 'torus', 'swiss_roll_candidate', 'double_graph', 'bowl4')
MAIN = CONNECTED + ('linked_tori',)
EPSILONS = (1., 2., 4., 8.)
REPS = 20


def name_of(c):
    return (f"{c['geometry']}_D{c['D']}_n{c['n']}_e{c['epsilon']:g}_a{c['a']:g}_s{c['sigma']:g}"
            f"_{c['noise']}_r{c['replicate']}")


def build_manifest():
    cells = {}
    def add(role, geometry, *, n=None, epsilon=2., a=.08, sigma=.04, noise='uniform_ball', D=None):
        for r in range(REPS):
            c = dict(geometry=geometry, D=D or AMBIENT[geometry], n=n or DEFAULT_N[geometry], epsilon=float(epsilon),
                     a=a, sigma=sigma, noise=noise, Q=20, replicate=r)
            cells.setdefault(name_of(c), dict(c, roles=[]))['roles'].append(role)
    for g in MAIN + ('fourier_torus',):
        for e in EPSILONS: add('epsilon_scan', g, epsilon=e)
        for n in NGRID[g]: add('n_scan', g, n=n)
        if g != 'fourier_torus':
            for a in (.08, .10, .12, .14): add('a_scan', g, a=a)
    for g in CONNECTED:
        for s in (.02, .04, .06, .08): add('sigma_scan', g, sigma=s)
        add('central_noise', g, noise='central')
    for g in ('circle',) + EXTRA: add('extra_geometry', g, n=2000)
    for D in (10, 20, 30, 40, 50):
        add('case6', 'trefoil', n=8000, D=D)
        add('case6', 'double_graph', n=16000, D=D)
    return dict(method=METHOD, pca=PCA_CONFIGS, cells=cells)


def geometry_id(geometry):
    if geometry in GEOMETRIES: return GEOMETRIES.index(geometry) + 1
    return 201 if geometry == 'fourier_torus' else 101 + SEED_ORDER.index(geometry)


def seed_schedule(c):
    # identical to campaign_2040.seed_schedule for the modules.synthetic geometries
    eps = c['epsilon']; g = c['geometry']
    prefix = [SEED, c['replicate'], geometry_id(g), int(eps * 1000)]
    key = f"{g}_D{c['D']}_n{c['n']}_eps{eps:g}_r{c['replicate']}_structured_shared"
    return dict(private_field=prefix + [81], auxiliary_key=key, pca={
        f"{p['mode']}-q{p['quantile']:g}": dict(
            clip=prefix + [951 if p['mode'] == 'split' else 953, int(p['quantile'] * 1000)],
            pca=prefix + [901], aux=prefix + [993]) for p in PCA_CONFIGS})


def run_private_pca(data, c, seeds):
    from modules.baselines.dp_pca.estimator import release_and_compare
    predictions = {}; diagnostics = {}
    for config in PCA_CONFIGS:
        label = f"{config['mode']}-q{config['quantile']:g}"
        rngs = {k: np.random.default_rng(np.random.SeedSequence(s)) for k, s in seeds['pca'][label].items()}
        result = release_and_compare(data['reference'], data['queries'], epsilon=c['epsilon'], delta=c['n'] ** -1.1,
                                     quantile=config['quantile'], privacy_mode=config['mode'],
                                     rng_clip=rngs['clip'], rng_pca=rngs['pca'], rng_aux=rngs['aux'],
                                     nu_draws=config['nu_draws'], alpha=config['alpha'])
        for rule, pred in result.pop('predictions').items(): predictions[pca_label(config, rule.capitalize())] = pred
        diagnostics[label] = {k: result[k] for k in ('radius', 'nu', 'noise_sd', 'private_spectrum') if k in result}
    return predictions, diagnostics


def run_unit(out, name, cell, cache=None):
    from modules.md.estimator import run
    target = Path(out) / 'units' / (name + '.json')
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists(): return
    begin = time.perf_counter()
    data = make_data(cell); seeds = seed_schedule(cell)
    # the memory guard does not change any computation; high-D Case 6 fields need more than the 6 GiB default
    method = dict(METHOD, memory_limit_gib=64.) if cell['D'] > 10 or cell['Q'] > 20 else METHOD
    md = run(data['reference'], data['queries'], epsilon=cell['epsilon'], intrinsic_dimension=data['d'],
             rng=np.random.default_rng(np.random.SeedSequence(seeds['private_field'])),
             config=MethodConfig(**method), unit_id=seeds['auxiliary_key'], public_field_cache=cache)
    md_seconds = time.perf_counter() - begin
    predictions, pca = run_private_pca(data, cell, seeds)
    predictions.update(md.estimates)
    predictions['RAW'] = data['queries']
    rmse = {k: latent_rmse(np.asarray(v, float), data['latent']) for k, v in predictions.items()}
    errors = {k: np.sum((np.asarray(v, float) - data['latent']) ** 2, axis=1) for k, v in predictions.items()}
    np.savez_compressed(target.with_suffix('.npz'), latent=data['latent'], queries=data['queries'],
                        **{'estimate_' + k: np.asarray(v, float) for k, v in predictions.items()})
    atomic_json(target, jsonable(dict(status='COMPLETE', name=name, cell=cell, method=METHOD, seeds=seeds,
        d=data['d'], rmse=rmse, squared_error=errors, selected_h=md.selected_bandwidths, scores=md.scores,
        pca=pca, md_seconds=md_seconds, seconds=time.perf_counter() - begin,
        peak_gib=peak_rss_bytes() / 2 ** 30)))
    print(name, f'{time.perf_counter() - begin:.0f} s', {k: round(rmse[k], 4) for k in ('RAW', 'NP-MD-S', 'DP-MD-S')},
          flush=True)


def main():
    out, command, *names = sys.argv[1:]
    Path(out, 'units').mkdir(parents=True, exist_ok=True)
    manifest = build_manifest()
    path = Path(out) / 'MANIFEST.json'
    if path.exists():
        if json.loads(path.read_text()) != jsonable(manifest): raise RuntimeError('manifest changed')
    else:
        atomic_json(path, jsonable(manifest))
    if command == 'manifest':
        print('\n'.join(manifest['cells']))
        return
    # as campaign_2040.run_worker: the public field factor depends only on public queries and bandwidths, so units
    # of one (geometry, D, a) group share a PublicFieldCache within this process
    from modules.md.public_field_cache import PublicFieldCache
    cache = group = None
    for name in names:
        cell = manifest['cells'][name]
        if (cell['geometry'], cell['D'], cell['a']) != group:
            cache = PublicFieldCache(); group = (cell['geometry'], cell['D'], cell['a'])
        run_unit(out, name, cell, cache)


if __name__ == '__main__':
    main()
