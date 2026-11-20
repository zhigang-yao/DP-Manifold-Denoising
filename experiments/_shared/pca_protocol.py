"""Shared DP-PCA comparison settings and seed schedule for the frozen simulations.

These definitions were extracted unchanged from campaign_2040.py so the
current experiment runner does not import historical campaign orchestration.
Do not change the seed schedule or baseline labels without a replay audit.
"""
import json
import numpy as np
from geometry.synthetic import GEOMETRIES

PCA_CONFIGS = [dict(mode=mode, quantile=q, role=('main' if (mode,q)==('split',.5) else 'sensitivity'),
                   nu_draws=10, alpha=.1)
               for mode,q in [('split',.5),('split',.75),('compose',.5),('compose',.75)]]

def pca_label(config, rule):
    return f"DP-PCA-{config['mode']}-q{config['quantile']:g}-{rule}"

def jsonable(value):
    return json.loads(json.dumps(value,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x),allow_nan=False))

def seed_schedule(unit):
    d=unit['data']; eps=unit['epsilon']; g=d['geometry']
    prefix=[d['seed'],d['replicate'],GEOMETRIES.index(g)+1,int(eps*1000)]
    key=f"{g}_D{d['D']}_n{d['n']}_eps{eps:g}_r{d['replicate']}_structured_shared"
    return dict(private_field=prefix+[81],auxiliary_key=key,pca={
        f"{c['mode']}-q{c['quantile']:g}":dict(
            clip=prefix+[951 if c['mode']=='split' else 953,int(c['quantile']*1000)],
            pca=prefix+[901],aux=prefix+[993]) for c in unit['pca']})

def run_private_pca(data, unit):
    from modules.baselines.dp_pca.estimator import release_and_compare
    predictions={}; diagnostics={}; seeds=seed_schedule(unit)['pca']
    for config in unit['pca']:
        mode=config['mode']; q=config['quantile']; label=f'{mode}-q{q:g}'
        rngs={k:np.random.default_rng(np.random.SeedSequence(s)) for k,s in seeds[label].items()}
        result=release_and_compare(data['reference'],data['queries'],epsilon=unit['epsilon'],
            delta=unit['data']['n']**-1.1,quantile=q,privacy_mode=mode,
            rng_clip=rngs['clip'],rng_pca=rngs['pca'],rng_aux=rngs['aux'],
            nu_draws=config['nu_draws'],alpha=config['alpha'])
        for rule,pred in result.pop('predictions').items(): predictions[pca_label(config,rule.capitalize())]=pred
        diagnostics[label]=result
    return predictions,diagnostics,seeds
