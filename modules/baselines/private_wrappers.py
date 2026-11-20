"""Explicit central-DP and local-DP wrappers around pinned MF2023/MF2025.

The wrapper analyses protect the reference data under bounded replace-one
adjacency and public queries. Nonprivate MF predictions used internally must
not be separately published with an SA release.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from modules.privacy import analytic_gaussian_sigma
from modules.baselines.manifold_fitting import FitResult,fit,MF2025_DEFAULT_RADIUS

DEFAULT_CLIP_RADIUS=1.3
DEFAULT_DISPLACEMENT_CAP=0.2
DEFAULT_BLOCKS=10

def _valid_privacy(epsilon,delta):
    if not np.isfinite(epsilon) or epsilon<=0 or not np.isfinite(delta) or not 0<delta<1:
        raise ValueError("epsilon>0 and 0<delta<1 required")

def clip_rows(y,radius:float=DEFAULT_CLIP_RADIUS):
    y=np.asarray(y,float)
    if y.ndim!=2 or not np.isfinite(y).all() or radius<=0:
        raise ValueError("finite row matrix and positive radius required")
    lengths=np.linalg.norm(y,axis=1)
    return y * np.minimum(1.,radius/np.maximum(lengths,np.finfo(float).tiny))[:,None]

def local_release(reference,epsilon,delta,rng,*,clip_radius=DEFAULT_CLIP_RADIUS):
    """One independent Gaussian release per person; no composition over persons."""
    _valid_privacy(epsilon,delta)
    base=clip_rows(reference,clip_radius)
    # One record changes by at most 2c in Euclidean norm.
    sd=float(analytic_gaussian_sigma(epsilon,delta,1.)[0])*2*clip_radius
    released=base+sd*rng.standard_normal(base.shape)
    return released,dict(adjacency="replace-one individual row",
        privacy_model="local_dp_independent_record_releases",
        coordinate_sd=sd,record_sensitivity=2*clip_radius,
        clip_radius=clip_radius,epsilon=epsilon,delta=delta)

def ldp_fit(method,reference,queries,*,d,sigma,epsilon,delta,rng,
            clip_radius=DEFAULT_CLIP_RADIUS,source_root=None):
    released,mechanism=local_release(reference,epsilon,delta,rng,clip_radius=clip_radius)
    fitted=fit(method,released,queries,d=d,sigma=sigma,
               source_root=source_root)
    metadata=dict(fitted.diagnostics)
    metadata.update(mechanism=mechanism,privacy="(epsilon,delta)-LDP, parallel per-person Gaussian releases")
    return FitResult(fitted.estimate,metadata)

def subsample_aggregate(method,reference,queries,*,d,sigma,epsilon,delta,rng,
                        blocks=DEFAULT_BLOCKS,displacement_cap=DEFAULT_DISPLACEMENT_CAP,
                        source_root=None):
    """Data-independent partition, clipped block predictions, *single* joint DP output.

    For Q public queries, each block emits a stacked Q*D displacement vector
    with norm <= displacement_cap*sqrt(Q). Only one block can change under a
    replace-one adjacency for a fixed partition, so the block *mean* has
    sensitivity <=2*displacement_cap*sqrt(Q)/blocks.
    """
    _valid_privacy(epsilon,delta)
    y=np.asarray(reference,float);x=np.asarray(queries,float)
    if y.ndim!=2 or x.ndim!=2 or x.shape[1]!=y.shape[1] or not np.isfinite(y).all() or not np.isfinite(x).all():
        raise ValueError("finite compatible reference/query matrices required")
    if isinstance(blocks,bool) or not isinstance(blocks,int) or blocks<2 or len(y)//blocks<5:
        raise ValueError("need >=2 fixed blocks and at least 5 rows in each block")
    if not np.isfinite(displacement_cap) or displacement_cap<=0:
        raise ValueError("positive displacement cap required")
    Q=len(x)
    assignments=np.array_split(rng.permutation(len(y)),blocks)
    values=[]
    flag_count=0
    for idx in assignments:
        r=fit(method,y[idx],x,d=d,sigma=sigma,source_root=source_root)
        displacement=clip_rows(r.estimate-x,displacement_cap)
        values.append(displacement)
        flag_count+=r.diagnostics["non_move_count"]
    mean=np.mean(np.stack(values,axis=0),axis=0)
    sensitivity=2*displacement_cap*np.sqrt(Q)/blocks
    sd=float(analytic_gaussian_sigma(epsilon,delta,1.)[0])*sensitivity
    private=x+mean+sd*rng.standard_normal(x.shape)
    if not np.isfinite(private).all():
        raise FloatingPointError("nonfinite SA release")
    return FitResult(private,dict(method=f"SA-{method}",privacy="central (epsilon,delta)-DP",
        mechanism=dict(adjacency="replace-one",epsilon=epsilon,delta=delta,
                       blocks=blocks,block_min_count=min(map(len,assignments)),
                       displacement_cap=displacement_cap,query_count=Q,
                       vector_l2_sensitivity=sensitivity,gaussian_coordinate_sd=sd,
                       partition="data-independent random partition"),
        component_non_move_count=flag_count,upstream_nonprivate_intermediates_not_releasable=True))

def run_method(method,mode,reference,queries,*,d,sigma,epsilon,delta,rng,
               source_root=None,blocks=DEFAULT_BLOCKS):
    if mode=="NP":
        return fit(method,reference,queries,d=d,sigma=sigma,source_root=source_root)
    if mode=="LDP":
        return ldp_fit(method,reference,queries,d=d,sigma=sigma,epsilon=epsilon,
                       delta=delta,rng=rng,source_root=source_root)
    if mode=="SA":
        return subsample_aggregate(method,reference,queries,d=d,sigma=sigma,
                    epsilon=epsilon,delta=delta,rng=rng,source_root=source_root,blocks=blocks)
    raise ValueError("mode must be NP, LDP, or SA")
