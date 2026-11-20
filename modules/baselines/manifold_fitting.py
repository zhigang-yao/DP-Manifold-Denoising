"""Lossless upstream MF2023 / MF2025 interfaces with audited allocation fixes.

No invented substitute estimator, no ground-truth tuning, no implicit download.
All original source bytes are SHA-verified before use. GPL-3.0 upstream files
reside in external/manifold-fitting, outside this artifact's authored modules.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
import importlib.util
import io
from pathlib import Path
import sys
import types
import numpy as np

from modules.baselines.external_mf_sources import installed_root,file_manifest,UPSTREAM_COMMIT

METHODS=("MF2023","MF2025")
MF2025_DEFAULT_OPTIONS=dict(maxiter=10,epsilon=0.01,alpha0=1,initer=10)
MF2025_DEFAULT_RADIUS=0.2

@dataclass
class FitResult:
    estimate: np.ndarray
    diagnostics: dict

def compatibility_patch(method:str,source:str) -> tuple[str,list[str]]:
    """Apply only audited memory/ambient-shape fixes, guarded by exact strings."""
    edits=[]
    def fix(before,after,label):
        nonlocal source
        if source.count(before)!=1:
            raise RuntimeError(f"Upstream source changed: expected one {label} site")
        source=source.replace(before,after)
        edits.append(label)
    if method=="MF2023":
        fix("    for ii in range(N):",
            "    nbrs = NearestNeighbors(n_neighbors=5).fit(sample)\n\n    for ii in range(N):",
            "fit_knn_once")
        fix("dists = squareform(pdist(np.vstack([x, sample])))[0, 1:]",
            "dists = np.linalg.norm(sample - x[None, :],axis=1)",
            "replace_full_pairwise_distance_by_query_to_sample")
        fix("        nbrs = NearestNeighbors(n_neighbors=5).fit(sample)\n",
            "", "remove_repeated_knn_fit")
    elif method=="MF2025":
        fix("Mout = np.empty(((d + 1), 0))",
            "Mout = np.empty((data.shape[0], 0))",
            "allocate_output_in_ambient_dimension")
    else:
        raise ValueError("unknown external manifold-fitting method")
    return source,edits

@contextlib.contextmanager
def _external_import_path(paths):
    paths=[str(Path(p)) for p in paths]
    # External imports use bare module names, so isolate and discard them.
    names=("getUi","GD_PutM","get_obj_grad")
    prior={name:sys.modules.pop(name,None) for name in names}
    sys.path[:0]=paths
    try:
        yield
    finally:
        for path in paths:
            sys.path.remove(path)
        for name in names:
            sys.modules.pop(name,None)
            if prior[name] is not None:
                sys.modules[name]=prior[name]

def fit(method,reference,queries,*,sigma:float,d:int,
        radius:float=MF2025_DEFAULT_RADIUS,source_root=None,options=None) -> FitResult:
    """Return all-Q original-algorithm predictions and explicit per-query flags.

    MF2023: sigma/sqrt(D+2) as in the archived author comparison.
    MF2025: r=0.2, maxiter=10, epsilon=.01, alpha0=1, initer=10, beta=d+2.
    A false MF2025 moveflag is reported, not relabeled a successful correction.
    """
    if method not in METHODS:raise ValueError(f"Unknown external method {method}")
    ref=np.asarray(reference,float);x=np.asarray(queries,float)
    if (ref.ndim!=2 or x.ndim!=2 or len(ref)<5 or not len(x)
            or ref.shape[1]!=x.shape[1] or not np.isfinite(ref).all()
            or not np.isfinite(x).all() or not 0<d<x.shape[1]):
        raise ValueError("finite [n,D] references, [Q,D] queries, n>=5 and valid d required")
    if not np.isfinite(sigma) or sigma<=0 or not np.isfinite(radius) or radius<=0:
        raise ValueError("positive finite sigma/radius required")
    root=installed_root(source_root)
    file_manifest(root) # validation on EVERY external invocation, no unchecked files
    if method=="MF2023":
        rel="Manifold Fitting/Python/mf/manfit_ours.py"
        external_paths=[root/"Manifold Fitting/Python/mf"]
    else:
        rel="Manifold Fitting under Unbounded Noise/Python/algo/manfit_our.py"
        external_paths=[root/"Manifold Fitting under Unbounded Noise/Python/algo",
                        root/"Manifold Fitting under Unbounded Noise/Python/utils"]
    source=(root/rel).read_text(encoding="utf-8")
    patched,changes=compatibility_patch(method,source)
    module=types.ModuleType("dpmd_external_"+method.lower())
    module.__file__=str(root/rel)
    options=dict(MF2025_DEFAULT_OPTIONS if options is None else options)
    # Avoid leaking massive upstream stdout in batch runs; nothing is hidden
    # in the returned false-moveflag count.
    transcript=io.StringIO()
    with _external_import_path(external_paths),contextlib.redirect_stdout(transcript):
        exec(compile(patched,module.__file__+" [audited allocation compatibility]", "exec"),module.__dict__)
        if method=="MF2023":
            coord_sd=float(sigma/np.sqrt(ref.shape[1]+2))
            if not 0<coord_sd<1:
                raise ValueError("MF2023 original logarithmic radius needs 0<sigma/sqrt(D+2)<1")
            y=module.manfit_ours(ref,coord_sd,x)
            failed=np.zeros(len(x),bool)
            hyperparams=dict(coordinate_noise_sd=coord_sd,
                             radius_policy="upstream fixed sig-dependent radii; no oracle h")
        else:
            options.setdefault("beta",int(d)+2)
            predictions,info=module.manfit_our(ref.T,int(d),float(radius),x.T,options)
            y=np.asarray(predictions).T
            flags=np.asarray(info["moveflag"],bool).reshape(-1)
            if flags.shape!=(len(x),):
                raise ArithmeticError("MF2025 upstream moveflag cardinality mismatch")
            failed=~flags
            hyperparams=dict(radius=radius,options=options)
    y=np.asarray(y,float)
    if y.shape!=x.shape or not np.isfinite(y).all():
        raise FloatingPointError("upstream external fitter emitted invalid output (no RAW substitution)")
    return FitResult(y,dict(method=method,
            upstream_commit=UPSTREAM_COMMIT,source_file=rel,
            compatibility_edits=changes,hyperparameters=hyperparams,
            non_move_count=int(failed.sum()),move_rate=float((~failed).mean()),
            upstream_stdout_excerpt=transcript.getvalue()[:500],
            privacy="nonprivate; any DP guarantee comes from an explicit wrapper"))
