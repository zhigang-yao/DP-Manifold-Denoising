"""E0 + variance/h^2, always choose an actual bandwidth (no global RAW option)."""
import math
from fractions import Fraction
import numpy as np

def evaluate(original_masses, corrected_masses, variances, bandwidths, n):
    q = np.asarray(original_masses, float)
    y = np.asarray(corrected_masses, float)
    V = np.asarray(variances, float)
    hs = np.asarray(bandwidths, float)
    if (q.ndim != 2 or y.ndim != 3 or V.shape != (len(q),len(hs))
            or y.shape != (len(q),len(hs),len(hs))
            or q.shape != (len(q),len(hs)) or n < 2):
        raise ValueError("expected original [Q,H], corrected [Q,H,H], variance [Q,H]")
    if (not all(np.isfinite(v).all() for v in (q, y, V, hs))
            or np.any(V < 0) or np.any(hs <= 0)):
        raise FloatingPointError("nonfinite/negative selection input; no RAW substitution")
    floor = 1.0/n
    Q, H = len(q), len(hs)
    E = np.array([math.fsum(
        math.log(max(float(y[i,j,b]),floor)) - math.log(max(float(q[i,b]),floor))
        for i in range(Q) for b in range(H)) / (Q*H) for j in range(H)])
    P = np.mean(V / (hs[None,:]**2), axis=0)
    loss = -E+P
    if not np.isfinite(loss).all():
        raise FloatingPointError("no finite bandwidth score")
    # Exact comparisons on binary input floats, ties resolved by smaller bandwidth.
    keys = [(Fraction(float(loss[j])), float(hs[j])) for j in range(H)]
    j = min(range(H), key=lambda j: keys[j])
    return dict(status="ok",index=int(j),h=float(hs[j]),loss=loss.tolist(),
                E0=E.tolist(),penalty=P.tolist(),raw_allowed=False,
                tie_rule="smallest h",mass_floor=floor)

def select_estimate(candidates, score):
    return np.asarray(candidates)[:,score["index"]].copy()

def position_variance(outputs):
    out=np.asarray(outputs,float)
    if out.ndim < 3 or out.shape[0] < 2 or not np.isfinite(out).all():
        raise ValueError("expected at least two finite auxiliary draws")
    return np.var(out,axis=0,ddof=1).sum(axis=-1)
