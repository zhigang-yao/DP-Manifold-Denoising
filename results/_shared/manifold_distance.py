"""Distance from points in R^D to the true manifold of the final-campaign geometries (manifold RMSE).

A point is split into its native coordinates x = y @ E and the residual orthogonal to the embedding E used by
modules.synthetic.generate; the squared distance is |residual|^2 + dist(x, native manifold)^2. Native distances:
torus closed form; trefoil by a dense parameter grid refined with a bounded scalar search; Swiss roll by the
planar spiral distance (grid + refinement over theta in [0, 3*pi]) combined with the clipped height; double graph
and bowl as graphs over u in R^3 by local minimisation from u = x[:3] (queries lie well inside the unit ball).
"""
import numpy as np
from scipy.optimize import minimize, minimize_scalar

from geometry import candidate_swiss as cs
from geometry.synthetic import DIMENSIONS, DOUBLE_GRAPH_ALPHA, _rng

SEED = 20261105
EYE = ('double_graph', 'linked_tori', 'bowl4', 'swiss_roll_candidate')


def embedding(g, D):
    native = DIMENSIONS[g][1]
    if D == native or g in EYE: return np.eye(D, native)
    return np.linalg.qr(_rng(SEED, g, 0, 7).standard_normal((D, native)))[0]


def _trefoil(t):
    return np.stack([np.sin(t) + 2 * np.sin(2 * t), np.cos(t) - 2 * np.cos(2 * t), -np.sin(3 * t)], -1) / 3


def _curve_distance(x, curve, lo, hi, grid=4000):
    ts = np.linspace(lo, hi, grid)
    out = []
    for p in x:
        i = int(np.argmin(np.sum((curve(ts) - p) ** 2, 1)))
        a, b = ts[max(i - 1, 0)], ts[min(i + 1, grid - 1)]
        res = minimize_scalar(lambda t: np.sum((curve(np.array([t]))[0] - p) ** 2), bounds=(a, b), method='bounded',
                              options=dict(xatol=1e-12))
        out.append(np.sqrt(min(res.fun, np.sum((curve(ts[i:i + 1])[0] - p) ** 2))))
    return np.array(out)


def _spiral(t):
    r = cs.RADIUS_START + cs.RADIAL_SLOPE * t
    return np.stack([r * np.cos(t), r * np.sin(t)], -1)


def _graph_distance(x, lift):
    out = []
    for p in x:
        f = lambda u: np.sum((lift(u) - p) ** 2)
        res = minimize(f, p[:3], method='BFGS', options=dict(gtol=1e-12))
        out.append(np.sqrt(res.fun))
    return np.array(out)


def native_distance(g, x):
    if g == 'torus':
        R, r = 2 / 3, 1 / 3
        return np.abs(np.hypot(np.hypot(x[:, 0], x[:, 1]) - R, x[:, 2]) - r)
    if g == 'trefoil':
        return _curve_distance(x, _trefoil, 0, 2 * np.pi)
    if g == 'swiss_roll_candidate':
        p = x + np.asarray(cs.SURFACE_AREA_CENTER)
        planar = _curve_distance(p[:, :2], _spiral, 0, cs.THETA_MAX)
        over = np.maximum(np.abs(p[:, 2]) - cs.HALF_HEIGHT, 0)
        return np.hypot(planar, over)
    if g == 'double_graph':
        al = DOUBLE_GRAPH_ALPHA
        return _graph_distance(x, lambda u: np.array([*u, al * (u[0] ** 2 - u[1] ** 2), al * u[1] * u[2]]))
    if g == 'bowl4':
        return _graph_distance(x, lambda u: np.array([*u, .4 * np.dot(u, u)]))
    raise ValueError(g)


def distance(g, D, y):
    E = embedding(g, D)
    x = y @ E
    residual = y - x @ E.T
    return np.hypot(np.linalg.norm(residual, axis=1), native_distance(g, x))
