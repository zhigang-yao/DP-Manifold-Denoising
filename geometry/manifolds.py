"""Research-only diameter-two bank with volume sampling and projection targets.

The open S sheet, one-turn spiral sheet and shallow 3D graph generate only
interior public queries. References still cover the full manifold. They do not
acquire the closed-manifold reach certificate.
Noise is added after a fixed geometric normalization, never a sample-based fit.
"""

from dataclasses import replace
from functools import lru_cache
import heapq
import math
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from geometry.geometry_core import (
    geometry_spec,
    sample_geometry,
    project_to_geometry,
    isometric_embedding,
)

CASES = (
    "circle",
    "ellipse",
    "sphere2",
    "s_curve",
    "swiss_roll",
    "torus",
    "sphere3",
    "bowl3",
)
_DIMS = {
    "circle": 1,
    "ellipse": 1,
    "sphere2": 2,
    "s_curve": 2,
    "swiss_roll": 2,
    "torus": 2,
    "sphere3": 3,
    "bowl3": 3,
}
QUERY_NOISE = 0.05
QUERY_BOUNDARY_MARGIN = 0.1
VERSION = "easy_diameter2_interior_queries_v2"


def display_grid(case):
    """Deterministic true-manifold lines/surfaces for manuscript plots only."""
    if case == "affine":
        return np.c_[np.linspace(-1, 1, 100), np.zeros((100, 2))]
    if case == "circle":
        t = np.linspace(0, 2 * np.pi, 240)
        raw = np.c_[np.cos(t), np.sin(t)]
    elif case == "torus":
        theta, phi = np.meshgrid(
            np.linspace(0, 2 * np.pi, 45), np.linspace(0, 2 * np.pi, 23)
        )
        raw = np.stack(
            (
                (2 / 3 + np.cos(phi) / 3) * np.cos(theta),
                (2 / 3 + np.cos(phi) / 3) * np.sin(theta),
                np.sin(phi) / 3,
            ),
            -1,
        )
    elif case == "s_curve":
        t, u = np.meshgrid(np.linspace(-1, 1, 50), np.linspace(-0.6, 0.6, 18))
        raw = np.stack((t, u, 0.15 * np.sin(np.pi * t)), -1)
    else:
        raise ValueError("no paper display grid for geometry")
    meta = geometry_metadata(case)
    return (raw - np.asarray(meta["raw_center"])) * meta["scale"]


def _case(case):
    if case not in _DIMS:
        raise ValueError("unknown easy geometry")
    return _DIMS[case]


@lru_cache(maxsize=8)
def _closed_spec(case):
    if case == "ellipse":
        return replace(
            geometry_spec("ellipse065"),
            name="easy_ellipse080",
            reach_lower_bound=0.64,
            parameters=(("axis_a", 1.0), ("axis_b", 0.8)),
        )
    if case == "torus":
        return replace(
            geometry_spec("torus065035"),
            name="easy_torus_2to1",
            reach_lower_bound=1 / 3,
            parameters=(("major_radius", 2 / 3), ("minor_radius", 1 / 3)),
        )
    return geometry_spec(case)


def _spiral(t):
    t = np.asarray(t)
    r = 2 + t / np.pi
    return np.stack((r * np.cos(t), r * np.sin(t)), axis=-1)


def _spiral_speed(t):
    return np.sqrt((2 + np.asarray(t) / np.pi) ** 2 + 1 / np.pi**2)


def _spiral_length(t):
    b = 1 / np.pi
    r = 2 + np.asarray(t) / np.pi
    return (r * np.sqrt(r * r + b * b) + b * b * np.arcsinh(r / b)) / (2 * b)


@lru_cache(maxsize=8)
def _metadata(case):
    d = _case(case)
    scale = 1.0
    center = np.zeros(d + 1)
    boundary = False
    angle = None
    reach = {
        "circle": 1.0,
        "ellipse": 0.64,
        "sphere2": 1.0,
        "torus": 1 / 3,
        "sphere3": 1.0,
    }.get(case)
    if case in ("circle", "sphere2", "sphere3"):
        rms = 1.0
        curvature = 1.0
    elif case == "ellipse":
        w = lambda t: np.sqrt(np.sin(t) ** 2 + 0.8**2 * np.cos(t) ** 2)
        area = quad(w, 0, 2 * np.pi, epsabs=1e-12)[0]
        rms = np.sqrt(
            quad(
                lambda t: (np.cos(t) ** 2 + 0.8**2 * np.sin(t) ** 2) * w(t),
                0,
                2 * np.pi,
                epsabs=1e-12,
            )[0]
            / area
        )
        curvature = 1 / 0.64
    elif case == "torus":
        rms = np.sqrt((2 / 3) ** 2 + 2 * (1 / 3) ** 2)
        curvature = 3.0
    elif case == "s_curve":
        boundary = True
        a = 0.15
        b = 0.6
        scale = 1 / np.sqrt(1 + b * b)
        w = lambda t: np.sqrt(1 + (a * np.pi * np.cos(np.pi * t)) ** 2)
        area = quad(w, -1, 1, epsabs=1e-12)[0]
        rms = scale * np.sqrt(
            quad(
                lambda t: (t * t + a * a * np.sin(np.pi * t) ** 2) * w(t),
                -1,
                1,
                epsabs=1e-12,
            )[0]
            / area
            + b * b / 3
        )
        curvature = a * np.pi**2 / scale
    elif case == "swiss_roll":
        boundary = True
        b = 1 / np.pi
        # For a fixed angular separation, shifting both points towards the outer
        # endpoint increases distance. The remaining one-variable maximum lies
        # in [pi/2,pi]; the derivative there is strictly decreasing.
        derivative = lambda dt: (
            2 * b * b * dt - 8 * b * (1 - np.cos(dt)) + 8 * (4 - b * dt) * np.sin(dt)
        )
        angle = brentq(derivative, np.pi / 2, np.pi, xtol=5e-15, rtol=1e-14)
        diameter2 = (b * angle) ** 2 + 8 * (4 - b * angle) * (1 - np.cos(angle)) + 4
        scale = 2 / np.sqrt(diameter2)
        area = quad(_spiral_speed, 0, 2 * np.pi, epsabs=1e-12)[0]
        center = np.array(
            [
                quad(
                    lambda t: _spiral(t)[0] * _spiral_speed(t),
                    0,
                    2 * np.pi,
                    epsabs=1e-12,
                )[0]
                / area,
                0.0,
                quad(
                    lambda t: _spiral(t)[1] * _spiral_speed(t),
                    0,
                    2 * np.pi,
                    epsabs=1e-12,
                )[0]
                / area,
            ]
        )
        rms = scale * np.sqrt(
            quad(
                lambda t: (2 + t / np.pi) ** 2 * _spiral_speed(t),
                0,
                2 * np.pi,
                epsabs=1e-12,
            )[0]
            / area
            + 1 / 3
            - center @ center
        )
        curvature = (4 + 2 * b * b) / (4 + b * b) ** 1.5 / scale
    else:
        boundary = True
        a = 0.1
        w = lambda r: r * r * np.sqrt(1 + 4 * a * a * r * r)
        area = quad(w, 0, 1, epsabs=1e-13)[0]
        r2 = quad(lambda r: r * r * w(r), 0, 1, epsabs=1e-13)[0] / area
        r4 = quad(lambda r: r**4 * w(r), 0, 1, epsabs=1e-13)[0] / area
        center[3] = a * r2
        rms = np.sqrt(r2 + a * a * (r4 - r2 * r2))
        curvature = 2 * a
    return dict(
        case=case,
        d=d,
        native_dimension=d + 1,
        diameter=2.0,
        scale=float(scale),
        raw_center=center.tolist(),
        rms_radius=float(rms),
        has_boundary=boundary,
        reach_lower_bound=reach,
        local_curvature_upper=float(curvature),
        diameter_angle_difference=None if angle is None else float(angle),
        sampling_measure="manifold_volume",
        generator_version=VERSION,
        query_sampling_measure="noisy_volume_conditioned_on_projected_interior"
        if boundary
        else "noisy_manifold_volume",
        query_boundary_margin=QUERY_BOUNDARY_MARGIN if boundary else 0.0,
        theory_scope="boundary_benchmark" if boundary else "closed_regular",
        parameters={"s_amplitude": 0.15, "s_half_height": 0.6}
        if case == "s_curve"
        else {
            "inner_radius": 2.0,
            "pitch_per_turn": 2.0,
            "turns": 1.0,
            "half_height": 1.0,
        }
        if case == "swiss_roll"
        else {"quadratic_coefficient": 0.1, "domain_radius": 1.0}
        if case == "bowl3"
        else dict(_closed_spec(case).parameters),
    )


def geometry_metadata(case):
    m = _metadata(case)
    return {
        **m,
        "raw_center": list(m["raw_center"]),
        "parameters": dict(m["parameters"]),
    }


@lru_cache(maxsize=16)
def _embedding(case, D):
    native = _case(case) + 1
    if not isinstance(D, (int, np.integer)) or isinstance(D, bool) or D < native:
        raise ValueError("ambient dimension smaller than native dimension")
    if D == native:
        return np.eye(native)
    return isometric_embedding(
        native, int(D), np.random.SeedSequence([66, 7103, CASES.index(case), int(D)])
    )


def _accept(count, rng, low, high, weight, maximum):
    out = []
    left = count
    while left:
        x = rng.uniform(low, high, max(64, 2 * left))
        kept = x[rng.random(len(x)) <= weight(x) / maximum][:left]
        out.append(kept)
        left -= len(kept)
    return np.concatenate(out)


def _sample_raw(case, count, rng):
    if case not in ("s_curve", "swiss_roll", "bowl3"):
        return sample_geometry(_closed_spec(case), count, rng)
    if case == "s_curve":
        u = _accept(
            count,
            rng,
            -1.0,
            1.0,
            lambda x: np.sqrt(1 + (0.15 * np.pi * np.cos(np.pi * x)) ** 2),
            np.sqrt(1 + (0.15 * np.pi) ** 2),
        )
        return np.c_[u, rng.uniform(-0.6, 0.6, count), 0.15 * np.sin(np.pi * u)]
    if case == "swiss_roll":
        t = _accept(
            count, rng, 0.0, 2 * np.pi, _spiral_speed, np.sqrt(16 + 1 / np.pi**2)
        )
        c = _spiral(t)
        return np.c_[c[:, 0], rng.uniform(-1.0, 1.0, count), c[:, 1]]
    # Uniform ball proposals, then exact graph-volume Jacobian rejection.
    out = []
    left = count
    while left:
        batch = max(64, 2 * left)
        u = rng.standard_normal((batch, 3))
        u /= np.linalg.norm(u, axis=1)[:, None]
        r = rng.random(batch) ** (1 / 3)
        u *= r[:, None]
        keep = rng.random(batch) <= np.sqrt(1 + 0.04 * r * r) / np.sqrt(1.04)
        p = np.c_[u[keep], 0.1 * r[keep] ** 2][:left]
        out.append(p)
        left -= len(p)
    return np.concatenate(out)


def _ellipse_projection(points):
    out = np.empty_like(points)
    a = 1.0
    b = 0.8
    difference = a * a - b * b
    for j, p in enumerate(points):
        x, y = np.abs(p)
        if x == 0:
            px, py = 0.0, b
        elif y == 0:
            c = min(a * x / difference, 1.0)
            px, py = a * c, b * np.sqrt(1 - c * c)
        else:

            def f(t):
                if t == 0:
                    return -b * y
                if t == np.pi / 2:
                    return a * x
                return (
                    -difference * np.sin(t) * np.cos(t)
                    + a * x * np.sin(t)
                    - b * y * np.cos(t)
                )

            t = brentq(f, 0, np.pi / 2, xtol=5e-15, rtol=1e-14)
            px, py = a * np.cos(t), b * np.sin(t)
        out[j] = [np.copysign(px, p[0]), np.copysign(py, p[1])]
    return out


def _spiral_project(q):
    """Numerical global objective certificate from bounded second derivatives.

    Root candidates give upper bounds only. Taylor lower bounds cover the
    entire parameter interval, so a missing stationary point cannot be hidden.
    Ordinary float guards are used; this is not an interval-arithmetic proof.
    """
    b = 1 / math.pi
    Q = math.hypot(*q)

    def f(t):
        r = 2 + b * t
        return (r * math.cos(t) - q[0]) ** 2 + (r * math.sin(t) - q[1]) ** 2

    def df(t):
        r = 2 + b * t
        c = math.cos(t)
        s = math.sin(t)
        return 2 * (b * r - q[0] * (b * c - r * s) - q[1] * (b * s + r * c))

    grid = np.linspace(0, 2 * np.pi, 65)
    grad = [df(t) for t in grid]
    candidates = [0.0, 2 * np.pi]
    for lo, hi, gl, gh in zip(grid[:-1], grid[1:], grad[:-1], grad[1:]):
        if gl <= 0 and gh >= 0:
            candidates.append(brentq(df, lo, hi, xtol=5e-15, rtol=1e-14))
    tbest = min(candidates, key=f)
    best = f(tbest)
    second_bound = 2 * b * b + 2 * Q * (2 * b + 4)
    guard = 64 * np.finfo(float).eps * (1 + Q * Q + 16)

    def lower(lo, hi):
        mid = (lo + hi) / 2
        half = (hi - lo) / 2
        return max(
            0.0, f(mid) - abs(df(mid)) * half - second_bound * half * half / 2 - guard
        )

    heap = [
        (lower(lo, hi), float(lo), float(hi)) for lo, hi in zip(grid[:-1], grid[1:])
    ]
    heapq.heapify(heap)
    tol = max(2e-12, 4 * guard)
    steps = 0
    while heap and best - heap[0][0] > tol:
        bound, lo, hi = heapq.heappop(heap)
        if bound >= best:
            continue
        mid = (lo + hi) / 2
        value = f(mid)
        if value < best:
            best, tbest = value, mid
        for a, c in ((lo, mid), (mid, hi)):
            bound = lower(a, c)
            if bound < best:
                heapq.heappush(heap, (bound, a, c))
        steps += 1
        if steps > 100000:
            raise RuntimeError(
                "spiral projection global objective audit did not finish"
            )
    gap = max(0.0, best - heap[0][0]) if heap else 0.0
    return tbest, gap, steps


def project_with_audit(case, points, *, ambient_dimension=None, compute_boundary=True):
    """Research-only nearest-point target, global-gap and boundary diagnostics."""
    d = _case(case)
    z = np.asarray(points, float)
    if z.ndim != 2 or not np.all(np.isfinite(z)):
        raise ValueError("finite query matrix required")
    D = z.shape[1] if ambient_dimension is None else ambient_dimension
    if D != z.shape[1]:
        raise ValueError("query ambient dimension mismatch")
    Q = _embedding(case, D)
    meta = _metadata(case)
    scale = meta["scale"]
    center = np.array(meta["raw_center"])
    native = z @ Q
    raw = native / scale + center
    gaps = np.zeros(len(z))
    boundary = np.full(len(z), np.inf)
    iterations = np.zeros(len(z), int)
    if case in ("circle", "sphere2", "sphere3", "torus"):
        target = project_to_geometry(_closed_spec(case), raw, np.eye(d + 1))
    elif case == "ellipse":
        target = _ellipse_projection(raw)
    elif case == "s_curve":
        target = np.empty_like(raw)
        for j, p in enumerate(raw):
            # f''/2 >= 1-(a+|z_3|)*a*pi^2. Every generated query satisfies this.
            if 1 - (0.15 + abs(p[2])) * 0.15 * np.pi**2 <= 0:
                raise ValueError(
                    "S target outside the certified convex projection region"
                )

            def derivative(u):
                return (
                    u
                    - p[0]
                    + (0.15 * np.sin(np.pi * u) - p[2])
                    * 0.15
                    * np.pi
                    * np.cos(np.pi * u)
                )

            u = (
                -1.0
                if derivative(-1) >= 0
                else 1.0
                if derivative(1) <= 0
                else brentq(derivative, -1.0, 1.0, xtol=5e-15, rtol=1e-14)
            )
            y = np.clip(p[1], -0.6, 0.6)
            target[j] = [u, y, 0.15 * np.sin(np.pi * u)]
            arc = lambda t: np.sqrt(1 + (0.15 * np.pi * np.cos(np.pi * t)) ** 2)
            if compute_boundary:
                boundary[j] = scale * min(
                    quad(arc, -1, u)[0], quad(arc, u, 1)[0], 0.6 - abs(y)
                )
    elif case == "swiss_roll":
        target = np.empty_like(raw)
        for j, p in enumerate(raw):
            t, gap, steps = _spiral_project((p[0], p[2]))
            c = _spiral(t)
            y = np.clip(p[1], -1.0, 1.0)
            target[j] = [c[0], y, c[1]]
            gaps[j] = gap * scale * scale
            iterations[j] = steps
            boundary[j] = scale * min(
                _spiral_length(t) - _spiral_length(0),
                _spiral_length(2 * np.pi) - _spiral_length(t),
                1 - abs(y),
            )
    else:
        target = np.empty_like(raw)
        for j, p in enumerate(raw):
            if 1 - 0.2 * p[3] <= 0:
                raise ValueError("bowl target outside the convex projection region")
            rho = np.linalg.norm(p[:3])
            f = lambda r: r - rho + 0.2 * r * (0.1 * r * r - p[3])
            r = (
                1.0
                if f(1) <= 0
                else brentq(f, 0.0, 1.0, xtol=5e-15, rtol=1e-14)
                if rho > 0
                else 0.0
            )
            target[j] = np.r_[
                p[:3] * (r / rho) if rho > 0 else np.zeros(3), 0.1 * r * r
            ]
            length = lambda u: (
                (u * np.sqrt(1 + 0.04 * u * u) + np.arcsinh(0.2 * u) / 0.2) / 2
            )
            boundary[j] = length(1) - length(r)
    return dict(
        points=((target - center) * scale) @ Q.T,
        objective_gap_upper=gaps,
        boundary_distance=np.maximum(boundary, 0.0),
        projection_iterations=iterations,
    )
