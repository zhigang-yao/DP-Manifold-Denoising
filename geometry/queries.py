"""Synthetic fixed-distance queries in the complete normal space.

True geometry and distance are generation/evaluation inputs only, never tuning.
Open-bank projection audits are numerical checks, not global reach certificates.
"""

import numpy as np
from . import manifolds as g

VERSION = "easy_fixed_normal_distance_v1"


def _normal(case, points):
    meta = g.geometry_metadata(case)
    Q = g._embedding(case, points.shape[1])
    raw = points @ Q / meta["scale"] + np.asarray(meta["raw_center"])
    if case in ("circle", "sphere2", "sphere3"):
        n = raw.copy()
    elif case == "ellipse":
        n = raw / np.array([1.0, 0.64])
    elif case == "torus":
        rho = np.linalg.norm(raw[:, :2], axis=1)
        n = np.c_[(rho - 2 / 3)[:, None] * raw[:, :2] / rho[:, None], raw[:, 2]]
    elif case == "s_curve":
        n = np.c_[
            -0.15 * np.pi * np.cos(np.pi * raw[:, 0]),
            np.zeros(len(raw)),
            np.ones(len(raw)),
        ]
    elif case == "swiss_roll":
        t = np.mod(np.arctan2(raw[:, 2], raw[:, 0]), 2 * np.pi)
        r = 2 + t / np.pi
        b = 1 / np.pi
        n = np.c_[
            -(b * np.sin(t) + r * np.cos(t)),
            np.zeros(len(raw)),
            b * np.cos(t) - r * np.sin(t),
        ]
    else:
        n = np.c_[-0.2 * raw[:, :3], np.ones(len(raw))]
    n /= np.linalg.norm(n, axis=1)[:, None]
    return n @ Q.T, Q


def make_query_bank(case, count, seed, distances, *, ambient_dimension=None):
    meta = g.geometry_metadata(case)
    D = meta["native_dimension"] if ambient_dimension is None else ambient_dimension
    Q = g._embedding(case, D)
    if (
        isinstance(count, (bool, np.bool_))
        or not isinstance(count, (int, np.integer))
        or not 1 <= count <= 100
    ):
        raise ValueError("query count must be in [1,100]")
    if (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
        or seed < 0
    ):
        raise ValueError("nonnegative seed required")
    distances = list(map(float, distances))
    if (
        not distances
        or len(set(distances)) != len(distances)
        or not np.isfinite(distances).all()
        or min(distances) < 0
    ):
        raise ValueError("distinct finite nonnegative distances required")
    reach = meta["reach_lower_bound"]
    if reach is not None and max(distances) >= reach:
        raise ValueError("distance must be strictly below known reach")
    rng = np.random.default_rng(
        np.random.SeedSequence([67, 821, int(seed), g.CASES.index(case), 0])
    )
    points = []
    left = count
    rejected = 0
    for _ in range(1024):
        raw = g._sample_raw(case, left, rng)
        base = (raw - np.asarray(meta["raw_center"])) * meta["scale"] @ Q.T
        audit = g.project_with_audit(case, base)
        good = (
            audit["boundary_distance"] > g.QUERY_BOUNDARY_MARGIN
            if meta["has_boundary"]
            else np.ones(left, bool)
        )
        rejected += int(np.sum(~good))
        points.append(base[good])
        left -= int(np.sum(good))
        if not left:
            break
    else:
        raise RuntimeError("interior base generation incomplete")
    base = np.concatenate(points)
    normal, Q = _normal(case, base)
    direction_rng = np.random.default_rng(
        np.random.SeedSequence([67, 821, int(seed), g.CASES.index(case), 1, int(D)])
    )
    if D == meta["native_dimension"]:
        sign = np.r_[np.ones(count // 2), -np.ones(count - count // 2)]
        direction_rng.shuffle(sign)
        direction = normal * sign[:, None]
    else:
        ambient = direction_rng.standard_normal((count, D))
        extra = ambient - (ambient @ Q) @ Q.T
        coefficient = np.einsum("mi,mi->m", ambient, normal)
        direction = extra + normal * coefficient[:, None]
        direction /= np.linalg.norm(direction, axis=1)[:, None]
        sign = np.sign(coefficient)
    z = {}
    audits = {}
    discrepancy = []
    for a in distances:
        z[a] = base + a * direction
        audits[a] = g.project_with_audit(case, z[a])
        error = np.linalg.norm(audits[a]["points"] - base, axis=1)
        discrepancy.append(float(error.max()))
        if error.max() > 3e-7:
            raise ValueError(
                "normal-query projection differs from base; retain failed batch"
            )
        if (
            meta["has_boundary"]
            and np.min(audits[a]["boundary_distance"]) <= g.QUERY_BOUNDARY_MARGIN
        ):
            raise ValueError(
                "normal-query projection violates interior generation constraint"
            )
    return dict(
        case=case,
        d=meta["d"],
        D=int(D),
        base=base,
        directions=direction,
        points=z,
        audits=audits,
        distances=distances,
        direction_sign=sign,
        has_boundary=meta["has_boundary"],
        base_boundary_distance=g.project_with_audit(case, base)["boundary_distance"],
        base_rejections=rejected,
        projection_target_discrepancy=discrepancy,
        generator_version=VERSION,
    )
