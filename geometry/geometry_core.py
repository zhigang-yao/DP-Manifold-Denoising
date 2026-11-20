"""Canonical geometry bank, manifold-volume sampling, and exact targets."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]


class UnsupportedProjectionError(NotImplementedError):
    """Raised when a maintained geometry has no canonical exact projection."""


@dataclass(frozen=True)
class GeometrySpec:
    """Immutable metadata for one maintained diameter-two geometry."""

    name: str
    family: str
    intrinsic_dimension: int
    native_dimension: int
    diameter_target: float
    reach_lower_bound: float
    reach_certificate: str
    sampling_measure: str
    parameters: tuple[tuple[str, float], ...]
    tau0_primary: float = 0.25
    generator_version: str = "primary_regular_v1"

    @property
    def regularity_pass(self) -> bool:
        return self.reach_lower_bound >= self.tau0_primary

    def parameter(self, name: str) -> float:
        """Return one named, frozen geometric parameter."""

        for parameter_name, value in self.parameters:
            if parameter_name == name:
                return value
        raise KeyError(f"geometry {self.name!r} has no parameter {name!r}")


_SQRT_HALF = 1.0 / np.sqrt(2.0)
_PRIMARY_GEOMETRIES: Final[tuple[GeometrySpec, ...]] = (
    GeometrySpec(
        "circle",
        "circle",
        1,
        2,
        2.0,
        1.0,
        "exact circle reach",
        "manifold_volume",
        (("radius", 1.0),),
    ),
    GeometrySpec(
        "ellipse065",
        "ellipse",
        1,
        2,
        2.0,
        0.65**2,
        "exact ellipse min curvature radius",
        "manifold_volume",
        (("axis_a", 1.0), ("axis_b", 0.65)),
    ),
    GeometrySpec(
        "sphere2",
        "sphere",
        2,
        3,
        2.0,
        1.0,
        "exact sphere reach",
        "manifold_volume",
        (("radius", 1.0),),
    ),
    GeometrySpec(
        "torus065035",
        "ring_torus",
        2,
        3,
        2.0,
        0.30,
        "analytic ring-torus reach",
        "manifold_volume",
        (("major_radius", 0.65), ("minor_radius", 0.35)),
    ),
    GeometrySpec(
        "ellipsoid08065",
        "ellipsoid",
        2,
        3,
        2.0,
        0.65**2,
        "convex principal-radius lower bound",
        "manifold_volume",
        (("axis_a", 1.0), ("axis_b", 0.8), ("axis_c", 0.65)),
    ),
    GeometrySpec(
        "sphere3",
        "sphere",
        3,
        4,
        2.0,
        1.0,
        "exact sphere reach",
        "manifold_volume",
        (("radius", 1.0),),
    ),
    GeometrySpec(
        "s1xs2_equal",
        "sphere_product",
        3,
        5,
        2.0,
        _SQRT_HALF,
        "orthogonal-product reach lower bound",
        "manifold_volume",
        (("circle_radius", _SQRT_HALF), ("sphere_radius", _SQRT_HALF)),
    ),
    GeometrySpec(
        "sphere5",
        "sphere",
        5,
        6,
        2.0,
        1.0,
        "exact sphere reach",
        "manifold_volume",
        (("radius", 1.0),),
    ),
)
_GEOMETRY_BY_NAME: Final = MappingProxyType(
    {spec.name: spec for spec in _PRIMARY_GEOMETRIES}
)


def primary_geometry_bank() -> tuple[GeometrySpec, ...]:
    """Return the frozen maintained primary geometry bank."""

    return _PRIMARY_GEOMETRIES


def geometry_spec(name: str) -> GeometrySpec:
    """Resolve one maintained geometry by its canonical name."""

    try:
        return _GEOMETRY_BY_NAME[name]
    except KeyError as error:
        raise KeyError(f"unknown primary geometry {name!r}") from error


def _validate_count(count: int) -> int:
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError("count must be a positive integer")
    return int(count)


def _unit_sphere_samples(count: int, dimension: int, rng: np.random.Generator) -> Array:
    samples = np.asarray(rng.standard_normal((count, dimension)), dtype=float)
    norms = np.linalg.norm(samples, axis=1)
    while np.any(norms == 0.0):
        zero = norms == 0.0
        samples[zero] = rng.standard_normal((int(np.sum(zero)), dimension))
        norms[zero] = np.linalg.norm(samples[zero], axis=1)
    return samples / norms[:, None]


def _rejection_angles(
    count: int,
    rng: np.random.Generator,
    acceptance_probability,
    oversampling: float,
) -> Array:
    accepted: list[Array] = []
    remaining = count
    while remaining:
        batch = max(64, int(np.ceil(oversampling * remaining)))
        candidates = 2.0 * np.pi * rng.random(batch)
        probabilities = np.asarray(acceptance_probability(candidates), dtype=float)
        values = candidates[rng.random(batch) <= probabilities]
        take = min(remaining, values.size)
        if take:
            accepted.append(values[:take])
            remaining -= take
    return np.concatenate(accepted)


def sample_geometry(spec: GeometrySpec, count: int, rng: np.random.Generator) -> Array:
    """Draw row-oriented samples from the manifold-volume measure."""

    count = _validate_count(count)
    if not isinstance(spec, GeometrySpec):
        raise TypeError("spec must be a GeometrySpec")
    if not spec.regularity_pass:
        raise ValueError(f"geometry {spec.name!r} fails the primary reach floor")

    if spec.family == "circle":
        radius = spec.parameter("radius")
        angle = 2.0 * np.pi * rng.random(count)
        samples = radius * np.column_stack((np.cos(angle), np.sin(angle)))
    elif spec.family == "ellipse":
        axis_a = spec.parameter("axis_a")
        axis_b = spec.parameter("axis_b")
        angle = _rejection_angles(
            count,
            rng,
            lambda t: (
                np.sqrt((axis_a * np.sin(t)) ** 2 + (axis_b * np.cos(t)) ** 2) / axis_a
            ),
            1.8,
        )
        samples = np.column_stack((axis_a * np.cos(angle), axis_b * np.sin(angle)))
    elif spec.family == "sphere":
        samples = spec.parameter("radius") * _unit_sphere_samples(
            count, spec.native_dimension, rng
        )
    elif spec.family == "ring_torus":
        major = spec.parameter("major_radius")
        minor = spec.parameter("minor_radius")
        u = 2.0 * np.pi * rng.random(count)
        v = _rejection_angles(
            count,
            rng,
            lambda angle: (major + minor * np.cos(angle)) / (major + minor),
            1.6,
        )
        radial = major + minor * np.cos(v)
        samples = np.column_stack(
            (radial * np.cos(u), radial * np.sin(u), minor * np.sin(v))
        )
    elif spec.family == "ellipsoid":
        axes = np.array(
            [
                spec.parameter("axis_a"),
                spec.parameter("axis_b"),
                spec.parameter("axis_c"),
            ]
        )
        accepted: list[Array] = []
        remaining = count
        max_weight = 1.0 / np.min(axes)
        while remaining:
            batch = max(64, int(np.ceil(1.8 * remaining)))
            directions = _unit_sphere_samples(batch, 3, rng)
            weights = np.linalg.norm(directions / axes, axis=1)
            keep = directions[rng.random(batch) <= weights / max_weight]
            take = min(remaining, len(keep))
            if take:
                accepted.append(keep[:take])
                remaining -= take
        samples = np.concatenate(accepted, axis=0) * axes
    elif spec.family == "sphere_product":
        circle_radius = spec.parameter("circle_radius")
        sphere_radius = spec.parameter("sphere_radius")
        angle = 2.0 * np.pi * rng.random(count)
        circle = circle_radius * np.column_stack((np.cos(angle), np.sin(angle)))
        sphere = sphere_radius * _unit_sphere_samples(count, 3, rng)
        samples = np.column_stack((circle, sphere))
    else:  # pragma: no cover - bank construction makes this unreachable
        raise ValueError(f"unsupported geometry family {spec.family!r}")

    samples = np.asarray(samples, dtype=float)
    if samples.shape != (count, spec.native_dimension) or not np.all(
        np.isfinite(samples)
    ):
        raise RuntimeError("geometry sampler returned invalid samples")
    return samples


def isometric_embedding(
    native_dimension: int,
    ambient_dimension: int,
    rng_or_seed: np.random.Generator | np.random.SeedSequence | int,
) -> Array:
    """Create a deterministic-sign linear isometry ``Q`` of shape ``(D,D0)``."""

    if native_dimension < 1 or ambient_dimension < native_dimension:
        raise ValueError("require ambient_dimension >= native_dimension >= 1")
    rng = (
        rng_or_seed
        if isinstance(rng_or_seed, np.random.Generator)
        else np.random.Generator(np.random.PCG64(rng_or_seed))
    )
    gaussian = rng.standard_normal((ambient_dimension, native_dimension))
    q, r = np.linalg.qr(gaussian, mode="reduced")
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return np.asarray(q * signs, dtype=float)


def _project_radially(points: Array, radius: float) -> Array:
    norms = np.linalg.norm(points, axis=1)
    projected = np.zeros_like(points)
    nonzero = norms > 100.0 * np.finfo(float).eps
    projected[nonzero] = radius * points[nonzero] / norms[nonzero, None]
    projected[~nonzero, 0] = radius
    return projected


def _project_ring_torus(points: Array, major: float, minor: float) -> Array:
    rho = np.linalg.norm(points[:, :2], axis=1)
    radial_direction = np.zeros((len(points), 2), dtype=float)
    nonzero_rho = rho > 100.0 * np.finfo(float).eps
    radial_direction[nonzero_rho] = points[nonzero_rho, :2] / rho[nonzero_rho, None]
    radial_direction[~nonzero_rho, 0] = 1.0

    meridian = np.column_stack((rho - major, points[:, 2]))
    meridian_norm = np.linalg.norm(meridian, axis=1)
    meridian_direction = np.zeros_like(meridian)
    nonzero_meridian = meridian_norm > 100.0 * np.finfo(float).eps
    meridian_direction[nonzero_meridian] = (
        meridian[nonzero_meridian] / meridian_norm[nonzero_meridian, None]
    )
    meridian_direction[~nonzero_meridian, 0] = 1.0
    rho_projected = major + minor * meridian_direction[:, 0]
    return np.column_stack(
        (radial_direction * rho_projected[:, None], minor * meridian_direction[:, 1])
    )


def project_to_geometry(
    spec: GeometrySpec, ambient_query: ArrayLike, embedding: ArrayLike
) -> Array:
    """Return exact synthetic nearest-point targets for supported families."""

    query = np.asarray(ambient_query, dtype=float)
    q = np.asarray(embedding, dtype=float)
    if query.ndim != 2 or not np.all(np.isfinite(query)):
        raise ValueError("ambient_query must be a finite (M,D) array")
    if q.shape != (query.shape[1], spec.native_dimension) or not np.all(np.isfinite(q)):
        raise ValueError("embedding must have shape (D,native_dimension)")
    gram_error = np.linalg.norm(q.T @ q - np.eye(spec.native_dimension), ord="fro")
    if gram_error > 1e-8 * max(1, spec.native_dimension):
        raise ValueError("embedding columns must be orthonormal")

    native = query @ q
    if spec.family in {"circle", "sphere"}:
        projected_native = _project_radially(native, spec.parameter("radius"))
    elif spec.family == "ring_torus":
        projected_native = _project_ring_torus(
            native,
            spec.parameter("major_radius"),
            spec.parameter("minor_radius"),
        )
    elif spec.family == "sphere_product":
        projected_native = np.column_stack(
            (
                _project_radially(native[:, :2], spec.parameter("circle_radius")),
                _project_radially(native[:, 2:5], spec.parameter("sphere_radius")),
            )
        )
    else:
        raise UnsupportedProjectionError(
            f"exact projection is not implemented for family {spec.family!r}"
        )
    return projected_native @ q.T
