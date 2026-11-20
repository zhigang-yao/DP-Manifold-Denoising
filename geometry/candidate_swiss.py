"""Approved diameter-two, surface-area sampled Archimedean Swiss roll.

The native surface is a generalized cylinder, so its intrinsic metric is
``ds**2 + dh**2``, with s the planar spiral's arc length. Public interior
queries exclude a strict 0.1 strip along all four intrinsic boundaries.
The numerical reach indicator is descriptive, not a reach certificate.
"""
import operator
import numpy as np

RADIUS_START = .21
RADIAL_PITCH = .48
RADIAL_SLOPE = RADIAL_PITCH / (2*np.pi)
THETA_MAX = 3*np.pi
HALF_HEIGHT = .5813653147689595
SURFACE_AREA_CENTER = (-.032093196915720004, .16421521119771754, 0.)
QUERY_BOUNDARY_MARGIN = .1


def _arc_primitive(theta):
    r = RADIUS_START + RADIAL_SLOPE*theta
    b = RADIAL_SLOPE
    return (r*np.hypot(r, b) + b*b*np.arcsinh(r/b))/(2*b)


def _inverse_arc(value):
    lo, hi = 0., THETA_MAX
    for _ in range(56):
        mid = (lo + hi)/2
        if _arc_primitive(mid) < value:
            lo = mid
        else:
            hi = mid
    return (lo + hi)/2


INTERIOR_THETA_BOUNDS = (
    _inverse_arc(_arc_primitive(0.) + QUERY_BOUNDARY_MARGIN),
    _inverse_arc(_arc_primitive(THETA_MAX) - QUERY_BOUNDARY_MARGIN),
)


def metadata():
    """Return JSON-safe parameters without overstating geometric guarantees."""
    return {
        'geometry': 'swiss_roll_candidate',
        'version': 1,
        'native_dimension': 3,
        'intrinsic_dimension': 2,
        'radius_start': RADIUS_START,
        'radial_pitch_per_turn': RADIAL_PITCH,
        'radial_slope': float(RADIAL_SLOPE),
        'theta_bounds': [0., float(THETA_MAX)],
        'half_height': HALF_HEIGHT,
        'surface_area_center': list(SURFACE_AREA_CENTER),
        'coordinate_equations': '(r*cos(theta), r*sin(theta), h) - surface_area_center; r = radius_start + radial_slope*theta',
        'sampling_measure': 'surface_area',
        'angular_density_proportional_to': 'sqrt(r**2 + radial_slope**2)',
        'query_boundary_margin': QUERY_BOUNDARY_MARGIN,
        'interior_theta_bounds': [float(x) for x in INTERIOR_THETA_BOUNDS],
        'interior_half_height': HALF_HEIGHT - QUERY_BOUNDARY_MARGIN,
        'boundary_distance_metric': 'intrinsic; minimum of arc distances to angular endpoints and height endpoints',
        'diameter': 2.,
        'diameter_status': 'numerically normalized design target',
        'numerical_reach_indicator': .2001,
        'reach_is_certified': False,
    }


def sample_native(count, rng, *, interior=False):
    """Sample area-uniform native points and orthonormal (theta, h) tangents.

    Rejection against the increasing angular Jacobian avoids discretized CDF
    bias. Interior samples are uniform for area restricted to the interior.
    Only the supplied NumPy random generator is consumed.
    """
    count = operator.index(count)
    if count < 0:
        raise ValueError('count must be nonnegative')
    lo, hi = INTERIOR_THETA_BOUNDS if interior else (0., THETA_MAX)
    half_height = HALF_HEIGHT - (QUERY_BOUNDARY_MARGIN if interior else 0.)
    theta = np.empty(count)
    filled = 0
    maximum_weight = np.hypot(RADIUS_START + RADIAL_SLOPE*hi, RADIAL_SLOPE)
    while filled < count:
        batch = max(64, 2*(count-filled))
        proposal = rng.uniform(lo, hi, batch)
        weight = np.hypot(RADIUS_START + RADIAL_SLOPE*proposal, RADIAL_SLOPE)
        keep = rng.random(batch)*maximum_weight < weight
        if interior:
            keep &= (proposal > lo) & (proposal < hi)
        selected = proposal[keep][:count-filled]
        theta[filled:filled+len(selected)] = selected
        filled += len(selected)
    height = rng.uniform(-half_height, half_height, count)
    if interior:
        # Exclude finite-RNG endpoint atoms as well as the continuum boundary.
        endpoints = abs(height) >= half_height
        while np.any(endpoints):
            height[endpoints] = rng.uniform(-half_height, half_height, int(endpoints.sum()))
            endpoints = abs(height) >= half_height
    radius = RADIUS_START + RADIAL_SLOPE*theta
    cosine, sine = np.cos(theta), np.sin(theta)
    points = np.column_stack([radius*cosine, radius*sine, height])
    points -= SURFACE_AREA_CENTER
    tangent_frames = np.zeros((count, 2, 3))
    tangent_frames[:, 0, 0] = RADIAL_SLOPE*cosine - radius*sine
    tangent_frames[:, 0, 1] = RADIAL_SLOPE*sine + radius*cosine
    tangent_frames[:, 0] /= np.hypot(radius, RADIAL_SLOPE)[:, None]
    tangent_frames[:, 1, 2] = 1.
    return points, tangent_frames
