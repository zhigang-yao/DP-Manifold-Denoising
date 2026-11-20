"""Reuse one structured field's public factors without reusing a release.

The cached template is constructed solely from public queries and numerical
settings and is never initialized. Every request gets a separate history,
mutable diagnostics, and kernel evaluation scratch space. Dense public factors
are shared read-only; a different key discards the old template before building
its replacement.
"""
from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass
from time import perf_counter

import numpy as np

from modules.md.structured_shared_field import SharedStructuredField


def _freeze_arrays(value):
    """Make arrays reachable through public construction state read-only."""
    if isinstance(value, np.ndarray):
        value.setflags(write=False)
    elif isinstance(value, dict):
        for item in value.values():
            _freeze_arrays(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _freeze_arrays(item)
    elif is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            _freeze_arrays(getattr(value, field.name))


def _copy_containers(value):
    """Copy mutable public containers while retaining immutable factor arrays."""
    if isinstance(value, dict):
        return {key: _copy_containers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_containers(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_containers(item) for item in value)
    return value


def _fresh_kernel(kernel):
    if kernel is None:
        return None
    result = copy.copy(kernel)
    result.__dict__ = _copy_containers(kernel.__dict__)
    # D=2,3,4 radials memoize distances and their evaluated values. Terminal
    # positions can depend on a private release, so these caches must be local.
    result._shared_distances = None
    result._shared_values = []
    return result


def _array_key(array):
    return array.shape, array.dtype.str, array.tobytes(order="C")


def _number_key(value):
    # Preserve the exact numerical representation and distinguish bool from
    # int: otherwise a cache hit could bypass the constructor's order checks.
    return type(value), np.asarray(value).dtype.str, np.asarray(value).tobytes()


class PublicFieldCache:
    """A process-local, single-entry factory for fresh structured releases.

    Inputs are copied and converted to the same float representation used by
    ``SharedStructuredField``. Equality is exact, including signed zero, with
    no rounding or digest collisions. No reference data or random generator
    enters the key or the retained template. Live returned fields can retain
    an evicted factor, but the factory itself retains at most one template.
    """

    def __init__(self):
        self._key = None
        self._template = None

    def clear(self):
        """Release the factory's reference to its public template."""
        self._key = None
        self._template = None

    def new_field(self, queries, bandwidths, *, order=128,
                  memory_limit_gib=6, bandwidth_correlation="shared",
                  jitter=1e-9, span_tol=1e-10):
        """Return an uninitialized field with independent mutable state."""
        started = perf_counter()
        # The underlying constructor retains its hs argument. Own these arrays
        # so read-only protection never changes the caller's array flags.
        x = np.array(queries, dtype=float, order="C", copy=True)
        hs = np.array(bandwidths, dtype=float, order="C", copy=True)
        key = (_array_key(x), _array_key(hs), _number_key(order),
               _number_key(jitter), _number_key(memory_limit_gib),
               _number_key(span_tol), bandwidth_correlation)
        cache_hit = key == self._key
        if not cache_hit:
            # Do not retain the old dense factors during a replacement build.
            # A failed build deliberately leaves this cache empty.
            self.clear()
            template = SharedStructuredField(
                x, hs, order=order, jitter=jitter,
                memory_limit_gib=memory_limit_gib, span_tol=span_tol,
                bandwidth_correlation=bandwidth_correlation)
            _freeze_arrays(template.__dict__)
            for kernel in [template.kernel, *(template.single_kernels or [])]:
                _freeze_arrays(kernel.__dict__)
                kernel._shared_distances = None
                kernel._shared_values = []
            self._template = template
            self._key = key

        template = self._template
        result = copy.copy(template)
        result.__dict__ = _copy_containers(template.__dict__)
        result.plan = copy.deepcopy(template.plan)
        result.kernel = _fresh_kernel(template.kernel)
        result.single_kernels = (None if template.single_kernels is None else
                                [_fresh_kernel(k) for k in template.single_kernels])
        result.plan["public_factor_cache_hit"] = cache_hit
        result.plan["public_factor_construction_seconds"] = template.plan["setup_seconds"]
        result.plan["setup_seconds"] = perf_counter() - started
        return result
