"""One joint Gaussian history and its terminal conditional density extension.

Cache schema retains the e1df31d covariance source fingerprint. A scientific
change to covariance arithmetic requires a new schema; this migration does not.
"""

import copy, hashlib, json, os, time
from pathlib import Path
import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.linalg.blas import dsyrk
from modules.md.moments import empirical
from modules.md._kernel import _operators, MixtureKernel

COVARIANCE_SOURCE_DIGEST = bytes.fromhex(
    "b382990f36354bca4b4aa3a4e3b305948473cd5f1511dbb538124166765e52d0"
)


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''):
            digest.update(block)
    return digest.hexdigest()


class FullField:
    """Private simulation server. Decoder never gets the reference array."""

    def __init__(
        self, reference, hs, hmin, sd, rng, order=128, jitter=1e-9, cache_dir=None,
        kernel=None,
    ):
        self.reference = np.asarray(reference)
        self.hs = np.asarray(hs)
        self.sd = sd
        self.rng = rng
        self.kernel = kernel if kernel is not None else MixtureKernel(
            self.reference.shape[1], hs, order, spectrum_range=(hmin, 0.4)
        )
        self.jitter = jitter
        self.cache_dir = (
            Path(cache_dir)
            if cache_dir is not None
            else Path(
                os.environ.get(
                    "PAPER_GAUSSIAN_CACHE", Path.cwd() / "results/cache"
                )
            )
        )
        self.spectrum_range = (hmin, 0.4)
        self.blocks = []
        self.noise = np.empty(0)
        self.L = np.empty((0, 0))
        self.cache = {}
        self.cache_indices = {}
        self.max_conditional_jitter = 0.0

    @staticmethod
    def key(h, z):
        z = np.asarray(z, dtype=np.float64).copy()
        z[z == 0.0] = 0.0
        return (float(h), z.tobytes())

    def _cov(self, left, right):
        x, h, kind = left
        y, k, k2 = right
        return self.kernel.covariance(x, y, h, k, kind, k2)

    def initial(self, x):
        initial_started = time.perf_counter()
        self.initial_timings = dict(covariance_seconds=0.0, cholesky_seconds=0.0,
                                    cache_hit=False)
        if self.blocks:
            raise RuntimeError("field history cannot be reset")
        x = np.asarray(x, dtype=np.float64).copy()
        x[x == 0.0] = 0.0
        if len({z.tobytes() for z in x}) != len(x):
            raise ValueError("duplicate original query positions must be deduplicated")
        blocks = [(x.copy(), float(h), "moments") for h in self.hs]
        sizes = [len(x) * len(_operators(x.shape[1], h, "moments")) for h in self.hs]
        edges = np.r_[0, np.cumsum(sizes)]
        design = hashlib.sha256(
            np.array(x.shape, dtype=np.int64).tobytes()
            + x.dtype.str.encode()
            + x.tobytes()
            + self.hs.tobytes()
            + np.array([self.kernel.order, self.jitter, *self.spectrum_range]).tobytes()
            + COVARIANCE_SOURCE_DIGEST
            + b"native-mixture-cache-v2"
            + getattr(self.kernel, 'design_digest', b'')
        ).hexdigest()
        cache = self.cache_dir
        cache.mkdir(parents=True, exist_ok=True)
        path = cache / (design + ".npy")
        if path.exists():
            if hasattr(self.kernel, 'design_digest'):
                checksum = path.with_suffix('.sha256')
                if not checksum.exists() or _file_sha256(path) != checksum.read_text().strip():
                    raise ValueError('finite-field factor cache hash missing or invalid')
            self.initial_timings["cache_hit"] = True
            self.L = np.load(path, mmap_mode="r")
        else:
            # LAPACK consumes a Fortran-order matrix in place. Symmetrize only
            # the small diagonal blocks, not the full multi-GB joint matrix.
            K = np.empty((edges[-1], edges[-1]), order="F")
            for i, left in enumerate(blocks):
                for j in range(i + 1):
                    b = self._cov(left, blocks[j])
                    if not np.isfinite(b).all():
                        raise ValueError("nonfinite covariance block")
                    if i == j:
                        b = (b + b.T) / 2
                    K[edges[i] : edges[i + 1], edges[j] : edges[j + 1]] = b
                    K[edges[j] : edges[j + 1], edges[i] : edges[i + 1]] = b.T
            K[np.diag_indices(len(K))] += self.jitter * np.maximum(np.diag(K), 1.0)
            covariance_done = time.perf_counter()
            self.initial_timings["covariance_seconds"] = covariance_done - initial_started
            self.L = cholesky(K, lower=True, overwrite_a=True, check_finite=False)
            self.initial_timings["cholesky_seconds"] = time.perf_counter() - covariance_done
            del K, b
            temporary = cache / (design + f".{os.getpid()}.tmp.npy")
            np.save(temporary, self.L)
            os.replace(temporary, path)
            if hasattr(self.kernel, 'design_digest'):
                checksum = path.with_suffix('.sha256')
                tmp_checksum = checksum.with_name(checksum.name + f'.{os.getpid()}.tmp')
                tmp_checksum.write_text(_file_sha256(path))
                os.replace(tmp_checksum, checksum)
        self.local_covariances = []
        for h in self.hs:
            local = self.kernel.covariance(x[:1], x[:1], h, h)
            local += np.diag(self.jitter * np.maximum(np.diag(local), 1.0))
            self.local_covariances.append(local)
        sampling_started = time.perf_counter()
        self.noise = self.L @ self.rng.normal(size=len(self.L))
        self.initial_timings["sampling_seconds"] = time.perf_counter() - sampling_started
        self.blocks = blocks
        result = []
        for i, (xx, h, kind) in enumerate(blocks):
            val = empirical(self.reference, xx, h, kind) + self.sd * self.noise[
                edges[i] : edges[i + 1]
            ].reshape(len(xx), -1)
            result.append(val)
            # Density/gradient at exactly existing points is the original jet readout.
            for qi, (z, row) in enumerate(zip(xx, val)):
                key = self.key(h, z)
                self.cache[key] = np.r_[row[0], row[1 : 1 + x.shape[1]] / h]
                self.cache_indices[key] = int(edges[i] + qi * val.shape[1])
        self.initial_timings["total_seconds"] = time.perf_counter() - initial_started
        return np.stack(result, axis=1)

    def read(self, x, h):
        if getattr(self, '_terminal_sealed', False):
            raise RuntimeError('terminal transcript sealed; further reads require full terminal history restoration')
        if not self.blocks:
            raise RuntimeError("initialize the field before adaptive reads")
        x = np.asarray(x, dtype=np.float64).copy()
        x[x == 0.0] = 0.0
        keys = [self.key(h, z) for z in x]
        seen = set(self.cache)
        missing = []
        for i, key in enumerate(keys):
            if key not in seen:
                missing.append(i)
                seen.add(key)
        if missing:
            xx = x[missing]
            block = (xx, float(h), "density_gradient")
            # Construct the transposed cross covariance directly. A list plus
            # concatenate retained two copies, and C-order L made triangular
            # solves copy the whole history. Neither copy is needed.
            cross = np.empty((len(self.L), len(xx) * (x.shape[1] + 1)), order="F")
            offset = 0
            for previous in self.blocks:
                b = self._cov(block, previous)
                if not np.isfinite(b).all():
                    raise ValueError("nonfinite covariance block")
                cross[offset : offset + b.shape[1]] = b.T
                offset += b.shape[1]
            del b
            K = self._cov(block, block)
            K += np.diag(self.jitter * np.maximum(np.diag(K), 1.0))
            lower = solve_triangular(self.L, cross, lower=True, overwrite_b=True)
            del cross
            cond = K - lower.T @ lower
            cond = (cond + cond.T) / 2
            eigmin = float(np.linalg.eigvalsh(cond)[0])
            extra = max(0.0, -eigmin + self.jitter)
            if extra > 1e-7 * max(1.0, float(np.diag(K).max())):
                raise ArithmeticError(
                    f"conditional covariance needs excessive correction: {extra}"
                )
            self.max_conditional_jitter = max(self.max_conditional_jitter, extra)
            root = cholesky(cond + np.eye(len(cond)) * extra, lower=True)
            latent = lower.T @ solve_triangular(
                self.L, self.noise, lower=True
            ) + root @ self.rng.normal(size=len(cond))
            old = len(self.L)
            new = np.zeros((old + len(root), old + len(root)), order="F")
            new[:old, :old] = self.L
            new[old:, :old] = lower.T
            new[old:, old:] = root
            self.L = new
            self.noise = np.r_[self.noise, latent]
            self.blocks.append(block)
            val = empirical(
                self.reference, xx, h, "density_gradient"
            ) + self.sd * latent.reshape(len(xx), -1)
            for mi, (k, row) in enumerate(zip([keys[i] for i in missing], val)):
                self.cache[k] = row
                self.cache_indices[k] = old + mi * (x.shape[1] + 1)
        return np.array([self.cache[k] for k in keys])


def cross_density_readout(
    field,
    outputs,
    *,
    empirical,
    draw_identity,
    block_size=100,
    memory_limit_bytes=7 * 2**30,
    reserve_bytes=768 * 2**20,
    event=None,
    allow_candidate_count=False,
    seal_history=False,
):
    """Return f_b(Y_h) with axes (query, candidate h, density bandwidth b).

    ``empirical`` must be the frozen scientific density-mean function.
    ``field`` must retain the original FullField L/noise/blocks/cache/RNG state
    after ALL original adaptive reads. Readouts already in that cache are
    reused exactly. Missing keys use b-major, h-major, query-major order.

    Return scalar history separately; do not insert scalar entries into the
    original cache, whose values have density+gradient coordinates. A further
    extension must condition on this returned history as well; calling this
    function a second time on the original field is a distinct conditional
    draw, not a repeated read of the first extension.
    """
    started = time.perf_counter()
    if getattr(field, '_terminal_sealed', False):
        raise RuntimeError('cannot extend a sealed terminal transcript')
    if not isinstance(draw_identity, str) or not draw_identity.strip():
        raise ValueError(
            "an explicit new conditional extension draw identity is required"
        )
    if not isinstance(block_size, int) or block_size < 1 or block_size > 100:
        raise ValueError("block_size must be an integer in 1..100")
    outputs = np.asarray(outputs, dtype=np.float64)
    hs = np.asarray(field.hs, dtype=np.float64)
    if outputs.ndim != 3 or (not allow_candidate_count and outputs.shape[1] != len(hs)) or outputs.shape[0] < 1 or outputs.shape[1] < 1:
        raise ValueError("outputs must have shape (positive Q, len(field.hs), D)")
    if not np.isfinite(outputs).all() or not np.isfinite(hs).all() or np.any(hs <= 0):
        raise ValueError(
            "finite candidate coordinates and positive bandwidths required"
        )
    if len(np.unique(hs)) != len(hs):
        raise ValueError("density bandwidths must be distinct")
    N = len(field.L)
    if (
        not N
        or field.L.shape != (N, N)
        or np.asarray(field.noise).shape != (N,)
        or not field.blocks
    ):
        raise ValueError("complete initialized old field history required")
    if (
        not np.isfinite(field.sd)
        or field.sd <= 0
        or not np.isfinite(field.jitter)
        or field.jitter <= 0
    ):
        raise ValueError("positive finite frozen field scale and jitter required")
    if (
        not isinstance(memory_limit_bytes, int)
        or memory_limit_bytes <= 0
        or reserve_bytes < 0
    ):
        raise ValueError("positive memory limit and nonnegative reserve required")

    def emit(stage, **extra):
        if event is not None:
            event(stage, elapsed_seconds=time.perf_counter() - started, **extra)

    state_before = copy.deepcopy(field.rng.bit_generator.state)
    state_before_text = json.dumps(
        state_before,
        sort_keys=True,
        separators=(",", ":"),
        default=lambda x: x.tolist() if isinstance(x, np.ndarray) else int(x),
    )
    Q, H, D = outputs.shape
    values = np.empty((Q, H, len(hs)), dtype=np.float64)
    grid_to_new = np.full((Q, H, len(hs)), -1, dtype=np.int64)
    keys_to_new = {}
    cached_keys = set()
    coordinates = []
    bandwidths = []
    groups = []
    for bi, b in enumerate(hs):
        for hj in range(H):
            begin = len(coordinates)
            for qi in range(Q):
                z = outputs[qi, hj]
                key = field.key(float(b), z)
                if key in field.cache:
                    cached = np.asarray(field.cache[key])
                    if (
                        cached.ndim != 1
                        or cached.size < 1
                        or not np.isfinite(cached[0])
                    ):
                        raise ValueError("nonfinite or malformed old cached density")
                    values[qi, hj, bi] = cached[0]
                    cached_keys.add(key)
                else:
                    if key not in keys_to_new:
                        keys_to_new[key] = len(coordinates)
                        canonical = z.copy()
                        canonical[canonical == 0.0] = 0.0
                        coordinates.append(canonical)
                        bandwidths.append(float(b))
                    grid_to_new[qi, hj, bi] = keys_to_new[key]
            end = len(coordinates)
            for lo in range(begin, end, block_size):
                groups.append((lo, min(lo + block_size, end), float(b)))
    U = len(coordinates)
    new_locations = np.asarray(coordinates, dtype=np.float64).reshape(U, D)
    new_bandwidths = np.asarray(bandwidths, dtype=np.float64)
    # Group ONLY identical public coordinate arrays. Matrix indices and the
    # b-major functional/RNG ordering stay unchanged. The frozen radial kernel
    # retains one radius set and already computes all spectral pairs together.
    # Keep all bandwidth calls on that same coordinate pair adjacent.
    position_groups = {}
    for lo, hi, b in groups:
        coords = new_locations[lo:hi]
        identity = (coords.shape, hashlib.sha256(coords.tobytes()).digest())
        position_groups.setdefault(identity, []).append((lo, hi, b))
    new_position_groups = list(position_groups.values())
    old_position_groups = {}
    offset = 0
    for old_x, h, kind in field.blocks:
        old_x = np.asarray(old_x)
        width = len(old_x) * (
            1 + D + D * (D + 1) // 2
            if kind == "moments"
            else D + 1
            if kind == "density_gradient"
            else 1
        )
        identity = (old_x.shape, hashlib.sha256(old_x.tobytes()).digest())
        old_position_groups.setdefault(identity, []).append(
            (old_x, h, kind, offset, width)
        )
        offset += width
    # A moments block needs radial derivatives through order two. Evaluate it
    # before density+gradient (order one) at identical positions, so the latter
    # reuses the already computed higher-order radial arrays.
    derivative_priority = {"moments": 0, "density_gradient": 1, "density": 2}
    old_position_groups = [
        sorted(entries, key=lambda e: derivative_priority.get(e[2], 3))
        for entries in old_position_groups.values()
    ]
    # Include old L, a possible solve copy of V, three U^2 work matrices
    # (Schur/eigensolver/Cholesky), bounded covariance buffers, and runtime
    # reserve. This is a conservative live-array estimate, not a measured RSS.
    old_arrays = (
        np.asarray(field.L).nbytes
        + np.asarray(field.noise).nbytes
        + np.asarray(field.reference).nbytes
    )
    max_old_width = max(
        len(x)
        * (
            1 + D + D * (D + 1) // 2
            if kind == "moments"
            else D + 1
            if kind == "density_gradient"
            else 1
        )
        for x, _, kind in field.blocks
    )
    expected_old_width = sum(
        len(x)
        * (
            1 + D + D * (D + 1) // 2
            if kind == "moments"
            else D + 1
            if kind == "density_gradient"
            else 1
        )
        for x, _, kind in field.blocks
    )
    if expected_old_width != N or any(
        kind not in ("moments", "density_gradient", "density")
        for _, _, kind in field.blocks
    ):
        raise ValueError("old history block dimensions do not match old factor")
    cross_bytes = 8 * N * U
    schur_bytes = 8 * U * U
    factor_copy_bytes = (
        0 if np.asarray(field.L).flags.f_contiguous else np.asarray(field.L).nbytes
    )
    bounded_buffer_bytes = 8 * block_size * max(max_old_width, block_size) * 4
    planned = (
        old_arrays
        + factor_copy_bytes
        + 2 * cross_bytes
        + 3 * schur_bytes
        + bounded_buffer_bytes
        + reserve_bytes
    )
    metadata = dict(
        draw_identity=draw_identity,
        axes=["query", "candidate_h", "density_bandwidth_b"],
        ordering="density b-major, candidate h-major, query-major; first unique missing key",
        old_dimension=N,
        old_block_count=len(field.blocks),
        old_cache_key_count=len(field.cache),
        new_scalar_count=U,
        total_grid_readouts=Q * H * len(hs),
        cached_unique_count=len(cached_keys),
        distinct_new_position_blocks=len(new_position_groups),
        distinct_old_position_blocks=len(old_position_groups),
        covariance_assembly="identical coordinate pair first, then all bandwidth blocks; scalar order unchanged",
        new_duplicate_grid_readouts=int(np.sum(grid_to_new >= 0)) - U,
        maximum_combined_scalar_dimension=N + U,
        memory=dict(
            old_arrays_bytes=old_arrays,
            whitened_cross_bytes=cross_bytes,
            schur_bytes=schur_bytes,
            bounded_buffer_bytes=bounded_buffer_bytes,
            possible_old_factor_solve_copy_bytes=factor_copy_bytes,
            reserve_bytes=reserve_bytes,
            estimated_peak_bytes=planned,
            limit_bytes=memory_limit_bytes,
            estimate="old arrays + possible old factor copy + 2 cross buffers + 3 Schur buffers + bounded blocks + reserve; not measured RSS",
        ),
        rng_state_before=state_before,
        rng_state_sha256_before=hashlib.sha256(state_before_text.encode()).hexdigest(),
        original_jitter=float(field.jitter),
        conditional_extra_jitter=0.0,
        interpretation="New jointly conditioned density-only extension of reconstructed finite numerical field; not recovery of previously unsampled unique cross values; not certified machine DP",
    )
    if planned > memory_limit_bytes:
        raise MemoryError(
            f"density continuation memory estimate {planned} exceeds limit {memory_limit_bytes}"
        )
    emit(
        "cross_density_planned",
        old_dimension=N,
        new_scalar_count=U,
        estimated_peak_bytes=planned,
    )
    plan_done = time.perf_counter()

    if U:
        # V is filled transposed directly, with no list/concatenate copies.
        V = np.empty((N, U), dtype=np.float64, order="F")
        cross_pair_count = len(new_position_groups) * len(old_position_groups)
        pair_index = 0
        for new_entries in new_position_groups:
            for old_entries in old_position_groups:
                for old_x, h, kind, offset, width in old_entries:
                    for lo, hi, b in new_entries:
                        block = field.kernel.covariance(
                            new_locations[lo:hi], old_x, b, h, "density", kind
                        )
                        if (
                            block.shape != (hi - lo, width)
                            or not np.isfinite(block).all()
                        ):
                            raise ValueError(
                                "nonfinite or malformed old-new density covariance"
                            )
                        V[offset : offset + width, lo:hi] = block.T
                del block
                pair_index += 1
                emit(
                    "cross_density_cross_block",
                    block=pair_index,
                    blocks=cross_pair_count,
                )
        V = solve_triangular(
            field.L, V, lower=True, overwrite_b=True, check_finite=False
        )
        cross_done = time.perf_counter()
        emit("cross_density_whitened", new_scalar_count=U)

        # Fill lower triangle and its transpose in bounded coordinate groups.
        S = np.empty((U, U), dtype=np.float64, order="F")
        covariance_pair_count = (
            len(new_position_groups) * (len(new_position_groups) + 1) // 2
        )
        pair_index = 0
        for i, entries_a in enumerate(new_position_groups):
            for j in range(i + 1):
                entries_b = new_position_groups[j]
                for ai, entry_a in enumerate(entries_a):
                    for bj, entry_b in enumerate(entries_b):
                        if i == j and bj > ai:
                            continue
                        # Preserve exactly the old lower-block orientation;
                        # grouping changes call order only, never a kernel call.
                        left, right = (
                            (entry_a, entry_b)
                            if entry_a[0] >= entry_b[0]
                            else (entry_b, entry_a)
                        )
                        lo, hi, b = left
                        rlo, rhi, k = right
                        block = field.kernel.covariance(
                            new_locations[lo:hi],
                            new_locations[rlo:rhi],
                            b,
                            k,
                            "density",
                            "density",
                        )
                        if (
                            block.shape != (hi - lo, rhi - rlo)
                            or not np.isfinite(block).all()
                        ):
                            raise ValueError(
                                "nonfinite or malformed new-new density covariance"
                            )
                        if lo == rlo:
                            block = (block + block.T) / 2
                        S[lo:hi, rlo:rhi] = block
                        S[rlo:rhi, lo:hi] = block.T
                del block
                pair_index += 1
                emit(
                    "cross_density_covariance_block",
                    block=pair_index,
                    blocks=covariance_pair_count,
                )
        diagonal_indices = np.diag_indices(U)
        original_diagonal = S[diagonal_indices].copy()
        S[diagonal_indices] += field.jitter * np.maximum(original_diagonal, 1.0)
        maximum_K_diagonal = float(np.max(S[diagonal_indices]))
        # dsyrk updates lower(S) <- lower(S - V.T V), with no U^2 Gram temporary.
        S = dsyrk(-1.0, V, beta=1.0, c=S, trans=1, lower=1, overwrite_c=1)
        for lo in range(0, U, block_size):
            hi = min(U, lo + block_size)
            for rlo in range(0, lo, block_size):
                rhi = min(U, rlo + block_size)
                S[rlo:rhi, lo:hi] = S[lo:hi, rlo:rhi].T
            diagonal_block = S[lo:hi, lo:hi]
            for ii in range(hi - lo):
                diagonal_block[ii, ii + 1 :] = diagonal_block[ii + 1 :, ii]
        covariance_done = time.perf_counter()
        emit("cross_density_schur_ready", new_scalar_count=U)
        if not np.isfinite(S).all():
            raise ValueError("nonfinite density conditional covariance")
        eigmin = float(np.linalg.eigvalsh(S)[0])
        extra = max(0.0, -eigmin + field.jitter)
        guard = 1e-7 * max(1.0, maximum_K_diagonal)
        if extra > guard:
            raise ArithmeticError(
                f"conditional covariance needs excessive correction: {extra}"
            )
        metadata.update(
            conditional_min_eigenvalue=eigmin,
            conditional_extra_jitter=extra,
            conditional_correction_guard=guard,
            maximum_new_jittered_variance=maximum_K_diagonal,
        )
        S[diagonal_indices] += extra
        root = cholesky(S, lower=True, overwrite_a=True, check_finite=False)
        del S
        coefficients = solve_triangular(
            field.L, field.noise, lower=True, check_finite=False
        )
        conditional_mean = V.T @ coefficients
        del V, coefficients
        # Compute every empirical mean before advancing RNG; failures do not
        # consume a conditional draw or mutate old history.
        new_means = np.empty(U)
        for lo, hi, b in groups:
            means = np.asarray(
                empirical(field.reference, new_locations[lo:hi], b, "density")
            )
            if means.shape != (hi - lo, 1) or not np.isfinite(means).all():
                raise ValueError("nonfinite or malformed empirical density means")
            new_means[lo:hi] = means[:, 0]
        factor_done = time.perf_counter()
        emit(
            "cross_density_factor_ready",
            new_scalar_count=U,
            conditional_extra_jitter=extra,
        )
        new_noise = conditional_mean + root @ field.rng.normal(size=U)
        del root
        new_values = new_means + field.sd * new_noise
        if not np.isfinite(new_values).all():
            raise ArithmeticError("nonfinite conditional density values")
        mask = grid_to_new >= 0
        values[mask] = new_values[grid_to_new[mask]]
    else:
        new_noise = np.empty(0)
        new_values = np.empty(0)
        cross_done = covariance_done = factor_done = plan_done
    after = copy.deepcopy(field.rng.bit_generator.state)
    after_text = json.dumps(
        after,
        sort_keys=True,
        separators=(",", ":"),
        default=lambda x: x.tolist() if isinstance(x, np.ndarray) else int(x),
    )
    metadata.update(
        rng_state_after=after,
        rng_state_sha256_after=hashlib.sha256(after_text.encode()).hexdigest(),
        timings=dict(
            plan_seconds=plan_done - started,
            cross_solve_seconds=cross_done - plan_done,
            schur_seconds=covariance_done - cross_done,
            factor_and_means_seconds=factor_done - covariance_done,
            total_seconds=time.perf_counter() - started,
        ),
    )
    emit("cross_density_complete", new_scalar_count=U)
    if seal_history:
        field._terminal_sealed = True
    return dict(
        values=values,
        new_values=new_values,
        new_noise=new_noise,
        new_locations=new_locations,
        new_bandwidths=new_bandwidths,
        grid_to_new_index=grid_to_new,
        metadata=metadata,
    )
