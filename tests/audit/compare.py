"""Compare replayed simulation units with the frozen numerical archive.

RMSE values and available NPZ arrays are checked numerically with explicit
absolute and relative tolerances. Missing archived NPZ arrays are counted and
reported; use --require-arrays to fail when an array cannot be checked.

usage: python -m tests.audit.compare <replay root> [stored root]
       [--rtol 1e-9] [--atol 1e-12] [--require-arrays]

The default stored root is $RUNS, falling back to data/runs.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def numeric_match(a, b, *, rtol, atol):
    """Floating arrays use tolerance; integer/bool and nonnumeric arrays are exact."""
    if a.shape != b.shape:
        return False
    if np.issubdtype(a.dtype, np.inexact) and np.issubdtype(b.dtype, np.inexact):
        return bool(np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=False))
    return bool(np.array_equal(a, b))


def compare(replay, stored, *, rtol=1e-9, atol=1e-12, require_arrays=False):
    if not replay.is_dir():
        raise ValueError(f"replay directory does not exist: {replay}")
    if not stored.is_dir():
        raise ValueError(f"stored directory does not exist: {stored}")

    json_files = sorted(replay.rglob("*.json"))
    orphan_npz = sorted(p for p in replay.rglob("*.npz") if not p.with_suffix(".json").exists())
    if not json_files and not orphan_npz:
        raise ValueError(f"no replay units found in {replay}")

    failed = 0
    rmse_checked = 0
    arrays_checked = 0
    arrays_unavailable = 0
    for f in [*json_files, *orphan_npz]:
        rel = f.relative_to(replay)
        o = stored / rel
        if not o.is_file():
            print(f"{rel}: NO STORED UNIT")
            failed += 1
            continue

        line = f"{rel}:"
        if f.suffix == ".json":
            a = json.loads(f.read_text(encoding="utf-8")).get("rmse")
            b = json.loads(o.read_text(encoding="utf-8")).get("rmse")
            if (a is None) != (b is None):
                line += " RMSE RECORD MISSING"
                failed += 1
            elif a is not None:
                if not isinstance(a, dict) or not isinstance(b, dict) or not a or set(a) != set(b):
                    line += " RMSE METHOD KEYS MISMATCH"
                    failed += 1
                else:
                    keys = sorted(a)
                    good = all(np.isclose(a[k], b[k], rtol=rtol, atol=atol)
                               for k in keys)
                    rmse_checked += 1
                    failed += not good
                    line += f' rmse {"OK" if good else "MISMATCH"} ({len(keys)} methods)'

        p, q = f.with_suffix(".npz"), o.with_suffix(".npz")
        if p.is_file() and q.is_file():
            with np.load(p, allow_pickle=False) as x, np.load(q, allow_pickle=False) as y:
                if set(x.files) != set(y.files):
                    bad = sorted(set(x.files) ^ set(y.files))
                else:
                    bad = [k for k in sorted(x.files)
                           if not numeric_match(x[k], y[k], rtol=rtol, atol=atol)]
            arrays_checked += 1
            failed += bool(bad)
            line += f' npz {"OK" if not bad else "MISMATCH " + ",".join(bad[:4])}'
        elif p.is_file() != q.is_file():
            arrays_unavailable += 1
            line += " npz NOT COMPARABLE (one archive lacks arrays)"
            if require_arrays:
                failed += 1

        print(line)

    if rmse_checked + arrays_checked == 0:
        raise ValueError("no replay RMSE or NPZ arrays were checked")

    print(f"Checked {rmse_checked} RMSE records and {arrays_checked} NPZ bundles; "
          f"{arrays_unavailable} NPZ bundles unavailable in one archive.")
    if arrays_unavailable and not require_arrays:
        print("WARNING: Array-level verification is incomplete; use --require-arrays to enforce it.")
    print("PASS (within tolerance)" if not failed else f"{failed} MISMATCHES")
    return not failed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path, help="directory containing replayed units")
    parser.add_argument("stored", type=Path, nargs="?",
                        default=Path(os.environ.get("RUNS", str(ROOT / "data" / "runs"))))
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument("--require-arrays", action="store_true")
    args = parser.parse_args()
    if not (0 <= args.rtol < float("inf") and 0 <= args.atol < float("inf")):
        parser.error("rtol and atol must be finite nonnegative numbers")
    try:
        passed = compare(args.replay, args.stored, rtol=args.rtol, atol=args.atol,
                         require_arrays=args.require_arrays)
    except ValueError as exc:
        parser.exit(status=1, message=f"ERROR: {exc}\n")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
