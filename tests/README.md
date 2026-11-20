# Tests and replay checks

Run repository checks (imports, file layout, seed/configuration invariants, portable memory reporting and shell line endings):

```bash
uv run python -m unittest discover -s tests -p 'test_*.py'
```

The GitHub Actions smoke test also checks Bash syntax and executes one full trefoil paper unit on Linux.

Compare replayed units against the archived results using:

```bash
uv run python -m tests.audit.compare /path/to/replay data/runs --rtol 1e-9 --atol 1e-12
```

The comparison fails if no numerical units are checked or if method keys differ. It checks any NPZ arrays available on both sides. The smaller published plotting archive may not contain NPZ arrays for every unit; missing array pairs are reported explicitly and only RMSE can be verified for those units. Use `--require-arrays` to require matching NPZ bundles for every replayed unit.
