# Shared reproducibility environment
T="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export T
export PY="${PY:-python}"
export PYTHONPATH="$T${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 DPMD_THREADS=1
export RUNS="${RUNS:-$T/data/runs}"
