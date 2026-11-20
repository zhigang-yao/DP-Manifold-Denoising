"""Unified method dispatch for reproducibility experiments."""
from modules.md.estimator import run as run_proposed

def run_all(reference,queries,*,epsilon,d,rng,config,unit_id,cache_dir):
    return run_proposed(reference,queries,epsilon=epsilon,intrinsic_dimension=d,
                        rng=rng,config=config,unit_id=unit_id,cache_dir=cache_dir)
