"""Reproduce a paper simulation case or regenerate all paper plots."""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
CASES = ("reconstruction", "case1_epsilon", "case2_n", "case3_offset", "case4_release", "case5_disconnected", "case6_highd", "case7_fourier", "app_noise_sigma", "app_geometries", "app_baselines")

def main():
    parser = argparse.ArgumentParser(description="Frozen DP-MD simulations")
    parser.add_argument("--case", choices=(*CASES, "all"))
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    if not args.case and not args.plots:
        parser.error("Specify --case or --plots")
    if args.case:
        for case in CASES if args.case == "all" else (args.case,):
            subprocess.run([sys.executable, "-m", f"experiments.{case}.run"], cwd=ROOT, check=True)
    if args.plots:
        subprocess.run(["bash", str(ROOT / "results/_shared/make_paper_assets.sh")], cwd=ROOT, check=True)

if __name__ == "__main__":
    main()
