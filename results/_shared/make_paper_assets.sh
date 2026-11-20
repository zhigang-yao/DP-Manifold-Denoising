#!/bin/bash
# All paper tables/figures from $RUNS into results/_build (results._shared.plotting), then a copy of each
# into results/<case>/, following the paper order.
# usage: bash results/_shared/make_paper_assets.sh [out=results/_build]
set -eu
source "$(dirname "$0")/../../experiments/_shared/env.sh"
OUT="${1:-$T/results/_build}"; mkdir -p "$OUT"
"$PY" -B -m results._shared.case4_d10_csv "$RUNS/case4_d10/units" "$RUNS/case4_d10/per_dataset_rmse.csv"
"$PY" -B -W ignore -m results._shared.geometry_gallery "$OUT"
"$PY" -B -W ignore -m results._shared.plotting "$RUNS" "$RUNS/comp_m1_eps2" "$RUNS/case4_d10/per_dataset_rmse.csv" "$OUT" > "$OUT/assets.log"
place() { d="$T/results/$1"; mkdir -p "$d"; shift
  for f in "$@"; do case $f in tab_*) cp "$OUT/tables/$f.tex" "$d/";; *) cp "$OUT/figures/$f.pdf" "$OUT/preview/$f.png" "$d/";; esac; done; }
place reconstruction sim_reconstruction
place case1_epsilon sim_rmse_eps_half tab_app_sweep_eps
place case2_n sim_rmse_n_half tab_app_sweep_n
place case3_offset tab_sim_hard_soft_a tab_app_hard_soft_a_dg tab_app_hard_soft_a_manifold
place case4_release sim_release_m_grid tab_app_release_m
place case5_disconnected sim_disconnected app_sim_disconnected_view tab_app_disconnected
place case6_highd sim_highd tab_app_highd
place case7_fourier sim_fourier tab_app_fourier
place app_noise_sigma app_sim_sigma tab_app_sweep_sigma tab_app_noise
place app_geometries app_sim_geometries app_sim_extra_geometry tab_app_extra
place app_baselines tab_app_baselines_m20 tab_app_baselines
echo "assets in $OUT and results/<case>"
