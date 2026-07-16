#!/usr/bin/env bash
set -euo pipefail

DATA_RECENT="campaign.csv"
DATA_ALL_ARCHIVED="archival.csv"
SCRIPT="phot_var_twin_sine_QC_v12_mcmc.py"

# --- LCO-ONLY ---
python "$SCRIPT" "$DATA_RECENT" \
  --delim comma \
  --time-col jd --mag-col mag --mag-err-col uncertainty \
  --observer-col observer --filter-col band --band-include "Johnson V" \
  --observer-include DTYA,NFRA,TKVA,OPAA \
  --target WASP-52_LCO \
  --group gap --gap-hours 6 \
  --per-night-minN 20 --per-night-sigma 3 \
  --global-sigma 5 --global-iters 2 \
  --roll-window 0 --roll-sigma 4.0 \
  --err-median-factor 2.0 \
  --minP 3 --maxP 20 --nfreq 1000000 \
  --harmonic-window 0.25 \
  --P1-prior 10 --P1-prior-frac 0.25 \
  --exclude-periods "0.95-1.05;1.60-1.90;1.90-2.10" \
  --mcmc --mcmc-walkers 64 --mcmc-steps 12000 --mcmc-burn 3000 --mcmc-thin 5 --mcmc-period-prior-k 3.0 --mcmc-seed 42 --mcmc-moves de \
  --outdir phot_var_runs/WASP-52_LCO \
  --ic-mode gauss

# --- EPW-ONLY ---
python "$SCRIPT" "$DATA_RECENT" \
  --delim comma \
  --time-col jd --mag-col mag --mag-err-col uncertainty \
  --observer-col observer --filter-col band --band-include "Johnson V" \
  --observer-exclude DTYA,NFRA,TKVA,OPAA,ZDAB \
  --target WASP-52_EPW \
  --group gap --gap-hours 6 \
  --per-night-minN 20 --per-night-sigma 3 \
  --global-sigma 5 --global-iters 2 \
  --roll-window 0 --roll-sigma 4.0 \
  --err-median-factor 2.0 \
  --minP 3 --maxP 20 --nfreq 1000000 \
  --harmonic-window 0.25 \
  --P1-prior 10 --P1-prior-frac 0.25 \
  --exclude-periods "0.95-1.05;1.60-1.90;1.90-2.10" \
  --mcmc --mcmc-walkers 64 --mcmc-steps 12000 --mcmc-burn 3000 --mcmc-thin 5 --mcmc-period-prior-k 3.0 --mcmc-seed 42 --mcmc-moves de \
  --outdir phot_var_runs/WASP-52_EPW \
  --ic-mode gauss

# --- ALL (LCO + EPW) ---
python "$SCRIPT" "$DATA_RECENT" \
  --delim comma \
  --time-col jd --mag-col mag --mag-err-col uncertainty \
  --observer-col observer --filter-col band --band-include "Johnson V" \
  --observer-exclude ZDAB \
  --target WASP-52_all \
  --group gap --gap-hours 6 \
  --per-night-minN 20 --per-night-sigma 3 \
  --global-sigma 5 --global-iters 2 \
  --roll-window 0 --roll-sigma 4.0 \
  --err-median-factor 2.0 \
  --minP 3 --maxP 20 --nfreq 1000000 \
  --harmonic-window 0.25 \
  --P1-prior 10 --P1-prior-frac 0.25 \
  --exclude-periods "0.95-1.05;1.60-1.90;1.90-2.10" \
  --mcmc --mcmc-walkers 64 --mcmc-steps 12000 --mcmc-burn 3000 --mcmc-thin 5 --mcmc-period-prior-k 3.0 --mcmc-seed 42 --mcmc-moves de \
  --outdir phot_var_runs/WASP-52_all \
  --ic-mode gauss

# --- ARCHIVED (2015-present, all data) ---
python "$SCRIPT" "$DATA_ALL_ARCHIVED" \
  --delim comma \
  --time-col jd --mag-col mag --mag-err-col uncertainty \
  --observer-col observer --filter-col band --band-include "Johnson V" \
  --observer-exclude ZDAB \
  --target WASP-52_archived \
  --group gap --gap-hours 6 \
  --per-night-minN 20 --per-night-sigma 3 \
  --global-sigma 5 --global-iters 2 \
  --roll-window 0 --roll-sigma 4.0 \
  --err-median-factor 2.0 \
  --minP 3 --maxP 20 --nfreq 1000000 \
  --harmonic-window 0.25 \
  --P1-prior 10 --P1-prior-frac 0.25 \
  --exclude-periods "0.95-1.05;1.60-1.90;1.90-2.10" \
  --mcmc --mcmc-walkers 64 --mcmc-steps 12000 --mcmc-burn 3000 --mcmc-thin 5 --mcmc-period-prior-k 3.0 --mcmc-seed 42 --mcmc-moves de \
  --outdir phot_var_runs/WASP-52_archived \
  --ic-mode gauss

cd phot_var_runs

python v10_compare_runs_to_overleaf_mcmc.py \
  WASP-52_LCO \
  WASP-52_EPW \
  WASP-52_all \
  WASP-52_archived \
  --labels "LCO only,EPW only,All combined,Archived (2015-present)" \
  --outdir compare_WASP-52