#!/usr/bin/env bash
set -euo pipefail

# Smoke test rapido do pipeline da Antares.
# Nao usar estes tempos como resultado cientifico.
# Executa 14 e 28 threads, WAIT/PROGRESS/CRITICAL,
# busy-wait/semaphore, com problema e amostragem pequenos.

N_VALUE="${N_VALUE:-512}" \
T_VALUE="${T_VALUE:-100}" \
TILE_VALUE="${TILE_VALUE:-32}" \
THREADS_LIST="${THREADS_LIST:-14,28}" \
REPS="${REPS:-1}" \
PROFILE_REPS="${PROFILE_REPS:-1}" \
CAL_SAMPLES="${CAL_SAMPLES:-2}" \
WAIT_CAL_SAMPLES="${WAIT_CAL_SAMPLES:-2}" \
CAL_T="${CAL_T:-50}" \
OUTDIR="${OUTDIR:-results/antares_smoke}" \
./run_antares_baseline.sh
