#!/usr/bin/env bash
set -euo pipefail

PARAM="${PARAM:-param.txt}"
THREADS="${THREADS:-4}"
N="${SMOKE_N:-512}"
T="${SMOKE_T:-100}"
TILE="${SMOKE_TILE:-32}"
CAL="${CALIBRATION_SAMPLES:-8}"
WCAL="${WAIT_CALIBRATION_SAMPLES:-8}"

[[ -f "${PARAM}" ]] || { echo "ERRO: ${PARAM} nao encontrado." >&2; exit 1; }
cp "${PARAM}" "${PARAM}.smoke.bak"
trap 'mv -f "${PARAM}.smoke.bak" "${PARAM}"; rm -f heat2d_cost_model.dat heat2d_wait_cost_busywait.dat heat2d_wait_cost_semaphore.dat output.txt' EXIT

python3 - "${PARAM}" "${N}" "${T}" "${TILE}" <<'PY'
import re,sys
p,N,T,tile=sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]
s=open(p).read()
for k,v in [('N',N),('T',T),('TILE',tile),('WRITE_OUTPUT','0')]:
    pat=rf'(?m)^\s*{k}\s*=.*$'
    if re.search(pat,s): s=re.sub(pat,f'{k} = {v}',s)
    else: s+=f'\n{k} = {v}\n'
open(p,'w').write(s)
PY

export OMP_NUM_THREADS="${THREADS}"
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_DYNAMIC=FALSE
export HEAT2D_MAX_LEAD=2

make core profiles

HEAT2D_ENABLE_PREDICT=1 HEAT2D_ENABLE_RECOMPUTE=1 HEAT2D_CALIBRATION_SAMPLES="${CAL}" ./heat2d_dependency_calibrate >/tmp/heat2d_cal_pr.log
HEAT2D_WAIT_CALIBRATION_SAMPLES="${WCAL}" ./heat2d_wait_calibrate_busywait >/tmp/heat2d_cal_bw.log
HEAT2D_WAIT_CALIBRATION_SAMPLES="${WCAL}" ./heat2d_wait_calibrate_semaphore >/tmp/heat2d_cal_sm.log

for exe in heat2d_explicit_omp_mpilike heat2d_explicit_omp_busywait_nobarrier_nofs heat2d_explicit_omp_sem_nobarrier_nofs; do
  echo "[smoke] ${exe}"; ./${exe} | grep -E '^(Variant|Tempo|Linf)'
done

for exe in heat2d_adaptive_busywait heat2d_adaptive_sem heat2d_adaptive_busywait_profile heat2d_adaptive_sem_profile; do
  echo "[smoke] ${exe}"
  HEAT2D_ENABLE_PREDICT=0 HEAT2D_ENABLE_RECOMPUTE=1 ./${exe} > /tmp/${exe}.log
  grep -q '^Adaptive progress-aware: on$' /tmp/${exe}.log
  grep -E '^(Adaptive progress-aware|Variant|Tempo|Linf|Progress-blocked RECOMPUTE)' /tmp/${exe}.log || true
done

echo "SMOKE OK"
