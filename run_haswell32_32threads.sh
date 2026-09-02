#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Heat2D - experimento principal na Haswell com 32 threads
# T deve ser fornecido explicitamente apos a etapa de selecao.
#
# Uso:
#   T_VALUE=250 ./run_haswell32_32threads.sh
#
# Por padrao reutiliza a calibracao gerada por
# run_haswell32_select_T.sh. Se ela nao existir, calibra de novo.
# ============================================================

THREADS=32
: "${T_VALUE:?ERRO: informe T_VALUE, por exemplo: T_VALUE=250 ./run_haswell32_32threads.sh}"
N_VALUE="${N_VALUE:-8192}"
TILE_VALUE="${TILE_VALUE:-32}"
REPS="${REPS:-10}"
PROFILE_REPS="${PROFILE_REPS:-3}"
PROGRESS_LAMBDA="${PROGRESS_LAMBDA:-1.0}"
PROGRESS_BOOTSTRAP="${PROGRESS_BOOTSTRAP:-8}"
CAL_SAMPLES="${CAL_SAMPLES:-16}"
WAIT_CAL_SAMPLES="${WAIT_CAL_SAMPLES:-16}"
CAL_T="${CAL_T:-250}"
CALIBRATION_DIR="${CALIBRATION_DIR:-results/haswell32_T_selection/calibration}"
OUTDIR="${OUTDIR:-results/haswell32_32threads_T${T_VALUE}}"

BW_REF="./heat2d_adaptive_busywait"
SM_REF="./heat2d_adaptive_sem"
BW_PROF="./heat2d_adaptive_busywait_profile"
SM_PROF="./heat2d_adaptive_sem_profile"
PR_CAL="./heat2d_dependency_calibrate"
BW_WCAL="./heat2d_wait_calibrate_busywait"
SM_WCAL="./heat2d_wait_calibrate_semaphore"

export OMP_NUM_THREADS="${THREADS}"
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_DYNAMIC=FALSE
export HEAT2D_MAX_LEAD=2

command -v python3 >/dev/null || { echo "ERRO: python3 nao encontrado." >&2; exit 1; }
command -v make >/dev/null || { echo "ERRO: make nao encontrado." >&2; exit 1; }
[[ -f param.txt ]] || { echo "ERRO: param.txt nao encontrado." >&2; exit 1; }
mkdir -p "${OUTDIR}"

PARAM_BACKUP="$(mktemp ./param.txt.run32.XXXXXX)"
cp param.txt "${PARAM_BACKUP}"
restore_param() {
    if [[ -f "${PARAM_BACKUP}" ]]; then cp "${PARAM_BACKUP}" param.txt; rm -f "${PARAM_BACKUP}"; fi
}
trap restore_param EXIT INT TERM

set_param() {
    local tval="$1"
    python3 - "${N_VALUE}" "${tval}" "${TILE_VALUE}" <<'PY'
import re,sys
path='param.txt'; values={'N':sys.argv[1],'T':sys.argv[2],'TILE':sys.argv[3],'WRITE_OUTPUT':'0'}
with open(path,encoding='utf-8') as f: lines=f.readlines()
out=[]; seen=set()
for line in lines:
    m=re.match(r'^(\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*=\s*)(.*?)(\s*)$',line.rstrip('\n'))
    if m and m.group(2) in values:
        k=m.group(2); out.append(f'{m.group(1)}{k}{m.group(3)}{values[k]}\n'); seen.add(k)
    else: out.append(line)
for k,v in values.items():
    if k not in seen: out.append(f'{k} = {v}\n')
with open(path,'w',encoding='utf-8') as f: f.writelines(out)
PY
}

MAKE_JOBS="${MAKE_JOBS:-$(nproc 2>/dev/null || echo 4)}"
echo "[build] make -j ${MAKE_JOBS} core profiles"
make -j "${MAKE_JOBS}" core profiles

for exe in "${BW_REF}" "${SM_REF}" "${BW_PROF}" "${SM_PROF}" "${PR_CAL}" "${BW_WCAL}" "${SM_WCAL}"; do
    [[ -x "${exe}" ]] || { echo "ERRO: executavel ausente: ${exe}" >&2; exit 2; }
done

COST="${CALIBRATION_DIR}/heat2d_cost_model.dat"
BW_WAIT="${CALIBRATION_DIR}/heat2d_wait_cost_busywait.dat"
SM_WAIT="${CALIBRATION_DIR}/heat2d_wait_cost_semaphore.dat"

if [[ -s "${COST}" && -s "${BW_WAIT}" && -s "${SM_WAIT}" ]]; then
    echo "[calibration] reutilizando ${CALIBRATION_DIR}"
else
    echo "[calibration] arquivos nao encontrados; calibrando uma vez em p=${THREADS}, T=${CAL_T}"
    mkdir -p "${CALIBRATION_DIR}"
    set_param "${CAL_T}"
    HEAT2D_ENABLE_PREDICT=1 HEAT2D_ENABLE_RECOMPUTE=1 \
    HEAT2D_CALIBRATION_SAMPLES="${CAL_SAMPLES}" HEAT2D_COST_FILE="${COST}" \
        "${PR_CAL}" > "${CALIBRATION_DIR}/calibration_pr.log" 2>&1
    HEAT2D_WAIT_CALIBRATION_SAMPLES="${WAIT_CAL_SAMPLES}" HEAT2D_WAIT_COST_FILE="${BW_WAIT}" \
        "${BW_WCAL}" > "${CALIBRATION_DIR}/calibration_wait_busywait.log" 2>&1
    HEAT2D_WAIT_CALIBRATION_SAMPLES="${WAIT_CAL_SAMPLES}" HEAT2D_WAIT_COST_FILE="${SM_WAIT}" \
        "${SM_WCAL}" > "${CALIBRATION_DIR}/calibration_wait_semaphore.log" 2>&1
fi

set_param "${T_VALUE}"
cp param.txt "${OUTDIR}/param.effective.txt"

run_solver() {
    local exe="$1" backend="$2" policy="$3" log="$4"
    local recompute criticality waitcost
    case "${policy}" in
        wait) recompute=0; criticality=0 ;;
        progress) recompute=1; criticality=0 ;;
        critical) recompute=1; criticality=1 ;;
        *) echo "ERRO: policy invalida ${policy}" >&2; exit 4 ;;
    esac
    if [[ "${backend}" == "busywait" ]]; then waitcost="${BW_WAIT}"; else waitcost="${SM_WAIT}"; fi
    HEAT2D_ENABLE_PREDICT=0 \
    HEAT2D_ENABLE_RECOMPUTE="${recompute}" \
    HEAT2D_ENABLE_CRITICALITY="${criticality}" \
    HEAT2D_COST_FILE="${COST}" \
    HEAT2D_WAIT_COST_FILE="${waitcost}" \
    HEAT2D_PROGRESS_LAMBDA="${PROGRESS_LAMBDA}" \
    HEAT2D_PROGRESS_BOOTSTRAP_SAMPLES="${PROGRESS_BOOTSTRAP}" \
        "${exe}" > "${log}" 2>&1
}

echo "[warm-up] 6 cases, p=${THREADS}, T=${T_VALUE}"
for backend in busywait semaphore; do
    for policy in wait progress critical; do
        if [[ "${backend}" == "busywait" ]]; then exe="${BW_REF}"; else exe="${SM_REF}"; fi
        run_solver "${exe}" "${backend}" "${policy}" "${OUTDIR}/${backend}_${policy}_warmup.log"
    done
done

echo "[timing] ${REPS} clean repetitions per case"
for ((rep=1; rep<=REPS; ++rep)); do
    if (( rep % 2 == 1 )); then
        cases=("busywait wait" "busywait progress" "busywait critical" "semaphore wait" "semaphore progress" "semaphore critical")
    else
        cases=("semaphore critical" "semaphore progress" "semaphore wait" "busywait critical" "busywait progress" "busywait wait")
    fi
    for c in "${cases[@]}"; do
        read -r backend policy <<< "${c}"
        if [[ "${backend}" == "busywait" ]]; then exe="${BW_REF}"; else exe="${SM_REF}"; fi
        run_solver "${exe}" "${backend}" "${policy}" "${OUTDIR}/${backend}_${policy}_ref_r$(printf '%02d' "${rep}").log"
    done
    echo "timing rep=${rep}/${REPS} complete"
done

echo "[profiling] ${PROFILE_REPS} repetitions per case"
for ((rep=1; rep<=PROFILE_REPS; ++rep)); do
    for backend in busywait semaphore; do
        for policy in wait progress critical; do
            if [[ "${backend}" == "busywait" ]]; then exe="${BW_PROF}"; else exe="${SM_PROF}"; fi
            run_solver "${exe}" "${backend}" "${policy}" "${OUTDIR}/${backend}_${policy}_prof_r$(printf '%02d' "${rep}").log"
        done
    done
done

python3 - "${OUTDIR}" "${REPS}" "${PROFILE_REPS}" "${T_VALUE}" <<'PY'
import csv,glob,os,re,statistics,sys
outdir=sys.argv[1]; reps=int(sys.argv[2]); preps=int(sys.argv[3]); T=int(sys.argv[4])

def readlog(path,profile=False):
    x={}
    mapping={'Adaptive actions RECOMPUTE':'recompute','Adaptive actions WAIT':'wait','Adaptive lead-guard waits':'lead_wait','Progress-blocked RECOMPUTE':'progress_blocked','Progress penalty observed mean':'penalty_mean_ticks','Progress penalty nonzero fraction':'penalty_nonzero_fraction','Critical RECOMPUTE':'critical_recompute','Noncritical RECOMPUTE rejected':'noncritical_recompute_rejected','Criticality-forced WAIT':'criticality_forced_wait'}
    with open(path,errors='replace') as f:
        for line in f:
            m=re.match(r'^Tempo\s*:\s*([-+0-9.eE]+)',line)
            if m: x['time_s']=float(m.group(1))
            m=re.match(r'^Linf:\s*([-+0-9.eE]+)',line)
            if m: x['linf']=float(m.group(1))
            if profile:
                for p,k in mapping.items():
                    if line.startswith(p+':'):
                        x[k]=float(line.split(':',1)[1].strip().split()[0]); break
    return x

times=[]; prof=[]
for path in glob.glob(os.path.join(outdir,'*_ref_r*.log')):
    m=re.fullmatch(r'(busywait|semaphore)_(wait|progress|critical)_ref_r(\d+)\.log',os.path.basename(path))
    if m: times.append({'backend':m.group(1),'policy':m.group(2),'rep':int(m.group(3)),**readlog(path)})
for path in glob.glob(os.path.join(outdir,'*_prof_r*.log')):
    m=re.fullmatch(r'(busywait|semaphore)_(wait|progress|critical)_prof_r(\d+)\.log',os.path.basename(path))
    if m: prof.append({'backend':m.group(1),'policy':m.group(2),'rep':int(m.group(3)),**readlog(path,True)})

with open(os.path.join(outdir,'times.csv'),'w',newline='') as f:
    cols=['backend','policy','rep','time_s','linf']; w=csv.DictWriter(f,fieldnames=cols); w.writeheader(); w.writerows(times)

summary=[]
for backend in ('busywait','semaphore'):
    wait=[r['time_s'] for r in times if r['backend']==backend and r['policy']=='wait']; medw=statistics.median(wait)
    for policy in ('wait','progress','critical'):
        rr=[r for r in times if r['backend']==backend and r['policy']==policy]
        pp=[r for r in prof if r['backend']==backend and r['policy']==policy]
        if len(rr)!=reps or len(pp)!=preps: raise SystemExit(f'Reps incompletas {backend} {policy}')
        medt=statistics.median(r['time_s'] for r in rr); med=lambda k: statistics.median(float(r.get(k,0.0)) for r in pp)
        summary.append({'threads':32,'T':T,'backend':backend,'policy':policy,'median_time_s':medt,'speedup_vs_wait':medw/medt,'time_reduction_vs_wait_pct':(medw-medt)/medw*100,'median_recompute':med('recompute'),'median_wait':med('wait'),'median_lead_wait':med('lead_wait'),'median_progress_blocked':med('progress_blocked'),'median_penalty_mean_ticks':med('penalty_mean_ticks'),'median_penalty_nonzero_fraction':med('penalty_nonzero_fraction'),'median_critical_recompute':med('critical_recompute'),'median_noncritical_recompute_rejected':med('noncritical_recompute_rejected'),'median_criticality_forced_wait':med('criticality_forced_wait'),'median_linf':statistics.median(r['linf'] for r in rr)})
with open(os.path.join(outdir,'summary.csv'),'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)

print('\nSpeedup vs WAIT:')
for r in summary:
    if r['policy']!='wait': print(f"{r['backend']:9s} {r['policy']:8s} S={r['speedup_vs_wait']:.5f} delta={r['time_reduction_vs_wait_pct']:+.2f}%")
print(f'\nSummary: {os.path.join(outdir,"summary.csv")}')
PY

{
    echo "date=$(date --iso-8601=seconds 2>/dev/null || date)"
    echo "host=$(hostname)"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "threads=${THREADS}"
    echo "N=${N_VALUE}"
    echo "T=${T_VALUE}"
    echo "TILE=${TILE_VALUE}"
    echo "REPS=${REPS}"
    echo "PROFILE_REPS=${PROFILE_REPS}"
    echo "CALIBRATION_DIR=${CALIBRATION_DIR}"
} > "${OUTDIR}/run_metadata.txt"

echo "Haswell 32-thread experiment complete."
