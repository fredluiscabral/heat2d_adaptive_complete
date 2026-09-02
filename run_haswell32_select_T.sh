#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Heat2D - selecao do menor T que preserva o comportamento
# Plataforma: Haswell, 32 threads, sem SLURM
#
# Objetivo:
#   comparar T em {100,250,500,1000} mantendo fixos N, TILE,
#   numero de threads e, principalmente, os mesmos modelos de
#   custo C_R/C_W para todos os T.
#
# Criterio automatico:
#   escolhe o menor T cujo speedup vs WAIT difere em no maximo
#   SPEEDUP_TOL do resultado em T=max(T_LIST), simultaneamente
#   para PROGRESS e CRITICAL nos dois backends.
# ============================================================

THREADS=32
T_LIST_STR="${T_LIST:-100 250 500 1000}"
read -r -a T_LIST <<< "${T_LIST_STR}"
N_VALUE="${N_VALUE:-8192}"
TILE_VALUE="${TILE_VALUE:-32}"
REPS="${REPS:-3}"
PROFILE_REPS="${PROFILE_REPS:-1}"
CAL_T="${CAL_T:-250}"
CAL_SAMPLES="${CAL_SAMPLES:-16}"
WAIT_CAL_SAMPLES="${WAIT_CAL_SAMPLES:-16}"
PROGRESS_LAMBDA="${PROGRESS_LAMBDA:-1.0}"
PROGRESS_BOOTSTRAP="${PROGRESS_BOOTSTRAP:-8}"
SPEEDUP_TOL="${SPEEDUP_TOL:-0.03}"
OUTDIR="${OUTDIR:-results/haswell32_T_selection}"

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

mkdir -p "${OUTDIR}" "${OUTDIR}/calibration"

PARAM_BACKUP="$(mktemp ./param.txt.selectT.XXXXXX)"
cp param.txt "${PARAM_BACKUP}"
restore_param() {
    if [[ -f "${PARAM_BACKUP}" ]]; then
        cp "${PARAM_BACKUP}" param.txt
        rm -f "${PARAM_BACKUP}"
    fi
}
trap restore_param EXIT INT TERM

set_param() {
    local tval="$1"
    python3 - "${N_VALUE}" "${tval}" "${TILE_VALUE}" <<'PY'
import re, sys
path='param.txt'
values={'N':sys.argv[1], 'T':sys.argv[2], 'TILE':sys.argv[3], 'WRITE_OUTPUT':'0'}
with open(path, encoding='utf-8') as f: lines=f.readlines()
out=[]; seen=set()
for line in lines:
    m=re.match(r'^(\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*=\s*)(.*?)(\s*)$', line.rstrip('\n'))
    if m and m.group(2) in values:
        k=m.group(2); out.append(f'{m.group(1)}{k}{m.group(3)}{values[k]}\n'); seen.add(k)
    else:
        out.append(line)
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

# ------------------------------------------------------------
# Calibracao UNICA. Isso evita confundir efeito de T com
# diferencas nos custos calibrados.
# ------------------------------------------------------------
CALDIR="${OUTDIR}/calibration"
COST="${CALDIR}/heat2d_cost_model.dat"
BW_WAIT="${CALDIR}/heat2d_wait_cost_busywait.dat"
SM_WAIT="${CALDIR}/heat2d_wait_cost_semaphore.dat"

set_param "${CAL_T}"
echo "[calibration] unica em p=${THREADS}, T=${CAL_T}"
HEAT2D_ENABLE_PREDICT=1 \
HEAT2D_ENABLE_RECOMPUTE=1 \
HEAT2D_CALIBRATION_SAMPLES="${CAL_SAMPLES}" \
HEAT2D_COST_FILE="${COST}" \
    "${PR_CAL}" > "${CALDIR}/calibration_pr.log" 2>&1

HEAT2D_WAIT_CALIBRATION_SAMPLES="${WAIT_CAL_SAMPLES}" \
HEAT2D_WAIT_COST_FILE="${BW_WAIT}" \
    "${BW_WCAL}" > "${CALDIR}/calibration_wait_busywait.log" 2>&1

HEAT2D_WAIT_CALIBRATION_SAMPLES="${WAIT_CAL_SAMPLES}" \
HEAT2D_WAIT_COST_FILE="${SM_WAIT}" \
    "${SM_WCAL}" > "${CALDIR}/calibration_wait_semaphore.log" 2>&1

for f in "${COST}" "${BW_WAIT}" "${SM_WAIT}"; do
    [[ -s "${f}" ]] || { echo "ERRO: calibracao nao gerou ${f}" >&2; exit 3; }
done

run_solver() {
    local exe="$1" backend="$2" policy="$3" log="$4"
    local recompute criticality waitcost
    case "${policy}" in
        wait)     recompute=0; criticality=0 ;;
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

for tval in "${T_LIST[@]}"; do
    set_param "${tval}"
    tdir="${OUTDIR}/T_$(printf '%04d' "${tval}")"
    mkdir -p "${tdir}"
    rm -f "${tdir}"/*.log 2>/dev/null || true

    echo
    echo "============================================================"
    echo "T = ${tval}, threads = ${THREADS}"
    echo "============================================================"

    echo "[warm-up] 6 cases"
    for backend in busywait semaphore; do
        for policy in wait progress critical; do
            if [[ "${backend}" == "busywait" ]]; then exe="${BW_REF}"; else exe="${SM_REF}"; fi
            run_solver "${exe}" "${backend}" "${policy}" "${tdir}/${backend}_${policy}_warmup.log"
        done
    done

    echo "[timing] ${REPS} repetitions"
    for ((rep=1; rep<=REPS; ++rep)); do
        if (( rep % 2 == 1 )); then
            cases=("busywait wait" "busywait progress" "busywait critical" "semaphore wait" "semaphore progress" "semaphore critical")
        else
            cases=("semaphore critical" "semaphore progress" "semaphore wait" "busywait critical" "busywait progress" "busywait wait")
        fi
        for c in "${cases[@]}"; do
            read -r backend policy <<< "${c}"
            if [[ "${backend}" == "busywait" ]]; then exe="${BW_REF}"; else exe="${SM_REF}"; fi
            run_solver "${exe}" "${backend}" "${policy}" "${tdir}/${backend}_${policy}_ref_r$(printf '%02d' "${rep}").log"
        done
        echo "T=${tval} timing rep=${rep}/${REPS} complete"
    done

    echo "[profiling] ${PROFILE_REPS} repetitions"
    for ((rep=1; rep<=PROFILE_REPS; ++rep)); do
        for backend in busywait semaphore; do
            for policy in wait progress critical; do
                if [[ "${backend}" == "busywait" ]]; then exe="${BW_PROF}"; else exe="${SM_PROF}"; fi
                run_solver "${exe}" "${backend}" "${policy}" "${tdir}/${backend}_${policy}_prof_r$(printf '%02d' "${rep}").log"
            done
        done
    done
done

python3 - "${OUTDIR}" "${REPS}" "${PROFILE_REPS}" "${SPEEDUP_TOL}" <<'PY'
import csv, glob, os, re, statistics, sys
outdir=sys.argv[1]; reps=int(sys.argv[2]); preps=int(sys.argv[3]); tol=float(sys.argv[4])

def parse_log(path, profile=False):
    d={}
    with open(path,errors='replace') as f:
        for line in f:
            m=re.match(r'^Tempo\s*:\s*([-+0-9.eE]+)',line)
            if m: d['time_s']=float(m.group(1))
            m=re.match(r'^Linf:\s*([-+0-9.eE]+)',line)
            if m: d['linf']=float(m.group(1))
            if profile:
                mapping={
                    'Adaptive actions RECOMPUTE':'recompute',
                    'Adaptive actions WAIT':'wait',
                    'Adaptive lead-guard waits':'lead_wait',
                    'Progress penalty nonzero fraction':'penalty_nonzero_fraction',
                    'Critical RECOMPUTE':'critical_recompute',
                    'Noncritical RECOMPUTE rejected':'noncritical_recompute_rejected',
                }
                for prefix,key in mapping.items():
                    if line.startswith(prefix+':'):
                        d[key]=float(line.split(':',1)[1].strip().split()[0]); break
    return d

times=[]; prof=[]
for path in glob.glob(os.path.join(outdir,'T_*','*_ref_r*.log')):
    b=os.path.basename(path); m=re.fullmatch(r'(busywait|semaphore)_(wait|progress|critical)_ref_r(\d+)\.log',b)
    if not m: continue
    T=int(os.path.basename(os.path.dirname(path)).split('_')[1])
    x=parse_log(path); times.append({'T':T,'backend':m.group(1),'policy':m.group(2),'rep':int(m.group(3)),**x})
for path in glob.glob(os.path.join(outdir,'T_*','*_prof_r*.log')):
    b=os.path.basename(path); m=re.fullmatch(r'(busywait|semaphore)_(wait|progress|critical)_prof_r(\d+)\.log',b)
    if not m: continue
    T=int(os.path.basename(os.path.dirname(path)).split('_')[1])
    x=parse_log(path,True); prof.append({'T':T,'backend':m.group(1),'policy':m.group(2),'rep':int(m.group(3)),**x})

with open(os.path.join(outdir,'times.csv'),'w',newline='') as f:
    cols=['T','backend','policy','rep','time_s','linf']; w=csv.DictWriter(f,fieldnames=cols); w.writeheader(); w.writerows(times)

pcols=['T','backend','policy','rep','time_s','linf','recompute','wait','lead_wait','penalty_nonzero_fraction','critical_recompute','noncritical_recompute_rejected']
for r in prof:
    for c in pcols: r.setdefault(c,0.0)
with open(os.path.join(outdir,'profile_per_run.csv'),'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=pcols); w.writeheader(); w.writerows(prof)

Ts=sorted({r['T'] for r in times}); rows=[]
for T in Ts:
    for backend in ('busywait','semaphore'):
        wait_times=[r['time_s'] for r in times if r['T']==T and r['backend']==backend and r['policy']=='wait']
        if len(wait_times)!=reps: raise SystemExit(f'WAIT reps incompletas T={T} {backend}')
        med_wait=statistics.median(wait_times)
        for policy in ('wait','progress','critical'):
            rr=[r for r in times if r['T']==T and r['backend']==backend and r['policy']==policy]
            pp=[r for r in prof if r['T']==T and r['backend']==backend and r['policy']==policy]
            if len(rr)!=reps or len(pp)!=preps: raise SystemExit(f'reps incompletas T={T} {backend} {policy}')
            med_t=statistics.median(r['time_s'] for r in rr)
            med=lambda k: statistics.median(float(r.get(k,0.0)) for r in pp)
            rows.append({
                'T':T,'backend':backend,'policy':policy,
                'median_time_s':med_t,
                'speedup_vs_wait':med_wait/med_t,
                'time_reduction_vs_wait_pct':(med_wait-med_t)/med_wait*100,
                'recompute_per_step':med('recompute')/T,
                'wait_per_step':med('wait')/T,
                'lead_wait_per_step':med('lead_wait')/T,
                'penalty_nonzero_fraction':med('penalty_nonzero_fraction'),
                'critical_recompute_per_step':med('critical_recompute')/T,
                'noncritical_rejected_per_step':med('noncritical_recompute_rejected')/T,
                'median_linf':statistics.median(r['linf'] for r in rr),
            })

with open(os.path.join(outdir,'summary.csv'),'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

Tref=max(Ts)
ref={(r['backend'],r['policy']):r['speedup_vs_wait'] for r in rows if r['T']==Tref and r['policy']!='wait'}
recommend=None; detail=[]
for T in Ts:
    ok=True; diffs=[]
    for backend in ('busywait','semaphore'):
        for policy in ('progress','critical'):
            cur=next(r['speedup_vs_wait'] for r in rows if r['T']==T and r['backend']==backend and r['policy']==policy)
            d=abs(cur-ref[(backend,policy)])
            diffs.append((backend,policy,cur,ref[(backend,policy)],d))
            if d>tol: ok=False
    detail.append((T,ok,diffs))
    if ok and recommend is None: recommend=T
if recommend is None: recommend=Tref

recpath=os.path.join(outdir,'t_recommendation.txt')
with open(recpath,'w') as f:
    f.write(f'Reference T = {Tref}\n')
    f.write(f'Speedup absolute tolerance = {tol}\n')
    for T,ok,diffs in detail:
        f.write(f'T={T}: {"STABLE" if ok else "NOT_STABLE"}\n')
        for b,p,cur,rf,d in diffs:
            f.write(f'  {b} {p}: S(T)={cur:.6f}, S(ref)={rf:.6f}, |delta|={d:.6f}\n')
    f.write(f'\nRECOMMENDED_T={recommend}\n')

print('\nSpeedup vs WAIT by T:')
for r in rows:
    if r['policy']!='wait':
        print(f"T={r['T']:4d} {r['backend']:9s} {r['policy']:8s} S={r['speedup_vs_wait']:.5f} delta={r['time_reduction_vs_wait_pct']:+.2f}%")
print(f'\nRecommended T = {recommend} (reference={Tref}, tolerance={tol})')
print(f'Results: {os.path.join(outdir,"summary.csv")}')
print(f'Recommendation: {recpath}')
PY

{
    echo "date=$(date --iso-8601=seconds 2>/dev/null || date)"
    echo "host=$(hostname)"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "threads=${THREADS}"
    echo "N=${N_VALUE}"
    echo "T_LIST=${T_LIST_STR}"
    echo "TILE=${TILE_VALUE}"
    echo "REPS=${REPS}"
    echo "PROFILE_REPS=${PROFILE_REPS}"
    echo "CAL_T=${CAL_T}"
    echo "CAL_SAMPLES=${CAL_SAMPLES}"
    echo "WAIT_CAL_SAMPLES=${WAIT_CAL_SAMPLES}"
    echo "SPEEDUP_TOL=${SPEEDUP_TOL}"
} > "${OUTDIR}/run_metadata.txt"

echo "T-selection complete."
