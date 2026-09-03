#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Heat2D - Antares (2 x Xeon Gold 5120, 14 cores/socket, SMT2)
# Experimento principal: 14 threads (1 socket) e 28 threads
# (2 sockets), sempre 1 thread por core fisico e sem SMT.
#
# T deve ser fornecido explicitamente:
#   T_VALUE=250 ./run_antares_baseline.sh
#
# Afinidade:
#   14 threads -> CPUs 0-13  (socket/NUMA node 0)
#   28 threads -> CPUs 0-27  (14 cores de cada socket)
#
# A calibracao de C_P/C_R e C_W e refeita separadamente para
# cada numero de threads, pois os custos dependem da maquina e p.
# PREDICT permanece desligado no experimento principal.
# ============================================================

: "${T_VALUE:?ERRO: informe T_VALUE, por exemplo: T_VALUE=250 ./run_antares_baseline.sh}"

THREADS_LIST_STR="${THREADS_LIST:-14,28}"
IFS=',' read -r -a THREADS_LIST <<< "${THREADS_LIST_STR}"

N_VALUE="${N_VALUE:-8192}"
TILE_VALUE="${TILE_VALUE:-32}"
REPS="${REPS:-10}"
PROFILE_REPS="${PROFILE_REPS:-3}"
PROGRESS_LAMBDA="${PROGRESS_LAMBDA:-1.0}"
PROGRESS_BOOTSTRAP="${PROGRESS_BOOTSTRAP:-8}"
CAL_SAMPLES="${CAL_SAMPLES:-16}"
WAIT_CAL_SAMPLES="${WAIT_CAL_SAMPLES:-16}"
CAL_T="${CAL_T:-250}"
OUTDIR="${OUTDIR:-results/antares_T${T_VALUE}}"

BW_REF="./heat2d_adaptive_busywait"
SM_REF="./heat2d_adaptive_sem"
BW_PROF="./heat2d_adaptive_busywait_profile"
SM_PROF="./heat2d_adaptive_sem_profile"
PR_CAL="./heat2d_dependency_calibrate"
BW_WCAL="./heat2d_wait_calibrate_busywait"
SM_WCAL="./heat2d_wait_calibrate_semaphore"

export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_DYNAMIC=FALSE
export HEAT2D_MAX_LEAD=2

command -v python3 >/dev/null || { echo "ERRO: python3 nao encontrado." >&2; exit 1; }
command -v make >/dev/null || { echo "ERRO: make nao encontrado." >&2; exit 1; }
command -v taskset >/dev/null || { echo "ERRO: taskset nao encontrado (pacote util-linux)." >&2; exit 1; }
[[ -f param.txt ]] || { echo "ERRO: param.txt nao encontrado." >&2; exit 1; }
mkdir -p "${OUTDIR}"

PARAM_BACKUP="$(mktemp ./param.txt.antares.XXXXXX)"
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
import re,sys
path='param.txt'
values={'N':sys.argv[1],'T':sys.argv[2],'TILE':sys.argv[3],'WRITE_OUTPUT':'0'}
with open(path,encoding='utf-8') as f:
    lines=f.readlines()
out=[]; seen=set()
for line in lines:
    m=re.match(r'^(\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*=\s*)(.*?)(\s*)$',line.rstrip('\n'))
    if m and m.group(2) in values:
        k=m.group(2)
        out.append(f'{m.group(1)}{k}{m.group(3)}{values[k]}\n')
        seen.add(k)
    else:
        out.append(line)
for k,v in values.items():
    if k not in seen:
        out.append(f'{k} = {v}\n')
with open(path,'w',encoding='utf-8') as f:
    f.writelines(out)
PY
}

cpu_list_for_threads() {
    case "$1" in
        14) echo "0-13" ;;
        28) echo "0-27" ;;
        *)
            echo "ERRO: THREADS_LIST suporta 14 e/ou 28 neste script; recebido '$1'." >&2
            exit 2
            ;;
    esac
}

topology_label_for_threads() {
    case "$1" in
        14) echo "one_socket_no_smt" ;;
        28) echo "two_socket_no_smt" ;;
    esac
}

MAKE_JOBS="${MAKE_JOBS:-$(nproc 2>/dev/null || echo 4)}"
echo "[build] make -j ${MAKE_JOBS} core profiles"
make -j "${MAKE_JOBS}" core profiles

for exe in "${BW_REF}" "${SM_REF}" "${BW_PROF}" "${SM_PROF}" "${PR_CAL}" "${BW_WCAL}" "${SM_WCAL}"; do
    [[ -x "${exe}" ]] || { echo "ERRO: executavel ausente: ${exe}" >&2; exit 3; }
done

# Registra a topologia real da maquina usada.
lscpu > "${OUTDIR}/lscpu.txt" 2>&1 || true
lscpu -e=CPU,NODE,SOCKET,CORE > "${OUTDIR}/lscpu_extended.txt" 2>&1 || true
if command -v numactl >/dev/null; then
    numactl --hardware > "${OUTDIR}/numactl_hardware.txt" 2>&1 || true
fi

run_pinned_env() {
    local cpus="$1" log="$2"
    shift 2
    taskset -c "${cpus}" env "$@" > "${log}" 2>&1
}

run_solver() {
    local exe="$1" backend="$2" policy="$3" log="$4" threads="$5" cpus="$6" cost="$7" bw_wait="$8" sm_wait="$9"
    local recompute criticality waitcost

    case "${policy}" in
        wait)     recompute=0; criticality=0 ;;
        progress) recompute=1; criticality=0 ;;
        critical) recompute=1; criticality=1 ;;
        *) echo "ERRO: policy invalida ${policy}" >&2; exit 4 ;;
    esac

    if [[ "${backend}" == "busywait" ]]; then
        waitcost="${bw_wait}"
    else
        waitcost="${sm_wait}"
    fi

    run_pinned_env "${cpus}" "${log}" \
        OMP_NUM_THREADS="${threads}" \
        OMP_PLACES="${OMP_PLACES}" \
        OMP_PROC_BIND="${OMP_PROC_BIND}" \
        OMP_DYNAMIC="${OMP_DYNAMIC}" \
        HEAT2D_MAX_LEAD="${HEAT2D_MAX_LEAD}" \
        HEAT2D_ENABLE_PREDICT=0 \
        HEAT2D_ENABLE_RECOMPUTE="${recompute}" \
        HEAT2D_ENABLE_CRITICALITY="${criticality}" \
        HEAT2D_COST_FILE="${cost}" \
        HEAT2D_WAIT_COST_FILE="${waitcost}" \
        HEAT2D_PROGRESS_LAMBDA="${PROGRESS_LAMBDA}" \
        HEAT2D_PROGRESS_BOOTSTRAP_SAMPLES="${PROGRESS_BOOTSTRAP}" \
        "${exe}"
}

for threads in "${THREADS_LIST[@]}"; do
    cpus="$(cpu_list_for_threads "${threads}")"
    topology="$(topology_label_for_threads "${threads}")"
    tdir="${OUTDIR}/threads_$(printf '%03d' "${threads}")"
    caldir="${tdir}/calibration"
    mkdir -p "${tdir}" "${caldir}"

    cost="${caldir}/heat2d_cost_model.dat"
    bw_wait="${caldir}/heat2d_wait_cost_busywait.dat"
    sm_wait="${caldir}/heat2d_wait_cost_semaphore.dat"

    echo
    echo "============================================================"
    echo "ANTARES: threads=${threads} topology=${topology} cpus=${cpus}"
    echo "============================================================"

    # Calibracao independente para esta configuracao.
    set_param "${CAL_T}"
    echo "[calibration] PREDICT/RECOMPUTE (p=${threads}, CPUs ${cpus})"
    run_pinned_env "${cpus}" "${caldir}/calibration_pr.log" \
        OMP_NUM_THREADS="${threads}" OMP_PLACES="${OMP_PLACES}" OMP_PROC_BIND="${OMP_PROC_BIND}" OMP_DYNAMIC="${OMP_DYNAMIC}" \
        HEAT2D_MAX_LEAD="${HEAT2D_MAX_LEAD}" \
        HEAT2D_ENABLE_PREDICT=1 HEAT2D_ENABLE_RECOMPUTE=1 \
        HEAT2D_CALIBRATION_SAMPLES="${CAL_SAMPLES}" HEAT2D_COST_FILE="${cost}" \
        "${PR_CAL}"

    echo "[calibration] WAIT busy-wait"
    run_pinned_env "${cpus}" "${caldir}/calibration_wait_busywait.log" \
        OMP_NUM_THREADS="${threads}" OMP_PLACES="${OMP_PLACES}" OMP_PROC_BIND="${OMP_PROC_BIND}" OMP_DYNAMIC="${OMP_DYNAMIC}" \
        HEAT2D_MAX_LEAD="${HEAT2D_MAX_LEAD}" \
        HEAT2D_WAIT_CALIBRATION_SAMPLES="${WAIT_CAL_SAMPLES}" HEAT2D_WAIT_COST_FILE="${bw_wait}" \
        "${BW_WCAL}"

    echo "[calibration] WAIT semaphore"
    run_pinned_env "${cpus}" "${caldir}/calibration_wait_semaphore.log" \
        OMP_NUM_THREADS="${threads}" OMP_PLACES="${OMP_PLACES}" OMP_PROC_BIND="${OMP_PROC_BIND}" OMP_DYNAMIC="${OMP_DYNAMIC}" \
        HEAT2D_MAX_LEAD="${HEAT2D_MAX_LEAD}" \
        HEAT2D_WAIT_CALIBRATION_SAMPLES="${WAIT_CAL_SAMPLES}" HEAT2D_WAIT_COST_FILE="${sm_wait}" \
        "${SM_WCAL}"

    for f in "${cost}" "${bw_wait}" "${sm_wait}"; do
        [[ -s "${f}" ]] || { echo "ERRO: calibracao nao gerou ${f}" >&2; exit 5; }
    done

    set_param "${T_VALUE}"
    cp param.txt "${tdir}/param.effective.txt"

    echo "[warm-up] 6 cases"
    for backend in busywait semaphore; do
        for policy in wait progress critical; do
            if [[ "${backend}" == "busywait" ]]; then exe="${BW_REF}"; else exe="${SM_REF}"; fi
            run_solver "${exe}" "${backend}" "${policy}" "${tdir}/${backend}_${policy}_warmup.log" "${threads}" "${cpus}" "${cost}" "${bw_wait}" "${sm_wait}"
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
            run_solver "${exe}" "${backend}" "${policy}" "${tdir}/${backend}_${policy}_ref_r$(printf '%02d' "${rep}").log" "${threads}" "${cpus}" "${cost}" "${bw_wait}" "${sm_wait}"
        done
        echo "threads=${threads} timing rep=${rep}/${REPS} complete"
    done

    echo "[profiling] ${PROFILE_REPS} repetitions per case"
    for ((rep=1; rep<=PROFILE_REPS; ++rep)); do
        for backend in busywait semaphore; do
            for policy in wait progress critical; do
                if [[ "${backend}" == "busywait" ]]; then exe="${BW_PROF}"; else exe="${SM_PROF}"; fi
                run_solver "${exe}" "${backend}" "${policy}" "${tdir}/${backend}_${policy}_prof_r$(printf '%02d' "${rep}").log" "${threads}" "${cpus}" "${cost}" "${bw_wait}" "${sm_wait}"
            done
        done
    done

done

python3 - "${OUTDIR}" "${REPS}" "${PROFILE_REPS}" "${T_VALUE}" <<'PY'
import csv,glob,os,re,statistics,sys
outdir=sys.argv[1]; reps=int(sys.argv[2]); preps=int(sys.argv[3]); T=int(sys.argv[4])

def readlog(path,profile=False):
    x={}
    mapping={
        'Adaptive actions RECOMPUTE':'recompute',
        'Adaptive actions WAIT':'wait',
        'Adaptive lead-guard waits':'lead_wait',
        'Progress-blocked RECOMPUTE':'progress_blocked',
        'Progress penalty observed mean':'penalty_mean_ticks',
        'Progress penalty nonzero fraction':'penalty_nonzero_fraction',
        'Critical RECOMPUTE':'critical_recompute',
        'Noncritical RECOMPUTE rejected':'noncritical_recompute_rejected',
        'Criticality-forced WAIT':'criticality_forced_wait',
    }
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
for path in glob.glob(os.path.join(outdir,'threads_*','*_ref_r*.log')):
    m=re.fullmatch(r'(busywait|semaphore)_(wait|progress|critical)_ref_r(\d+)\.log',os.path.basename(path))
    if not m: continue
    threads=int(os.path.basename(os.path.dirname(path)).split('_')[1])
    times.append({'threads':threads,'T':T,'backend':m.group(1),'policy':m.group(2),'rep':int(m.group(3)),**readlog(path)})
for path in glob.glob(os.path.join(outdir,'threads_*','*_prof_r*.log')):
    m=re.fullmatch(r'(busywait|semaphore)_(wait|progress|critical)_prof_r(\d+)\.log',os.path.basename(path))
    if not m: continue
    threads=int(os.path.basename(os.path.dirname(path)).split('_')[1])
    prof.append({'threads':threads,'T':T,'backend':m.group(1),'policy':m.group(2),'rep':int(m.group(3)),**readlog(path,True)})

with open(os.path.join(outdir,'times.csv'),'w',newline='') as f:
    cols=['threads','T','backend','policy','rep','time_s','linf']
    w=csv.DictWriter(f,fieldnames=cols); w.writeheader(); w.writerows(sorted(times,key=lambda r:(r['threads'],r['backend'],r['policy'],r['rep'])))

pcols=['threads','T','backend','policy','rep','time_s','linf','recompute','wait','lead_wait','progress_blocked','penalty_mean_ticks','penalty_nonzero_fraction','critical_recompute','noncritical_recompute_rejected','criticality_forced_wait']
for r in prof:
    for c in pcols: r.setdefault(c,0.0)
with open(os.path.join(outdir,'profile_per_run.csv'),'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=pcols); w.writeheader(); w.writerows(sorted(prof,key=lambda r:(r['threads'],r['backend'],r['policy'],r['rep'])))

summary=[]
for threads in sorted({r['threads'] for r in times}):
    for backend in ('busywait','semaphore'):
        wait=[r['time_s'] for r in times if r['threads']==threads and r['backend']==backend and r['policy']=='wait']
        if len(wait)!=reps: raise SystemExit(f'WAIT reps incompletas p={threads} {backend}: {len(wait)}/{reps}')
        medw=statistics.median(wait)
        for policy in ('wait','progress','critical'):
            rr=[r for r in times if r['threads']==threads and r['backend']==backend and r['policy']==policy]
            pp=[r for r in prof if r['threads']==threads and r['backend']==backend and r['policy']==policy]
            if len(rr)!=reps or len(pp)!=preps:
                raise SystemExit(f'Reps incompletas p={threads} {backend} {policy}: timing={len(rr)}/{reps}, profile={len(pp)}/{preps}')
            medt=statistics.median(r['time_s'] for r in rr)
            med=lambda k: statistics.median(float(r.get(k,0.0)) for r in pp)
            summary.append({
                'threads':threads,
                'topology':'one_socket_no_smt' if threads==14 else 'two_socket_no_smt',
                'T':T,
                'backend':backend,
                'policy':policy,
                'median_time_s':medt,
                'speedup_vs_wait':medw/medt,
                'time_reduction_vs_wait_pct':(medw-medt)/medw*100,
                'median_recompute':med('recompute'),
                'median_wait':med('wait'),
                'median_lead_wait':med('lead_wait'),
                'median_progress_blocked':med('progress_blocked'),
                'median_penalty_mean_ticks':med('penalty_mean_ticks'),
                'median_penalty_nonzero_fraction':med('penalty_nonzero_fraction'),
                'median_critical_recompute':med('critical_recompute'),
                'median_noncritical_recompute_rejected':med('noncritical_recompute_rejected'),
                'median_criticality_forced_wait':med('criticality_forced_wait'),
                'median_linf':statistics.median(r['linf'] for r in rr),
            })

with open(os.path.join(outdir,'summary.csv'),'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)

print('\nSpeedup vs WAIT:')
for r in summary:
    if r['policy']!='wait':
        print(f"p={r['threads']:2d} {r['backend']:9s} {r['policy']:8s} S={r['speedup_vs_wait']:.5f} delta={r['time_reduction_vs_wait_pct']:+.2f}%")
print(f'\nSummary: {os.path.join(outdir,"summary.csv")}')
PY

{
    echo "date=$(date --iso-8601=seconds 2>/dev/null || date)"
    echo "host=$(hostname)"
    echo "machine=Antares"
    echo "cpu=Intel Xeon Gold 5120"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "threads_list=${THREADS_LIST_STR}"
    echo "cpu_binding_14=0-13"
    echo "cpu_binding_28=0-27"
    echo "smt_used=no"
    echo "memory_policy=system_default_first_touch"
    echo "N=${N_VALUE}"
    echo "T=${T_VALUE}"
    echo "TILE=${TILE_VALUE}"
    echo "REPS=${REPS}"
    echo "PROFILE_REPS=${PROFILE_REPS}"
    echo "CAL_T=${CAL_T}"
    echo "CAL_SAMPLES=${CAL_SAMPLES}"
    echo "WAIT_CAL_SAMPLES=${WAIT_CAL_SAMPLES}"
    echo "PROGRESS_LAMBDA=${PROGRESS_LAMBDA}"
    echo "PROGRESS_BOOTSTRAP=${PROGRESS_BOOTSTRAP}"
} > "${OUTDIR}/run_metadata.txt"

echo
echo "Results:"
echo "  ${OUTDIR}/times.csv"
echo "  ${OUTDIR}/profile_per_run.csv"
echo "  ${OUTDIR}/summary.csv"
echo
echo "Antares baseline complete."
