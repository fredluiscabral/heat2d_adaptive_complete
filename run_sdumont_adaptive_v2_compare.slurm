#!/bin/bash
#SBATCH --job-name=heat2d_v2_cmp
#SBATCH --partition=lncc-cpu_amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --output=heat2d_v2_cmp-%j.out
#SBATCH --error=heat2d_v2_cmp-%j.err

set -euo pipefail

# =============================================================================
# SDumont II — Adaptive-V2 comparison under OMP spread
#
# Compares, without modifying the historical implementations:
#   BASE     = native no-false-sharing solver
#   WAIT     = current adaptive infrastructure with RECOMPUTE disabled
#   CRITICAL = current progress+binary-criticality controller
#   V2       = online C_R/C_W + progress penalty + continuous phi controller
#
# Default matrix:
#   p       = 64 160
#   backend = busywait semaphore
#   policy  = base wait critical v2
#   N/T/TILE = 8192 / 500 / 32
#   PREDICT = off
#   max_lead = 2
#   OMP_PLACES=cores, OMP_PROC_BIND=spread
#
# Per p:
#   - P/R calibrated once;
#   - WAIT calibrated once per backend;
#   - one warm-up per policy/backend;
#   - 20 timing repetitions with rotated order;
#   - 3 profiling repetitions for WAIT, CRITICAL and V2.
# =============================================================================

THREADS_LIST_STR="${THREADS_LIST:-64 160}"
read -r -a THREADS_LIST <<< "${THREADS_LIST_STR}"
REPS="${REPS:-20}"
PROFILE_REPS="${PROFILE_REPS:-3}"
WARMUPS="${WARMUPS:-1}"
CAL_SAMPLES="${CAL_SAMPLES:-32}"
WAIT_CAL_SAMPLES="${WAIT_CAL_SAMPLES:-32}"

PROGRESS_LAMBDA="${PROGRESS_LAMBDA:-1.0}"
PROGRESS_BOOTSTRAP="${PROGRESS_BOOTSTRAP:-8}"
V2_ONLINE_BETA="${V2_ONLINE_BETA:-0.05}"
V2_PROGRESS_BETA="${V2_PROGRESS_BETA:-0.05}"
V2_PHI_BETA="${V2_PHI_BETA:-0.10}"
V2_PHI_FLOOR="${V2_PHI_FLOOR:-0.0}"
V2_PHI_RESAMPLE_PERIOD="${V2_PHI_RESAMPLE_PERIOD:-16}"
V2_ONLINE_BOOTSTRAP="${V2_ONLINE_BOOTSTRAP:-4}"

OUTDIR="${OUTDIR:-results/sdumont_adaptive_v2_compare_spread_T500}"

BW_BASE="./heat2d_explicit_omp_busywait_nobarrier_nofs"
SM_BASE="./heat2d_explicit_omp_sem_nobarrier_nofs"
BW_CUR="./heat2d_adaptive_busywait"
SM_CUR="./heat2d_adaptive_sem"
BW_CUR_PROF="./heat2d_adaptive_busywait_profile"
SM_CUR_PROF="./heat2d_adaptive_sem_profile"
BW_V2="./heat2d_adaptive_v2_busywait"
SM_V2="./heat2d_adaptive_v2_sem"
BW_V2_PROF="./heat2d_adaptive_v2_busywait_profile"
SM_V2_PROF="./heat2d_adaptive_v2_sem_profile"
PR_CAL="./heat2d_dependency_calibrate"
BW_WCAL="./heat2d_wait_calibrate_busywait"
SM_WCAL="./heat2d_wait_calibrate_semaphore"

cd "${SLURM_SUBMIT_DIR}"

export OMP_PLACES=cores
export OMP_PROC_BIND=spread
export OMP_DYNAMIC=FALSE
export HEAT2D_MAX_LEAD=2

get_param() {
    local key="$1"
    awk -F= -v k="$key" '
        $1 ~ "^[[:space:]]*" k "[[:space:]]*$" {
            gsub(/[[:space:]]/, "", $2); print $2; exit
        }' param.txt
}

[[ -f param.txt ]] || { echo "ERRO: param.txt ausente." >&2; exit 1; }
N_VALUE="$(get_param N)"
T_VALUE="$(get_param T)"
TILE_VALUE="$(get_param TILE)"
WRITE_OUTPUT_VALUE="$(get_param WRITE_OUTPUT | tr '[:upper:]' '[:lower:]')"

[[ "${N_VALUE}" == "8192" ]] || { echo "ERRO: N deve ser 8192; atual=${N_VALUE}." >&2; exit 2; }
[[ "${T_VALUE}" == "500" ]] || { echo "ERRO: T deve ser 500; atual=${T_VALUE}." >&2; exit 2; }
[[ "${TILE_VALUE}" == "32" ]] || { echo "ERRO: TILE deve ser 32; atual=${TILE_VALUE}." >&2; exit 2; }
case "${WRITE_OUTPUT_VALUE}" in
    0|off|false|no) ;;
    *) echo "ERRO: WRITE_OUTPUT deve ser 0." >&2; exit 2 ;;
esac

for p in "${THREADS_LIST[@]}"; do
    (( p >= 1 && p <= SLURM_CPUS_PER_TASK )) || {
        echo "ERRO: numero de threads invalido: ${p}." >&2; exit 2;
    }
done

mkdir -p "${OUTDIR}"
cp param.txt "${OUTDIR}/param.effective.txt"
lscpu > "${OUTDIR}/lscpu.txt" 2>&1 || true
lscpu -e=CPU,NODE,SOCKET,CORE,ONLINE > "${OUTDIR}/lscpu_extended.txt" 2>&1 || true

MAKE_JOBS="${MAKE_JOBS:-${SLURM_CPUS_PER_TASK}}"
echo "[build] core + profiles + Adaptive-V2"
make -j "${MAKE_JOBS}" core profiles adaptive_v2 profiles_v2

for exe in \
    "${BW_BASE}" "${SM_BASE}" \
    "${BW_CUR}" "${SM_CUR}" "${BW_CUR_PROF}" "${SM_CUR_PROF}" \
    "${BW_V2}" "${SM_V2}" "${BW_V2_PROF}" "${SM_V2_PROF}" \
    "${PR_CAL}" "${BW_WCAL}" "${SM_WCAL}"
do
    [[ -x "${exe}" ]] || { echo "ERRO: executavel ausente: ${exe}" >&2; exit 3; }
done

printf 'threads,backend,rep,order\n' > "${OUTDIR}/order.csv"

check_common_log() {
    local log="$1" threads="$2"
    grep -q '^Tempo[[:space:]]*:' "${log}" || { echo "ERRO: Tempo ausente em ${log}" >&2; exit 4; }
    grep -q '^Linf:' "${log}" || { echo "ERRO: Linf ausente em ${log}" >&2; exit 4; }
    grep -q "^Threads: ${threads}$" "${log}" || { echo "ERRO: Threads inesperado em ${log}" >&2; exit 4; }
}

run_base() {
    local backend="$1" threads="$2" log="$3"
    local exe
    if [[ "${backend}" == "busywait" ]]; then exe="${BW_BASE}"; else exe="${SM_BASE}"; fi
    OMP_NUM_THREADS="${threads}" "${exe}" > "${log}" 2>&1
    check_common_log "${log}" "${threads}"
}

run_current() {
    local backend="$1" policy="$2" threads="$3" cost="$4" waitcost="$5" log="$6" profile="$7"
    local exe recompute criticality
    if [[ "${profile}" == "1" ]]; then
        if [[ "${backend}" == "busywait" ]]; then exe="${BW_CUR_PROF}"; else exe="${SM_CUR_PROF}"; fi
    else
        if [[ "${backend}" == "busywait" ]]; then exe="${BW_CUR}"; else exe="${SM_CUR}"; fi
    fi
    case "${policy}" in
        wait)     recompute=0; criticality=0 ;;
        critical) recompute=1; criticality=1 ;;
        *) echo "ERRO interno: current policy ${policy}" >&2; exit 5 ;;
    esac

    env \
        OMP_NUM_THREADS="${threads}" \
        HEAT2D_ENABLE_PREDICT=0 \
        HEAT2D_ENABLE_RECOMPUTE="${recompute}" \
        HEAT2D_ENABLE_CRITICALITY="${criticality}" \
        HEAT2D_COST_FILE="${cost}" \
        HEAT2D_WAIT_COST_FILE="${waitcost}" \
        HEAT2D_PROGRESS_LAMBDA="${PROGRESS_LAMBDA}" \
        HEAT2D_PROGRESS_BOOTSTRAP_SAMPLES="${PROGRESS_BOOTSTRAP}" \
        "${exe}" > "${log}" 2>&1

    check_common_log "${log}" "${threads}"
    grep -q '^Adaptive predict: off$' "${log}" || { echo "ERRO: PREDICT deveria estar off em ${log}" >&2; exit 5; }
    if [[ "${policy}" == "critical" ]]; then
        grep -q '^Adaptive criticality-aware: on$' "${log}" || { echo "ERRO: criticality deveria estar on em ${log}" >&2; exit 5; }
    fi
}

run_v2() {
    local backend="$1" threads="$2" cost="$3" waitcost="$4" log="$5" profile="$6"
    local exe
    if [[ "${profile}" == "1" ]]; then
        if [[ "${backend}" == "busywait" ]]; then exe="${BW_V2_PROF}"; else exe="${SM_V2_PROF}"; fi
    else
        if [[ "${backend}" == "busywait" ]]; then exe="${BW_V2}"; else exe="${SM_V2}"; fi
    fi

    env \
        OMP_NUM_THREADS="${threads}" \
        HEAT2D_ENABLE_PREDICT=0 \
        HEAT2D_ENABLE_RECOMPUTE=1 \
        HEAT2D_ENABLE_PHI=1 \
        HEAT2D_COST_FILE="${cost}" \
        HEAT2D_WAIT_COST_FILE="${waitcost}" \
        HEAT2D_PROGRESS_LAMBDA="${PROGRESS_LAMBDA}" \
        HEAT2D_PROGRESS_BOOTSTRAP_SAMPLES="${PROGRESS_BOOTSTRAP}" \
        HEAT2D_ONLINE_BETA="${V2_ONLINE_BETA}" \
        HEAT2D_PROGRESS_BETA="${V2_PROGRESS_BETA}" \
        HEAT2D_PHI_BETA="${V2_PHI_BETA}" \
        HEAT2D_PHI_FLOOR="${V2_PHI_FLOOR}" \
        HEAT2D_PHI_RESAMPLE_PERIOD="${V2_PHI_RESAMPLE_PERIOD}" \
        HEAT2D_ONLINE_BOOTSTRAP_SAMPLES="${V2_ONLINE_BOOTSTRAP}" \
        "${exe}" > "${log}" 2>&1

    check_common_log "${log}" "${threads}"
    grep -q '^Adaptive predict: off$' "${log}" || { echo "ERRO: V2 PREDICT deveria estar off em ${log}" >&2; exit 6; }
    grep -q '^Adaptive V2 phi-aware: on$' "${log}" || { echo "ERRO: phi deveria estar on em ${log}" >&2; exit 6; }
}

run_policy() {
    local backend="$1" policy="$2" threads="$3" cost="$4" waitcost="$5" log="$6" profile="$7"
    case "${policy}" in
        base)     run_base "${backend}" "${threads}" "${log}" ;;
        wait)     run_current "${backend}" wait "${threads}" "${cost}" "${waitcost}" "${log}" "${profile}" ;;
        critical) run_current "${backend}" critical "${threads}" "${cost}" "${waitcost}" "${log}" "${profile}" ;;
        v2)       run_v2 "${backend}" "${threads}" "${cost}" "${waitcost}" "${log}" "${profile}" ;;
        *) echo "ERRO interno: policy=${policy}" >&2; exit 7 ;;
    esac
}

for threads in "${THREADS_LIST[@]}"; do
    pdir="${OUTDIR}/threads_$(printf '%03d' "${threads}")"
    caldir="${pdir}/calibration"
    mkdir -p "${pdir}" "${caldir}"

    cost="${caldir}/heat2d_cost_model.dat"
    bw_wait="${caldir}/heat2d_wait_cost_busywait.dat"
    sm_wait="${caldir}/heat2d_wait_cost_semaphore.dat"

    echo
    echo "============================================================"
    echo "threads=${threads} affinity=spread"
    echo "============================================================"

    echo "[calibration] P/R"
    env OMP_NUM_THREADS="${threads}" \
        HEAT2D_ENABLE_PREDICT=1 HEAT2D_ENABLE_RECOMPUTE=1 \
        HEAT2D_CALIBRATION_SAMPLES="${CAL_SAMPLES}" \
        HEAT2D_COST_FILE="${cost}" \
        "${PR_CAL}" > "${caldir}/pr.log" 2>&1

    echo "[calibration] WAIT busywait"
    env OMP_NUM_THREADS="${threads}" \
        HEAT2D_WAIT_CALIBRATION_SAMPLES="${WAIT_CAL_SAMPLES}" \
        HEAT2D_WAIT_COST_FILE="${bw_wait}" \
        "${BW_WCAL}" > "${caldir}/wait_busywait.log" 2>&1

    echo "[calibration] WAIT semaphore"
    env OMP_NUM_THREADS="${threads}" \
        HEAT2D_WAIT_CALIBRATION_SAMPLES="${WAIT_CAL_SAMPLES}" \
        HEAT2D_WAIT_COST_FILE="${sm_wait}" \
        "${SM_WCAL}" > "${caldir}/wait_semaphore.log" 2>&1

    for f in "${cost}" "${bw_wait}" "${sm_wait}"; do
        [[ -s "${f}" ]] || { echo "ERRO: calibracao nao gerou ${f}" >&2; exit 8; }
    done

    for backend in busywait semaphore; do
        if [[ "${backend}" == "busywait" ]]; then waitcost="${bw_wait}"; else waitcost="${sm_wait}"; fi

        echo "[warm-up] ${backend}"
        for ((w=1; w<=WARMUPS; ++w)); do
            for policy in base wait critical v2; do
                log="${pdir}/${backend}_${policy}_warmup_$(printf '%02d' "${w}").log"
                run_policy "${backend}" "${policy}" "${threads}" "${cost}" "${waitcost}" "${log}" 0
            done
        done

        echo "[timing] ${backend}: ${REPS} rotated repetitions"
        policies=(base wait critical v2)
        for ((rep=1; rep<=REPS; ++rep)); do
            shift=$(( (rep - 1) % 4 ))
            order=()
            for ((k=0; k<4; ++k)); do
                order+=("${policies[$(( (k + shift) % 4 ))]}")
            done
            for policy in "${order[@]}"; do
                log="${pdir}/${backend}_${policy}_timing_r$(printf '%02d' "${rep}").log"
                run_policy "${backend}" "${policy}" "${threads}" "${cost}" "${waitcost}" "${log}" 0
            done
            order_string="$(IFS='>'; echo "${order[*]}")"
            printf '%s,%s,%s,%s\n' "${threads}" "${backend}" "${rep}" "${order_string}" >> "${OUTDIR}/order.csv"
        done

        echo "[profile] ${backend}: WAIT, CRITICAL, V2"
        for policy in wait critical v2; do
            for ((rep=1; rep<=PROFILE_REPS; ++rep)); do
                log="${pdir}/${backend}_${policy}_profile_r$(printf '%02d' "${rep}").log"
                run_policy "${backend}" "${policy}" "${threads}" "${cost}" "${waitcost}" "${log}" 1
            done
        done
    done
done

python3 - "${OUTDIR}" "${REPS}" "${PROFILE_REPS}" <<'PY'
import csv, glob, math, os, random, re, statistics, sys

outdir=sys.argv[1]
reps=int(sys.argv[2])
profile_reps=int(sys.argv[3])

num=r'([-+0-9.eE]+)'

def parse_common(path):
    d={'time_s':None,'linf':None}
    with open(path, errors='replace') as f:
        for line in f:
            m=re.match(r'^Tempo\s*:\s*'+num,line)
            if m: d['time_s']=float(m.group(1))
            m=re.match(r'^Linf:\s*'+num,line)
            if m: d['linf']=float(m.group(1))
    if d['time_s'] is None or d['linf'] is None:
        raise RuntimeError(f'incomplete log: {path}')
    return d

def read_scalar(lines,label,default=0.0):
    prefix=label+':'
    for line in lines:
        if line.startswith(prefix):
            tail=line[len(prefix):].strip().split()
            if tail:
                try: return float(tail[0])
                except ValueError: pass
    return default

rows=[]
for pdir in sorted(glob.glob(os.path.join(outdir,'threads_*'))):
    threads=int(os.path.basename(pdir).split('_')[1])
    for backend in ('busywait','semaphore'):
        for policy in ('base','wait','critical','v2'):
            for rep in range(1,reps+1):
                path=os.path.join(pdir,f'{backend}_{policy}_timing_r{rep:02d}.log')
                x=parse_common(path)
                rows.append({'threads':threads,'backend':backend,'policy':policy,'rep':rep,**x})

times_path=os.path.join(outdir,'times.csv')
with open(times_path,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

def boot_ci(values, seed=20260903, B=20000):
    if not values: return (math.nan,math.nan)
    rng=random.Random(seed)
    n=len(values); meds=[]
    for _ in range(B):
        meds.append(statistics.median(values[rng.randrange(n)] for _ in range(n)))
    meds.sort()
    return meds[int(.025*B)], meds[min(B-1,int(.975*B))]

groups={}
for r in rows:
    groups.setdefault((r['threads'],r['backend'],r['policy']),[]).append(r)

summary=[]
for threads in sorted({r['threads'] for r in rows}):
    for backend in ('busywait','semaphore'):
        by={p:sorted(groups[(threads,backend,p)],key=lambda x:x['rep']) for p in ('base','wait','critical','v2')}
        med={p:statistics.median(x['time_s'] for x in rr) for p,rr in by.items()}
        out={'threads':threads,'backend':backend,'reps':reps}
        for p in ('base','wait','critical','v2'):
            out[f'median_{p}_s']=med[p]
            out[f'linf_{p}']=statistics.median(x['linf'] for x in by[p])
        comparisons=(('base','v2'),('wait','critical'),('wait','v2'),('critical','v2'))
        for a,b in comparisons:
            ratio=med[a]/med[b]
            paired=[x['time_s']/y['time_s'] for x,y in zip(by[a],by[b])]
            lo,hi=boot_ci(paired,seed=20260903+threads+sum(map(ord,backend+a+b)))
            tag=f'{a}_over_{b}'
            out[f'{tag}_ratio_of_medians']=ratio
            out[f'{tag}_gain_pct']=(ratio-1.0)*100.0
            out[f'{tag}_paired_median']=statistics.median(paired)
            out[f'{tag}_wins_b']=sum(1 for x in paired if x>1.0)
            out[f'{tag}_bootstrap95_lo']=lo
            out[f'{tag}_bootstrap95_hi']=hi
        summary.append(out)

summary_path=os.path.join(outdir,'summary.csv')
with open(summary_path,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)

profile=[]
for pdir in sorted(glob.glob(os.path.join(outdir,'threads_*'))):
    threads=int(os.path.basename(pdir).split('_')[1])
    for backend in ('busywait','semaphore'):
        for policy in ('wait','critical','v2'):
            for rep in range(1,profile_reps+1):
                path=os.path.join(pdir,f'{backend}_{policy}_profile_r{rep:02d}.log')
                common=parse_common(path)
                with open(path,errors='replace') as f: lines=f.readlines()
                profile.append({
                    'threads':threads,'backend':backend,'policy':policy,'rep':rep,
                    'time_s':common['time_s'],'linf':common['linf'],
                    'read':read_scalar(lines,'Adaptive actions READ'),
                    'recompute':read_scalar(lines,'Adaptive actions RECOMPUTE'),
                    'predict':read_scalar(lines,'Adaptive actions PREDICT'),
                    'wait':read_scalar(lines,'Adaptive actions WAIT'),
                    'lead_wait':read_scalar(lines,'Adaptive lead-guard waits'),
                    'lead_wait_ticks_sum':read_scalar(lines,'Lead-guard wait ticks sum'),
                    'progress_blocked':read_scalar(lines,'Progress-blocked RECOMPUTE'),
                    'critical_recompute':read_scalar(lines,'Critical RECOMPUTE'),
                    'noncritical_recompute_rejected':read_scalar(lines,'Noncritical RECOMPUTE rejected'),
                    'v2_recompute_rejected':read_scalar(lines,'V2 RECOMPUTE rejected by model'),
                    'v2_model_evaluations':read_scalar(lines,'V2 model evaluations'),
                    'v2_phi_mean':read_scalar(lines,'V2 phi mean'),
                    'v2_phi_min':read_scalar(lines,'V2 phi min'),
                    'v2_phi_max':read_scalar(lines,'V2 phi max'),
                    'v2_lhs_mean':read_scalar(lines,'V2 LHS mean'),
                    'v2_rhs_mean':read_scalar(lines,'V2 RHS mean'),
                    'v2_recompute_obs_samples':read_scalar(lines,'V2 observed RECOMPUTE samples'),
                    'v2_recompute_obs_mean':read_scalar(lines,'V2 observed RECOMPUTE mean'),
                    'v2_wait_obs_samples':read_scalar(lines,'V2 observed WAIT samples'),
                    'v2_wait_obs_mean':read_scalar(lines,'V2 observed WAIT mean'),
                    'penalty_samples':read_scalar(lines,'Progress penalty samples'),
                    'penalty_nonzero_samples':read_scalar(lines,'Progress penalty nonzero samples'),
                    'penalty_mean_ticks':read_scalar(lines,'Progress penalty observed mean'),
                })

profile_path=os.path.join(outdir,'profile_per_run.csv')
with open(profile_path,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(profile[0].keys())); w.writeheader(); w.writerows(profile)

pg={}
for r in profile: pg.setdefault((r['threads'],r['backend'],r['policy']),[]).append(r)
prof_summary=[]
for key in sorted(pg):
    rr=pg[key]; threads,backend,policy=key
    row={'threads':threads,'backend':backend,'policy':policy,'reps':len(rr)}
    for field in profile[0]:
        if field in ('threads','backend','policy','rep'): continue
        row['median_'+field]=statistics.median(float(x[field]) for x in rr)
    prof_summary.append(row)

prof_summary_path=os.path.join(outdir,'profile_summary.csv')
with open(prof_summary_path,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(prof_summary[0].keys())); w.writeheader(); w.writerows(prof_summary)

print('\n=== ADAPTIVE-V2 COMPARISON / SPREAD ===')
print(' p backend     BASE      WAIT      CRIT       V2    WAIT/V2   BASE/V2  CRIT/V2')
for r in summary:
    print(f"{r['threads']:3d} {r['backend']:9s} "
          f"{r['median_base_s']:8.5f} {r['median_wait_s']:8.5f} "
          f"{r['median_critical_s']:8.5f} {r['median_v2_s']:8.5f} "
          f"{r['wait_over_v2_ratio_of_medians']:9.4f} "
          f"{r['base_over_v2_ratio_of_medians']:9.4f} "
          f"{r['critical_over_v2_ratio_of_medians']:9.4f}")
print('\nFiles:')
for p in (times_path,summary_path,profile_path,prof_summary_path): print(' ',p)
PY

{
    echo "date=$(date --iso-8601=seconds 2>/dev/null || date)"
    echo "job_id=${SLURM_JOB_ID}"
    echo "node=${SLURM_NODELIST}"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "experiment=adaptive_v2_compare"
    echo "threads=${THREADS_LIST[*]}"
    echo "N=${N_VALUE}"
    echo "T=${T_VALUE}"
    echo "TILE=${TILE_VALUE}"
    echo "REPS=${REPS}"
    echo "PROFILE_REPS=${PROFILE_REPS}"
    echo "CAL_SAMPLES=${CAL_SAMPLES}"
    echo "WAIT_CAL_SAMPLES=${WAIT_CAL_SAMPLES}"
    echo "OMP_PLACES=${OMP_PLACES}"
    echo "OMP_PROC_BIND=${OMP_PROC_BIND}"
    echo "PREDICT=0"
    echo "HEAT2D_MAX_LEAD=${HEAT2D_MAX_LEAD}"
    echo "PROGRESS_LAMBDA=${PROGRESS_LAMBDA}"
    echo "PROGRESS_BOOTSTRAP=${PROGRESS_BOOTSTRAP}"
    echo "V2_ONLINE_BETA=${V2_ONLINE_BETA}"
    echo "V2_PROGRESS_BETA=${V2_PROGRESS_BETA}"
    echo "V2_PHI_BETA=${V2_PHI_BETA}"
    echo "V2_PHI_FLOOR=${V2_PHI_FLOOR}"
    echo "V2_PHI_RESAMPLE_PERIOD=${V2_PHI_RESAMPLE_PERIOD}"
    echo "V2_ONLINE_BOOTSTRAP=${V2_ONLINE_BOOTSTRAP}"
} > "${OUTDIR}/run_metadata.txt"

sha256sum \
    heat2d_explicit_omp_busywait_nobarrier_nofs_adaptive.cpp \
    heat2d_explicit_omp_sem_nobarrier_nofs_adaptive.cpp \
    heat2d_explicit_omp_busywait_nobarrier_nofs_adaptive_v2.cpp \
    heat2d_explicit_omp_sem_nobarrier_nofs_adaptive_v2.cpp \
    > "${OUTDIR}/source_sha256.txt"

echo
echo "============================================================"
echo "Adaptive-V2 comparison complete."
echo "  ${OUTDIR}/summary.csv"
echo "  ${OUTDIR}/profile_summary.csv"
echo "============================================================"
