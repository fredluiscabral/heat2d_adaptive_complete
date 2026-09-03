#!/bin/bash
#SBATCH --job-name=heat2d_sd_full_sweep
#SBATCH --partition=lncc-cpu_amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=heat2d_sd_full_sweep-%j.out
#SBATCH --error=heat2d_sd_full_sweep-%j.err

set -euo pipefail

# =============================================================================
# SDumont II — sweep completo do controlador RECOMPUTE/WAIT
#
# Objetivo:
#   mapear, em uma unica submissao, o efeito de:
#     - numero de threads;
#     - backend de espera (busy-wait / semaphore);
#     - infraestrutura adaptativa;
#     - progress-awareness;
#     - criticality-awareness.
#
# PREDICT permanece desligado durante as medicoes.
#
# Matriz padrao:
#   p = 32 48 64 80 96 112 128 160 192
#   backend = busywait semaphore
#   policy  = base wait progress critical
#
# Para cada p:
#   - calibra P/R uma vez;
#   - calibra WAIT separadamente para busy-wait e semaphore;
#   - 1 warm-up por caso;
#   - 10 repeticoes temporizadas, com ordem rotacionada;
#   - 3 repeticoes de profiling para wait/progress/critical.
#
# Parametros numericos exigidos:
#   N=8192, T=500, TILE=32, WRITE_OUTPUT=0.
#
# Saida:
#   results/sdumont_criticality_sweep_complete/
#       times.csv
#       profile_per_run.csv
#       profile_summary.csv
#       summary.csv
#       run_metadata.txt
#       threads_XXX/*.log
#       threads_XXX/*.dat
# =============================================================================

REPS="${REPS:-10}"
PROFILE_REPS="${PROFILE_REPS:-3}"
WARMUPS="${WARMUPS:-1}"

THREADS_LIST_STR="${THREADS_LIST:-32 48 64 80 96 112 128 160 192}"
read -r -a THREADS_LIST <<< "${THREADS_LIST_STR}"

CAL_SAMPLES="${CAL_SAMPLES:-32}"
WAIT_CAL_SAMPLES="${WAIT_CAL_SAMPLES:-32}"
PROGRESS_LAMBDA="${PROGRESS_LAMBDA:-1.0}"
PROGRESS_BOOTSTRAP="${PROGRESS_BOOTSTRAP:-8}"

OUTDIR="${OUTDIR:-results/sdumont_criticality_sweep_complete}"

BW_BASE="./heat2d_explicit_omp_busywait_nobarrier_nofs"
SM_BASE="./heat2d_explicit_omp_sem_nobarrier_nofs"

BW_REF="./heat2d_adaptive_busywait"
SM_REF="./heat2d_adaptive_sem"

BW_PROF="./heat2d_adaptive_busywait_profile"
SM_PROF="./heat2d_adaptive_sem_profile"

PR_CAL="./heat2d_dependency_calibrate"
BW_WCAL="./heat2d_wait_calibrate_busywait"
SM_WCAL="./heat2d_wait_calibrate_semaphore"

cd "${SLURM_SUBMIT_DIR}"

export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_DYNAMIC=FALSE
export HEAT2D_MAX_LEAD=2

get_param() {
    local key="$1"
    awk -F= -v k="$key" '
        $1 ~ "^[[:space:]]*" k "[[:space:]]*$" {
            gsub(/[[:space:]]/, "", $2)
            print $2
            exit
        }
    ' param.txt
}

[[ -f param.txt ]] || {
    echo "ERRO: param.txt nao encontrado em ${SLURM_SUBMIT_DIR}." >&2
    exit 1
}

N_VALUE="$(get_param N)"
T_VALUE="$(get_param T)"
TILE_VALUE="$(get_param TILE)"
WRITE_OUTPUT_VALUE="$(get_param WRITE_OUTPUT | tr '[:upper:]' '[:lower:]')"

[[ "${N_VALUE}" == "8192" ]] || {
    echo "ERRO: este sweep exige N=8192; atual=${N_VALUE:-ausente}." >&2
    exit 2
}
[[ "${T_VALUE}" == "500" ]] || {
    echo "ERRO: este sweep exige T=500; atual=${T_VALUE:-ausente}." >&2
    exit 3
}
[[ "${TILE_VALUE}" == "32" ]] || {
    echo "ERRO: este sweep exige TILE=32; atual=${TILE_VALUE:-ausente}." >&2
    exit 4
}
case "${WRITE_OUTPUT_VALUE}" in
    0|off|false|no) ;;
    *)
        echo "ERRO: este sweep exige WRITE_OUTPUT=0; atual=${WRITE_OUTPUT_VALUE:-ausente}." >&2
        exit 5
        ;;
esac

for threads in "${THREADS_LIST[@]}"; do
    if (( threads < 1 || threads > SLURM_CPUS_PER_TASK )); then
        echo "ERRO: THREADS_LIST contem valor invalido: ${threads}." >&2
        exit 6
    fi
done

mkdir -p "${OUTDIR}"

# Nao apaga o diretorio inteiro, mas remove produtos de uma execucao anterior
# com o mesmo OUTDIR para impedir mistura silenciosa de resultados.
rm -f "${OUTDIR}/times.csv" \
      "${OUTDIR}/profile_per_run.csv" \
      "${OUTDIR}/profile_summary.csv" \
      "${OUTDIR}/summary.csv" \
      "${OUTDIR}/run_metadata.txt" 2>/dev/null || true

for threads in "${THREADS_LIST[@]}"; do
    tag="$(printf '%03d' "${threads}")"
    pdir="${OUTDIR}/threads_${tag}"
    mkdir -p "${pdir}"
    rm -f "${pdir}"/*.log "${pdir}"/*.dat 2>/dev/null || true
done

echo "============================================================"
echo "SDUMONT II — COMPLETE CRITICALITY/PROGRESS SWEEP"
echo "============================================================"
echo "Job ID              : ${SLURM_JOB_ID}"
echo "Node                : ${SLURM_NODELIST}"
echo "CPUs/task           : ${SLURM_CPUS_PER_TASK}"
echo "Max wall time       : ${SLURM_TIMELIMIT:-04:00:00}"
echo "Threads             : ${THREADS_LIST[*]}"
echo "Timed repetitions   : ${REPS}"
echo "Profile repetitions : ${PROFILE_REPS}"
echo "Warm-ups/case       : ${WARMUPS}"
echo "P/R cal samples     : ${CAL_SAMPLES}"
echo "WAIT cal samples    : ${WAIT_CAL_SAMPLES}"
echo "Progress lambda     : ${PROGRESS_LAMBDA}"
echo "Progress bootstrap  : ${PROGRESS_BOOTSTRAP}"
echo "N/T/TILE            : ${N_VALUE}/${T_VALUE}/${TILE_VALUE}"
echo "PREDICT             : OFF during measurements"
echo "OMP_PLACES          : ${OMP_PLACES}"
echo "OMP_PROC_BIND       : ${OMP_PROC_BIND}"
echo "Output              : ${OUTDIR}"
echo "============================================================"

{
    echo "job_id=${SLURM_JOB_ID}"
    echo "node=${SLURM_NODELIST}"
    echo "cpus_per_task=${SLURM_CPUS_PER_TASK}"
    echo "slurm_time_limit=${SLURM_TIMELIMIT:-04:00:00}"
    echo "threads=${THREADS_LIST[*]}"
    echo "reps=${REPS}"
    echo "profile_reps=${PROFILE_REPS}"
    echo "warmups=${WARMUPS}"
    echo "cal_samples=${CAL_SAMPLES}"
    echo "wait_cal_samples=${WAIT_CAL_SAMPLES}"
    echo "progress_lambda=${PROGRESS_LAMBDA}"
    echo "progress_bootstrap=${PROGRESS_BOOTSTRAP}"
    echo "N=${N_VALUE}"
    echo "T=${T_VALUE}"
    echo "TILE=${TILE_VALUE}"
    echo "predict=0"
    echo "omp_places=${OMP_PLACES}"
    echo "omp_proc_bind=${OMP_PROC_BIND}"
    echo "date_start=$(date --iso-8601=seconds)"
    echo "uname=$(uname -a)"
} > "${OUTDIR}/run_metadata.txt"

echo
echo "================ BUILD ======================================="
make -j "${SLURM_CPUS_PER_TASK}" core profiles
make -j "${SLURM_CPUS_PER_TASK}" \
    heat2d_explicit_omp_busywait_nobarrier_nofs \
    heat2d_explicit_omp_sem_nobarrier_nofs

for exe in \
    "${BW_BASE}" "${SM_BASE}" \
    "${BW_REF}" "${SM_REF}" \
    "${BW_PROF}" "${SM_PROF}" \
    "${PR_CAL}" "${BW_WCAL}" "${SM_WCAL}"
do
    [[ -x "${exe}" ]] || {
        echo "ERRO: executavel ausente: ${exe}" >&2
        exit 7
    }
done

check_common_output() {
    local log="$1"

    grep -q '^Tempo[[:space:]]*:' "${log}" || {
        echo "ERRO: Tempo ausente em ${log}" >&2
        exit 8
    }
    grep -q '^Linf:' "${log}" || {
        echo "ERRO: Linf ausente em ${log}" >&2
        exit 9
    }
}

run_base() {
    local exe="$1"
    local backend="$2"
    local threads="$3"
    local rep="$4"
    local pdir="$5"
    local kind="$6"

    local log="${pdir}/${backend}_base_${kind}_r$(printf '%02d' "${rep}").log"

    "${exe}" > "${log}" 2>&1
    check_common_output "${log}"
}

run_adaptive() {
    local exe="$1"
    local backend="$2"
    local policy="$3"
    local threads="$4"
    local rep="$5"
    local pdir="$6"
    local cost="$7"
    local waitcost="$8"
    local kind="$9"

    local recompute criticality
    case "${policy}" in
        wait)
            recompute=0
            criticality=0
            ;;
        progress)
            recompute=1
            criticality=0
            ;;
        critical)
            recompute=1
            criticality=1
            ;;
        *)
            echo "ERRO interno: policy invalida: ${policy}" >&2
            exit 10
            ;;
    esac

    local log="${pdir}/${backend}_${policy}_${kind}_r$(printf '%02d' "${rep}").log"

    HEAT2D_ENABLE_PREDICT=0 \
    HEAT2D_ENABLE_RECOMPUTE="${recompute}" \
    HEAT2D_ENABLE_CRITICALITY="${criticality}" \
    HEAT2D_COST_FILE="${cost}" \
    HEAT2D_WAIT_COST_FILE="${waitcost}" \
    HEAT2D_PROGRESS_LAMBDA="${PROGRESS_LAMBDA}" \
    HEAT2D_PROGRESS_BOOTSTRAP_SAMPLES="${PROGRESS_BOOTSTRAP}" \
        "${exe}" > "${log}" 2>&1

    check_common_output "${log}"

    grep -q '^Adaptive predict: off$' "${log}" || {
        echo "ERRO: PREDICT deveria estar OFF em ${log}" >&2
        exit 11
    }

    if [[ "${recompute}" == "1" ]]; then
        grep -q '^Adaptive recompute: on$' "${log}" || {
            echo "ERRO: RECOMPUTE deveria estar ON em ${log}" >&2
            exit 12
        }
    else
        grep -q '^Adaptive recompute: off$' "${log}" || {
            echo "ERRO: RECOMPUTE deveria estar OFF em ${log}" >&2
            exit 13
        }
    fi

    if [[ "${criticality}" == "1" ]]; then
        grep -q '^Adaptive criticality-aware: on$' "${log}" || {
            echo "ERRO: criticality deveria estar ON em ${log}" >&2
            exit 14
        }
    else
        grep -q '^Adaptive criticality-aware: off$' "${log}" || {
            echo "ERRO: criticality deveria estar OFF em ${log}" >&2
            exit 15
        }
    fi
}

run_case() {
    local backend="$1"
    local policy="$2"
    local threads="$3"
    local rep="$4"
    local pdir="$5"
    local cost="$6"
    local bw_wait="$7"
    local sm_wait="$8"
    local kind="$9"

    case "${backend}:${policy}:${kind}" in
        busywait:base:*)
            run_base "${BW_BASE}" busywait "${threads}" "${rep}" "${pdir}" "${kind}"
            ;;
        semaphore:base:*)
            run_base "${SM_BASE}" semaphore "${threads}" "${rep}" "${pdir}" "${kind}"
            ;;
        busywait:*:ref|busywait:*:warmup)
            run_adaptive "${BW_REF}" busywait "${policy}" "${threads}" "${rep}" \
                         "${pdir}" "${cost}" "${bw_wait}" "${kind}"
            ;;
        semaphore:*:ref|semaphore:*:warmup)
            run_adaptive "${SM_REF}" semaphore "${policy}" "${threads}" "${rep}" \
                         "${pdir}" "${cost}" "${sm_wait}" "${kind}"
            ;;
        busywait:*:prof)
            run_adaptive "${BW_PROF}" busywait "${policy}" "${threads}" "${rep}" \
                         "${pdir}" "${cost}" "${bw_wait}" "${kind}"
            ;;
        semaphore:*:prof)
            run_adaptive "${SM_PROF}" semaphore "${policy}" "${threads}" "${rep}" \
                         "${pdir}" "${cost}" "${sm_wait}" "${kind}"
            ;;
        *)
            echo "ERRO interno: caso desconhecido ${backend}:${policy}:${kind}" >&2
            exit 16
            ;;
    esac
}

# Ordem deliberadamente intercalada entre backends.
TIMED_CASES=(
    "busywait:base"
    "semaphore:base"
    "busywait:wait"
    "semaphore:wait"
    "busywait:progress"
    "semaphore:progress"
    "busywait:critical"
    "semaphore:critical"
)

PROFILE_CASES=(
    "busywait:wait"
    "semaphore:wait"
    "busywait:progress"
    "semaphore:progress"
    "busywait:critical"
    "semaphore:critical"
)

for threads in "${THREADS_LIST[@]}"; do
    export OMP_NUM_THREADS="${threads}"

    tag="$(printf '%03d' "${threads}")"
    pdir="${OUTDIR}/threads_${tag}"

    cost="${pdir}/heat2d_cost_model.dat"
    bw_wait="${pdir}/heat2d_wait_cost_busywait.dat"
    sm_wait="${pdir}/heat2d_wait_cost_semaphore.dat"

    echo
    echo "============================================================"
    echo "THREADS = ${threads}"
    echo "============================================================"

    echo "[calibration] PREDICT/RECOMPUTE"
    HEAT2D_ENABLE_PREDICT=1 \
    HEAT2D_ENABLE_RECOMPUTE=1 \
    HEAT2D_CALIBRATION_SAMPLES="${CAL_SAMPLES}" \
    HEAT2D_COST_FILE="${cost}" \
        "${PR_CAL}" > "${pdir}/calibration_pr.log" 2>&1

    [[ -s "${cost}" ]] || {
        echo "ERRO: modelo P/R nao gerado para p=${threads}." >&2
        exit 17
    }

    echo "[calibration] WAIT busy-wait"
    HEAT2D_WAIT_CALIBRATION_SAMPLES="${WAIT_CAL_SAMPLES}" \
    HEAT2D_WAIT_COST_FILE="${bw_wait}" \
        "${BW_WCAL}" > "${pdir}/calibration_wait_busywait.log" 2>&1

    [[ -s "${bw_wait}" ]] || {
        echo "ERRO: modelo WAIT busy-wait nao gerado para p=${threads}." >&2
        exit 18
    }

    echo "[calibration] WAIT semaphore"
    HEAT2D_WAIT_CALIBRATION_SAMPLES="${WAIT_CAL_SAMPLES}" \
    HEAT2D_WAIT_COST_FILE="${sm_wait}" \
        "${SM_WCAL}" > "${pdir}/calibration_wait_semaphore.log" 2>&1

    [[ -s "${sm_wait}" ]] || {
        echo "ERRO: modelo WAIT semaphore nao gerado para p=${threads}." >&2
        exit 19
    }

    echo "[warm-up] ${WARMUPS} x 8 cases"
    for ((w=1; w<=WARMUPS; ++w)); do
        shift=$(( (w - 1) % ${#TIMED_CASES[@]} ))
        for ((k=0; k<${#TIMED_CASES[@]}; ++k)); do
            idx=$(( (k + shift) % ${#TIMED_CASES[@]} ))
            IFS=: read -r backend policy <<< "${TIMED_CASES[$idx]}"
            run_case "${backend}" "${policy}" "${threads}" "${w}" \
                     "${pdir}" "${cost}" "${bw_wait}" "${sm_wait}" warmup
        done
    done

    echo "[timing] ${REPS} rotated blocks"
    for ((rep=1; rep<=REPS; ++rep)); do
        shift=$(( (rep - 1) % ${#TIMED_CASES[@]} ))

        for ((k=0; k<${#TIMED_CASES[@]}; ++k)); do
            idx=$(( (k + shift) % ${#TIMED_CASES[@]} ))
            IFS=: read -r backend policy <<< "${TIMED_CASES[$idx]}"

            run_case "${backend}" "${policy}" "${threads}" "${rep}" \
                     "${pdir}" "${cost}" "${bw_wait}" "${sm_wait}" ref
        done

        echo "threads=${threads} timing rep=${rep}/${REPS} complete"
    done

    echo "[profiling] ${PROFILE_REPS} rotated blocks"
    for ((rep=1; rep<=PROFILE_REPS; ++rep)); do
        shift=$(( (rep - 1) % ${#PROFILE_CASES[@]} ))

        for ((k=0; k<${#PROFILE_CASES[@]}; ++k)); do
            idx=$(( (k + shift) % ${#PROFILE_CASES[@]} ))
            IFS=: read -r backend policy <<< "${PROFILE_CASES[$idx]}"

            run_case "${backend}" "${policy}" "${threads}" "${rep}" \
                     "${pdir}" "${cost}" "${bw_wait}" "${sm_wait}" prof
        done

        echo "threads=${threads} profile rep=${rep}/${PROFILE_REPS} complete"
    done
done

python3 - "${OUTDIR}" "${REPS}" "${PROFILE_REPS}" <<'PY'
import csv
import glob
import math
import os
import random
import re
import statistics
import sys

outdir = sys.argv[1]
reps = int(sys.argv[2])
profile_reps = int(sys.argv[3])

TIME_RE = re.compile(r'^Tempo\s*:\s*([-+0-9.eE]+)')
LINF_RE = re.compile(r'^Linf:\s*([-+0-9.eE]+)')

def parse_time_linf(path):
    time_s = None
    linf = None
    with open(path, errors='replace') as f:
        for line in f:
            m = TIME_RE.match(line)
            if m:
                time_s = float(m.group(1))
            m = LINF_RE.match(line)
            if m:
                linf = float(m.group(1))
    if time_s is None or linf is None:
        raise RuntimeError(f'incomplete timing log: {path}')
    return time_s, linf

def parse_profile(path):
    time_s = None
    linf = None
    vals = {
        'recompute': 0.0,
        'wait': 0.0,
        'lead_wait': 0.0,
        'progress_blocked': 0.0,
        'penalty_samples': 0.0,
        'penalty_nonzero_samples': 0.0,
        'penalty_mean_ticks': 0.0,
        'penalty_nonzero_fraction': 0.0,
        'critical_recompute': 0.0,
        'noncritical_recompute_rejected': 0.0,
        'criticality_forced_wait': 0.0,
    }

    fields = {
        'Adaptive actions RECOMPUTE': 'recompute',
        'Adaptive actions WAIT': 'wait',
        'Adaptive lead-guard waits': 'lead_wait',
        'Progress-blocked RECOMPUTE': 'progress_blocked',
        'Progress penalty samples': 'penalty_samples',
        'Progress penalty nonzero samples': 'penalty_nonzero_samples',
        'Progress penalty observed mean': 'penalty_mean_ticks',
        'Progress penalty nonzero fraction': 'penalty_nonzero_fraction',
        'Critical RECOMPUTE': 'critical_recompute',
        'Noncritical RECOMPUTE rejected': 'noncritical_recompute_rejected',
        'Criticality-forced WAIT': 'criticality_forced_wait',
    }

    with open(path, errors='replace') as f:
        for line in f:
            m = TIME_RE.match(line)
            if m:
                time_s = float(m.group(1))
            m = LINF_RE.match(line)
            if m:
                linf = float(m.group(1))

            for prefix, key in fields.items():
                if line.startswith(prefix + ':'):
                    tok = line.split(':', 1)[1].strip().split()[0]
                    vals[key] = float(tok)
                    break

    if time_s is None or linf is None:
        raise RuntimeError(f'incomplete profile log: {path}')

    vals['profile_time_s'] = time_s
    vals['linf'] = linf
    return vals

# -------------------------------------------------------------------------
# Timing rows
# -------------------------------------------------------------------------
timing_rows = []

for path in sorted(glob.glob(os.path.join(outdir, 'threads_*', '*_ref_r*.log'))):
    base = os.path.basename(path)
    m = re.fullmatch(
        r'(busywait|semaphore)_(base|wait|progress|critical)_ref_r(\d+)\.log',
        base
    )
    if not m:
        continue

    backend, policy, rep = m.group(1), m.group(2), int(m.group(3))
    threads = int(os.path.basename(os.path.dirname(path)).split('_')[-1])
    time_s, linf = parse_time_linf(path)

    timing_rows.append({
        'threads': threads,
        'backend': backend,
        'policy': policy,
        'rep': rep,
        'time_s': time_s,
        'linf': linf,
    })

if not timing_rows:
    raise SystemExit('No timing rows found')

expected_timing = {}
for r in timing_rows:
    key = (r['threads'], r['backend'], r['policy'])
    expected_timing.setdefault(key, []).append(r)

for key, rr in expected_timing.items():
    if len(rr) != reps:
        raise SystemExit(f'Expected {reps} timing reps for {key}, got {len(rr)}')

times_path = os.path.join(outdir, 'times.csv')
with open(times_path, 'w', newline='') as f:
    cols = ['threads', 'backend', 'policy', 'rep', 'time_s', 'linf']
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(sorted(
        timing_rows,
        key=lambda r: (r['threads'], r['backend'], r['policy'], r['rep'])
    ))

# -------------------------------------------------------------------------
# Profile rows
# -------------------------------------------------------------------------
profile_rows = []

for path in sorted(glob.glob(os.path.join(outdir, 'threads_*', '*_prof_r*.log'))):
    base = os.path.basename(path)
    m = re.fullmatch(
        r'(busywait|semaphore)_(wait|progress|critical)_prof_r(\d+)\.log',
        base
    )
    if not m:
        continue

    backend, policy, rep = m.group(1), m.group(2), int(m.group(3))
    threads = int(os.path.basename(os.path.dirname(path)).split('_')[-1])
    vals = parse_profile(path)
    vals.update({
        'threads': threads,
        'backend': backend,
        'policy': policy,
        'rep': rep,
    })
    profile_rows.append(vals)

expected_profile = {}
for r in profile_rows:
    key = (r['threads'], r['backend'], r['policy'])
    expected_profile.setdefault(key, []).append(r)

for key, rr in expected_profile.items():
    if len(rr) != profile_reps:
        raise SystemExit(
            f'Expected {profile_reps} profile reps for {key}, got {len(rr)}'
        )

profile_cols = [
    'threads', 'backend', 'policy', 'rep',
    'profile_time_s', 'linf',
    'recompute', 'wait', 'lead_wait', 'progress_blocked',
    'penalty_samples', 'penalty_nonzero_samples',
    'penalty_mean_ticks', 'penalty_nonzero_fraction',
    'critical_recompute', 'noncritical_recompute_rejected',
    'criticality_forced_wait',
]

profile_per_run_path = os.path.join(outdir, 'profile_per_run.csv')
with open(profile_per_run_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=profile_cols)
    w.writeheader()
    w.writerows(sorted(
        profile_rows,
        key=lambda r: (r['threads'], r['backend'], r['policy'], r['rep'])
    ))

profile_summary = []
for key in sorted(expected_profile):
    rr = expected_profile[key]
    med = lambda field: statistics.median(float(r[field]) for r in rr)
    profile_summary.append({
        'threads': key[0],
        'backend': key[1],
        'policy': key[2],
        'reps': len(rr),
        'median_profile_time_s': med('profile_time_s'),
        'median_recompute': med('recompute'),
        'median_wait': med('wait'),
        'median_lead_wait': med('lead_wait'),
        'median_progress_blocked': med('progress_blocked'),
        'median_penalty_samples': med('penalty_samples'),
        'median_penalty_nonzero_samples': med('penalty_nonzero_samples'),
        'median_penalty_mean_ticks': med('penalty_mean_ticks'),
        'median_penalty_nonzero_fraction': med('penalty_nonzero_fraction'),
        'median_critical_recompute': med('critical_recompute'),
        'median_noncritical_recompute_rejected': med('noncritical_recompute_rejected'),
        'median_criticality_forced_wait': med('criticality_forced_wait'),
        'median_linf': med('linf'),
    })

profile_summary_path = os.path.join(outdir, 'profile_summary.csv')
with open(profile_summary_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(profile_summary[0].keys()))
    w.writeheader()
    w.writerows(profile_summary)

# -------------------------------------------------------------------------
# Comparison summary with paired statistics
# -------------------------------------------------------------------------
def median(xs):
    return statistics.median(xs)

def paired_values(rows_by_policy, numerator, denominator):
    num = {int(r['rep']): float(r['time_s']) for r in rows_by_policy[numerator]}
    den = {int(r['rep']): float(r['time_s']) for r in rows_by_policy[denominator]}
    common = sorted(set(num) & set(den))
    if len(common) != reps:
        raise RuntimeError(
            f'paired comparison {numerator}/{denominator}: '
            f'expected {reps} common reps, got {len(common)}'
        )
    return [num[i] / den[i] for i in common]

def bootstrap_median_ci(values, seed, nboot=20000, alpha=0.05):
    rng = random.Random(seed)
    n = len(values)
    boots = []
    for _ in range(nboot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boots.append(statistics.median(sample))
    boots.sort()

    lo_idx = max(0, int((alpha / 2.0) * nboot))
    hi_idx = min(nboot - 1, int((1.0 - alpha / 2.0) * nboot) - 1)
    return boots[lo_idx], boots[hi_idx]

grouped = {}
for r in timing_rows:
    grouped.setdefault((r['threads'], r['backend']), {}).setdefault(
        r['policy'], []
    ).append(r)

summary_rows = []

for (threads, backend) in sorted(grouped):
    byp = grouped[(threads, backend)]
    required = {'base', 'wait', 'progress', 'critical'}
    if set(byp) != required:
        raise RuntimeError(
            f'{threads}/{backend}: policies={sorted(byp)}, expected={sorted(required)}'
        )

    med = {
        p: median([float(r['time_s']) for r in byp[p]])
        for p in required
    }

    wait_over_base = paired_values(byp, 'wait', 'base')
    wait_over_progress = paired_values(byp, 'wait', 'progress')
    wait_over_critical = paired_values(byp, 'wait', 'critical')
    progress_over_critical = paired_values(byp, 'progress', 'critical')

    seed_base = threads * 100 + (1 if backend == 'busywait' else 2)

    ci_base = bootstrap_median_ci(wait_over_base, seed_base + 10)
    ci_prog = bootstrap_median_ci(wait_over_progress, seed_base + 20)
    ci_crit = bootstrap_median_ci(wait_over_critical, seed_base + 30)
    ci_cp = bootstrap_median_ci(progress_over_critical, seed_base + 40)

    summary_rows.append({
        'threads': threads,
        'backend': backend,
        'reps': reps,

        'median_base_s': med['base'],
        'median_wait_s': med['wait'],
        'median_progress_s': med['progress'],
        'median_critical_s': med['critical'],

        'wait_overhead_vs_base_pct':
            100.0 * (med['wait'] / med['base'] - 1.0),

        'progress_speedup_vs_wait_ratio_of_medians':
            med['wait'] / med['progress'],
        'critical_speedup_vs_wait_ratio_of_medians':
            med['wait'] / med['critical'],
        'critical_speedup_vs_progress_ratio_of_medians':
            med['progress'] / med['critical'],

        'median_paired_wait_over_base':
            median(wait_over_base),
        'bootstrap95_wait_over_base_lo':
            ci_base[0],
        'bootstrap95_wait_over_base_hi':
            ci_base[1],

        'median_paired_wait_over_progress':
            median(wait_over_progress),
        'progress_wins_vs_wait':
            sum(1 for x in wait_over_progress if x > 1.0),
        'bootstrap95_wait_over_progress_lo':
            ci_prog[0],
        'bootstrap95_wait_over_progress_hi':
            ci_prog[1],

        'median_paired_wait_over_critical':
            median(wait_over_critical),
        'critical_wins_vs_wait':
            sum(1 for x in wait_over_critical if x > 1.0),
        'bootstrap95_wait_over_critical_lo':
            ci_crit[0],
        'bootstrap95_wait_over_critical_hi':
            ci_crit[1],

        'median_paired_progress_over_critical':
            median(progress_over_critical),
        'critical_wins_vs_progress':
            sum(1 for x in progress_over_critical if x > 1.0),
        'bootstrap95_progress_over_critical_lo':
            ci_cp[0],
        'bootstrap95_progress_over_critical_hi':
            ci_cp[1],

        'linf_base': median([float(r['linf']) for r in byp['base']]),
        'linf_wait': median([float(r['linf']) for r in byp['wait']]),
        'linf_progress': median([float(r['linf']) for r in byp['progress']]),
        'linf_critical': median([float(r['linf']) for r in byp['critical']]),
    })

summary_path = os.path.join(outdir, 'summary.csv')
with open(summary_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
    w.writeheader()
    w.writerows(summary_rows)

print()
print('=== FINAL FILES ===')
print('Timing per run      :', times_path)
print('Profile per run     :', profile_per_run_path)
print('Profile summary     :', profile_summary_path)
print('Comparison summary  :', summary_path)

print()
print('=== CRITICAL vs WAIT ===')
for r in summary_rows:
    print(
        f"p={r['threads']:3d} {r['backend']:9s} "
        f"S_med={r['critical_speedup_vs_wait_ratio_of_medians']:.5f} "
        f"S_paired={r['median_paired_wait_over_critical']:.5f} "
        f"wins={r['critical_wins_vs_wait']}/{r['reps']} "
        f"CI95=[{r['bootstrap95_wait_over_critical_lo']:.5f},"
        f"{r['bootstrap95_wait_over_critical_hi']:.5f}]"
    )
PY

{
    echo "date_end=$(date --iso-8601=seconds)"
} >> "${OUTDIR}/run_metadata.txt"

echo
echo "============================================================"
echo "SDumont complete sweep finished."
echo "Main files:"
echo "  ${OUTDIR}/times.csv"
echo "  ${OUTDIR}/profile_per_run.csv"
echo "  ${OUTDIR}/profile_summary.csv"
echo "  ${OUTDIR}/summary.csv"
echo "  ${OUTDIR}/run_metadata.txt"
echo "============================================================"
