#!/bin/bash
#SBATCH --job-name=heat2d_sd_affinity
#SBATCH --partition=lncc-cpu_amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=heat2d_sd_affinity-%j.out
#SBATCH --error=heat2d_sd_affinity-%j.err

set -euo pipefail

# =============================================================================
# SDumont II — experimento de afinidade OpenMP
#
# Objetivo:
#   testar se a politica de afinidade altera o regime de desempenho do
#   controlador CRITICAL em relacao a WAIT-only.
#
# Matriz padrao:
#   threads  = 64 96 160
#   affinity = close spread
#   backend  = busywait semaphore
#   policy   = wait critical
#
# Metodologia:
#   - N=8192, T=500, TILE=32, WRITE_OUTPUT=0;
#   - OMP_PLACES=cores;
#   - PREDICT OFF nas medicoes;
#   - max_lead=2;
#   - calibracao P/R separada para cada (threads, affinity);
#   - calibracao WAIT separada para cada (threads, affinity, backend);
#   - 1 warm-up por caso;
#   - 20 repeticoes temporizadas, pareadas e com ordem alternada;
#   - 3 repeticoes de profiling por caso;
#   - um probe de afinidade por (threads, affinity), usando
#     OMP_DISPLAY_AFFINITY=TRUE, fora das medicoes.
#
# Saida padrao:
#   results/sdumont_affinity_close_spread_T500/
#       times.csv
#       summary.csv
#       affinity_summary.csv
#       profile_per_run.csv
#       profile_summary.csv
#       run_metadata.txt
#       lscpu_topology.txt
#       affinity_close/threads_XXX/*.log, *.dat
#       affinity_spread/threads_XXX/*.log, *.dat
# =============================================================================

REPS="${REPS:-20}"
PROFILE_REPS="${PROFILE_REPS:-3}"
WARMUPS="${WARMUPS:-1}"

THREADS_LIST_STR="${THREADS_LIST:-64 96 160}"
read -r -a THREADS_LIST <<< "${THREADS_LIST_STR}"

AFFINITIES=(close spread)

CAL_SAMPLES="${CAL_SAMPLES:-32}"
WAIT_CAL_SAMPLES="${WAIT_CAL_SAMPLES:-32}"
PROGRESS_LAMBDA="${PROGRESS_LAMBDA:-1.0}"
PROGRESS_BOOTSTRAP="${PROGRESS_BOOTSTRAP:-8}"

OUTDIR="${OUTDIR:-results/sdumont_affinity_close_spread_T500}"

BW_SOLVER="./heat2d_adaptive_busywait"
SM_SOLVER="./heat2d_adaptive_sem"
BW_PROF="./heat2d_adaptive_busywait_profile"
SM_PROF="./heat2d_adaptive_sem_profile"

PR_CAL="./heat2d_dependency_calibrate"
BW_WCAL="./heat2d_wait_calibrate_busywait"
SM_WCAL="./heat2d_wait_calibrate_semaphore"

cd "${SLURM_SUBMIT_DIR}"

export OMP_PLACES=cores
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
    echo "ERRO: este teste exige N=8192; atual=${N_VALUE:-ausente}." >&2
    exit 2
}
[[ "${T_VALUE}" == "500" ]] || {
    echo "ERRO: este teste exige T=500; atual=${T_VALUE:-ausente}." >&2
    exit 3
}
[[ "${TILE_VALUE}" == "32" ]] || {
    echo "ERRO: este teste exige TILE=32; atual=${TILE_VALUE:-ausente}." >&2
    exit 4
}
case "${WRITE_OUTPUT_VALUE}" in
    0|off|false|no) ;;
    *)
        echo "ERRO: este teste exige WRITE_OUTPUT=0; atual=${WRITE_OUTPUT_VALUE:-ausente}." >&2
        exit 5
        ;;
esac

for threads in "${THREADS_LIST[@]}"; do
    if (( threads < 1 || threads > SLURM_CPUS_PER_TASK )); then
        echo "ERRO: THREADS_LIST contem valor invalido: ${threads}." >&2
        exit 6
    fi
done

for affinity in "${AFFINITIES[@]}"; do
    case "${affinity}" in
        close|spread) ;;
        *)
            echo "ERRO: afinidade invalida: ${affinity}; use close ou spread." >&2
            exit 7
            ;;
    esac
done

mkdir -p "${OUTDIR}"

rm -f \
    "${OUTDIR}/times.csv" \
    "${OUTDIR}/summary.csv" \
    "${OUTDIR}/affinity_summary.csv" \
    "${OUTDIR}/profile_per_run.csv" \
    "${OUTDIR}/profile_summary.csv" \
    "${OUTDIR}/run_metadata.txt" \
    "${OUTDIR}/lscpu_topology.txt" 2>/dev/null || true

for affinity in "${AFFINITIES[@]}"; do
    for threads in "${THREADS_LIST[@]}"; do
        tag="$(printf '%03d' "${threads}")"
        pdir="${OUTDIR}/affinity_${affinity}/threads_${tag}"
        mkdir -p "${pdir}"
        rm -f "${pdir}"/*.log "${pdir}"/*.dat 2>/dev/null || true
    done
done

lscpu > "${OUTDIR}/lscpu_topology.txt" 2>&1 || true
{
    echo
    echo "===== lscpu -e=CPU,CORE,SOCKET,NODE ====="
    lscpu -e=CPU,CORE,SOCKET,NODE 2>&1 || true
} >> "${OUTDIR}/lscpu_topology.txt"

echo "============================================================"
echo "SDUMONT II — OPENMP AFFINITY EXPERIMENT"
echo "============================================================"
echo "Job ID              : ${SLURM_JOB_ID}"
echo "Node                : ${SLURM_NODELIST}"
echo "CPUs/task           : ${SLURM_CPUS_PER_TASK}"
echo "Threads             : ${THREADS_LIST[*]}"
echo "Affinities          : ${AFFINITIES[*]}"
echo "Backends            : busywait semaphore"
echo "Policies            : wait critical"
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
echo "OMP_PROC_BIND       : varied: ${AFFINITIES[*]}"
echo "Output              : ${OUTDIR}"
echo "============================================================"

{
    echo "job_id=${SLURM_JOB_ID}"
    echo "node=${SLURM_NODELIST}"
    echo "cpus_per_task=${SLURM_CPUS_PER_TASK}"
    echo "threads=${THREADS_LIST[*]}"
    echo "affinities=${AFFINITIES[*]}"
    echo "backends=busywait semaphore"
    echo "policies=wait critical"
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
    echo "max_lead=${HEAT2D_MAX_LEAD}"
    echo "omp_places=${OMP_PLACES}"
    echo "omp_proc_bind=varied:${AFFINITIES[*]}"
    echo "date_start=$(date --iso-8601=seconds)"
    echo "uname=$(uname -a)"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
} > "${OUTDIR}/run_metadata.txt"

echo
echo "================ BUILD ======================================="
make -j "${SLURM_CPUS_PER_TASK}" core profiles

for exe in \
    "${BW_SOLVER}" "${SM_SOLVER}" \
    "${BW_PROF}" "${SM_PROF}" \
    "${PR_CAL}" "${BW_WCAL}" "${SM_WCAL}"
do
    [[ -x "${exe}" ]] || {
        echo "ERRO: executavel ausente: ${exe}" >&2
        exit 8
    }
done

check_common_output() {
    local log="$1"

    grep -q '^Tempo[[:space:]]*:' "${log}" || {
        echo "ERRO: Tempo ausente em ${log}" >&2
        exit 9
    }
    grep -q '^Linf:' "${log}" || {
        echo "ERRO: Linf ausente em ${log}" >&2
        exit 10
    }
}

run_adaptive() {
    local affinity="$1"
    local threads="$2"
    local backend="$3"
    local policy="$4"
    local kind="$5"
    local rep="$6"
    local pdir="$7"
    local cost_file="$8"
    local wait_file="$9"

    local exe recompute criticality

    case "${backend}" in
        busywait)
            if [[ "${kind}" == "prof" ]]; then
                exe="${BW_PROF}"
            else
                exe="${BW_SOLVER}"
            fi
            ;;
        semaphore)
            if [[ "${kind}" == "prof" ]]; then
                exe="${SM_PROF}"
            else
                exe="${SM_SOLVER}"
            fi
            ;;
        *)
            echo "ERRO interno: backend invalido: ${backend}" >&2
            exit 11
            ;;
    esac

    case "${policy}" in
        wait)
            recompute=0
            criticality=0
            ;;
        critical)
            recompute=1
            criticality=1
            ;;
        *)
            echo "ERRO interno: policy invalida: ${policy}" >&2
            exit 12
            ;;
    esac

    export OMP_NUM_THREADS="${threads}"
    export OMP_PROC_BIND="${affinity}"

    local rtag
    rtag="$(printf '%02d' "${rep}")"
    local log="${pdir}/${backend}_${policy}_${kind}_r${rtag}.log"

    HEAT2D_ENABLE_PREDICT=0 \
    HEAT2D_ENABLE_RECOMPUTE="${recompute}" \
    HEAT2D_ENABLE_CRITICALITY="${criticality}" \
    HEAT2D_COST_FILE="${cost_file}" \
    HEAT2D_WAIT_COST_FILE="${wait_file}" \
    HEAT2D_PROGRESS_LAMBDA="${PROGRESS_LAMBDA}" \
    HEAT2D_PROGRESS_BOOTSTRAP_SAMPLES="${PROGRESS_BOOTSTRAP}" \
        "${exe}" > "${log}" 2>&1

    check_common_output "${log}"

    grep -q '^Adaptive predict: off$' "${log}" || {
        echo "ERRO: PREDICT deveria estar OFF em ${log}" >&2
        exit 13
    }

    if [[ "${recompute}" == "1" ]]; then
        grep -q '^Adaptive recompute: on$' "${log}" || {
            echo "ERRO: RECOMPUTE deveria estar ON em ${log}" >&2
            exit 14
        }
    else
        grep -q '^Adaptive recompute: off$' "${log}" || {
            echo "ERRO: RECOMPUTE deveria estar OFF em ${log}" >&2
            exit 15
        }
    fi

    if [[ "${criticality}" == "1" ]]; then
        grep -q '^Adaptive criticality-aware: on$' "${log}" || {
            echo "ERRO: criticality deveria estar ON em ${log}" >&2
            exit 16
        }
    else
        grep -q '^Adaptive criticality-aware: off$' "${log}" || {
            echo "ERRO: criticality deveria estar OFF em ${log}" >&2
            exit 17
        }
    fi
}

calibrate_affinity_threads() {
    local affinity="$1"
    local threads="$2"
    local pdir="$3"

    local cost_file="${pdir}/heat2d_cost_model.dat"
    local bw_wait="${pdir}/heat2d_wait_cost_busywait.dat"
    local sm_wait="${pdir}/heat2d_wait_cost_semaphore.dat"

    export OMP_NUM_THREADS="${threads}"
    export OMP_PROC_BIND="${affinity}"

    echo "[calibration] affinity=${affinity} p=${threads} PREDICT/RECOMPUTE"
    HEAT2D_ENABLE_PREDICT=1 \
    HEAT2D_ENABLE_RECOMPUTE=1 \
    HEAT2D_CALIBRATION_SAMPLES="${CAL_SAMPLES}" \
    HEAT2D_COST_FILE="${cost_file}" \
        "${PR_CAL}" > "${pdir}/calibration_pr.log" 2>&1

    [[ -s "${cost_file}" ]] || {
        echo "ERRO: modelo P/R nao gerado para ${affinity}/p=${threads}." >&2
        exit 18
    }

    echo "[calibration] affinity=${affinity} p=${threads} WAIT busywait"
    HEAT2D_WAIT_CALIBRATION_SAMPLES="${WAIT_CAL_SAMPLES}" \
    HEAT2D_WAIT_COST_FILE="${bw_wait}" \
        "${BW_WCAL}" > "${pdir}/calibration_wait_busywait.log" 2>&1

    [[ -s "${bw_wait}" ]] || {
        echo "ERRO: modelo WAIT busywait nao gerado para ${affinity}/p=${threads}." >&2
        exit 19
    }

    echo "[calibration] affinity=${affinity} p=${threads} WAIT semaphore"
    HEAT2D_WAIT_CALIBRATION_SAMPLES="${WAIT_CAL_SAMPLES}" \
    HEAT2D_WAIT_COST_FILE="${sm_wait}" \
        "${SM_WCAL}" > "${pdir}/calibration_wait_semaphore.log" 2>&1

    [[ -s "${sm_wait}" ]] || {
        echo "ERRO: modelo WAIT semaphore nao gerado para ${affinity}/p=${threads}." >&2
        exit 20
    }
}

affinity_probe() {
    local affinity="$1"
    local threads="$2"
    local pdir="$3"

    local cost_file="${pdir}/heat2d_cost_model.dat"
    local wait_file="${pdir}/heat2d_wait_cost_busywait.dat"
    local log="${pdir}/affinity_map.log"

    export OMP_NUM_THREADS="${threads}"
    export OMP_PROC_BIND="${affinity}"

    echo "[affinity probe] affinity=${affinity} p=${threads}"

    OMP_DISPLAY_AFFINITY=TRUE \
    HEAT2D_ENABLE_PREDICT=0 \
    HEAT2D_ENABLE_RECOMPUTE=0 \
    HEAT2D_ENABLE_CRITICALITY=0 \
    HEAT2D_COST_FILE="${cost_file}" \
    HEAT2D_WAIT_COST_FILE="${wait_file}" \
    HEAT2D_PROGRESS_LAMBDA="${PROGRESS_LAMBDA}" \
    HEAT2D_PROGRESS_BOOTSTRAP_SAMPLES="${PROGRESS_BOOTSTRAP}" \
        "${BW_SOLVER}" > "${log}" 2>&1

    check_common_output "${log}"
}

get_case_files() {
    local affinity="$1"
    local threads="$2"

    local tag
    tag="$(printf '%03d' "${threads}")"
    CASE_DIR="${OUTDIR}/affinity_${affinity}/threads_${tag}"
    CASE_COST="${CASE_DIR}/heat2d_cost_model.dat"
    CASE_BW_WAIT="${CASE_DIR}/heat2d_wait_cost_busywait.dat"
    CASE_SM_WAIT="${CASE_DIR}/heat2d_wait_cost_semaphore.dat"
}

run_backend_pair() {
    local affinity="$1"
    local threads="$2"
    local backend="$3"
    local kind="$4"
    local rep="$5"
    local reverse="$6"

    get_case_files "${affinity}" "${threads}"

    local wait_file
    case "${backend}" in
        busywait)  wait_file="${CASE_BW_WAIT}" ;;
        semaphore) wait_file="${CASE_SM_WAIT}" ;;
        *)
            echo "ERRO interno: backend invalido em run_backend_pair: ${backend}" >&2
            exit 21
            ;;
    esac

    if [[ "${reverse}" == "0" ]]; then
        run_adaptive "${affinity}" "${threads}" "${backend}" wait \
            "${kind}" "${rep}" "${CASE_DIR}" "${CASE_COST}" "${wait_file}"
        run_adaptive "${affinity}" "${threads}" "${backend}" critical \
            "${kind}" "${rep}" "${CASE_DIR}" "${CASE_COST}" "${wait_file}"
    else
        run_adaptive "${affinity}" "${threads}" "${backend}" critical \
            "${kind}" "${rep}" "${CASE_DIR}" "${CASE_COST}" "${wait_file}"
        run_adaptive "${affinity}" "${threads}" "${backend}" wait \
            "${kind}" "${rep}" "${CASE_DIR}" "${CASE_COST}" "${wait_file}"
    fi
}

# =============================================================================
# Experimento
# =============================================================================

for threads in "${THREADS_LIST[@]}"; do
    echo
    echo "============================================================"
    echo "THREADS = ${threads}"
    echo "============================================================"

    # Calibra ambas as afinidades antes das medicoes deste p.
    for affinity in "${AFFINITIES[@]}"; do
        get_case_files "${affinity}" "${threads}"
        calibrate_affinity_threads "${affinity}" "${threads}" "${CASE_DIR}"
        affinity_probe "${affinity}" "${threads}" "${CASE_DIR}"
    done

    echo
    echo "================ WARM-UP p=${threads} ========================"
    for ((w=1; w<=WARMUPS; ++w)); do
        if (( w % 2 == 1 )); then
            run_backend_pair close  "${threads}" busywait  warmup "${w}" 0
            run_backend_pair close  "${threads}" semaphore warmup "${w}" 0
            run_backend_pair spread "${threads}" busywait  warmup "${w}" 0
            run_backend_pair spread "${threads}" semaphore warmup "${w}" 0
        else
            run_backend_pair spread "${threads}" semaphore warmup "${w}" 1
            run_backend_pair spread "${threads}" busywait  warmup "${w}" 1
            run_backend_pair close  "${threads}" semaphore warmup "${w}" 1
            run_backend_pair close  "${threads}" busywait  warmup "${w}" 1
        fi
    done

    echo
    echo "================ TIMING p=${threads} ========================="
    for ((rep=1; rep<=REPS; ++rep)); do
        if (( rep % 2 == 1 )); then
            run_backend_pair close  "${threads}" busywait  ref "${rep}" 0
            run_backend_pair close  "${threads}" semaphore ref "${rep}" 0
            run_backend_pair spread "${threads}" busywait  ref "${rep}" 0
            run_backend_pair spread "${threads}" semaphore ref "${rep}" 0
            order="close->spread / wait->critical"
        else
            run_backend_pair spread "${threads}" semaphore ref "${rep}" 1
            run_backend_pair spread "${threads}" busywait  ref "${rep}" 1
            run_backend_pair close  "${threads}" semaphore ref "${rep}" 1
            run_backend_pair close  "${threads}" busywait  ref "${rep}" 1
            order="spread->close / critical->wait"
        fi
        echo "threads=${threads} timing rep=${rep}/${REPS} complete (${order})"
    done

    echo
    echo "================ PROFILING p=${threads} ======================"
    for ((rep=1; rep<=PROFILE_REPS; ++rep)); do
        if (( rep % 2 == 1 )); then
            run_backend_pair close  "${threads}" busywait  prof "${rep}" 0
            run_backend_pair close  "${threads}" semaphore prof "${rep}" 0
            run_backend_pair spread "${threads}" busywait  prof "${rep}" 0
            run_backend_pair spread "${threads}" semaphore prof "${rep}" 0
        else
            run_backend_pair spread "${threads}" semaphore prof "${rep}" 1
            run_backend_pair spread "${threads}" busywait  prof "${rep}" 1
            run_backend_pair close  "${threads}" semaphore prof "${rep}" 1
            run_backend_pair close  "${threads}" busywait  prof "${rep}" 1
        fi
        echo "threads=${threads} profile rep=${rep}/${PROFILE_REPS} complete"
    done
done

# =============================================================================
# Consolidacao e estatistica
# =============================================================================

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
        'read': 0.0,
        'recompute': 0.0,
        'predict': 0.0,
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
        'Adaptive actions READ': 'read',
        'Adaptive actions RECOMPUTE': 'recompute',
        'Adaptive actions PREDICT': 'predict',
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


def affinity_from_path(path):
    parts = os.path.normpath(path).split(os.sep)
    for p in parts:
        if p.startswith('affinity_'):
            return p[len('affinity_'):]
    raise RuntimeError(f'affinity not found in path: {path}')


def threads_from_path(path):
    parts = os.path.normpath(path).split(os.sep)
    for p in parts:
        if p.startswith('threads_'):
            return int(p.split('_', 1)[1])
    raise RuntimeError(f'threads not found in path: {path}')


# -------------------------------------------------------------------------
# Timing rows
# -------------------------------------------------------------------------
timing_rows = []

for path in sorted(glob.glob(
    os.path.join(outdir, 'affinity_*', 'threads_*', '*_ref_r*.log')
)):
    base = os.path.basename(path)
    m = re.fullmatch(
        r'(busywait|semaphore)_(wait|critical)_ref_r(\d+)\.log',
        base
    )
    if not m:
        continue

    backend, policy, rep = m.group(1), m.group(2), int(m.group(3))
    affinity = affinity_from_path(path)
    threads = threads_from_path(path)
    time_s, linf = parse_time_linf(path)

    timing_rows.append({
        'affinity': affinity,
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
    key = (r['affinity'], r['threads'], r['backend'], r['policy'])
    expected_timing.setdefault(key, []).append(r)

for key, rr in expected_timing.items():
    if len(rr) != reps:
        raise SystemExit(f'Expected {reps} timing reps for {key}, got {len(rr)}')

times_path = os.path.join(outdir, 'times.csv')
with open(times_path, 'w', newline='') as f:
    cols = ['affinity', 'threads', 'backend', 'policy', 'rep', 'time_s', 'linf']
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(sorted(
        timing_rows,
        key=lambda r: (
            r['threads'], r['affinity'], r['backend'], r['policy'], r['rep']
        )
    ))


# -------------------------------------------------------------------------
# Profile rows
# -------------------------------------------------------------------------
profile_rows = []

for path in sorted(glob.glob(
    os.path.join(outdir, 'affinity_*', 'threads_*', '*_prof_r*.log')
)):
    base = os.path.basename(path)
    m = re.fullmatch(
        r'(busywait|semaphore)_(wait|critical)_prof_r(\d+)\.log',
        base
    )
    if not m:
        continue

    backend, policy, rep = m.group(1), m.group(2), int(m.group(3))
    affinity = affinity_from_path(path)
    threads = threads_from_path(path)
    vals = parse_profile(path)
    vals.update({
        'affinity': affinity,
        'threads': threads,
        'backend': backend,
        'policy': policy,
        'rep': rep,
    })
    profile_rows.append(vals)

expected_profile = {}
for r in profile_rows:
    key = (r['affinity'], r['threads'], r['backend'], r['policy'])
    expected_profile.setdefault(key, []).append(r)

for key, rr in expected_profile.items():
    if len(rr) != profile_reps:
        raise SystemExit(
            f'Expected {profile_reps} profile reps for {key}, got {len(rr)}'
        )

profile_cols = [
    'affinity', 'threads', 'backend', 'policy', 'rep',
    'profile_time_s', 'linf',
    'read', 'recompute', 'predict', 'wait', 'lead_wait', 'progress_blocked',
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
        key=lambda r: (
            r['threads'], r['affinity'], r['backend'], r['policy'], r['rep']
        )
    ))

profile_summary_rows = []
for key in sorted(expected_profile):
    rr = expected_profile[key]
    med = lambda field: statistics.median(float(r[field]) for r in rr)
    profile_summary_rows.append({
        'affinity': key[0],
        'threads': key[1],
        'backend': key[2],
        'policy': key[3],
        'reps': len(rr),
        'median_profile_time_s': med('profile_time_s'),
        'median_read': med('read'),
        'median_recompute': med('recompute'),
        'median_predict': med('predict'),
        'median_wait': med('wait'),
        'median_lead_wait': med('lead_wait'),
        'median_progress_blocked': med('progress_blocked'),
        'median_penalty_samples': med('penalty_samples'),
        'median_penalty_nonzero_samples': med('penalty_nonzero_samples'),
        'median_penalty_mean_ticks': med('penalty_mean_ticks'),
        'median_penalty_nonzero_fraction': med('penalty_nonzero_fraction'),
        'median_critical_recompute': med('critical_recompute'),
        'median_noncritical_recompute_rejected':
            med('noncritical_recompute_rejected'),
        'median_criticality_forced_wait': med('criticality_forced_wait'),
        'median_linf': med('linf'),
    })

profile_summary_path = os.path.join(outdir, 'profile_summary.csv')
with open(profile_summary_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(profile_summary_rows[0].keys()))
    w.writeheader()
    w.writerows(profile_summary_rows)


# -------------------------------------------------------------------------
# WAIT vs CRITICAL, separadamente por afinidade
# -------------------------------------------------------------------------
grouped = {}
for r in timing_rows:
    grouped.setdefault(
        (r['affinity'], r['threads'], r['backend']), {}
    ).setdefault(r['policy'], []).append(r)

summary_rows = []

for (affinity, threads, backend) in sorted(grouped):
    byp = grouped[(affinity, threads, backend)]
    if set(byp) != {'wait', 'critical'}:
        raise RuntimeError(
            f'{affinity}/{threads}/{backend}: policies={sorted(byp)}'
        )

    wait_by_rep = {int(r['rep']): float(r['time_s']) for r in byp['wait']}
    crit_by_rep = {int(r['rep']): float(r['time_s']) for r in byp['critical']}
    common = sorted(set(wait_by_rep) & set(crit_by_rep))
    if len(common) != reps:
        raise RuntimeError(
            f'{affinity}/{threads}/{backend}: expected {reps} paired reps, '
            f'got {len(common)}'
        )

    paired = [wait_by_rep[i] / crit_by_rep[i] for i in common]
    wait_times = list(wait_by_rep.values())
    crit_times = list(crit_by_rep.values())

    med_wait = statistics.median(wait_times)
    med_crit = statistics.median(crit_times)

    seed = (
        threads * 1000
        + (1 if backend == 'busywait' else 2) * 100
        + (1 if affinity == 'close' else 2)
    )
    lo, hi = bootstrap_median_ci(paired, seed=20260903 + seed)

    linf_wait = statistics.median(float(r['linf']) for r in byp['wait'])
    linf_crit = statistics.median(float(r['linf']) for r in byp['critical'])

    if not math.isclose(linf_wait, linf_crit, rel_tol=0.0, abs_tol=1e-15):
        raise RuntimeError(
            f'Linf mismatch {affinity}/{threads}/{backend}: '
            f'wait={linf_wait}, critical={linf_crit}'
        )

    summary_rows.append({
        'affinity': affinity,
        'threads': threads,
        'backend': backend,
        'reps': reps,
        'median_wait_s': med_wait,
        'median_critical_s': med_crit,
        'speedup_wait_over_critical_ratio_of_medians': med_wait / med_crit,
        'critical_gain_pct_ratio_of_medians':
            (1.0 - med_crit / med_wait) * 100.0,
        'median_paired_wait_over_critical': statistics.median(paired),
        'critical_wins_vs_wait': sum(x > 1.0 for x in paired),
        'bootstrap95_paired_speedup_lo': lo,
        'bootstrap95_paired_speedup_hi': hi,
        'min_paired_speedup': min(paired),
        'max_paired_speedup': max(paired),
        'linf_wait': linf_wait,
        'linf_critical': linf_crit,
    })

summary_path = os.path.join(outdir, 'summary.csv')
with open(summary_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
    w.writeheader()
    w.writerows(summary_rows)


# -------------------------------------------------------------------------
# Efeito da afinidade: close vs spread, por politica
#
# Razao close/spread > 1 significa que SPREAD foi mais rapido.
# -------------------------------------------------------------------------
by_affinity = {}
for r in timing_rows:
    by_affinity.setdefault(
        (r['threads'], r['backend'], r['policy']), {}
    ).setdefault(r['affinity'], []).append(r)

affinity_rows = []

for (threads, backend, policy) in sorted(by_affinity):
    bya = by_affinity[(threads, backend, policy)]
    if set(bya) != {'close', 'spread'}:
        raise RuntimeError(
            f'{threads}/{backend}/{policy}: affinities={sorted(bya)}'
        )

    close_by_rep = {
        int(r['rep']): float(r['time_s']) for r in bya['close']
    }
    spread_by_rep = {
        int(r['rep']): float(r['time_s']) for r in bya['spread']
    }
    common = sorted(set(close_by_rep) & set(spread_by_rep))
    if len(common) != reps:
        raise RuntimeError(
            f'{threads}/{backend}/{policy}: expected {reps} affinity pairs, '
            f'got {len(common)}'
        )

    paired = [close_by_rep[i] / spread_by_rep[i] for i in common]
    med_close = statistics.median(close_by_rep.values())
    med_spread = statistics.median(spread_by_rep.values())

    seed = (
        threads * 1000
        + (1 if backend == 'busywait' else 2) * 100
        + (1 if policy == 'wait' else 2) * 10
    )
    lo, hi = bootstrap_median_ci(paired, seed=20260903 + seed)

    affinity_rows.append({
        'threads': threads,
        'backend': backend,
        'policy': policy,
        'reps': reps,
        'median_close_s': med_close,
        'median_spread_s': med_spread,
        'speedup_close_over_spread_ratio_of_medians':
            med_close / med_spread,
        'spread_gain_pct_ratio_of_medians':
            (1.0 - med_spread / med_close) * 100.0,
        'median_paired_close_over_spread': statistics.median(paired),
        'spread_wins_vs_close': sum(x > 1.0 for x in paired),
        'bootstrap95_paired_close_over_spread_lo': lo,
        'bootstrap95_paired_close_over_spread_hi': hi,
        'min_paired_close_over_spread': min(paired),
        'max_paired_close_over_spread': max(paired),
    })

affinity_summary_path = os.path.join(outdir, 'affinity_summary.csv')
with open(affinity_summary_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(affinity_rows[0].keys()))
    w.writeheader()
    w.writerows(affinity_rows)


print()
print('=== CRITICAL vs WAIT-only BY AFFINITY ===')
for r in summary_rows:
    print(
        f"p={r['threads']:3d} {r['affinity']:6s} {r['backend']:9s} "
        f"S_med={r['speedup_wait_over_critical_ratio_of_medians']:.6f} "
        f"S_paired={r['median_paired_wait_over_critical']:.6f} "
        f"wins={r['critical_wins_vs_wait']}/{r['reps']} "
        f"CI95=[{r['bootstrap95_paired_speedup_lo']:.6f},"
        f"{r['bootstrap95_paired_speedup_hi']:.6f}]"
    )

print()
print('=== SPREAD vs CLOSE ===')
print('ratio close/spread > 1 => spread faster')
for r in affinity_rows:
    print(
        f"p={r['threads']:3d} {r['backend']:9s} {r['policy']:8s} "
        f"S_med={r['speedup_close_over_spread_ratio_of_medians']:.6f} "
        f"S_paired={r['median_paired_close_over_spread']:.6f} "
        f"wins={r['spread_wins_vs_close']}/{r['reps']} "
        f"CI95=[{r['bootstrap95_paired_close_over_spread_lo']:.6f},"
        f"{r['bootstrap95_paired_close_over_spread_hi']:.6f}]"
    )

print()
print('=== FINAL FILES ===')
print('Timing per run      :', times_path)
print('WAIT vs CRITICAL    :', summary_path)
print('Affinity comparison :', affinity_summary_path)
print('Profile per run     :', profile_per_run_path)
print('Profile summary     :', profile_summary_path)
PY

{
    echo "date_end=$(date --iso-8601=seconds)"
} >> "${OUTDIR}/run_metadata.txt"

echo
echo "============================================================"
echo "SDumont affinity experiment complete."
echo "Main files:"
echo "  ${OUTDIR}/summary.csv"
echo "  ${OUTDIR}/affinity_summary.csv"
echo "  ${OUTDIR}/profile_summary.csv"
echo "  ${OUTDIR}/profile_per_run.csv"
echo "  ${OUTDIR}/times.csv"
echo "  ${OUTDIR}/run_metadata.txt"
echo "  ${OUTDIR}/lscpu_topology.txt"
echo "============================================================"
