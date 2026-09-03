#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Heat2D — Antares: topology + SMT experiment
#
# Machine:
#   2 x Intel Xeon Gold 5120
#   14 physical cores/socket, SMT2
#
# CPU topology used:
#   one_socket_no_smt_14t : 14 threads, 14 physical cores, socket 0
#                           CPUs 0-13
#   one_socket_smt_28t    : 28 threads, same 14 physical cores, SMT2, socket 0
#                           CPUs 0-13,28-41
#   two_socket_no_smt_28t : 28 threads, 28 physical cores, two sockets, no SMT
#                           CPUs 0-27
#   two_socket_smt_56t    : 56 threads, same 28 physical cores, SMT2, two sockets
#                           CPUs 0-55
#
# Scientific questions:
#   1) confirm the previous 28T/no-SMT result with more repetitions;
#   2) measure the effect of SMT on WAIT/PROGRESS/CRITICAL;
#   3) test interaction between SMT and waiting backend (busy-wait/semaphore);
#   4) inspect whether CRITICAL suppresses harmful lead-guard waits under SMT.
#
# Main numerical configuration:
#   N=8192, T=500, TILE=32
#
# Policies:
#   WAIT      : RECOMPUTE=0, CRITICALITY=0
#   PROGRESS  : RECOMPUTE=1, CRITICALITY=0
#   CRITICAL  : RECOMPUTE=1, CRITICALITY=1
#
# PREDICT remains OFF during all measured and profile runs.
#
# Measurement design:
#   - calibration is redone independently for EACH topology;
#   - 1 warm-up per topology/backend/policy;
#   - 20 clean timing repetitions by default;
#   - 3 profile repetitions by default;
#   - topology order rotates between timing blocks;
#   - policy/backend order rotates inside each topology;
#   - primary analysis uses paired speedups and bootstrap median CI.
#
# Important affinity choice:
#   OMP_PLACES=threads is intentional here.  With SMT enabled, each OpenMP
#   hardware thread must be an explicit place.  taskset restricts the exact
#   logical CPUs visible to the run.
#
# Default output:
#   results/antares_smt_topology_T500/
#       times.csv
#       profile_per_run.csv
#       profile_summary.csv
#       summary.csv
#       smt_summary.csv
#       order.csv
#       run_metadata.txt
#       lscpu.txt
#       lscpu_extended.txt
#       numactl_hardware.txt          (if numactl exists)
#       <topology>/...
# =============================================================================

N_VALUE="${N_VALUE:-8192}"
T_VALUE="${T_VALUE:-500}"
TILE_VALUE="${TILE_VALUE:-32}"

REPS="${REPS:-20}"
PROFILE_REPS="${PROFILE_REPS:-3}"
WARMUPS="${WARMUPS:-1}"

PROGRESS_LAMBDA="${PROGRESS_LAMBDA:-1.0}"
PROGRESS_BOOTSTRAP="${PROGRESS_BOOTSTRAP:-8}"

CAL_SAMPLES="${CAL_SAMPLES:-16}"
WAIT_CAL_SAMPLES="${WAIT_CAL_SAMPLES:-16}"
CAL_T="${CAL_T:-250}"

OUTDIR="${OUTDIR:-results/antares_smt_topology_T${T_VALUE}}"

BW_REF="./heat2d_adaptive_busywait"
SM_REF="./heat2d_adaptive_sem"
BW_PROF="./heat2d_adaptive_busywait_profile"
SM_PROF="./heat2d_adaptive_sem_profile"

PR_CAL="./heat2d_dependency_calibrate"
BW_WCAL="./heat2d_wait_calibrate_busywait"
SM_WCAL="./heat2d_wait_calibrate_semaphore"

# Format:
# label|threads|physical_cores|smt|socket_scope|cpu_list
CONFIGS=(
    "one_socket_no_smt_14t|14|14|no|one_socket|0-13"
    "one_socket_smt_28t|28|14|yes|one_socket|0-13,28-41"
    "two_socket_no_smt_28t|28|28|no|two_socket|0-27"
    "two_socket_smt_56t|56|28|yes|two_socket|0-55"
)

# Interleave backends to avoid measuring an entire backend hours before the other.
CASES=(
    "busywait:wait"
    "semaphore:wait"
    "busywait:progress"
    "semaphore:progress"
    "busywait:critical"
    "semaphore:critical"
)

# Exact logical-thread places are required for the SMT cases.
export OMP_PLACES=threads
export OMP_PROC_BIND=close
export OMP_DYNAMIC=FALSE
export HEAT2D_MAX_LEAD=2

# Avoid hidden affinity variables overriding the intended OpenMP placement.
unset GOMP_CPU_AFFINITY || true
unset KMP_AFFINITY || true

command -v python3 >/dev/null || {
    echo "ERRO: python3 nao encontrado." >&2
    exit 1
}
command -v make >/dev/null || {
    echo "ERRO: make nao encontrado." >&2
    exit 1
}
command -v taskset >/dev/null || {
    echo "ERRO: taskset nao encontrado (pacote util-linux)." >&2
    exit 1
}
command -v lscpu >/dev/null || {
    echo "ERRO: lscpu nao encontrado." >&2
    exit 1
}
[[ -f param.txt ]] || {
    echo "ERRO: param.txt nao encontrado." >&2
    exit 1
}

if (( REPS < 2 )); then
    echo "ERRO: REPS deve ser >= 2 para analise pareada." >&2
    exit 2
fi
if (( PROFILE_REPS < 1 )); then
    echo "ERRO: PROFILE_REPS deve ser >= 1." >&2
    exit 2
fi
if (( WARMUPS < 1 )); then
    echo "ERRO: WARMUPS deve ser >= 1." >&2
    exit 2
fi

mkdir -p "${OUTDIR}"

# Preserve the user's param.txt exactly.
PARAM_BACKUP="$(mktemp ./param.txt.antares_smt.XXXXXX)"
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
import re
import sys

path = 'param.txt'
values = {
    'N': sys.argv[1],
    'T': sys.argv[2],
    'TILE': sys.argv[3],
    'WRITE_OUTPUT': '0',
}

with open(path, encoding='utf-8') as f:
    lines = f.readlines()

out = []
seen = set()

for line in lines:
    raw = line.rstrip('\n')
    m = re.match(
        r'^(\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*=\s*)(.*?)(\s*)$',
        raw
    )
    if m and m.group(2) in values:
        key = m.group(2)
        out.append(f'{m.group(1)}{key}{m.group(3)}{values[key]}\n')
        seen.add(key)
    else:
        out.append(line)

for key, value in values.items():
    if key not in seen:
        out.append(f'{key} = {value}\n')

with open(path, 'w', encoding='utf-8') as f:
    f.writelines(out)
PY
}

expand_cpu_list_count() {
    python3 - "$1" <<'PY'
import sys

s = sys.argv[1]
cpus = []

for token in s.split(','):
    token = token.strip()
    if not token:
        continue
    if '-' in token:
        a, b = map(int, token.split('-', 1))
        if b < a:
            raise SystemExit(f'invalid CPU range: {token}')
        cpus.extend(range(a, b + 1))
    else:
        cpus.append(int(token))

if len(cpus) != len(set(cpus)):
    raise SystemExit(f'duplicate CPU in set: {s}')

print(len(cpus))
PY
}

validate_config() {
    local spec="$1"
    local label threads physical_cores smt socket_scope cpus
    IFS='|' read -r label threads physical_cores smt socket_scope cpus <<< "${spec}"

    local count
    count="$(expand_cpu_list_count "${cpus}")"

    if [[ "${count}" != "${threads}" ]]; then
        echo "ERRO: ${label}: CPU set '${cpus}' contem ${count} CPUs, mas threads=${threads}." >&2
        exit 3
    fi

    # Test whether the operating system accepts the mask.
    taskset -c "${cpus}" true >/dev/null 2>&1 || {
        echo "ERRO: ${label}: taskset rejeitou CPUs '${cpus}'." >&2
        exit 3
    }

    echo "[topology] ${label}: threads=${threads}, physical_cores=${physical_cores}, smt=${smt}, scope=${socket_scope}, CPUs=${cpus}"
}

for spec in "${CONFIGS[@]}"; do
    validate_config "${spec}"
done

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
MAKE_JOBS="${MAKE_JOBS:-$(nproc 2>/dev/null || echo 4)}"

echo
echo "============================================================"
echo "ANTARES — TOPOLOGY + SMT EXPERIMENT"
echo "============================================================"
echo "N/T/TILE           : ${N_VALUE}/${T_VALUE}/${TILE_VALUE}"
echo "Timing reps        : ${REPS}"
echo "Profile reps       : ${PROFILE_REPS}"
echo "Warm-ups           : ${WARMUPS}"
echo "Calibration T      : ${CAL_T}"
echo "P/R samples        : ${CAL_SAMPLES}"
echo "WAIT samples       : ${WAIT_CAL_SAMPLES}"
echo "Progress lambda    : ${PROGRESS_LAMBDA}"
echo "Progress bootstrap : ${PROGRESS_BOOTSTRAP}"
echo "OMP_PLACES         : ${OMP_PLACES}"
echo "OMP_PROC_BIND      : ${OMP_PROC_BIND}"
echo "Output             : ${OUTDIR}"
echo "============================================================"

echo
echo "[build] make -j ${MAKE_JOBS} core profiles"
make -j "${MAKE_JOBS}" core profiles

for exe in \
    "${BW_REF}" "${SM_REF}" \
    "${BW_PROF}" "${SM_PROF}" \
    "${PR_CAL}" "${BW_WCAL}" "${SM_WCAL}"
do
    [[ -x "${exe}" ]] || {
        echo "ERRO: executavel ausente: ${exe}" >&2
        exit 4
    }
done

# ---------------------------------------------------------------------------
# Record machine topology
# ---------------------------------------------------------------------------
lscpu > "${OUTDIR}/lscpu.txt" 2>&1 || true
lscpu -e=CPU,NODE,SOCKET,CORE,ONLINE > "${OUTDIR}/lscpu_extended.txt" 2>&1 || true

if command -v numactl >/dev/null; then
    numactl --hardware > "${OUTDIR}/numactl_hardware.txt" 2>&1 || true
fi

# ---------------------------------------------------------------------------
# Run helpers
# ---------------------------------------------------------------------------
run_pinned_env() {
    local cpus="$1"
    local log="$2"
    shift 2

    taskset -c "${cpus}" env "$@" > "${log}" 2>&1
}

check_solver_log() {
    local log="$1"
    local threads="$2"
    local predict="$3"
    local recompute="$4"
    local criticality="$5"

    grep -q '^Tempo[[:space:]]*:' "${log}" || {
        echo "ERRO: Tempo ausente em ${log}" >&2
        exit 5
    }

    grep -q '^Linf:' "${log}" || {
        echo "ERRO: Linf ausente em ${log}" >&2
        exit 5
    }

    grep -q "^Threads: ${threads}$" "${log}" || {
        echo "ERRO: numero de threads inesperado em ${log}; esperado ${threads}." >&2
        exit 5
    }

    if [[ "${predict}" == "0" ]]; then
        grep -q '^Adaptive predict: off$' "${log}" || {
            echo "ERRO: PREDICT deveria estar OFF em ${log}" >&2
            exit 5
        }
    fi

    if [[ "${recompute}" == "1" ]]; then
        grep -q '^Adaptive recompute: on$' "${log}" || {
            echo "ERRO: RECOMPUTE deveria estar ON em ${log}" >&2
            exit 5
        }
    else
        grep -q '^Adaptive recompute: off$' "${log}" || {
            echo "ERRO: RECOMPUTE deveria estar OFF em ${log}" >&2
            exit 5
        }
    fi

    if [[ "${criticality}" == "1" ]]; then
        grep -q '^Adaptive criticality-aware: on$' "${log}" || {
            echo "ERRO: criticality deveria estar ON em ${log}" >&2
            exit 5
        }
    else
        grep -q '^Adaptive criticality-aware: off$' "${log}" || {
            echo "ERRO: criticality deveria estar OFF em ${log}" >&2
            exit 5
        }
    fi
}

run_solver() {
    local exe="$1"
    local backend="$2"
    local policy="$3"
    local log="$4"
    local threads="$5"
    local cpus="$6"
    local cost="$7"
    local bw_wait="$8"
    local sm_wait="$9"

    local recompute criticality waitcost

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
            echo "ERRO: policy invalida: ${policy}" >&2
            exit 6
            ;;
    esac

    case "${backend}" in
        busywait)
            waitcost="${bw_wait}"
            ;;
        semaphore)
            waitcost="${sm_wait}"
            ;;
        *)
            echo "ERRO: backend invalido: ${backend}" >&2
            exit 6
            ;;
    esac

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

    check_solver_log "${log}" "${threads}" 0 "${recompute}" "${criticality}"
}

# ---------------------------------------------------------------------------
# Prepare output directories
# ---------------------------------------------------------------------------
rm -f \
    "${OUTDIR}/times.csv" \
    "${OUTDIR}/profile_per_run.csv" \
    "${OUTDIR}/profile_summary.csv" \
    "${OUTDIR}/summary.csv" \
    "${OUTDIR}/smt_summary.csv" \
    "${OUTDIR}/order.csv" \
    "${OUTDIR}/run_metadata.txt" 2>/dev/null || true

for spec in "${CONFIGS[@]}"; do
    IFS='|' read -r label threads physical_cores smt socket_scope cpus <<< "${spec}"

    tdir="${OUTDIR}/${label}"
    caldir="${tdir}/calibration"

    mkdir -p "${tdir}" "${caldir}"
    rm -f "${tdir}"/*.log "${tdir}"/*.txt "${caldir}"/*.log "${caldir}"/*.dat 2>/dev/null || true
done

echo "phase,rep,position,topology,threads,backend,policy" > "${OUTDIR}/order.csv"

# ---------------------------------------------------------------------------
# Calibration: independently for every topology
# ---------------------------------------------------------------------------
set_param "${CAL_T}"

for spec in "${CONFIGS[@]}"; do
    IFS='|' read -r label threads physical_cores smt socket_scope cpus <<< "${spec}"

    tdir="${OUTDIR}/${label}"
    caldir="${tdir}/calibration"

    cost="${caldir}/heat2d_cost_model.dat"
    bw_wait="${caldir}/heat2d_wait_cost_busywait.dat"
    sm_wait="${caldir}/heat2d_wait_cost_semaphore.dat"

    echo
    echo "============================================================"
    echo "CALIBRATION: ${label}"
    echo "threads=${threads} physical_cores=${physical_cores} smt=${smt}"
    echo "scope=${socket_scope} CPUs=${cpus}"
    echo "============================================================"

    echo "[calibration] PREDICT/RECOMPUTE"
    run_pinned_env "${cpus}" "${caldir}/calibration_pr.log" \
        OMP_NUM_THREADS="${threads}" \
        OMP_PLACES="${OMP_PLACES}" \
        OMP_PROC_BIND="${OMP_PROC_BIND}" \
        OMP_DYNAMIC="${OMP_DYNAMIC}" \
        HEAT2D_MAX_LEAD="${HEAT2D_MAX_LEAD}" \
        HEAT2D_ENABLE_PREDICT=1 \
        HEAT2D_ENABLE_RECOMPUTE=1 \
        HEAT2D_CALIBRATION_SAMPLES="${CAL_SAMPLES}" \
        HEAT2D_COST_FILE="${cost}" \
        "${PR_CAL}"

    echo "[calibration] WAIT busy-wait"
    run_pinned_env "${cpus}" "${caldir}/calibration_wait_busywait.log" \
        OMP_NUM_THREADS="${threads}" \
        OMP_PLACES="${OMP_PLACES}" \
        OMP_PROC_BIND="${OMP_PROC_BIND}" \
        OMP_DYNAMIC="${OMP_DYNAMIC}" \
        HEAT2D_MAX_LEAD="${HEAT2D_MAX_LEAD}" \
        HEAT2D_WAIT_CALIBRATION_SAMPLES="${WAIT_CAL_SAMPLES}" \
        HEAT2D_WAIT_COST_FILE="${bw_wait}" \
        "${BW_WCAL}"

    echo "[calibration] WAIT semaphore"
    run_pinned_env "${cpus}" "${caldir}/calibration_wait_semaphore.log" \
        OMP_NUM_THREADS="${threads}" \
        OMP_PLACES="${OMP_PLACES}" \
        OMP_PROC_BIND="${OMP_PROC_BIND}" \
        OMP_DYNAMIC="${OMP_DYNAMIC}" \
        HEAT2D_MAX_LEAD="${HEAT2D_MAX_LEAD}" \
        HEAT2D_WAIT_CALIBRATION_SAMPLES="${WAIT_CAL_SAMPLES}" \
        HEAT2D_WAIT_COST_FILE="${sm_wait}" \
        "${SM_WCAL}"

    for f in "${cost}" "${bw_wait}" "${sm_wait}"; do
        [[ -s "${f}" ]] || {
            echo "ERRO: calibracao nao gerou ${f}" >&2
            exit 7
        }
    done
done

# Restore measured T.
set_param "${T_VALUE}"

for spec in "${CONFIGS[@]}"; do
    IFS='|' read -r label threads physical_cores smt socket_scope cpus <<< "${spec}"
    cp param.txt "${OUTDIR}/${label}/param.effective.txt"
done

# ---------------------------------------------------------------------------
# Warm-up
# ---------------------------------------------------------------------------
echo
echo "============================================================"
echo "WARM-UP"
echo "============================================================"

for ((w=1; w<=WARMUPS; ++w)); do
    config_shift=$(( (w - 1) % ${#CONFIGS[@]} ))

    for ((ci=0; ci<${#CONFIGS[@]}; ++ci)); do
        cidx=$(( (ci + config_shift) % ${#CONFIGS[@]} ))
        spec="${CONFIGS[$cidx]}"
        IFS='|' read -r label threads physical_cores smt socket_scope cpus <<< "${spec}"

        tdir="${OUTDIR}/${label}"
        caldir="${tdir}/calibration"
        cost="${caldir}/heat2d_cost_model.dat"
        bw_wait="${caldir}/heat2d_wait_cost_busywait.dat"
        sm_wait="${caldir}/heat2d_wait_cost_semaphore.dat"

        case_shift=$(( (w + ci - 1) % ${#CASES[@]} ))

        for ((ki=0; ki<${#CASES[@]}; ++ki)); do
            kidx=$(( (ki + case_shift) % ${#CASES[@]} ))
            IFS=: read -r backend policy <<< "${CASES[$kidx]}"

            if [[ "${backend}" == "busywait" ]]; then
                exe="${BW_REF}"
            else
                exe="${SM_REF}"
            fi

            log="${tdir}/${backend}_${policy}_warmup_r$(printf '%02d' "${w}").log"

            run_solver \
                "${exe}" "${backend}" "${policy}" "${log}" \
                "${threads}" "${cpus}" "${cost}" "${bw_wait}" "${sm_wait}"
        done
    done
done

# ---------------------------------------------------------------------------
# Clean timing runs
#
# The repetition number is a SUPER-BLOCK:
#   every topology is measured once in each rep.
# This makes cross-topology paired analysis more defensible under temporal drift.
# ---------------------------------------------------------------------------
echo
echo "============================================================"
echo "TIMING: ${REPS} rotated super-blocks"
echo "============================================================"

for ((rep=1; rep<=REPS; ++rep)); do
    config_shift=$(( (rep - 1) % ${#CONFIGS[@]} ))
    global_position=0

    for ((ci=0; ci<${#CONFIGS[@]}; ++ci)); do
        cidx=$(( (ci + config_shift) % ${#CONFIGS[@]} ))
        spec="${CONFIGS[$cidx]}"
        IFS='|' read -r label threads physical_cores smt socket_scope cpus <<< "${spec}"

        tdir="${OUTDIR}/${label}"
        caldir="${tdir}/calibration"
        cost="${caldir}/heat2d_cost_model.dat"
        bw_wait="${caldir}/heat2d_wait_cost_busywait.dat"
        sm_wait="${caldir}/heat2d_wait_cost_semaphore.dat"

        # Rotate the six cases independently from topology order.
        case_shift=$(( (rep + ci - 1) % ${#CASES[@]} ))

        for ((ki=0; ki<${#CASES[@]}; ++ki)); do
            kidx=$(( (ki + case_shift) % ${#CASES[@]} ))
            IFS=: read -r backend policy <<< "${CASES[$kidx]}"

            if [[ "${backend}" == "busywait" ]]; then
                exe="${BW_REF}"
            else
                exe="${SM_REF}"
            fi

            global_position=$((global_position + 1))
            echo "timing,${rep},${global_position},${label},${threads},${backend},${policy}" >> "${OUTDIR}/order.csv"

            log="${tdir}/${backend}_${policy}_ref_r$(printf '%02d' "${rep}").log"

            run_solver \
                "${exe}" "${backend}" "${policy}" "${log}" \
                "${threads}" "${cpus}" "${cost}" "${bw_wait}" "${sm_wait}"
        done
    done

    echo "timing super-block rep=${rep}/${REPS} complete"
done

# ---------------------------------------------------------------------------
# Profile runs
# ---------------------------------------------------------------------------
echo
echo "============================================================"
echo "PROFILING: ${PROFILE_REPS} rotated super-blocks"
echo "============================================================"

for ((rep=1; rep<=PROFILE_REPS; ++rep)); do
    config_shift=$(( (rep - 1) % ${#CONFIGS[@]} ))
    global_position=0

    for ((ci=0; ci<${#CONFIGS[@]}; ++ci)); do
        cidx=$(( (ci + config_shift) % ${#CONFIGS[@]} ))
        spec="${CONFIGS[$cidx]}"
        IFS='|' read -r label threads physical_cores smt socket_scope cpus <<< "${spec}"

        tdir="${OUTDIR}/${label}"
        caldir="${tdir}/calibration"
        cost="${caldir}/heat2d_cost_model.dat"
        bw_wait="${caldir}/heat2d_wait_cost_busywait.dat"
        sm_wait="${caldir}/heat2d_wait_cost_semaphore.dat"

        case_shift=$(( (rep + ci - 1) % ${#CASES[@]} ))

        for ((ki=0; ki<${#CASES[@]}; ++ki)); do
            kidx=$(( (ki + case_shift) % ${#CASES[@]} ))
            IFS=: read -r backend policy <<< "${CASES[$kidx]}"

            if [[ "${backend}" == "busywait" ]]; then
                exe="${BW_PROF}"
            else
                exe="${SM_PROF}"
            fi

            global_position=$((global_position + 1))
            echo "profile,${rep},${global_position},${label},${threads},${backend},${policy}" >> "${OUTDIR}/order.csv"

            log="${tdir}/${backend}_${policy}_prof_r$(printf '%02d' "${rep}").log"

            run_solver \
                "${exe}" "${backend}" "${policy}" "${log}" \
                "${threads}" "${cpus}" "${cost}" "${bw_wait}" "${sm_wait}"
        done
    done

    echo "profile super-block rep=${rep}/${PROFILE_REPS} complete"
done

# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------
python3 - "${OUTDIR}" "${REPS}" "${PROFILE_REPS}" "${T_VALUE}" <<'PY'
import csv
import glob
import os
import random
import re
import statistics
import sys

outdir = sys.argv[1]
reps = int(sys.argv[2])
profile_reps = int(sys.argv[3])
T = int(sys.argv[4])

TOPOLOGY = {
    'one_socket_no_smt_14t': {
        'threads': 14,
        'physical_cores': 14,
        'smt': 'no',
        'socket_scope': 'one_socket',
        'cpus': '0-13',
    },
    'one_socket_smt_28t': {
        'threads': 28,
        'physical_cores': 14,
        'smt': 'yes',
        'socket_scope': 'one_socket',
        'cpus': '0-13,28-41',
    },
    'two_socket_no_smt_28t': {
        'threads': 28,
        'physical_cores': 28,
        'smt': 'no',
        'socket_scope': 'two_socket',
        'cpus': '0-27',
    },
    'two_socket_smt_56t': {
        'threads': 56,
        'physical_cores': 28,
        'smt': 'yes',
        'socket_scope': 'two_socket',
        'cpus': '0-55',
    },
}

TOPOLOGY_ORDER = [
    'one_socket_no_smt_14t',
    'one_socket_smt_28t',
    'two_socket_no_smt_28t',
    'two_socket_smt_56t',
]

TIME_RE = re.compile(r'^Tempo\s*:\s*([-+0-9.eE]+)')
LINF_RE = re.compile(r'^Linf:\s*([-+0-9.eE]+)')

PROFILE_FIELDS = {
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

def parse_log(path, profile=False):
    x = {}

    if profile:
        for key in PROFILE_FIELDS.values():
            x[key] = 0.0

    with open(path, errors='replace') as f:
        for line in f:
            m = TIME_RE.match(line)
            if m:
                x['time_s'] = float(m.group(1))

            m = LINF_RE.match(line)
            if m:
                x['linf'] = float(m.group(1))

            if profile:
                for prefix, key in PROFILE_FIELDS.items():
                    if line.startswith(prefix + ':'):
                        token = line.split(':', 1)[1].strip().split()[0]
                        x[key] = float(token)
                        break

    if 'time_s' not in x or 'linf' not in x:
        raise RuntimeError(f'incomplete log: {path}')

    return x

def median(xs):
    return statistics.median(xs)

def percentile(sorted_values, p):
    if not sorted_values:
        raise ValueError('empty values')
    if len(sorted_values) == 1:
        return sorted_values[0]

    k = (len(sorted_values) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)

    if f == c:
        return sorted_values[f]

    return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)

def bootstrap_median_ci(values, seed, nboot=20000):
    rng = random.Random(seed)
    n = len(values)
    boots = []

    for _ in range(nboot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boots.append(statistics.median(sample))

    boots.sort()
    return percentile(boots, 0.025), percentile(boots, 0.975)

def paired_ratio(rows_num, rows_den, expected):
    num = {int(r['rep']): float(r['time_s']) for r in rows_num}
    den = {int(r['rep']): float(r['time_s']) for r in rows_den}
    common = sorted(set(num) & set(den))

    if len(common) != expected:
        raise RuntimeError(
            f'paired ratio expected {expected} reps, got {len(common)}'
        )

    return [num[i] / den[i] for i in common]

# -------------------------------------------------------------------------
# Timing CSV
# -------------------------------------------------------------------------
times = []

for topology in TOPOLOGY_ORDER:
    tdir = os.path.join(outdir, topology)

    for path in glob.glob(os.path.join(tdir, '*_ref_r*.log')):
        m = re.fullmatch(
            r'(busywait|semaphore)_(wait|progress|critical)_ref_r(\d+)\.log',
            os.path.basename(path)
        )
        if not m:
            continue

        backend, policy, rep = m.group(1), m.group(2), int(m.group(3))
        meta = TOPOLOGY[topology]

        times.append({
            'topology': topology,
            'threads': meta['threads'],
            'physical_cores': meta['physical_cores'],
            'smt': meta['smt'],
            'socket_scope': meta['socket_scope'],
            'cpus': meta['cpus'],
            'T': T,
            'backend': backend,
            'policy': policy,
            'rep': rep,
            **parse_log(path, profile=False),
        })

expected_groups = {}
for r in times:
    key = (r['topology'], r['backend'], r['policy'])
    expected_groups.setdefault(key, []).append(r)

expected_group_count = len(TOPOLOGY_ORDER) * 2 * 3
if len(expected_groups) != expected_group_count:
    raise SystemExit(
        f'Expected {expected_group_count} timing groups, found {len(expected_groups)}'
    )

for key, rr in expected_groups.items():
    if len(rr) != reps:
        raise SystemExit(
            f'Incomplete timing group {key}: {len(rr)}/{reps}'
        )

times_cols = [
    'topology', 'threads', 'physical_cores', 'smt', 'socket_scope', 'cpus',
    'T', 'backend', 'policy', 'rep', 'time_s', 'linf'
]

times_path = os.path.join(outdir, 'times.csv')
with open(times_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=times_cols)
    w.writeheader()
    w.writerows(sorted(
        times,
        key=lambda r: (
            TOPOLOGY_ORDER.index(r['topology']),
            r['backend'], r['policy'], r['rep']
        )
    ))

# -------------------------------------------------------------------------
# Profile CSV
# -------------------------------------------------------------------------
prof = []

for topology in TOPOLOGY_ORDER:
    tdir = os.path.join(outdir, topology)

    for path in glob.glob(os.path.join(tdir, '*_prof_r*.log')):
        m = re.fullmatch(
            r'(busywait|semaphore)_(wait|progress|critical)_prof_r(\d+)\.log',
            os.path.basename(path)
        )
        if not m:
            continue

        backend, policy, rep = m.group(1), m.group(2), int(m.group(3))
        meta = TOPOLOGY[topology]

        prof.append({
            'topology': topology,
            'threads': meta['threads'],
            'physical_cores': meta['physical_cores'],
            'smt': meta['smt'],
            'socket_scope': meta['socket_scope'],
            'cpus': meta['cpus'],
            'T': T,
            'backend': backend,
            'policy': policy,
            'rep': rep,
            **parse_log(path, profile=True),
        })

profile_groups = {}
for r in prof:
    key = (r['topology'], r['backend'], r['policy'])
    profile_groups.setdefault(key, []).append(r)

if len(profile_groups) != expected_group_count:
    raise SystemExit(
        f'Expected {expected_group_count} profile groups, found {len(profile_groups)}'
    )

for key, rr in profile_groups.items():
    if len(rr) != profile_reps:
        raise SystemExit(
            f'Incomplete profile group {key}: {len(rr)}/{profile_reps}'
        )

profile_cols = [
    'topology', 'threads', 'physical_cores', 'smt', 'socket_scope', 'cpus',
    'T', 'backend', 'policy', 'rep', 'time_s', 'linf',
    'recompute', 'wait', 'lead_wait', 'progress_blocked',
    'penalty_samples', 'penalty_nonzero_samples',
    'penalty_mean_ticks', 'penalty_nonzero_fraction',
    'critical_recompute', 'noncritical_recompute_rejected',
    'criticality_forced_wait'
]

profile_per_run_path = os.path.join(outdir, 'profile_per_run.csv')
with open(profile_per_run_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=profile_cols)
    w.writeheader()
    w.writerows(sorted(
        prof,
        key=lambda r: (
            TOPOLOGY_ORDER.index(r['topology']),
            r['backend'], r['policy'], r['rep']
        )
    ))

profile_summary = []

for key in sorted(
    profile_groups,
    key=lambda k: (TOPOLOGY_ORDER.index(k[0]), k[1], k[2])
):
    rr = profile_groups[key]
    topology, backend, policy = key
    meta = TOPOLOGY[topology]
    med = lambda field: median([float(r[field]) for r in rr])

    profile_summary.append({
        'topology': topology,
        'threads': meta['threads'],
        'physical_cores': meta['physical_cores'],
        'smt': meta['smt'],
        'socket_scope': meta['socket_scope'],
        'backend': backend,
        'policy': policy,
        'reps': len(rr),
        'median_profile_time_s': med('time_s'),
        'median_recompute': med('recompute'),
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
        'median_criticality_forced_wait':
            med('criticality_forced_wait'),
        'median_linf': med('linf'),
    })

profile_summary_path = os.path.join(outdir, 'profile_summary.csv')
with open(profile_summary_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(profile_summary[0].keys()))
    w.writeheader()
    w.writerows(profile_summary)

# -------------------------------------------------------------------------
# Policy summary within each topology/backend
# -------------------------------------------------------------------------
summary = []

for topology in TOPOLOGY_ORDER:
    meta = TOPOLOGY[topology]

    for backend in ('busywait', 'semaphore'):
        by_policy = {}

        for policy in ('wait', 'progress', 'critical'):
            rr = [
                r for r in times
                if r['topology'] == topology
                and r['backend'] == backend
                and r['policy'] == policy
            ]

            if len(rr) != reps:
                raise RuntimeError(
                    f'{topology}/{backend}/{policy}: {len(rr)}/{reps}'
                )

            by_policy[policy] = rr

        med_wait = median([r['time_s'] for r in by_policy['wait']])
        med_progress = median([r['time_s'] for r in by_policy['progress']])
        med_critical = median([r['time_s'] for r in by_policy['critical']])

        wait_over_progress = paired_ratio(
            by_policy['wait'], by_policy['progress'], reps
        )
        wait_over_critical = paired_ratio(
            by_policy['wait'], by_policy['critical'], reps
        )
        progress_over_critical = paired_ratio(
            by_policy['progress'], by_policy['critical'], reps
        )

        seed = (
            TOPOLOGY_ORDER.index(topology) * 100
            + (1 if backend == 'busywait' else 2)
        )

        ci_wp = bootstrap_median_ci(wait_over_progress, seed + 10)
        ci_wc = bootstrap_median_ci(wait_over_critical, seed + 20)
        ci_pc = bootstrap_median_ci(progress_over_critical, seed + 30)

        summary.append({
            'topology': topology,
            'threads': meta['threads'],
            'physical_cores': meta['physical_cores'],
            'smt': meta['smt'],
            'socket_scope': meta['socket_scope'],
            'backend': backend,
            'reps': reps,

            'median_wait_s': med_wait,
            'median_progress_s': med_progress,
            'median_critical_s': med_critical,

            'progress_speedup_vs_wait_ratio_of_medians':
                med_wait / med_progress,
            'critical_speedup_vs_wait_ratio_of_medians':
                med_wait / med_critical,
            'critical_speedup_vs_progress_ratio_of_medians':
                med_progress / med_critical,

            'median_paired_wait_over_progress':
                median(wait_over_progress),
            'progress_wins_vs_wait':
                sum(1 for x in wait_over_progress if x > 1.0),
            'bootstrap95_wait_over_progress_lo': ci_wp[0],
            'bootstrap95_wait_over_progress_hi': ci_wp[1],

            'median_paired_wait_over_critical':
                median(wait_over_critical),
            'critical_wins_vs_wait':
                sum(1 for x in wait_over_critical if x > 1.0),
            'bootstrap95_wait_over_critical_lo': ci_wc[0],
            'bootstrap95_wait_over_critical_hi': ci_wc[1],

            'median_paired_progress_over_critical':
                median(progress_over_critical),
            'critical_wins_vs_progress':
                sum(1 for x in progress_over_critical if x > 1.0),
            'bootstrap95_progress_over_critical_lo': ci_pc[0],
            'bootstrap95_progress_over_critical_hi': ci_pc[1],

            'linf_wait':
                median([r['linf'] for r in by_policy['wait']]),
            'linf_progress':
                median([r['linf'] for r in by_policy['progress']]),
            'linf_critical':
                median([r['linf'] for r in by_policy['critical']]),
        })

summary_path = os.path.join(outdir, 'summary.csv')
with open(summary_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    w.writeheader()
    w.writerows(summary)

# -------------------------------------------------------------------------
# SMT effect: same physical cores, SMT off -> SMT on.
#
# one socket:
#   14 physical cores, 14T no-SMT -> same 14 physical cores, 28T SMT
#
# two sockets:
#   28 physical cores, 28T no-SMT -> same 28 physical cores, 56T SMT
#
# Ratio > 1 means SMT configuration is faster.
# -------------------------------------------------------------------------
smt_pairs = [
    (
        'one_socket',
        'one_socket_no_smt_14t',
        'one_socket_smt_28t',
    ),
    (
        'two_socket',
        'two_socket_no_smt_28t',
        'two_socket_smt_56t',
    ),
]

smt_summary = []

for scope, no_smt_topology, smt_topology in smt_pairs:
    for backend in ('busywait', 'semaphore'):
        for policy in ('wait', 'progress', 'critical'):
            no_smt_rows = [
                r for r in times
                if r['topology'] == no_smt_topology
                and r['backend'] == backend
                and r['policy'] == policy
            ]
            smt_rows = [
                r for r in times
                if r['topology'] == smt_topology
                and r['backend'] == backend
                and r['policy'] == policy
            ]

            paired = paired_ratio(no_smt_rows, smt_rows, reps)
            seed = (
                (1 if scope == 'one_socket' else 2) * 1000
                + (1 if backend == 'busywait' else 2) * 100
                + {'wait': 1, 'progress': 2, 'critical': 3}[policy]
            )
            ci = bootstrap_median_ci(paired, seed)

            med_no_smt = median([r['time_s'] for r in no_smt_rows])
            med_smt = median([r['time_s'] for r in smt_rows])

            smt_summary.append({
                'socket_scope': scope,
                'backend': backend,
                'policy': policy,
                'reps': reps,
                'no_smt_topology': no_smt_topology,
                'smt_topology': smt_topology,
                'median_no_smt_s': med_no_smt,
                'median_smt_s': med_smt,
                'smt_speedup_ratio_of_medians': med_no_smt / med_smt,
                'median_paired_no_smt_over_smt': median(paired),
                'smt_wins': sum(1 for x in paired if x > 1.0),
                'bootstrap95_no_smt_over_smt_lo': ci[0],
                'bootstrap95_no_smt_over_smt_hi': ci[1],
            })

smt_summary_path = os.path.join(outdir, 'smt_summary.csv')
with open(smt_summary_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(smt_summary[0].keys()))
    w.writeheader()
    w.writerows(smt_summary)

print()
print('=== POLICY RESULTS: CRITICAL vs WAIT ===')
for r in summary:
    print(
        f"{r['topology']:24s} {r['backend']:9s} "
        f"S_med={r['critical_speedup_vs_wait_ratio_of_medians']:.5f} "
        f"S_paired={r['median_paired_wait_over_critical']:.5f} "
        f"wins={r['critical_wins_vs_wait']}/{r['reps']} "
        f"CI95=[{r['bootstrap95_wait_over_critical_lo']:.5f},"
        f"{r['bootstrap95_wait_over_critical_hi']:.5f}]"
    )

print()
print('=== SMT EFFECT ===')
for r in smt_summary:
    print(
        f"{r['socket_scope']:10s} {r['backend']:9s} {r['policy']:8s} "
        f"S_med={r['smt_speedup_ratio_of_medians']:.5f} "
        f"S_paired={r['median_paired_no_smt_over_smt']:.5f} "
        f"wins={r['smt_wins']}/{r['reps']} "
        f"CI95=[{r['bootstrap95_no_smt_over_smt_lo']:.5f},"
        f"{r['bootstrap95_no_smt_over_smt_hi']:.5f}]"
    )

print()
print('=== FILES ===')
print('times             :', times_path)
print('profile_per_run   :', profile_per_run_path)
print('profile_summary   :', profile_summary_path)
print('summary           :', summary_path)
print('smt_summary       :', smt_summary_path)
PY

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
{
    echo "date=$(date --iso-8601=seconds 2>/dev/null || date)"
    echo "host=$(hostname)"
    echo "machine=Antares"
    echo "cpu=Intel Xeon Gold 5120"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "memory_policy=system_default_first_touch"
    echo "N=${N_VALUE}"
    echo "T=${T_VALUE}"
    echo "TILE=${TILE_VALUE}"
    echo "REPS=${REPS}"
    echo "PROFILE_REPS=${PROFILE_REPS}"
    echo "WARMUPS=${WARMUPS}"
    echo "CAL_T=${CAL_T}"
    echo "CAL_SAMPLES=${CAL_SAMPLES}"
    echo "WAIT_CAL_SAMPLES=${WAIT_CAL_SAMPLES}"
    echo "PROGRESS_LAMBDA=${PROGRESS_LAMBDA}"
    echo "PROGRESS_BOOTSTRAP=${PROGRESS_BOOTSTRAP}"
    echo "OMP_PLACES=${OMP_PLACES}"
    echo "OMP_PROC_BIND=${OMP_PROC_BIND}"
    echo "PREDICT=0"
    echo "config_one_socket_no_smt_14t=threads14;physical_cores14;smt_no;cpus_0-13"
    echo "config_one_socket_smt_28t=threads28;physical_cores14;smt_yes;cpus_0-13,28-41"
    echo "config_two_socket_no_smt_28t=threads28;physical_cores28;smt_no;cpus_0-27"
    echo "config_two_socket_smt_56t=threads56;physical_cores28;smt_yes;cpus_0-55"
} > "${OUTDIR}/run_metadata.txt"

echo
echo "============================================================"
echo "Antares topology + SMT experiment complete."
echo "Results:"
echo "  ${OUTDIR}/times.csv"
echo "  ${OUTDIR}/profile_per_run.csv"
echo "  ${OUTDIR}/profile_summary.csv"
echo "  ${OUTDIR}/summary.csv"
echo "  ${OUTDIR}/smt_summary.csv"
echo "  ${OUTDIR}/order.csv"
echo "  ${OUTDIR}/run_metadata.txt"
echo "============================================================"
