#!/bin/bash
#SBATCH --job-name=heat2d_sd_confirm_48s64b
#SBATCH --partition=lncc-cpu_amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=heat2d_sd_confirm_48s64b-%j.out
#SBATCH --error=heat2d_sd_confirm_48s64b-%j.err

set -euo pipefail

# =============================================================================
# SDumont II — confirmacao focada do CRITICAL
#
# Casos confirmatorios:
#   1) 48 threads / semaphore
#   2) 64 threads / busy-wait
#
# Para cada caso:
#   - calibra PREDICT/RECOMPUTE para o numero de threads correspondente;
#   - calibra WAIT para o backend correspondente;
#   - PREDICT permanece OFF durante as medicoes;
#   - compara somente WAIT-only vs CRITICAL;
#   - 1 warm-up por politica;
#   - 20 repeticoes validas e pareadas;
#   - alterna a ordem WAIT/CRITICAL e a ordem dos dois casos;
#   - gera estatistica pareada e bootstrap 95% (20.000 reamostragens).
#
# Parametros numericos exigidos:
#   N=8192, T=500, TILE=32, WRITE_OUTPUT=0.
#
# Saida padrao:
#   results/sdumont_confirm_48sem_64bw_T500/
#       times.csv
#       summary.csv
#       run_metadata.txt
#       p048_semaphore/*.log, *.dat
#       p064_busywait/*.log, *.dat
# =============================================================================

REPS="${REPS:-20}"
WARMUPS="${WARMUPS:-1}"
CAL_SAMPLES="${CAL_SAMPLES:-32}"
WAIT_CAL_SAMPLES="${WAIT_CAL_SAMPLES:-32}"
PROGRESS_LAMBDA="${PROGRESS_LAMBDA:-1.0}"
PROGRESS_BOOTSTRAP="${PROGRESS_BOOTSTRAP:-8}"
OUTDIR="${OUTDIR:-results/sdumont_confirm_48sem_64bw_T500}"

BW_SOLVER="./heat2d_adaptive_busywait"
SM_SOLVER="./heat2d_adaptive_sem"
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

mkdir -p "${OUTDIR}"
P48_DIR="${OUTDIR}/p048_semaphore"
P64_DIR="${OUTDIR}/p064_busywait"
mkdir -p "${P48_DIR}" "${P64_DIR}"

# Evita mistura silenciosa com uma execucao anterior usando o mesmo OUTDIR.
rm -f "${OUTDIR}/times.csv" \
      "${OUTDIR}/summary.csv" \
      "${OUTDIR}/run_metadata.txt" 2>/dev/null || true
rm -f "${P48_DIR}"/*.log "${P48_DIR}"/*.dat 2>/dev/null || true
rm -f "${P64_DIR}"/*.log "${P64_DIR}"/*.dat 2>/dev/null || true

echo "============================================================"
echo "SDUMONT II — CONFIRMATORY CRITICALITY CAMPAIGN"
echo "============================================================"
echo "Job ID              : ${SLURM_JOB_ID}"
echo "Node                : ${SLURM_NODELIST}"
echo "CPUs/task           : ${SLURM_CPUS_PER_TASK}"
echo "Cases               : 48T semaphore; 64T busywait"
echo "Timed repetitions   : ${REPS}"
echo "Warm-ups/policy     : ${WARMUPS}"
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
    echo "reps=${REPS}"
    echo "warmups=${WARMUPS}"
    echo "cases=48:semaphore 64:busywait"
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
    echo "omp_proc_bind=${OMP_PROC_BIND}"
    echo "date_start=$(date --iso-8601=seconds)"
    echo "uname=$(uname -a)"
} > "${OUTDIR}/run_metadata.txt"

echo
echo "================ BUILD ======================================="
make -j "${SLURM_CPUS_PER_TASK}" core

for exe in "${BW_SOLVER}" "${SM_SOLVER}" "${PR_CAL}" "${BW_WCAL}" "${SM_WCAL}"; do
    [[ -x "${exe}" ]] || {
        echo "ERRO: executavel ausente: ${exe}" >&2
        exit 6
    }
done

check_common_output() {
    local log="$1"

    grep -q '^Tempo[[:space:]]*:' "${log}" || {
        echo "ERRO: Tempo ausente em ${log}" >&2
        exit 7
    }
    grep -q '^Linf:' "${log}" || {
        echo "ERRO: Linf ausente em ${log}" >&2
        exit 8
    }
}

run_adaptive() {
    local threads="$1"
    local backend="$2"
    local policy="$3"
    local tag="$4"
    local pdir="$5"
    local cost_file="$6"
    local wait_file="$7"

    local exe recompute criticality

    case "${backend}" in
        busywait)  exe="${BW_SOLVER}" ;;
        semaphore) exe="${SM_SOLVER}" ;;
        *)
            echo "ERRO interno: backend invalido: ${backend}" >&2
            exit 9
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
            exit 10
            ;;
    esac

    export OMP_NUM_THREADS="${threads}"

    local log="${pdir}/${backend}_${policy}_${tag}.log"

    HEAT2D_ENABLE_PREDICT=0 \
    HEAT2D_ENABLE_RECOMPUTE="${recompute}" \
    HEAT2D_ENABLE_CRITICALITY="${criticality}" \
    HEAT2D_COST_FILE="${cost_file}" \
    HEAT2D_WAIT_COST_FILE="${wait_file}" \
    HEAT2D_PROGRESS_LAMBDA="${PROGRESS_LAMBDA}" \
    HEAT2D_PROGRESS_BOOTSTRAP_SAMPLES="${PROGRESS_BOOTSTRAP}" \
        "${exe}" > "${log}" 2>&1

    check_common_output "${log}"

    grep -q '^Adaptive progress-aware: on$' "${log}" || {
        echo "ERRO: controlador progress-aware ausente em ${log}" >&2
        exit 11
    }
    grep -q '^Adaptive predict: off$' "${log}" || {
        echo "ERRO: PREDICT deveria estar OFF em ${log}" >&2
        exit 12
    }

    if [[ "${recompute}" == "1" ]]; then
        grep -q '^Adaptive recompute: on$' "${log}" || {
            echo "ERRO: RECOMPUTE deveria estar ON em ${log}" >&2
            exit 13
        }
    else
        grep -q '^Adaptive recompute: off$' "${log}" || {
            echo "ERRO: RECOMPUTE deveria estar OFF em ${log}" >&2
            exit 14
        }
    fi

    if [[ "${criticality}" == "1" ]]; then
        grep -q '^Adaptive criticality-aware: on$' "${log}" || {
            echo "ERRO: criticality deveria estar ON em ${log}" >&2
            exit 15
        }
    else
        grep -q '^Adaptive criticality-aware: off$' "${log}" || {
            echo "ERRO: criticality deveria estar OFF em ${log}" >&2
            exit 16
        }
    fi
}

calibrate_case() {
    local threads="$1"
    local backend="$2"
    local pdir="$3"
    local cost_file="$4"
    local wait_file="$5"

    export OMP_NUM_THREADS="${threads}"

    echo "[calibration] p=${threads} PREDICT/RECOMPUTE"
    HEAT2D_ENABLE_PREDICT=1 \
    HEAT2D_ENABLE_RECOMPUTE=1 \
    HEAT2D_CALIBRATION_SAMPLES="${CAL_SAMPLES}" \
    HEAT2D_COST_FILE="${cost_file}" \
        "${PR_CAL}" > "${pdir}/calibration_pr.log" 2>&1

    [[ -s "${cost_file}" ]] || {
        echo "ERRO: modelo P/R nao gerado para p=${threads}." >&2
        exit 17
    }

    echo "[calibration] p=${threads} WAIT ${backend}"
    case "${backend}" in
        busywait)
            HEAT2D_WAIT_CALIBRATION_SAMPLES="${WAIT_CAL_SAMPLES}" \
            HEAT2D_WAIT_COST_FILE="${wait_file}" \
                "${BW_WCAL}" > "${pdir}/calibration_wait_busywait.log" 2>&1
            ;;
        semaphore)
            HEAT2D_WAIT_CALIBRATION_SAMPLES="${WAIT_CAL_SAMPLES}" \
            HEAT2D_WAIT_COST_FILE="${wait_file}" \
                "${SM_WCAL}" > "${pdir}/calibration_wait_semaphore.log" 2>&1
            ;;
        *)
            echo "ERRO interno: backend invalido na calibracao: ${backend}" >&2
            exit 18
            ;;
    esac

    [[ -s "${wait_file}" ]] || {
        echo "ERRO: modelo WAIT nao gerado para p=${threads}/${backend}." >&2
        exit 19
    }
}

P48_COST="${P48_DIR}/heat2d_cost_model.dat"
P48_WAIT="${P48_DIR}/heat2d_wait_cost_semaphore.dat"
P64_COST="${P64_DIR}/heat2d_cost_model.dat"
P64_WAIT="${P64_DIR}/heat2d_wait_cost_busywait.dat"

echo
echo "================ CALIBRATION ================================="
calibrate_case 48 semaphore "${P48_DIR}" "${P48_COST}" "${P48_WAIT}"
calibrate_case 64 busywait  "${P64_DIR}" "${P64_COST}" "${P64_WAIT}"

echo
echo "================ WARM-UP ====================================="
for ((w=1; w<=WARMUPS; ++w)); do
    wtag="warmup_$(printf '%02d' "${w}")"

    # Alterna tanto a ordem das politicas quanto a ordem dos dois casos.
    if (( w % 2 == 1 )); then
        run_adaptive 48 semaphore wait     "${wtag}" "${P48_DIR}" "${P48_COST}" "${P48_WAIT}"
        run_adaptive 48 semaphore critical "${wtag}" "${P48_DIR}" "${P48_COST}" "${P48_WAIT}"
        run_adaptive 64 busywait  wait     "${wtag}" "${P64_DIR}" "${P64_COST}" "${P64_WAIT}"
        run_adaptive 64 busywait  critical "${wtag}" "${P64_DIR}" "${P64_COST}" "${P64_WAIT}"
    else
        run_adaptive 64 busywait  critical "${wtag}" "${P64_DIR}" "${P64_COST}" "${P64_WAIT}"
        run_adaptive 64 busywait  wait     "${wtag}" "${P64_DIR}" "${P64_COST}" "${P64_WAIT}"
        run_adaptive 48 semaphore critical "${wtag}" "${P48_DIR}" "${P48_COST}" "${P48_WAIT}"
        run_adaptive 48 semaphore wait     "${wtag}" "${P48_DIR}" "${P48_COST}" "${P48_WAIT}"
    fi

done

echo
echo "================ TIMING ======================================"
echo "[timing] ${REPS} paired repetitions"

for ((rep=1; rep<=REPS; ++rep)); do
    rtag="r$(printf '%02d' "${rep}")"

    # Mantem WAIT e CRITICAL adjacentes dentro de cada par.
    # A ordem interna do par e a ordem entre os dois casos alternam a cada rep.
    if (( rep % 2 == 1 )); then
        run_adaptive 48 semaphore wait     "${rtag}" "${P48_DIR}" "${P48_COST}" "${P48_WAIT}"
        run_adaptive 48 semaphore critical "${rtag}" "${P48_DIR}" "${P48_COST}" "${P48_WAIT}"
        run_adaptive 64 busywait  wait     "${rtag}" "${P64_DIR}" "${P64_COST}" "${P64_WAIT}"
        run_adaptive 64 busywait  critical "${rtag}" "${P64_DIR}" "${P64_COST}" "${P64_WAIT}"
        order="48:wait->critical | 64:wait->critical"
    else
        run_adaptive 64 busywait  critical "${rtag}" "${P64_DIR}" "${P64_COST}" "${P64_WAIT}"
        run_adaptive 64 busywait  wait     "${rtag}" "${P64_DIR}" "${P64_COST}" "${P64_WAIT}"
        run_adaptive 48 semaphore critical "${rtag}" "${P48_DIR}" "${P48_COST}" "${P48_WAIT}"
        run_adaptive 48 semaphore wait     "${rtag}" "${P48_DIR}" "${P48_COST}" "${P48_WAIT}"
        order="64:critical->wait | 48:critical->wait"
    fi

    echo "rep=${rep}/${REPS} complete (${order})"
done

python3 - "${OUTDIR}" "${REPS}" <<'PY'
import csv
import math
import os
import random
import re
import statistics
import sys

outdir = sys.argv[1]
reps = int(sys.argv[2])

TIME_RE = re.compile(r'^Tempo\s*:\s*([-+0-9.eE]+)')
LINF_RE = re.compile(r'^Linf:\s*([-+0-9.eE]+)')

CASES = [
    {
        'threads': 48,
        'backend': 'semaphore',
        'subdir': 'p048_semaphore',
        'seed': 4802,
    },
    {
        'threads': 64,
        'backend': 'busywait',
        'subdir': 'p064_busywait',
        'seed': 6401,
    },
]

def parse_log(path):
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
        raise RuntimeError(f'incomplete log: {path}')
    return time_s, linf


def percentile_bootstrap_median(values, seed, nboot=20000, alpha=0.05):
    rng = random.Random(seed)
    n = len(values)
    boot = []
    for _ in range(nboot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boot.append(statistics.median(sample))
    boot.sort()
    lo_idx = max(0, int((alpha / 2.0) * nboot))
    hi_idx = min(nboot - 1, int((1.0 - alpha / 2.0) * nboot) - 1)
    return boot[lo_idx], boot[hi_idx]


timing_rows = []
summary_rows = []

for case in CASES:
    threads = case['threads']
    backend = case['backend']
    pdir = os.path.join(outdir, case['subdir'])

    wait_times = []
    crit_times = []
    paired_speedups = []
    linfs = []

    for rep in range(1, reps + 1):
        tag = f'r{rep:02d}'
        wait_path = os.path.join(pdir, f'{backend}_wait_{tag}.log')
        crit_path = os.path.join(pdir, f'{backend}_critical_{tag}.log')

        wt, wl = parse_log(wait_path)
        ct, cl = parse_log(crit_path)

        if not math.isclose(wl, cl, rel_tol=0.0, abs_tol=1e-15):
            raise RuntimeError(
                f'Linf mismatch p={threads}/{backend} rep={rep}: '
                f'wait={wl}, critical={cl}'
            )

        speedup = wt / ct
        gain_pct = (1.0 - ct / wt) * 100.0

        timing_rows.append({
            'threads': threads,
            'backend': backend,
            'rep': rep,
            'wait_time_s': wt,
            'critical_time_s': ct,
            'paired_speedup_wait_over_critical': speedup,
            'critical_gain_pct': gain_pct,
            'linf': wl,
        })

        wait_times.append(wt)
        crit_times.append(ct)
        paired_speedups.append(speedup)
        linfs.append(wl)

    med_wait = statistics.median(wait_times)
    med_crit = statistics.median(crit_times)
    ratio_of_medians = med_wait / med_crit
    median_paired = statistics.median(paired_speedups)
    wins = sum(x > 1.0 for x in paired_speedups)
    lo, hi = percentile_bootstrap_median(
        paired_speedups,
        seed=20260903 + case['seed'],
        nboot=20000,
    )

    summary_rows.append({
        'threads': threads,
        'backend': backend,
        'reps': reps,
        'median_wait_s': med_wait,
        'median_critical_s': med_crit,
        'speedup_wait_over_critical_ratio_of_medians': ratio_of_medians,
        'critical_gain_pct_ratio_of_medians': (1.0 - med_crit / med_wait) * 100.0,
        'median_paired_wait_over_critical': median_paired,
        'critical_wins_vs_wait': wins,
        'bootstrap95_paired_speedup_lo': lo,
        'bootstrap95_paired_speedup_hi': hi,
        'min_paired_speedup': min(paired_speedups),
        'max_paired_speedup': max(paired_speedups),
        'linf': statistics.median(linfs),
    })

with open(os.path.join(outdir, 'times.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(timing_rows[0].keys()))
    w.writeheader()
    w.writerows(timing_rows)

with open(os.path.join(outdir, 'summary.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
    w.writeheader()
    w.writerows(summary_rows)

print()
print('=== CONFIRMATORY RESULTS: CRITICAL vs WAIT-only ===')
for r in summary_rows:
    print(
        f"p={r['threads']:3d} {r['backend']:9s} "
        f"S_med={r['speedup_wait_over_critical_ratio_of_medians']:.6f} "
        f"S_paired={r['median_paired_wait_over_critical']:.6f} "
        f"wins={r['critical_wins_vs_wait']}/{r['reps']} "
        f"CI95=[{r['bootstrap95_paired_speedup_lo']:.6f},"
        f"{r['bootstrap95_paired_speedup_hi']:.6f}] "
        f"gain_med={r['critical_gain_pct_ratio_of_medians']:.3f}%"
    )
PY

{
    echo "date_end=$(date --iso-8601=seconds)"
} >> "${OUTDIR}/run_metadata.txt"

echo
echo "============================================================"
echo "SDumont focused confirmatory campaign complete."
echo "Main files:"
echo "  ${OUTDIR}/summary.csv"
echo "  ${OUTDIR}/times.csv"
echo "  ${OUTDIR}/run_metadata.txt"
echo "============================================================"
