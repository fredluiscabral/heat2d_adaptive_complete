#!/bin/bash
#SBATCH --job-name=heat2d_c64_sem_confirm
#SBATCH --partition=lncc-cpu_amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=heat2d_c64_sem_confirm-%j.out
#SBATCH --error=heat2d_c64_sem_confirm-%j.err

set -euo pipefail

REPS="${REPS:-10}"
WARMUPS="${WARMUPS:-1}"
THREADS=64
CAL_SAMPLES="${CAL_SAMPLES:-32}"
WAIT_CAL_SAMPLES="${WAIT_CAL_SAMPLES:-32}"
PROGRESS_LAMBDA="${PROGRESS_LAMBDA:-1.0}"
PROGRESS_BOOTSTRAP="${PROGRESS_BOOTSTRAP:-8}"
OUTDIR="${OUTDIR:-results/criticality_confirm_64_semaphore}"

SOLVER="./heat2d_adaptive_sem"
PR_CAL="./heat2d_dependency_calibrate"
WAIT_CAL="./heat2d_wait_calibrate_semaphore"

cd "${SLURM_SUBMIT_DIR}"

export OMP_NUM_THREADS="${THREADS}"
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_DYNAMIC=FALSE
export HEAT2D_MAX_LEAD=2

[[ -f param.txt ]] || { echo "ERRO: param.txt nao encontrado." >&2; exit 1; }

get_param() {
    local key="$1"
    awk -F= -v k="$key" '
        $1 ~ "^[[:space:]]*" k "[[:space:]]*$" {
            gsub(/[[:space:]]/, "", $2); print $2; exit
        }
    ' param.txt
}

N_VALUE="$(get_param N)"
T_VALUE="$(get_param T)"
TILE_VALUE="$(get_param TILE)"
WRITE_OUTPUT_VALUE="$(get_param WRITE_OUTPUT | tr '[:upper:]' '[:lower:]')"

[[ "${N_VALUE}" == "8192" ]] || { echo "ERRO: este teste exige N=8192; atual=${N_VALUE:-ausente}." >&2; exit 2; }
[[ "${T_VALUE}" == "1000" ]] || { echo "ERRO: este teste exige T=1000; atual=${T_VALUE:-ausente}." >&2; exit 3; }
[[ "${TILE_VALUE}" == "32" ]] || { echo "ERRO: este teste exige TILE=32; atual=${TILE_VALUE:-ausente}." >&2; exit 4; }
case "${WRITE_OUTPUT_VALUE}" in
    0|off|false|no) ;;
    *) echo "ERRO: configure WRITE_OUTPUT=0 em param.txt." >&2; exit 5 ;;
esac

mkdir -p "${OUTDIR}"
rm -f "${OUTDIR}"/*.log "${OUTDIR}"/*.dat "${OUTDIR}"/*.csv 2>/dev/null || true

# Mantem o mesmo ambiente de compilacao/execucao usado nos testes anteriores.
make -j "${SLURM_CPUS_PER_TASK}" core

for exe in "${SOLVER}" "${PR_CAL}" "${WAIT_CAL}"; do
    [[ -x "${exe}" ]] || { echo "ERRO: executavel ausente: ${exe}" >&2; exit 6; }
done

COST_FILE="${OUTDIR}/heat2d_cost_model.dat"
WAIT_FILE="${OUTDIR}/heat2d_wait_cost_semaphore.dat"

printf '=== Confirmacao 64 threads / semaphore ===\n'
printf 'N=%s T=%s TILE=%s threads=%s reps=%s warmups=%s\n' \
       "${N_VALUE}" "${T_VALUE}" "${TILE_VALUE}" "${THREADS}" "${REPS}" "${WARMUPS}"

printf '[calibration] PREDICT/RECOMPUTE\n'
HEAT2D_ENABLE_PREDICT=1 \
HEAT2D_ENABLE_RECOMPUTE=1 \
HEAT2D_CALIBRATION_SAMPLES="${CAL_SAMPLES}" \
HEAT2D_COST_FILE="${COST_FILE}" \
    "${PR_CAL}" > "${OUTDIR}/calibration_pr.log" 2>&1

printf '[calibration] WAIT semaphore\n'
HEAT2D_WAIT_CALIBRATION_SAMPLES="${WAIT_CAL_SAMPLES}" \
HEAT2D_WAIT_COST_FILE="${WAIT_FILE}" \
    "${WAIT_CAL}" > "${OUTDIR}/calibration_wait_semaphore.log" 2>&1

run_case() {
    local policy="$1"
    local tag="$2"
    local recompute criticality

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
            exit 7
            ;;
    esac

    local log="${OUTDIR}/${policy}_${tag}.log"

    HEAT2D_ENABLE_PREDICT=0 \
    HEAT2D_ENABLE_RECOMPUTE="${recompute}" \
    HEAT2D_ENABLE_CRITICALITY="${criticality}" \
    HEAT2D_COST_FILE="${COST_FILE}" \
    HEAT2D_WAIT_COST_FILE="${WAIT_FILE}" \
    HEAT2D_PROGRESS_LAMBDA="${PROGRESS_LAMBDA}" \
    HEAT2D_PROGRESS_BOOTSTRAP_SAMPLES="${PROGRESS_BOOTSTRAP}" \
        "${SOLVER}" > "${log}" 2>&1

    grep -q '^Adaptive progress-aware: on$' "${log}" || {
        echo "ERRO: controlador progress-aware ausente em ${log}" >&2
        exit 8
    }

    if [[ "${criticality}" == "1" ]]; then
        grep -q '^Adaptive criticality-aware: on$' "${log}" || {
            echo "ERRO: criticality deveria estar ON em ${log}" >&2
            exit 9
        }
    else
        grep -q '^Adaptive criticality-aware: off$' "${log}" || {
            echo "ERRO: criticality deveria estar OFF em ${log}" >&2
            exit 10
        }
    fi

    grep -q '^Tempo[[:space:]]*:' "${log}" || {
        echo "ERRO: Tempo ausente em ${log}" >&2
        exit 11
    }
    grep -q '^Linf:' "${log}" || {
        echo "ERRO: Linf ausente em ${log}" >&2
        exit 12
    }
}

printf '[warm-up]\n'
for ((w=1; w<=WARMUPS; ++w)); do
    # Alterna tambem a ordem dos warm-ups.
    if (( w % 2 == 1 )); then
        run_case wait     "warmup_$(printf '%02d' "${w}")"
        run_case critical "warmup_$(printf '%02d' "${w}")"
    else
        run_case critical "warmup_$(printf '%02d' "${w}")"
        run_case wait     "warmup_$(printf '%02d' "${w}")"
    fi
done

printf '[10 measured paired runs]\n'
for ((rep=1; rep<=REPS; ++rep)); do
    rtag="r$(printf '%02d' "${rep}")"

    # Ordem alternada para reduzir viés de aquecimento/deriva temporal do nó.
    if (( rep % 2 == 1 )); then
        run_case wait     "${rtag}"
        run_case critical "${rtag}"
        order="wait->critical"
    else
        run_case critical "${rtag}"
        run_case wait     "${rtag}"
        order="critical->wait"
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

def parse_log(path):
    time_s = None
    linf = None
    with open(path, errors='replace') as f:
        for line in f:
            m = re.match(r'^Tempo\s*:\s*([-+0-9.eE]+)', line)
            if m:
                time_s = float(m.group(1))
            m = re.match(r'^Linf:\s*([-+0-9.eE]+)', line)
            if m:
                linf = float(m.group(1))
    if time_s is None or linf is None:
        raise RuntimeError(f'incomplete log: {path}')
    return time_s, linf

rows = []
wait_times = []
crit_times = []
for rep in range(1, reps + 1):
    wt, wl = parse_log(os.path.join(outdir, f'wait_r{rep:02d}.log'))
    ct, cl = parse_log(os.path.join(outdir, f'critical_r{rep:02d}.log'))
    if not math.isclose(wl, cl, rel_tol=0.0, abs_tol=1e-15):
        raise RuntimeError(f'Linf mismatch in rep {rep}: wait={wl}, critical={cl}')
    speedup = wt / ct
    rows.append({
        'rep': rep,
        'wait_time_s': wt,
        'critical_time_s': ct,
        'paired_speedup': speedup,
        'critical_gain_pct': (1.0 - ct / wt) * 100.0,
        'linf': wl,
    })
    wait_times.append(wt)
    crit_times.append(ct)

with open(os.path.join(outdir, 'times.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

med_wait = statistics.median(wait_times)
med_crit = statistics.median(crit_times)
speedup = med_wait / med_crit
gain = (1.0 - med_crit / med_wait) * 100.0
paired_wins = sum(c < w for w, c in zip(wait_times, crit_times))
paired_speedups = [w / c for w, c in zip(wait_times, crit_times)]

# Bootstrap pareado da razao de medianas. Mantem cada par da mesma repeticao.
rng = random.Random(20260902)
boot = []
idx = list(range(reps))
for _ in range(20000):
    sample = [rng.choice(idx) for _ in idx]
    bw = [wait_times[i] for i in sample]
    bc = [crit_times[i] for i in sample]
    boot.append(statistics.median(bw) / statistics.median(bc))
boot.sort()
lo = boot[int(0.025 * len(boot))]
hi = boot[int(0.975 * len(boot)) - 1]

summary = {
    'threads': 64,
    'backend': 'semaphore',
    'reps': reps,
    'median_wait_s': med_wait,
    'median_critical_s': med_crit,
    'speedup_wait_over_critical': speedup,
    'critical_gain_pct': gain,
    'paired_critical_wins': paired_wins,
    'median_paired_speedup': statistics.median(paired_speedups),
    'bootstrap95_speedup_lo': lo,
    'bootstrap95_speedup_hi': hi,
    'linf': rows[0]['linf'],
}

with open(os.path.join(outdir, 'summary.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(summary.keys()))
    w.writeheader()
    w.writerow(summary)

print('\n=== RESULTADO ===')
print(f"WAIT median      : {med_wait:.9f} s")
print(f"CRITICAL median  : {med_crit:.9f} s")
print(f"Speedup          : {speedup:.6f}x")
print(f"Ganho CRITICAL   : {gain:.3f}%")
print(f"Wins pareados     : {paired_wins}/{reps}")
print(f"Bootstrap 95% CI : [{lo:.6f}, {hi:.6f}] x")
print(f"Linf              : {rows[0]['linf']:.17g}")
PY

printf '\nArquivos principais:\n'
printf '  %s/summary.csv\n' "${OUTDIR}"
printf '  %s/times.csv\n' "${OUTDIR}"
