#!/bin/bash
#SBATCH --job-name=heat2d_imbalance
#SBATCH --partition=lncc-cpu_amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=heat2d_imbalance-%j.out
#SBATCH --error=heat2d_imbalance-%j.err

set -euo pipefail

REPS="${REPS:-3}"
THREADS_LIST_STR="${THREADS_LIST:-64 128 160 192}"
read -r -a THREADS_LIST <<< "${THREADS_LIST_STR}"
CAL_SAMPLES="${CAL_SAMPLES:-32}"
WAIT_CAL_SAMPLES="${WAIT_CAL_SAMPLES:-32}"
OUTDIR="${OUTDIR:-imbalance_results_sd}"
MAKEFILE="${MAKEFILE:-Makefile_imbalance}"

BW_EXE="./heat2d_imbalance_busywait"
SEM_EXE="./heat2d_imbalance_semaphore"
PR_CAL="./heat2d_dependency_calibrate"
BW_WAIT_CAL="./heat2d_wait_calibrate_busywait"
SEM_WAIT_CAL="./heat2d_wait_calibrate_semaphore"

cd "${SLURM_SUBMIT_DIR}"

export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_DYNAMIC=FALSE
export HEAT2D_ENABLE_PREDICT=0
export HEAT2D_MAX_LEAD=2

if [[ ! -f param.txt ]]; then
    echo "ERRO: param.txt nao encontrado." >&2
    exit 1
fi

WRITE_OUTPUT_VALUE="$(awk -F= '
    /^[[:space:]]*WRITE_OUTPUT[[:space:]]*=/ {
        gsub(/[[:space:]]/, "", $2); print tolower($2); exit
    }' param.txt)"
case "${WRITE_OUTPUT_VALUE}" in
    0|off|false|no) ;;
    *) echo "ERRO: configure WRITE_OUTPUT = 0 em param.txt." >&2; exit 2 ;;
esac

mkdir -p "${OUTDIR}"

printf '%s\n' "============================================================"
printf '%s\n' "HEAT2D - IMBALANCE / PROGRESS-SKEW DIAGNOSTIC"
printf '%s\n' "============================================================"
echo "Job ID           : ${SLURM_JOB_ID}"
echo "Node             : ${SLURM_NODELIST}"
echo "Threads          : ${THREADS_LIST[*]}"
echo "Repetitions      : ${REPS}"
echo "P/R cal samples  : ${CAL_SAMPLES}"
echo "WAIT cal samples : ${WAIT_CAL_SAMPLES}"
echo "OMP_PLACES       : ${OMP_PLACES}"
echo "OMP_PROC_BIND    : ${OMP_PROC_BIND}"
echo
echo "Parameters:"
cat param.txt

printf '\n%s\n' "================ BUILD ====================="
make -f "${MAKEFILE}" -j "${SLURM_CPUS_PER_TASK}" all

for exe in "${BW_EXE}" "${SEM_EXE}" "${PR_CAL}" "${BW_WAIT_CAL}" "${SEM_WAIT_CAL}"; do
    [[ -x "${exe}" ]] || { echo "ERRO: executavel ausente: ${exe}" >&2; exit 3; }
done

run_case() {
    local backend="$1"
    local policy="$2"
    local exe="$3"
    local threads="$4"
    local rep="$5"
    local pdir="$6"
    local cost_model="$7"
    local wait_model="$8"
    local warmup="${9:-0}"

    local recompute_enable=0
    if [[ "${policy}" == "recompute" ]]; then
        recompute_enable=1
    fi

    local suffix="r$(printf '%02d' "${rep}")"
    [[ "${warmup}" == "1" ]] && suffix="warmup"
    local log="${pdir}/${backend}_${policy}_${suffix}.log"

    rm -f output.txt
    HEAT2D_ENABLE_RECOMPUTE="${recompute_enable}" \
    HEAT2D_COST_FILE="${cost_model}" \
    HEAT2D_WAIT_COST_FILE="${wait_model}" \
        "${exe}" > "${log}" 2>&1

    if [[ -e output.txt ]]; then
        echo "ERRO: ${exe} criou output.txt com WRITE_OUTPUT=0." >&2
        exit 4
    fi

    if [[ "${policy}" == "recompute" ]]; then
        if ! grep -q '^Adaptive direct R/W choice: RECOMPUTE$' "${log}"; then
            echo "ERRO: modelo local nao escolheu RECOMPUTE para ${threads}/${backend}." >&2
            echo "Veja ${log}" >&2
            exit 5
        fi
    fi

    if [[ "${warmup}" == "0" ]]; then
        printf 'thr=%3d rep=%02d/%02d %-9s %-9s ' "${threads}" "${rep}" "${REPS}" "${backend}" "${policy}"
        grep -E '^(Tempo|LBE useful|Adaptive lead-guard waits|Lead-guard wait time sum|Neighbor skew avg|Neighbor skew max)' "${log}" \
            | tr '\n' ' '
        echo
    fi
}

for threads in "${THREADS_LIST[@]}"; do
    if (( threads < 1 || threads > SLURM_CPUS_PER_TASK )); then
        echo "ERRO: thread count invalido: ${threads}" >&2
        exit 6
    fi

    export OMP_NUM_THREADS="${threads}"
    tag="$(printf '%03d' "${threads}")"
    pdir="${OUTDIR}/threads_${tag}"
    mkdir -p "${pdir}"
    rm -f "${pdir}"/*.log "${pdir}"/*.dat 2>/dev/null || true

    printf '\n%s\n' "============================================================"
    echo "THREADS = ${threads}"
    printf '%s\n' "============================================================"

    cost_model="${pdir}/heat2d_cost_model.dat"
    bw_wait_model="${pdir}/heat2d_wait_cost_busywait.dat"
    sem_wait_model="${pdir}/heat2d_wait_cost_semaphore.dat"

    echo "[calibration] PREDICT/RECOMPUTE"
    rm -f output.txt
    HEAT2D_ENABLE_PREDICT=1 \
    HEAT2D_ENABLE_RECOMPUTE=1 \
    HEAT2D_COST_BOOTSTRAP_SAMPLES="${CAL_SAMPLES}" \
    HEAT2D_COST_FILE="${cost_model}" \
        "${PR_CAL}" > "${pdir}/calibration_pr.log" 2>&1
    rm -f output.txt
    [[ -s "${cost_model}" ]] || { echo "ERRO: modelo P/R nao gerado." >&2; exit 7; }

    echo "[calibration] WAIT busy-wait"
    HEAT2D_WAIT_CALIBRATION_SAMPLES="${WAIT_CAL_SAMPLES}" \
    HEAT2D_WAIT_COST_FILE="${bw_wait_model}" \
        "${BW_WAIT_CAL}" > "${pdir}/calibration_wait_busywait.log" 2>&1
    rm -f output.txt
    [[ -s "${bw_wait_model}" ]] || { echo "ERRO: modelo WAIT busy-wait nao gerado." >&2; exit 8; }

    echo "[calibration] WAIT semaphore"
    HEAT2D_WAIT_CALIBRATION_SAMPLES="${WAIT_CAL_SAMPLES}" \
    HEAT2D_WAIT_COST_FILE="${sem_wait_model}" \
        "${SEM_WAIT_CAL}" > "${pdir}/calibration_wait_semaphore.log" 2>&1
    rm -f output.txt
    [[ -s "${sem_wait_model}" ]] || { echo "ERRO: modelo WAIT semaphore nao gerado." >&2; exit 9; }

    echo "[warm-up] four diagnostic cases"
    run_case busywait  wait      "${BW_EXE}"  "${threads}" 0 "${pdir}" "${cost_model}" "${bw_wait_model}" 1
    run_case busywait  recompute "${BW_EXE}"  "${threads}" 0 "${pdir}" "${cost_model}" "${bw_wait_model}" 1
    run_case semaphore wait      "${SEM_EXE}" "${threads}" 0 "${pdir}" "${cost_model}" "${sem_wait_model}" 1
    run_case semaphore recompute "${SEM_EXE}" "${threads}" 0 "${pdir}" "${cost_model}" "${sem_wait_model}" 1

    for ((rep=1; rep<=REPS; ++rep)); do
        if (( rep % 2 == 1 )); then
            run_case busywait  wait      "${BW_EXE}"  "${threads}" "${rep}" "${pdir}" "${cost_model}" "${bw_wait_model}"
            run_case busywait  recompute "${BW_EXE}"  "${threads}" "${rep}" "${pdir}" "${cost_model}" "${bw_wait_model}"
            run_case semaphore wait      "${SEM_EXE}" "${threads}" "${rep}" "${pdir}" "${cost_model}" "${sem_wait_model}"
            run_case semaphore recompute "${SEM_EXE}" "${threads}" "${rep}" "${pdir}" "${cost_model}" "${sem_wait_model}"
        else
            run_case semaphore recompute "${SEM_EXE}" "${threads}" "${rep}" "${pdir}" "${cost_model}" "${sem_wait_model}"
            run_case semaphore wait      "${SEM_EXE}" "${threads}" "${rep}" "${pdir}" "${cost_model}" "${sem_wait_model}"
            run_case busywait  recompute "${BW_EXE}"  "${threads}" "${rep}" "${pdir}" "${cost_model}" "${bw_wait_model}"
            run_case busywait  wait      "${BW_EXE}"  "${threads}" "${rep}" "${pdir}" "${cost_model}" "${bw_wait_model}"
        fi
    done
done

python3 - "${OUTDIR}" "${REPS}" <<'PY'
import csv
import glob
import os
import re
import statistics
import sys

outdir = sys.argv[1]
expected_reps = int(sys.argv[2])

fields = {
    "Adaptive actions READ": "read",
    "Adaptive actions RECOMPUTE": "recompute",
    "Adaptive actions PREDICT": "predict",
    "Adaptive actions WAIT": "wait",
    "Adaptive lead-guard waits": "lead_wait",
    "Lead-guard wait time sum": "lead_wait_time_sum_s",
    "Useful time avg": "useful_avg_s",
    "Useful time max": "useful_max_s",
    "LBE useful": "lbe",
    "Neighbor skew samples": "skew_samples",
    "Neighbor skew avg": "skew_avg",
    "Neighbor skew max": "skew_max",
    "Neighbor skew frac 0": "skew_frac0",
    "Neighbor skew frac 1": "skew_frac1",
    "Neighbor skew frac >=2": "skew_frac2plus",
    "Linf": "linf",
}

int_fields = {"read", "recompute", "predict", "wait", "lead_wait", "skew_samples", "skew_max"}
rows = []

for path in sorted(glob.glob(os.path.join(outdir, "threads_*", "*_r*.log"))):
    base = os.path.basename(path)
    m = re.fullmatch(r"(busywait|semaphore)_(wait|recompute)_r(\d+)\.log", base)
    if not m:
        continue
    backend, policy, rep = m.group(1), m.group(2), int(m.group(3))
    threads = int(os.path.basename(os.path.dirname(path)).split("_")[-1])
    vals = {"threads": threads, "backend": backend, "policy": policy, "rep": rep}

    with open(path, errors="replace") as f:
        for line in f:
            if line.startswith("Tempo"):
                mt = re.match(r"^Tempo\s*:\s*([-+0-9.eE]+)", line)
                if mt:
                    vals["time_s"] = float(mt.group(1))
                continue
            for prefix, key in fields.items():
                if line.startswith(prefix + ":"):
                    token = line.split(":", 1)[1].strip().split()[0]
                    vals[key] = int(float(token)) if key in int_fields else float(token)
                    break

    required = ["time_s", *fields.values()]
    missing = [k for k in required if k not in vals]
    if missing:
        raise SystemExit(f"Missing fields in {path}: {missing}")
    rows.append(vals)

if not rows:
    raise SystemExit("No diagnostic logs found")

per_run_cols = [
    "threads", "backend", "policy", "rep", "time_s",
    "read", "recompute", "predict", "wait", "lead_wait", "lead_wait_time_sum_s",
    "useful_avg_s", "useful_max_s", "lbe",
    "skew_samples", "skew_avg", "skew_max", "skew_frac0", "skew_frac1", "skew_frac2plus",
    "linf",
]
per_run_path = os.path.join(outdir, "per_run.csv")
with open(per_run_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=per_run_cols)
    w.writeheader()
    w.writerows(rows)

groups = {}
for r in rows:
    groups.setdefault((r["threads"], r["backend"], r["policy"]), []).append(r)

summary_cols = [
    "threads", "backend", "policy", "reps",
    "median_time_s", "median_lbe", "median_lead_wait", "median_lead_wait_time_sum_s",
    "median_skew_avg", "median_skew_max", "median_skew_frac0", "median_skew_frac1",
    "median_skew_frac2plus", "median_recompute", "median_wait", "median_linf",
]
summary_rows = []
for (threads, backend, policy), grp in sorted(groups.items()):
    if len(grp) != expected_reps:
        raise SystemExit(
            f"Expected {expected_reps} reps for {threads}/{backend}/{policy}, found {len(grp)}"
        )
    med = lambda key: statistics.median(r[key] for r in grp)
    summary_rows.append({
        "threads": threads,
        "backend": backend,
        "policy": policy,
        "reps": len(grp),
        "median_time_s": med("time_s"),
        "median_lbe": med("lbe"),
        "median_lead_wait": med("lead_wait"),
        "median_lead_wait_time_sum_s": med("lead_wait_time_sum_s"),
        "median_skew_avg": med("skew_avg"),
        "median_skew_max": med("skew_max"),
        "median_skew_frac0": med("skew_frac0"),
        "median_skew_frac1": med("skew_frac1"),
        "median_skew_frac2plus": med("skew_frac2plus"),
        "median_recompute": med("recompute"),
        "median_wait": med("wait"),
        "median_linf": med("linf"),
    })

summary_path = os.path.join(outdir, "summary.csv")
with open(summary_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=summary_cols)
    w.writeheader()
    w.writerows(summary_rows)

print("\n" + "=" * 132)
print("IMBALANCE / PROGRESS-SKEW SUMMARY (medians across repetitions)")
print("=" * 132)
print(
    f'{"thr":>4} {"backend":>10} {"policy":>10} {"time(s)":>10} {"LBE":>9} '
    f'{"lead#":>10} {"lead_s(sum)":>12} {"skew_avg":>10} {"skew_max":>9} {"skew>=2%":>10} {"R":>10} {"W":>10}'
)
for r in summary_rows:
    print(
        f'{r["threads"]:4d} {r["backend"]:>10} {r["policy"]:>10} '
        f'{r["median_time_s"]:10.6f} {r["median_lbe"]:9.5f} '
        f'{r["median_lead_wait"]:10.0f} {r["median_lead_wait_time_sum_s"]:12.6f} '
        f'{r["median_skew_avg"]:10.5f} {r["median_skew_max"]:9.0f} '
        f'{100*r["median_skew_frac2plus"]:9.3f}% {r["median_recompute"]:10.0f} {r["median_wait"]:10.0f}'
    )
print("=" * 132)
print(f"Per-run CSV : {per_run_path}")
print(f"Summary CSV : {summary_path}")
PY

echo
echo "Imbalance diagnostic complete."
echo "Results: ${OUTDIR}/"
echo "  ${OUTDIR}/per_run.csv"
echo "  ${OUTDIR}/summary.csv"