#!/bin/bash
#SBATCH --job-name=heat2d_models
#SBATCH --partition=lncc-cpu_amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=heat2d_models-%j.out
#SBATCH --error=heat2d_models-%j.err

set -euo pipefail

# ============================================================
# Heat2D dependency-model benchmark - SDumont II
#
# Self-contained workflow:
#   1. compile;
#   2. offline calibration on SDumont;
#   3. 1 warm-up per variant;
#   4. 10 valid runs per variant;
#   5. rotate variant order;
#   6. compute medians and relative metrics.
#
# Uses N, T, TILE, alpha, theta etc. from param.txt.
# Requires WRITE_OUTPUT = 0.
# ============================================================

REPS="${REPS:-10}"
OUTDIR="${OUTDIR:-benchmark_models_results_sd}"
COST_MODEL="heat2d_cost_model.dat"

BASELINE="./heat2d_explicit_omp_mpilike"
ORACLE="./heat2d_dependency_oracle"
CALIBRATOR="./heat2d_dependency_calibrate"
CALIBRATED="./heat2d_adaptive_calibrated"
ONLINE="./heat2d_adaptive_online"

cd "${SLURM_SUBMIT_DIR}"

echo "============================================================"
echo "SDUMONT HEAT2D BENCHMARK"
echo "============================================================"
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Node         : ${SLURM_NODELIST}"
echo "CPUs/task    : ${SLURM_CPUS_PER_TASK}"
echo "Submit dir   : ${SLURM_SUBMIT_DIR}"
echo "Repetitions  : ${REPS}"
echo "============================================================"

# ------------------------------------------------------------
# OpenMP
# ------------------------------------------------------------
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_DYNAMIC=FALSE

echo
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "OMP_PLACES=${OMP_PLACES}"
echo "OMP_PROC_BIND=${OMP_PROC_BIND}"
echo "OMP_DYNAMIC=${OMP_DYNAMIC}"

# ------------------------------------------------------------
# Basic checks
# ------------------------------------------------------------
if [[ ! -f param.txt ]]; then
    echo "ERRO: param.txt não encontrado." >&2
    exit 1
fi

if ! command -v g++ >/dev/null 2>&1; then
    echo "ERRO: g++ não está disponível no ambiente atual." >&2
    exit 2
fi

WRITE_OUTPUT_VALUE="$(
    awk -F= '
        /^[[:space:]]*WRITE_OUTPUT[[:space:]]*=/ {
            gsub(/[[:space:]]/, "", $2);
            print tolower($2);
            exit
        }
    ' param.txt
)"

case "${WRITE_OUTPUT_VALUE}" in
    0|off|false|no)
        ;;
    *)
        echo "ERRO: configure WRITE_OUTPUT = 0 em param.txt." >&2
        exit 3
        ;;
esac

echo
echo "============================================================"
echo "PARAMETERS"
echo "============================================================"
cat param.txt

# ------------------------------------------------------------
# Build
# ------------------------------------------------------------
echo
echo "============================================================"
echo "BUILD"
echo "============================================================"

make clean
make -j "${SLURM_CPUS_PER_TASK}"

for exe in \
    "${BASELINE}" "${ORACLE}" "${CALIBRATOR}" \
    "${CALIBRATED}" "${ONLINE}"
do
    if [[ ! -x "${exe}" ]]; then
        echo "ERRO: executável ausente após compilação: ${exe}" >&2
        exit 4
    fi
done

# ------------------------------------------------------------
# Clean previous run artifacts
# ------------------------------------------------------------
rm -f "${COST_MODEL}" output.txt adaptive_stats.txt
rm -rf "${OUTDIR}"
mkdir -p "${OUTDIR}"

# ------------------------------------------------------------
# Offline calibration
# ------------------------------------------------------------
echo
echo "============================================================"
echo "OFFLINE CALIBRATION"
echo "============================================================"

HEAT2D_CALIBRATION_SAMPLES=32 \
    "${CALIBRATOR}" \
    | tee "${OUTDIR}/calibration.log"

if [[ ! -s "${COST_MODEL}" ]]; then
    echo "ERRO: ${COST_MODEL} não foi gerado." >&2
    exit 5
fi

cp "${COST_MODEL}" "${OUTDIR}/heat2d_cost_model.dat"

echo
echo "Cost model used:"
cat "${COST_MODEL}"

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
extract_time() {
    local log="$1"
    awk '
        /^Tempo[[:space:]]*:/ {
            print $3
            exit
        }
    ' "${log}"
}

run_one() {
    local exe="$1"
    local log="$2"

    rm -f output.txt
    "${exe}" > "${log}" 2>&1

    if [[ -e output.txt ]]; then
        echo "ERRO: ${exe} criou output.txt com WRITE_OUTPUT=0." >&2
        exit 6
    fi

    local t
    t="$(extract_time "${log}")"

    if [[ -z "${t}" ]]; then
        echo "ERRO: Tempo não encontrado em ${log}." >&2
        exit 7
    fi

    printf '%s\n' "${t}"
}

run_variant() {
    local variant="$1"
    local rep="$2"
    local exe log t

    case "${variant}" in
        baseline)   exe="${BASELINE}" ;;
        oracle)     exe="${ORACLE}" ;;
        calibrated) exe="${CALIBRATED}" ;;
        online)     exe="${ONLINE}" ;;
        *)
            echo "ERRO interno: variante ${variant}" >&2
            exit 8
            ;;
    esac

    log="${OUTDIR}/${variant}_r${rep}.log"
    t="$(run_one "${exe}" "${log}")"
    echo "${variant},${rep},${t}" >> "${OUTDIR}/raw_results.csv"
    echo "rep=${rep}/${REPS}  ${variant}=${t}s"
}

# ------------------------------------------------------------
# Warm-up
# ------------------------------------------------------------
echo
echo "============================================================"
echo "WARM-UP"
echo "============================================================"

echo "[warm-up] baseline"
run_one "${BASELINE}" "${OUTDIR}/warmup_baseline.log" >/dev/null

echo "[warm-up] oracle"
run_one "${ORACLE}" "${OUTDIR}/warmup_oracle.log" >/dev/null

echo "[warm-up] calibrated"
run_one "${CALIBRATED}" "${OUTDIR}/warmup_calibrated.log" >/dev/null

echo "[warm-up] online"
run_one "${ONLINE}" "${OUTDIR}/warmup_online.log" >/dev/null

# ------------------------------------------------------------
# Valid repetitions
# ------------------------------------------------------------
echo "variant,rep,time_s" > "${OUTDIR}/raw_results.csv"

variants=(baseline oracle calibrated online)

echo
echo "============================================================"
echo "10 VALID RUNS PER VARIANT"
echo "============================================================"

for ((rep=1; rep<=REPS; ++rep)); do
    shift_by=$(( (rep - 1) % 4 ))

    for ((k=0; k<4; ++k)); do
        idx=$(( (k + shift_by) % 4 ))
        run_variant "${variants[$idx]}" "${rep}"
    done
done

# ------------------------------------------------------------
# Statistics
# ------------------------------------------------------------
python3 - "${OUTDIR}/raw_results.csv" "${OUTDIR}/summary.csv" <<'PY'
import csv
import statistics
import sys
from collections import defaultdict

raw_path, summary_path = sys.argv[1:3]

data = defaultdict(list)

with open(raw_path, newline="") as f:
    for row in csv.DictReader(f):
        data[row["variant"]].append(float(row["time_s"]))

order = ["baseline", "oracle", "calibrated", "online"]

for v in order:
    if not data[v]:
        raise SystemExit(f"Missing results for {v}")

stats = {}
for v in order:
    x = data[v]
    stats[v] = {
        "n": len(x),
        "median": statistics.median(x),
        "mean": statistics.mean(x),
        "min": min(x),
        "max": max(x),
    }

tb = stats["baseline"]["median"]
t0 = stats["oracle"]["median"]
tc = stats["calibrated"]["median"]
to = stats["online"]["median"]

def pct(a, b):
    return 100.0 * (a / b - 1.0)

def recovery(ta):
    d = tb - t0
    return float("nan") if d == 0 else (tb - ta) / d

print()
print("=" * 84)
print("FINAL BENCHMARK SUMMARY")
print("=" * 84)
print(f'{"variant":>14} {"n":>4} {"median(s)":>14} {"mean(s)":>14} {"min(s)":>14} {"max(s)":>14}')
print("-" * 82)

for v in order:
    s = stats[v]
    print(
        f'{v:>14} {s["n"]:4d} '
        f'{s["median"]:14.8f} {s["mean"]:14.8f} '
        f'{s["min"]:14.8f} {s["max"]:14.8f}'
    )

print()
print("Relative to baseline:")
print(f"  oracle     : {t0/tb:.6f}  ({pct(t0,tb):+.3f}%)")
print(f"  calibrated : {tc/tb:.6f}  ({pct(tc,tb):+.3f}%)")
print(f"  online     : {to/tb:.6f}  ({pct(to,tb):+.3f}%)")

print()
print("Baseline -> oracle opportunity:")
print(f"  T_B                  = {tb:.9f} s")
print(f"  T_0                  = {t0:.9f} s")
print(f"  T_B - T_0            = {tb-t0:.9f} s")
print(f"  empirical max speedup= {tb/t0:.6f} x")

print()
print("Gap recovered:")
print(f"  calibrated = {recovery(tc):.6f}")
print(f"  online     = {recovery(to):.6f}")

fields = [
    "variant","n","median_s","mean_s","min_s","max_s",
    "ratio_to_baseline","delta_vs_baseline_pct",
    "gap_recovered_fraction"
]

rows = []
for v in order:
    s = stats[v]
    med = s["median"]
    rows.append({
        "variant": v,
        "n": s["n"],
        "median_s": med,
        "mean_s": s["mean"],
        "min_s": s["min"],
        "max_s": s["max"],
        "ratio_to_baseline": med / tb,
        "delta_vs_baseline_pct": pct(med, tb),
        "gap_recovered_fraction": (
            "" if v in ("baseline", "oracle")
            else recovery(med)
        ),
    })

with open(summary_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
PY

echo
echo "============================================================"
echo "SUMMARY.CSV"
echo "============================================================"
cat "${OUTDIR}/summary.csv"

echo
echo "============================================================"
echo "JOB COMPLETED"
echo "============================================================"
echo "Results: ${OUTDIR}/"
