#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Heat2D benchmark:
#   baseline
#   oracle
#   adaptive calibrated
#   adaptive online
#
# Methodology:
#   - 1 warm-up per variant
#   - 10 measured runs per variant by default
#   - variants are rotated across repetitions to reduce
#     systematic bias from machine drift
#   - median is the primary reported statistic
#
# Assumptions:
#   - param.txt is already configured for the experiment
#   - WRITE_OUTPUT = 0
#   - heat2d_cost_model.dat already exists and matches the
#     configuration used by adaptive_calibrated
# ============================================================

PARAM="${PARAM:-param.txt}"
COST_MODEL="${COST_MODEL:-heat2d_cost_model.dat}"

BASELINE="${BASELINE:-./heat2d_explicit_omp_mpilike}"
ORACLE="${ORACLE:-./heat2d_dependency_oracle}"
CALIBRATED="${CALIBRATED:-./heat2d_adaptive_calibrated}"
ONLINE="${ONLINE:-./heat2d_adaptive_online}"

REPS="${REPS:-10}"
OUTDIR="${OUTDIR:-benchmark_models_results}"

mkdir -p "${OUTDIR}"

RAW="${OUTDIR}/raw_results.csv"
SUMMARY="${OUTDIR}/summary.csv"

# ------------------------------------------------------------
# Checks
# ------------------------------------------------------------
if [[ ! -f "${PARAM}" ]]; then
    echo "ERRO: ${PARAM} não encontrado." >&2
    exit 1
fi

if [[ ! -f "${COST_MODEL}" ]]; then
    echo "ERRO: ${COST_MODEL} não encontrado." >&2
    echo "Execute o calibrador antes do benchmark." >&2
    exit 2
fi

for exe in "${BASELINE}" "${ORACLE}" "${CALIBRATED}" "${ONLINE}"; do
    if [[ ! -x "${exe}" ]]; then
        echo "ERRO: executável não encontrado: ${exe}" >&2
        exit 3
    fi
done

# Require WRITE_OUTPUT=0 for performance runs.
WRITE_OUTPUT_VALUE="$(
    awk -F= '
        /^[[:space:]]*WRITE_OUTPUT[[:space:]]*=/ {
            gsub(/[[:space:]]/, "", $2);
            print tolower($2);
            exit
        }
    ' "${PARAM}"
)"

case "${WRITE_OUTPUT_VALUE}" in
    0|off|false|no)
        ;;
    *)
        echo "ERRO: para benchmark, configure WRITE_OUTPUT = 0 em ${PARAM}." >&2
        exit 4
        ;;
esac

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
        exit 5
    fi

    local t
    t="$(extract_time "${log}")"

    if [[ -z "${t}" ]]; then
        echo "ERRO: não foi possível extrair Tempo de ${log}" >&2
        exit 6
    fi

    printf '%s\n' "${t}"
}

# ------------------------------------------------------------
# Environment display
# ------------------------------------------------------------
echo
echo "============================================================"
echo "Heat2D benchmark"
echo "============================================================"
echo "OMP_NUM_THREADS = ${OMP_NUM_THREADS:-not-set}"
echo "OMP_PLACES      = ${OMP_PLACES:-not-set}"
echo "OMP_PROC_BIND   = ${OMP_PROC_BIND:-not-set}"
echo "repetitions     = ${REPS}"
echo "cost model      = ${COST_MODEL}"
echo "results         = ${OUTDIR}"
echo "============================================================"

# ------------------------------------------------------------
# Warm-up
# ------------------------------------------------------------
echo
echo "[warm-up] baseline"
run_one "${BASELINE}" "${OUTDIR}/warmup_baseline.log" >/dev/null

echo "[warm-up] oracle"
run_one "${ORACLE}" "${OUTDIR}/warmup_oracle.log" >/dev/null

echo "[warm-up] adaptive calibrated"
run_one "${CALIBRATED}" "${OUTDIR}/warmup_calibrated.log" >/dev/null

echo "[warm-up] adaptive online"
run_one "${ONLINE}" "${OUTDIR}/warmup_online.log" >/dev/null

# ------------------------------------------------------------
# Measured runs
# ------------------------------------------------------------
echo "variant,rep,time_s" > "${RAW}"

# Rotate execution order by repetition to reduce systematic
# ordering/drift effects.
variants=(baseline oracle calibrated online)

run_variant() {
    local variant="$1"
    local rep="$2"
    local exe log t

    case "${variant}" in
        baseline)
            exe="${BASELINE}"
            ;;
        oracle)
            exe="${ORACLE}"
            ;;
        calibrated)
            exe="${CALIBRATED}"
            ;;
        online)
            exe="${ONLINE}"
            ;;
        *)
            echo "ERRO interno: variante inválida ${variant}" >&2
            exit 7
            ;;
    esac

    log="${OUTDIR}/${variant}_r${rep}.log"
    t="$(run_one "${exe}" "${log}")"
    echo "${variant},${rep},${t}" >> "${RAW}"
    echo "rep=${rep}/${REPS}  ${variant}=${t}s"
}

echo
echo "[execuções válidas]"

for ((rep=1; rep<=REPS; ++rep)); do
    shift_by=$(( (rep - 1) % 4 ))

    for ((k=0; k<4; ++k)); do
        idx=$(( (k + shift_by) % 4 ))
        run_variant "${variants[$idx]}" "${rep}"
    done
done

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------
python3 - "${RAW}" "${SUMMARY}" <<'PY'
import csv
import statistics
import sys
from collections import defaultdict

raw_path, summary_path = sys.argv[1], sys.argv[2]

data = defaultdict(list)

with open(raw_path, newline="") as f:
    for row in csv.DictReader(f):
        data[row["variant"]].append(float(row["time_s"]))

order = ["baseline", "oracle", "calibrated", "online"]

missing = [v for v in order if not data[v]]
if missing:
    raise SystemExit(f"Missing data for: {', '.join(missing)}")

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

def pct_over(a, b):
    return 100.0 * (a / b - 1.0)

def recovery(tb, ta, t0):
    denom = tb - t0
    if denom == 0.0:
        return float("nan")
    return (tb - ta) / denom

print()
print("================================================================================")
print("BENCHMARK SUMMARY — MEDIAN OF MEASURED RUNS")
print("================================================================================")
print(f'{"variant":>14} {"n":>5} {"median(s)":>14} {"mean(s)":>14} {"min(s)":>14} {"max(s)":>14}')
print("-" * 82)

for v in order:
    s = stats[v]
    print(
        f'{v:>14} {s["n"]:5d} '
        f'{s["median"]:14.8f} {s["mean"]:14.8f} '
        f'{s["min"]:14.8f} {s["max"]:14.8f}'
    )

print()
print("Relative to baseline:")
print(f"  oracle      : {t0/tb:.6f} x baseline  ({pct_over(t0,tb):+.3f}%)")
print(f"  calibrated  : {tc/tb:.6f} x baseline  ({pct_over(tc,tb):+.3f}%)")
print(f"  online      : {to/tb:.6f} x baseline  ({pct_over(to,tb):+.3f}%)")

print()
print("Dependency-resolution opportunity:")
print(f"  baseline median T_B : {tb:.9f} s")
print(f"  oracle median   T_0 : {t0:.9f} s")
print(f"  removable gap       : {tb - t0:.9f} s")
print(f"  empirical max speedup T_B/T_0 : {tb/t0:.6f} x")

print()
print("Fraction of baseline→oracle gap recovered:")
print(f"  calibrated : {recovery(tb, tc, t0):.6f}")
print(f"  online     : {recovery(tb, to, t0):.6f}")

fieldnames = [
    "variant", "n", "median_s", "mean_s", "min_s", "max_s",
    "ratio_to_baseline", "delta_vs_baseline_pct",
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
        "delta_vs_baseline_pct": pct_over(med, tb),
        "gap_recovered_fraction": (
            "" if v in ("baseline", "oracle")
            else recovery(tb, med, t0)
        ),
    })

with open(summary_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)

print()
print(f"Raw results : {raw_path}")
print(f"Summary     : {summary_path}")
PY

echo
echo "============================================================"
echo "BENCHMARK COMPLETED"
echo "============================================================"
