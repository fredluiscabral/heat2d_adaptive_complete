#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Heat2D local robustness/performance check
#
# Fixed:
#   N = 8192
#
# Sweep:
#   T = 100, 1000, 10000
#
# For each T:
#   - 1 warm-up baseline
#   - 1 warm-up adaptive
#   - 10 measured baseline runs
#   - 10 measured adaptive runs
#   - median reported at the end
#
# The measured runs are interleaved:
#   baseline, adaptive, baseline, adaptive, ...
#
# Uses all other values already present in param.txt.
# ============================================================

BASELINE="${BASELINE:-./heat2d_explicit_omp_mpilike}"
ADAPTIVE="${ADAPTIVE:-./heat2d_adaptive_perf}"
PARAM="${PARAM:-param.txt}"

N_FIXED="${N_FIXED:-8192}"
T_VALUES=(100 1000 10000)
REPS="${REPS:-10}"

OUTDIR="${OUTDIR:-long_T_results}"
RAW="${OUTDIR}/raw_results.csv"
SUMMARY="${OUTDIR}/summary.csv"

mkdir -p "${OUTDIR}"

if [[ ! -x "${BASELINE}" ]]; then
    echo "ERRO: baseline não encontrado: ${BASELINE}" >&2
    exit 1
fi

if [[ ! -x "${ADAPTIVE}" ]]; then
    echo "ERRO: adaptativo não encontrado: ${ADAPTIVE}" >&2
    exit 1
fi

if [[ ! -f "${PARAM}" ]]; then
    echo "ERRO: ${PARAM} não encontrado." >&2
    exit 1
fi

# ------------------------------------------------------------
# Preserve original param.txt
# ------------------------------------------------------------
BACKUP="${PARAM}.bak_longT_$$"
cp "${PARAM}" "${BACKUP}"

restore_param() {
    mv -f "${BACKUP}" "${PARAM}"
}
trap restore_param EXIT

# ------------------------------------------------------------
# Change one key in param.txt, preserving everything else
# ------------------------------------------------------------
set_param_value() {
    local key="$1"
    local value="$2"

    if grep -Eq "^[[:space:]]*${key}[[:space:]]*=" "${PARAM}"; then
        sed -i -E \
            "s|^[[:space:]]*${key}[[:space:]]*=.*$|${key} = ${value}|" \
            "${PARAM}"
    else
        echo "${key} = ${value}" >> "${PARAM}"
    fi
}

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
extract_time() {
    local log="$1"
    awk '/^Tempo[[:space:]]*:/ {print $3; exit}' "${log}"
}

extract_field() {
    local pattern="$1"
    local log="$2"
    awk -v pat="${pattern}" '
        index($0, pat) == 1 {
            print $NF
            exit
        }
    ' "${log}"
}

cleanup_output() {
    rm -f output.txt adaptive_stats.txt
}

run_one() {
    local exe="$1"
    local log="$2"

    "${exe}" > "${log}"
    cleanup_output

    local t
    t="$(extract_time "${log}")"

    if [[ -z "${t}" ]]; then
        echo "ERRO: não foi possível extrair Tempo de ${log}" >&2
        exit 10
    fi

    printf '%s\n' "${t}"
}

# ------------------------------------------------------------
# Force N = 8192
# ------------------------------------------------------------
set_param_value N "${N_FIXED}"

# ------------------------------------------------------------
# CSV
# ------------------------------------------------------------
echo "T,variant,rep,time_s,cost_models_ready,cost_models_total,tryP_median_cycles,recompute_median_cycles,acceptance_median,periodic_samples,tryP_R_ratio" > "${RAW}"

echo
echo "============================================================"
echo "Heat2D - T sweep"
echo "============================================================"
echo "N       = ${N_FIXED}"
echo "T       = ${T_VALUES[*]}"
echo "warm-up = 1 por variante"
echo "reps    = ${REPS} por variante"
echo "============================================================"

for T in "${T_VALUES[@]}"; do

    set_param_value T "${T}"

    echo
    echo "============================================================"
    echo "N=${N_FIXED}  T=${T}"
    echo "============================================================"

    echo
    echo "[warm-up baseline]"
    run_one "${BASELINE}" "${OUTDIR}/warmup_baseline_T${T}.log" >/dev/null

    echo "[warm-up adaptive]"
    run_one "${ADAPTIVE}" "${OUTDIR}/warmup_adaptive_T${T}.log" >/dev/null

    echo
    echo "[10 execuções válidas]"

    for ((r=1; r<=REPS; ++r)); do

        B_LOG="${OUTDIR}/baseline_T${T}_r${r}.log"
        A_LOG="${OUTDIR}/adaptive_T${T}_r${r}.log"

        BT="$(run_one "${BASELINE}" "${B_LOG}")"
        echo "${T},baseline,${r},${BT},,,,,,," >> "${RAW}"

        AT="$(run_one "${ADAPTIVE}" "${A_LOG}")"

        READY_LINE="$(grep -m1 '^Adaptive cost models ready:' "${A_LOG}" || true)"
        if [[ -n "${READY_LINE}" ]]; then
            READY="$(awk '{print $(NF-2)}' <<< "${READY_LINE}")"
            TOTAL="$(awk '{print $NF}' <<< "${READY_LINE}")"
        else
            READY=""
            TOTAL=""
        fi

        TRYP="$(extract_field "Adaptive learned try-PREDICT median:" "${A_LOG}")"
        REC="$(extract_field "Adaptive learned RECOMPUTE median:" "${A_LOG}")"
        ACC="$(extract_field "Adaptive learned PREDICT acceptance median:" "${A_LOG}")"
        PERIODIC="$(extract_field "Adaptive periodic cost samples:" "${A_LOG}")"
        RATIO="$(extract_field "Adaptive learned tryP/R ratio:" "${A_LOG}")"

        echo "${T},adaptive,${r},${AT},${READY},${TOTAL},${TRYP},${REC},${ACC},${PERIODIC},${RATIO}" >> "${RAW}"

        echo "T=${T} rep=${r}/${REPS}  baseline=${BT}s  adaptive=${AT}s"
    done
done

# ------------------------------------------------------------
# Statistics
# ------------------------------------------------------------
python3 - "${RAW}" "${SUMMARY}" <<'PY'
import csv
import math
import statistics
import sys
from collections import defaultdict

raw_path, summary_path = sys.argv[1], sys.argv[2]

times = defaultdict(list)
adaptive_meta = defaultdict(lambda: defaultdict(list))

with open(raw_path, newline="") as f:
    for row in csv.DictReader(f):
        T = int(row["T"])
        variant = row["variant"]
        times[(T, variant)].append(float(row["time_s"]))

        if variant == "adaptive":
            fields = {
                "cost_models_ready": int,
                "cost_models_total": int,
                "tryP_median_cycles": float,
                "recompute_median_cycles": float,
                "acceptance_median": float,
                "periodic_samples": int,
                "tryP_R_ratio": float,
            }
            for key, typ in fields.items():
                val = row.get(key, "")
                if val not in ("", None):
                    adaptive_meta[T][key].append(typ(val))

T_values = sorted({T for T, _ in times.keys()})

summary_rows = []

print()
print("==========================================================================")
print("SUMMARY - MEDIAN OF 10 MEASURED RUNS")
print("==========================================================================")
print(
    f'{"T":>8}  {"baseline(s)":>14}  {"adaptive(s)":>14}  '
    f'{"A/B":>9}  {"delta":>10}  {"gain B/A":>10}'
)
print("-" * 78)

for T in T_values:
    b = times[(T, "baseline")]
    a = times[(T, "adaptive")]

    mb = statistics.median(b)
    ma = statistics.median(a)

    ratio = ma / mb
    delta = 100.0 * (ratio - 1.0)
    gain = mb / ma

    row = {
        "T": T,
        "baseline_median_s": mb,
        "adaptive_median_s": ma,
        "adaptive_over_baseline": ratio,
        "delta_pct": delta,
        "baseline_over_adaptive": gain,
    }

    # Add medians of learned-cost metadata when available.
    for key in (
        "cost_models_ready",
        "cost_models_total",
        "tryP_median_cycles",
        "recompute_median_cycles",
        "acceptance_median",
        "periodic_samples",
        "tryP_R_ratio",
    ):
        vals = adaptive_meta[T].get(key, [])
        row[key + "_median"] = statistics.median(vals) if vals else ""

    summary_rows.append(row)

    print(
        f"{T:8d}  {mb:14.8f}  {ma:14.8f}  "
        f"{ratio:9.4f}  {delta:+9.2f}%  {gain:10.4f}"
    )

print()
print("Adaptive learned-cost medians:")
print(
    f'{"T":>8}  {"ready":>9}  {"tryP cyc":>14}  {"R cyc":>14}  '
    f'{"accept":>9}  {"periodic":>10}  {"tryP/R":>9}'
)
print("-" * 84)

for row in summary_rows:
    ready = row["cost_models_ready_median"]
    total = row["cost_models_total_median"]

    if ready != "" and total != "":
        ready_text = f"{int(ready)}/{int(total)}"
    else:
        ready_text = "-"

    def fmt(v, pattern):
        return pattern.format(v) if v != "" else "-"

    print(
        f'{row["T"]:8d}  {ready_text:>9}  '
        f'{fmt(row["tryP_median_cycles_median"], "{:.1f}"):>14}  '
        f'{fmt(row["recompute_median_cycles_median"], "{:.1f}"):>14}  '
        f'{fmt(row["acceptance_median_median"], "{:.4f}"):>9}  '
        f'{fmt(row["periodic_samples_median"], "{:.0f}"):>10}  '
        f'{fmt(row["tryP_R_ratio_median"], "{:.4f}"):>9}'
    )

fieldnames = [
    "T",
    "baseline_median_s",
    "adaptive_median_s",
    "adaptive_over_baseline",
    "delta_pct",
    "baseline_over_adaptive",
    "cost_models_ready_median",
    "cost_models_total_median",
    "tryP_median_cycles_median",
    "recompute_median_cycles_median",
    "acceptance_median_median",
    "periodic_samples_median",
    "tryP_R_ratio_median",
]

with open(summary_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(summary_rows)

print()
print("Raw results :", raw_path)
print("Summary     :", summary_path)
PY

echo
echo "============================================================"
echo "TEST COMPLETED"
echo "============================================================"
