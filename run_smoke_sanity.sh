#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Heat2D - smoke test + numerical sanity/convergence check
#
# Expected executables:
#   ./heat2d_explicit_omp_mpilike
#   ./heat2d_dependency_oracle
#   ./heat2d_dependency_calibrate
#   ./heat2d_adaptive_calibrated
#   ./heat2d_adaptive_online
#
# The script:
#   1) builds the project;
#   2) runs a smoke test on all executables;
#   3) checks WRITE_OUTPUT=0 (output.txt must not be created);
#   4) checks that the calibrator creates heat2d_cost_model.dat;
#   5) runs a convergence sanity test for:
#        baseline
#        adaptive_calibrated
#        adaptive_online
#   6) keeps approximately the same physical final time while
#      refining the mesh by scaling T ~ (N-1)^2;
#   7) computes observed orders in L1, L2 and Linf.
#
# Oracle is smoke-tested only. It is intentionally NOT included
# in numerical convergence checks.
# ============================================================

ROOT="$(pwd)"
PARAM="${PARAM:-param.txt}"

BASELINE="${BASELINE:-./heat2d_explicit_omp_mpilike}"
ORACLE="${ORACLE:-./heat2d_dependency_oracle}"
CALIBRATOR="${CALIBRATOR:-./heat2d_dependency_calibrate}"
CALIBRATED="${CALIBRATED:-./heat2d_adaptive_calibrated}"
ONLINE="${ONLINE:-./heat2d_adaptive_online}"

THREADS="${THREADS:-4}"

SMOKE_N="${SMOKE_N:-512}"
SMOKE_T="${SMOKE_T:-100}"
SMOKE_TILE="${SMOKE_TILE:-32}"

# Convergence grid sequence.
N_VALUES=(${N_VALUES:-128 256 512 1024})

# T used at the first N in N_VALUES.
# For the other meshes:
#   T(N) = BASE_T * ((N-1)/(N0-1))^2
# so final_time stays approximately constant because dt ~ h^2.
BASE_T="${BASE_T:-100}"
SANITY_TILE="${SANITY_TILE:-32}"

# Pass/fail criteria.
MIN_ORDER="${MIN_ORDER:-1.70}"
MAX_ADAPTIVE_ERROR_RATIO="${MAX_ADAPTIVE_ERROR_RATIO:-1.20}"

# Cost calibration samples used by the test.
CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-8}"

OUTDIR="${OUTDIR:-smoke_sanity_results}"
COST_MODEL="${COST_MODEL:-heat2d_cost_model.dat}"

mkdir -p "${OUTDIR}"

# ------------------------------------------------------------
# Preserve files that this test modifies.
# ------------------------------------------------------------
if [[ ! -f "${PARAM}" ]]; then
    echo "ERRO: ${PARAM} não encontrado." >&2
    exit 1
fi

PARAM_BACKUP="$(mktemp)"
cp "${PARAM}" "${PARAM_BACKUP}"

COST_BACKUP=""
HAD_COST_MODEL=0
if [[ -f "${COST_MODEL}" ]]; then
    COST_BACKUP="$(mktemp)"
    cp "${COST_MODEL}" "${COST_BACKUP}"
    HAD_COST_MODEL=1
fi

cleanup() {
    cp "${PARAM_BACKUP}" "${PARAM}"
    rm -f "${PARAM_BACKUP}"

    if [[ "${HAD_COST_MODEL}" -eq 1 ]]; then
        cp "${COST_BACKUP}" "${COST_MODEL}"
        rm -f "${COST_BACKUP}"
    else
        rm -f "${COST_MODEL}"
    fi

    # This script must never leave the huge field-output file behind.
    rm -f output.txt
}
trap cleanup EXIT

# ------------------------------------------------------------
# Param-file helper.
# Accepts lines such as:
#   N = 8192
#   T=1000
# ------------------------------------------------------------
set_param() {
    local key="$1"
    local value="$2"

    if grep -Eq "^[[:space:]]*${key}[[:space:]]*=" "${PARAM}"; then
        sed -i -E \
            "s|^[[:space:]]*${key}[[:space:]]*=.*$|${key} = ${value}|" \
            "${PARAM}"
    else
        printf '%s = %s\n' "${key}" "${value}" >> "${PARAM}"
    fi
}

# ------------------------------------------------------------
# Output parsers.
# ------------------------------------------------------------
extract_scalar() {
    local key="$1"
    local file="$2"

    awk -v key="${key}" '
        index($0, key) == 1 {
            for (i = 1; i <= NF; ++i) {
                if ($i ~ /^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$/) {
                    value = $i
                }
            }
            if (value != "") {
                print value
                exit
            }
        }
    ' "${file}"
}

require_scalar() {
    local key="$1"
    local file="$2"
    local value
    value="$(extract_scalar "${key}" "${file}")"
    if [[ -z "${value}" ]]; then
        echo "ERRO: campo '${key}' não encontrado em ${file}" >&2
        exit 20
    fi
    printf '%s\n' "${value}"
}

check_no_output_file() {
    local where="$1"
    if [[ -e output.txt ]]; then
        echo "ERRO: output.txt foi criado em ${where}, apesar de WRITE_OUTPUT=0." >&2
        exit 21
    fi
}

run_logged() {
    local exe="$1"
    local log="$2"

    rm -f output.txt
    "${exe}" > "${log}" 2>&1
    check_no_output_file "${exe}"
}

# ------------------------------------------------------------
# Build.
# ------------------------------------------------------------
echo
echo "============================================================"
echo "BUILD"
echo "============================================================"
make -j

for exe in \
    "${BASELINE}" \
    "${ORACLE}" \
    "${CALIBRATOR}" \
    "${CALIBRATED}" \
    "${ONLINE}"
do
    if [[ ! -x "${exe}" ]]; then
        echo "ERRO: executável não encontrado após make: ${exe}" >&2
        exit 2
    fi
done

export OMP_NUM_THREADS="${THREADS}"
export OMP_PLACES="${OMP_PLACES:-cores}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"

echo
echo "OpenMP threads : ${OMP_NUM_THREADS}"
echo "OMP_PLACES     : ${OMP_PLACES}"
echo "OMP_PROC_BIND  : ${OMP_PROC_BIND}"

# ============================================================
# 1. SMOKE TEST
# ============================================================
echo
echo "============================================================"
echo "SMOKE TEST"
echo "============================================================"

set_param N "${SMOKE_N}"
set_param T "${SMOKE_T}"
set_param TILE "${SMOKE_TILE}"
set_param WRITE_OUTPUT 0

echo "N=${SMOKE_N} T=${SMOKE_T} TILE=${SMOKE_TILE} WRITE_OUTPUT=0"

echo "[1/5] baseline"
run_logged "${BASELINE}" "${OUTDIR}/smoke_baseline.log"
require_scalar "Tempo" "${OUTDIR}/smoke_baseline.log" >/dev/null
require_scalar "L1_mean" "${OUTDIR}/smoke_baseline.log" >/dev/null
require_scalar "L2_rms" "${OUTDIR}/smoke_baseline.log" >/dev/null
require_scalar "Linf" "${OUTDIR}/smoke_baseline.log" >/dev/null

echo "[2/5] oracle"
run_logged "${ORACLE}" "${OUTDIR}/smoke_oracle.log"
require_scalar "Tempo" "${OUTDIR}/smoke_oracle.log" >/dev/null

echo "[3/5] calibrator"
rm -f "${COST_MODEL}"
rm -f output.txt
HEAT2D_CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES}" \
    "${CALIBRATOR}" > "${OUTDIR}/smoke_calibrator.log" 2>&1
check_no_output_file "${CALIBRATOR}"

if [[ ! -s "${COST_MODEL}" ]]; then
    echo "ERRO: calibrador não gerou ${COST_MODEL}." >&2
    exit 22
fi

echo "[4/5] adaptive calibrated"
run_logged "${CALIBRATED}" "${OUTDIR}/smoke_calibrated.log"
require_scalar "Tempo" "${OUTDIR}/smoke_calibrated.log" >/dev/null
require_scalar "L1_mean" "${OUTDIR}/smoke_calibrated.log" >/dev/null
require_scalar "L2_rms" "${OUTDIR}/smoke_calibrated.log" >/dev/null
require_scalar "Linf" "${OUTDIR}/smoke_calibrated.log" >/dev/null

echo "[5/5] adaptive online"
run_logged "${ONLINE}" "${OUTDIR}/smoke_online.log"
require_scalar "Tempo" "${OUTDIR}/smoke_online.log" >/dev/null
require_scalar "L1_mean" "${OUTDIR}/smoke_online.log" >/dev/null
require_scalar "L2_rms" "${OUTDIR}/smoke_online.log" >/dev/null
require_scalar "Linf" "${OUTDIR}/smoke_online.log" >/dev/null

echo
echo "SMOKE TEST: PASS"

# ============================================================
# 2. NUMERICAL SANITY / CONVERGENCE
# ============================================================
echo
echo "============================================================"
echo "NUMERICAL SANITY / CONVERGENCE"
echo "============================================================"
echo "N values    : ${N_VALUES[*]}"
echo "Base T      : ${BASE_T}"
echo "Min. order  : ${MIN_ORDER}"
echo "Max adaptive/baseline error ratio: ${MAX_ADAPTIVE_ERROR_RATIO}"
echo

CSV="${OUTDIR}/convergence.csv"
SUMMARY="${OUTDIR}/convergence_summary.txt"

echo "N,T,variant,L1,L2,Linf,time_s" > "${CSV}"

N0="${N_VALUES[0]}"

for N in "${N_VALUES[@]}"; do
    # Keep approximately the same final physical time:
    # T ~ (N-1)^2.
    T="$(python3 - "${BASE_T}" "${N0}" "${N}" <<'PY'
import sys
base_t = int(sys.argv[1])
n0 = int(sys.argv[2])
n = int(sys.argv[3])
t = round(base_t * ((n - 1) / (n0 - 1))**2)
print(max(1, t))
PY
)"

    set_param N "${N}"
    set_param T "${T}"
    set_param TILE "${SANITY_TILE}"
    set_param WRITE_OUTPUT 0

    echo "------------------------------------------------------------"
    echo "N=${N} T=${T}"
    echo "------------------------------------------------------------"

    # Baseline
    B_LOG="${OUTDIR}/baseline_N${N}.log"
    echo "  baseline"
    run_logged "${BASELINE}" "${B_LOG}"

    B_L1="$(require_scalar "L1_mean" "${B_LOG}")"
    B_L2="$(require_scalar "L2_rms" "${B_LOG}")"
    B_LI="$(require_scalar "Linf" "${B_LOG}")"
    B_T="$(require_scalar "Tempo" "${B_LOG}")"

    echo "${N},${T},baseline,${B_L1},${B_L2},${B_LI},${B_T}" >> "${CSV}"

    # Recalibrate for THIS N before adaptive_calibrated.
    C_LOG="${OUTDIR}/calibrator_N${N}.log"
    echo "  calibrator"
    rm -f "${COST_MODEL}" output.txt
    HEAT2D_CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES}" \
        "${CALIBRATOR}" > "${C_LOG}" 2>&1
    check_no_output_file "${CALIBRATOR}"

    if [[ ! -s "${COST_MODEL}" ]]; then
        echo "ERRO: calibrador não gerou ${COST_MODEL} para N=${N}." >&2
        exit 23
    fi

    # Adaptive calibrated
    A_LOG="${OUTDIR}/calibrated_N${N}.log"
    echo "  adaptive calibrated"
    run_logged "${CALIBRATED}" "${A_LOG}"

    A_L1="$(require_scalar "L1_mean" "${A_LOG}")"
    A_L2="$(require_scalar "L2_rms" "${A_LOG}")"
    A_LI="$(require_scalar "Linf" "${A_LOG}")"
    A_T="$(require_scalar "Tempo" "${A_LOG}")"

    echo "${N},${T},adaptive_calibrated,${A_L1},${A_L2},${A_LI},${A_T}" >> "${CSV}"

    # Adaptive online
    O_LOG="${OUTDIR}/online_N${N}.log"
    echo "  adaptive online"
    run_logged "${ONLINE}" "${O_LOG}"

    O_L1="$(require_scalar "L1_mean" "${O_LOG}")"
    O_L2="$(require_scalar "L2_rms" "${O_LOG}")"
    O_LI="$(require_scalar "Linf" "${O_LOG}")"
    O_T="$(require_scalar "Tempo" "${O_LOG}")"

    echo "${N},${T},adaptive_online,${O_L1},${O_L2},${O_LI},${O_T}" >> "${CSV}"
done

# ------------------------------------------------------------
# Analyze convergence and sanity criteria.
# ------------------------------------------------------------
python3 - "${CSV}" "${SUMMARY}" "${MIN_ORDER}" "${MAX_ADAPTIVE_ERROR_RATIO}" <<'PY'
import csv
import math
import sys
from collections import defaultdict

csv_path, summary_path, min_order_s, max_ratio_s = sys.argv[1:]
min_order = float(min_order_s)
max_ratio = float(max_ratio_s)

rows = []
with open(csv_path, newline="") as f:
    for r in csv.DictReader(f):
        r["N"] = int(r["N"])
        r["T"] = int(r["T"])
        for k in ("L1", "L2", "Linf", "time_s"):
            r[k] = float(r[k])
        rows.append(r)

by_variant = defaultdict(list)
by_N = defaultdict(dict)

for r in rows:
    by_variant[r["variant"]].append(r)
    by_N[r["N"]][r["variant"]] = r

for v in by_variant:
    by_variant[v].sort(key=lambda x: x["N"])

variants = ["baseline", "adaptive_calibrated", "adaptive_online"]
metrics = ["L1", "L2", "Linf"]

failures = []
lines = []

lines.append("=" * 84)
lines.append("NUMERICAL SANITY / CONVERGENCE SUMMARY")
lines.append("=" * 84)
lines.append("")

for v in variants:
    data = by_variant.get(v, [])
    if len(data) < 2:
        failures.append(f"{v}: insufficient convergence points")
        continue

    lines.append(v)
    lines.append("-" * len(v))
    lines.append(
        f'{"N":>7} {"T":>9} {"L1":>15} {"L2":>15} {"Linf":>15}'
    )
    for r in data:
        lines.append(
            f'{r["N"]:7d} {r["T"]:9d} '
            f'{r["L1"]:15.8e} {r["L2"]:15.8e} {r["Linf"]:15.8e}'
        )

    lines.append("")
    lines.append("Observed orders:")
    lines.append(
        f'{"refinement":>18} {"p(L1)":>12} {"p(L2)":>12} {"p(Linf)":>12}'
    )

    for coarse, fine in zip(data[:-1], data[1:]):
        # h = 1/(N-1); use exact mesh-ratio denominator.
        h_ratio = (fine["N"] - 1) / (coarse["N"] - 1)

        ps = {}
        for m in metrics:
            ec = coarse[m]
            ef = fine[m]
            if not (math.isfinite(ec) and math.isfinite(ef) and ec > 0 and ef > 0):
                p = float("nan")
            else:
                p = math.log(ec / ef) / math.log(h_ratio)
            ps[m] = p

            if not math.isfinite(p) or p < min_order:
                failures.append(
                    f'{v} {coarse["N"]}->{fine["N"]} {m}: '
                    f'order={p:.6g} < {min_order}'
                )

        lines.append(
            f'{coarse["N"]}->{fine["N"]: <10d} '
            f'{ps["L1"]:12.6f} {ps["L2"]:12.6f} {ps["Linf"]:12.6f}'
        )

    lines.append("")

lines.append("=" * 84)
lines.append("ADAPTIVE ERROR / BASELINE ERROR")
lines.append("=" * 84)
lines.append(
    f'{"N":>7} {"variant":>22} {"L1 ratio":>12} {"L2 ratio":>12} {"Linf ratio":>12}'
)

for N in sorted(by_N):
    base = by_N[N].get("baseline")
    if base is None:
        failures.append(f"N={N}: missing baseline")
        continue

    for v in ("adaptive_calibrated", "adaptive_online"):
        r = by_N[N].get(v)
        if r is None:
            failures.append(f"N={N}: missing {v}")
            continue

        ratios = {m: r[m] / base[m] for m in metrics}

        lines.append(
            f'{N:7d} {v:>22} '
            f'{ratios["L1"]:12.6f} {ratios["L2"]:12.6f} {ratios["Linf"]:12.6f}'
        )

        for m, ratio in ratios.items():
            if not math.isfinite(ratio) or ratio > max_ratio:
                failures.append(
                    f"N={N} {v} {m}: adaptive/baseline ratio "
                    f"{ratio:.6g} > {max_ratio}"
                )

lines.append("")
lines.append("=" * 84)

if failures:
    lines.append("SANITY RESULT: FAIL")
    lines.append("=" * 84)
    for f in failures:
        lines.append(" - " + f)
    rc = 1
else:
    lines.append("SANITY RESULT: PASS")
    lines.append("=" * 84)
    lines.append(
        f"All observed orders >= {min_order:.3f} and all adaptive/baseline "
        f"error ratios <= {max_ratio:.3f}."
    )
    rc = 0

text = "\n".join(lines)
print(text)

with open(summary_path, "w") as f:
    f.write(text + "\n")

sys.exit(rc)
PY

echo
echo "============================================================"
echo "ALL TESTS PASSED"
echo "============================================================"
echo "Logs and tables: ${OUTDIR}/"
echo "  ${CSV}"
echo "  ${SUMMARY}"
echo
echo "param.txt was restored automatically."
echo "Any pre-existing ${COST_MODEL} is also restored automatically."
