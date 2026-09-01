#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Heat2D - complete smoke test + numerical sanity/convergence check
#
# The script validates every binary produced by the current Makefile:
#   MPI-like family:
#     ./heat2d_explicit_omp_mpilike
#     ./heat2d_dependency_oracle
#     ./heat2d_dependency_calibrate
#     ./heat2d_adaptive_calibrated
#     ./heat2d_adaptive_online
#     ./heat2d_adaptive_compact
#   no-false-sharing baselines:
#     ./heat2d_explicit_omp_busywait_nobarrier_nofs
#     ./heat2d_explicit_omp_sem_nobarrier_nofs
#   backend WAIT calibrators:
#     ./heat2d_wait_calibrate_busywait
#     ./heat2d_wait_calibrate_sem
#   adaptive busy-wait / semaphore:
#     ./heat2d_adaptive_busywait
#     ./heat2d_adaptive_sem
#   diagnostic/profile builds:
#     ./heat2d_adaptive_busywait_profile
#     ./heat2d_adaptive_sem_profile
#     ./heat2d_residual_profile_busywait
#     ./heat2d_residual_profile_sem
#
# Smoke stage:
#   - builds and runs all 16 executables;
#   - checks WRITE_OUTPUT=0 for every execution;
#   - checks generation of all three cost-model files;
#   - checks that profile binaries report READ/RECOMPUTE/PREDICT/WAIT actions.
#
# Numerical sanity/convergence stage:
#   - oracle and calibrators are intentionally excluded from convergence;
#   - profile binaries are equivalent diagnostic builds and are smoke-tested only;
#   - convergence is checked for the 8 numerically meaningful solver variants:
#       mpilike_baseline, mpilike_calibrated, mpilike_online, mpilike_compact,
#       busywait_nofs, semaphore_nofs, busywait_adaptive, semaphore_adaptive;
#   - cost models are recalibrated for every N before adaptive runs;
#   - T scales as (N-1)^2 to keep final physical time approximately constant.
# =============================================================================

PARAM="${PARAM:-param.txt}"
THREADS="${THREADS:-4}"

SMOKE_N="${SMOKE_N:-512}"
SMOKE_T="${SMOKE_T:-100}"
SMOKE_TILE="${SMOKE_TILE:-32}"

N_VALUES=(${N_VALUES:-128 256 512 1024})
BASE_T="${BASE_T:-100}"
SANITY_TILE="${SANITY_TILE:-32}"

MIN_ORDER="${MIN_ORDER:-1.70}"
MAX_ERROR_RATIO="${MAX_ERROR_RATIO:-1.20}"
CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-8}"
WAIT_CALIBRATION_SAMPLES="${WAIT_CALIBRATION_SAMPLES:-8}"

OUTDIR="${OUTDIR:-smoke_sanity_results}"
mkdir -p "${OUTDIR}"

MPI_BASE="${MPI_BASE:-./heat2d_explicit_omp_mpilike}"
MPI_ORACLE="${MPI_ORACLE:-./heat2d_dependency_oracle}"
MPI_CAL="${MPI_CAL:-./heat2d_dependency_calibrate}"
MPI_OFF="${MPI_OFF:-./heat2d_adaptive_calibrated}"
MPI_ONL="${MPI_ONL:-./heat2d_adaptive_online}"
MPI_CMP="${MPI_CMP:-./heat2d_adaptive_compact}"
BW_BASE="${BW_BASE:-./heat2d_explicit_omp_busywait_nobarrier_nofs}"
SM_BASE="${SM_BASE:-./heat2d_explicit_omp_sem_nobarrier_nofs}"
BW_WCAL="${BW_WCAL:-./heat2d_wait_calibrate_busywait}"
SM_WCAL="${SM_WCAL:-./heat2d_wait_calibrate_sem}"
BW_ADP="${BW_ADP:-./heat2d_adaptive_busywait}"
SM_ADP="${SM_ADP:-./heat2d_adaptive_sem}"
BW_PROF="${BW_PROF:-./heat2d_adaptive_busywait_profile}"
SM_PROF="${SM_PROF:-./heat2d_adaptive_sem_profile}"
BW_RES="${BW_RES:-./heat2d_residual_profile_busywait}"
SM_RES="${SM_RES:-./heat2d_residual_profile_sem}"

COST_MODEL="${COST_MODEL:-heat2d_cost_model.dat}"
BW_WAIT_MODEL="${BW_WAIT_MODEL:-heat2d_wait_cost_busywait.dat}"
SM_WAIT_MODEL="${SM_WAIT_MODEL:-heat2d_wait_cost_semaphore.dat}"

if [[ ! -f "${PARAM}" ]]; then
    echo "ERRO: ${PARAM} nao encontrado." >&2
    exit 1
fi

# -----------------------------------------------------------------------------
# Preserve every file that this test may modify.
# -----------------------------------------------------------------------------
PARAM_BACKUP="$(mktemp)"
cp "${PARAM}" "${PARAM_BACKUP}"
MODEL_BACKUP_DIR="$(mktemp -d)"

for f in "${COST_MODEL}" "${BW_WAIT_MODEL}" "${SM_WAIT_MODEL}"; do
    if [[ -f "${f}" ]]; then
        cp "${f}" "${MODEL_BACKUP_DIR}/$(basename "${f}")"
    fi
done

cleanup() {
    cp "${PARAM_BACKUP}" "${PARAM}"
    rm -f "${PARAM_BACKUP}"

    for f in "${COST_MODEL}" "${BW_WAIT_MODEL}" "${SM_WAIT_MODEL}"; do
        b="${MODEL_BACKUP_DIR}/$(basename "${f}")"
        if [[ -f "${b}" ]]; then
            cp "${b}" "${f}"
        else
            rm -f "${f}"
        fi
    done
    rm -rf "${MODEL_BACKUP_DIR}"

    rm -f output.txt adaptive_stats.txt
}
trap cleanup EXIT

set_param() {
    local key="$1"
    local value="$2"
    if grep -Eq "^[[:space:]]*${key}[[:space:]]*=" "${PARAM}"; then
        sed -i -E "s|^[[:space:]]*${key}[[:space:]]*=.*$|${key} = ${value}|" "${PARAM}"
    else
        printf '%s = %s\n' "${key}" "${value}" >> "${PARAM}"
    fi
}

extract_scalar() {
    local key="$1"
    local file="$2"
    awk -v key="${key}" '
        index($0,key)==1 {
            for (i=1;i<=NF;++i) {
                if ($i ~ /^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$/) value=$i
            }
            if (value!="") { print value; exit }
        }
    ' "${file}"
}

require_scalar() {
    local key="$1"
    local file="$2"
    local value
    value="$(extract_scalar "${key}" "${file}")"
    if [[ -z "${value}" ]]; then
        echo "ERRO: campo '${key}' nao encontrado em ${file}." >&2
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
    shift 2
    rm -f output.txt adaptive_stats.txt
    env "$@" "${exe}" > "${log}" 2>&1
    check_no_output_file "${exe}"
}

require_solver_metrics() {
    local log="$1"
    require_scalar "Tempo" "${log}" >/dev/null
    require_scalar "L1_mean" "${log}" >/dev/null
    require_scalar "L2_rms" "${log}" >/dev/null
    require_scalar "Linf" "${log}" >/dev/null
}

# =============================================================================
# BUILD
# =============================================================================
echo
echo "============================================================"
echo "BUILD"
echo "============================================================"
make -j

ALL_EXES=(
    "${MPI_BASE}" "${MPI_ORACLE}" "${MPI_CAL}" "${MPI_OFF}" "${MPI_ONL}" "${MPI_CMP}"
    "${BW_BASE}" "${SM_BASE}" "${BW_WCAL}" "${SM_WCAL}"
    "${BW_ADP}" "${SM_ADP}" "${BW_PROF}" "${SM_PROF}"
    "${BW_RES}" "${SM_RES}"
)
for exe in "${ALL_EXES[@]}"; do
    if [[ ! -x "${exe}" ]]; then
        echo "ERRO: executavel nao encontrado apos make: ${exe}" >&2
        exit 2
    fi
done

echo "Executables checked: ${#ALL_EXES[@]}"

export OMP_NUM_THREADS="${THREADS}"
export OMP_PLACES="${OMP_PLACES:-cores}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_DYNAMIC=FALSE

echo "OpenMP threads : ${OMP_NUM_THREADS}"
echo "OMP_PLACES     : ${OMP_PLACES}"
echo "OMP_PROC_BIND  : ${OMP_PROC_BIND}"

# =============================================================================
# 1. COMPLETE SMOKE TEST
# =============================================================================
echo
echo "============================================================"
echo "COMPLETE SMOKE TEST - 16 EXECUTABLES"
echo "============================================================"

set_param N "${SMOKE_N}"
set_param T "${SMOKE_T}"
set_param TILE "${SMOKE_TILE}"
set_param WRITE_OUTPUT 0

rm -f "${COST_MODEL}" "${BW_WAIT_MODEL}" "${SM_WAIT_MODEL}" output.txt adaptive_stats.txt

echo "N=${SMOKE_N} T=${SMOKE_T} TILE=${SMOKE_TILE} WRITE_OUTPUT=0"

step=0
TOTAL=16
announce() { step=$((step+1)); printf '[%02d/%02d] %s\n' "${step}" "${TOTAL}" "$1"; }

announce "MPI-like baseline"
run_logged "${MPI_BASE}" "${OUTDIR}/smoke_mpilike_baseline.log"
require_solver_metrics "${OUTDIR}/smoke_mpilike_baseline.log"

announce "MPI-like oracle"
run_logged "${MPI_ORACLE}" "${OUTDIR}/smoke_mpilike_oracle.log"
require_scalar "Tempo" "${OUTDIR}/smoke_mpilike_oracle.log" >/dev/null

announce "PREDICT/RECOMPUTE calibrator"
run_logged "${MPI_CAL}" "${OUTDIR}/smoke_calibration_pr.log" \
    HEAT2D_CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES}" \
    HEAT2D_COST_FILE="${COST_MODEL}"
[[ -s "${COST_MODEL}" ]] || { echo "ERRO: ${MPI_CAL} nao gerou ${COST_MODEL}." >&2; exit 22; }
require_solver_metrics "${OUTDIR}/smoke_calibration_pr.log"
RECOMPUTE_CYCLES="$(awk '$1 == "recompute_ticks" {print $2; exit}' "${COST_MODEL}")"
[[ -n "${RECOMPUTE_CYCLES}" ]] || { echo "ERRO: recompute_ticks ausente em ${COST_MODEL}." >&2; exit 25; }

announce "busy-wait residual-WAIT profile"
BW_RES_CSV="${OUTDIR}/smoke_residual_busywait.csv"
rm -f "${BW_RES_CSV}"
run_logged "${BW_RES}" "${OUTDIR}/smoke_residual_busywait.log" \
    HEAT2D_RECOMPUTE_CYCLES="${RECOMPUTE_CYCLES}" \
    HEAT2D_RESIDUAL_FILE="${BW_RES_CSV}"
require_solver_metrics "${OUTDIR}/smoke_residual_busywait.log"
require_scalar "Residual initial misses" "${OUTDIR}/smoke_residual_busywait.log" >/dev/null
[[ -s "${BW_RES_CSV}" ]] || { echo "ERRO: perfil residual busy-wait nao gerou CSV." >&2; exit 26; }

announce "semaphore residual-WAIT profile"
SM_RES_CSV="${OUTDIR}/smoke_residual_semaphore.csv"
rm -f "${SM_RES_CSV}"
run_logged "${SM_RES}" "${OUTDIR}/smoke_residual_semaphore.log" \
    HEAT2D_RECOMPUTE_CYCLES="${RECOMPUTE_CYCLES}" \
    HEAT2D_RESIDUAL_FILE="${SM_RES_CSV}"
require_solver_metrics "${OUTDIR}/smoke_residual_semaphore.log"
require_scalar "Residual initial misses" "${OUTDIR}/smoke_residual_semaphore.log" >/dev/null
[[ -s "${SM_RES_CSV}" ]] || { echo "ERRO: perfil residual semaforo nao gerou CSV." >&2; exit 27; }

announce "MPI-like adaptive calibrated"
run_logged "${MPI_OFF}" "${OUTDIR}/smoke_mpilike_calibrated.log" \
    HEAT2D_COST_FILE="${COST_MODEL}"
require_solver_metrics "${OUTDIR}/smoke_mpilike_calibrated.log"

announce "MPI-like adaptive online"
run_logged "${MPI_ONL}" "${OUTDIR}/smoke_mpilike_online.log"
require_solver_metrics "${OUTDIR}/smoke_mpilike_online.log"

announce "MPI-like adaptive compact"
run_logged "${MPI_CMP}" "${OUTDIR}/smoke_mpilike_compact.log" \
    HEAT2D_COST_FILE="${COST_MODEL}"
require_solver_metrics "${OUTDIR}/smoke_mpilike_compact.log"

announce "busy-wait no-FS baseline"
run_logged "${BW_BASE}" "${OUTDIR}/smoke_busywait_nofs.log"
require_solver_metrics "${OUTDIR}/smoke_busywait_nofs.log"

announce "semaphore no-FS baseline"
run_logged "${SM_BASE}" "${OUTDIR}/smoke_semaphore_nofs.log"
require_solver_metrics "${OUTDIR}/smoke_semaphore_nofs.log"

announce "busy-wait WAIT calibrator"
run_logged "${BW_WCAL}" "${OUTDIR}/smoke_wait_calibration_busywait.log" \
    HEAT2D_WAIT_CALIBRATION_SAMPLES="${WAIT_CALIBRATION_SAMPLES}" \
    HEAT2D_WAIT_COST_FILE="${BW_WAIT_MODEL}"
[[ -s "${BW_WAIT_MODEL}" ]] || { echo "ERRO: ${BW_WCAL} nao gerou ${BW_WAIT_MODEL}." >&2; exit 23; }
require_solver_metrics "${OUTDIR}/smoke_wait_calibration_busywait.log"

announce "semaphore WAIT calibrator"
run_logged "${SM_WCAL}" "${OUTDIR}/smoke_wait_calibration_semaphore.log" \
    HEAT2D_WAIT_CALIBRATION_SAMPLES="${WAIT_CALIBRATION_SAMPLES}" \
    HEAT2D_WAIT_COST_FILE="${SM_WAIT_MODEL}"
[[ -s "${SM_WAIT_MODEL}" ]] || { echo "ERRO: ${SM_WCAL} nao gerou ${SM_WAIT_MODEL}." >&2; exit 24; }
require_solver_metrics "${OUTDIR}/smoke_wait_calibration_semaphore.log"

announce "busy-wait adaptive"
run_logged "${BW_ADP}" "${OUTDIR}/smoke_busywait_adaptive.log" \
    HEAT2D_COST_FILE="${COST_MODEL}" HEAT2D_WAIT_COST_FILE="${BW_WAIT_MODEL}"
require_solver_metrics "${OUTDIR}/smoke_busywait_adaptive.log"

announce "semaphore adaptive"
run_logged "${SM_ADP}" "${OUTDIR}/smoke_semaphore_adaptive.log" \
    HEAT2D_COST_FILE="${COST_MODEL}" HEAT2D_WAIT_COST_FILE="${SM_WAIT_MODEL}"
require_solver_metrics "${OUTDIR}/smoke_semaphore_adaptive.log"

announce "busy-wait adaptive profile"
run_logged "${BW_PROF}" "${OUTDIR}/smoke_busywait_profile.log" \
    HEAT2D_COST_FILE="${COST_MODEL}" HEAT2D_WAIT_COST_FILE="${BW_WAIT_MODEL}"
require_solver_metrics "${OUTDIR}/smoke_busywait_profile.log"
for key in "Adaptive actions READ" "Adaptive actions RECOMPUTE" "Adaptive actions PREDICT" "Adaptive actions WAIT"; do
    require_scalar "${key}" "${OUTDIR}/smoke_busywait_profile.log" >/dev/null
done

announce "semaphore adaptive profile"
run_logged "${SM_PROF}" "${OUTDIR}/smoke_semaphore_profile.log" \
    HEAT2D_COST_FILE="${COST_MODEL}" HEAT2D_WAIT_COST_FILE="${SM_WAIT_MODEL}"
require_solver_metrics "${OUTDIR}/smoke_semaphore_profile.log"
for key in "Adaptive actions READ" "Adaptive actions RECOMPUTE" "Adaptive actions PREDICT" "Adaptive actions WAIT"; do
    require_scalar "${key}" "${OUTDIR}/smoke_semaphore_profile.log" >/dev/null
done

echo
echo "COMPLETE SMOKE TEST: PASS"

# =============================================================================
# 2. NUMERICAL SANITY / CONVERGENCE
# =============================================================================
echo
echo "============================================================"
echo "NUMERICAL SANITY / CONVERGENCE"
echo "============================================================"
echo "N values      : ${N_VALUES[*]}"
echo "Base T        : ${BASE_T}"
echo "Min. order    : ${MIN_ORDER}"
echo "Max error/base: ${MAX_ERROR_RATIO}"
echo

CSV="${OUTDIR}/convergence.csv"
SUMMARY="${OUTDIR}/convergence_summary.txt"
echo "N,T,variant,L1,L2,Linf,time_s" > "${CSV}"

append_solver_row() {
    local N="$1" T="$2" variant="$3" log="$4"
    local l1 l2 li ts
    l1="$(require_scalar "L1_mean" "${log}")"
    l2="$(require_scalar "L2_rms" "${log}")"
    li="$(require_scalar "Linf" "${log}")"
    ts="$(require_scalar "Tempo" "${log}")"
    echo "${N},${T},${variant},${l1},${l2},${li},${ts}" >> "${CSV}"
}

N0="${N_VALUES[0]}"
for N in "${N_VALUES[@]}"; do
    T="$(python3 - "${BASE_T}" "${N0}" "${N}" <<'PY'
import sys
base_t = int(sys.argv[1]); n0 = int(sys.argv[2]); n = int(sys.argv[3])
t = round(base_t * ((n - 1) / (n0 - 1))**2)
print(max(1, t))
PY
)"

    set_param N "${N}"
    set_param T "${T}"
    set_param TILE "${SANITY_TILE}"
    set_param WRITE_OUTPUT 0

    case_dir="${OUTDIR}/N${N}"
    mkdir -p "${case_dir}"
    pr_model="${case_dir}/heat2d_cost_model.dat"
    bw_wait="${case_dir}/heat2d_wait_cost_busywait.dat"
    sm_wait="${case_dir}/heat2d_wait_cost_semaphore.dat"
    rm -f "${pr_model}" "${bw_wait}" "${sm_wait}" output.txt adaptive_stats.txt

    echo "------------------------------------------------------------"
    echo "N=${N} T=${T}"
    echo "------------------------------------------------------------"

    echo "  calibrate P/R"
    run_logged "${MPI_CAL}" "${case_dir}/calibration_pr.log" \
        HEAT2D_CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES}" \
        HEAT2D_COST_FILE="${pr_model}"
    [[ -s "${pr_model}" ]] || { echo "ERRO: modelo P/R nao gerado para N=${N}." >&2; exit 30; }

    echo "  calibrate busy-wait WAIT"
    run_logged "${BW_WCAL}" "${case_dir}/calibration_wait_busywait.log" \
        HEAT2D_WAIT_CALIBRATION_SAMPLES="${WAIT_CALIBRATION_SAMPLES}" \
        HEAT2D_WAIT_COST_FILE="${bw_wait}"
    [[ -s "${bw_wait}" ]] || { echo "ERRO: modelo WAIT busy-wait nao gerado para N=${N}." >&2; exit 31; }

    echo "  calibrate semaphore WAIT"
    run_logged "${SM_WCAL}" "${case_dir}/calibration_wait_semaphore.log" \
        HEAT2D_WAIT_CALIBRATION_SAMPLES="${WAIT_CALIBRATION_SAMPLES}" \
        HEAT2D_WAIT_COST_FILE="${sm_wait}"
    [[ -s "${sm_wait}" ]] || { echo "ERRO: modelo WAIT semaphore nao gerado para N=${N}." >&2; exit 32; }

    echo "  mpilike baseline"
    log="${case_dir}/mpilike_baseline.log"
    run_logged "${MPI_BASE}" "${log}"
    append_solver_row "${N}" "${T}" "mpilike_baseline" "${log}"

    echo "  mpilike calibrated"
    log="${case_dir}/mpilike_calibrated.log"
    run_logged "${MPI_OFF}" "${log}" HEAT2D_COST_FILE="${pr_model}"
    append_solver_row "${N}" "${T}" "mpilike_calibrated" "${log}"

    echo "  mpilike online"
    log="${case_dir}/mpilike_online.log"
    run_logged "${MPI_ONL}" "${log}"
    append_solver_row "${N}" "${T}" "mpilike_online" "${log}"

    echo "  mpilike compact"
    log="${case_dir}/mpilike_compact.log"
    run_logged "${MPI_CMP}" "${log}" HEAT2D_COST_FILE="${pr_model}"
    append_solver_row "${N}" "${T}" "mpilike_compact" "${log}"

    echo "  busy-wait no-FS baseline"
    log="${case_dir}/busywait_nofs.log"
    run_logged "${BW_BASE}" "${log}"
    append_solver_row "${N}" "${T}" "busywait_nofs" "${log}"

    echo "  semaphore no-FS baseline"
    log="${case_dir}/semaphore_nofs.log"
    run_logged "${SM_BASE}" "${log}"
    append_solver_row "${N}" "${T}" "semaphore_nofs" "${log}"

    echo "  busy-wait adaptive"
    log="${case_dir}/busywait_adaptive.log"
    run_logged "${BW_ADP}" "${log}" \
        HEAT2D_COST_FILE="${pr_model}" HEAT2D_WAIT_COST_FILE="${bw_wait}"
    append_solver_row "${N}" "${T}" "busywait_adaptive" "${log}"

    echo "  semaphore adaptive"
    log="${case_dir}/semaphore_adaptive.log"
    run_logged "${SM_ADP}" "${log}" \
        HEAT2D_COST_FILE="${pr_model}" HEAT2D_WAIT_COST_FILE="${sm_wait}"
    append_solver_row "${N}" "${T}" "semaphore_adaptive" "${log}"
done

python3 - "${CSV}" "${SUMMARY}" "${MIN_ORDER}" "${MAX_ERROR_RATIO}" <<'PY'
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
        r["N"] = int(r["N"]); r["T"] = int(r["T"])
        for k in ("L1", "L2", "Linf", "time_s"):
            r[k] = float(r[k])
        rows.append(r)

variants = [
    "mpilike_baseline", "mpilike_calibrated", "mpilike_online", "mpilike_compact",
    "busywait_nofs", "semaphore_nofs", "busywait_adaptive", "semaphore_adaptive",
]
metrics = ["L1", "L2", "Linf"]
by_variant = defaultdict(list)
by_N = defaultdict(dict)
for r in rows:
    by_variant[r["variant"]].append(r)
    by_N[r["N"]][r["variant"]] = r
for v in by_variant:
    by_variant[v].sort(key=lambda x: x["N"])

failures = []
lines = []
lines += ["="*96, "NUMERICAL SANITY / CONVERGENCE SUMMARY", "="*96, ""]

for v in variants:
    data = by_variant.get(v, [])
    if len(data) < 2:
        failures.append(f"{v}: insufficient convergence points")
        continue
    lines.append(v)
    lines.append("-"*len(v))
    lines.append(f'{"N":>7} {"T":>9} {"L1":>15} {"L2":>15} {"Linf":>15}')
    for r in data:
        lines.append(f'{r["N"]:7d} {r["T"]:9d} {r["L1"]:15.8e} {r["L2"]:15.8e} {r["Linf"]:15.8e}')
    lines.append("")
    lines.append("Observed orders:")
    lines.append(f'{"refinement":>18} {"p(L1)":>12} {"p(L2)":>12} {"p(Linf)":>12}')
    for coarse, fine in zip(data[:-1], data[1:]):
        h_ratio = (fine["N"] - 1) / (coarse["N"] - 1)
        ps = {}
        for m in metrics:
            ec, ef = coarse[m], fine[m]
            p = math.log(ec/ef)/math.log(h_ratio) if ec > 0 and ef > 0 else float("nan")
            ps[m] = p
            if not math.isfinite(p) or p < min_order:
                failures.append(f'{v} {coarse["N"]}->{fine["N"]} {m}: order={p:.6g} < {min_order}')
        lines.append(f'{coarse["N"]}->{fine["N"]:<10d} {ps["L1"]:12.6f} {ps["L2"]:12.6f} {ps["Linf"]:12.6f}')
    lines.append("")

lines += ["="*96, "ERROR RATIO RELATIVE TO MPI-LIKE BASELINE", "="*96]
lines.append(f'{"N":>7} {"variant":>24} {"L1 ratio":>12} {"L2 ratio":>12} {"Linf ratio":>12}')
for N in sorted(by_N):
    base = by_N[N].get("mpilike_baseline")
    if base is None:
        failures.append(f"N={N}: missing mpilike_baseline")
        continue
    for v in variants[1:]:
        r = by_N[N].get(v)
        if r is None:
            failures.append(f"N={N}: missing {v}")
            continue
        ratios = {m: r[m]/base[m] for m in metrics}
        lines.append(f'{N:7d} {v:>24} {ratios["L1"]:12.6f} {ratios["L2"]:12.6f} {ratios["Linf"]:12.6f}')
        for m, ratio in ratios.items():
            if not math.isfinite(ratio) or ratio > max_ratio:
                failures.append(f"N={N} {v} {m}: error/base ratio {ratio:.6g} > {max_ratio}")

lines += ["", "="*96]
if failures:
    lines += ["SANITY RESULT: FAIL", "="*96]
    lines.extend(" - " + x for x in failures)
    rc = 1
else:
    lines += ["SANITY RESULT: PASS", "="*96]
    lines.append(f"All observed orders >= {min_order:.3f} and all error/base ratios <= {max_ratio:.3f}.")
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
echo "param.txt and any pre-existing cost models were restored automatically."
