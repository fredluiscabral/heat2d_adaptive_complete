#!/bin/bash
#SBATCH --job-name=heat2d_allmodels
#SBATCH --partition=lncc-cpu_amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=heat2d_allmodels-%j.out
#SBATCH --error=heat2d_allmodels-%j.err

set -euo pipefail

# =============================================================================
# Heat2D complete dependency-resolution study - SDumont II
#
# Thread counts: 64, 128, 160, 192 (override with THREADS_LIST="...")
# Valid runs : 10 per variant by default (override with REPS=...)
#
# Current MPI-like family:
#   mpilike_baseline, mpilike_oracle, mpilike_calibrated,
#   mpilike_online, mpilike_compact
#
# Shared-array no-false-sharing families:
#   busywait_nofs, busywait_adaptive
#   semaphore_nofs, semaphore_adaptive
#
# Per thread count the script performs:
#   1) PREDICT/RECOMPUTE offline calibration (existing MPI-like calibrator);
#   2) observed busy-wait WAIT calibration;
#   3) observed semaphore WAIT calibration;
#   4) one diagnostic profile run of each new adaptive backend;
#   5) one warm-up per timed variant;
#   6) REPS valid runs, with rotated variant order;
#   7) per-thread and combined CSV summaries.
#
# Requires WRITE_OUTPUT = 0 in param.txt.
# The result root remains benchmark_models_results_sd; no rm -rf is used.
# =============================================================================

REPS="${REPS:-10}"
THREADS_LIST_STR="${THREADS_LIST:-64 128 160 192}"
read -r -a THREADS_LIST <<< "${THREADS_LIST_STR}"
OUTDIR="${OUTDIR:-benchmark_models_results_sd}"
CAL_SAMPLES="${CAL_SAMPLES:-32}"
WAIT_CAL_SAMPLES="${WAIT_CAL_SAMPLES:-32}"

MPI_BASE="./heat2d_explicit_omp_mpilike"
MPI_ORACLE="./heat2d_dependency_oracle"
MPI_CAL="./heat2d_dependency_calibrate"
MPI_OFF="./heat2d_adaptive_calibrated"
MPI_ONL="./heat2d_adaptive_online"
MPI_CMP="./heat2d_adaptive_compact"
BW_BASE="./heat2d_explicit_omp_busywait_nobarrier_nofs"
SM_BASE="./heat2d_explicit_omp_sem_nobarrier_nofs"
BW_WCAL="./heat2d_wait_calibrate_busywait"
SM_WCAL="./heat2d_wait_calibrate_sem"
BW_ADP="./heat2d_adaptive_busywait"
SM_ADP="./heat2d_adaptive_sem"
BW_PROF="./heat2d_adaptive_busywait_profile"
SM_PROF="./heat2d_adaptive_sem_profile"

cd "${SLURM_SUBMIT_DIR}"

export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_DYNAMIC=FALSE

mkdir -p "${OUTDIR}"
ROOT_RAW="${OUTDIR}/raw_results.csv"
ROOT_SUMMARY="${OUTDIR}/summary.csv"
echo "threads,variant,rep,time_s" > "${ROOT_RAW}"

printf '%s\n' "============================================================"
printf '%s\n' "SDUMONT HEAT2D - COMPLETE DEPENDENCY STUDY"
printf '%s\n' "============================================================"
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Node         : ${SLURM_NODELIST}"
echo "CPUs/task    : ${SLURM_CPUS_PER_TASK}"
echo "Submit dir   : ${SLURM_SUBMIT_DIR}"
echo "Threads      : ${THREADS_LIST[*]}"
echo "Repetitions  : ${REPS}"
echo "P/R samples  : ${CAL_SAMPLES}"
echo "WAIT samples : ${WAIT_CAL_SAMPLES}"
echo "OMP_PLACES   : ${OMP_PLACES}"
echo "OMP_PROC_BIND: ${OMP_PROC_BIND}"

if [[ ! -f param.txt ]]; then
    echo "ERRO: param.txt nao encontrado." >&2
    exit 1
fi
if ! command -v g++ >/dev/null 2>&1; then
    echo "ERRO: g++ nao encontrado." >&2
    exit 2
fi

WRITE_OUTPUT_VALUE="$(awk -F= '
    /^[[:space:]]*WRITE_OUTPUT[[:space:]]*=/ {
        gsub(/[[:space:]]/, "", $2); print tolower($2); exit
    }' param.txt)"
case "${WRITE_OUTPUT_VALUE}" in
    0|off|false|no) ;;
    *) echo "ERRO: configure WRITE_OUTPUT = 0 em param.txt." >&2; exit 3 ;;
esac

printf '\n%s\n' "================ PARAMETERS ================"
cat param.txt

printf '\n%s\n' "================ BUILD ====================="
make clean
make -j "${SLURM_CPUS_PER_TASK}"

ALL_EXES=(
    "${MPI_BASE}" "${MPI_ORACLE}" "${MPI_CAL}" "${MPI_OFF}" "${MPI_ONL}" "${MPI_CMP}"
    "${BW_BASE}" "${SM_BASE}" "${BW_WCAL}" "${SM_WCAL}"
    "${BW_ADP}" "${SM_ADP}" "${BW_PROF}" "${SM_PROF}"
)
for exe in "${ALL_EXES[@]}"; do
    [[ -x "${exe}" ]] || { echo "ERRO: executavel ausente: ${exe}" >&2; exit 4; }
done

extract_time() {
    local log="$1"
    awk '/^Tempo[[:space:]]*:/ {print $3; exit}' "${log}"
}

check_no_output() {
    local exe="$1"
    if [[ -e output.txt ]]; then
        echo "ERRO: ${exe} criou output.txt com WRITE_OUTPUT=0." >&2
        exit 5
    fi
}

run_plain() {
    local exe="$1" log="$2"
    shift 2
    rm -f output.txt adaptive_stats.txt
    env "$@" "${exe}" > "${log}" 2>&1
    check_no_output "${exe}"
}

run_timed_variant() {
    local threads="$1" pdir="$2" variant="$3" rep="$4"
    local exe log t
    local pr_model="${pdir}/heat2d_cost_model.dat"
    local bw_wait="${pdir}/heat2d_wait_cost_busywait.dat"
    local sm_wait="${pdir}/heat2d_wait_cost_semaphore.dat"

    case "${variant}" in
        mpilike_baseline)   exe="${MPI_BASE}" ;;
        mpilike_oracle)     exe="${MPI_ORACLE}" ;;
        mpilike_calibrated) exe="${MPI_OFF}" ;;
        mpilike_online)     exe="${MPI_ONL}" ;;
        mpilike_compact)    exe="${MPI_CMP}" ;;
        busywait_nofs)      exe="${BW_BASE}" ;;
        semaphore_nofs)     exe="${SM_BASE}" ;;
        busywait_adaptive)  exe="${BW_ADP}" ;;
        semaphore_adaptive) exe="${SM_ADP}" ;;
        *) echo "ERRO interno: variante ${variant}" >&2; exit 6 ;;
    esac

    log="${pdir}/${variant}_r${rep}.log"
    rm -f output.txt adaptive_stats.txt

    case "${variant}" in
        mpilike_calibrated|mpilike_compact)
            HEAT2D_COST_FILE="${pr_model}" "${exe}" > "${log}" 2>&1 ;;
        busywait_adaptive)
            HEAT2D_COST_FILE="${pr_model}" HEAT2D_WAIT_COST_FILE="${bw_wait}" \
                "${exe}" > "${log}" 2>&1 ;;
        semaphore_adaptive)
            HEAT2D_COST_FILE="${pr_model}" HEAT2D_WAIT_COST_FILE="${sm_wait}" \
                "${exe}" > "${log}" 2>&1 ;;
        *)
            "${exe}" > "${log}" 2>&1 ;;
    esac

    check_no_output "${exe}"
    t="$(extract_time "${log}")"
    if [[ -z "${t}" ]]; then
        echo "ERRO: Tempo nao encontrado em ${log}." >&2
        exit 7
    fi

    echo "${variant},${rep},${t}" >> "${pdir}/raw_results.csv"
    echo "${threads},${variant},${rep},${t}" >> "${ROOT_RAW}"
    printf 'threads=%3d rep=%02d/%02d %-22s %ss\n' \
        "${threads}" "${rep}" "${REPS}" "${variant}" "${t}"
}

VARIANTS=(
    mpilike_baseline
    mpilike_oracle
    mpilike_calibrated
    mpilike_online
    mpilike_compact
    busywait_nofs
    semaphore_nofs
    busywait_adaptive
    semaphore_adaptive
)
NV=${#VARIANTS[@]}

for threads in "${THREADS_LIST[@]}"; do
    if (( threads < 1 || threads > SLURM_CPUS_PER_TASK )); then
        echo "ERRO: thread count invalido: ${threads}" >&2
        exit 8
    fi

    export OMP_NUM_THREADS="${threads}"
    tag="$(printf '%03d' "${threads}")"
    pdir="${OUTDIR}/threads_${tag}"
    mkdir -p "${pdir}"

    # Remove somente artefatos conhecidos deste caso; nunca o diretorio inteiro.
    rm -f "${pdir}"/*.log "${pdir}"/*.csv "${pdir}"/*.dat 2>/dev/null || true
    echo "variant,rep,time_s" > "${pdir}/raw_results.csv"

    pr_model="${pdir}/heat2d_cost_model.dat"
    bw_wait="${pdir}/heat2d_wait_cost_busywait.dat"
    sm_wait="${pdir}/heat2d_wait_cost_semaphore.dat"

    printf '\n%s\n' "============================================================"
    echo "THREADS = ${threads}"
    printf '%s\n' "============================================================"

    # -------------------------------------------------------------------------
    # Offline cost calibration
    # -------------------------------------------------------------------------
    echo "[calibration] PREDICT / RECOMPUTE"
    run_plain "${MPI_CAL}" "${pdir}/calibration_pr.log" \
        HEAT2D_CALIBRATION_SAMPLES="${CAL_SAMPLES}" \
        HEAT2D_COST_FILE="${pr_model}"
    [[ -s "${pr_model}" ]] || { echo "ERRO: modelo P/R nao gerado." >&2; exit 9; }

    echo "[calibration] busy-wait WAIT"
    run_plain "${BW_WCAL}" "${pdir}/calibration_wait_busywait.log" \
        HEAT2D_WAIT_CALIBRATION_SAMPLES="${WAIT_CAL_SAMPLES}" \
        HEAT2D_WAIT_COST_FILE="${bw_wait}"
    [[ -s "${bw_wait}" ]] || { echo "ERRO: modelo WAIT busy-wait nao gerado." >&2; exit 10; }

    echo "[calibration] semaphore WAIT"
    run_plain "${SM_WCAL}" "${pdir}/calibration_wait_semaphore.log" \
        HEAT2D_WAIT_CALIBRATION_SAMPLES="${WAIT_CAL_SAMPLES}" \
        HEAT2D_WAIT_COST_FILE="${sm_wait}"
    [[ -s "${sm_wait}" ]] || { echo "ERRO: modelo WAIT semaphore nao gerado." >&2; exit 11; }

    echo "--- P/R model ---"; cat "${pr_model}"
    echo "--- busy-wait WAIT model ---"; cat "${bw_wait}"
    echo "--- semaphore WAIT model ---"; cat "${sm_wait}"

    # -------------------------------------------------------------------------
    # Diagnostic profile: not included in benchmark timing.
    # -------------------------------------------------------------------------
    echo "[diagnostic] busywait adaptive actions"
    run_plain "${BW_PROF}" "${pdir}/diagnostic_busywait_adaptive.log" \
        HEAT2D_COST_FILE="${pr_model}" HEAT2D_WAIT_COST_FILE="${bw_wait}"

    echo "[diagnostic] semaphore adaptive actions"
    run_plain "${SM_PROF}" "${pdir}/diagnostic_semaphore_adaptive.log" \
        HEAT2D_COST_FILE="${pr_model}" HEAT2D_WAIT_COST_FILE="${sm_wait}"

    echo "Busy-wait policy/actions:"
    grep -E '^Adaptive (try-PREDICT|RECOMPUTE:|WAIT:|direct R/W choice:|actions READ:|actions RECOMPUTE:|actions PREDICT:|actions WAIT:|predict rejected:)' \
        "${pdir}/diagnostic_busywait_adaptive.log" || true
    echo "Semaphore policy/actions:"
    grep -E '^Adaptive (try-PREDICT|RECOMPUTE:|WAIT:|direct R/W choice:|actions READ:|actions RECOMPUTE:|actions PREDICT:|actions WAIT:|predict rejected:)' \
        "${pdir}/diagnostic_semaphore_adaptive.log" || true

    # -------------------------------------------------------------------------
    # Warm-up: one per timed variant.
    # -------------------------------------------------------------------------
    echo "[warm-up] ${NV} variants"
    for variant in "${VARIANTS[@]}"; do
        # use rep=0 then remove the record immediately; this reuses exactly the
        # same execution path as valid runs and validates model compatibility.
        run_timed_variant "${threads}" "${pdir}" "${variant}" 0 >/dev/null
    done
    # warm-up must not enter the measured CSVs
    awk -F, 'NR==1 || $2 != 0' "${pdir}/raw_results.csv" > "${pdir}/raw_results.tmp"
    mv "${pdir}/raw_results.tmp" "${pdir}/raw_results.csv"
    awk -F, 'NR==1 || $3 != 0' "${ROOT_RAW}" > "${ROOT_RAW}.tmp"
    mv "${ROOT_RAW}.tmp" "${ROOT_RAW}"

    echo "MPI-like compact policy:"
    grep -E '^Adaptive calibrated prefer try-PREDICT:|^Adaptive compact fast path:|^Adaptive compact history slots:' \
        "${pdir}/mpilike_compact_r0.log" || true

    # -------------------------------------------------------------------------
    # Valid repetitions with a rotating 9-way order.
    # -------------------------------------------------------------------------
    echo "[benchmark] ${REPS} valid runs x ${NV} variants"
    for ((rep=1; rep<=REPS; ++rep)); do
        shift_by=$(( (rep - 1) % NV ))
        for ((k=0; k<NV; ++k)); do
            idx=$(( (k + shift_by) % NV ))
            run_timed_variant "${threads}" "${pdir}" "${VARIANTS[$idx]}" "${rep}"
        done
    done

done

# =============================================================================
# Combined statistics + one per-thread summary.csv
# =============================================================================
python3 - "${ROOT_RAW}" "${ROOT_SUMMARY}" "${OUTDIR}" <<'PY'
import csv
import math
import os
import statistics
import sys
from collections import defaultdict

raw_path, summary_path, outdir = sys.argv[1:4]
data = defaultdict(list)
with open(raw_path, newline="") as f:
    for row in csv.DictReader(f):
        key = (int(row["threads"]), row["variant"])
        data[key].append(float(row["time_s"]))

order = [
    "mpilike_baseline", "mpilike_oracle", "mpilike_calibrated",
    "mpilike_online", "mpilike_compact", "busywait_nofs",
    "busywait_adaptive", "semaphore_nofs", "semaphore_adaptive",
]

family_baseline = {
    "mpilike_baseline": "mpilike_baseline",
    "mpilike_oracle": "mpilike_baseline",
    "mpilike_calibrated": "mpilike_baseline",
    "mpilike_online": "mpilike_baseline",
    "mpilike_compact": "mpilike_baseline",
    "busywait_nofs": "busywait_nofs",
    "busywait_adaptive": "busywait_nofs",
    "semaphore_nofs": "semaphore_nofs",
    "semaphore_adaptive": "semaphore_nofs",
}

threads_values = sorted({t for t, _ in data})
rows = []
for t in threads_values:
    stats = {}
    for v in order:
        x = data.get((t, v), [])
        if not x:
            raise SystemExit(f"Missing results for threads={t}, variant={v}")
        stats[v] = {
            "n": len(x), "median": statistics.median(x),
            "mean": statistics.mean(x), "min": min(x), "max": max(x),
        }

    tb = stats["mpilike_baseline"]["median"]
    t0 = stats["mpilike_oracle"]["median"]
    gap = tb - t0

    for v in order:
        s = stats[v]
        med = s["median"]
        fb = family_baseline[v]
        fmed = stats[fb]["median"]
        recovery = ""
        if v in ("mpilike_calibrated", "mpilike_online", "mpilike_compact") and gap != 0.0:
            recovery = (tb - med) / gap
        rows.append({
            "threads": t,
            "variant": v,
            "n": s["n"],
            "median_s": med,
            "mean_s": s["mean"],
            "min_s": s["min"],
            "max_s": s["max"],
            "family_baseline": fb,
            "ratio_to_family_baseline": med / fmed,
            "delta_vs_family_baseline_pct": 100.0 * (med / fmed - 1.0),
            "ratio_to_mpilike_baseline": med / tb,
            "mpilike_gap_recovered_fraction": recovery,
        })

fields = [
    "threads", "variant", "n", "median_s", "mean_s", "min_s", "max_s",
    "family_baseline", "ratio_to_family_baseline",
    "delta_vs_family_baseline_pct", "ratio_to_mpilike_baseline",
    "mpilike_gap_recovered_fraction",
]
with open(summary_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader(); w.writerows(rows)

for t in threads_values:
    pdir = os.path.join(outdir, f"threads_{t:03d}")
    with open(os.path.join(pdir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows([r for r in rows if r["threads"] == t])

print("\n" + "="*104)
print("FINAL MEDIAN SUMMARY")
print("="*104)
for t in threads_values:
    print(f"\nTHREADS={t}")
    print(f'{"variant":>22} {"median(s)":>12} {"family ratio":>13} {"delta family":>13}')
    print("-"*66)
    for r in [x for x in rows if x["threads"] == t]:
        print(f'{r["variant"]:>22} {r["median_s"]:12.8f} '
              f'{r["ratio_to_family_baseline"]:13.6f} '
              f'{r["delta_vs_family_baseline_pct"]:+12.3f}%')
PY

printf '\n%s\n' "================ SUMMARY.CSV ==============="
cat "${ROOT_SUMMARY}"
printf '\n%s\n' "================ JOB COMPLETED =============="
echo "Results: ${OUTDIR}/"
