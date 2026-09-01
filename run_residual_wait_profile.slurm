#!/bin/bash
#SBATCH --job-name=heat2d_residual_wait
#SBATCH --partition=lncc-cpu_amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=heat2d_residual_wait-%j.out
#SBATCH --error=heat2d_residual_wait-%j.err

set -euo pipefail

# =============================================================================
# Heat2D - residual WAIT diagnostic on SDumont II
#
# Measures, in the ORIGINAL no-FS busy-wait and semaphore synchronization
# patterns, how much elapsed time remains after the first availability probe
# reports that a dependency is not ready.
#
# This is a diagnostic experiment, not a performance benchmark. The profile
# binaries are instrumented and their Tempo values must not replace baseline
# benchmark times.
#
# Override examples:
#   REPS=5 sbatch run_residual_wait_profile.slurm
#   THREADS_LIST="128 192" sbatch run_residual_wait_profile.slurm
#   SAVE_SAMPLES=1 sbatch run_residual_wait_profile.slurm
# =============================================================================

REPS="${REPS:-3}"
THREADS_LIST_STR="${THREADS_LIST:-64 128 160 192}"
read -r -a THREADS_LIST <<< "${THREADS_LIST_STR}"
CAL_SAMPLES="${CAL_SAMPLES:-32}"
SAVE_SAMPLES="${SAVE_SAMPLES:-0}"
OUTDIR="${OUTDIR:-residual_wait_results_sd}"

MPI_CAL="./heat2d_dependency_calibrate"
BW_RES="./heat2d_residual_profile_busywait"
SM_RES="./heat2d_residual_profile_sem"

cd "${SLURM_SUBMIT_DIR}"

export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_DYNAMIC=FALSE

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
printf '%s\n' "HEAT2D - RESIDUAL WAIT PROFILE"
printf '%s\n' "============================================================"
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Node         : ${SLURM_NODELIST}"
echo "Threads      : ${THREADS_LIST[*]}"
echo "Repetitions  : ${REPS}"
echo "P/R samples  : ${CAL_SAMPLES}"
echo "Save samples : ${SAVE_SAMPLES}"
echo "OMP_PLACES   : ${OMP_PLACES}"
echo "OMP_PROC_BIND: ${OMP_PROC_BIND}"
echo
echo "Parameters:"
cat param.txt

printf '\n%s\n' "================ BUILD ====================="
make -j "${SLURM_CPUS_PER_TASK}" \
    heat2d_dependency_calibrate \
    heat2d_residual_profile_busywait \
    heat2d_residual_profile_sem

for exe in "${MPI_CAL}" "${BW_RES}" "${SM_RES}"; do
    [[ -x "${exe}" ]] || { echo "ERRO: executavel ausente: ${exe}" >&2; exit 3; }
done

run_profile() {
    local backend="$1" exe="$2" threads="$3" rep="$4" pdir="$5" recompute="$6"
    local log="${pdir}/${backend}_r${rep}.log"
    local -a extra_env=()

    if [[ "${SAVE_SAMPLES}" == "1" ]]; then
        extra_env+=("HEAT2D_RESIDUAL_FILE=${pdir}/${backend}_r${rep}_samples.csv")
    fi

    rm -f output.txt
    env HEAT2D_RECOMPUTE_CYCLES="${recompute}" \
        "${extra_env[@]}" "${exe}" > "${log}" 2>&1

    if [[ -e output.txt ]]; then
        echo "ERRO: ${exe} criou output.txt com WRITE_OUTPUT=0." >&2
        exit 4
    fi

    printf 'threads=%3d rep=%02d/%02d %-10s ' "${threads}" "${rep}" "${REPS}" "${backend}"
    grep -E '^Residual (miss fraction:|wait p50:|wait p90:|wait p95:|waits below RECOMPUTE fraction:)' "${log}" \
        | sed 's/^Residual //' | tr '\n' ' '
    echo
}

for threads in "${THREADS_LIST[@]}"; do
    if (( threads < 1 || threads > SLURM_CPUS_PER_TASK )); then
        echo "ERRO: thread count invalido: ${threads}" >&2
        exit 5
    fi

    export OMP_NUM_THREADS="${threads}"
    tag="$(printf '%03d' "${threads}")"
    pdir="${OUTDIR}/threads_${tag}"
    mkdir -p "${pdir}"
    rm -f "${pdir}"/*.log "${pdir}"/*.dat "${pdir}"/*.csv 2>/dev/null || true

    printf '\n%s\n' "============================================================"
    echo "THREADS = ${threads}"
    printf '%s\n' "============================================================"

    cost_model="${pdir}/heat2d_cost_model.dat"
    echo "[calibration] RECOMPUTE cost"
    rm -f output.txt
    HEAT2D_CALIBRATION_SAMPLES="${CAL_SAMPLES}" \
    HEAT2D_COST_FILE="${cost_model}" \
        "${MPI_CAL}" > "${pdir}/calibration_pr.log" 2>&1
    [[ -s "${cost_model}" ]] || { echo "ERRO: modelo P/R nao gerado." >&2; exit 6; }

    recompute="$(awk '$1 == "recompute_ticks" {print $2; exit}' "${cost_model}")"
    tick_unit="$(awk '$1 == "tick_unit" {print $2; exit}' "${cost_model}")"
    [[ -n "${recompute}" ]] || { echo "ERRO: recompute_ticks ausente." >&2; exit 7; }
    [[ "${tick_unit}" == "cycles" ]] || {
        echo "ERRO: esperado tick_unit=cycles no SDumont; obtido ${tick_unit}." >&2
        exit 8
    }
    echo "RECOMPUTE = ${recompute} cycles"

    # Warm-up diagnostic. No raw samples are saved.
    echo "[warm-up] residual profiles"
    HEAT2D_RECOMPUTE_CYCLES="${recompute}" "${BW_RES}" > "${pdir}/busywait_warmup.log" 2>&1
    HEAT2D_RECOMPUTE_CYCLES="${recompute}" "${SM_RES}" > "${pdir}/semaphore_warmup.log" 2>&1

    # Alternate backend order across repetitions.
    for ((rep=1; rep<=REPS; ++rep)); do
        if (( rep % 2 == 1 )); then
            run_profile busywait "${BW_RES}" "${threads}" "${rep}" "${pdir}" "${recompute}"
            run_profile semaphore "${SM_RES}" "${threads}" "${rep}" "${pdir}" "${recompute}"
        else
            run_profile semaphore "${SM_RES}" "${threads}" "${rep}" "${pdir}" "${recompute}"
            run_profile busywait "${BW_RES}" "${threads}" "${rep}" "${pdir}" "${recompute}"
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
    "Residual dependencies total": "dependencies",
    "Residual immediate ready": "immediate",
    "Residual initial misses": "misses",
    "Residual miss fraction": "miss_fraction",
    "Residual wait min": "min",
    "Residual wait mean": "mean",
    "Residual wait p50": "p50",
    "Residual wait p75": "p75",
    "Residual wait p90": "p90",
    "Residual wait p95": "p95",
    "Residual wait p99": "p99",
    "Residual wait max": "max",
    "Residual recompute threshold": "recompute",
    "Residual waits below RECOMPUTE fraction": "frac_below_recompute",
    "Residual waits >= RECOMPUTE fraction": "frac_above_equal_recompute",
    "Tempo": "profile_time_s",
}

int_fields = {"dependencies", "immediate", "misses", "min", "p50", "p75", "p90", "p95", "p99", "max"}
rows = []

for path in sorted(glob.glob(os.path.join(outdir, "threads_*", "*_r*.log"))):
    base = os.path.basename(path)
    m = re.fullmatch(r"(busywait|semaphore)_r(\d+)\.log", base)
    if not m:
        continue
    backend, rep = m.group(1), int(m.group(2))
    thread_tag = os.path.basename(os.path.dirname(path)).split("_")[-1]
    threads = int(thread_tag)
    vals = {"threads": threads, "backend": backend, "rep": rep}
    with open(path, errors="replace") as f:
        for line in f:
            if line.startswith("Tempo"):
                mtime = re.match(r"^Tempo\s*:\s*([-+0-9.eE]+)", line)
                if mtime:
                    vals["profile_time_s"] = float(mtime.group(1))
                    continue
            for prefix, key in fields.items():
                if prefix == "Tempo":
                    continue
                if line.startswith(prefix + ":"):
                    value = line.split(":", 1)[1].strip().split()[0]
                    vals[key] = int(float(value)) if key in int_fields else float(value)
                    break
    missing = [k for k in fields.values() if k not in vals]
    if missing:
        raise SystemExit(f"Missing fields in {path}: {missing}")
    rows.append(vals)

if not rows:
    raise SystemExit("No residual profile logs found")

per_run_path = os.path.join(outdir, "per_run.csv")
columns = [
    "threads", "backend", "rep", "dependencies", "immediate", "misses",
    "miss_fraction", "min", "mean", "p50", "p75", "p90", "p95", "p99", "max",
    "recompute", "frac_below_recompute", "frac_above_equal_recompute", "profile_time_s",
]
with open(per_run_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=columns)
    w.writeheader(); w.writerows(rows)

summary_path = os.path.join(outdir, "summary.csv")
summary_cols = [
    "threads", "backend", "reps",
    "median_dependencies", "median_immediate", "median_misses", "median_miss_fraction",
    "median_min", "median_mean", "median_p50", "median_p75", "median_p90", "median_p95",
    "median_p99", "median_max", "recompute_cycles",
    "median_frac_below_recompute", "median_frac_above_equal_recompute",
]

groups = {}
for r in rows:
    groups.setdefault((r["threads"], r["backend"]), []).append(r)

summary_rows = []
for (threads, backend), grp in sorted(groups.items()):
    if len(grp) != expected_reps:
        raise SystemExit(f"Expected {expected_reps} reps for {threads}/{backend}, found {len(grp)}")
    med = lambda key: statistics.median(r[key] for r in grp)
    summary_rows.append({
        "threads": threads,
        "backend": backend,
        "reps": len(grp),
        "median_dependencies": med("dependencies"),
        "median_immediate": med("immediate"),
        "median_misses": med("misses"),
        "median_miss_fraction": med("miss_fraction"),
        "median_min": med("min"),
        "median_mean": med("mean"),
        "median_p50": med("p50"),
        "median_p75": med("p75"),
        "median_p90": med("p90"),
        "median_p95": med("p95"),
        "median_p99": med("p99"),
        "median_max": med("max"),
        "recompute_cycles": med("recompute"),
        "median_frac_below_recompute": med("frac_below_recompute"),
        "median_frac_above_equal_recompute": med("frac_above_equal_recompute"),
    })

with open(summary_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=summary_cols)
    w.writeheader(); w.writerows(summary_rows)

print("\n" + "="*118)
print("RESIDUAL WAIT SUMMARY (median across repetitions)")
print("="*118)
print(f'{"thr":>4} {"backend":>10} {"miss%":>8} {"p50":>12} {"p90":>12} {"p95":>12} {"p99":>12} {"C_R":>12} {"P(W<C_R)":>12}')
for r in summary_rows:
    print(f'{r["threads"]:4d} {r["backend"]:>10} {100*r["median_miss_fraction"]:8.2f} '
          f'{r["median_p50"]:12.0f} {r["median_p90"]:12.0f} {r["median_p95"]:12.0f} {r["median_p99"]:12.0f} '
          f'{r["recompute_cycles"]:12.1f} {100*r["median_frac_below_recompute"]:11.2f}%')
print("="*118)
print(f"Per-run CSV : {per_run_path}")
print(f"Summary CSV : {summary_path}")
PY

echo
echo "Residual-WAIT diagnostic complete."
echo "Results: ${OUTDIR}/"
echo "  ${OUTDIR}/per_run.csv"
echo "  ${OUTDIR}/summary.csv"
if [[ "${SAVE_SAMPLES}" == "1" ]]; then
    echo "  Raw sample CSVs were also saved inside threads_*/"
fi
