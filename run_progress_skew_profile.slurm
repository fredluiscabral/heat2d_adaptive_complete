#!/bin/bash
#SBATCH --job-name=heat2d_progress_skew
#SBATCH --partition=lncc-cpu_amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=heat2d_progress_skew-%j.out
#SBATCH --error=heat2d_progress_skew-%j.err

set -euo pipefail

# Initial experiment: N=8192, T=1000, TILE=32, 192 threads.
# Compares, under identical progress instrumentation:
#   baseline WAIT
#   adaptive forced to WAIT
#   adaptive with PREDICT off and RECOMPUTE enabled
# for busy-wait and semaphore backends.

REPS="${REPS:-3}"
CAL_SAMPLES="${CAL_SAMPLES:-32}"
WAIT_CAL_SAMPLES="${WAIT_CAL_SAMPLES:-32}"
WINDOW_STEPS="${WINDOW_STEPS:-100}"
OUTDIR="${OUTDIR:-progress_skew_results_sd}"

cd "${SLURM_SUBMIT_DIR}"

export OMP_NUM_THREADS=192
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_DYNAMIC=FALSE

get_param() {
    local key="$1"
    awk -F= -v k="${key}" '
        $1 ~ "^[[:space:]]*" k "[[:space:]]*$" {
            gsub(/[[:space:]]/, "", $2); print $2; exit
        }' param.txt
}

[[ -f param.txt ]] || { echo "ERRO: param.txt nao encontrado." >&2; exit 1; }
[[ "$(get_param N)" == "8192" ]] || { echo "ERRO: este teste inicial requer N=8192." >&2; exit 2; }
[[ "$(get_param T)" == "1000" ]] || { echo "ERRO: este teste inicial requer T=1000." >&2; exit 3; }
[[ "$(get_param TILE)" == "32" ]] || { echo "ERRO: este teste inicial requer TILE=32." >&2; exit 4; }
case "$(get_param WRITE_OUTPUT | tr '[:upper:]' '[:lower:]')" in
    0|off|false|no) ;;
    *) echo "ERRO: configure WRITE_OUTPUT=0." >&2; exit 5 ;;
esac

MPI_CAL=./heat2d_dependency_calibrate
BW_WCAL=./heat2d_wait_calibrate_busywait
SM_WCAL=./heat2d_wait_calibrate_sem

BW_BASE=./heat2d_explicit_omp_busywait_nobarrier_nofs
SM_BASE=./heat2d_explicit_omp_sem_nobarrier_nofs
BW_ADP=./heat2d_adaptive_busywait
SM_ADP=./heat2d_adaptive_sem

BW_BASE_PROF=./heat2d_progress_profile_busywait_baseline
SM_BASE_PROF=./heat2d_progress_profile_sem_baseline
BW_ADP_PROF=./heat2d_progress_profile_busywait_adaptive
SM_ADP_PROF=./heat2d_progress_profile_sem_adaptive

mkdir -p "${OUTDIR}"
rm -f "${OUTDIR}"/*.log "${OUTDIR}"/*.csv "${OUTDIR}"/*.dat 2>/dev/null || true

printf '%s\n' "============================================================"
printf '%s\n' "HEAT2D - PROGRESS SKEW PROFILE"
printf '%s\n' "============================================================"
echo "Job ID        : ${SLURM_JOB_ID}"
echo "Node          : ${SLURM_NODELIST}"
echo "Threads       : ${OMP_NUM_THREADS}"
echo "Repetitions   : ${REPS}"
echo "Window steps  : ${WINDOW_STEPS}"
echo "OMP_PLACES    : ${OMP_PLACES}"
echo "OMP_PROC_BIND : ${OMP_PROC_BIND}"
echo
cat param.txt

echo
echo "================ BUILD ===================="
make -j "${SLURM_CPUS_PER_TASK}" \
    heat2d_dependency_calibrate \
    heat2d_wait_calibrate_busywait \
    heat2d_wait_calibrate_sem \
    heat2d_explicit_omp_busywait_nobarrier_nofs \
    heat2d_explicit_omp_sem_nobarrier_nofs \
    heat2d_adaptive_busywait \
    heat2d_adaptive_sem \
    progress_diagnostics

for exe in "${MPI_CAL}" "${BW_WCAL}" "${SM_WCAL}" \
           "${BW_BASE}" "${SM_BASE}" "${BW_ADP}" "${SM_ADP}" \
           "${BW_BASE_PROF}" "${SM_BASE_PROF}" "${BW_ADP_PROF}" "${SM_ADP_PROF}"; do
    [[ -x "${exe}" ]] || { echo "ERRO: executavel ausente: ${exe}" >&2; exit 6; }
done

PR_MODEL="${OUTDIR}/heat2d_cost_model.dat"
BW_WAIT="${OUTDIR}/heat2d_wait_cost_busywait.dat"
SM_WAIT="${OUTDIR}/heat2d_wait_cost_semaphore.dat"

echo
echo "================ CALIBRATION =============="
HEAT2D_CALIBRATION_SAMPLES="${CAL_SAMPLES}" HEAT2D_COST_FILE="${PR_MODEL}" \
    "${MPI_CAL}" > "${OUTDIR}/calibration_pr.log" 2>&1
HEAT2D_WAIT_CALIBRATION_SAMPLES="${WAIT_CAL_SAMPLES}" HEAT2D_WAIT_COST_FILE="${BW_WAIT}" \
    "${BW_WCAL}" > "${OUTDIR}/calibration_wait_busywait.log" 2>&1
HEAT2D_WAIT_CALIBRATION_SAMPLES="${WAIT_CAL_SAMPLES}" HEAT2D_WAIT_COST_FILE="${SM_WAIT}" \
    "${SM_WCAL}" > "${OUTDIR}/calibration_wait_semaphore.log" 2>&1

CR="$(awk '$1=="recompute_ticks" {print $2; exit}' "${PR_MODEL}")"
CW_BW="$(awk '$1=="wait_ticks" {print $2; exit}' "${BW_WAIT}")"
CW_SM="$(awk '$1=="wait_ticks" {print $2; exit}' "${SM_WAIT}")"
[[ -n "${CR}" && -n "${CW_BW}" && -n "${CW_SM}" ]] || { echo "ERRO: calibracao incompleta." >&2; exit 7; }

echo "C_R       = ${CR} cycles"
echo "C_W busy  = ${CW_BW} cycles"
echo "C_W sem   = ${CW_SM} cycles"

python3 - "${CR}" "${CW_BW}" "${CW_SM}" <<'PY'
import sys
cr,bw,sm=map(float,sys.argv[1:])
if not cr < bw:
    raise SystemExit(f"ERRO: C_R={cr} nao e menor que C_W_busy={bw}; modo RECOMPUTE nao seria selecionado.")
if not cr < sm:
    raise SystemExit(f"ERRO: C_R={cr} nao e menor que C_W_sem={sm}; modo RECOMPUTE nao seria selecionado.")
PY

extract_time() {
    awk '/^Tempo[[:space:]]*:/ {print $3; exit}' "$1"
}

run_reference() {
    local backend="$1" mode="$2" rep="$3" log="$4"
    local exe wait_file
    if [[ "${backend}" == "busywait" ]]; then
        wait_file="${BW_WAIT}"
        [[ "${mode}" == "baseline_wait" ]] && exe="${BW_BASE}" || exe="${BW_ADP}"
    else
        wait_file="${SM_WAIT}"
        [[ "${mode}" == "baseline_wait" ]] && exe="${SM_BASE}" || exe="${SM_ADP}"
    fi

    rm -f output.txt
    if [[ "${mode}" == "baseline_wait" ]]; then
        "${exe}" > "${log}" 2>&1
    elif [[ "${mode}" == "adaptive_wait" ]]; then
        HEAT2D_COST_FILE="${PR_MODEL}" HEAT2D_WAIT_COST_FILE="${wait_file}" \
        HEAT2D_ENABLE_PREDICT=0 HEAT2D_ENABLE_RECOMPUTE=0 \
            "${exe}" > "${log}" 2>&1
    else
        HEAT2D_COST_FILE="${PR_MODEL}" HEAT2D_WAIT_COST_FILE="${wait_file}" \
        HEAT2D_ENABLE_PREDICT=0 HEAT2D_ENABLE_RECOMPUTE=1 \
            "${exe}" > "${log}" 2>&1
    fi
    [[ ! -e output.txt ]] || { echo "ERRO: output.txt criado por ${exe}." >&2; exit 8; }
    local t
    t="$(extract_time "${log}")"
    [[ -n "${t}" ]] || { echo "ERRO: Tempo ausente em ${log}." >&2; exit 9; }
    echo "${t}"
}

run_profile() {
    local backend="$1" mode="$2" rep="$3" log="$4" summary="$5" windows="$6"
    local exe wait_file
    if [[ "${backend}" == "busywait" ]]; then
        wait_file="${BW_WAIT}"
        [[ "${mode}" == "baseline_wait" ]] && exe="${BW_BASE_PROF}" || exe="${BW_ADP_PROF}"
    else
        wait_file="${SM_WAIT}"
        [[ "${mode}" == "baseline_wait" ]] && exe="${SM_BASE_PROF}" || exe="${SM_ADP_PROF}"
    fi

    rm -f output.txt "${summary}" "${windows}"
    if [[ "${mode}" == "baseline_wait" ]]; then
        HEAT2D_PROGRESS_SUMMARY_FILE="${summary}" \
        HEAT2D_PROGRESS_WINDOWS_FILE="${windows}" \
        HEAT2D_PROGRESS_WINDOW_STEPS="${WINDOW_STEPS}" \
            "${exe}" > "${log}" 2>&1
    elif [[ "${mode}" == "adaptive_wait" ]]; then
        HEAT2D_COST_FILE="${PR_MODEL}" HEAT2D_WAIT_COST_FILE="${wait_file}" \
        HEAT2D_ENABLE_PREDICT=0 HEAT2D_ENABLE_RECOMPUTE=0 \
        HEAT2D_PROGRESS_SUMMARY_FILE="${summary}" \
        HEAT2D_PROGRESS_WINDOWS_FILE="${windows}" \
        HEAT2D_PROGRESS_WINDOW_STEPS="${WINDOW_STEPS}" \
            "${exe}" > "${log}" 2>&1
    else
        HEAT2D_COST_FILE="${PR_MODEL}" HEAT2D_WAIT_COST_FILE="${wait_file}" \
        HEAT2D_ENABLE_PREDICT=0 HEAT2D_ENABLE_RECOMPUTE=1 \
        HEAT2D_PROGRESS_SUMMARY_FILE="${summary}" \
        HEAT2D_PROGRESS_WINDOWS_FILE="${windows}" \
        HEAT2D_PROGRESS_WINDOW_STEPS="${WINDOW_STEPS}" \
            "${exe}" > "${log}" 2>&1
    fi
    [[ ! -e output.txt ]] || { echo "ERRO: output.txt criado por ${exe}." >&2; exit 10; }
    [[ -s "${summary}" && -s "${windows}" ]] || { echo "ERRO: arquivos de progresso ausentes." >&2; exit 11; }

    if [[ "${mode}" == "adaptive_wait" ]]; then
        grep -q '^Adaptive actions RECOMPUTE: 0$' "${log}" || { echo "ERRO: adaptive_wait executou RECOMPUTE." >&2; exit 12; }
        grep -q '^Adaptive actions PREDICT: 0$' "${log}" || { echo "ERRO: adaptive_wait executou PREDICT." >&2; exit 13; }
    elif [[ "${mode}" == "adaptive_recompute" ]]; then
        local rc wt pr
        rc="$(awk '/^Adaptive actions RECOMPUTE:/ {print $4; exit}' "${log}")"
        wt="$(awk '/^Adaptive actions WAIT:/ {print $4; exit}' "${log}")"
        pr="$(awk '/^Adaptive actions PREDICT:/ {print $4; exit}' "${log}")"
        [[ "${rc:-0}" -gt 0 && "${wt:-1}" -eq 0 && "${pr:-1}" -eq 0 ]] || {
            echo "ERRO: adaptive_recompute nao ficou em READ/RECOMPUTE puro (R=${rc:-?} W=${wt:-?} P=${pr:-?})." >&2
            exit 14
        }
    fi
}

MODES=(baseline_wait adaptive_wait adaptive_recompute)
BACKENDS=(busywait semaphore)
RAW="${OUTDIR}/raw_times.csv"
echo "backend,mode,rep,reference_s,profile_s" > "${RAW}"

# Warm-up, no files kept.
echo
echo "================ WARM-UP =================="
for backend in "${BACKENDS[@]}"; do
    for mode in "${MODES[@]}"; do
        echo "warm-up ${backend} ${mode}"
        run_reference "${backend}" "${mode}" 0 "${OUTDIR}/warmup_ref_${backend}_${mode}.log" >/dev/null
        run_profile "${backend}" "${mode}" 0 \
            "${OUTDIR}/warmup_prof_${backend}_${mode}.log" \
            "${OUTDIR}/warmup_${backend}_${mode}_summary.csv" \
            "${OUTDIR}/warmup_${backend}_${mode}_windows.csv"
        rm -f "${OUTDIR}/warmup_${backend}_${mode}_summary.csv" "${OUTDIR}/warmup_${backend}_${mode}_windows.csv"
    done
done

echo
echo "================ MEASURED ================="
for ((rep=1; rep<=REPS; ++rep)); do
    # Rotate backend order to reduce ordering bias.
    if (( rep % 2 == 1 )); then ORDER=(busywait semaphore); else ORDER=(semaphore busywait); fi
    for backend in "${ORDER[@]}"; do
        for mode in "${MODES[@]}"; do
            ref_log="${OUTDIR}/ref_${backend}_${mode}_r${rep}.log"
            prof_log="${OUTDIR}/profile_${backend}_${mode}_r${rep}.log"
            psum="${OUTDIR}/profile_${backend}_${mode}_r${rep}_summary.csv"
            pwin="${OUTDIR}/profile_${backend}_${mode}_r${rep}_windows.csv"
            tref="$(run_reference "${backend}" "${mode}" "${rep}" "${ref_log}")"
            run_profile "${backend}" "${mode}" "${rep}" "${prof_log}" "${psum}" "${pwin}"
            tprof="$(extract_time "${prof_log}")"
            echo "${backend},${mode},${rep},${tref},${tprof}" >> "${RAW}"
            printf 'rep=%d/%d %-9s %-18s ref=%ss profile=%ss\n' \
                "${rep}" "${REPS}" "${backend}" "${mode}" "${tref}" "${tprof}"
        done
    done
done

python3 - "${OUTDIR}" "${REPS}" <<'PY'
import csv, glob, os, statistics, sys
from collections import defaultdict
outdir=sys.argv[1]
reps=int(sys.argv[2])

# Times
with open(os.path.join(outdir,'raw_times.csv'), newline='') as f:
    times=list(csv.DictReader(f))

groups=defaultdict(list)
for r in times:
    groups[(r['backend'],r['mode'])].append(r)

progress={}
for path in glob.glob(os.path.join(outdir,'profile_*_r*_summary.csv')):
    base=os.path.basename(path)
    # profile_<backend>_<mode>_rN_summary.csv
    stem=base[:-len('_summary.csv')]
    prefix='profile_'
    rest=stem[len(prefix):]
    backend, tail = rest.split('_',1)
    mode, rep_s = tail.rsplit('_r',1)
    rep=int(rep_s)
    with open(path,newline='') as f:
        progress[(backend,mode,rep)]=next(csv.DictReader(f))

summary_path=os.path.join(outdir,'summary.csv')
with open(summary_path,'w',newline='') as f:
    fields=['backend','mode','reps','median_reference_s','median_profile_s','profile_overhead_pct',
            'median_tau_step_ns','median_global_range_mean_norm','median_global_range_p95_ns',
            'median_neighbor_rms_norm','median_neighbor_rms_ns']
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
    for backend in ('busywait','semaphore'):
        for mode in ('baseline_wait','adaptive_wait','adaptive_recompute'):
            rows=groups[(backend,mode)]
            if len(rows)!=reps: raise SystemExit(f'missing time rows for {backend}/{mode}')
            ref=[float(x['reference_s']) for x in rows]
            prof=[float(x['profile_s']) for x in rows]
            ps=[progress[(backend,mode,int(x['rep']))] for x in rows]
            medref=statistics.median(ref); medprof=statistics.median(prof)
            w.writerow({
                'backend':backend,'mode':mode,'reps':reps,
                'median_reference_s':medref,'median_profile_s':medprof,
                'profile_overhead_pct':100.0*(medprof/medref-1.0),
                'median_tau_step_ns':statistics.median(float(x['tau_step_median_ns']) for x in ps),
                'median_global_range_mean_norm':statistics.median(float(x['global_range_mean_norm']) for x in ps),
                'median_global_range_p95_ns':statistics.median(float(x['global_range_p95_ns']) for x in ps),
                'median_neighbor_rms_norm':statistics.median(float(x['neighbor_rms_norm']) for x in ps),
                'median_neighbor_rms_ns':statistics.median(float(x['neighbor_rms_ns']) for x in ps),
            })

# Consolidate all window traces, preserving mode/rep.
win_out=os.path.join(outdir,'windows_all.csv')
first=True
with open(win_out,'w',newline='') as fo:
    writer=None
    for path in sorted(glob.glob(os.path.join(outdir,'profile_*_r*_windows.csv'))):
        base=os.path.basename(path)
        stem=base[:-len('_windows.csv')]
        rest=stem[len('profile_'):]
        backend, tail=rest.split('_',1)
        mode, rep_s=tail.rsplit('_r',1)
        rep=int(rep_s)
        with open(path,newline='') as fi:
            rows=list(csv.DictReader(fi))
        if not rows: continue
        fields=['backend','mode','rep']+list(rows[0].keys())
        if writer is None:
            writer=csv.DictWriter(fo,fieldnames=fields); writer.writeheader()
        for r in rows:
            writer.writerow({'backend':backend,'mode':mode,'rep':rep,**r})

print('\nSUMMARY')
with open(summary_path) as f:
    print(f.read(), end='')
PY

echo
echo "Resultados:"
echo "  ${OUTDIR}/summary.csv"
echo "  ${OUTDIR}/raw_times.csv"
echo "  ${OUTDIR}/windows_all.csv"
