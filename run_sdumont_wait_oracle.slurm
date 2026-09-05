#!/bin/bash
#SBATCH --job-name=heat2d_wait_oracle
#SBATCH --partition=lncc-cpu_amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=heat2d_wait_oracle-%j.out
#SBATCH --error=heat2d_wait_oracle-%j.err

set -euo pipefail

# =============================================================================
# SDumont II — WAIT critical-path oracle diagnostic
#
# Fixed experiment:
#   N=8192, T=500, TILE=32, threads=64
#   affinity = close, spread
#   backend  = busywait, semaphore
#
# For each case:
#   - native WAIT-only runs quantify tracing perturbation;
#   - trace WAIT-only runs record per-step timing and actual blocking WAITs;
#   - all trace events stay in per-thread memory during the timed region;
#   - offline Python reconstructs the layered causal DAG and critical paths.
#
# The trace execution times are DIAGNOSTIC ONLY and must not replace the normal
# performance measurements in the paper.
# =============================================================================

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
THREADS="${THREADS:-64}"
REPS="${REPS:-3}"
BASE_OUTDIR="${OUTDIR:-results/sdumont_wait_oracle_T500}"
RUN_DIR="${BASE_OUTDIR}/job_${SLURM_JOB_ID}"

BW_NATIVE="${PROJECT_DIR}/heat2d_explicit_omp_busywait_nobarrier_nofs"
SM_NATIVE="${PROJECT_DIR}/heat2d_explicit_omp_sem_nobarrier_nofs"
BW_TRACE="${PROJECT_DIR}/heat2d_wait_trace_busywait"
SM_TRACE="${PROJECT_DIR}/heat2d_wait_trace_semaphore"
ANALYZER="${PROJECT_DIR}/analyze_heat2d_wait_oracle.py"

cd "${PROJECT_DIR}"
export OMP_PLACES=cores
export OMP_DYNAMIC=FALSE

[[ -f param.txt ]] || { echo "ERRO: param.txt ausente." >&2; exit 1; }
[[ -f heat2d_wait_trace.hpp ]] || { echo "ERRO: heat2d_wait_trace.hpp ausente." >&2; exit 1; }
[[ -f heat2d_explicit_omp_busywait_nobarrier_nofs_wait_trace.cpp ]] || { echo "ERRO: fonte trace busywait ausente." >&2; exit 1; }
[[ -f heat2d_explicit_omp_sem_nobarrier_nofs_wait_trace.cpp ]] || { echo "ERRO: fonte trace semaphore ausente." >&2; exit 1; }
[[ -f "${ANALYZER}" ]] || { echo "ERRO: analisador ausente: ${ANALYZER}" >&2; exit 1; }

get_param() {
    local key="$1"
    awk -F= -v k="$key" '
        $1 ~ "^[[:space:]]*" k "[[:space:]]*$" {
            gsub(/[[:space:]]/, "", $2); print $2; exit
        }' param.txt
}

N_VALUE="$(get_param N)"
T_VALUE="$(get_param T)"
TILE_VALUE="$(get_param TILE)"
WRITE_OUTPUT_VALUE="$(get_param WRITE_OUTPUT | tr '[:upper:]' '[:lower:]')"

[[ "${N_VALUE}" == "8192" ]] || { echo "ERRO: N deve ser 8192; atual=${N_VALUE}." >&2; exit 2; }
[[ "${T_VALUE}" == "500" ]] || { echo "ERRO: T deve ser 500; atual=${T_VALUE}." >&2; exit 2; }
[[ "${TILE_VALUE}" == "32" ]] || { echo "ERRO: TILE deve ser 32; atual=${TILE_VALUE}." >&2; exit 2; }
case "${WRITE_OUTPUT_VALUE}" in
    0|off|false|no) ;;
    *) echo "ERRO: WRITE_OUTPUT deve ser 0; atual=${WRITE_OUTPUT_VALUE}." >&2; exit 2 ;;
esac
(( THREADS == 64 )) || { echo "ERRO: este diagnostico foi definido para THREADS=64." >&2; exit 2; }
(( REPS >= 1 )) || { echo "ERRO: REPS deve ser >=1." >&2; exit 2; }

mkdir -p "${RUN_DIR}"
cp param.txt "${RUN_DIR}/param.txt"
lscpu > "${RUN_DIR}/lscpu.txt" 2>&1 || true
lscpu -e=CPU,NODE,SOCKET,CORE,ONLINE > "${RUN_DIR}/lscpu_extended.txt" 2>&1 || true

MAKE_JOBS="${MAKE_JOBS:-${SLURM_CPUS_PER_TASK}}"
echo "[build] baselines + trace"
make -j "${MAKE_JOBS}" baselines trace

for exe in "${BW_NATIVE}" "${SM_NATIVE}" "${BW_TRACE}" "${SM_TRACE}"; do
    [[ -x "${exe}" ]] || { echo "ERRO: executavel ausente: ${exe}" >&2; exit 3; }
done

printf 'backend,affinity,rep,time_s,linf\n' > "${RUN_DIR}/native_times.csv"

extract_time() {
    awk '/^Tempo[[:space:]]*:/ { for (i=1;i<=NF;i++) if ($i==":" && i<NF) {print $(i+1); exit} }' "$1"
}
extract_linf() {
    awk -F: '/^Linf:/ {gsub(/[[:space:]]/,"",$2); print $2; exit}' "$1"
}
check_log() {
    local log="$1"
    grep -q '^Tempo[[:space:]]*:' "$log" || { echo "ERRO: Tempo ausente em ${log}" >&2; exit 4; }
    grep -q '^Linf:' "$log" || { echo "ERRO: Linf ausente em ${log}" >&2; exit 4; }
    grep -q "^Threads: ${THREADS}$" "$log" || { echo "ERRO: Threads inesperado em ${log}" >&2; exit 4; }
}

run_native() {
    local backend="$1" affinity="$2" rep="$3"
    local exe log
    if [[ "$backend" == "busywait" ]]; then exe="${BW_NATIVE}"; else exe="${SM_NATIVE}"; fi
    log="${RUN_DIR}/native_${backend}_${affinity}_rep$(printf '%02d' "$rep").log"
    env OMP_NUM_THREADS="${THREADS}" OMP_PROC_BIND="${affinity}" \
        "${exe}" > "${log}" 2>&1
    check_log "${log}"
    local tm li
    tm="$(extract_time "${log}")"
    li="$(extract_linf "${log}")"
    [[ -n "${tm}" && -n "${li}" ]] || { echo "ERRO: parser native falhou: ${log}" >&2; exit 4; }
    printf '%s,%s,%d,%s,%s\n' "$backend" "$affinity" "$rep" "$tm" "$li" >> "${RUN_DIR}/native_times.csv"
}

run_trace() {
    local backend="$1" affinity="$2" rep="$3"
    local exe tag log
    if [[ "$backend" == "busywait" ]]; then exe="${BW_TRACE}"; else exe="${SM_TRACE}"; fi
    tag="${backend}_${affinity}_rep$(printf '%02d' "$rep")"
    log="${RUN_DIR}/${tag}.log"

    env \
        OMP_NUM_THREADS="${THREADS}" \
        OMP_PROC_BIND="${affinity}" \
        HEAT2D_TRACE_DIR="${RUN_DIR}" \
        HEAT2D_TRACE_TAG="${tag}" \
        HEAT2D_TRACE_AFFINITY="${affinity}" \
        HEAT2D_TRACE_REP="${rep}" \
        "${exe}" > "${log}" 2>&1

    check_log "${log}"
    [[ -s "${RUN_DIR}/${tag}_steps.csv" ]] || { echo "ERRO: steps ausente para ${tag}" >&2; exit 5; }
    [[ -s "${RUN_DIR}/${tag}_waits.csv" ]] || { echo "ERRO: waits ausente para ${tag}" >&2; exit 5; }
    [[ -s "${RUN_DIR}/${tag}_meta.csv" ]] || { echo "ERRO: meta ausente para ${tag}" >&2; exit 5; }
}

for affinity in close spread; do
    echo
    echo "=================================================================="
    echo "affinity=${affinity} threads=${THREADS}"
    echo "=================================================================="

    # One affinity report outside the diagnostic timing interpretation.
    env OMP_NUM_THREADS="${THREADS}" OMP_PROC_BIND="${affinity}" OMP_DISPLAY_AFFINITY=TRUE \
        "${BW_NATIVE}" > "${RUN_DIR}/affinity_${affinity}.log" 2>&1

    for rep in $(seq 1 "${REPS}"); do
        # Alternate backend order across repetitions.
        if (( rep % 2 == 1 )); then
            order=(busywait semaphore)
        else
            order=(semaphore busywait)
        fi

        for backend in "${order[@]}"; do
            echo "[native] affinity=${affinity} backend=${backend} rep=${rep}"
            run_native "$backend" "$affinity" "$rep"
            echo "[trace ] affinity=${affinity} backend=${backend} rep=${rep}"
            run_trace "$backend" "$affinity" "$rep"
        done
    done
done

echo "[analysis] reconstructing critical DAG"
python3 "${ANALYZER}" --root "${RUN_DIR}"

mkdir -p "${BASE_OUTDIR}"
ln -sfn "job_${SLURM_JOB_ID}" "${BASE_OUTDIR}/latest"

cat > "${RUN_DIR}/README_RESULTS.txt" <<TXT
Diagnostic WAIT critical-path oracle.
Job: ${SLURM_JOB_ID}
Threads: ${THREADS}
Reps: ${REPS}

Main outputs:
  aggregate_summary.csv    aggregate critical-path statistics
  run_summary.csv          per-trace-run statistics
  wait_oracle.csv          one row per actual blocking WAIT
  feature_auc.csv          discrimination of critical WAITs by local features
  perturbation_summary.csv tracing overhead vs native WAIT-only
  native_times.csv         native WAIT-only diagnostic timings

Trace timings are diagnostic only; use the normal experiment scripts for paper performance numbers.
TXT

echo
echo "DONE"
echo "Results: ${RUN_DIR}"
echo "Latest:  ${BASE_OUTDIR}/latest"
echo "Show me:"
echo "  cat ${BASE_OUTDIR}/latest/aggregate_summary.csv"
echo "  cat ${BASE_OUTDIR}/latest/feature_auc.csv"
echo "  cat ${BASE_OUTDIR}/latest/perturbation_summary.csv"
