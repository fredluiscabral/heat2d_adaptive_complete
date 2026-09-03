#!/bin/bash
#SBATCH --job-name=heat2d_sd_imbalance
#SBATCH --partition=lncc-cpu_amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH --output=heat2d_sd_imbalance-%j.out
#SBATCH --error=heat2d_sd_imbalance-%j.err

set -euo pipefail

# =============================================================================
# SDumont II — controlled synthetic progress-imbalance experiment
#
# Scientific question
# -------------------
# Does the usefulness of CRITICAL change causally when a controlled progress
# imbalance is introduced, while the PDE, numerical parameters, controller
# parameters and machine remain unchanged?
#
# Default matrix
# --------------
#   p            = 32 64 96 160
#   load         = 0 10 20 40 80 (%)
#   backend      = busywait semaphore
#   policy       = WAIT-only CRITICAL
#   PREDICT      = OFF
#   max_lead     = 2
#   N/T/TILE     = 8192 / 500 / 32
#
# Synthetic load
# --------------
# For each p, slow_thread=floor(p/2). After the real update_region() and before
# publication of interfaces/progress, that thread executes an additional
# READ-ONLY five-point-stencil-equivalent pass over a configurable fraction of
# its local rows. The accumulated value is written only to a padded volatile
# sink. Therefore the numerical fields are unchanged.
#
# IMPORTANT: L=20 means approximately 20% of the selected thread's LOCAL rows
# are revisited once per time step. It does NOT mean 20% wall-clock slowdown.
# load_effect_summary.csv measures the realized whole-program effect.
#
# Calibration
# -----------
# P/R is calibrated once per p and WAIT once per p/backend, with NO synthetic
# load. Those calibration files are then held fixed across all load levels.
# This is deliberate: the independent variable is the controlled progress
# perturbation, not a re-fitted cost model.
#
# Repetitions
# -----------
#   1 warm-up per case
#   20 clean timing repetitions per case
#   3 profile repetitions per case
#
# Repository sources are NOT modified. Instrumented copies and dedicated
# binaries are generated under OUTDIR/build/.
# =============================================================================

THREADS_LIST_STR="${THREADS_LIST:-32 64 96 160}"
read -r -a THREADS_LIST <<< "${THREADS_LIST_STR}"

LOAD_LEVELS_STR="${LOAD_LEVELS:-0 10 20 40 80}"
read -r -a LOAD_LEVELS <<< "${LOAD_LEVELS_STR}"

REPS="${REPS:-20}"
PROFILE_REPS="${PROFILE_REPS:-3}"
WARMUPS="${WARMUPS:-1}"

CAL_SAMPLES="${CAL_SAMPLES:-32}"
WAIT_CAL_SAMPLES="${WAIT_CAL_SAMPLES:-32}"
PROGRESS_LAMBDA="${PROGRESS_LAMBDA:-1.0}"
PROGRESS_BOOTSTRAP="${PROGRESS_BOOTSTRAP:-8}"

OUTDIR="${OUTDIR:-results/sdumont_controlled_imbalance_T500}"

BW_SRC="heat2d_explicit_omp_busywait_nobarrier_nofs_adaptive.cpp"
SM_SRC="heat2d_explicit_omp_sem_nobarrier_nofs_adaptive.cpp"
COMMON="heat2d_explicit_common.hpp"

PR_CAL="./heat2d_dependency_calibrate"
BW_WCAL="./heat2d_wait_calibrate_busywait"
SM_WCAL="./heat2d_wait_calibrate_semaphore"

BUILD_DIR="${OUTDIR}/build"
BW_SYN_SRC="${BUILD_DIR}/heat2d_adaptive_busywait_synthetic.cpp"
SM_SYN_SRC="${BUILD_DIR}/heat2d_adaptive_sem_synthetic.cpp"
BW_BIN="${BUILD_DIR}/heat2d_adaptive_busywait_synthetic"
SM_BIN="${BUILD_DIR}/heat2d_adaptive_sem_synthetic"
BW_PROF="${BUILD_DIR}/heat2d_adaptive_busywait_synthetic_profile"
SM_PROF="${BUILD_DIR}/heat2d_adaptive_sem_synthetic_profile"

CXX="${CXX:-g++}"
CXXFLAGS_SYN="${CXXFLAGS_SYN:--O3 -std=c++17 -fopenmp -Wall -Wextra -Wpedantic}"
LDFLAGS_SYN="${LDFLAGS_SYN:--fopenmp -pthread}"

cd "${SLURM_SUBMIT_DIR}"

export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_DYNAMIC=FALSE
export HEAT2D_MAX_LEAD=2

command -v python3 >/dev/null || { echo "ERRO: python3 nao encontrado." >&2; exit 1; }
command -v "${CXX}" >/dev/null || { echo "ERRO: compilador '${CXX}' nao encontrado." >&2; exit 1; }

get_param() {
    local key="$1"
    awk -F= -v k="$key" '
        $1 ~ "^[[:space:]]*" k "[[:space:]]*$" {
            gsub(/[[:space:]]/, "", $2)
            print $2
            exit
        }
    ' param.txt
}

[[ -f param.txt ]] || {
    echo "ERRO: param.txt nao encontrado em ${SLURM_SUBMIT_DIR}." >&2
    exit 2
}

N_VALUE="$(get_param N)"
T_VALUE="$(get_param T)"
TILE_VALUE="$(get_param TILE)"
WRITE_OUTPUT_VALUE="$(get_param WRITE_OUTPUT | tr '[:upper:]' '[:lower:]')"

[[ "${N_VALUE}" == "8192" ]] || {
    echo "ERRO: este experimento exige N=8192; atual=${N_VALUE:-ausente}." >&2
    exit 3
}
[[ "${T_VALUE}" == "500" ]] || {
    echo "ERRO: este experimento exige T=500; atual=${T_VALUE:-ausente}." >&2
    exit 4
}
[[ "${TILE_VALUE}" == "32" ]] || {
    echo "ERRO: este experimento exige TILE=32; atual=${TILE_VALUE:-ausente}." >&2
    exit 5
}
case "${WRITE_OUTPUT_VALUE}" in
    0|off|false|no) ;;
    *)
        echo "ERRO: configure WRITE_OUTPUT=0 em param.txt." >&2
        exit 6
        ;;
esac

(( REPS >= 2 )) || { echo "ERRO: REPS deve ser >= 2." >&2; exit 7; }
(( PROFILE_REPS >= 1 )) || { echo "ERRO: PROFILE_REPS deve ser >= 1." >&2; exit 7; }
(( WARMUPS >= 1 )) || { echo "ERRO: WARMUPS deve ser >= 1." >&2; exit 7; }

for p in "${THREADS_LIST[@]}"; do
    [[ "${p}" =~ ^[0-9]+$ ]] || { echo "ERRO: p invalido '${p}'." >&2; exit 7; }
    (( p >= 3 && p <= SLURM_CPUS_PER_TASK )) || {
        echo "ERRO: p=${p} fora de [3,${SLURM_CPUS_PER_TASK}]." >&2
        exit 7
    }
done

have_zero=0
for pct in "${LOAD_LEVELS[@]}"; do
    [[ "${pct}" =~ ^[0-9]+$ ]] || { echo "ERRO: carga invalida '${pct}'." >&2; exit 7; }
    (( pct >= 0 && pct <= 100 )) || { echo "ERRO: carga ${pct}% fora de [0,100]." >&2; exit 7; }
    (( pct == 0 )) && have_zero=1
done
(( have_zero == 1 )) || {
    echo "ERRO: LOAD_LEVELS deve conter 0 para o controle interno." >&2
    exit 7
}

for f in "${BW_SRC}" "${SM_SRC}" "${COMMON}"; do
    [[ -f "${f}" ]] || { echo "ERRO: fonte obrigatoria ausente: ${f}" >&2; exit 8; }
done

# Clean output safely to avoid mixing campaigns.
if [[ -z "${OUTDIR}" || "${OUTDIR}" == "/" || "${OUTDIR}" == "." ]]; then
    echo "ERRO: OUTDIR inseguro: '${OUTDIR}'." >&2
    exit 9
fi
rm -rf -- "${OUTDIR}"
mkdir -p "${BUILD_DIR}"
cp param.txt "${OUTDIR}/param.effective.txt"

# =============================================================================
# Build calibrators and generate instrumented experiment-only solvers.
# =============================================================================

echo "============================================================"
echo "SDUMONT II — CONTROLLED SYNTHETIC PROGRESS IMBALANCE"
echo "============================================================"
echo "Job ID              : ${SLURM_JOB_ID}"
echo "Node                : ${SLURM_NODELIST}"
echo "CPUs/task           : ${SLURM_CPUS_PER_TASK}"
echo "Threads             : ${THREADS_LIST[*]}"
echo "Load levels (%)     : ${LOAD_LEVELS[*]}"
echo "Timed repetitions   : ${REPS}"
echo "Profile repetitions : ${PROFILE_REPS}"
echo "Warm-ups/case       : ${WARMUPS}"
echo "P/R cal samples     : ${CAL_SAMPLES}"
echo "WAIT cal samples    : ${WAIT_CAL_SAMPLES}"
echo "Progress lambda     : ${PROGRESS_LAMBDA}"
echo "Progress bootstrap  : ${PROGRESS_BOOTSTRAP}"
echo "N/T/TILE            : ${N_VALUE}/${T_VALUE}/${TILE_VALUE}"
echo "PREDICT             : OFF during measurements"
echo "OMP_PLACES          : ${OMP_PLACES}"
echo "OMP_PROC_BIND       : ${OMP_PROC_BIND}"
echo "Output              : ${OUTDIR}"
echo "============================================================"

echo
echo "================ BUILD CORE =================================="
make -j "${SLURM_CPUS_PER_TASK}" core

for exe in "${PR_CAL}" "${BW_WCAL}" "${SM_WCAL}"; do
    [[ -x "${exe}" ]] || { echo "ERRO: executavel de calibracao ausente: ${exe}" >&2; exit 10; }
done

echo "[build] generating instrumented sources from current repository"

python3 - "${BW_SRC}" "${BW_SYN_SRC}" "${SM_SRC}" "${SM_SYN_SRC}" <<'PY'
from pathlib import Path
import re
import sys

pairs = [
    (Path(sys.argv[1]), Path(sys.argv[2]), "busywait"),
    (Path(sys.argv[3]), Path(sys.argv[4]), "semaphore"),
]

helper = r'''
// -----------------------------------------------------------------------------
// EXPERIMENT-ONLY controlled synthetic progress imbalance.
// Generated by run_sdumont_controlled_imbalance.slurm.
// -----------------------------------------------------------------------------
struct alignas(heat2d::CACHELINE_BYTES) SyntheticLoadSink {
    volatile double value;
    char padding[heat2d::CACHELINE_BYTES - sizeof(double)];
    SyntheticLoadSink() noexcept : value(0.0), padding{} {}
};
static_assert(sizeof(SyntheticLoadSink) == heat2d::CACHELINE_BYTES,
              "SyntheticLoadSink must occupy one cache line");

static inline int synthetic_load_thread_from_env(int nt) noexcept {
    int tid = nt / 2;
    if (const char* s = std::getenv("HEAT2D_SYNTHETIC_THREAD"); s && *s) {
        char* end = nullptr;
        const long x = std::strtol(s, &end, 10);
        if (end && *end == 0 && x >= 0 && x < nt)
            tid = static_cast<int>(x);
    }
    return tid;
}

static inline double synthetic_load_fraction_from_env() noexcept {
    double pct = 0.0;
    if (const char* s = std::getenv("HEAT2D_SYNTHETIC_LOAD_PCT"); s && *s) {
        char* end = nullptr;
        const double x = std::strtod(s, &end);
        if (end && *end == 0 && std::isfinite(x) && x >= 0.0)
            pct = x;
    }
    if (pct > 100.0) pct = 100.0;
    return pct / 100.0;
}

static inline void synthetic_stencil_equivalent_work(
        const Region& rg,
        const double* src,
        std::size_t ld,
        int N,
        double mu,
        double fraction,
        volatile double& sink) noexcept {
    if (!(fraction > 0.0) || rg.ni <= 0 || N <= 2)
        return;

    int rows = static_cast<int>(
        std::llround(fraction * static_cast<double>(rg.ni))
    );
    rows = std::max(1, std::min(rows, rg.ni));

    // Center the extra rows inside the selected thread's region so the
    // artificial work is not biased toward either interface.
    const int first = rg.first + (rg.ni - rows) / 2;
    const int last = first + rows - 1;

    double acc = 0.0;
    for (int i = first; i <= last; ++i) {
        for (int j = 1; j <= N - 2; ++j) {
            const std::size_t k = heat2d::idx2(i, j, ld);
            const double c = src[k];
            acc += c + mu * (
                src[heat2d::idx2(i + 1, j, ld)]
              + src[heat2d::idx2(i - 1, j, ld)]
              + src[heat2d::idx2(i, j + 1, ld)]
              + src[heat2d::idx2(i, j - 1, ld)]
              - 4.0 * c
            );
        }
    }
    sink = acc;
}
'''

setup = r'''
    const int synthetic_load_thread =
        synthetic_load_thread_from_env(ntmax);
    const double synthetic_load_fraction =
        synthetic_load_fraction_from_env();

    std::unique_ptr<SyntheticLoadSink[]> synthetic_load_sink(
        new SyntheticLoadSink[static_cast<std::size_t>(ntmax)]
    );

    std::cout << "Synthetic imbalance: on\n"
              << "Synthetic imbalance thread: "
              << synthetic_load_thread << '\n'
              << std::setprecision(16)
              << "Synthetic imbalance load pct: "
              << (100.0 * synthetic_load_fraction) << '\n';

'''

call = r'''
            // EXPERIMENT-ONLY: extra read-only stencil work after the real
            // update and before publishing the new interface/progress level.
            if (tid == synthetic_load_thread &&
                synthetic_load_fraction > 0.0) {
                synthetic_stencil_equivalent_work(
                    rg, src, ld, N, mu, synthetic_load_fraction,
                    synthetic_load_sink[static_cast<std::size_t>(tid)].value
                );
            }
'''

for src, dst, backend in pairs:
    text = src.read_text(encoding="utf-8")

    if "Politica progress-aware + criticality-aware" not in text:
        raise SystemExit(
            f"{src}: current progress+criticality controller marker not found"
        )

    region_re = re.compile(
        r"(struct\s+Region\s*\{\s*"
        r"int\s+first\s*=\s*1\s*;\s*"
        r"int\s+last\s*=\s*0\s*;\s*"
        r"int\s+ni\s*=\s*0\s*;\s*"
        r"\};)",
        re.S,
    )
    text, n_region = region_re.subn(r"\1\n" + helper, text, count=1)
    if n_region != 1:
        raise SystemExit(f"{src}: Region anchor count={n_region}, expected 1")

    anchor = "    WaitBackend wait_backend(ntmax);"
    if text.count(anchor) != 1:
        raise SystemExit(
            f"{src}: WaitBackend anchor count={text.count(anchor)}, expected 1"
        )
    text = text.replace(anchor, setup + "\n" + anchor, 1)

    update_re = re.compile(
        r"(\s*update_region\s*\(\s*rg\s*,\s*src\s*,\s*dst\s*,\s*ld\s*,\s*N\s*,"
        r"\s*TILE\s*,\s*mu\s*,\s*rn\.halo\s*,\s*rs\.halo\s*\)\s*;)"
    )
    text, n_update = update_re.subn(r"\1\n" + call, text, count=1)
    if n_update != 1:
        raise SystemExit(
            f"{src}: update_region anchor count={n_update}, expected 1"
        )

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8")
    print(f"generated {backend}: {dst}")
PY

sha256sum "${BW_SRC}" "${SM_SRC}" "${BW_SYN_SRC}" "${SM_SYN_SRC}" \
    > "${BUILD_DIR}/source_sha256.txt"

echo "[build] compiling dedicated timing/profile binaries"
# shellcheck disable=SC2086
"${CXX}" ${CXXFLAGS_SYN} -I. -DHEAT2D_PROFILE_STATS=0 \
    "${BW_SYN_SRC}" -o "${BW_BIN}" ${LDFLAGS_SYN}
# shellcheck disable=SC2086
"${CXX}" ${CXXFLAGS_SYN} -I. -DHEAT2D_PROFILE_STATS=0 \
    "${SM_SYN_SRC}" -o "${SM_BIN}" ${LDFLAGS_SYN}
# shellcheck disable=SC2086
"${CXX}" ${CXXFLAGS_SYN} -I. -DHEAT2D_PROFILE_STATS=1 \
    "${BW_SYN_SRC}" -o "${BW_PROF}" ${LDFLAGS_SYN}
# shellcheck disable=SC2086
"${CXX}" ${CXXFLAGS_SYN} -I. -DHEAT2D_PROFILE_STATS=1 \
    "${SM_SYN_SRC}" -o "${SM_PROF}" ${LDFLAGS_SYN}

for exe in "${BW_BIN}" "${SM_BIN}" "${BW_PROF}" "${SM_PROF}"; do
    [[ -x "${exe}" ]] || {
        echo "ERRO: executavel sintetico ausente: ${exe}" >&2
        exit 11
    }
done

# =============================================================================
# Calibration: once per p, fixed across load levels.
# =============================================================================

calibrate_p() {
    local p="$1"
    local ptag pdir cost bw_wait sm_wait
    ptag="$(printf '%03d' "${p}")"
    pdir="${OUTDIR}/p${ptag}"
    mkdir -p "${pdir}"

    cost="${pdir}/heat2d_cost_model.dat"
    bw_wait="${pdir}/heat2d_wait_cost_busywait.dat"
    sm_wait="${pdir}/heat2d_wait_cost_semaphore.dat"

    export OMP_NUM_THREADS="${p}"

    echo "[calibration] p=${p} PREDICT/RECOMPUTE (synthetic load OFF)"
    HEAT2D_ENABLE_PREDICT=1 \
    HEAT2D_ENABLE_RECOMPUTE=1 \
    HEAT2D_CALIBRATION_SAMPLES="${CAL_SAMPLES}" \
    HEAT2D_COST_FILE="${cost}" \
        "${PR_CAL}" > "${pdir}/calibration_pr.log" 2>&1
    [[ -s "${cost}" ]] || { echo "ERRO: modelo P/R nao gerado p=${p}." >&2; exit 12; }

    echo "[calibration] p=${p} WAIT busy-wait (synthetic load OFF)"
    HEAT2D_WAIT_CALIBRATION_SAMPLES="${WAIT_CAL_SAMPLES}" \
    HEAT2D_WAIT_COST_FILE="${bw_wait}" \
        "${BW_WCAL}" > "${pdir}/calibration_wait_busywait.log" 2>&1
    [[ -s "${bw_wait}" ]] || { echo "ERRO: WAIT busy-wait nao gerado p=${p}." >&2; exit 13; }

    echo "[calibration] p=${p} WAIT semaphore (synthetic load OFF)"
    HEAT2D_WAIT_CALIBRATION_SAMPLES="${WAIT_CAL_SAMPLES}" \
    HEAT2D_WAIT_COST_FILE="${sm_wait}" \
        "${SM_WCAL}" > "${pdir}/calibration_wait_semaphore.log" 2>&1
    [[ -s "${sm_wait}" ]] || { echo "ERRO: WAIT semaphore nao gerado p=${p}." >&2; exit 14; }
}

echo
echo "================ CALIBRATION ================================"
for p in "${THREADS_LIST[@]}"; do
    calibrate_p "${p}"
done

# =============================================================================
# Execution helpers.
# =============================================================================

check_log() {
    local log="$1" expected_thread="$2" expected_pct="$3" recompute="$4" criticality="$5"

    grep -q '^Tempo[[:space:]]*:' "${log}" || {
        echo "ERRO: Tempo ausente em ${log}" >&2; exit 15;
    }
    grep -q '^Linf:' "${log}" || {
        echo "ERRO: Linf ausente em ${log}" >&2; exit 15;
    }
    grep -q '^Synthetic imbalance: on$' "${log}" || {
        echo "ERRO: instrumentacao ausente em ${log}" >&2; exit 15;
    }
    grep -q "^Synthetic imbalance thread: ${expected_thread}$" "${log}" || {
        echo "ERRO: thread sintetica inesperada em ${log}" >&2; exit 15;
    }
    grep -q '^Adaptive predict: off$' "${log}" || {
        echo "ERRO: PREDICT deveria estar OFF em ${log}" >&2; exit 15;
    }

    if [[ "${recompute}" == "1" ]]; then
        grep -q '^Adaptive recompute: on$' "${log}" || {
            echo "ERRO: RECOMPUTE deveria estar ON em ${log}" >&2; exit 15;
        }
    else
        grep -q '^Adaptive recompute: off$' "${log}" || {
            echo "ERRO: RECOMPUTE deveria estar OFF em ${log}" >&2; exit 15;
        }
    fi

    if [[ "${criticality}" == "1" ]]; then
        grep -q '^Adaptive criticality-aware: on$' "${log}" || {
            echo "ERRO: CRITICALITY deveria estar ON em ${log}" >&2; exit 15;
        }
    else
        grep -q '^Adaptive criticality-aware: off$' "${log}" || {
            echo "ERRO: CRITICALITY deveria estar OFF em ${log}" >&2; exit 15;
        }
    fi

    python3 - "${log}" "${expected_pct}" <<'PY'
import re
import sys

path = sys.argv[1]
expected = float(sys.argv[2])
value = None
with open(path, errors="replace") as f:
    for line in f:
        m = re.match(r"^Synthetic imbalance load pct:\s*([-+0-9.eE]+)", line)
        if m:
            value = float(m.group(1))
            break
if value is None:
    raise SystemExit(f"missing synthetic load pct in {path}")
if abs(value - expected) > 1.0e-10:
    raise SystemExit(f"synthetic load mismatch in {path}: {value} != {expected}")
PY
}

exe_for() {
    local backend="$1" kind="$2"
    case "${backend}:${kind}" in
        busywait:ref|busywait:warmup) echo "${BW_BIN}" ;;
        semaphore:ref|semaphore:warmup) echo "${SM_BIN}" ;;
        busywait:prof) echo "${BW_PROF}" ;;
        semaphore:prof) echo "${SM_PROF}" ;;
        *) echo "ERRO interno: exe_for ${backend}:${kind}" >&2; exit 16 ;;
    esac
}

run_case() {
    local p="$1" backend="$2" policy="$3" pct="$4" kind="$5" rep="$6"
    local ptag ltag pdir ldir cost waitcost exe log recompute criticality slow_thread

    ptag="$(printf '%03d' "${p}")"
    ltag="$(printf '%03d' "${pct}")"
    pdir="${OUTDIR}/p${ptag}"
    ldir="${pdir}/load_${ltag}"
    mkdir -p "${ldir}"

    cost="${pdir}/heat2d_cost_model.dat"
    case "${backend}" in
        busywait)  waitcost="${pdir}/heat2d_wait_cost_busywait.dat" ;;
        semaphore) waitcost="${pdir}/heat2d_wait_cost_semaphore.dat" ;;
        *) echo "ERRO: backend invalido '${backend}'." >&2; exit 17 ;;
    esac

    case "${policy}" in
        wait)     recompute=0; criticality=0 ;;
        critical) recompute=1; criticality=1 ;;
        *) echo "ERRO: policy invalida '${policy}'." >&2; exit 17 ;;
    esac

    slow_thread=$(( p / 2 ))
    export OMP_NUM_THREADS="${p}"
    exe="$(exe_for "${backend}" "${kind}")"
    log="${ldir}/${backend}_${policy}_${kind}_r$(printf '%02d' "${rep}").log"

    HEAT2D_ENABLE_PREDICT=0 \
    HEAT2D_ENABLE_RECOMPUTE="${recompute}" \
    HEAT2D_ENABLE_CRITICALITY="${criticality}" \
    HEAT2D_COST_FILE="${cost}" \
    HEAT2D_WAIT_COST_FILE="${waitcost}" \
    HEAT2D_PROGRESS_LAMBDA="${PROGRESS_LAMBDA}" \
    HEAT2D_PROGRESS_BOOTSTRAP_SAMPLES="${PROGRESS_BOOTSTRAP}" \
    HEAT2D_SYNTHETIC_THREAD="${slow_thread}" \
    HEAT2D_SYNTHETIC_LOAD_PCT="${pct}" \
        "${exe}" > "${log}" 2>&1

    check_log "${log}" "${slow_thread}" "${pct}" "${recompute}" "${criticality}"
}

# Pair order is kept adjacent for the paired WAIT-vs-CRITICAL comparison.
BACKENDS=(busywait semaphore)
POLICIES=(wait critical)

rm -f "${OUTDIR}/times.csv" \
      "${OUTDIR}/profile_per_run.csv" \
      "${OUTDIR}/summary.csv" \
      "${OUTDIR}/profile_summary.csv" \
      "${OUTDIR}/load_effect_summary.csv" \
      "${OUTDIR}/order.csv" \
      "${OUTDIR}/run_metadata.txt" 2>/dev/null || true

echo "phase,threads,load_pct,rep,position,backend,policy" > "${OUTDIR}/order.csv"

# =============================================================================
# Warm-up.
# =============================================================================

echo
echo "================ WARM-UP ====================================="
for p in "${THREADS_LIST[@]}"; do
    for pct in "${LOAD_LEVELS[@]}"; do
        for ((w=1; w<=WARMUPS; ++w)); do
            for bi in 0 1; do
                backend="${BACKENDS[$bi]}"
                if (( (w + p + pct + bi) % 2 == 0 )); then
                    run_case "${p}" "${backend}" wait     "${pct}" warmup "${w}"
                    run_case "${p}" "${backend}" critical "${pct}" warmup "${w}"
                else
                    run_case "${p}" "${backend}" critical "${pct}" warmup "${w}"
                    run_case "${p}" "${backend}" wait     "${pct}" warmup "${w}"
                fi
            done
        done
        echo "warm-up p=${p} load=${pct}% complete"
    done
done

# =============================================================================
# Timing: rotate p, load and backend orders across global repetitions. WAIT and
# CRITICAL remain adjacent within each p/load/backend pair, with pair direction
# alternating to reduce temporal bias.
# =============================================================================

echo
echo "================ TIMING ======================================"
np="${#THREADS_LIST[@]}"
nl="${#LOAD_LEVELS[@]}"
nb="${#BACKENDS[@]}"

for ((rep=1; rep<=REPS; ++rep)); do
    p_shift=$(( (rep - 1) % np ))
    position=0

    for ((pk=0; pk<np; ++pk)); do
        pi=$(( (pk + p_shift) % np ))
        p="${THREADS_LIST[$pi]}"

        load_shift=$(( (rep + pk - 1) % nl ))
        for ((lk=0; lk<nl; ++lk)); do
            li=$(( (lk + load_shift) % nl ))
            pct="${LOAD_LEVELS[$li]}"

            backend_shift=$(( (rep + pk + lk - 1) % nb ))
            for ((bk=0; bk<nb; ++bk)); do
                bi=$(( (bk + backend_shift) % nb ))
                backend="${BACKENDS[$bi]}"

                if (( (rep + pk + lk + bk) % 2 == 0 )); then
                    pair=(wait critical)
                else
                    pair=(critical wait)
                fi

                for policy in "${pair[@]}"; do
                    position=$((position + 1))
                    echo "timing,${p},${pct},${rep},${position},${backend},${policy}" \
                        >> "${OUTDIR}/order.csv"
                    run_case "${p}" "${backend}" "${policy}" "${pct}" ref "${rep}"
                done
            done
        done
    done
    echo "timing global rep=${rep}/${REPS} complete"
done

# =============================================================================
# Profiling: same matrix, fewer repetitions.
# =============================================================================

echo
echo "================ PROFILING ==================================="
for ((rep=1; rep<=PROFILE_REPS; ++rep)); do
    p_shift=$(( (rep - 1) % np ))
    position=0

    for ((pk=0; pk<np; ++pk)); do
        pi=$(( (pk + p_shift) % np ))
        p="${THREADS_LIST[$pi]}"

        load_shift=$(( (rep + pk - 1) % nl ))
        for ((lk=0; lk<nl; ++lk)); do
            li=$(( (lk + load_shift) % nl ))
            pct="${LOAD_LEVELS[$li]}"

            backend_shift=$(( (rep + pk + lk - 1) % nb ))
            for ((bk=0; bk<nb; ++bk)); do
                bi=$(( (bk + backend_shift) % nb ))
                backend="${BACKENDS[$bi]}"

                if (( (rep + pk + lk + bk) % 2 == 0 )); then
                    pair=(wait critical)
                else
                    pair=(critical wait)
                fi

                for policy in "${pair[@]}"; do
                    position=$((position + 1))
                    echo "profile,${p},${pct},${rep},${position},${backend},${policy}" \
                        >> "${OUTDIR}/order.csv"
                    run_case "${p}" "${backend}" "${policy}" "${pct}" prof "${rep}"
                done
            done
        done
    done
    echo "profile global rep=${rep}/${PROFILE_REPS} complete"
done

# =============================================================================
# Consolidation and paired statistics.
# =============================================================================

python3 - "${OUTDIR}" "${REPS}" "${PROFILE_REPS}" "${T_VALUE}" \
    "${THREADS_LIST_STR}" "${LOAD_LEVELS_STR}" <<'PY'
import csv
import glob
import math
import os
import random
import re
import statistics
import sys

outdir = sys.argv[1]
reps = int(sys.argv[2])
preps = int(sys.argv[3])
T = int(sys.argv[4])
threads_list = [int(x) for x in sys.argv[5].split()]
load_levels = [float(x) for x in sys.argv[6].split()]

TIME_RE = re.compile(r"^Tempo\s*:\s*([-+0-9.eE]+)")
LINF_RE = re.compile(r"^Linf:\s*([-+0-9.eE]+)")

PROFILE_FIELDS = {
    "Adaptive actions READ": "read",
    "Adaptive actions RECOMPUTE": "recompute",
    "Adaptive actions PREDICT": "predict",
    "Adaptive actions WAIT": "wait",
    "Adaptive lead-guard waits": "lead_wait",
    "Lead-guard wait ticks sum": "lead_wait_ticks_sum",
    "Progress-blocked RECOMPUTE": "progress_blocked",
    "Progress penalty samples": "penalty_samples",
    "Progress penalty nonzero samples": "penalty_nonzero_samples",
    "Progress penalty observed mean": "penalty_mean_ticks",
    "Progress penalty nonzero fraction": "penalty_nonzero_fraction",
    "Critical RECOMPUTE": "critical_recompute",
    "Noncritical RECOMPUTE rejected": "noncritical_recompute_rejected",
    "Criticality-forced WAIT": "criticality_forced_wait",
}


def parse_log(path, profile=False):
    out = {}
    if profile:
        for key in PROFILE_FIELDS.values():
            out[key] = 0.0
    with open(path, errors="replace") as f:
        for line in f:
            m = TIME_RE.match(line)
            if m:
                out["time_s"] = float(m.group(1))
            m = LINF_RE.match(line)
            if m:
                out["linf"] = float(m.group(1))
            if profile:
                for prefix, key in PROFILE_FIELDS.items():
                    if line.startswith(prefix + ":"):
                        out[key] = float(line.split(":", 1)[1].strip().split()[0])
                        break
    if "time_s" not in out or "linf" not in out:
        raise RuntimeError(f"incomplete log: {path}")
    return out


def percentile(vals, p):
    if len(vals) == 1:
        return vals[0]
    k = (len(vals) - 1) * p
    f = int(k)
    c = min(f + 1, len(vals) - 1)
    if f == c:
        return vals[f]
    return vals[f] * (c - k) + vals[c] * (k - f)


def bootstrap_median_ci(values, seed, nboot=20000):
    rng = random.Random(seed)
    n = len(values)
    boots = []
    for _ in range(nboot):
        boots.append(statistics.median(values[rng.randrange(n)] for _ in range(n)))
    boots.sort()
    return percentile(boots, 0.025), percentile(boots, 0.975)


def p_load_from_path(path):
    ldir = os.path.basename(os.path.dirname(path))
    pdir = os.path.basename(os.path.dirname(os.path.dirname(path)))
    mp = re.fullmatch(r"p(\d+)", pdir)
    ml = re.fullmatch(r"load_(\d+)", ldir)
    if not mp or not ml:
        raise RuntimeError(f"bad p/load path: {path}")
    return int(mp.group(1)), float(int(ml.group(1)))


def select(rows, p, load, backend, policy):
    return [
        r for r in rows
        if r["threads"] == p
        and r["load_pct"] == load
        and r["backend"] == backend
        and r["policy"] == policy
    ]


def paired_ratio(num_rows, den_rows, expected):
    num = {int(r["rep"]): float(r["time_s"]) for r in num_rows}
    den = {int(r["rep"]): float(r["time_s"]) for r in den_rows}
    common = sorted(set(num) & set(den))
    if len(common) != expected:
        raise RuntimeError(f"paired comparison incomplete: {len(common)}/{expected}")
    return [num[r] / den[r] for r in common]


# -------------------------------------------------------------------------
# Timing rows
# -------------------------------------------------------------------------
times = []
for path in glob.glob(os.path.join(outdir, "p*", "load_*", "*_ref_r*.log")):
    m = re.fullmatch(
        r"(busywait|semaphore)_(wait|critical)_ref_r(\d+)\.log",
        os.path.basename(path),
    )
    if not m:
        continue
    p, load = p_load_from_path(path)
    times.append({
        "threads": p,
        "T": T,
        "slow_thread": p // 2,
        "load_pct": load,
        "backend": m.group(1),
        "policy": m.group(2),
        "rep": int(m.group(3)),
        **parse_log(path, False),
    })

expected_groups = len(threads_list) * len(load_levels) * 2 * 2
groups = {}
for r in times:
    groups.setdefault((r["threads"], r["load_pct"], r["backend"], r["policy"]), []).append(r)
if len(groups) != expected_groups:
    raise SystemExit(f"Expected {expected_groups} timing groups, got {len(groups)}")
for key, rr in groups.items():
    if len(rr) != reps:
        raise SystemExit(f"Incomplete timing group {key}: {len(rr)}/{reps}")

# Numerical equality inside every paired repetition.
for p in threads_list:
    for load in load_levels:
        for backend in ("busywait", "semaphore"):
            wait = select(times, p, load, backend, "wait")
            crit = select(times, p, load, backend, "critical")
            w = {int(r["rep"]): float(r["linf"]) for r in wait}
            c = {int(r["rep"]): float(r["linf"]) for r in crit}
            for rep in range(1, reps + 1):
                if not math.isclose(w[rep], c[rep], rel_tol=0.0, abs_tol=1e-15):
                    raise RuntimeError(
                        f"Linf mismatch p={p} load={load} backend={backend} rep={rep}: "
                        f"wait={w[rep]} critical={c[rep]}"
                    )

times_path = os.path.join(outdir, "times.csv")
with open(times_path, "w", newline="") as f:
    cols = ["threads", "T", "slow_thread", "load_pct", "backend", "policy", "rep", "time_s", "linf"]
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(sorted(times, key=lambda r: (
        r["threads"], r["load_pct"], r["backend"], r["policy"], r["rep"]
    )))

# -------------------------------------------------------------------------
# Summary: primary endpoint Scrit = WAIT / CRITICAL.
# -------------------------------------------------------------------------
summary = []
for p in threads_list:
    for load in load_levels:
        for backend in ("busywait", "semaphore"):
            wait = select(times, p, load, backend, "wait")
            crit = select(times, p, load, backend, "critical")
            med_wait = statistics.median(r["time_s"] for r in wait)
            med_crit = statistics.median(r["time_s"] for r in crit)
            wc = paired_ratio(wait, crit, reps)
            seed = 20260903 + p * 100000 + int(load) * 100 + (1 if backend == "busywait" else 2)
            ci = bootstrap_median_ci(wc, seed)

            summary.append({
                "threads": p,
                "T": T,
                "slow_thread": p // 2,
                "load_pct": load,
                "backend": backend,
                "reps": reps,
                "median_wait_s": med_wait,
                "median_critical_s": med_crit,
                "speedup_wait_over_critical_ratio_of_medians": med_wait / med_crit,
                "critical_gain_pct_ratio_of_medians": (1.0 - med_crit / med_wait) * 100.0,
                "median_paired_wait_over_critical": statistics.median(wc),
                "critical_wins_vs_wait": sum(x > 1.0 for x in wc),
                "bootstrap95_paired_speedup_lo": ci[0],
                "bootstrap95_paired_speedup_hi": ci[1],
                "min_paired_speedup": min(wc),
                "max_paired_speedup": max(wc),
                "linf_wait": statistics.median(r["linf"] for r in wait),
                "linf_critical": statistics.median(r["linf"] for r in crit),
            })

summary_path = os.path.join(outdir, "summary.csv")
with open(summary_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    w.writeheader()
    w.writerows(summary)

# -------------------------------------------------------------------------
# Whole-program effect of the injected load relative to load=0.
# This is not a direct measurement of per-thread slowdown; it quantifies the
# realized effect on each full-program policy/backend configuration.
# -------------------------------------------------------------------------
load_effect = []
for p in threads_list:
    for backend in ("busywait", "semaphore"):
        for policy in ("wait", "critical"):
            ref = select(times, p, 0.0, backend, policy)
            med_ref = statistics.median(r["time_s"] for r in ref)
            for load in load_levels:
                cur = select(times, p, load, backend, policy)
                med_cur = statistics.median(r["time_s"] for r in cur)
                ratios = paired_ratio(cur, ref, reps)
                seed = (
                    20260903 + p * 100000 + int(load) * 100
                    + (10 if backend == "busywait" else 20)
                    + (1 if policy == "wait" else 2)
                )
                ci = bootstrap_median_ci(ratios, seed)
                load_effect.append({
                    "threads": p,
                    "T": T,
                    "slow_thread": p // 2,
                    "load_pct": load,
                    "backend": backend,
                    "policy": policy,
                    "median_time_s": med_cur,
                    "median_zero_load_s": med_ref,
                    "slowdown_ratio_of_medians": med_cur / med_ref,
                    "slowdown_pct_ratio_of_medians": (med_cur / med_ref - 1.0) * 100.0,
                    "median_paired_slowdown_ratio": statistics.median(ratios),
                    "bootstrap95_paired_slowdown_lo": ci[0],
                    "bootstrap95_paired_slowdown_hi": ci[1],
                })

load_effect_path = os.path.join(outdir, "load_effect_summary.csv")
with open(load_effect_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(load_effect[0].keys()))
    w.writeheader()
    w.writerows(load_effect)

# -------------------------------------------------------------------------
# Profiling rows and medians.
# -------------------------------------------------------------------------
prof = []
for path in glob.glob(os.path.join(outdir, "p*", "load_*", "*_prof_r*.log")):
    m = re.fullmatch(
        r"(busywait|semaphore)_(wait|critical)_prof_r(\d+)\.log",
        os.path.basename(path),
    )
    if not m:
        continue
    p, load = p_load_from_path(path)
    prof.append({
        "threads": p,
        "T": T,
        "slow_thread": p // 2,
        "load_pct": load,
        "backend": m.group(1),
        "policy": m.group(2),
        "rep": int(m.group(3)),
        **parse_log(path, True),
    })

pgroups = {}
for r in prof:
    pgroups.setdefault((r["threads"], r["load_pct"], r["backend"], r["policy"]), []).append(r)
if len(pgroups) != expected_groups:
    raise SystemExit(f"Expected {expected_groups} profile groups, got {len(pgroups)}")
for key, rr in pgroups.items():
    if len(rr) != preps:
        raise SystemExit(f"Incomplete profile group {key}: {len(rr)}/{preps}")

profile_cols = [
    "threads", "T", "slow_thread", "load_pct", "backend", "policy", "rep",
    "time_s", "linf", "read", "recompute", "predict", "wait", "lead_wait",
    "lead_wait_ticks_sum", "progress_blocked", "penalty_samples",
    "penalty_nonzero_samples", "penalty_mean_ticks", "penalty_nonzero_fraction",
    "critical_recompute", "noncritical_recompute_rejected", "criticality_forced_wait",
]
profile_path = os.path.join(outdir, "profile_per_run.csv")
with open(profile_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=profile_cols)
    w.writeheader()
    w.writerows(sorted(prof, key=lambda r: (
        r["threads"], r["load_pct"], r["backend"], r["policy"], r["rep"]
    )))

profile_summary = []
for key in sorted(pgroups):
    p, load, backend, policy = key
    rr = pgroups[key]
    med = lambda field: statistics.median(float(r[field]) for r in rr)
    profile_summary.append({
        "threads": p,
        "T": T,
        "slow_thread": p // 2,
        "load_pct": load,
        "backend": backend,
        "policy": policy,
        "reps": len(rr),
        "median_profile_time_s": med("time_s"),
        "median_read": med("read"),
        "median_recompute": med("recompute"),
        "median_predict": med("predict"),
        "median_wait": med("wait"),
        "median_lead_wait": med("lead_wait"),
        "median_lead_wait_ticks_sum": med("lead_wait_ticks_sum"),
        "median_progress_blocked": med("progress_blocked"),
        "median_penalty_samples": med("penalty_samples"),
        "median_penalty_nonzero_samples": med("penalty_nonzero_samples"),
        "median_penalty_mean_ticks": med("penalty_mean_ticks"),
        "median_penalty_nonzero_fraction": med("penalty_nonzero_fraction"),
        "median_critical_recompute": med("critical_recompute"),
        "median_noncritical_recompute_rejected": med("noncritical_recompute_rejected"),
        "median_criticality_forced_wait": med("criticality_forced_wait"),
        "median_linf": med("linf"),
    })

profile_summary_path = os.path.join(outdir, "profile_summary.csv")
with open(profile_summary_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(profile_summary[0].keys()))
    w.writeheader()
    w.writerows(profile_summary)

print()
print("=== SDUMONT CONTROLLED IMBALANCE: CRITICAL vs WAIT-only ===")
print(" p  load% backend    Scrit(med)  paired-Scrit  wins   bootstrap95")
for r in summary:
    print(
        f"{r['threads']:3d} {r['load_pct']:5.0f} {r['backend']:10s} "
        f"{r['speedup_wait_over_critical_ratio_of_medians']:10.5f} "
        f"{r['median_paired_wait_over_critical']:13.5f} "
        f"{r['critical_wins_vs_wait']:2d}/{r['reps']:2d} "
        f"[{r['bootstrap95_paired_speedup_lo']:.6f},"
        f"{r['bootstrap95_paired_speedup_hi']:.6f}]"
    )

print()
print("Files:")
for path in (times_path, summary_path, load_effect_path, profile_path, profile_summary_path):
    print(" ", path)
PY

# =============================================================================
# Metadata after successful completion.
# =============================================================================

{
    echo "date_start_or_end=$(date --iso-8601=seconds 2>/dev/null || date)"
    echo "job_id=${SLURM_JOB_ID}"
    echo "node=${SLURM_NODELIST}"
    echo "cpus_per_task=${SLURM_CPUS_PER_TASK}"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "machine=SDumontII"
    echo "experiment=controlled_synthetic_progress_imbalance"
    echo "purpose=mechanistic_causal_sweep"
    echo "threads=${THREADS_LIST[*]}"
    echo "load_levels=${LOAD_LEVELS[*]}"
    echo "N=${N_VALUE}"
    echo "T=${T_VALUE}"
    echo "TILE=${TILE_VALUE}"
    echo "REPS=${REPS}"
    echo "PROFILE_REPS=${PROFILE_REPS}"
    echo "WARMUPS=${WARMUPS}"
    echo "CAL_SAMPLES=${CAL_SAMPLES}"
    echo "WAIT_CAL_SAMPLES=${WAIT_CAL_SAMPLES}"
    echo "PREDICT=0"
    echo "HEAT2D_MAX_LEAD=${HEAT2D_MAX_LEAD}"
    echo "PROGRESS_LAMBDA=${PROGRESS_LAMBDA}"
    echo "PROGRESS_BOOTSTRAP=${PROGRESS_BOOTSTRAP}"
    echo "calibration_per_p=yes"
    echo "calibration_reused_across_load_levels=yes"
    echo "calibration_synthetic_load=off"
    echo "slow_thread_rule=floor(p/2)"
    echo "synthetic_load_definition=read-only stencil-equivalent rows after update and before publication"
    echo "synthetic_load_pct_semantics=percentage of selected-thread local rows revisited once per time step"
    echo "CXX=${CXX}"
    echo "CXXFLAGS_SYN=${CXXFLAGS_SYN}"
    echo "LDFLAGS_SYN=${LDFLAGS_SYN}"
    echo "OMP_PLACES=${OMP_PLACES}"
    echo "OMP_PROC_BIND=${OMP_PROC_BIND}"
} > "${OUTDIR}/run_metadata.txt"

echo
echo "============================================================"
echo "SDumont controlled-imbalance experiment complete."
echo "Main files:"
echo "  ${OUTDIR}/summary.csv"
echo "  ${OUTDIR}/times.csv"
echo "  ${OUTDIR}/load_effect_summary.csv"
echo "  ${OUTDIR}/profile_summary.csv"
echo "  ${OUTDIR}/profile_per_run.csv"
echo "  ${OUTDIR}/order.csv"
echo "  ${OUTDIR}/run_metadata.txt"
echo "============================================================"
