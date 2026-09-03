#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Heat2D — Haswell 32: controlled synthetic-imbalance sweep
#
# Scientific question
# -------------------
# Does CRITICAL become useful on the same Haswell-32 system when a controlled
# progress imbalance is introduced, while all numerical data and solver
# parameters remain unchanged?
#
# Design
# ------
#   threads      : 32
#   N            : 8192
#   T            : 500
#   TILE         : 32
#   slow thread  : 16 (zero-based)
#   load levels  : 0 5 10 20 40 (% of one local stencil-equivalent sweep)
#   backends     : busywait, semaphore
#   policies     : wait, progress, critical
#   PREDICT      : OFF
#   max_lead     : 2
#
# Exploratory sweep:
#   1 warm-up per case
#   10 clean timing repetitions per case
#   3 profile repetitions per case
#
# Synthetic load:
#   after the real update_region() and before interface publication, the
#   selected thread performs an additional read-only five-point-stencil pass
#   over a configurable fraction of its local rows. The accumulated value is
#   stored once in a per-thread padded volatile sink.
#
# Therefore the numerical fields are unchanged. "5%" means approximately 5%
# of one LOCAL stencil sweep in extra rows, not 5% wall-clock slowdown.
#
# The repository sources are NOT modified. Instrumented source copies and four
# dedicated binaries are generated under OUTDIR/build/.
# =============================================================================

THREADS="${THREADS:-32}"
N_VALUE="${N_VALUE:-8192}"
T_VALUE="${T_VALUE:-500}"
TILE_VALUE="${TILE_VALUE:-32}"

SLOW_THREAD="${SLOW_THREAD:-16}"
LOAD_LEVELS_STR="${LOAD_LEVELS:-0 5 10 20 40}"
read -r -a LOAD_LEVELS <<< "${LOAD_LEVELS_STR}"

REPS="${REPS:-10}"
PROFILE_REPS="${PROFILE_REPS:-3}"
WARMUPS="${WARMUPS:-1}"

PROGRESS_LAMBDA="${PROGRESS_LAMBDA:-1.0}"
PROGRESS_BOOTSTRAP="${PROGRESS_BOOTSTRAP:-8}"

CALIBRATION_DIR="${CALIBRATION_DIR:-results/haswell32_T_selection/calibration}"
OUTDIR="${OUTDIR:-results/haswell32_controlled_imbalance_T${T_VALUE}}"

BW_SRC="heat2d_explicit_omp_busywait_nobarrier_nofs_adaptive.cpp"
SM_SRC="heat2d_explicit_omp_sem_nobarrier_nofs_adaptive.cpp"
COMMON="heat2d_explicit_common.hpp"

COST="${CALIBRATION_DIR}/heat2d_cost_model.dat"
BW_WAIT="${CALIBRATION_DIR}/heat2d_wait_cost_busywait.dat"
SM_WAIT="${CALIBRATION_DIR}/heat2d_wait_cost_semaphore.dat"

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

export OMP_NUM_THREADS="${THREADS}"
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_DYNAMIC=FALSE
export HEAT2D_MAX_LEAD=2

command -v python3 >/dev/null || { echo "ERRO: python3 nao encontrado." >&2; exit 1; }
command -v "${CXX}" >/dev/null || { echo "ERRO: compilador '${CXX}' nao encontrado." >&2; exit 1; }

[[ -f param.txt ]] || { echo "ERRO: param.txt nao encontrado." >&2; exit 1; }

for f in "${BW_SRC}" "${SM_SRC}" "${COMMON}"; do
    [[ -f "${f}" ]] || { echo "ERRO: fonte obrigatoria nao encontrada: ${f}" >&2; exit 1; }
done

for f in "${COST}" "${BW_WAIT}" "${SM_WAIT}"; do
    [[ -s "${f}" ]] || {
        echo "ERRO: calibracao Haswell p=32 nao encontrada: ${f}" >&2
        echo "CALIBRATION_DIR=${CALIBRATION_DIR}" >&2
        exit 2
    }
done

[[ "${THREADS}" == "32" ]] || { echo "ERRO: este experimento requer THREADS=32." >&2; exit 2; }

if (( SLOW_THREAD < 0 || SLOW_THREAD >= THREADS )); then
    echo "ERRO: SLOW_THREAD=${SLOW_THREAD} fora de [0,$((THREADS-1))]." >&2
    exit 2
fi

(( REPS >= 2 )) || { echo "ERRO: REPS deve ser >= 2." >&2; exit 2; }
(( PROFILE_REPS >= 1 )) || { echo "ERRO: PROFILE_REPS deve ser >= 1." >&2; exit 2; }

for pct in "${LOAD_LEVELS[@]}"; do
    [[ "${pct}" =~ ^[0-9]+$ ]] || {
        echo "ERRO: carga invalida '${pct}'." >&2
        exit 2
    }
done

mkdir -p "${OUTDIR}" "${BUILD_DIR}"

# Preserve param.txt.
PARAM_BACKUP="$(mktemp ./param.txt.haswell_synthetic.XXXXXX)"
cp param.txt "${PARAM_BACKUP}"
restore_param() {
    if [[ -f "${PARAM_BACKUP}" ]]; then
        cp "${PARAM_BACKUP}" param.txt
        rm -f "${PARAM_BACKUP}"
    fi
}
trap restore_param EXIT INT TERM

python3 - "${N_VALUE}" "${T_VALUE}" "${TILE_VALUE}" <<'PY'
import re, sys
path="param.txt"
values={"N":sys.argv[1],"T":sys.argv[2],"TILE":sys.argv[3],"WRITE_OUTPUT":"0"}
with open(path,encoding="utf-8") as f:
    lines=f.readlines()
out=[]; seen=set()
for line in lines:
    raw=line.rstrip("\n")
    m=re.match(r"^(\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*=\s*)(.*?)(\s*)$",raw)
    if m and m.group(2) in values:
        k=m.group(2)
        out.append(f"{m.group(1)}{k}{m.group(3)}{values[k]}\n")
        seen.add(k)
    else:
        out.append(line)
for k,v in values.items():
    if k not in seen:
        out.append(f"{k} = {v}\n")
with open(path,"w",encoding="utf-8") as f:
    f.writelines(out)
PY

cp param.txt "${OUTDIR}/param.effective.txt"

echo "[build] generating instrumented sources from current repository"

python3 - "${BW_SRC}" "${BW_SYN_SRC}" "${SM_SRC}" "${SM_SYN_SRC}" <<'PY'
from pathlib import Path
import re, sys

pairs=[
    (Path(sys.argv[1]),Path(sys.argv[2]),"busywait"),
    (Path(sys.argv[3]),Path(sys.argv[4]),"semaphore"),
]

helper=r'''
// -----------------------------------------------------------------------------
// EXPERIMENT-ONLY controlled synthetic imbalance.
// Generated by run_haswell32_controlled_imbalance.sh.
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
        if (end && *end == '\0' && x >= 0 && x < nt)
            tid = static_cast<int>(x);
    }
    return tid;
}

static inline double synthetic_load_fraction_from_env() noexcept {
    double pct = 0.0;
    if (const char* s = std::getenv("HEAT2D_SYNTHETIC_LOAD_PCT"); s && *s) {
        char* end = nullptr;
        const double x = std::strtod(s, &end);
        if (end && *end == '\0' && std::isfinite(x) && x >= 0.0)
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

setup=r'''
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

call=r'''
            // EXPERIMENT-ONLY: artificial work after the real update and
            // before publishing the new interface/progress level.
            if (tid == synthetic_load_thread &&
                synthetic_load_fraction > 0.0) {
                synthetic_stencil_equivalent_work(
                    rg, src, ld, N, mu, synthetic_load_fraction,
                    synthetic_load_sink[static_cast<std::size_t>(tid)].value
                );
            }
'''

for src,dst,backend in pairs:
    text=src.read_text(encoding="utf-8")

    if "Politica progress-aware + criticality-aware" not in text:
        raise SystemExit(
            f"{src}: current progress+criticality controller marker not found"
        )

    region_re=re.compile(
        r"(struct\s+Region\s*\{\s*"
        r"int\s+first\s*=\s*1\s*;\s*"
        r"int\s+last\s*=\s*0\s*;\s*"
        r"int\s+ni\s*=\s*0\s*;\s*"
        r"\};)", re.S
    )
    text,n_region=region_re.subn(r"\1\n"+helper,text,count=1)
    if n_region != 1:
        raise SystemExit(f"{src}: Region anchor count={n_region}, expected 1")

    anchor="    WaitBackend wait_backend(ntmax);"
    if text.count(anchor) != 1:
        raise SystemExit(
            f"{src}: WaitBackend anchor count={text.count(anchor)}, expected 1"
        )
    text=text.replace(anchor,setup+"\n"+anchor,1)

    update_re=re.compile(
        r"(\s*update_region\s*\(\s*rg\s*,\s*src\s*,\s*dst\s*,\s*ld\s*,\s*N\s*,"
        r"\s*TILE\s*,\s*mu\s*,\s*rn\.halo\s*,\s*rs\.halo\s*\)\s*;)"
    )
    text,n_update=update_re.subn(r"\1\n"+call,text,count=1)
    if n_update != 1:
        raise SystemExit(
            f"{src}: update_region anchor count={n_update}, expected 1"
        )

    dst.parent.mkdir(parents=True,exist_ok=True)
    dst.write_text(text,encoding="utf-8")
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
    [[ -x "${exe}" ]] || { echo "ERRO: executavel sintetico ausente: ${exe}" >&2; exit 3; }
done

check_log() {
    local log="$1" pct="$2" recompute="$3" criticality="$4"

    grep -q '^Tempo[[:space:]]*:' "${log}" || { echo "ERRO: Tempo ausente em ${log}" >&2; exit 4; }
    grep -q '^Linf:' "${log}" || { echo "ERRO: Linf ausente em ${log}" >&2; exit 4; }
    grep -q '^Synthetic imbalance: on$' "${log}" || { echo "ERRO: instrumentacao ausente em ${log}" >&2; exit 4; }
    grep -q "^Synthetic imbalance thread: ${SLOW_THREAD}$" "${log}" || { echo "ERRO: thread sintetica inesperada em ${log}" >&2; exit 4; }
    grep -q '^Adaptive predict: off$' "${log}" || { echo "ERRO: PREDICT deveria estar OFF em ${log}" >&2; exit 4; }

    if [[ "${recompute}" == "1" ]]; then
        grep -q '^Adaptive recompute: on$' "${log}" || { echo "ERRO: RECOMPUTE deveria estar ON em ${log}" >&2; exit 4; }
    else
        grep -q '^Adaptive recompute: off$' "${log}" || { echo "ERRO: RECOMPUTE deveria estar OFF em ${log}" >&2; exit 4; }
    fi

    if [[ "${criticality}" == "1" ]]; then
        grep -q '^Adaptive criticality-aware: on$' "${log}" || { echo "ERRO: CRITICALITY deveria estar ON em ${log}" >&2; exit 4; }
    else
        grep -q '^Adaptive criticality-aware: off$' "${log}" || { echo "ERRO: CRITICALITY deveria estar OFF em ${log}" >&2; exit 4; }
    fi

    python3 - "${log}" "${pct}" <<'PY'
import re,sys
path=sys.argv[1]; expected=float(sys.argv[2]); value=None
with open(path,errors="replace") as f:
    for line in f:
        m=re.match(r"^Synthetic imbalance load pct:\s*([-+0-9.eE]+)",line)
        if m:
            value=float(m.group(1)); break
if value is None:
    raise SystemExit(f"missing synthetic load pct in {path}")
if abs(value-expected)>1.0e-10:
    raise SystemExit(f"synthetic load mismatch in {path}: {value} != {expected}")
PY
}

run_case() {
    local backend="$1" policy="$2" pct="$3" exe="$4" log="$5"
    local recompute criticality waitcost

    case "${policy}" in
        wait) recompute=0; criticality=0 ;;
        progress) recompute=1; criticality=0 ;;
        critical) recompute=1; criticality=1 ;;
        *) echo "ERRO: policy invalida: ${policy}" >&2; exit 5 ;;
    esac

    case "${backend}" in
        busywait) waitcost="${BW_WAIT}" ;;
        semaphore) waitcost="${SM_WAIT}" ;;
        *) echo "ERRO: backend invalido: ${backend}" >&2; exit 5 ;;
    esac

    HEAT2D_ENABLE_PREDICT=0 \
    HEAT2D_ENABLE_RECOMPUTE="${recompute}" \
    HEAT2D_ENABLE_CRITICALITY="${criticality}" \
    HEAT2D_COST_FILE="${COST}" \
    HEAT2D_WAIT_COST_FILE="${waitcost}" \
    HEAT2D_PROGRESS_LAMBDA="${PROGRESS_LAMBDA}" \
    HEAT2D_PROGRESS_BOOTSTRAP_SAMPLES="${PROGRESS_BOOTSTRAP}" \
    HEAT2D_SYNTHETIC_THREAD="${SLOW_THREAD}" \
    HEAT2D_SYNTHETIC_LOAD_PCT="${pct}" \
        "${exe}" > "${log}" 2>&1

    check_log "${log}" "${pct}" "${recompute}" "${criticality}"
}

exe_for() {
    local backend="$1" kind="$2"
    case "${backend}:${kind}" in
        busywait:ref|busywait:warmup) echo "${BW_BIN}" ;;
        semaphore:ref|semaphore:warmup) echo "${SM_BIN}" ;;
        busywait:prof) echo "${BW_PROF}" ;;
        semaphore:prof) echo "${SM_PROF}" ;;
        *) echo "ERRO interno: exe_for ${backend}:${kind}" >&2; exit 6 ;;
    esac
}

CASES=(
    "busywait:wait"
    "semaphore:wait"
    "busywait:progress"
    "semaphore:progress"
    "busywait:critical"
    "semaphore:critical"
)

rm -f "${OUTDIR}/times.csv" "${OUTDIR}/profile_per_run.csv" \
      "${OUTDIR}/summary.csv" "${OUTDIR}/profile_summary.csv" \
      "${OUTDIR}/load_effect_summary.csv" "${OUTDIR}/order.csv" \
      "${OUTDIR}/run_metadata.txt" 2>/dev/null || true

echo "phase,load_pct,rep,position,backend,policy" > "${OUTDIR}/order.csv"

echo
echo "============================================================"
echo "HASWELL 32 — CONTROLLED SYNTHETIC IMBALANCE"
echo "============================================================"
echo "threads=${THREADS}"
echo "N=${N_VALUE}"
echo "T=${T_VALUE}"
echo "TILE=${TILE_VALUE}"
echo "slow_thread=${SLOW_THREAD}"
echo "load_levels=${LOAD_LEVELS[*]}"
echo "timing_reps=${REPS}"
echo "profile_reps=${PROFILE_REPS}"
echo "============================================================"

echo
echo "[warm-up]"
for pct in "${LOAD_LEVELS[@]}"; do
    tag="$(printf '%03.0f' "${pct}")"
    ldir="${OUTDIR}/load_${tag}"
    mkdir -p "${ldir}"

    for ((w=1; w<=WARMUPS; ++w)); do
        shift=$(( (w - 1) % ${#CASES[@]} ))
        for ((k=0; k<${#CASES[@]}; ++k)); do
            idx=$(( (k + shift) % ${#CASES[@]} ))
            IFS=: read -r backend policy <<< "${CASES[$idx]}"
            exe="$(exe_for "${backend}" warmup)"
            log="${ldir}/${backend}_${policy}_warmup_r$(printf '%02d' "${w}").log"
            run_case "${backend}" "${policy}" "${pct}" "${exe}" "${log}"
        done
    done
    echo "warm-up load=${pct}% complete"
done

echo
echo "[timing] ${REPS} super-blocks"
nloads="${#LOAD_LEVELS[@]}"
ncases="${#CASES[@]}"

for ((rep=1; rep<=REPS; ++rep)); do
    load_shift=$(( (rep - 1) % nloads ))
    for ((lk=0; lk<nloads; ++lk)); do
        li=$(( (lk + load_shift) % nloads ))
        pct="${LOAD_LEVELS[$li]}"
        tag="$(printf '%03.0f' "${pct}")"
        ldir="${OUTDIR}/load_${tag}"
        case_shift=$(( (rep + lk - 2) % ncases ))
        position=0

        for ((ck=0; ck<ncases; ++ck)); do
            ci=$(( (ck + case_shift) % ncases ))
            IFS=: read -r backend policy <<< "${CASES[$ci]}"
            position=$((position + 1))
            echo "timing,${pct},${rep},${position},${backend},${policy}" >> "${OUTDIR}/order.csv"

            exe="$(exe_for "${backend}" ref)"
            log="${ldir}/${backend}_${policy}_ref_r$(printf '%02d' "${rep}").log"
            run_case "${backend}" "${policy}" "${pct}" "${exe}" "${log}"
        done
    done
    echo "timing super-block rep=${rep}/${REPS} complete"
done

echo
echo "[profiling] ${PROFILE_REPS} super-blocks"
for ((rep=1; rep<=PROFILE_REPS; ++rep)); do
    load_shift=$(( (rep - 1) % nloads ))
    for ((lk=0; lk<nloads; ++lk)); do
        li=$(( (lk + load_shift) % nloads ))
        pct="${LOAD_LEVELS[$li]}"
        tag="$(printf '%03.0f' "${pct}")"
        ldir="${OUTDIR}/load_${tag}"
        case_shift=$(( (rep + lk - 2) % ncases ))
        position=0

        for ((ck=0; ck<ncases; ++ck)); do
            ci=$(( (ck + case_shift) % ncases ))
            IFS=: read -r backend policy <<< "${CASES[$ci]}"
            position=$((position + 1))
            echo "profile,${pct},${rep},${position},${backend},${policy}" >> "${OUTDIR}/order.csv"

            exe="$(exe_for "${backend}" prof)"
            log="${ldir}/${backend}_${policy}_prof_r$(printf '%02d' "${rep}").log"
            run_case "${backend}" "${policy}" "${pct}" "${exe}" "${log}"
        done
    done
    echo "profile super-block rep=${rep}/${PROFILE_REPS} complete"
done

python3 - "${OUTDIR}" "${REPS}" "${PROFILE_REPS}" "${THREADS}" "${T_VALUE}" \
    "${SLOW_THREAD}" "${LOAD_LEVELS_STR}" <<'PY'
import csv,glob,os,random,re,statistics,sys

outdir=sys.argv[1]
reps=int(sys.argv[2])
preps=int(sys.argv[3])
threads=int(sys.argv[4])
T=int(sys.argv[5])
slow_thread=int(sys.argv[6])
load_levels=[float(x) for x in sys.argv[7].split()]

TIME_RE=re.compile(r"^Tempo\s*:\s*([-+0-9.eE]+)")
LINF_RE=re.compile(r"^Linf:\s*([-+0-9.eE]+)")

PROFILE_FIELDS={
    "Adaptive actions READ":"read",
    "Adaptive actions RECOMPUTE":"recompute",
    "Adaptive actions PREDICT":"predict",
    "Adaptive actions WAIT":"wait",
    "Adaptive lead-guard waits":"lead_wait",
    "Lead-guard wait ticks sum":"lead_wait_ticks_sum",
    "Progress-blocked RECOMPUTE":"progress_blocked",
    "Progress penalty samples":"penalty_samples",
    "Progress penalty nonzero samples":"penalty_nonzero_samples",
    "Progress penalty observed mean":"penalty_mean_ticks",
    "Progress penalty nonzero fraction":"penalty_nonzero_fraction",
    "Critical RECOMPUTE":"critical_recompute",
    "Noncritical RECOMPUTE rejected":"noncritical_recompute_rejected",
    "Criticality-forced WAIT":"criticality_forced_wait",
}

def parse_log(path,profile=False):
    out={}
    if profile:
        for k in PROFILE_FIELDS.values():
            out[k]=0.0
    with open(path,errors="replace") as f:
        for line in f:
            m=TIME_RE.match(line)
            if m: out["time_s"]=float(m.group(1))
            m=LINF_RE.match(line)
            if m: out["linf"]=float(m.group(1))
            if profile:
                for prefix,key in PROFILE_FIELDS.items():
                    if line.startswith(prefix+":"):
                        out[key]=float(line.split(":",1)[1].strip().split()[0])
                        break
    if "time_s" not in out or "linf" not in out:
        raise RuntimeError(f"incomplete log: {path}")
    return out

def percentile(vals,p):
    if len(vals)==1: return vals[0]
    k=(len(vals)-1)*p
    f=int(k); c=min(f+1,len(vals)-1)
    if f==c: return vals[f]
    return vals[f]*(c-k)+vals[c]*(k-f)

def bootstrap_median_ci(values,seed,nboot=20000):
    rng=random.Random(seed); n=len(values); boots=[]
    for _ in range(nboot):
        boots.append(statistics.median(values[rng.randrange(n)] for _ in range(n)))
    boots.sort()
    return percentile(boots,0.025),percentile(boots,0.975)

def load_from_dir(path):
    parent=os.path.basename(os.path.dirname(path))
    m=re.fullmatch(r"load_(\d+)",parent)
    if not m: raise RuntimeError(f"bad load dir: {path}")
    return float(int(m.group(1)))

def select(rows,load,backend,policy):
    return [r for r in rows if r["load_pct"]==load and r["backend"]==backend and r["policy"]==policy]

def paired_ratio(num_rows,den_rows,expected):
    num={int(r["rep"]):float(r["time_s"]) for r in num_rows}
    den={int(r["rep"]):float(r["time_s"]) for r in den_rows}
    common=sorted(set(num)&set(den))
    if len(common)!=expected:
        raise RuntimeError(f"paired comparison incomplete: {len(common)}/{expected}")
    return [num[r]/den[r] for r in common]

times=[]
for path in glob.glob(os.path.join(outdir,"load_*","*_ref_r*.log")):
    m=re.fullmatch(r"(busywait|semaphore)_(wait|progress|critical)_ref_r(\d+)\.log",os.path.basename(path))
    if m:
        times.append({
            "threads":threads,"T":T,"slow_thread":slow_thread,
            "load_pct":load_from_dir(path),"backend":m.group(1),
            "policy":m.group(2),"rep":int(m.group(3)),
            **parse_log(path,False)
        })

expected_groups=len(load_levels)*2*3
groups={}
for r in times:
    groups.setdefault((r["load_pct"],r["backend"],r["policy"]),[]).append(r)
if len(groups)!=expected_groups:
    raise SystemExit(f"Expected {expected_groups} timing groups, got {len(groups)}")
for key,rr in groups.items():
    if len(rr)!=reps:
        raise SystemExit(f"Incomplete timing group {key}: {len(rr)}/{reps}")

times_path=os.path.join(outdir,"times.csv")
with open(times_path,"w",newline="") as f:
    cols=["threads","T","slow_thread","load_pct","backend","policy","rep","time_s","linf"]
    w=csv.DictWriter(f,fieldnames=cols); w.writeheader()
    w.writerows(sorted(times,key=lambda r:(r["load_pct"],r["backend"],r["policy"],r["rep"])))

prof=[]
for path in glob.glob(os.path.join(outdir,"load_*","*_prof_r*.log")):
    m=re.fullmatch(r"(busywait|semaphore)_(wait|progress|critical)_prof_r(\d+)\.log",os.path.basename(path))
    if m:
        prof.append({
            "threads":threads,"T":T,"slow_thread":slow_thread,
            "load_pct":load_from_dir(path),"backend":m.group(1),
            "policy":m.group(2),"rep":int(m.group(3)),
            **parse_log(path,True)
        })

pgroups={}
for r in prof:
    pgroups.setdefault((r["load_pct"],r["backend"],r["policy"]),[]).append(r)
if len(pgroups)!=expected_groups:
    raise SystemExit(f"Expected {expected_groups} profile groups, got {len(pgroups)}")
for key,rr in pgroups.items():
    if len(rr)!=preps:
        raise SystemExit(f"Incomplete profile group {key}: {len(rr)}/{preps}")

profile_path=os.path.join(outdir,"profile_per_run.csv")
profile_cols=[
    "threads","T","slow_thread","load_pct","backend","policy","rep","time_s","linf",
    "read","recompute","predict","wait","lead_wait","lead_wait_ticks_sum",
    "progress_blocked","penalty_samples","penalty_nonzero_samples","penalty_mean_ticks",
    "penalty_nonzero_fraction","critical_recompute","noncritical_recompute_rejected",
    "criticality_forced_wait"
]
with open(profile_path,"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=profile_cols); w.writeheader()
    w.writerows(sorted(prof,key=lambda r:(r["load_pct"],r["backend"],r["policy"],r["rep"])))

summary=[]
for load in load_levels:
    for backend in ("busywait","semaphore"):
        wait=select(times,load,backend,"wait")
        prog=select(times,load,backend,"progress")
        crit=select(times,load,backend,"critical")
        med_wait=statistics.median(r["time_s"] for r in wait)
        med_prog=statistics.median(r["time_s"] for r in prog)
        med_crit=statistics.median(r["time_s"] for r in crit)

        wp=paired_ratio(wait,prog,reps)
        wc=paired_ratio(wait,crit,reps)
        pc=paired_ratio(prog,crit,reps)
        seed=int(load*1000)+(0 if backend=="busywait" else 100000)
        ci_wp=bootstrap_median_ci(wp,seed+1)
        ci_wc=bootstrap_median_ci(wc,seed+2)
        ci_pc=bootstrap_median_ci(pc,seed+3)

        summary.append({
            "threads":threads,"T":T,"slow_thread":slow_thread,"load_pct":load,
            "backend":backend,"reps":reps,
            "median_wait_s":med_wait,"median_progress_s":med_prog,"median_critical_s":med_crit,
            "speedup_progress_ratio_of_medians":med_wait/med_prog,
            "speedup_critical_ratio_of_medians":med_wait/med_crit,
            "speedup_critical_vs_progress_ratio_of_medians":med_prog/med_crit,
            "critical_gain_vs_wait_pct_ratio_of_medians":100.0*(med_wait/med_crit-1.0),
            "median_paired_wait_over_progress":statistics.median(wp),
            "progress_wins_vs_wait":sum(x>1.0 for x in wp),
            "bootstrap95_wait_over_progress_lo":ci_wp[0],
            "bootstrap95_wait_over_progress_hi":ci_wp[1],
            "median_paired_wait_over_critical":statistics.median(wc),
            "critical_wins_vs_wait":sum(x>1.0 for x in wc),
            "bootstrap95_wait_over_critical_lo":ci_wc[0],
            "bootstrap95_wait_over_critical_hi":ci_wc[1],
            "median_paired_progress_over_critical":statistics.median(pc),
            "critical_wins_vs_progress":sum(x>1.0 for x in pc),
            "bootstrap95_progress_over_critical_lo":ci_pc[0],
            "bootstrap95_progress_over_critical_hi":ci_pc[1],
            "linf_wait":statistics.median(r["linf"] for r in wait),
            "linf_progress":statistics.median(r["linf"] for r in prog),
            "linf_critical":statistics.median(r["linf"] for r in crit),
        })

summary_path=os.path.join(outdir,"summary.csv")
with open(summary_path,"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)

if 0.0 not in load_levels:
    raise SystemExit("load=0 control is required")

load_effect=[]
for backend in ("busywait","semaphore"):
    for policy in ("wait","progress","critical"):
        ref=select(times,0.0,backend,policy)
        med_ref=statistics.median(r["time_s"] for r in ref)
        for load in load_levels:
            cur=select(times,load,backend,policy)
            med_cur=statistics.median(r["time_s"] for r in cur)
            ratios=paired_ratio(cur,ref,reps)
            seed=int(load*1000)+(0 if backend=="busywait" else 100000)+{"wait":10,"progress":20,"critical":30}[policy]
            ci=bootstrap_median_ci(ratios,seed)
            load_effect.append({
                "threads":threads,"T":T,"slow_thread":slow_thread,"load_pct":load,
                "backend":backend,"policy":policy,
                "median_time_s":med_cur,"median_zero_load_s":med_ref,
                "slowdown_ratio_of_medians":med_cur/med_ref,
                "slowdown_pct_ratio_of_medians":100.0*(med_cur/med_ref-1.0),
                "median_paired_slowdown_ratio":statistics.median(ratios),
                "bootstrap95_paired_slowdown_lo":ci[0],
                "bootstrap95_paired_slowdown_hi":ci[1],
            })

load_effect_path=os.path.join(outdir,"load_effect_summary.csv")
with open(load_effect_path,"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(load_effect[0].keys())); w.writeheader(); w.writerows(load_effect)

profile_summary=[]
for key in sorted(pgroups):
    load,backend,policy=key; rr=pgroups[key]
    med=lambda field: statistics.median(float(r[field]) for r in rr)
    profile_summary.append({
        "threads":threads,"T":T,"slow_thread":slow_thread,"load_pct":load,
        "backend":backend,"policy":policy,"reps":len(rr),
        "median_profile_time_s":med("time_s"),"median_read":med("read"),
        "median_recompute":med("recompute"),"median_predict":med("predict"),
        "median_wait":med("wait"),"median_lead_wait":med("lead_wait"),
        "median_lead_wait_ticks_sum":med("lead_wait_ticks_sum"),
        "median_progress_blocked":med("progress_blocked"),
        "median_penalty_samples":med("penalty_samples"),
        "median_penalty_nonzero_samples":med("penalty_nonzero_samples"),
        "median_penalty_mean_ticks":med("penalty_mean_ticks"),
        "median_penalty_nonzero_fraction":med("penalty_nonzero_fraction"),
        "median_critical_recompute":med("critical_recompute"),
        "median_noncritical_recompute_rejected":med("noncritical_recompute_rejected"),
        "median_criticality_forced_wait":med("criticality_forced_wait"),
        "median_linf":med("linf"),
    })

profile_summary_path=os.path.join(outdir,"profile_summary.csv")
with open(profile_summary_path,"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(profile_summary[0].keys())); w.writeheader(); w.writerows(profile_summary)

print()
print("=== HASWELL 32 CONTROLLED IMBALANCE ===")
print("load% backend    Scrit(med)  paired-Scrit  wins  bootstrap95")
for r in summary:
    print(
        f"{r['load_pct']:5.0f} {r['backend']:10s} "
        f"{r['speedup_critical_ratio_of_medians']:10.5f} "
        f"{r['median_paired_wait_over_critical']:13.5f} "
        f"{r['critical_wins_vs_wait']:2d}/{r['reps']:2d} "
        f"[{r['bootstrap95_wait_over_critical_lo']:.6f},"
        f"{r['bootstrap95_wait_over_critical_hi']:.6f}]"
    )

print()
print("Files:")
for p in (times_path,profile_path,summary_path,profile_summary_path,load_effect_path):
    print(" ",p)
PY

{
    echo "date=$(date --iso-8601=seconds 2>/dev/null || date)"
    echo "host=$(hostname)"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "machine=Haswell32"
    echo "experiment=controlled_synthetic_imbalance"
    echo "exploratory=yes"
    echo "threads=${THREADS}"
    echo "N=${N_VALUE}"
    echo "T=${T_VALUE}"
    echo "TILE=${TILE_VALUE}"
    echo "slow_thread=${SLOW_THREAD}"
    echo "load_levels=${LOAD_LEVELS[*]}"
    echo "REPS=${REPS}"
    echo "PROFILE_REPS=${PROFILE_REPS}"
    echo "WARMUPS=${WARMUPS}"
    echo "PREDICT=0"
    echo "HEAT2D_MAX_LEAD=2"
    echo "PROGRESS_LAMBDA=${PROGRESS_LAMBDA}"
    echo "PROGRESS_BOOTSTRAP=${PROGRESS_BOOTSTRAP}"
    echo "CALIBRATION_DIR=${CALIBRATION_DIR}"
    echo "calibration_reused=yes"
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
echo "Haswell 32 controlled-imbalance sweep complete."
echo "Results:"
echo "  ${OUTDIR}/times.csv"
echo "  ${OUTDIR}/profile_per_run.csv"
echo "  ${OUTDIR}/summary.csv"
echo "  ${OUTDIR}/profile_summary.csv"
echo "  ${OUTDIR}/load_effect_summary.csv"
echo "  ${OUTDIR}/order.csv"
echo "  ${OUTDIR}/run_metadata.txt"
echo "============================================================"
