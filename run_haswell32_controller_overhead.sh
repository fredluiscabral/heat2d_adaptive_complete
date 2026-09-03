#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Heat2D — Haswell 32: decomposicao do overhead do controlador adaptativo
#
# Objetivo:
#   medir separadamente:
#
#     O_adaptive =
#       T(adaptive_wait) / T(base) - 1
#
#     O_critical =
#       T(crit_infra) / T(adaptive_wait) - 1
#
# Casos por backend:
#
#   base
#       executavel original nao adaptativo
#
#   adaptive_wait
#       HEAT2D_ENABLE_PREDICT=0
#       HEAT2D_ENABLE_RECOMPUTE=0
#       HEAT2D_ENABLE_CRITICALITY=0
#
#   crit_infra
#       HEAT2D_ENABLE_PREDICT=0
#       HEAT2D_ENABLE_RECOMPUTE=0
#       HEAT2D_ENABLE_CRITICALITY=1
#
# Portanto, crit_infra mede o custo da infraestrutura de criticidade
# (blocked_on[] e atomicos associados) SEM permitir qualquer beneficio
# de RECOMPUTE.
#
# Configuracao principal:
#   p=32, N=8192, T=500, TILE=32
#
# Backends:
#   busywait
#   semaphore
#
# Desenho:
#   - 1 warm-up por caso;
#   - 20 repeticoes temporizadas por caso;
#   - ordem rotacionada entre os 6 casos;
#   - 3 repeticoes de profiling APENAS para adaptive_wait e crit_infra;
#   - analise pareada + IC bootstrap 95%.
#
# Calibracao:
#   Como RECOMPUTE permanece OFF em ambos os casos adaptativos, este
#   experimento nao precisa recalibrar C_P/C_R/C_W. Os arquivos ja
#   existentes sao apenas passados ao executavel por compatibilidade.
#
# Saida padrao:
#   results/haswell32_controller_overhead_T500/
#       times.csv
#       profile_per_run.csv
#       profile_summary.csv
#       summary.csv
#       order.csv
#       param.effective.txt
#       run_metadata.txt
# =============================================================================

THREADS="${THREADS:-32}"
N_VALUE="${N_VALUE:-8192}"
T_VALUE="${T_VALUE:-500}"
TILE_VALUE="${TILE_VALUE:-32}"

REPS="${REPS:-20}"
PROFILE_REPS="${PROFILE_REPS:-3}"
WARMUPS="${WARMUPS:-1}"

PROGRESS_LAMBDA="${PROGRESS_LAMBDA:-1.0}"
PROGRESS_BOOTSTRAP="${PROGRESS_BOOTSTRAP:-8}"

CALIBRATION_DIR="${CALIBRATION_DIR:-results/haswell32_T_selection/calibration}"
OUTDIR="${OUTDIR:-results/haswell32_controller_overhead_T${T_VALUE}}"

BW_BASE="./heat2d_explicit_omp_busywait_nobarrier_nofs"
SM_BASE="./heat2d_explicit_omp_sem_nobarrier_nofs"

BW_REF="./heat2d_adaptive_busywait"
SM_REF="./heat2d_adaptive_sem"

BW_PROF="./heat2d_adaptive_busywait_profile"
SM_PROF="./heat2d_adaptive_sem_profile"

COST="${CALIBRATION_DIR}/heat2d_cost_model.dat"
BW_WAIT="${CALIBRATION_DIR}/heat2d_wait_cost_busywait.dat"
SM_WAIT="${CALIBRATION_DIR}/heat2d_wait_cost_semaphore.dat"

export OMP_NUM_THREADS="${THREADS}"
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_DYNAMIC=FALSE
export HEAT2D_MAX_LEAD=2

command -v python3 >/dev/null || {
    echo "ERRO: python3 nao encontrado." >&2
    exit 1
}
command -v make >/dev/null || {
    echo "ERRO: make nao encontrado." >&2
    exit 1
}
[[ -f param.txt ]] || {
    echo "ERRO: param.txt nao encontrado." >&2
    exit 1
}

if [[ "${THREADS}" != "32" ]]; then
    echo "ERRO: este experimento foi definido para p=32; THREADS=${THREADS}." >&2
    exit 2
fi

if (( REPS < 2 )); then
    echo "ERRO: REPS deve ser >= 2 para analise pareada." >&2
    exit 2
fi

if (( PROFILE_REPS < 1 )); then
    echo "ERRO: PROFILE_REPS deve ser >= 1." >&2
    exit 2
fi

if (( WARMUPS < 1 )); then
    echo "ERRO: WARMUPS deve ser >= 1." >&2
    exit 2
fi

for f in "${COST}" "${BW_WAIT}" "${SM_WAIT}"; do
    [[ -s "${f}" ]] || {
        echo "ERRO: arquivo de calibracao existente nao encontrado: ${f}" >&2
        echo "Esperado em CALIBRATION_DIR=${CALIBRATION_DIR}" >&2
        exit 3
    }
done

mkdir -p "${OUTDIR}"

# ---------------------------------------------------------------------------
# Preserva param.txt e fixa a configuracao efetiva.
# ---------------------------------------------------------------------------
PARAM_BACKUP="$(mktemp ./param.txt.haswell_overhead.XXXXXX)"
cp param.txt "${PARAM_BACKUP}"

restore_param() {
    if [[ -f "${PARAM_BACKUP}" ]]; then
        cp "${PARAM_BACKUP}" param.txt
        rm -f "${PARAM_BACKUP}"
    fi
}
trap restore_param EXIT INT TERM

set_param() {
    python3 - "${N_VALUE}" "${T_VALUE}" "${TILE_VALUE}" <<'PY'
import re
import sys

path = 'param.txt'
values = {
    'N': sys.argv[1],
    'T': sys.argv[2],
    'TILE': sys.argv[3],
    'WRITE_OUTPUT': '0',
}

with open(path, encoding='utf-8') as f:
    lines = f.readlines()

out = []
seen = set()

for line in lines:
    raw = line.rstrip('\n')
    m = re.match(
        r'^(\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*=\s*)(.*?)(\s*)$',
        raw
    )
    if m and m.group(2) in values:
        key = m.group(2)
        out.append(
            f'{m.group(1)}{key}{m.group(3)}{values[key]}\n'
        )
        seen.add(key)
    else:
        out.append(line)

for key, value in values.items():
    if key not in seen:
        out.append(f'{key} = {value}\n')

with open(path, 'w', encoding='utf-8') as f:
    f.writelines(out)
PY
}

set_param
cp param.txt "${OUTDIR}/param.effective.txt"

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
MAKE_JOBS="${MAKE_JOBS:-$(nproc 2>/dev/null || echo 4)}"

echo "============================================================"
echo "HASWELL 32 — CONTROLLER OVERHEAD DECOMPOSITION"
echo "============================================================"
echo "Threads             : ${THREADS}"
echo "N/T/TILE            : ${N_VALUE}/${T_VALUE}/${TILE_VALUE}"
echo "Timed repetitions   : ${REPS}"
echo "Profile repetitions : ${PROFILE_REPS}"
echo "Warm-ups            : ${WARMUPS}"
echo "OMP_PLACES          : ${OMP_PLACES}"
echo "OMP_PROC_BIND       : ${OMP_PROC_BIND}"
echo "Calibration dir     : ${CALIBRATION_DIR}"
echo "Output              : ${OUTDIR}"
echo "============================================================"

echo
echo "[build] core + profiles + native baselines"
make -j "${MAKE_JOBS}" core profiles
make -j "${MAKE_JOBS}" \
    heat2d_explicit_omp_busywait_nobarrier_nofs \
    heat2d_explicit_omp_sem_nobarrier_nofs

for exe in \
    "${BW_BASE}" "${SM_BASE}" \
    "${BW_REF}" "${SM_REF}" \
    "${BW_PROF}" "${SM_PROF}"
do
    [[ -x "${exe}" ]] || {
        echo "ERRO: executavel ausente: ${exe}" >&2
        exit 4
    }
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
check_common_output() {
    local log="$1"

    grep -q '^Tempo[[:space:]]*:' "${log}" || {
        echo "ERRO: Tempo ausente em ${log}" >&2
        exit 5
    }

    grep -q '^Linf:' "${log}" || {
        echo "ERRO: Linf ausente em ${log}" >&2
        exit 5
    }
}

check_adaptive_flags() {
    local log="$1"
    local criticality="$2"

    grep -q '^Adaptive predict: off$' "${log}" || {
        echo "ERRO: PREDICT deveria estar OFF em ${log}" >&2
        exit 6
    }

    grep -q '^Adaptive recompute: off$' "${log}" || {
        echo "ERRO: RECOMPUTE deveria estar OFF em ${log}" >&2
        exit 6
    }

    if [[ "${criticality}" == "1" ]]; then
        grep -q '^Adaptive criticality-aware: on$' "${log}" || {
            echo "ERRO: CRITICALITY deveria estar ON em ${log}" >&2
            exit 6
        }
    else
        grep -q '^Adaptive criticality-aware: off$' "${log}" || {
            echo "ERRO: CRITICALITY deveria estar OFF em ${log}" >&2
            exit 6
        }
    fi
}

run_base() {
    local backend="$1"
    local log="$2"
    local exe

    case "${backend}" in
        busywait)  exe="${BW_BASE}" ;;
        semaphore) exe="${SM_BASE}" ;;
        *)
            echo "ERRO: backend invalido: ${backend}" >&2
            exit 7
            ;;
    esac

    "${exe}" > "${log}" 2>&1
    check_common_output "${log}"
}

run_adaptive() {
    local backend="$1"
    local policy="$2"
    local exe="$3"
    local log="$4"

    local criticality waitcost

    case "${policy}" in
        adaptive_wait)
            criticality=0
            ;;
        crit_infra)
            criticality=1
            ;;
        *)
            echo "ERRO: policy adaptativa invalida: ${policy}" >&2
            exit 7
            ;;
    esac

    case "${backend}" in
        busywait)
            waitcost="${BW_WAIT}"
            ;;
        semaphore)
            waitcost="${SM_WAIT}"
            ;;
        *)
            echo "ERRO: backend invalido: ${backend}" >&2
            exit 7
            ;;
    esac

    HEAT2D_ENABLE_PREDICT=0 \
    HEAT2D_ENABLE_RECOMPUTE=0 \
    HEAT2D_ENABLE_CRITICALITY="${criticality}" \
    HEAT2D_COST_FILE="${COST}" \
    HEAT2D_WAIT_COST_FILE="${waitcost}" \
    HEAT2D_PROGRESS_LAMBDA="${PROGRESS_LAMBDA}" \
    HEAT2D_PROGRESS_BOOTSTRAP_SAMPLES="${PROGRESS_BOOTSTRAP}" \
        "${exe}" > "${log}" 2>&1

    check_common_output "${log}"
    check_adaptive_flags "${log}" "${criticality}"
}

run_timing_case() {
    local backend="$1"
    local policy="$2"
    local tag="$3"
    local log="${OUTDIR}/${backend}_${policy}_${tag}.log"

    case "${policy}" in
        base)
            run_base "${backend}" "${log}"
            ;;
        adaptive_wait|crit_infra)
            if [[ "${backend}" == "busywait" ]]; then
                exe="${BW_REF}"
            else
                exe="${SM_REF}"
            fi
            run_adaptive "${backend}" "${policy}" "${exe}" "${log}"
            ;;
        *)
            echo "ERRO: timing policy invalida: ${policy}" >&2
            exit 8
            ;;
    esac
}

run_profile_case() {
    local backend="$1"
    local policy="$2"
    local tag="$3"
    local log="${OUTDIR}/${backend}_${policy}_${tag}.log"
    local exe

    case "${backend}" in
        busywait)  exe="${BW_PROF}" ;;
        semaphore) exe="${SM_PROF}" ;;
        *)
            echo "ERRO: backend invalido: ${backend}" >&2
            exit 8
            ;;
    esac

    run_adaptive "${backend}" "${policy}" "${exe}" "${log}"
}

# Intercala os backends e os tres niveis de infraestrutura.
TIMING_CASES=(
    "busywait:base"
    "semaphore:base"
    "busywait:adaptive_wait"
    "semaphore:adaptive_wait"
    "busywait:crit_infra"
    "semaphore:crit_infra"
)

PROFILE_CASES=(
    "busywait:adaptive_wait"
    "semaphore:adaptive_wait"
    "busywait:crit_infra"
    "semaphore:crit_infra"
)

rm -f \
    "${OUTDIR}"/*_warmup_r*.log \
    "${OUTDIR}"/*_ref_r*.log \
    "${OUTDIR}"/*_prof_r*.log \
    "${OUTDIR}/times.csv" \
    "${OUTDIR}/profile_per_run.csv" \
    "${OUTDIR}/profile_summary.csv" \
    "${OUTDIR}/summary.csv" \
    "${OUTDIR}/order.csv" \
    "${OUTDIR}/run_metadata.txt" 2>/dev/null || true

echo "phase,rep,position,backend,policy" > "${OUTDIR}/order.csv"

# ---------------------------------------------------------------------------
# Warm-up
# ---------------------------------------------------------------------------
echo
echo "[warm-up] ${WARMUPS} x 6 cases"

for ((w=1; w<=WARMUPS; ++w)); do
    shift=$(( (w - 1) % ${#TIMING_CASES[@]} ))

    for ((k=0; k<${#TIMING_CASES[@]}; ++k)); do
        idx=$(( (k + shift) % ${#TIMING_CASES[@]} ))
        IFS=: read -r backend policy <<< "${TIMING_CASES[$idx]}"

        run_timing_case \
            "${backend}" "${policy}" \
            "warmup_r$(printf '%02d' "${w}")"
    done
done

# ---------------------------------------------------------------------------
# Clean timing
# ---------------------------------------------------------------------------
echo
echo "[timing] ${REPS} rotated blocks"

for ((rep=1; rep<=REPS; ++rep)); do
    shift=$(( (rep - 1) % ${#TIMING_CASES[@]} ))
    position=0

    for ((k=0; k<${#TIMING_CASES[@]}; ++k)); do
        idx=$(( (k + shift) % ${#TIMING_CASES[@]} ))
        IFS=: read -r backend policy <<< "${TIMING_CASES[$idx]}"

        position=$((position + 1))
        echo "timing,${rep},${position},${backend},${policy}" \
            >> "${OUTDIR}/order.csv"

        run_timing_case \
            "${backend}" "${policy}" \
            "ref_r$(printf '%02d' "${rep}")"
    done

    echo "timing rep=${rep}/${REPS} complete"
done

# ---------------------------------------------------------------------------
# Profiling: only adaptive_wait and crit_infra
# ---------------------------------------------------------------------------
echo
echo "[profiling] ${PROFILE_REPS} rotated blocks"

for ((rep=1; rep<=PROFILE_REPS; ++rep)); do
    shift=$(( (rep - 1) % ${#PROFILE_CASES[@]} ))
    position=0

    for ((k=0; k<${#PROFILE_CASES[@]}; ++k)); do
        idx=$(( (k + shift) % ${#PROFILE_CASES[@]} ))
        IFS=: read -r backend policy <<< "${PROFILE_CASES[$idx]}"

        position=$((position + 1))
        echo "profile,${rep},${position},${backend},${policy}" \
            >> "${OUTDIR}/order.csv"

        run_profile_case \
            "${backend}" "${policy}" \
            "prof_r$(printf '%02d' "${rep}")"
    done

    echo "profile rep=${rep}/${PROFILE_REPS} complete"
done

# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------
python3 - "${OUTDIR}" "${REPS}" "${PROFILE_REPS}" "${THREADS}" "${T_VALUE}" <<'PY'
import csv
import glob
import os
import random
import re
import statistics
import sys

outdir = sys.argv[1]
reps = int(sys.argv[2])
profile_reps = int(sys.argv[3])
threads = int(sys.argv[4])
T = int(sys.argv[5])

TIME_RE = re.compile(r'^Tempo\s*:\s*([-+0-9.eE]+)')
LINF_RE = re.compile(r'^Linf:\s*([-+0-9.eE]+)')

PROFILE_FIELDS = {
    'Adaptive actions RECOMPUTE': 'recompute',
    'Adaptive actions WAIT': 'wait',
    'Adaptive lead-guard waits': 'lead_wait',
    'Progress-blocked RECOMPUTE': 'progress_blocked',
    'Progress penalty samples': 'penalty_samples',
    'Progress penalty nonzero samples': 'penalty_nonzero_samples',
    'Progress penalty observed mean': 'penalty_mean_ticks',
    'Progress penalty nonzero fraction': 'penalty_nonzero_fraction',
    'Critical RECOMPUTE': 'critical_recompute',
    'Noncritical RECOMPUTE rejected': 'noncritical_recompute_rejected',
    'Criticality-forced WAIT': 'criticality_forced_wait',
}

def parse_log(path, profile=False):
    x = {}

    if profile:
        for key in PROFILE_FIELDS.values():
            x[key] = 0.0

    with open(path, errors='replace') as f:
        for line in f:
            m = TIME_RE.match(line)
            if m:
                x['time_s'] = float(m.group(1))

            m = LINF_RE.match(line)
            if m:
                x['linf'] = float(m.group(1))

            if profile:
                for prefix, key in PROFILE_FIELDS.items():
                    if line.startswith(prefix + ':'):
                        tok = line.split(':', 1)[1].strip().split()[0]
                        x[key] = float(tok)
                        break

    if 'time_s' not in x or 'linf' not in x:
        raise RuntimeError(f'incomplete log: {path}')

    return x

def percentile(sorted_values, p):
    if len(sorted_values) == 1:
        return sorted_values[0]

    k = (len(sorted_values) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)

    if f == c:
        return sorted_values[f]

    return (
        sorted_values[f] * (c - k)
        + sorted_values[c] * (k - f)
    )

def bootstrap_median_ci(values, seed, nboot=20000):
    rng = random.Random(seed)
    n = len(values)
    boots = []

    for _ in range(nboot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boots.append(statistics.median(sample))

    boots.sort()
    return percentile(boots, 0.025), percentile(boots, 0.975)

def rows_for(rows, backend, policy):
    return [
        r for r in rows
        if r['backend'] == backend and r['policy'] == policy
    ]

def paired_ratio(num_rows, den_rows, expected):
    num = {int(r['rep']): float(r['time_s']) for r in num_rows}
    den = {int(r['rep']): float(r['time_s']) for r in den_rows}
    common = sorted(set(num) & set(den))

    if len(common) != expected:
        raise RuntimeError(
            f'paired comparison incomplete: {len(common)}/{expected}'
        )

    return [num[i] / den[i] for i in common]

# -------------------------------------------------------------------------
# Timing rows
# -------------------------------------------------------------------------
times = []

for path in glob.glob(os.path.join(outdir, '*_ref_r*.log')):
    m = re.fullmatch(
        r'(busywait|semaphore)_(base|adaptive_wait|crit_infra)_ref_r(\d+)\.log',
        os.path.basename(path)
    )
    if not m:
        continue

    backend, policy, rep = m.group(1), m.group(2), int(m.group(3))
    times.append({
        'threads': threads,
        'T': T,
        'backend': backend,
        'policy': policy,
        'rep': rep,
        **parse_log(path, profile=False),
    })

expected_timing_groups = 2 * 3
groups = {}

for r in times:
    groups.setdefault((r['backend'], r['policy']), []).append(r)

if len(groups) != expected_timing_groups:
    raise SystemExit(
        f'Expected {expected_timing_groups} timing groups, found {len(groups)}'
    )

for key, rr in groups.items():
    if len(rr) != reps:
        raise SystemExit(
            f'Incomplete timing group {key}: {len(rr)}/{reps}'
        )

times_path = os.path.join(outdir, 'times.csv')
with open(times_path, 'w', newline='') as f:
    cols = [
        'threads', 'T', 'backend', 'policy',
        'rep', 'time_s', 'linf'
    ]
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(sorted(
        times,
        key=lambda r: (r['backend'], r['policy'], r['rep'])
    ))

# -------------------------------------------------------------------------
# Profile rows
# -------------------------------------------------------------------------
prof = []

for path in glob.glob(os.path.join(outdir, '*_prof_r*.log')):
    m = re.fullmatch(
        r'(busywait|semaphore)_(adaptive_wait|crit_infra)_prof_r(\d+)\.log',
        os.path.basename(path)
    )
    if not m:
        continue

    backend, policy, rep = m.group(1), m.group(2), int(m.group(3))

    prof.append({
        'threads': threads,
        'T': T,
        'backend': backend,
        'policy': policy,
        'rep': rep,
        **parse_log(path, profile=True),
    })

profile_groups = {}
for r in prof:
    profile_groups.setdefault((r['backend'], r['policy']), []).append(r)

expected_profile_groups = 2 * 2
if len(profile_groups) != expected_profile_groups:
    raise SystemExit(
        f'Expected {expected_profile_groups} profile groups, '
        f'found {len(profile_groups)}'
    )

for key, rr in profile_groups.items():
    if len(rr) != profile_reps:
        raise SystemExit(
            f'Incomplete profile group {key}: {len(rr)}/{profile_reps}'
        )

profile_cols = [
    'threads', 'T', 'backend', 'policy', 'rep',
    'time_s', 'linf',
    'recompute', 'wait', 'lead_wait', 'progress_blocked',
    'penalty_samples', 'penalty_nonzero_samples',
    'penalty_mean_ticks', 'penalty_nonzero_fraction',
    'critical_recompute', 'noncritical_recompute_rejected',
    'criticality_forced_wait',
]

profile_per_run_path = os.path.join(outdir, 'profile_per_run.csv')
with open(profile_per_run_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=profile_cols)
    w.writeheader()
    w.writerows(sorted(
        prof,
        key=lambda r: (r['backend'], r['policy'], r['rep'])
    ))

profile_summary = []

for key in sorted(profile_groups):
    rr = profile_groups[key]
    backend, policy = key
    med = lambda field: statistics.median(
        float(r[field]) for r in rr
    )

    profile_summary.append({
        'threads': threads,
        'T': T,
        'backend': backend,
        'policy': policy,
        'reps': len(rr),
        'median_profile_time_s': med('time_s'),
        'median_recompute': med('recompute'),
        'median_wait': med('wait'),
        'median_lead_wait': med('lead_wait'),
        'median_progress_blocked': med('progress_blocked'),
        'median_penalty_samples': med('penalty_samples'),
        'median_penalty_nonzero_samples':
            med('penalty_nonzero_samples'),
        'median_penalty_mean_ticks': med('penalty_mean_ticks'),
        'median_penalty_nonzero_fraction':
            med('penalty_nonzero_fraction'),
        'median_critical_recompute': med('critical_recompute'),
        'median_noncritical_recompute_rejected':
            med('noncritical_recompute_rejected'),
        'median_criticality_forced_wait':
            med('criticality_forced_wait'),
        'median_linf': med('linf'),
    })

profile_summary_path = os.path.join(outdir, 'profile_summary.csv')
with open(profile_summary_path, 'w', newline='') as f:
    w = csv.DictWriter(
        f,
        fieldnames=list(profile_summary[0].keys())
    )
    w.writeheader()
    w.writerows(profile_summary)

# -------------------------------------------------------------------------
# Overhead summary
# -------------------------------------------------------------------------
summary = []

for backend in ('busywait', 'semaphore'):
    base = rows_for(times, backend, 'base')
    aw = rows_for(times, backend, 'adaptive_wait')
    ci = rows_for(times, backend, 'crit_infra')

    if len(base) != reps or len(aw) != reps or len(ci) != reps:
        raise RuntimeError(f'incomplete timing rows for {backend}')

    med_base = statistics.median(r['time_s'] for r in base)
    med_aw = statistics.median(r['time_s'] for r in aw)
    med_ci = statistics.median(r['time_s'] for r in ci)

    # Paired ratios:
    #   adaptive/base > 1 => adaptive infrastructure costs time.
    #   crit/adaptive > 1 => criticality infrastructure costs time.
    aw_over_base = paired_ratio(aw, base, reps)
    ci_over_aw = paired_ratio(ci, aw, reps)
    ci_over_base = paired_ratio(ci, base, reps)

    seed = 100 if backend == 'busywait' else 200
    ci_aw_base = bootstrap_median_ci(aw_over_base, seed + 1)
    ci_ci_aw = bootstrap_median_ci(ci_over_aw, seed + 2)
    ci_ci_base = bootstrap_median_ci(ci_over_base, seed + 3)

    summary.append({
        'threads': threads,
        'T': T,
        'backend': backend,
        'reps': reps,

        'median_base_s': med_base,
        'median_adaptive_wait_s': med_aw,
        'median_crit_infra_s': med_ci,

        'adaptive_overhead_pct_ratio_of_medians':
            100.0 * (med_aw / med_base - 1.0),

        'criticality_overhead_pct_ratio_of_medians':
            100.0 * (med_ci / med_aw - 1.0),

        'total_crit_infra_overhead_pct_ratio_of_medians':
            100.0 * (med_ci / med_base - 1.0),

        'median_paired_adaptive_over_base':
            statistics.median(aw_over_base),

        'median_paired_adaptive_overhead_pct':
            100.0 * (statistics.median(aw_over_base) - 1.0),

        'adaptive_slower_than_base_pairs':
            sum(1 for x in aw_over_base if x > 1.0),

        'bootstrap95_adaptive_over_base_lo':
            ci_aw_base[0],

        'bootstrap95_adaptive_over_base_hi':
            ci_aw_base[1],

        'median_paired_crit_infra_over_adaptive':
            statistics.median(ci_over_aw),

        'median_paired_criticality_overhead_pct':
            100.0 * (statistics.median(ci_over_aw) - 1.0),

        'crit_infra_slower_than_adaptive_pairs':
            sum(1 for x in ci_over_aw if x > 1.0),

        'bootstrap95_crit_infra_over_adaptive_lo':
            ci_ci_aw[0],

        'bootstrap95_crit_infra_over_adaptive_hi':
            ci_ci_aw[1],

        'median_paired_crit_infra_over_base':
            statistics.median(ci_over_base),

        'median_paired_total_overhead_pct':
            100.0 * (statistics.median(ci_over_base) - 1.0),

        'crit_infra_slower_than_base_pairs':
            sum(1 for x in ci_over_base if x > 1.0),

        'bootstrap95_crit_infra_over_base_lo':
            ci_ci_base[0],

        'bootstrap95_crit_infra_over_base_hi':
            ci_ci_base[1],

        'linf_base':
            statistics.median(r['linf'] for r in base),

        'linf_adaptive_wait':
            statistics.median(r['linf'] for r in aw),

        'linf_crit_infra':
            statistics.median(r['linf'] for r in ci),
    })

summary_path = os.path.join(outdir, 'summary.csv')
with open(summary_path, 'w', newline='') as f:
    w = csv.DictWriter(
        f,
        fieldnames=list(summary[0].keys())
    )
    w.writeheader()
    w.writerows(summary)

print()
print('=== HASWELL 32 — OVERHEAD ===')

for r in summary:
    print(
        f"{r['backend']:9s} "
        f"O_adaptive_med={r['adaptive_overhead_pct_ratio_of_medians']:+.4f}% "
        f"O_adaptive_paired={r['median_paired_adaptive_overhead_pct']:+.4f}% "
        f"pairs_slower={r['adaptive_slower_than_base_pairs']}/{r['reps']} "
        f"CIratio=[{r['bootstrap95_adaptive_over_base_lo']:.6f},"
        f"{r['bootstrap95_adaptive_over_base_hi']:.6f}]"
    )

    print(
        f"{'':9s} "
        f"O_critical_med={r['criticality_overhead_pct_ratio_of_medians']:+.4f}% "
        f"O_critical_paired={r['median_paired_criticality_overhead_pct']:+.4f}% "
        f"pairs_slower={r['crit_infra_slower_than_adaptive_pairs']}/{r['reps']} "
        f"CIratio=[{r['bootstrap95_crit_infra_over_adaptive_lo']:.6f},"
        f"{r['bootstrap95_crit_infra_over_adaptive_hi']:.6f}]"
    )

print()
print('Files:')
print(' ', times_path)
print(' ', profile_per_run_path)
print(' ', profile_summary_path)
print(' ', summary_path)
PY

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
{
    echo "date=$(date --iso-8601=seconds 2>/dev/null || date)"
    echo "host=$(hostname)"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "machine=Haswell32"
    echo "threads=${THREADS}"
    echo "N=${N_VALUE}"
    echo "T=${T_VALUE}"
    echo "TILE=${TILE_VALUE}"
    echo "REPS=${REPS}"
    echo "PROFILE_REPS=${PROFILE_REPS}"
    echo "WARMUPS=${WARMUPS}"
    echo "CALIBRATION_DIR=${CALIBRATION_DIR}"
    echo "calibration_reused=yes"
    echo "recalibration_performed=no"
    echo "PREDICT=0"
    echo "RECOMPUTE=0"
    echo "OMP_PLACES=${OMP_PLACES}"
    echo "OMP_PROC_BIND=${OMP_PROC_BIND}"
    echo "PROGRESS_LAMBDA=${PROGRESS_LAMBDA}"
    echo "PROGRESS_BOOTSTRAP=${PROGRESS_BOOTSTRAP}"
} > "${OUTDIR}/run_metadata.txt"

echo
echo "============================================================"
echo "Haswell 32 controller-overhead experiment complete."
echo "Results:"
echo "  ${OUTDIR}/times.csv"
echo "  ${OUTDIR}/profile_per_run.csv"
echo "  ${OUTDIR}/profile_summary.csv"
echo "  ${OUTDIR}/summary.csv"
echo "  ${OUTDIR}/order.csv"
echo "  ${OUTDIR}/param.effective.txt"
echo "  ${OUTDIR}/run_metadata.txt"
echo "============================================================"
