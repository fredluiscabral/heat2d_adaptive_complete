#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Heat2D adaptive policies - Haswell 32 baseline
# No SLURM required.
#
# Default study:
#   N=8192, T=1000, TILE=32
#   threads = 8,16,24,32
#   policies = WAIT, PROGRESS, CRITICAL
#   backends = busy-wait, semaphore
#   1 warm-up per case
#   10 clean timing runs per case
#   3 profiling runs per case
#
# Override examples:
#   REPS=5 ./run_haswell32_baseline.sh
#   THREADS_LIST="16 32" REPS=10 ./run_haswell32_baseline.sh
# ============================================================

REPS="${REPS:-10}"
PROFILE_REPS="${PROFILE_REPS:-3}"
THREADS_LIST_STR="${THREADS_LIST:-8 16 24 32}"
read -r -a THREADS_LIST <<< "${THREADS_LIST_STR}"

N_VALUE="${N_VALUE:-8192}"
T_VALUE="${T_VALUE:-1000}"
TILE_VALUE="${TILE_VALUE:-32}"

CAL_SAMPLES="${CAL_SAMPLES:-32}"
WAIT_CAL_SAMPLES="${WAIT_CAL_SAMPLES:-32}"
PROGRESS_LAMBDA="${PROGRESS_LAMBDA:-1.0}"
PROGRESS_BOOTSTRAP="${PROGRESS_BOOTSTRAP:-8}"
OUTDIR="${OUTDIR:-results/haswell32_baseline}"

BW_REF="./heat2d_adaptive_busywait"
SM_REF="./heat2d_adaptive_sem"
BW_PROF="./heat2d_adaptive_busywait_profile"
SM_PROF="./heat2d_adaptive_sem_profile"
PR_CAL="./heat2d_dependency_calibrate"
BW_WCAL="./heat2d_wait_calibrate_busywait"
SM_WCAL="./heat2d_wait_calibrate_semaphore"

export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_DYNAMIC=FALSE
export HEAT2D_MAX_LEAD=2

command -v python3 >/dev/null || { echo "ERRO: python3 nao encontrado." >&2; exit 1; }
command -v make >/dev/null || { echo "ERRO: make nao encontrado." >&2; exit 1; }
[[ -f param.txt ]] || { echo "ERRO: param.txt nao encontrado no diretorio atual." >&2; exit 1; }

mkdir -p "${OUTDIR}"

# Record machine/software context before the run.
{
    echo "date=$(date --iso-8601=seconds 2>/dev/null || date)"
    echo "host=$(hostname)"
    echo "pwd=$(pwd)"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "compiler=$(g++ --version 2>/dev/null | head -n1 || echo unknown)"
    echo "REPS=${REPS}"
    echo "PROFILE_REPS=${PROFILE_REPS}"
    echo "THREADS_LIST=${THREADS_LIST_STR}"
    echo "N=${N_VALUE}"
    echo "T=${T_VALUE}"
    echo "TILE=${TILE_VALUE}"
    echo "CAL_SAMPLES=${CAL_SAMPLES}"
    echo "WAIT_CAL_SAMPLES=${WAIT_CAL_SAMPLES}"
    echo "PROGRESS_LAMBDA=${PROGRESS_LAMBDA}"
    echo "PROGRESS_BOOTSTRAP=${PROGRESS_BOOTSTRAP}"
    echo "OMP_PLACES=${OMP_PLACES}"
    echo "OMP_PROC_BIND=${OMP_PROC_BIND}"
} > "${OUTDIR}/run_metadata.txt"

lscpu > "${OUTDIR}/lscpu.txt" 2>/dev/null || true

# Make the experiment self-contained without permanently changing param.txt.
PARAM_BACKUP="$(mktemp ./param.txt.haswell32.XXXXXX)"
cp param.txt "${PARAM_BACKUP}"
restore_param() {
    if [[ -f "${PARAM_BACKUP}" ]]; then
        cp "${PARAM_BACKUP}" param.txt
        rm -f "${PARAM_BACKUP}"
    fi
}
trap restore_param EXIT INT TERM

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

with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

seen = set()
out = []
for line in lines:
    m = re.match(r'^(\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*=\s*)(.*?)(\s*)$', line.rstrip('\n'))
    if m and m.group(2) in values:
        key = m.group(2)
        out.append(f'{m.group(1)}{key}{m.group(3)}{values[key]}\n')
        seen.add(key)
    else:
        out.append(line)

for key, value in values.items():
    if key not in seen:
        out.append(f'{key} = {value}\n')

with open(path, 'w', encoding='utf-8') as f:
    f.writelines(out)
PY

cp param.txt "${OUTDIR}/param.effective.txt"

# Build the active binaries and profiling variants.
MAKE_JOBS="${MAKE_JOBS:-$(nproc 2>/dev/null || echo 4)}"
echo "[build] make -j ${MAKE_JOBS} core profiles"
make -j "${MAKE_JOBS}" core profiles

for exe in "${BW_REF}" "${SM_REF}" "${BW_PROF}" "${SM_PROF}" "${PR_CAL}" "${BW_WCAL}" "${SM_WCAL}"; do
    [[ -x "${exe}" ]] || { echo "ERRO: executavel ausente: ${exe}" >&2; exit 2; }
done

run_solver() {
    local exe="$1"
    local backend="$2"
    local policy="$3"
    local rep="$4"
    local pdir="$5"
    local cost="$6"
    local waitcost="$7"
    local kind="$8"

    local recompute criticality
    case "${policy}" in
        wait)
            recompute=0
            criticality=0
            ;;
        progress)
            recompute=1
            criticality=0
            ;;
        critical)
            recompute=1
            criticality=1
            ;;
        *)
            echo "ERRO interno: policy invalida: ${policy}" >&2
            exit 3
            ;;
    esac

    local log="${pdir}/${backend}_${policy}_${kind}_r$(printf '%02d' "${rep}").log"

    HEAT2D_ENABLE_PREDICT=0 \
    HEAT2D_ENABLE_RECOMPUTE="${recompute}" \
    HEAT2D_ENABLE_CRITICALITY="${criticality}" \
    HEAT2D_COST_FILE="${cost}" \
    HEAT2D_WAIT_COST_FILE="${waitcost}" \
    HEAT2D_PROGRESS_LAMBDA="${PROGRESS_LAMBDA}" \
    HEAT2D_PROGRESS_BOOTSTRAP_SAMPLES="${PROGRESS_BOOTSTRAP}" \
        "${exe}" > "${log}" 2>&1

    grep -q '^Adaptive progress-aware: on$' "${log}" || {
        echo "ERRO: controlador progress-aware ausente em ${log}" >&2
        exit 4
    }

    if [[ "${criticality}" == "1" ]]; then
        grep -q '^Adaptive criticality-aware: on$' "${log}" || {
            echo "ERRO: criticality deveria estar ON em ${log}" >&2
            exit 5
        }
    else
        grep -q '^Adaptive criticality-aware: off$' "${log}" || {
            echo "ERRO: criticality deveria estar OFF em ${log}" >&2
            exit 6
        }
    fi
}

run_case() {
    local backend="$1"
    local policy="$2"
    local rep="$3"
    local kind="$4"
    local pdir="$5"
    local cost="$6"
    local bw_wait="$7"
    local sm_wait="$8"
    local exe waitcost

    if [[ "${backend}" == "busywait" ]]; then
        if [[ "${kind}" == "ref" ]]; then exe="${BW_REF}"; else exe="${BW_PROF}"; fi
        waitcost="${bw_wait}"
    else
        if [[ "${kind}" == "ref" ]]; then exe="${SM_REF}"; else exe="${SM_PROF}"; fi
        waitcost="${sm_wait}"
    fi

    run_solver "${exe}" "${backend}" "${policy}" "${rep}" "${pdir}" "${cost}" "${waitcost}" "${kind}"
}

for threads in "${THREADS_LIST[@]}"; do
    export OMP_NUM_THREADS="${threads}"
    tag="$(printf '%03d' "${threads}")"
    pdir="${OUTDIR}/threads_${tag}"
    mkdir -p "${pdir}"
    rm -f "${pdir}"/*.log "${pdir}"/*.dat 2>/dev/null || true

    cost="${pdir}/heat2d_cost_model.dat"
    bw_wait="${pdir}/heat2d_wait_cost_busywait.dat"
    sm_wait="${pdir}/heat2d_wait_cost_semaphore.dat"

    echo
    echo "============================================================"
    echo "THREADS = ${threads}"
    echo "============================================================"

    echo "[calibration] PREDICT/RECOMPUTE"
    HEAT2D_ENABLE_PREDICT=1 \
    HEAT2D_ENABLE_RECOMPUTE=1 \
    HEAT2D_CALIBRATION_SAMPLES="${CAL_SAMPLES}" \
    HEAT2D_COST_FILE="${cost}" \
        "${PR_CAL}" > "${pdir}/calibration_pr.log" 2>&1

    echo "[calibration] WAIT busy-wait"
    HEAT2D_WAIT_CALIBRATION_SAMPLES="${WAIT_CAL_SAMPLES}" \
    HEAT2D_WAIT_COST_FILE="${bw_wait}" \
        "${BW_WCAL}" > "${pdir}/calibration_wait_busywait.log" 2>&1

    echo "[calibration] WAIT semaphore"
    HEAT2D_WAIT_CALIBRATION_SAMPLES="${WAIT_CAL_SAMPLES}" \
    HEAT2D_WAIT_COST_FILE="${sm_wait}" \
        "${SM_WCAL}" > "${pdir}/calibration_wait_semaphore.log" 2>&1

    echo "[warm-up] 6 cases"
    for backend in busywait semaphore; do
        for policy in wait progress critical; do
            run_case "${backend}" "${policy}" 0 warmup "${pdir}" "${cost}" "${bw_wait}" "${sm_wait}"
        done
    done

    echo "[timing] ${REPS} clean repetitions per case"
    for ((rep=1; rep<=REPS; ++rep)); do
        # Alternate order to reduce systematic time-order bias.
        if (( rep % 2 == 1 )); then
            cases=(
                "busywait wait" "busywait progress" "busywait critical"
                "semaphore wait" "semaphore progress" "semaphore critical"
            )
        else
            cases=(
                "semaphore critical" "semaphore progress" "semaphore wait"
                "busywait critical" "busywait progress" "busywait wait"
            )
        fi

        for c in "${cases[@]}"; do
            read -r backend policy <<< "${c}"
            run_case "${backend}" "${policy}" "${rep}" ref "${pdir}" "${cost}" "${bw_wait}" "${sm_wait}"
        done
        echo "threads=${threads} timing rep=${rep}/${REPS} complete"
    done

    echo "[profiling] ${PROFILE_REPS} repetitions per case"
    for ((rep=1; rep<=PROFILE_REPS; ++rep)); do
        if (( rep % 2 == 1 )); then
            cases=(
                "busywait wait" "busywait progress" "busywait critical"
                "semaphore wait" "semaphore progress" "semaphore critical"
            )
        else
            cases=(
                "semaphore critical" "semaphore progress" "semaphore wait"
                "busywait critical" "busywait progress" "busywait wait"
            )
        fi
        for c in "${cases[@]}"; do
            read -r backend policy <<< "${c}"
            run_case "${backend}" "${policy}" "${rep}" prof "${pdir}" "${cost}" "${bw_wait}" "${sm_wait}"
        done
        echo "threads=${threads} profile rep=${rep}/${PROFILE_REPS} complete"
    done
done

python3 - "${OUTDIR}" "${REPS}" "${PROFILE_REPS}" <<'PY'
import csv
import glob
import os
import re
import statistics
import sys

outdir = sys.argv[1]
reps = int(sys.argv[2])
profile_reps = int(sys.argv[3])

# -------------------- clean timings --------------------
times = []
for path in sorted(glob.glob(os.path.join(outdir, 'threads_*', '*_ref_r*.log'))):
    b = os.path.basename(path)
    m = re.fullmatch(r'(busywait|semaphore)_(wait|progress|critical)_ref_r(\d+)\.log', b)
    if not m:
        continue
    backend, policy, rep = m.group(1), m.group(2), int(m.group(3))
    threads = int(os.path.basename(os.path.dirname(path)).split('_')[-1])
    t = None
    linf = None
    with open(path, errors='replace') as f:
        for line in f:
            mm = re.match(r'^Tempo\s*:\s*([-+0-9.eE]+)', line)
            if mm:
                t = float(mm.group(1))
            mm = re.match(r'^Linf:\s*([-+0-9.eE]+)', line)
            if mm:
                linf = float(mm.group(1))
    if t is None or linf is None:
        raise SystemExit(f'Missing Tempo/Linf in {path}')
    times.append(dict(threads=threads, backend=backend, policy=policy, rep=rep,
                      time_s=t, linf=linf))

if not times:
    raise SystemExit('No clean timing logs found')

with open(os.path.join(outdir, 'times.csv'), 'w', newline='') as f:
    cols = ['threads','backend','policy','rep','time_s','linf']
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader(); w.writerows(times)

# -------------------- profiling data --------------------
prof_rows = []
fields = {
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
    'Linf': 'linf',
}

for path in sorted(glob.glob(os.path.join(outdir, 'threads_*', '*_prof_r*.log'))):
    b = os.path.basename(path)
    m = re.fullmatch(r'(busywait|semaphore)_(wait|progress|critical)_prof_r(\d+)\.log', b)
    if not m:
        continue
    backend, policy, rep = m.group(1), m.group(2), int(m.group(3))
    threads = int(os.path.basename(os.path.dirname(path)).split('_')[-1])
    row = dict(threads=threads, backend=backend, policy=policy, rep=rep)
    with open(path, errors='replace') as f:
        for line in f:
            mm = re.match(r'^Tempo\s*:\s*([-+0-9.eE]+)', line)
            if mm:
                row['profile_time_s'] = float(mm.group(1))
            for prefix, key in fields.items():
                if line.startswith(prefix + ':'):
                    tok = line.split(':',1)[1].strip().split()[0]
                    row[key] = float(tok)
                    break
    prof_rows.append(row)

prof_cols = [
    'threads','backend','policy','rep','profile_time_s',
    'recompute','wait','lead_wait','progress_blocked',
    'penalty_samples','penalty_nonzero_samples','penalty_mean_ticks',
    'penalty_nonzero_fraction','critical_recompute',
    'noncritical_recompute_rejected','criticality_forced_wait','linf'
]
for r in prof_rows:
    for c in prof_cols:
        if c not in r:
            if c in ('threads','backend','policy','rep','profile_time_s','linf'):
                raise SystemExit(f'Missing required profile field {c} in {r}')
            r[c] = 0.0

with open(os.path.join(outdir, 'profile_per_run.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=prof_cols)
    w.writeheader(); w.writerows(prof_rows)

# -------------------- summary + speedups --------------------
keys = sorted({(r['threads'],r['backend'],r['policy']) for r in times})
summary = []
for threads, backend, policy in keys:
    tt = [r for r in times if (r['threads'],r['backend'],r['policy']) == (threads,backend,policy)]
    pp = [r for r in prof_rows if (r['threads'],r['backend'],r['policy']) == (threads,backend,policy)]
    if len(tt) != reps:
        raise SystemExit(f'Expected {reps} timing reps for {(threads,backend,policy)}, got {len(tt)}')
    if len(pp) != profile_reps:
        raise SystemExit(f'Expected {profile_reps} profile reps for {(threads,backend,policy)}, got {len(pp)}')

    med_t = statistics.median(r['time_s'] for r in tt)
    wait_tt = [r for r in times if (r['threads'],r['backend'],r['policy']) == (threads,backend,'wait')]
    med_wait = statistics.median(r['time_s'] for r in wait_tt)
    speedup = med_wait / med_t
    reduction = (med_wait - med_t) / med_wait * 100.0

    med = lambda k: statistics.median(float(r[k]) for r in pp)
    summary.append({
        'threads': threads,
        'backend': backend,
        'policy': policy,
        'timing_reps': len(tt),
        'profile_reps': len(pp),
        'median_time_s': med_t,
        'speedup_vs_wait': speedup,
        'time_reduction_vs_wait_pct': reduction,
        'median_profile_time_s': med('profile_time_s'),
        'median_recompute': med('recompute'),
        'median_wait': med('wait'),
        'median_lead_wait': med('lead_wait'),
        'median_progress_blocked': med('progress_blocked'),
        'median_penalty_mean_ticks': med('penalty_mean_ticks'),
        'median_penalty_nonzero_fraction': med('penalty_nonzero_fraction'),
        'median_critical_recompute': med('critical_recompute'),
        'median_noncritical_recompute_rejected': med('noncritical_recompute_rejected'),
        'median_criticality_forced_wait': med('criticality_forced_wait'),
        'median_linf': med('linf'),
    })

summary_path = os.path.join(outdir, 'summary.csv')
with open(summary_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    w.writeheader(); w.writerows(summary)

print('\nResults:')
print('  ', os.path.join(outdir, 'times.csv'))
print('  ', os.path.join(outdir, 'profile_per_run.csv'))
print('  ', summary_path)
print('\nSpeedup vs WAIT (median times):')
for r in summary:
    if r['policy'] != 'wait':
        print(f"  p={r['threads']:2d} {r['backend']:9s} {r['policy']:8s} "
              f"S={r['speedup_vs_wait']:.5f} "
              f"delta={r['time_reduction_vs_wait_pct']:+.2f}%")
PY

echo
echo "Haswell-32 baseline complete."
