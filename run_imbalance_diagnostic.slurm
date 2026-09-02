#!/bin/bash
#SBATCH --job-name=heat2d_criticality
#SBATCH --partition=lncc-cpu_amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=heat2d_criticality-%j.out
#SBATCH --error=heat2d_criticality-%j.err

set -euo pipefail

REPS="${REPS:-3}"
THREADS_LIST_STR="${THREADS_LIST:-64 128 160 192}"
read -r -a THREADS_LIST <<< "${THREADS_LIST_STR}"
CAL_SAMPLES="${CAL_SAMPLES:-32}"
WAIT_CAL_SAMPLES="${WAIT_CAL_SAMPLES:-32}"
PROGRESS_LAMBDA="${PROGRESS_LAMBDA:-1.0}"
PROGRESS_BOOTSTRAP="${PROGRESS_BOOTSTRAP:-8}"
OUTDIR="${OUTDIR:-results/criticality_policy}"

BW_REF="./heat2d_adaptive_busywait"
SM_REF="./heat2d_adaptive_sem"
BW_PROF="./heat2d_adaptive_busywait_profile"
SM_PROF="./heat2d_adaptive_sem_profile"
PR_CAL="./heat2d_dependency_calibrate"
BW_WCAL="./heat2d_wait_calibrate_busywait"
SM_WCAL="./heat2d_wait_calibrate_semaphore"

cd "${SLURM_SUBMIT_DIR}"
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_DYNAMIC=FALSE
export HEAT2D_MAX_LEAD=2

[[ -f param.txt ]] || { echo "ERRO: param.txt nao encontrado." >&2; exit 1; }
WRITE_OUTPUT_VALUE="$(awk -F= '/^[[:space:]]*WRITE_OUTPUT[[:space:]]*=/{gsub(/[[:space:]]/,"",$2); print tolower($2); exit}' param.txt)"
case "${WRITE_OUTPUT_VALUE}" in
    0|off|false|no) ;;
    *) echo "ERRO: configure WRITE_OUTPUT = 0 em param.txt." >&2; exit 2 ;;
esac

mkdir -p "${OUTDIR}"

# O alvo padrao 'all' nao inclui os builds de profiling. Este experimento usa ambos.
make -j "${SLURM_CPUS_PER_TASK}" core profiles

for exe in "${BW_REF}" "${SM_REF}" "${BW_PROF}" "${SM_PROF}" "${PR_CAL}" "${BW_WCAL}" "${SM_WCAL}"; do
    [[ -x "${exe}" ]] || { echo "ERRO: executavel ausente: ${exe}" >&2; exit 3; }
done

run_solver() {
    local exe="$1"
    local backend="$2"
    local policy="$3"
    local threads="$4"
    local rep="$5"
    local pdir="$6"
    local cost="$7"
    local waitcost="$8"
    local kind="$9"

    local recompute criticality
    case "${policy}" in
        wait)
            recompute=0
            criticality=1
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
            exit 4
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
        exit 5
    }

    if [[ "${criticality}" == "1" ]]; then
        grep -q '^Adaptive criticality-aware: on$' "${log}" || {
            echo "ERRO: controlador criticality-aware nao esta ativo em ${log}" >&2
            exit 6
        }
    else
        grep -q '^Adaptive criticality-aware: off$' "${log}" || {
            echo "ERRO: controlador criticality-aware deveria estar desligado em ${log}" >&2
            exit 7
        }
    fi
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

    echo "[warm-up] six cases"
    for policy in wait progress critical; do
        run_solver "${BW_REF}" busywait  "${policy}" "${threads}" 0 "${pdir}" "${cost}" "${bw_wait}" warmup
        run_solver "${SM_REF}" semaphore "${policy}" "${threads}" 0 "${pdir}" "${cost}" "${sm_wait}" warmup
    done

    echo "[measured runs]"
    for ((rep=1; rep<=REPS; ++rep)); do
        # Clean reference timing.
        for policy in wait progress critical; do
            run_solver "${BW_REF}" busywait  "${policy}" "${threads}" "${rep}" "${pdir}" "${cost}" "${bw_wait}" ref
            run_solver "${SM_REF}" semaphore "${policy}" "${threads}" "${rep}" "${pdir}" "${cost}" "${sm_wait}" ref
        done

        # Same policies with statistics enabled at compile time.
        for policy in wait progress critical; do
            run_solver "${BW_PROF}" busywait  "${policy}" "${threads}" "${rep}" "${pdir}" "${cost}" "${bw_wait}" prof
            run_solver "${SM_PROF}" semaphore "${policy}" "${threads}" "${rep}" "${pdir}" "${cost}" "${sm_wait}" prof
        done

        echo "threads=${threads} rep=${rep}/${REPS} complete"
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
reps = int(sys.argv[2])
rows = []

for ref in sorted(glob.glob(os.path.join(outdir, 'threads_*', '*_ref_r*.log'))):
    b = os.path.basename(ref)
    m = re.fullmatch(r'(busywait|semaphore)_(wait|progress|critical)_ref_r(\d+)\.log', b)
    if not m:
        continue

    backend, policy, rep = m.group(1), m.group(2), int(m.group(3))
    threads = int(os.path.basename(os.path.dirname(ref)).split('_')[-1])
    prof = os.path.join(os.path.dirname(ref), f'{backend}_{policy}_prof_r{rep:02d}.log')

    vals = {
        'threads': threads,
        'backend': backend,
        'policy': policy,
        'rep': rep,
    }

    for path, isprof in [(ref, False), (prof, True)]:
        with open(path, errors='replace') as f:
            for line in f:
                if line.startswith('Tempo'):
                    mm = re.match(r'^Tempo\s*:\s*([-+0-9.eE]+)', line)
                    if mm:
                        vals['profile_time_s' if isprof else 'time_s'] = float(mm.group(1))

                if not isprof:
                    continue

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

                for prefix, key in fields.items():
                    if line.startswith(prefix + ':'):
                        tok = line.split(':', 1)[1].strip().split()[0]
                        vals[key] = float(tok)
                        break

    rows.append(vals)

if not rows:
    raise SystemExit('No result logs found')

cols = [
    'threads', 'backend', 'policy', 'rep',
    'time_s', 'profile_time_s',
    'recompute', 'wait', 'lead_wait', 'progress_blocked',
    'penalty_samples', 'penalty_nonzero_samples',
    'penalty_mean_ticks', 'penalty_nonzero_fraction',
    'critical_recompute', 'noncritical_recompute_rejected',
    'criticality_forced_wait', 'linf',
]

for r in rows:
    for c in cols:
        if c not in r:
            if c in ('threads', 'backend', 'policy', 'rep', 'time_s', 'profile_time_s', 'linf'):
                raise SystemExit(f'Missing required field {c} in row {r}')
            r[c] = 0.0

per_run_path = os.path.join(outdir, 'per_run.csv')
with open(per_run_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)

summary = []
keys = sorted({(r['threads'], r['backend'], r['policy']) for r in rows})
for key in keys:
    rr = [r for r in rows if (r['threads'], r['backend'], r['policy']) == key]
    if len(rr) != reps:
        raise SystemExit(f'Expected {reps} reps for {key}, got {len(rr)}')

    med = lambda k: statistics.median(float(r[k]) for r in rr)

    summary.append({
        'threads': key[0],
        'backend': key[1],
        'policy': key[2],
        'reps': len(rr),
        'median_time_s': med('time_s'),
        'median_profile_time_s': med('profile_time_s'),
        'median_recompute': med('recompute'),
        'median_wait': med('wait'),
        'median_lead_wait': med('lead_wait'),
        'median_progress_blocked': med('progress_blocked'),
        'median_penalty_samples': med('penalty_samples'),
        'median_penalty_nonzero_fraction': med('penalty_nonzero_fraction'),
        'median_penalty_mean_ticks': med('penalty_mean_ticks'),
        'median_critical_recompute': med('critical_recompute'),
        'median_noncritical_recompute_rejected': med('noncritical_recompute_rejected'),
        'median_criticality_forced_wait': med('criticality_forced_wait'),
        'median_linf': med('linf'),
    })

summary_path = os.path.join(outdir, 'summary.csv')
with open(summary_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    w.writeheader()
    w.writerows(summary)

print('Per-run CSV :', per_run_path)
print('Summary CSV :', summary_path)
PY

echo "Criticality-policy diagnostic complete."
