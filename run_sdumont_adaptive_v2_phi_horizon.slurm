#!/bin/bash
#SBATCH --job-name=heat2d_v2_phiT
#SBATCH --partition=lncc-cpu_amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --output=heat2d_v2_phiT-%j.out
#SBATCH --error=heat2d_v2_phiT-%j.err

set -euo pipefail

# =============================================================================
# SDumont II — Adaptive-V2 phi horizon diagnostic
#
# Objetivo:
#   verificar se o estimador phi cresce ao longo da execução e como esse
#   comportamento se relaciona com RECOMPUTE, WAIT, bootstrap e desempenho.
#
# Mantido fixo:
#   N=8192, TILE=32, p=64, OMP_PROC_BIND=spread, PREDICT=off, max_lead=2
#
# Sweep padrão:
#   T = 50 100 200 300 500 1000
#   backend = busywait semaphore
#   policies = WAIT e Adaptive-V2
#
# Por caso:
#   - calibra P/R e WAIT para o T efetivo;
#   - 1 warm-up por política;
#   - 10 repetições de tempo;
#   - 5 repetições de profiling;
#
# Não altera o código do controlador nem o param.txt original.
# =============================================================================

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
OUTDIR="${OUTDIR:-results/sdumont_adaptive_v2_phi_horizon}"
THREADS="${THREADS:-64}"
T_LIST_STR="${T_LIST:-50 100 200 300 500 1000}"
read -r -a T_LIST <<< "${T_LIST_STR}"
REPS="${REPS:-10}"
PROFILE_REPS="${PROFILE_REPS:-5}"
WARMUPS="${WARMUPS:-1}"
CAL_SAMPLES="${CAL_SAMPLES:-32}"
WAIT_CAL_SAMPLES="${WAIT_CAL_SAMPLES:-32}"

PROGRESS_LAMBDA="${PROGRESS_LAMBDA:-1.0}"
V2_PROGRESS_BOOTSTRAP="${V2_PROGRESS_BOOTSTRAP:-8}"
V2_EXPLORE_PERIOD="${V2_EXPLORE_PERIOD:-16}"
V2_ONLINE_BETA="${V2_ONLINE_BETA:-0.20}"
V2_PROGRESS_BETA="${V2_PROGRESS_BETA:-0.20}"
V2_PHI_BETA="${V2_PHI_BETA:-0.25}"
V2_PHI_FLOOR="${V2_PHI_FLOOR:-0.0}"
V2_ONLINE_CLIP_FACTOR="${V2_ONLINE_CLIP_FACTOR:-4.0}"
V2_CR_SAMPLE_PERIOD="${V2_CR_SAMPLE_PERIOD:-32}"
V2_CW_SAMPLE_PERIOD="${V2_CW_SAMPLE_PERIOD:-32}"
V2_PROGRESS_SAMPLE_PERIOD="${V2_PROGRESS_SAMPLE_PERIOD:-8}"
V2_PHI_SAMPLE_PERIOD="${V2_PHI_SAMPLE_PERIOD:-2}"
V2_PHI_GLOBAL_REFRESH_STEPS="${V2_PHI_GLOBAL_REFRESH_STEPS:-4}"
V2_TIMER_CALIBRATION_SAMPLES="${V2_TIMER_CALIBRATION_SAMPLES:-128}"

BW_WAIT="${PROJECT_DIR}/heat2d_adaptive_busywait"
SM_WAIT="${PROJECT_DIR}/heat2d_adaptive_sem"
BW_WAIT_PROF="${PROJECT_DIR}/heat2d_adaptive_busywait_profile"
SM_WAIT_PROF="${PROJECT_DIR}/heat2d_adaptive_sem_profile"
BW_V2="${PROJECT_DIR}/heat2d_adaptive_v2_busywait"
SM_V2="${PROJECT_DIR}/heat2d_adaptive_v2_sem"
BW_V2_PROF="${PROJECT_DIR}/heat2d_adaptive_v2_busywait_profile"
SM_V2_PROF="${PROJECT_DIR}/heat2d_adaptive_v2_sem_profile"
PR_CAL="${PROJECT_DIR}/heat2d_dependency_calibrate"
BW_WCAL="${PROJECT_DIR}/heat2d_wait_calibrate_busywait"
SM_WCAL="${PROJECT_DIR}/heat2d_wait_calibrate_semaphore"

export OMP_PLACES=cores
export OMP_PROC_BIND=spread
export OMP_DYNAMIC=FALSE
export HEAT2D_MAX_LEAD=2

cd "${PROJECT_DIR}"
[[ -f param.txt ]] || { echo "ERRO: param.txt ausente em ${PROJECT_DIR}." >&2; exit 1; }

# O teste exige a discretização usada nos experimentos atuais.
get_param() {
    local file="$1" key="$2"
    awk -F= -v k="$key" '
        $1 ~ "^[[:space:]]*" k "[[:space:]]*$" {
            gsub(/[[:space:]]/, "", $2); print $2; exit
        }' "$file"
}

N_VALUE="$(get_param param.txt N)"
TILE_VALUE="$(get_param param.txt TILE)"
WRITE_OUTPUT_VALUE="$(get_param param.txt WRITE_OUTPUT | tr '[:upper:]' '[:lower:]')"
[[ "${N_VALUE}" == "8192" ]] || { echo "ERRO: N deve ser 8192; atual=${N_VALUE}." >&2; exit 2; }
[[ "${TILE_VALUE}" == "32" ]] || { echo "ERRO: TILE deve ser 32; atual=${TILE_VALUE}." >&2; exit 2; }
case "${WRITE_OUTPUT_VALUE}" in
    0|off|false|no) ;;
    *) echo "ERRO: WRITE_OUTPUT deve ser 0." >&2; exit 2 ;;
esac
(( THREADS >= 1 && THREADS <= SLURM_CPUS_PER_TASK )) || {
    echo "ERRO: THREADS invalido: ${THREADS}." >&2; exit 2;
}

mkdir -p "${OUTDIR}"
cp param.txt "${OUTDIR}/param.original.txt"
lscpu > "${OUTDIR}/lscpu.txt" 2>&1 || true
lscpu -e=CPU,NODE,SOCKET,CORE,ONLINE > "${OUTDIR}/lscpu_extended.txt" 2>&1 || true

MAKE_JOBS="${MAKE_JOBS:-${SLURM_CPUS_PER_TASK}}"
echo "[build] core + profiles + Adaptive-V2"
make -j "${MAKE_JOBS}" core profiles adaptive_v2 profiles_v2

for exe in \
    "${BW_WAIT}" "${SM_WAIT}" "${BW_WAIT_PROF}" "${SM_WAIT_PROF}" \
    "${BW_V2}" "${SM_V2}" "${BW_V2_PROF}" "${SM_V2_PROF}" \
    "${PR_CAL}" "${BW_WCAL}" "${SM_WCAL}"
do
    [[ -x "${exe}" ]] || { echo "ERRO: executavel ausente: ${exe}" >&2; exit 3; }
done

make_param_for_T() {
    local src="$1" dst="$2" t="$3"
    awk -v newT="$t" '
        BEGIN { done=0 }
        /^[[:space:]]*T[[:space:]]*=/ {
            print "T=" newT; done=1; next
        }
        { print }
        END {
            if (!done) print "T=" newT
        }
    ' "$src" > "$dst"
}

check_log() {
    local log="$1"
    grep -q '^Tempo[[:space:]]*:' "$log" || { echo "ERRO: Tempo ausente em ${log}" >&2; exit 4; }
    grep -q '^Linf:' "$log" || { echo "ERRO: Linf ausente em ${log}" >&2; exit 4; }
    grep -q "^Threads: ${THREADS}$" "$log" || { echo "ERRO: Threads inesperado em ${log}" >&2; exit 4; }
}

run_wait() {
    local backend="$1" cost="$2" waitcost="$3" log="$4" profile="$5"
    local exe
    if [[ "$backend" == "busywait" ]]; then
        [[ "$profile" == "1" ]] && exe="$BW_WAIT_PROF" || exe="$BW_WAIT"
    else
        [[ "$profile" == "1" ]] && exe="$SM_WAIT_PROF" || exe="$SM_WAIT"
    fi

    env \
        OMP_NUM_THREADS="$THREADS" \
        HEAT2D_ENABLE_PREDICT=0 \
        HEAT2D_ENABLE_RECOMPUTE=0 \
        HEAT2D_ENABLE_CRITICALITY=0 \
        HEAT2D_COST_FILE="$cost" \
        HEAT2D_WAIT_COST_FILE="$waitcost" \
        HEAT2D_PROGRESS_LAMBDA="$PROGRESS_LAMBDA" \
        "$exe" > "$log" 2>&1
    check_log "$log"
}

run_v2() {
    local backend="$1" cost="$2" waitcost="$3" log="$4" profile="$5"
    local exe
    if [[ "$backend" == "busywait" ]]; then
        [[ "$profile" == "1" ]] && exe="$BW_V2_PROF" || exe="$BW_V2"
    else
        [[ "$profile" == "1" ]] && exe="$SM_V2_PROF" || exe="$SM_V2"
    fi

    env \
        OMP_NUM_THREADS="$THREADS" \
        HEAT2D_ENABLE_PREDICT=0 \
        HEAT2D_ENABLE_RECOMPUTE=1 \
        HEAT2D_ENABLE_PHI=1 \
        HEAT2D_COST_FILE="$cost" \
        HEAT2D_WAIT_COST_FILE="$waitcost" \
        HEAT2D_PROGRESS_LAMBDA="$PROGRESS_LAMBDA" \
        HEAT2D_V2_PROGRESS_BOOTSTRAP="$V2_PROGRESS_BOOTSTRAP" \
        HEAT2D_V2_EXPLORE_PERIOD="$V2_EXPLORE_PERIOD" \
        HEAT2D_ONLINE_BETA="$V2_ONLINE_BETA" \
        HEAT2D_PROGRESS_BETA="$V2_PROGRESS_BETA" \
        HEAT2D_PHI_BETA="$V2_PHI_BETA" \
        HEAT2D_PHI_FLOOR="$V2_PHI_FLOOR" \
        HEAT2D_ONLINE_CLIP_FACTOR="$V2_ONLINE_CLIP_FACTOR" \
        HEAT2D_CR_SAMPLE_PERIOD="$V2_CR_SAMPLE_PERIOD" \
        HEAT2D_CW_SAMPLE_PERIOD="$V2_CW_SAMPLE_PERIOD" \
        HEAT2D_PROGRESS_SAMPLE_PERIOD="$V2_PROGRESS_SAMPLE_PERIOD" \
        HEAT2D_PHI_SAMPLE_PERIOD="$V2_PHI_SAMPLE_PERIOD" \
        HEAT2D_PHI_GLOBAL_REFRESH_STEPS="$V2_PHI_GLOBAL_REFRESH_STEPS" \
        HEAT2D_TIMER_CALIBRATION_SAMPLES="$V2_TIMER_CALIBRATION_SAMPLES" \
        "$exe" > "$log" 2>&1

    check_log "$log"
    grep -q '^Adaptive V2 phi-aware: on$' "$log" || {
        echo "ERRO: phi deveria estar on em ${log}" >&2; exit 5;
    }
}

for T_NOW in "${T_LIST[@]}"; do
    case_dir="${PROJECT_DIR}/${OUTDIR}/T_$(printf '%04d' "$T_NOW")"
    mkdir -p "$case_dir/calibration" "$case_dir/timing" "$case_dir/profile"
    make_param_for_T "${PROJECT_DIR}/param.txt" "$case_dir/param.txt" "$T_NOW"

    cost="$case_dir/calibration/heat2d_cost_model.dat"
    bw_waitcost="$case_dir/calibration/heat2d_wait_cost_busywait.dat"
    sm_waitcost="$case_dir/calibration/heat2d_wait_cost_semaphore.dat"

    echo
    echo "=================================================================="
    echo "T=${T_NOW} threads=${THREADS} affinity=spread"
    echo "=================================================================="

    # Todos os executaveis leem param.txt do cwd; cada horizonte usa seu diretorio.
    pushd "$case_dir" >/dev/null

    echo "[calibration] P/R"
    env OMP_NUM_THREADS="$THREADS" \
        HEAT2D_ENABLE_PREDICT=1 \
        HEAT2D_ENABLE_RECOMPUTE=1 \
        HEAT2D_CALIBRATION_SAMPLES="$CAL_SAMPLES" \
        HEAT2D_COST_FILE="$cost" \
        "$PR_CAL" > calibration/pr.log 2>&1

    echo "[calibration] WAIT busywait"
    env OMP_NUM_THREADS="$THREADS" \
        HEAT2D_WAIT_CALIBRATION_SAMPLES="$WAIT_CAL_SAMPLES" \
        HEAT2D_WAIT_COST_FILE="$bw_waitcost" \
        "$BW_WCAL" > calibration/wait_busywait.log 2>&1

    echo "[calibration] WAIT semaphore"
    env OMP_NUM_THREADS="$THREADS" \
        HEAT2D_WAIT_CALIBRATION_SAMPLES="$WAIT_CAL_SAMPLES" \
        HEAT2D_WAIT_COST_FILE="$sm_waitcost" \
        "$SM_WCAL" > calibration/wait_semaphore.log 2>&1

    for backend in busywait semaphore; do
        if [[ "$backend" == "busywait" ]]; then waitcost="$bw_waitcost"; else waitcost="$sm_waitcost"; fi

        for ((w=1; w<=WARMUPS; ++w)); do
            run_wait "$backend" "$cost" "$waitcost" "timing/${backend}_wait_warmup_${w}.log" 0
            run_v2   "$backend" "$cost" "$waitcost" "timing/${backend}_v2_warmup_${w}.log" 0
        done

        # Alterna a ordem por repeticao para reduzir viés temporal.
        for ((r=1; r<=REPS; ++r)); do
            if (( r % 2 == 1 )); then
                run_wait "$backend" "$cost" "$waitcost" "timing/${backend}_wait_rep_${r}.log" 0
                run_v2   "$backend" "$cost" "$waitcost" "timing/${backend}_v2_rep_${r}.log" 0
            else
                run_v2   "$backend" "$cost" "$waitcost" "timing/${backend}_v2_rep_${r}.log" 0
                run_wait "$backend" "$cost" "$waitcost" "timing/${backend}_wait_rep_${r}.log" 0
            fi
        done

        for ((r=1; r<=PROFILE_REPS; ++r)); do
            run_wait "$backend" "$cost" "$waitcost" "profile/${backend}_wait_profile_${r}.log" 1
            run_v2   "$backend" "$cost" "$waitcost" "profile/${backend}_v2_profile_${r}.log" 1
        done
    done

    popd >/dev/null
done

# Consolida apenas com a biblioteca padrao do Python.
python3 - "$PROJECT_DIR/$OUTDIR" "$THREADS" "${T_LIST[*]}" "$REPS" "$PROFILE_REPS" <<'PY'
import csv, glob, os, re, statistics, sys

root = sys.argv[1]
threads = int(sys.argv[2])
t_values = [int(x) for x in sys.argv[3].split()]
reps = int(sys.argv[4])
profile_reps = int(sys.argv[5])

def value(log, label, default=0.0):
    pat = re.compile(r'^' + re.escape(label) + r':\s*([-+0-9.eE]+)', re.M)
    m = pat.search(open(log, encoding='utf-8', errors='replace').read())
    return float(m.group(1)) if m else default

def tempo(log):
    return value(log, 'Tempo')

def linf(log):
    return value(log, 'Linf')

def med(xs):
    return statistics.median(xs) if xs else 0.0

summary_path = os.path.join(root, 'summary.csv')
with open(summary_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['T','threads','backend','reps','median_wait_s','median_v2_s',
                'wait_over_v2_ratio','v2_gain_vs_wait_pct','paired_ratio_median',
                'v2_wins','linf_wait','linf_v2'])
    for T in t_values:
        d = os.path.join(root, f'T_{T:04d}', 'timing')
        for b in ('busywait','semaphore'):
            wt = [tempo(os.path.join(d, f'{b}_wait_rep_{r}.log')) for r in range(1,reps+1)]
            vt = [tempo(os.path.join(d, f'{b}_v2_rep_{r}.log')) for r in range(1,reps+1)]
            ratios = [a/bb for a,bb in zip(wt,vt)]
            mw, mv = med(wt), med(vt)
            ratio = mw/mv
            w.writerow([T,threads,b,reps,mw,mv,ratio,(ratio-1.0)*100.0,
                        med(ratios),sum(a>bb for a,bb in zip(wt,vt)),
                        linf(os.path.join(d,f'{b}_wait_rep_1.log')),
                        linf(os.path.join(d,f'{b}_v2_rep_1.log'))])

fields = [
    ('read','Adaptive actions READ'),
    ('recompute','Adaptive actions RECOMPUTE'),
    ('wait','Adaptive actions WAIT'),
    ('lead_wait','Adaptive lead-guard waits'),
    ('sampled_lead_wait_ticks','V2 sampled lead-guard wait ticks sum'),
    ('global_scans','V2 global progress scans'),
    ('progress_blocked','Progress-blocked RECOMPUTE'),
    ('value_blocked','V2 immediate-value blocked RECOMPUTE'),
    ('model_rejected','V2 RECOMPUTE rejected by model'),
    ('exploratory_recompute','V2 exploratory RECOMPUTE'),
    ('bootstrap_wait','V2 bootstrap WAIT decisions'),
    ('model_evaluations','V2 model evaluations'),
    ('phi_mean','V2 phi mean'),
    ('phi_min','V2 phi min'),
    ('phi_max','V2 phi max'),
    ('lhs_mean','V2 LHS mean'),
    ('rhs_mean','V2 RHS mean'),
    ('cr_obs_samples','V2 observed RECOMPUTE samples'),
    ('cr_obs_mean','V2 observed RECOMPUTE mean'),
    ('cw_obs_samples','V2 observed WAIT samples'),
    ('cw_obs_mean','V2 observed WAIT mean'),
    ('final_cr_hat','V2 final CR hat mean'),
    ('final_cw_hat','V2 final CW hat mean'),
    ('final_contention_proxy','V2 final contention proxy mean'),
    ('final_phi_hat','V2 final phi hat mean'),
    ('model_cr_samples','V2 model CR samples'),
    ('model_cw_samples','V2 model CW samples'),
    ('model_phi_updates','V2 model phi updates'),
    ('bootstrap_ready_sides','V2 progress bootstrap ready sides'),
    ('bootstrap_total_sides','V2 progress bootstrap total sides'),
    ('penalty_samples','Progress penalty samples'),
    ('penalty_nonzero_samples','Progress penalty nonzero samples'),
    ('penalty_mean_ticks','Progress penalty observed mean'),
]

prof_path = os.path.join(root, 'profile_summary.csv')
with open(prof_path, 'w', newline='') as f:
    w = csv.writer(f)
    header = ['T','threads','backend','policy','profile_reps','median_time_s','median_linf'] + ['median_'+a for a,_ in fields]
    w.writerow(header)
    for T in t_values:
        d = os.path.join(root, f'T_{T:04d}', 'profile')
        for b in ('busywait','semaphore'):
            for policy in ('wait','v2'):
                logs = [os.path.join(d,f'{b}_{policy}_profile_{r}.log') for r in range(1,profile_reps+1)]
                row = [T,threads,b,policy,profile_reps,med([tempo(x) for x in logs]),med([linf(x) for x in logs])]
                for _,label in fields:
                    row.append(med([value(x,label,0.0) for x in logs]))
                w.writerow(row)

print(f'Gerado: {summary_path}')
print(f'Gerado: {prof_path}')
PY

echo
echo "=== summary.csv ==="
cat "${OUTDIR}/summary.csv"
echo
echo "=== profile_summary.csv ==="
cat "${OUTDIR}/profile_summary.csv"
