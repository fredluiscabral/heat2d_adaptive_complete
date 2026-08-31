#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Sanity + convergence tests for local notebook execution
# Baseline x adaptive validate x adaptive perf
# ============================================================

BASELINE="${BASELINE:-./heat2d_explicit_omp_mpilike}"
VALIDATE="${VALIDATE:-./heat2d_adaptive_validate}"
PERF="${PERF:-./heat2d_adaptive_perf}"
PARAM="${PARAM:-param.txt}"

THREADS="${THREADS:-4}"
TILE="${TILE:-32}"
ALPHA="${ALPHA:-0.1}"
THETA="${THETA:-0.9}"
ETA="${ETA:-0.5}"

NREF="${NREF:-512}"
TREF="${TREF:-100}"

OUTDIR="${OUTDIR:-sanity_results}"
CSV="${OUTDIR}/convergence.csv"

if [[ "${FULL:-0}" == "1" ]]; then
    N_VALUES=(128 256 512 1024 2048)
else
    N_VALUES=(128 256 512 1024)
fi

mkdir -p "${OUTDIR}"

# ------------------------------------------------------------
# Preserve current param.txt
# ------------------------------------------------------------
BACKUP=""
if [[ -f "${PARAM}" ]]; then
    BACKUP="${PARAM}.bak_sanity_$$"
    cp "${PARAM}" "${BACKUP}"
fi

restore_param() {
    if [[ -n "${BACKUP}" && -f "${BACKUP}" ]]; then
        mv -f "${BACKUP}" "${PARAM}"
    fi
}
trap restore_param EXIT

# ------------------------------------------------------------
# OpenMP placement
# ------------------------------------------------------------
export OMP_NUM_THREADS="${THREADS}"
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# Keep controller parameters explicit if the program supports them.
export HEAT2D_ETA="${ETA}"

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
write_param() {
    local N="$1"
    local T="$2"
    cat > "${PARAM}" <<EOF
N = ${N}
T = ${T}
TILE = ${TILE}
alpha = ${ALPHA}
theta = ${THETA}
EOF
}

extract_baseline() {
    local log="$1"
    awk '
        /L1_error[[:space:]]*:/ {l1=$3}
        /L1_mean[[:space:]]*:/  {l1=$2}

        /L2_error[[:space:]]*:/ {l2=$3}
        /L2_rms[[:space:]]*:/   {l2=$2}

        /Linf_error[[:space:]]*:/ {li=$3}
        /^Linf[[:space:]]*:/      {li=$2}

        /Tempo[[:space:]]*:/ {tm=$3}

        END {print l1, l2, li, tm}
    ' "${log}"
}

extract_adaptive() {
    local log="$1"
    awk '
        /L1_mean[[:space:]]*:/ {l1=$2}
        /L2_rms[[:space:]]*:/  {l2=$2}
        /Linf[[:space:]]*:/    {li=$2}
        /Tempo[[:space:]]*:/   {tm=$3}
        END {print l1, l2, li, tm}
    ' "${log}"
}

cleanup_output() {
    # output.txt is outside the timed region and can become very large.
    rm -f output.txt
}

# ------------------------------------------------------------
# 1. Rebuild everything
# ------------------------------------------------------------
echo
echo "============================================================"
echo "1) BUILD"
echo "============================================================"

make clean
make

for exe in "${BASELINE}" "${VALIDATE}" "${PERF}"; do
    if [[ ! -x "${exe}" ]]; then
        echo "ERRO: executável não encontrado: ${exe}" >&2
        exit 10
    fi
done

echo "OK: os três binários foram gerados."

# ------------------------------------------------------------
# 2. Reference physical final time from NREF,TREF
# ------------------------------------------------------------
TFINAL="$(
python3 - "${NREF}" "${TREF}" "${ALPHA}" "${THETA}" <<'PY'
import sys
N = int(sys.argv[1])
T = int(sys.argv[2])
alpha = float(sys.argv[3])
theta = float(sys.argv[4])
h = 1.0/(N-1)
mu = theta/4.0
dt = mu*h*h/alpha
print(f"{T*dt:.17g}")
PY
)"

echo "Reference final_time = ${TFINAL}"

# ------------------------------------------------------------
# 3. Smoke test at NREF,TREF
# ------------------------------------------------------------
echo
echo "============================================================"
echo "2) SMOKE TEST: baseline / validate / perf"
echo "============================================================"

write_param "${NREF}" "${TREF}"

B_LOG="${OUTDIR}/smoke_baseline.log"
V_LOG="${OUTDIR}/smoke_validate.log"
P_LOG="${OUTDIR}/smoke_perf.log"

echo
echo "[baseline]"
"${BASELINE}" | tee "${B_LOG}"
cleanup_output

echo
echo "[adaptive validate]"
"${VALIDATE}" | tee "${V_LOG}"
cleanup_output

echo
echo "[adaptive perf]"
"${PERF}" | tee "${P_LOG}"
cleanup_output

if ! grep -q "Adaptive profile stats: on" "${V_LOG}"; then
    echo "ERRO: validate não está com profiling ON." >&2
    exit 20
fi

if ! grep -q "Adaptive profile stats: off" "${P_LOG}"; then
    echo "ERRO: perf não está com profiling OFF." >&2
    exit 21
fi

if grep -q "Adaptive actions READ:" "${P_LOG}"; then
    echo "ERRO: o binário perf ainda está imprimindo estatísticas detalhadas." >&2
    exit 22
fi

if grep -q "Accepted PREDICT max" "${P_LOG}"; then
    echo "ERRO: o binário perf ainda está calculando/imprimindo ratios diagnósticos." >&2
    exit 23
fi

ACCEPTED_MAX="$(awk '/Accepted PREDICT max Esync\/\(eta\*B\):/ {print $NF; exit}' "${V_LOG}")"

if [[ -z "${ACCEPTED_MAX}" ]]; then
    echo "ERRO: não foi possível obter o máximo de PREDICT aceito." >&2
    exit 24
fi

python3 - "${ACCEPTED_MAX}" <<'PY'
import sys
r=float(sys.argv[1])
if r > 1.0 + 1e-12:
    raise SystemExit(f"ERRO: PREDICT aceito excedeu orçamento: ratio={r}")
print(f"OK: max accepted Esync/(eta*B) = {r:.12g} <= 1")
PY

read -r B_L1 B_L2 B_LINF B_TIME < <(extract_baseline "${B_LOG}")
read -r V_L1 V_L2 V_LINF V_TIME < <(extract_adaptive "${V_LOG}")
read -r P_L1 P_L2 P_LINF P_TIME < <(extract_adaptive "${P_LOG}")

for v in B_L1 B_L2 B_LINF B_TIME V_L1 V_L2 V_LINF V_TIME P_L1 P_L2 P_LINF P_TIME; do
    if [[ -z "${!v:-}" ]]; then
        echo "ERRO: não foi possível extrair ${v} dos logs do smoke test." >&2
        exit 25
    fi
done

python3 - \
    "${B_L1}" "${B_L2}" "${B_LINF}" "${B_TIME}" \
    "${V_L1}" "${V_L2}" "${V_LINF}" "${V_TIME}" \
    "${P_L1}" "${P_L2}" "${P_LINF}" "${P_TIME}" <<'PY'
import sys
b1,b2,bi,bt,v1,v2,vi,vt,p1,p2,pi,pt = map(float,sys.argv[1:])

def pct(x,ref):
    return 100.0*(x/ref-1.0)

print("\nSMOKE SUMMARY")
print("               L1              L2              Linf            Tempo(s)")
print(f"baseline  {b1: .6e}   {b2: .6e}   {bi: .6e}   {bt:.6e}")
print(f"validate  {v1: .6e}   {v2: .6e}   {vi: .6e}   {vt:.6e}")
print(f"perf      {p1: .6e}   {p2: .6e}   {pi: .6e}   {pt:.6e}")
print()
print("Diferença relativa de erro vs baseline:")
print(f"validate: L1={pct(v1,b1):+.3f}%  L2={pct(v2,b2):+.3f}%  Linf={pct(vi,bi):+.3f}%")
print(f"perf:     L1={pct(p1,b1):+.3f}%  L2={pct(p2,b2):+.3f}%  Linf={pct(pi,bi):+.3f}%")
PY

# ------------------------------------------------------------
# 4. Convergence baseline x adaptive validate
# ------------------------------------------------------------
echo
echo "============================================================"
echo "3) CONVERGENCE TEST"
echo "============================================================"

echo "variant,N,h,T,dt,final_time,L1,L2,Linf,tempo_s,predict_count,accepted_max_ratio" > "${CSV}"

for N in "${N_VALUES[@]}"; do
    read -r H DT T ACTUAL_TFINAL < <(
        python3 - "${N}" "${ALPHA}" "${THETA}" "${TFINAL}" <<'PY'
import sys
N = int(sys.argv[1])
alpha = float(sys.argv[2])
theta = float(sys.argv[3])
tfinal = float(sys.argv[4])
h = 1.0/(N-1)
mu = theta/4.0
dt = mu*h*h/alpha
T = max(1, int(round(tfinal/dt)))
print(f"{h:.17g} {dt:.17g} {T:d} {T*dt:.17g}")
PY
    )

    write_param "${N}" "${T}"

    echo
    echo "------------------------------------------------------------"
    echo "N=${N}  T=${T}  h=${H}  dt=${DT}"
    echo "------------------------------------------------------------"

    BLOG="${OUTDIR}/conv_baseline_N${N}.log"
    VLOG="${OUTDIR}/conv_validate_N${N}.log"

    echo "[baseline]"
    "${BASELINE}" | tee "${BLOG}"
    cleanup_output

    echo "[adaptive validate]"
    "${VALIDATE}" | tee "${VLOG}"
    cleanup_output

    read -r L1 L2 LI TM < <(extract_baseline "${BLOG}")
    echo "baseline,${N},${H},${T},${DT},${ACTUAL_TFINAL},${L1},${L2},${LI},${TM},0,0" >> "${CSV}"

    read -r L1 L2 LI TM < <(extract_adaptive "${VLOG}")
    PC="$(awk '/Adaptive actions PREDICT:/ {print $NF; exit}' "${VLOG}")"
    MR="$(awk '/Accepted PREDICT max Esync\/\(eta\*B\):/ {print $NF; exit}' "${VLOG}")"
    PC="${PC:-0}"
    MR="${MR:-0}"

    python3 - "${MR}" "${N}" <<'PY'
import sys
r=float(sys.argv[1])
N=sys.argv[2]
if r > 1.0 + 1e-12:
    raise SystemExit(f"ERRO N={N}: prediction aceita acima do orçamento: {r}")
PY

    echo "adaptive,${N},${H},${T},${DT},${ACTUAL_TFINAL},${L1},${L2},${LI},${TM},${PC},${MR}" >> "${CSV}"
done

# ------------------------------------------------------------
# 5. Compute observed orders
# ------------------------------------------------------------
echo
echo "============================================================"
echo "RESULTADOS DE CONVERGÊNCIA"
echo "============================================================"

python3 - "${CSV}" <<'PY'
import csv, math, sys

path=sys.argv[1]
data={"baseline":[], "adaptive":[]}

with open(path,newline="") as f:
    for row in csv.DictReader(f):
        row["N"]=int(row["N"])
        for k in ("h","L1","L2","Linf","tempo_s","accepted_max_ratio"):
            row[k]=float(row[k])
        row["predict_count"]=int(row["predict_count"])
        data[row["variant"]].append(row)

for variant in ("baseline","adaptive"):
    rows=sorted(data[variant], key=lambda r:r["N"])
    print(f"\n{variant.upper()}")
    if variant=="adaptive":
        print("N       L1              p(L1)    L2              p(L2)    Linf            p(Linf)   PREDICT   maxR")
    else:
        print("N       L1              p(L1)    L2              p(L2)    Linf            p(Linf)")
    print("-"*112)

    prev=None
    orders=[]
    for r in rows:
        if prev is None:
            p1=p2=pi="-"
        else:
            den=math.log(prev["h"]/r["h"])
            vals=[
                math.log(prev["L1"]/r["L1"])/den,
                math.log(prev["L2"]/r["L2"])/den,
                math.log(prev["Linf"]/r["Linf"])/den
            ]
            orders.append(vals)
            p1,p2,pi=(f"{x:7.4f}" for x in vals)

        base=(f'{r["N"]:4d}  {r["L1"]:.6e}  {p1:>7}  '
              f'{r["L2"]:.6e}  {p2:>7}  '
              f'{r["Linf"]:.6e}  {pi:>7}')
        if variant=="adaptive":
            base += f'  {r["predict_count"]:7d}  {r["accepted_max_ratio"]:.6f}'
        print(base)
        prev=r

    # Sanity threshold only on the finest transition.
    if orders:
        finest=orders[-1]
        if any(x < 1.80 for x in finest):
            raise SystemExit(
                f"ERRO: ordem assintótica final de {variant} abaixo de 1.80: {finest}"
            )

print("\nOK: baseline e adaptativo preservam ordem aproximadamente 2 no refinamento final.")
PY

echo
echo "============================================================"
echo "SANITY TESTS CONCLUÍDOS COM SUCESSO"
echo "============================================================"
echo "CSV : ${CSV}"
echo "Logs: ${OUTDIR}/"
echo
echo "Para incluir N=2048:"
echo "  FULL=1 ./run_convergence.sh"
