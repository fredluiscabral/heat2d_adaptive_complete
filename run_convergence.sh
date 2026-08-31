#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Teste de convergência: FTCS baseline x versão adaptativa
# Execução local (sem SLURM)
# ------------------------------------------------------------

BASELINE="${BASELINE:-./heat2d_explicit_omp_mpilike}"
ADAPTIVE="${ADAPTIVE:-./heat2d_adaptive}"
PARAM="${PARAM:-param.txt}"

THREADS="${THREADS:-4}"
TILE="${TILE:-32}"
ALPHA="${ALPHA:-0.1}"
THETA="${THETA:-0.9}"

# Mesmo tempo físico do caso N=512, T=100 usado no primeiro teste.
TFINAL="${TFINAL:-0.000861669494219155}"

# Para o primeiro ensaio. Se 2048 ficar pesado, retire-o desta linha.
N_VALUES=(128 256 512 1024 2048)

OUTDIR="${OUTDIR:-convergence_results}"
CSV="${OUTDIR}/convergence.csv"

mkdir -p "${OUTDIR}"

if [[ ! -x "${BASELINE}" ]]; then
    echo "Erro: baseline não encontrado ou não executável: ${BASELINE}" >&2
    exit 1
fi

if [[ ! -x "${ADAPTIVE}" ]]; then
    echo "Erro: adaptativo não encontrado ou não executável: ${ADAPTIVE}" >&2
    exit 1
fi

# Preserva o param.txt atual.
BACKUP=""
if [[ -f "${PARAM}" ]]; then
    BACKUP="${PARAM}.bak_convergence_$$"
    cp "${PARAM}" "${BACKUP}"
fi

restore_param() {
    if [[ -n "${BACKUP}" && -f "${BACKUP}" ]]; then
        mv -f "${BACKUP}" "${PARAM}"
    fi
}
trap restore_param EXIT

export OMP_NUM_THREADS="${THREADS}"
export OMP_PLACES=cores
export OMP_PROC_BIND=close

echo "variant,N,h,T,dt,final_time,L1,L2,Linf,tempo_s" > "${CSV}"

extract_value() {
    local file="$1"
    local regex="$2"
    awk -v re="${regex}" '
        $0 ~ re {
            line=$0
            sub(/^.*:[[:space:]]*/, "", line)
            sub(/^.*=[[:space:]]*/, "", line)
            gsub(/[[:space:]]/, "", line)
            print line
            exit
        }
    ' "${file}"
}

for N in "${N_VALUES[@]}"; do
    read -r H DT T ACTUAL_TFINAL < <(
        python3 - "${N}" "${ALPHA}" "${THETA}" "${TFINAL}" <<'PY'
import sys, math
N = int(sys.argv[1])
alpha = float(sys.argv[2])
theta = float(sys.argv[3])
tfinal = float(sys.argv[4])

h = 1.0 / (N - 1)
mu = theta / 4.0
dt = mu * h*h / alpha
T = max(1, int(round(tfinal / dt)))
actual = T * dt
print(f"{h:.17g} {dt:.17g} {T:d} {actual:.17g}")
PY
    )

    cat > "${PARAM}" <<EOF
N = ${N}
T = ${T}
TILE = ${TILE}
alpha = ${ALPHA}
theta = ${THETA}
EOF

    echo
    echo "============================================================"
    echo "N=${N}  T=${T}  h=${H}  dt=${DT}"
    echo "final_time=${ACTUAL_TFINAL}"
    echo "============================================================"

    # ---------------- baseline ----------------
    BLOG="${OUTDIR}/baseline_N${N}.log"
    echo "[baseline]"
    "${BASELINE}" | tee "${BLOG}"

    B_L1="$(awk '/L1_error[[:space:]]*:/ {print $3; exit}' "${BLOG}")"
    B_L2="$(awk '/L2_error[[:space:]]*:/ {print $3; exit}' "${BLOG}")"
    B_LINF="$(awk '/Linf_error[[:space:]]*:/ {print $3; exit}' "${BLOG}")"
    B_TIME="$(awk '/Tempo[[:space:]]*:/ {print $3; exit}' "${BLOG}")"

    if [[ -z "${B_L1}" || -z "${B_L2}" || -z "${B_LINF}" || -z "${B_TIME}" ]]; then
        echo "Erro ao extrair resultados do baseline para N=${N}." >&2
        exit 2
    fi

    echo "baseline,${N},${H},${T},${DT},${ACTUAL_TFINAL},${B_L1},${B_L2},${B_LINF},${B_TIME}" >> "${CSV}"

    # ---------------- adaptativo ----------------
    ALOG="${OUTDIR}/adaptive_N${N}.log"
    echo "[adaptive]"
    "${ADAPTIVE}" | tee "${ALOG}"

    A_L1="$(awk '/L1_mean[[:space:]]*:/ {print $2; exit}' "${ALOG}")"
    A_L2="$(awk '/L2_rms[[:space:]]*:/ {print $2; exit}' "${ALOG}")"
    A_LINF="$(awk '/Linf[[:space:]]*:/ {print $2; exit}' "${ALOG}")"
    A_TIME="$(awk '/Tempo[[:space:]]*:/ {print $3; exit}' "${ALOG}")"

    if [[ -z "${A_L1}" || -z "${A_L2}" || -z "${A_LINF}" || -z "${A_TIME}" ]]; then
        echo "Erro ao extrair resultados adaptativos para N=${N}." >&2
        exit 3
    fi

    echo "adaptive,${N},${H},${T},${DT},${ACTUAL_TFINAL},${A_L1},${A_L2},${A_LINF},${A_TIME}" >> "${CSV}"
done

echo
echo "============================================================"
echo "RESULTADOS DE CONVERGÊNCIA"
echo "============================================================"

python3 - "${CSV}" <<'PY'
import csv, math, sys

path = sys.argv[1]
data = {"baseline": [], "adaptive": []}

with open(path, newline="") as f:
    for row in csv.DictReader(f):
        row = {k: (float(v) if k not in ("variant",) else v)
               for k, v in row.items()}
        data[row["variant"]].append(row)

for variant in ("baseline", "adaptive"):
    rows = sorted(data[variant], key=lambda r: r["N"])
    print(f"\n{variant.upper()}")
    print("N       L1              p(L1)    L2              p(L2)    Linf            p(Linf)")
    print("-" * 94)
    prev = None
    for r in rows:
        if prev is None:
            p1 = p2 = pi = "-"
        else:
            denom = math.log(prev["h"] / r["h"])
            p1 = f'{math.log(prev["L1"]/r["L1"])/denom:7.4f}'
            p2 = f'{math.log(prev["L2"]/r["L2"])/denom:7.4f}'
            pi = f'{math.log(prev["Linf"]/r["Linf"])/denom:7.4f}'
        print(f'{int(r["N"]):4d}  {r["L1"]:.6e}  {p1:>7}  '
              f'{r["L2"]:.6e}  {p2:>7}  '
              f'{r["Linf"]:.6e}  {pi:>7}')
        prev = r

print("\nEsperado sob Δt ~ h²: ordens próximas de 2 para baseline e adaptativo.")
PY

echo
echo "CSV: ${CSV}"
echo "Logs: ${OUTDIR}/"