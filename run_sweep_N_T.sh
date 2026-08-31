#!/usr/bin/env bash
set -euo pipefail

BASELINE="${BASELINE:-./heat2d_explicit_omp_mpilike}"
ADAPTIVE="${ADAPTIVE:-./heat2d_adaptive_validate}"
PARAM="${PARAM:-param.txt}"
OUTDIR="${OUTDIR:-sweep_N_T_results}"

# Edite estas listas como quiser.
N_VALUES=(128 256 512 1024)
T_VALUES=(100 1000 10000)

mkdir -p "${OUTDIR}"

if [[ ! -x "${BASELINE}" ]]; then
    echo "Erro: baseline não encontrado: ${BASELINE}" >&2
    exit 1
fi

if [[ ! -x "${ADAPTIVE}" ]]; then
    echo "Erro: adaptativo não encontrado: ${ADAPTIVE}" >&2
    exit 1
fi

if [[ ! -f "${PARAM}" ]]; then
    echo "Erro: ${PARAM} não encontrado." >&2
    exit 1
fi

# Guarda uma cópia do param.txt original.
BACKUP="${PARAM}.bak_sweep_$$"
cp "${PARAM}" "${BACKUP}"

restore_param() {
    mv -f "${BACKUP}" "${PARAM}"
}
trap restore_param EXIT

# Altera apenas N e T, mantendo os demais parâmetros.
set_param_value() {
    local key="$1"
    local value="$2"

    if grep -Eq "^[[:space:]]*${key}[[:space:]]*=" "${PARAM}"; then
        sed -i -E \
            "s|^[[:space:]]*${key}[[:space:]]*=.*$|${key} = ${value}|" \
            "${PARAM}"
    else
        echo "${key} = ${value}" >> "${PARAM}"
    fi
}

CSV="${OUTDIR}/results.csv"

echo "variant,N,T,h,dt,final_time,L1,L2,Linf,predict_count,max_ratio" > "${CSV}"

for N in "${N_VALUES[@]}"; do
    for T in "${T_VALUES[@]}"; do

        set_param_value N "${N}"
        set_param_value T "${T}"

        echo
        echo "============================================================"
        echo "N=${N}   T=${T}"
        echo "============================================================"

        B_LOG="${OUTDIR}/baseline_N${N}_T${T}.log"
        A_LOG="${OUTDIR}/adaptive_N${N}_T${T}.log"

        echo "[baseline]"
        "${BASELINE}" | tee "${B_LOG}"
        rm -f output.txt

        echo
        echo "[adaptive]"
        "${ADAPTIVE}" | tee "${A_LOG}"
        rm -f output.txt

        B_H="$(awk '/^h[[:space:]]*:/ {print $2; exit}' "${B_LOG}")"
        B_DT="$(awk '/^dt[[:space:]]*:/ {print $2; exit}' "${B_LOG}")"
        B_TF="$(awk '/^final_time[[:space:]]*:/ {print $2; exit}' "${B_LOG}")"
        B_L1="$(awk '/^L1_mean[[:space:]]*:/ {print $2; exit}' "${B_LOG}")"
        B_L2="$(awk '/^L2_rms[[:space:]]*:/ {print $2; exit}' "${B_LOG}")"
        B_LI="$(awk '/^Linf[[:space:]]*:/ {print $2; exit}' "${B_LOG}")"

        A_H="$(awk '/^h[[:space:]]*:/ {print $2; exit}' "${A_LOG}")"
        A_DT="$(awk '/^dt[[:space:]]*:/ {print $2; exit}' "${A_LOG}")"
        A_TF="$(awk '/^final_time[[:space:]]*:/ {print $2; exit}' "${A_LOG}")"
        A_L1="$(awk '/^L1_mean[[:space:]]*:/ {print $2; exit}' "${A_LOG}")"
        A_L2="$(awk '/^L2_rms[[:space:]]*:/ {print $2; exit}' "${A_LOG}")"
        A_LI="$(awk '/^Linf[[:space:]]*:/ {print $2; exit}' "${A_LOG}")"

        A_PRED="$(awk '/Adaptive actions PREDICT:/ {print $NF; exit}' "${A_LOG}")"
        A_MAXR="$(awk '/Accepted PREDICT max Esync\/\(eta\*B\):/ {print $NF; exit}' "${A_LOG}")"

        A_PRED="${A_PRED:-0}"
        A_MAXR="${A_MAXR:-0}"

        echo "baseline,${N},${T},${B_H},${B_DT},${B_TF},${B_L1},${B_L2},${B_LI},0,0" >> "${CSV}"
        echo "adaptive,${N},${T},${A_H},${A_DT},${A_TF},${A_L1},${A_L2},${A_LI},${A_PRED},${A_MAXR}" >> "${CSV}"

    done
done

echo
echo "============================================================"
echo "VARREDURA CONCLUÍDA"
echo "============================================================"
echo "Resultados: ${CSV}"
echo "Logs:       ${OUTDIR}/"
