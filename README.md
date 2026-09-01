Heat2D - modelos de custo e piso empirico de dependencias
==========================================================

Arquivos
--------

heat2d_explicit_omp_mpilike.cpp
    Baseline original com espera + copia de halos.

heat2d_explicit_omp_mpilike_oracle.cpp
    Referencia compute-only sem resolucao de dependencias.
    NAO E NUMERICAMENTE VALIDA. Nao espera, nao copia halos e nao executa
    READ/RECOMPUTE/PREDICT/WAIT. Os halos locais ficticios permanecem locais.
    Serve como piso empirico otimista do custo de computacao, nao como um
    limite inferior matematico rigoroso.

heat2d_explicit_omp_mpilike_calibrate.cpp
    Calibrador offline. Usa oportunidades reais da aplicacao para medir:
      - try_predict_ticks: bound + budget + admissibilidade + halo previsto;
      - recompute_ticks: reconstrucao da linha por RECOMPUTE;
      - accept_probability: fracao das tentativas numericamente admissiveis.
    Grava heat2d_cost_model.dat.

heat2d_explicit_omp_mpilike_adaptive_calibrated.cpp
    Adaptativo que le heat2d_cost_model.dat antes do cronometro.
    Nao faz bootstrap, RDTSC/RDTSCP, EWMA ou reamostragem online.
    Se READ falha e RECOMPUTE e possivel, tenta PREDICT somente se:

        margin * C_tryP < a * C_R

    onde a e a probabilidade de aceitacao obtida pelo calibrador.

heat2d_explicit_omp_mpilike_adaptive_online.cpp
    Versao anterior com aprendizado online por thread.

Compilacao
----------

    cp Makefile_dependency_models Makefile
    make -j

Uso basico
----------

1. Configure param.txt e OpenMP como no experimento final, por exemplo:

    export OMP_NUM_THREADS=32
    export OMP_PLACES=cores
    export OMP_PROC_BIND=close

2. Calibre na MESMA maquina, com o mesmo N, TILE, numero de threads,
   theta/mu, eta e kappa usados no teste:

    HEAT2D_CALIBRATION_SAMPLES=16 ./heat2d_dependency_calibrate

   Isso gera:

    heat2d_cost_model.dat

3. Rode a versao calibrada:

    ./heat2d_adaptive_calibrated

4. Rode baseline e piso empirico:

    ./heat2d_explicit_omp_mpilike
    ./heat2d_dependency_oracle

5. Se quiser comparar com aprendizado online:

    ./heat2d_adaptive_online

Arquivo de custos
-----------------

Exemplo:

    format heat2d_cost_model_v1
    tick_unit cycles
    N 8192
    T_calibration 1000
    TILE 32
    threads 32
    mu 0.225
    eta 0.5
    kappa 1
    samples_per_thread 16
    ready_models 32
    try_predict_ticks ...
    recompute_ticks ...
    accept_probability ...

A versao calibrada rejeita, por padrao, arquivos obtidos com N, TILE,
numero de threads, mu, eta ou kappa diferentes. Para deliberadamente ignorar
essa protecao:

    HEAT2D_ALLOW_COST_MISMATCH=1 ./heat2d_adaptive_calibrated

Isso nao e recomendado para os experimentos finais.

Custo esperado
--------------

A tentativa de PREDICT custa C_tryP. Se for rejeitada, ainda se paga C_R.
Com probabilidade de aceitacao a:

    E[C_try] = C_tryP + (1-a) C_R

comparado com RECOMPUTE direto:

    E[C_R] = C_R

Logo vale sequer testar PREDICT quando:

    C_tryP < a C_R.

Oracle / compute floor
----------------------

Defina:

    T0 = tempo do oracle compute-only
    TB = tempo do baseline
    TA = tempo adaptativo

Uma metrica experimental util e:

    recovery = (TB - TA) / (TB - T0)

quando TB > T0. Ela mede qual fracao do overhead potencialmente removivel
entre baseline e o piso empirico foi recuperada pela estrategia adaptativa.

Cuidado: T0 nao e um limite inferior matematico rigoroso. Remover
sincronizacao e comunicacao altera concorrencia, cache e trafego de memoria.
Ele deve ser descrito como "optimistic empirical compute-only floor".

OUTPUT DO CAMPO FINAL
=====================

Todos os executaveis usam o parametro WRITE_OUTPUT de param.txt:

    WRITE_OUTPUT = 0

Nao escreve output.txt. Este e o modo recomendado para testes de desempenho,
pois a escrita do campo completo pode dominar o tempo fora do kernel. As
metricas L1_mean, L2_rms e Linf continuam sendo calculadas normalmente.

Para gerar o campo final completo:

    WRITE_OUTPUT = 1

Tambem sao aceitos: on/off, true/false e yes/no. Se WRITE_OUTPUT estiver
ausente, o valor padrao e 0 (desabilitado).

