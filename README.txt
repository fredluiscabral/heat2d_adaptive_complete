FTCS 2D OpenMP MPI-like com READ / RECOMPUTE / PREDICT / WAIT
================================================================

Arquivos
--------
heat2d_explicit_omp_mpilike_adaptive.cpp : solver adaptativo
heat2d_explicit_common.hpp                : parametros, memoria alinhada, solucao exata e erros
param.txt                                 : parametros numericos
Makefile                                  : compilacao e execucao

Compilacao
----------
make

ou diretamente:

g++ -O3 -std=c++17 -fopenmp -march=native \
    heat2d_explicit_omp_mpilike_adaptive.cpp -o heat2d_adaptive

Execucao basica
---------------
export OMP_NUM_THREADS=8
export OMP_PLACES=cores
export OMP_PROC_BIND=close
./heat2d_adaptive

O param.txt usa:
N      = numero de pontos por direcao, incluindo o contorno
T      = numero de passos temporais
TILE   = tamanho do bloco espacial
alpha  = difusividade da equacao do calor
theta  = fracao do limite CFL do FTCS 2D

A relacao usada e
    mu = alpha*dt/h^2 = theta/4
    dt = theta*h^2/(4*alpha)
com 0 < theta <= 1.

Solucao de verificacao
----------------------
O problema usa dominio [0,1]^2, contorno de Dirichlet homogeneo e

    u(x,y,t) = sin(pi*x) sin(pi*y) exp(-2*pi^2*alpha*t).

Assim o programa calcula L1 medio, L2 RMS e Linf contra uma solucao conhecida.

Parametros adaptativos por variavel de ambiente
------------------------------------------------
HEAT2D_ETA=0.5
HEAT2D_KAPPA=1.0
HEAT2D_ENABLE_PREDICT=1
HEAT2D_ENABLE_RECOMPUTE=1
HEAT2D_COST_PREDICT=2.0
HEAT2D_COST_RECOMPUTE=6.0
HEAT2D_COST_WAIT=1000.0
HEAT2D_BUDGET_FLOOR=1e-30

Nesta versao HEAT2D_MAX_LEAD e efetivamente 2.

Teste com desequilibrio artificial
----------------------------------
Para provocar dependencias ainda nao prontas durante uma verificacao funcional:

export HEAT2D_DELAY_TID=1
export HEAT2D_DELAY_US=50
./heat2d_adaptive

Para experimentos reais de desempenho, retire o atraso:

unset HEAT2D_DELAY_TID
unset HEAT2D_DELAY_US

Saidas
------
output.txt         : campo final, em colunas x y U
adaptive_stats.txt : contagens READ/RECOMPUTE/PREDICT/WAIT e estatisticas do controlador
stdout             : parametros, tempo de execucao e erros contra a solucao exata

Para testes de desempenho grandes, output.txt pode gerar bastante I/O. Se desejado,
essa escrita pode ser desativada depois para que a ROI meça somente o solver.
