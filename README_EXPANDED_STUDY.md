# Heat2D — estudo completo de resolução de dependências

Este conjunto preserva o estudo MPI-like atual e acrescenta as famílias OpenMP
busy-wait e semáforo, ambas com mitigação de false sharing.

## Variantes temporizadas

1. `mpilike_baseline` -> `heat2d_explicit_omp_mpilike`
2. `mpilike_oracle` -> `heat2d_dependency_oracle`
3. `mpilike_calibrated` -> `heat2d_adaptive_calibrated`
4. `mpilike_online` -> `heat2d_adaptive_online`
5. `mpilike_compact` -> `heat2d_adaptive_compact`
6. `busywait_nofs` -> `heat2d_explicit_omp_busywait_nobarrier_nofs`
7. `busywait_adaptive` -> `heat2d_adaptive_busywait`
8. `semaphore_nofs` -> `heat2d_explicit_omp_sem_nobarrier_nofs`
9. `semaphore_adaptive` -> `heat2d_adaptive_sem`

## Calibração

Para cada número de threads, o script executa:

- o calibrador MPI-like existente para `C_tryP`, `C_R` e probabilidade de aceitação;
- um calibrador específico do WAIT busy-wait;
- um calibrador específico do WAIT por semáforo.

As novas variantes usam a decisão

`C_F = min(C_R, C_W)` quando RECOMPUTE está disponível, ou `C_F=C_W` caso contrário.

PREDICT só é tentado quando

`margin * C_tryP < a * C_F`

e continua sujeito ao orçamento numérico `E_sync <= eta * B_method`.

## False sharing

- campos globais alinhados e com leading dimension múltiplo da cache line;
- first-touch paralelo;
- `ProgressSlot` ocupa uma cache line;
- na família semáforo, cada `sem_t` ocupa um `SemSlot` de uma cache line;
- snapshots de interface têm metadados versionados alinhados.

## SDumont

O script `run_sdumont_models.slurm` usa uma alocação de 192 CPUs e executa
`OMP_NUM_THREADS = 64, 128, 160, 192`, com 1 warm-up e 10 execuções válidas
por variante. O diretório raiz continua sendo `benchmark_models_results_sd`.
Cada contagem de threads recebe um subdiretório `threads_064`, `threads_128`,
`threads_160` ou `threads_192`.

O script não usa `rm -rf` no diretório de resultados. Ele remove apenas artefatos
gerados conhecidos dentro do subdiretório da configuração corrente.

Antes de submeter, mantenha o seu `param.txt` atual com `WRITE_OUTPUT = 0`.

```bash
make -j 192
sbatch run_sdumont_models.slurm
```

Ao final:

```bash
cat benchmark_models_results_sd/summary.csv
```

## Residual WAIT diagnostic

The study now includes diagnostic builds that measure the **residual wait** seen
at the instant a dependency is first found unavailable:

- `heat2d_residual_profile_busywait`
- `heat2d_residual_profile_sem`

These are diagnostic binaries, not performance baselines. They preserve the
numerical FTCS update and the original synchronization family while adding an
availability probe and TSC-based timing around only the residual wait after an
initial miss. On x86, the reported unit is cycles.

The key comparison is against the independently calibrated RECOMPUTE cost:

`P(W_residual < C_RECOMPUTE)`.

A high value means that many immediate RECOMPUTE decisions are more expensive
than simply waiting for the dependency to become available. The profile also
reports min/mean/p50/p75/p90/p95/p99/max and the initial-miss fraction.

For a focused SDumont run use:

```bash
sbatch run_residual_wait_profile.slurm
```

Default thread counts are 64, 128, 160 and 192, with three profile repetitions.
Set `SAVE_SAMPLES=1` at submission time to retain every individual residual-wait
sample; otherwise only logs and consolidated CSV summaries are kept.
