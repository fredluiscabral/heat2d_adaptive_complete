# Heat2D WAIT critical-path oracle diagnostic

This diagnostic is separate from Adaptive-V2. It traces the native WAIT-only
busy-wait and semaphore solvers and reconstructs a layered causal DAG offline.

## Runtime trace

For each thread and time step the trace stores:

- `begin_ns`: before dependency checks/waits;
- `ready_ns`: after all dependencies are available;
- `end_ns`: after the local update has been published/completed.

For each actual blocking dependency it stores waiter, neighbor, side, expected
level, start/end time, and wait duration. Events stay in per-thread memory and
are written after the timed region.

## Offline model

For step k of thread i, the DAG contains dependencies from step k-1 of:

- thread i;
- thread i-1, when it exists;
- thread i+1, when it exists.

Node weight is the observed local work `ready_ns -> end_ns`. The analyzer
computes forward/backward longest paths and labels each observed WAIT edge as:

- `critical_path_member`: belongs to at least one modeled critical path;
- `all_critical_paths`: belongs to every modeled critical path.

`phi_star_upper` is deliberately named an upper bound. It is not claimed to be
an exact counterfactual makespan reduction after RECOMPUTE.

The analyzer also reconstructs, offline and without runtime scanning, the
structural ingredients used by the current phi idea: global minimum progress,
frontier, urgency, and whether the opposite neighbor is blocked on the current
thread. `feature_auc.csv` measures how well these local features discriminate
critical waits.

## SDumont experiment

Fixed defaults:

- N=8192
- T=500
- TILE=32
- 64 threads
- affinity: close, spread
- backend: busywait, semaphore
- 3 native runs and 3 trace runs per case

Trace timing is diagnostic only. `perturbation_summary.csv` quantifies trace
perturbation relative to the native WAIT-only runs.
