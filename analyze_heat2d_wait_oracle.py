#!/usr/bin/env python3
"""Offline critical-path analysis for Heat2D WAIT-only traces.

The trace executables record step timing and only actual blocking waits.  This
script builds the layered dependence DAG of the explicit stencil:

  (i, k-1) -> (i, k)
  (i-1, k-1) -> (i, k)
  (i+1, k-1) -> (i, k)

Node weight is the observed post-dependency local work for that thread/step
(ready_ns -> end_ns).  The model therefore separates dependency waiting from
local computation and lets us ask whether a WAIT edge belongs to a modeled
critical path.

Important: this is a diagnostic oracle for the recorded execution, not an exact
counterfactual prediction of the makespan after RECOMPUTE.  In particular,
phi_star_upper is explicitly an upper bound based on critical-path dominance
and the local predecessor gap.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import glob
import math
import os
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


def read_csv(path: str) -> List[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def as_int(row: dict, key: str) -> int:
    return int(row[key])


def median(xs: Iterable[float]) -> float:
    vals = list(xs)
    return float(statistics.median(vals)) if vals else 0.0


def mean(xs: Iterable[float]) -> float:
    vals = list(xs)
    return float(statistics.fmean(vals)) if vals else 0.0


def auc_binary(scores: List[float], labels: List[int]) -> float:
    """Mann-Whitney ROC AUC, with average ranks for ties."""
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    rank_sum_pos = 0.0
    i = 0
    n = len(pairs)
    while i < n:
        j = i + 1
        while j < n and pairs[j][0] == pairs[i][0]:
            j += 1
        # 1-based average rank of [i, j)
        avg_rank = ((i + 1) + j) / 2.0
        rank_sum_pos += avg_rank * sum(lbl for _, lbl in pairs[i:j])
        i = j

    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


@dataclass
class RunData:
    tag: str
    backend: str
    affinity: str
    rep: int
    N: int
    T: int
    TILE: int
    nt: int
    solver_time_s: float
    clock_overhead_ns: int
    steps: List[dict]
    waits: List[dict]


def load_runs(root: str) -> List[RunData]:
    runs: List[RunData] = []
    for meta_path in sorted(glob.glob(os.path.join(root, "*_meta.csv"))):
        tag = os.path.basename(meta_path)[:-9]  # strip _meta.csv
        steps_path = os.path.join(root, tag + "_steps.csv")
        waits_path = os.path.join(root, tag + "_waits.csv")
        if not os.path.exists(steps_path) or not os.path.exists(waits_path):
            raise RuntimeError(f"trace incompleto para {tag}")
        mrows = read_csv(meta_path)
        if len(mrows) != 1:
            raise RuntimeError(f"metadata invalido: {meta_path}")
        m = mrows[0]
        runs.append(RunData(
            tag=tag,
            backend=m["backend"],
            affinity=m["affinity"],
            rep=int(m["rep"]),
            N=int(m["N"]),
            T=int(m["T"]),
            TILE=int(m["TILE"]),
            nt=int(m["threads"]),
            solver_time_s=float(m["solver_time_s"]),
            clock_overhead_ns=int(m["clock_pair_overhead_ns"]),
            steps=read_csv(steps_path),
            waits=read_csv(waits_path),
        ))
    if not runs:
        raise RuntimeError(f"nenhum *_meta.csv encontrado em {root}")
    return runs


def analyze_run(run: RunData) -> Tuple[dict, List[dict]]:
    nt, T = run.nt, run.T
    if len(run.steps) != nt * T:
        raise RuntimeError(f"{run.tag}: esperado {nt*T} steps, recebido {len(run.steps)}")

    # Matrices indexed [step][tid].
    begin = [[0] * nt for _ in range(T)]
    ready = [[0] * nt for _ in range(T)]
    end = [[0] * nt for _ in range(T)]
    weight = [[0] * nt for _ in range(T)]

    for r in run.steps:
        t = int(r["tid"]); k = int(r["step"])
        begin[k][t] = int(r["begin_ns"])
        ready[k][t] = int(r["ready_ns"])
        end[k][t] = int(r["end_ns"])
        weight[k][t] = int(r["compute_ns"])

    first_begin = min(begin[0])
    source_offset = [max(0, begin[0][t] - first_begin) for t in range(nt)]

    # Forward longest-path DP.
    ef = [[0] * nt for _ in range(T)]
    es = [[0] * nt for _ in range(T)]
    for t in range(nt):
        es[0][t] = source_offset[t]
        ef[0][t] = source_offset[t] + weight[0][t]
    for k in range(1, T):
        prev = ef[k - 1]
        for t in range(nt):
            m = prev[t]
            if t > 0: m = max(m, prev[t - 1])
            if t + 1 < nt: m = max(m, prev[t + 1])
            es[k][t] = m
            ef[k][t] = m + weight[k][t]

    cp = max(ef[T - 1])

    # Best remaining path length after node completion.
    tail = [[0] * nt for _ in range(T)]
    for k in range(T - 2, -1, -1):
        for t in range(nt):
            best = weight[k + 1][t] + tail[k + 1][t]
            if t > 0:
                best = max(best, weight[k + 1][t - 1] + tail[k + 1][t - 1])
            if t + 1 < nt:
                best = max(best, weight[k + 1][t + 1] + tail[k + 1][t + 1])
            tail[k][t] = best

    # Critical-DAG path counts. Integers can grow large, which Python handles.
    node_critical = [[False] * nt for _ in range(T)]
    for k in range(T):
        for t in range(nt):
            node_critical[k][t] = (ef[k][t] + tail[k][t] == cp)

    count_to = [[0] * nt for _ in range(T)]
    for t in range(nt):
        if node_critical[0][t]:
            count_to[0][t] = 1
    for k in range(1, T):
        for t in range(nt):
            if not node_critical[k][t]:
                continue
            for u in (t - 1, t, t + 1):
                if 0 <= u < nt and node_critical[k - 1][u] and ef[k - 1][u] == es[k][t]:
                    count_to[k][t] += count_to[k - 1][u]

    count_from = [[0] * nt for _ in range(T)]
    for t in range(nt):
        if node_critical[T - 1][t] and ef[T - 1][t] == cp:
            count_from[T - 1][t] = 1
    for k in range(T - 2, -1, -1):
        for t in range(nt):
            if not node_critical[k][t]:
                continue
            for v in (t - 1, t, t + 1):
                if 0 <= v < nt and node_critical[k + 1][v] and ef[k][t] == es[k + 1][v]:
                    count_from[k][t] += count_from[k + 1][v]

    total_cp_paths = sum(count_to[T - 1][t] for t in range(nt) if ef[T - 1][t] == cp)
    if total_cp_paths <= 0:
        raise RuntimeError(f"{run.tag}: critical path count invalido")

    # Completion timestamps let us reconstruct progress at any wait start without
    # reading/scanning progress[] in the traced solver.
    end_by_tid = [[end[k][t] for k in range(T)] for t in range(nt)]

    # Wait intervals per waiter thread, for reconstructing blocked_on-like urgency.
    waits_by_tid: List[List[Tuple[int, int, int]]] = [[] for _ in range(nt)]
    for w in run.waits:
        wt = int(w["tid"])
        waits_by_tid[wt].append((int(w["start_ns"]), int(w["end_ns"]), int(w["neighbor"])))
    wait_starts_by_tid: List[List[int]] = []
    for t in range(nt):
        waits_by_tid[t].sort()
        wait_starts_by_tid.append([x[0] for x in waits_by_tid[t]])

    def progress_at(tid: int, ts: int) -> int:
        return bisect.bisect_right(end_by_tid[tid], ts)

    def blocked_on(waiter: int, target: int, ts: int) -> bool:
        if waiter < 0 or waiter >= nt:
            return False
        starts = wait_starts_by_tid[waiter]
        idx = bisect.bisect_right(starts, ts) - 1
        if idx < 0:
            return False
        a, b, nb = waits_by_tid[waiter][idx]
        return a <= ts <= b and nb == target

    oracle_rows: List[dict] = []
    for w in run.waits:
        tid = int(w["tid"])
        k = int(w["step"])
        nb = int(w["neighbor"])
        side = w["side"]
        ts = int(w["start_ns"])
        wait_raw = int(w["wait_ns"])
        wait_corr = int(w["wait_ns_corrected"])

        if k <= 0 or not (0 <= nb < nt):
            continue

        u_ef = ef[k - 1][nb]
        v_w = weight[k][tid]
        path_through = u_ef + v_w + tail[k][tid]
        edge_slack = max(0, cp - path_through)
        critical_member = int(edge_slack == 0 and u_ef == es[k][tid])

        other_preds = [ef[k - 1][tid]]
        if tid > 0 and tid - 1 != nb: other_preds.append(ef[k - 1][tid - 1])
        if tid + 1 < nt and tid + 1 != nb: other_preds.append(ef[k - 1][tid + 1])
        other_max = max(other_preds) if other_preds else 0
        local_gate = max(0, u_ef - other_max)

        edge_path_count = 0
        all_cp = 0
        if critical_member:
            edge_path_count = count_to[k - 1][nb] * count_from[k][tid]
            all_cp = int(edge_path_count == total_cp_paths)

        # Conservative/diagnostic upper bound, not an exact counterfactual.
        reducible_upper = min(wait_corr, local_gate) if all_cp else 0
        phi_star_upper = (reducible_upper / wait_corr) if wait_corr > 0 else 0.0

        progress = [progress_at(t, ts) for t in range(nt)]
        gmin = min(progress)
        level = int(w["expected_level"])
        frontier_slack = max(0, level - gmin)
        frontier = 1.0 / (1.0 + frontier_slack)

        other_nb = tid + 1 if nb == tid - 1 else tid - 1
        direct_block = int(blocked_on(other_nb, tid, ts)) if 0 <= other_nb < nt else 0
        if 0 <= other_nb < nt:
            other_progress = progress[other_nb]
            distance = max(0, level - other_progress)
            urgency = 1.0 if direct_block else 1.0 / (1.0 + distance)
        else:
            other_progress = -1
            urgency = 1.0
        phi_structural = frontier * urgency

        oracle_rows.append({
            "tag": run.tag,
            "backend": run.backend,
            "affinity": run.affinity,
            "rep": run.rep,
            "tid": tid,
            "step": k,
            "neighbor": nb,
            "side": side,
            "wait_ns": wait_raw,
            "wait_ns_corrected": wait_corr,
            "global_min_progress": gmin,
            "neighbor_progress": progress[nb],
            "other_neighbor": other_nb,
            "other_progress": other_progress,
            "other_directly_blocked_on_tid": direct_block,
            "frontier": frontier,
            "urgency": urgency,
            "phi_structural": phi_structural,
            "edge_slack_ns": edge_slack,
            "local_gate_ns": local_gate,
            "critical_path_member": critical_member,
            "all_critical_paths": all_cp,
            "oracle_reducible_upper_ns": reducible_upper,
            "phi_star_upper": phi_star_upper,
        })

    trace_start = min(begin[0])
    trace_end = max(end[T - 1])
    total_wait_ns = sum(int(w["wait_ns_corrected"]) for w in run.waits)
    critical_waits = [r for r in oracle_rows if r["critical_path_member"]]
    allcp_waits = [r for r in oracle_rows if r["all_critical_paths"]]

    summary = {
        "tag": run.tag,
        "backend": run.backend,
        "affinity": run.affinity,
        "rep": run.rep,
        "threads": nt,
        "T": T,
        "solver_time_s": run.solver_time_s,
        "trace_span_s": (trace_end - trace_start) / 1e9,
        "modeled_cp_s": cp / 1e9,
        "modeled_cp_over_trace_span": cp / max(1, trace_end - trace_start),
        "critical_path_count": str(total_cp_paths),
        "wait_events": len(run.waits),
        "wait_total_ms": total_wait_ns / 1e6,
        "critical_member_wait_events": len(critical_waits),
        "critical_member_wait_pct": 100.0 * len(critical_waits) / max(1, len(run.waits)),
        "critical_member_wait_ms": sum(r["wait_ns_corrected"] for r in critical_waits) / 1e6,
        "all_cp_wait_events": len(allcp_waits),
        "all_cp_wait_pct": 100.0 * len(allcp_waits) / max(1, len(run.waits)),
        "all_cp_wait_ms": sum(r["wait_ns_corrected"] for r in allcp_waits) / 1e6,
        "oracle_reducible_upper_ms": sum(r["oracle_reducible_upper_ns"] for r in oracle_rows) / 1e6,
        "phi_structural_mean": mean(r["phi_structural"] for r in oracle_rows),
        "phi_structural_critical_mean": mean(r["phi_structural"] for r in critical_waits),
        "phi_structural_noncritical_mean": mean(r["phi_structural"] for r in oracle_rows if not r["critical_path_member"]),
    }
    return summary, oracle_rows


def write_rows(path: str, rows: List[dict]) -> None:
    if not rows:
        with open(path, "w") as f:
            f.write("\n")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="directory containing *_steps.csv, *_waits.csv, *_meta.csv")
    args = ap.parse_args()
    root = os.path.abspath(args.root)

    runs = load_runs(root)
    run_summaries: List[dict] = []
    oracle_all: List[dict] = []
    for run in runs:
        s, rows = analyze_run(run)
        run_summaries.append(s)
        oracle_all.extend(rows)
        print(f"[oracle] {run.tag}: waits={s['wait_events']} critical={s['critical_member_wait_events']} allCP={s['all_cp_wait_events']}")

    write_rows(os.path.join(root, "run_summary.csv"), run_summaries)
    write_rows(os.path.join(root, "wait_oracle.csv"), oracle_all)

    grouped: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for s in run_summaries:
        grouped[(s["backend"], s["affinity"])].append(s)

    aggregate = []
    for (backend, affinity), rows in sorted(grouped.items()):
        aggregate.append({
            "backend": backend,
            "affinity": affinity,
            "reps": len(rows),
            "median_solver_time_s": median(r["solver_time_s"] for r in rows),
            "median_trace_span_s": median(r["trace_span_s"] for r in rows),
            "median_modeled_cp_s": median(r["modeled_cp_s"] for r in rows),
            "median_wait_events": median(r["wait_events"] for r in rows),
            "median_wait_total_ms": median(r["wait_total_ms"] for r in rows),
            "median_critical_member_wait_pct": median(r["critical_member_wait_pct"] for r in rows),
            "median_all_cp_wait_pct": median(r["all_cp_wait_pct"] for r in rows),
            "median_oracle_reducible_upper_ms": median(r["oracle_reducible_upper_ms"] for r in rows),
            "mean_phi_structural": mean(r["phi_structural_mean"] for r in rows),
            "mean_phi_structural_critical": mean(r["phi_structural_critical_mean"] for r in rows),
            "mean_phi_structural_noncritical": mean(r["phi_structural_noncritical_mean"] for r in rows),
        })
    write_rows(os.path.join(root, "aggregate_summary.csv"), aggregate)

    feature_rows = []
    waits_grouped: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for r in oracle_all:
        waits_grouped[(r["backend"], r["affinity"])].append(r)

    features = ["phi_structural", "wait_ns_corrected", "frontier", "urgency"]
    for (backend, affinity), rows in sorted(waits_grouped.items()):
        for label_name in ("critical_path_member", "all_critical_paths"):
            labels = [int(r[label_name]) for r in rows]
            for feat in features:
                scores = [float(r[feat]) for r in rows]
                auc = auc_binary(scores, labels)
                feature_rows.append({
                    "backend": backend,
                    "affinity": affinity,
                    "label": label_name,
                    "feature": feat,
                    "events": len(rows),
                    "positives": sum(labels),
                    "auc": auc,
                })
    write_rows(os.path.join(root, "feature_auc.csv"), feature_rows)

    print(f"[oracle] wrote {os.path.join(root, 'run_summary.csv')}")
    print(f"[oracle] wrote {os.path.join(root, 'aggregate_summary.csv')}")
    print(f"[oracle] wrote {os.path.join(root, 'wait_oracle.csv')}")
    print(f"[oracle] wrote {os.path.join(root, 'feature_auc.csv')}")

    native_path = os.path.join(root, "native_times.csv")
    if os.path.exists(native_path):
        native_rows = read_csv(native_path)
        ngrp: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        for r in native_rows:
            ngrp[(r["backend"], r["affinity"])].append(float(r["time_s"]))
        tgrp: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        for s in run_summaries:
            tgrp[(s["backend"], s["affinity"])].append(float(s["solver_time_s"]))
        perturb = []
        for key in sorted(set(ngrp) & set(tgrp)):
            backend, affinity = key
            nmed = median(ngrp[key])
            tmed = median(tgrp[key])
            perturb.append({
                "backend": backend,
                "affinity": affinity,
                "native_reps": len(ngrp[key]),
                "trace_reps": len(tgrp[key]),
                "median_native_s": nmed,
                "median_trace_s": tmed,
                "trace_over_native": (tmed / nmed) if nmed > 0 else float("nan"),
                "trace_overhead_pct": (100.0 * (tmed / nmed - 1.0)) if nmed > 0 else float("nan"),
            })
        write_rows(os.path.join(root, "perturbation_summary.csv"), perturb)
        print(f"[oracle] wrote {os.path.join(root, 'perturbation_summary.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
