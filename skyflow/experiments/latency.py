"""Latency measurement on the *current* machine (S7b).

Cached snapshots carry ``build_time_ms`` from whichever machine built the
cache, so a fair three-stage latency figure re-measures graph construction
here: simulate the deterministic test scenarios, observe them and time
``TKGBuilder.build`` + proximity candidate generation epoch by epoch.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from skyflow.config import SkyFlowConfig
from skyflow.data.tkg_builder import TKGSnapshot


def _sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_graph_build(cfg: SkyFlowConfig, n_epochs: int = 1000, device: torch.device = torch.device("cpu"),
                        num_uavs: Optional[int] = None, split: str = "test",
                        max_scenarios: Optional[int] = None) -> Dict:
    """Time graph construction for up to ``n_epochs`` consecutive observation
    epochs of the ``split`` scenarios (every epoch, not every 10th).

    Returns per-epoch ms for builder.build (``build_ms``) and candidate
    generation (``candidates_ms``), plus graph statistics."""
    sim = cfg.make_simulator(num_uavs=num_uavs, seed=getattr(cfg.data, "sim_seed", cfg.training.seed))
    builder = cfg.make_builder()
    obs = cfg.observation_params()
    kw = cfg.dataset_kwargs()
    n_scen = cfg.data.split_scenarios(split)
    if max_scenarios is not None:
        n_scen = min(n_scen, max_scenarios)
    per_scen = int(round(cfg.data.scenario_duration_s * 10))     # 10 Hz epochs
    n_scen = max(1, min(n_scen, int(np.ceil(n_epochs / per_scen))))
    logs = sim.simulate_logs(split, n_scen, cfg.data.scenario_duration_s)

    build_ms, cand_ms = [], []
    n_edges, n_cands, n_pairs_scored = [], [], []
    done = 0
    for log in logs:
        if log.infrastructure is not None:
            sim.load_infrastructure(log.infrastructure)
        builder.reset()
        for epoch in range(log.n_epochs):
            if done >= n_epochs:
                break
            state = sim.observe(log, epoch, obs)
            t0 = time.perf_counter()
            snap = builder.build(state, device=device)
            _sync(device)
            t1 = time.perf_counter()
            if kw["candidates"] == "proximity":
                ci, cj = sim.proximity_candidates(snap, obs, kw["proximity_margin_m"])
                n_c = int(ci.size)
            else:
                n_c = int(snap.num_pair_candidates or 0)
            t2 = time.perf_counter()
            build_ms.append((t1 - t0) * 1e3)
            cand_ms.append((t2 - t1) * 1e3)
            n_edges.append(int(sum(e.shape[1] for e in snap.edge_indices.values())))
            n_cands.append(n_c)
            done += 1
    n_uav = num_uavs or cfg.data.num_uavs
    b, c = np.asarray(build_ms), np.asarray(cand_ms)
    tot = b + c
    return {
        "n_epochs": int(done),
        "num_uavs": int(n_uav),
        "build_ms": build_ms,
        "candidates_ms": cand_ms,
        "graph_build_mean_ms": float(tot.mean()) if done else float("nan"),
        "graph_build_p95_ms": float(np.percentile(tot, 95)) if done else float("nan"),
        "builder_only_p95_ms": float(np.percentile(b, 95)) if done else float("nan"),
        "candidates_only_p95_ms": float(np.percentile(c, 95)) if done else float("nan"),
        "mean_edges": float(np.mean(n_edges)) if done else float("nan"),
        "mean_degree": float(np.mean(n_edges) / n_uav) if done else float("nan"),
        "mean_candidate_pairs": float(np.mean(n_cands)) if done else float("nan"),
    }


def build_latency_dataset(cfg: SkyFlowConfig, num_uavs: Optional[int] = None, n_distinct: int = 100,
                          device: torch.device = torch.device("cpu"), split: str = "test"):
    """Up to ``n_distinct`` labelled snapshots of one scenario at ``num_uavs``
    (epoch stride chosen to spread them over the scenario) for inference timing."""
    sim = cfg.make_simulator(num_uavs=num_uavs, seed=getattr(cfg.data, "sim_seed", cfg.training.seed))
    logs = sim.simulate_logs(split, 1, cfg.data.scenario_duration_s)
    step = max(1, logs[0].n_epochs // max(n_distinct, 1))
    return sim.dataset_from_logs(logs, device=device, epoch_step=step, **cfg.dataset_kwargs())[:n_distinct]


def cycle_to_length(data: List[Tuple[TKGSnapshot, torch.Tensor]], n: int) -> List[Tuple[TKGSnapshot, torch.Tensor]]:
    if not data:
        return data
    out = []
    while len(out) < n:
        out.extend(data)
    return out[:n]


def measure_inference(evaluate, data: List[Tuple[TKGSnapshot, torch.Tensor]], n_epochs: int = 1000,
                      warmup: int = 20, device: torch.device = torch.device("cpu")) -> Dict:
    """Run ``evaluate`` over the data cycled to ``n_epochs`` snapshots (after a
    warm-up pass) and return per-stage mean / P95 on this machine."""
    if warmup > 0:
        evaluate(cycle_to_length(data, min(warmup, len(data))))
        _sync(device)
    res = evaluate(cycle_to_length(data, n_epochs))
    _sync(device)
    return {
        "n_epochs": int(min(n_epochs, len(cycle_to_length(data, n_epochs)))),
        "gnn_forward_mean_ms": res.stage_ms.get("gnn_forward"),
        "gnn_forward_p95_ms": res.stage_p95_ms.get("gnn_forward"),
        "pair_scoring_mean_ms": res.stage_ms.get("pair_scoring"),
        "pair_scoring_p95_ms": res.stage_p95_ms.get("pair_scoring"),
        "inference_p95_ms": res.latency_ms,          # forward + scoring per snapshot
        "inference_mean_ms": res.latency_mean_ms,
    }
