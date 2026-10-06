"""Method table for the main experiment and ablations (S7a).

Every method is a *config transform* plus a *kind*:

  kind = "trgat"      TR-GAT family trained with ``SkyFlowTrainer``
  kind = "learned"    learned baseline from ``skyflow.baselines.registry``
  kind = "rule"       deterministic baseline (evaluate only; CPA-Rule may fit
                      thresholds on the validation split)

Learned baselines (LSTM-P, Tfm-P, STGCN, GAT-S) are retrained: the audit
(docs/repo_map.md) found they consumed the full 23-d feature vector,
including the leaking CPA features, so old results cannot be reused.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Dict, List

from skyflow.config import SkyFlowConfig


@dataclass(frozen=True)
class MethodSpec:
    name: str
    kind: str                                   # trgat | learned | rule
    transform: Callable[[SkyFlowConfig], SkyFlowConfig]
    baseline_name: str = ""                     # registry key for learned/rule
    group: str = "main"                         # main | ablation
    description: str = ""


def _identity(cfg: SkyFlowConfig) -> SkyFlowConfig:
    return cfg


def _set(path: str, value):
    def f(cfg: SkyFlowConfig) -> SkyFlowConfig:
        section, key = path.split(".")
        setattr(getattr(cfg, section), key, value)
        return cfg
    return f


def _chain(*fns):
    def f(cfg: SkyFlowConfig) -> SkyFlowConfig:
        for g in fns:
            cfg = g(cfg)
        return cfg
    return f


METHODS: Dict[str, MethodSpec] = {
    # ---- main comparison --------------------------------------------------
    "TR-GAT": MethodSpec("TR-GAT", "trgat", _identity, group="main",
                         description="Full model (leakage-free features, AoI δ, lookahead labels)."),
    "TR-GAT-NT": MethodSpec("TR-GAT-NT", "trgat", _set("model.use_temporal", False), group="main",
                            description="No temporal encoding φ(δ) in attention."),
    "GAT-S": MethodSpec("GAT-S", "learned", _identity, baseline_name="GAT-S", group="main",
                        description="Static GAT over the merged graph."),
    "STGCN": MethodSpec("STGCN", "learned", _identity, baseline_name="STGCN", group="main"),
    "LSTM-P": MethodSpec("LSTM-P", "learned", _identity, baseline_name="LSTM-P", group="main"),
    "Tfm-P": MethodSpec("Tfm-P", "learned", _identity, baseline_name="Tfm-P", group="main"),
    "CPA-Rule": MethodSpec("CPA-Rule", "rule", _identity, baseline_name="CPA-Rule", group="main"),
    "Plan-CPA": MethodSpec("Plan-CPA", "rule", _identity, baseline_name="Plan-CPA", group="main",
                           description="CPA interval test along the filed-plan polyline (S8d)."),
    "VO": MethodSpec("VO", "rule", _identity, baseline_name="VO", group="main"),
    # ---- ablations (TR-GAT variants) ---------------------------------------
    "abl_no_gating": MethodSpec("abl_no_gating", "trgat", _set("model.use_gating", False), group="ablation",
                                description="Uniform relation average instead of learned gate g_r."),
    "abl_no_gru": MethodSpec("abl_no_gru", "trgat", _set("model.use_gru", False), group="ablation",
                             description="No recurrence across the K-epoch window."),
    "abl_bce": MethodSpec("abl_bce", "trgat", _set("training.loss", "bce"), group="ablation",
                          description="Plain BCE instead of focal loss."),
    "abl_telemetry_only": MethodSpec("abl_telemetry_only", "trgat", _set("features.input_set", "telemetry_only"),
                                     group="ablation",
                                     description="UAV nodes + approaches edges only (no context streams)."),
    "abl_no_plan": MethodSpec("abl_no_plan", "trgat", _chain(_set("features.plan_context", False),
                                                             _set("features.pair_edge_features", "geometry")),
                              group="ablation",
                              description="Observation-only UAV and pair features (no filed-plan context; = S8c setting)."),
    # abl_no_temporal == TR-GAT-NT (reused, not re-run)
}

MAIN_METHODS: List[str] = ["TR-GAT", "TR-GAT-NT", "GAT-S", "STGCN", "LSTM-P", "Tfm-P", "CPA-Rule", "Plan-CPA", "VO"]
ABLATION_METHODS: List[str] = ["abl_no_gating", "abl_no_gru", "abl_bce", "abl_telemetry_only", "abl_no_plan"]
TRAINED_MAIN_METHODS: List[str] = ["TR-GAT", "TR-GAT-NT", "GAT-S", "STGCN", "LSTM-P", "Tfm-P"]


def method_config(name: str, base: SkyFlowConfig) -> SkyFlowConfig:
    if name not in METHODS:
        raise KeyError(f"unknown method {name!r}; available: {sorted(METHODS)}")
    return METHODS[name].transform(copy.deepcopy(base))


def is_trained(name: str) -> bool:
    return METHODS[name].kind in ("trgat", "learned")
