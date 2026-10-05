"""Baseline registry: one place that knows how to build every comparison
method from a :class:`SkyFlowConfig`, and whether it needs training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import torch

from skyflow.baselines.cpa_rule import CPARule
from skyflow.baselines.gat_static import GATStatic
from skyflow.baselines.lstm_pair import LSTMPair
from skyflow.baselines.stgcn import STGCN
from skyflow.baselines.transformer_pair import TransformerPair
from skyflow.baselines.velocity_obstacle import VelocityObstacle


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    factory: Callable[["SkyFlowConfig", torch.device], object]
    deterministic: bool          # no training, single run, predict() API
    description: str = ""


def _cpa_rule(cfg, device):
    bl = getattr(cfg, "baselines", None)
    mode = bl.cpa_rule_thresholds if bl is not None else "label"
    return CPARule(
        window_s=cfg.data.lookahead_seconds,
        h_thresh=cfg.data.conflict_h_sep_m,
        v_thresh=cfg.data.conflict_v_sep_m,
        threshold_mode=mode,
    )


REGISTRY: Dict[str, BaselineSpec] = {
    "CPA-Rule": BaselineSpec(
        "CPA-Rule", _cpa_rule, True,
        "Linear extrapolation of observed state over the label window; UTM-standard rule."),
    "VO": BaselineSpec(
        "VO", lambda cfg, device: VelocityObstacle(), True,
        "Reciprocal velocity obstacle score with 60 s look-ahead."),
    "LSTM-P": BaselineSpec(
        "LSTM-P", lambda cfg, device: LSTMPair(input_dim=cfg.uav_feature_dim()).to(device), False,
        "Pairwise LSTM over node features."),
    "Tfm-P": BaselineSpec(
        "Tfm-P", lambda cfg, device: TransformerPair(input_dim=cfg.uav_feature_dim()).to(device), False,
        "Pairwise Transformer over node features."),
    "STGCN": BaselineSpec(
        "STGCN", lambda cfg, device: STGCN(input_dim=cfg.uav_feature_dim()).to(device), False,
        "Spatio-temporal GCN."),
    "GAT-S": BaselineSpec(
        "GAT-S", lambda cfg, device: GATStatic(input_dim=cfg.uav_feature_dim()).to(device), False,
        "Static (non-temporal) GAT."),
}

BASELINE_ORDER: List[str] = ["CPA-Rule", "VO", "LSTM-P", "Tfm-P", "STGCN", "GAT-S"]


def get_baseline(name: str, cfg, device: torch.device):
    if name not in REGISTRY:
        raise KeyError(f"Unknown baseline {name!r}; available: {sorted(REGISTRY)}")
    return REGISTRY[name].factory(cfg, device)


def is_deterministic(name: str) -> bool:
    return REGISTRY[name].deterministic
