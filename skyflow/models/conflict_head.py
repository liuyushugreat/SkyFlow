"""Pairwise conflict scoring head — Equation (6) in the paper.

Implements:
  p_ij = σ( MLP( [h_i^L ‖ h_j^L ‖ s_i ‖ s_j ‖ e_ij] ) )

where h_i^L are final-layer TR-GAT embeddings (d=128), s_i are GRU
recurrent states (d_s=64), and e_ij is a pair edge feature.

Leakage-free e_ij (default, 7 dims):
  e_ij = [ Δp / 100 m  (3) ‖ Δv / 10 m/s  (3) ‖ δ_ij / 1 s  (1) ]
with Δp, Δv the *observed* relative position / velocity and δ_ij the pair
age of information (max of the two UAVs' AoI; 0 when AoI is unavailable).
It contains no CPA distance, CPA time or threshold decision.

A pair is flagged as a conflict when p_ij >= τ.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

PAIR_EDGE_FEATURE_DIM = 7          # "kinematics" mode (legacy TR-GAT)
PAIR_GEOMETRY_DIM = 5              # extra dims of the "geometry" mode (S8c)
PAIR_PLAN_DIM = 8                  # extra dims of the "geometry_plan" mode (S8d)
PLAN_HORIZONS_S = (10.0, 20.0, 30.0)
PLAN_FIRST_FEATURE = "pl10_dx"     # first of the 9 planned-position columns in the UAV features
POS_SCALE_M = 100.0
VEL_SCALE_MPS = 10.0
AOI_SCALE_S = 1.0
PAIR_FEATURE_MODES = ("none", "kinematics", "geometry", "geometry_plan")


def pair_edge_feature_dim(mode: str = "geometry_plan") -> int:
    """Width of e_ij for a ``features.pair_edge_features`` mode."""
    if mode not in PAIR_FEATURE_MODES:
        raise ValueError(f"pair_edge_features must be one of {PAIR_FEATURE_MODES}, got {mode!r}")
    return {"none": 0, "kinematics": PAIR_EDGE_FEATURE_DIM,
            "geometry": PAIR_EDGE_FEATURE_DIM + PAIR_GEOMETRY_DIM,
            "geometry_plan": PAIR_EDGE_FEATURE_DIM + PAIR_GEOMETRY_DIM + PAIR_PLAN_DIM}[mode]


def plan_column(feature_names) -> Optional[int]:
    """Index of the first planned-position column (``pl10_dx``) or None."""
    names = list(feature_names or [])
    return names.index(PLAN_FIRST_FEATURE) if PLAN_FIRST_FEATURE in names else None


def planned_pair_geometry(
    node_features: torch.Tensor, pairs: torch.Tensor, plan_col0: int,
    horizons_s=PLAN_HORIZONS_S,
) -> torch.Tensor:
    """Pair geometry along the *filed plans* (S8d): for each horizon the planned
    horizontal / vertical separation of the pair, the minimum planned
    horizontal separation over the horizons, and a flag that either UAV has no
    plan context (non-cooperative; such UAVs fall back to linear extrapolation
    of their observed velocity, as the Plan-CPA rule does).  Returns (P, 8)."""
    i, j = pairs[0], pairs[1]
    p = node_features[:, 0:3]
    v = node_features[:, 3:6]
    last = node_features[:, plan_col0 + 3 * (len(horizons_s) - 1): plan_col0 + 3 * len(horizons_s)]
    no_plan = last.norm(dim=-1) < 1e-6                                  # (N,)
    cols = []
    for k, tau in enumerate(horizons_s):
        off = node_features[:, plan_col0 + 3 * k: plan_col0 + 3 * k + 3]
        off = torch.where(no_plan.unsqueeze(-1), v * float(tau), off)
        q = p + off
        dq = q[j] - q[i]
        cols.append(dq[:, :2].norm(dim=-1) / POS_SCALE_M)
        cols.append(dq[:, 2].abs() / VEL_SCALE_MPS)
    dh = torch.stack(cols[0::2], dim=-1)
    cols.append(dh.min(dim=-1).values)
    cols.append((no_plan[i] | no_plan[j]).to(node_features.dtype))
    return torch.stack(cols, dim=-1)


def observed_cpa_geometry(dp_m: torch.Tensor, dv_mps: torch.Tensor, window_s: float) -> torch.Tensor:
    """Linear-extrapolation closest-point-of-approach geometry of a pair from
    *observed* relative position / velocity (metres, m/s) - exactly the inputs
    of the CPA rule, no label information.

    Returns (P, 5): [t_cpa / T, d_cpa_h / 100 m, |dz at t_cpa| / 10 m,
                     |dp_h| / 100 m, closing speed / 10 m/s]."""
    dph, dvh = dp_m[:, :2], dv_mps[:, :2]
    v2 = (dvh * dvh).sum(-1)
    t_cpa = torch.where(v2 > 1e-9, -(dph * dvh).sum(-1) / v2.clamp_min(1e-9), torch.zeros_like(v2))
    t_cpa = t_cpa.clamp(0.0, float(window_s))
    d_cpa_h = (dph + dvh * t_cpa.unsqueeze(-1)).norm(dim=-1)
    dz_cpa = (dp_m[:, 2] + dv_mps[:, 2] * t_cpa).abs()
    range_h = dph.norm(dim=-1)
    closing = -(dph * dvh).sum(-1) / range_h.clamp_min(1e-6)        # > 0 when converging
    return torch.stack([t_cpa / float(window_s), d_cpa_h / POS_SCALE_M, dz_cpa / VEL_SCALE_MPS,
                        range_h / POS_SCALE_M, closing / VEL_SCALE_MPS], dim=-1)


def build_pair_edge_features(
    node_features: torch.Tensor,
    pairs: torch.Tensor,
    uav_aoi: Optional[torch.Tensor] = None,
    mode: str = "kinematics",
    window_s: float = 30.0,
    plan_col0: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """Compute e_ij for each scored pair from observed kinematics (and filed plans).

    Args:
        node_features: (N, D) with columns 0:3 = position, 3:6 = velocity
            (raw metres / m/s, i.e. *before* any model-side standardisation).
        pairs: (2, P) UAV index pairs.
        uav_aoi: (N_uav,) age of information per UAV in seconds, or None.
        mode: "none" -> None (scorer sees node embeddings only; legacy baselines),
              "kinematics" -> [dp/100, dv/10, delta] (7; legacy TR-GAT),
              "geometry" -> kinematics + observed CPA geometry (12; S8c),
              "geometry_plan" -> geometry + planned pair separations (20; S8d default).
        window_s: look-ahead window for t_cpa clipping.
        plan_col0: index of the ``pl10_dx`` column (required for "geometry_plan").
    Returns:
        (P, pair_edge_feature_dim(mode)) tensor, or None for mode "none".
    """
    if mode not in PAIR_FEATURE_MODES:
        raise ValueError(f"pair_edge_features must be one of {PAIR_FEATURE_MODES}, got {mode!r}")
    if mode == "none":
        return None
    if mode == "geometry_plan" and plan_col0 is None:
        raise ValueError("pair_edge_features='geometry_plan' needs plan-context features "
                         "(features.plan_context: true); use 'geometry' without them")
    i, j = pairs[0], pairs[1]
    dp_m = node_features[j, 0:3] - node_features[i, 0:3]
    dv_mps = node_features[j, 3:6] - node_features[i, 3:6]
    if uav_aoi is None:
        delta = torch.zeros(i.size(0), 1, device=node_features.device, dtype=node_features.dtype)
    else:
        delta = torch.maximum(uav_aoi[i], uav_aoi[j]).unsqueeze(-1) / AOI_SCALE_S
    parts = [dp_m / POS_SCALE_M, dv_mps / VEL_SCALE_MPS, delta]
    if mode in ("geometry", "geometry_plan"):
        parts.append(observed_cpa_geometry(dp_m, dv_mps, window_s))
    if mode == "geometry_plan":
        parts.append(planned_pair_geometry(node_features, pairs, plan_col0))
    return torch.cat(parts, dim=-1)


def pair_scorer_input(
    embeddings: torch.Tensor,
    raw_node_features: torch.Tensor,
    pairs: torch.Tensor,
    uav_aoi: Optional[torch.Tensor],
    mode: str,
    window_s: float,
    plan_col0: Optional[int] = None,
) -> torch.Tensor:
    """[h_i ‖ h_j ‖ e_ij] for the baselines' pair scorers (same e_ij as TR-GAT).

    ``raw_node_features`` must be the un-standardised snapshot features
    (metres / m/s), i.e. ``snapshot.node_features`` before any ``InputStandardizer``.
    """
    parts = [embeddings[pairs[0]], embeddings[pairs[1]]]
    e = build_pair_edge_features(raw_node_features, pairs, uav_aoi, mode=mode, window_s=window_s,
                                 plan_col0=plan_col0)
    if e is not None:
        parts.append(e)
    return torch.cat(parts, dim=-1)


class ConflictScoringHead(nn.Module):
    """2-layer MLP producing per-pair conflict probability."""

    def __init__(
        self,
        embed_dim: int = 128,
        recurrent_dim: int = 64,
        edge_feature_dim: int = PAIR_EDGE_FEATURE_DIM,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.edge_feature_dim = edge_feature_dim
        in_dim = 2 * embed_dim + 2 * recurrent_dim + edge_feature_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.no_edge_token = nn.Parameter(torch.randn(edge_feature_dim) * 0.01)

    def forward(
        self,
        h_i: torch.Tensor,
        h_j: torch.Tensor,
        s_i: torch.Tensor,
        s_j: torch.Tensor,
        edge_feat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            h_i, h_j: (P, embed_dim) embeddings for source/target UAVs.
            s_i, s_j: (P, recurrent_dim) recurrent states.
            edge_feat: (P, edge_feature_dim) or None → uses no-edge token.
        Returns:
            (P,) conflict probabilities in [0, 1].
        """
        P = h_i.size(0)
        if edge_feat is None:
            edge_feat = self.no_edge_token.unsqueeze(0).expand(P, -1)
        x = torch.cat([h_i, h_j, s_i, s_j, edge_feat], dim=-1)
        return torch.sigmoid(self.net(x)).squeeze(-1)
