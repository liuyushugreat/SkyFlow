"""Pairwise conflict scoring head — Equation (6) in the paper.

Implements:
  p_ij = σ( MLP( [h_i^L ‖ h_j^L ‖ s_i ‖ s_j ‖ e_ij (‖ g_ij)] ) )

where h_i^L are final-layer TR-GAT embeddings (d=128), s_i are GRU
recurrent states (d_s=64), e_ij is the shared pair feature and g_ij the
optional intent-conformance-gated pair geometry (TR-GAT only, S8e).

Pair feature modes (``features.pair_edge_features``; every learned scorer
receives the same e_ij, computed from observed state and filed plans only):

  kinematics          7   [Δp/100 ‖ Δv/10 ‖ δ_ij]                        (legacy TR-GAT)
  geometry           12   + observed-state CPA geometry                    (S8c)
  geometry_plan      20   + planned pair separations from the filed plans  (S8d)
  geometry_plan_sync 22   same, but every report is first dead-reckoned to
                          the common epoch with its own age of information
                          (p_i' = p_i + v_i·AoI_i), and the two ages are
                          given separately                                  (S8e, default)

Nothing here uses the true trajectory or the label.  A pair is flagged as a
conflict when p_ij >= τ.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

PAIR_EDGE_FEATURE_DIM = 7          # "kinematics" mode (legacy TR-GAT)
PAIR_GEOMETRY_DIM = 5              # extra dims of the "geometry" mode (S8c)
PAIR_PLAN_DIM = 8                  # extra dims of the "geometry_plan" mode (S8d)
PAIR_SYNC_DIM = 2                  # extra dims of the "_sync" modes (S8e): AoI_i, AoI_j
PAIR_GATE_DIM = 9                  # intent-conformance-gated geometry (S8e): 3x(h,v) + min + w_i + w_j
GATE_RESIDUAL_DIM = 4              # per-UAV plan-conformance residual features fed to the gate
PLAN_HORIZONS_S = (10.0, 20.0, 30.0)
PLAN_FIRST_FEATURE = "pl10_dx"     # first of the 9 planned-position columns in the UAV features
POS_SCALE_M = 100.0
VEL_SCALE_MPS = 10.0
AOI_SCALE_S = 1.0
PAIR_FEATURE_MODES = ("none", "kinematics", "geometry", "geometry_plan", "geometry_plan_sync")


def pair_edge_feature_dim(mode: str = "geometry_plan_sync") -> int:
    """Width of e_ij for a ``features.pair_edge_features`` mode."""
    if mode not in PAIR_FEATURE_MODES:
        raise ValueError(f"pair_edge_features must be one of {PAIR_FEATURE_MODES}, got {mode!r}")
    return {"none": 0, "kinematics": PAIR_EDGE_FEATURE_DIM,
            "geometry": PAIR_EDGE_FEATURE_DIM + PAIR_GEOMETRY_DIM,
            "geometry_plan": PAIR_EDGE_FEATURE_DIM + PAIR_GEOMETRY_DIM + PAIR_PLAN_DIM,
            "geometry_plan_sync": PAIR_EDGE_FEATURE_DIM + PAIR_GEOMETRY_DIM + PAIR_PLAN_DIM + PAIR_SYNC_DIM}[mode]


def mode_uses_sync(mode: str) -> bool:
    return mode.endswith("_sync")


def plan_column(feature_names) -> Optional[int]:
    """Index of the first planned-position column (``pl10_dx``) or None."""
    names = list(feature_names or [])
    return names.index(PLAN_FIRST_FEATURE) if PLAN_FIRST_FEATURE in names else None


def node_ages(node_features: torch.Tensor, uav_aoi: Optional[torch.Tensor]) -> torch.Tensor:
    """(N,) age of information per node row (0 for non-UAV rows / no AoI)."""
    ages = torch.zeros(node_features.size(0), device=node_features.device, dtype=node_features.dtype)
    if uav_aoi is not None:
        n = min(uav_aoi.numel(), ages.numel())
        ages[:n] = uav_aoi[:n].to(ages.dtype)
    return ages


def synchronised_positions(node_features: torch.Tensor, uav_aoi: Optional[torch.Tensor],
                           enabled: bool) -> torch.Tensor:
    """Dead-reckon every report to the common (current) epoch: p' = p + v·AoI.

    With ADS-B latencies of 0.5–1.2 s (+3 s for non-cooperative targets) and
    speeds of 10–15 m/s, two reports of different age are misaligned by up to
    ~15 m - more than the 10 m separation threshold - if used as they are."""
    p = node_features[:, 0:3]
    if not enabled:
        return p
    v = node_features[:, 3:6]
    return p + v * node_ages(node_features, uav_aoi).unsqueeze(-1)


def planned_pair_geometry(
    node_features: torch.Tensor, pairs: torch.Tensor, plan_col0: int,
    horizons_s=PLAN_HORIZONS_S, positions: Optional[torch.Tensor] = None,
    plan_weight: Optional[torch.Tensor] = None, with_flag: bool = True,
) -> torch.Tensor:
    """Pair geometry along the *filed plans* (S8d): for each horizon the planned
    horizontal / vertical separation of the pair, the minimum planned
    horizontal separation over the horizons, and (``with_flag``) a flag that
    either UAV has no plan context (non-cooperative; such UAVs fall back to
    linear extrapolation of their observed velocity, as the Plan-CPA rule does).

    ``positions`` overrides the observed positions (e.g. AoI-synchronised).
    ``plan_weight`` (N,) in [0, 1] mixes the planned offset (1) with linear
    extrapolation (0) per UAV - the intent-conformance gate of S8e.
    Returns (P, 8) or (P, 7) without the flag."""
    i, j = pairs[0], pairs[1]
    p = node_features[:, 0:3] if positions is None else positions
    v = node_features[:, 3:6]
    last = node_features[:, plan_col0 + 3 * (len(horizons_s) - 1): plan_col0 + 3 * len(horizons_s)]
    no_plan = last.norm(dim=-1) < 1e-6                                  # (N,)
    if plan_weight is None:
        w = (~no_plan).to(node_features.dtype)
    else:
        w = plan_weight.to(node_features.dtype) * (~no_plan).to(node_features.dtype)
    w = w.unsqueeze(-1)
    cols = []
    for k, tau in enumerate(horizons_s):
        planned = node_features[:, plan_col0 + 3 * k: plan_col0 + 3 * k + 3]
        off = w * planned + (1.0 - w) * v * float(tau)
        q = p + off
        dq = q[j] - q[i]
        cols.append(dq[:, :2].norm(dim=-1) / POS_SCALE_M)
        cols.append(dq[:, 2].abs() / VEL_SCALE_MPS)
    dh = torch.stack(cols[0::2], dim=-1)
    cols.append(dh.min(dim=-1).values)
    if with_flag:
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
        mode: see module docstring.
        window_s: look-ahead window for t_cpa clipping.
        plan_col0: index of the ``pl10_dx`` column (required for the plan modes).
    Returns:
        (P, pair_edge_feature_dim(mode)) tensor, or None for mode "none".
    """
    if mode not in PAIR_FEATURE_MODES:
        raise ValueError(f"pair_edge_features must be one of {PAIR_FEATURE_MODES}, got {mode!r}")
    if mode == "none":
        return None
    if mode in ("geometry_plan", "geometry_plan_sync") and plan_col0 is None:
        raise ValueError(f"pair_edge_features={mode!r} needs plan-context features "
                         "(features.plan_context: true); use 'geometry' without them")
    i, j = pairs[0], pairs[1]
    sync = mode_uses_sync(mode)
    pos = synchronised_positions(node_features, uav_aoi, sync)
    dp_m = pos[j] - pos[i]
    dv_mps = node_features[j, 3:6] - node_features[i, 3:6]
    ages = node_ages(node_features, uav_aoi)
    delta = torch.maximum(ages[i], ages[j]).unsqueeze(-1) / AOI_SCALE_S
    parts = [dp_m / POS_SCALE_M, dv_mps / VEL_SCALE_MPS, delta]
    if mode in ("geometry", "geometry_plan", "geometry_plan_sync"):
        parts.append(observed_cpa_geometry(dp_m, dv_mps, window_s))
    if mode in ("geometry_plan", "geometry_plan_sync"):
        parts.append(planned_pair_geometry(node_features, pairs, plan_col0, positions=pos))
    if sync:
        parts.append(torch.stack([ages[i], ages[j]], dim=-1) / AOI_SCALE_S)
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


# --------------------------------------------------------------------------- #
# S8e: intent-conformance gate (TR-GAT only)
# --------------------------------------------------------------------------- #
def conformance_residuals(node_features: torch.Tensor, plan_col0: int) -> torch.Tensor:
    """Per-UAV disagreement between the filed plan and the observed motion,
    from the node features only: (N, 4) = [cos(heading, planned heading),
    (speed - planned speed)/10, (vz - planned vz)/3, no-plan flag].  The
    planned velocity is the mean velocity over the first 10 s of the filed
    route (pl10 / 10 s)."""
    v = node_features[:, 3:6]
    pl10 = node_features[:, plan_col0: plan_col0 + 3]
    vp = pl10 / float(PLAN_HORIZONS_S[0])
    no_plan = node_features[:, plan_col0 + 6: plan_col0 + 9].norm(dim=-1) < 1e-6
    vh, vph = v[:, :2], vp[:, :2]
    cos = (vh * vph).sum(-1) / (vh.norm(dim=-1) * vph.norm(dim=-1)).clamp_min(1e-6)
    dspeed = (vh.norm(dim=-1) - vph.norm(dim=-1)) / VEL_SCALE_MPS
    dvz = (v[:, 2] - vp[:, 2]) / 3.0
    keep = (~no_plan).to(node_features.dtype)
    return torch.stack([cos * keep, dspeed * keep, dvz * keep, no_plan.to(node_features.dtype)], dim=-1)


class ConformanceGate(nn.Module):
    """w_i = σ(MLP([s_i ‖ r_i])) ∈ (0,1): how much UAV *i* is expected to follow
    its filed plan over the look-ahead window, estimated from the recurrent
    state s_i (history over the K-snapshot window) and the current plan /
    telemetry residual r_i.  Forced to 0 for UAVs without a plan.  The pair
    geometry is then computed on the mixture  w·plan + (1-w)·extrapolation."""

    def __init__(self, recurrent_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(recurrent_dim + GATE_RESIDUAL_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, 2.0)          # start near "trust the plan" (w ≈ 0.88)

    def forward(self, rec_state: torch.Tensor, node_features: torch.Tensor, plan_col0: int) -> torch.Tensor:
        n = min(rec_state.size(0), node_features.size(0))
        r = conformance_residuals(node_features[:n], plan_col0)
        w = torch.sigmoid(self.net(torch.cat([rec_state[:n].to(r.dtype), r], dim=-1))).squeeze(-1)
        return w * (1.0 - r[:, 3])


def gated_pair_geometry(node_features: torch.Tensor, pairs: torch.Tensor, plan_col0: int,
                        plan_weight: torch.Tensor, uav_aoi: Optional[torch.Tensor],
                        sync: bool) -> torch.Tensor:
    """g_ij (P, 9): pair separations at +10/+20/+30 s and their horizontal
    minimum on the gated trajectories, plus the two gate values."""
    pos = synchronised_positions(node_features, uav_aoi, sync)
    w_full = torch.zeros(node_features.size(0), device=node_features.device, dtype=node_features.dtype)
    w_full[: plan_weight.numel()] = plan_weight.to(w_full.dtype)
    g = planned_pair_geometry(node_features, pairs, plan_col0, positions=pos,
                              plan_weight=w_full, with_flag=False)
    i, j = pairs[0], pairs[1]
    return torch.cat([g, w_full[i].unsqueeze(-1), w_full[j].unsqueeze(-1)], dim=-1)


class ConflictScoringHead(nn.Module):
    """2-layer MLP producing per-pair conflict probability (optionally with the
    intent-conformance gate of S8e)."""

    def __init__(
        self,
        embed_dim: int = 128,
        recurrent_dim: int = 64,
        edge_feature_dim: int = PAIR_EDGE_FEATURE_DIM,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        conformance_gate: bool = False,
    ):
        super().__init__()
        self.edge_feature_dim = edge_feature_dim
        self.conformance_gate = bool(conformance_gate)
        self.gate = ConformanceGate(recurrent_dim) if self.conformance_gate else None
        in_dim = 2 * embed_dim + 2 * recurrent_dim + edge_feature_dim + (PAIR_GATE_DIM if self.conformance_gate else 0)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.no_edge_token = nn.Parameter(torch.randn(edge_feature_dim) * 0.01)
        self.last_gate: Optional[torch.Tensor] = None      # (N_uav,) of the last forward, for analysis

    def gate_values(self, gate_ctx: Dict) -> torch.Tensor:
        return self.gate(gate_ctx["rec_state"], gate_ctx["node_features"], gate_ctx["plan_col0"])

    def forward(
        self,
        h_i: torch.Tensor,
        h_j: torch.Tensor,
        s_i: torch.Tensor,
        s_j: torch.Tensor,
        edge_feat: torch.Tensor | None = None,
        gate_ctx: Optional[Dict] = None,
    ) -> torch.Tensor:
        """
        Args:
            h_i, h_j: (P, embed_dim) embeddings for source/target UAVs.
            s_i, s_j: (P, recurrent_dim) recurrent states.
            edge_feat: (P, edge_feature_dim) or None → uses no-edge token.
            gate_ctx: required when the head was built with ``conformance_gate``:
                {"node_features", "pairs", "plan_col0", "uav_aoi", "rec_state", "sync"}.
        Returns:
            (P,) conflict probabilities in [0, 1].
        """
        P = h_i.size(0)
        if edge_feat is None:
            edge_feat = self.no_edge_token.unsqueeze(0).expand(P, -1)
        parts = [h_i, h_j, s_i, s_j, edge_feat]
        if self.conformance_gate:
            if gate_ctx is None or gate_ctx.get("plan_col0") is None:
                raise ValueError("conformance gate needs plan-context features and a gate_ctx")
            w = self.gate_values(gate_ctx)
            self.last_gate = w.detach()
            parts.append(gated_pair_geometry(gate_ctx["node_features"], gate_ctx["pairs"], gate_ctx["plan_col0"],
                                             w, gate_ctx.get("uav_aoi"), bool(gate_ctx.get("sync", False))))
        x = torch.cat(parts, dim=-1)
        return torch.sigmoid(self.net(x)).squeeze(-1)
