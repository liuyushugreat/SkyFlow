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

PAIR_EDGE_FEATURE_DIM = 7
POS_SCALE_M = 100.0
VEL_SCALE_MPS = 10.0
AOI_SCALE_S = 1.0


def build_pair_edge_features(
    node_features: torch.Tensor,
    pairs: torch.Tensor,
    uav_aoi: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute e_ij for each scored pair from observed kinematics.

    Args:
        node_features: (N, D) with columns 0:3 = position, 3:6 = velocity.
        pairs: (2, P) UAV index pairs.
        uav_aoi: (N_uav,) age of information per UAV in seconds, or None.
    Returns:
        (P, PAIR_EDGE_FEATURE_DIM) tensor.
    """
    i, j = pairs[0], pairs[1]
    dp = (node_features[j, 0:3] - node_features[i, 0:3]) / POS_SCALE_M
    dv = (node_features[j, 3:6] - node_features[i, 3:6]) / VEL_SCALE_MPS
    if uav_aoi is None:
        delta = torch.zeros(i.size(0), 1, device=node_features.device, dtype=node_features.dtype)
    else:
        delta = torch.maximum(uav_aoi[i], uav_aoi[j]).unsqueeze(-1) / AOI_SCALE_S
    return torch.cat([dp, dv, delta], dim=-1)


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
