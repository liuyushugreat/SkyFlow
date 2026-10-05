"""Transformer-Pair baseline.

Replaces the LSTM with a Transformer encoder using the same pairwise
scoring strategy. Enables parallelizable long-range dependency modeling
but treats each aircraft independently without relational structure.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from skyflow.data.tkg_builder import TKGSnapshot
from skyflow.models.conflict_head import pair_edge_feature_dim, pair_scorer_input
from skyflow.models.input_norm import InputStandardizer


class TransformerPair(nn.Module):
    """Transformer-based pairwise conflict detector."""

    def __init__(
        self,
        input_dim: int = 23,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        pair_edge_features: str = "geometry",
        window_s: float = 30.0,
    ):
        super().__init__()
        self.pair_edge_features = pair_edge_features
        self.window_s = float(window_s)
        self.input_norm = InputStandardizer(input_dim)
        self.input_proj = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.scorer = nn.Sequential(
            nn.Linear(embed_dim * 2 + pair_edge_feature_dim(pair_edge_features), 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        snapshot: TKGSnapshot,
        history: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        n_uav = snapshot.num_uavs
        feats = snapshot.node_features[:n_uav]

        if history is None:
            history = feats.unsqueeze(1)

        x = self.input_proj(self.input_norm(history))
        encoded = self.encoder(x)
        embeddings = encoded[:, -1, :]

        pairs = snapshot.conflict_pairs
        if pairs is None or pairs.size(1) == 0:
            return torch.zeros(0, device=feats.device)

        x = pair_scorer_input(embeddings, snapshot.node_features, pairs, snapshot.uav_aoi,
                              self.pair_edge_features, self.window_s)
        return torch.sigmoid(self.scorer(x)).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
