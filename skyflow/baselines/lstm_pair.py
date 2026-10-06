"""LSTM-Pair baseline.

Independent LSTM encoders on each aircraft's position-velocity history.
Conflict scores from concatenated final hidden states. Captures temporal
dynamics for individual trajectories but has no relational structure.
Parameter count matched to TR-GAT (~4.2M) for fair comparison.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from skyflow.data.tkg_builder import TKGSnapshot
from skyflow.models.conflict_head import pair_edge_feature_dim, pair_scorer_input, plan_column
from skyflow.models.input_norm import InputStandardizer


class LSTMPair(nn.Module):
    """LSTM-based pairwise conflict detector."""

    def __init__(
        self,
        input_dim: int = 23,
        hidden_dim: int = 192,
        num_layers: int = 2,
        dropout: float = 0.1,
        pair_edge_features: str = "geometry",
        window_s: float = 30.0,
    ):
        super().__init__()
        self.pair_edge_features = pair_edge_features
        self.window_s = float(window_s)
        self.input_norm = InputStandardizer(input_dim)
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2 + pair_edge_feature_dim(pair_edge_features), 256),
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
        """
        Args:
            snapshot: Current TKG snapshot.
            history: (N_uav, T, D) trajectory history or None (uses current).
        Returns:
            (P,) conflict probabilities.
        """
        n_uav = snapshot.num_uavs
        feats = snapshot.node_features[:n_uav]

        if history is None:
            history = feats.unsqueeze(1)

        # cuDNN is disabled for the RNN: the cuDNN dropout-state teardown of a
        # multi-layer LSTM aborts the process (0xC0000409) on Windows/CUDA 13
        # (torch 2.14).  History length is 1, so the native kernel costs nothing.
        with torch.backends.cudnn.flags(enabled=False):
            _, (h_n, _) = self.encoder(self.input_norm(history))
        embeddings = h_n[-1]

        pairs = snapshot.conflict_pairs
        if pairs is None or pairs.size(1) == 0:
            return torch.zeros(0, device=feats.device)

        x = pair_scorer_input(embeddings, snapshot.node_features, pairs, snapshot.uav_aoi,
                              self.pair_edge_features, self.window_s,
                              plan_col0=plan_column(getattr(snapshot, "feature_names", None)))
        return torch.sigmoid(self.scorer(x)).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
