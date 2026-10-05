"""Input standardisation for raw node features (S8 fix).

Raw node features mix metres (x, y up to 5000), m/s, fractions and flags; fed
straight into a Linear layer they made every learned model train very slowly
(val AUROC 0.72 after 2 epochs, predictions compressed into [0.09, 0.21]).

``InputStandardizer`` holds per-feature mean / std as *buffers*: fitted once
on the training split (``fit``), persisted in the checkpoint's state_dict,
applied identically at train / validation / test / robustness / scaling time.
With the default buffers (mean 0, std 1) it is the identity, i.e. the old
behaviour (``features.normalize_inputs: false`` simply skips ``fit``).
"""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn


class InputStandardizer(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, min_std: float = 1e-3):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))
        self.register_buffer("fitted", torch.tensor(False))
        self.eps = eps
        self.min_std = min_std

    @torch.no_grad()
    def fit(self, dataset: Iterable[Tuple[object, torch.Tensor]], max_snapshots: int = 400) -> "InputStandardizer":
        """Per-feature mean / std over node rows of up to ``max_snapshots``
        evenly spaced training snapshots (all node types, as the model sees them)."""
        data = list(dataset)
        if len(data) > max_snapshots:
            step = len(data) / max_snapshots
            data = [data[int(i * step)] for i in range(max_snapshots)]
        n = 0
        # accumulate on CPU in float64 regardless of where the model lives
        s = torch.zeros(self.mean.numel(), dtype=torch.float64)
        ss = torch.zeros(self.mean.numel(), dtype=torch.float64)
        for snapshot, _ in data:
            x = snapshot.node_features.detach().to("cpu", torch.float64)
            n += x.shape[0]
            s += x.sum(0)
            ss += (x * x).sum(0)
        if n == 0:
            return self
        mean = s / n
        var = (ss / n - mean * mean).clamp_min(0.0)
        std = var.sqrt()
        std = torch.where(std < self.min_std, torch.ones_like(std), std)   # constant features: leave scale
        self.mean.copy_(mean.to(self.mean.device, self.mean.dtype))
        self.std.copy_(std.to(self.std.device, self.std.dtype))
        self.fitted.fill_(True)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + self.eps)

    def extra_repr(self) -> str:
        return f"dim={self.mean.numel()}, fitted={bool(self.fitted)}"


def fit_input_norm(module: nn.Module, train_data, enabled: bool = True) -> bool:
    """Fit every InputStandardizer inside ``module`` on ``train_data``.
    Returns True if at least one was fitted."""
    if not enabled:
        return False
    done = False
    for m in module.modules():
        if isinstance(m, InputStandardizer):
            m.fit(train_data)
            done = True
    return done
