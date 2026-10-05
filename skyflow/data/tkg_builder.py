"""Temporal Knowledge Graph construction — Algorithm 1 in the paper.

Constructs typed entity-relation-time graphs G_t = (V_t, E_t, R, τ)
from four parallel data streams:
  - ADS-B telemetry → UAV nodes with a per-UAV state vector
  - Flight plans → corridor reservation edges (shares_corridor)
  - Weather grid → downwind influence edges (is_downwind_of)
  - Restricted-zone registry → proximity edges (is_restricted_by)

Feature set (``leakage_free``):
  * True (default, 20 dims): [x,y,z, vx,vy,vz, ψ,ψ̇, ax,ay,az, b,ḃ, p, c_id,
    wx,wy,wz, σ_gps, n_nbr].  No CPA distance / CPA time / avoidance flag.
  * False (legacy, 23 dims): the above + [d_min, t_cpa, f_avoid].

Relation vocabulary (``leakage_free``):
  * True (default, 5 types): approaches, shares_corridor, is_downwind_of,
    has_reserved, is_restricted_by.
  * False (legacy, 6 types): the above + conflicts_with (CPA < 0.3·D_appr).

``approaches`` edges are still gated by a linear-CPA test, but the CPA
*values* never enter any feature.  Each edge carries an elapsed time δ that
feeds the sinusoidal temporal encoding φ(δ).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


ENTITY_TYPES = {"uav": 0, "sector": 1, "weather": 2, "restricted": 3}

LEGACY_RELATION_VOCAB: Dict[str, int] = {
    "approaches": 0,
    "conflicts_with": 1,
    "shares_corridor": 2,
    "is_downwind_of": 3,
    "has_reserved": 4,
    "is_restricted_by": 5,
}

LEAKAGE_FREE_RELATION_VOCAB: Dict[str, int] = {
    "approaches": 0,
    "shares_corridor": 1,
    "is_downwind_of": 2,
    "has_reserved": 3,
    "is_restricted_by": 4,
}

# Backward-compatible alias (legacy numbering).
RELATION_VOCAB = LEGACY_RELATION_VOCAB

LEGACY_UAV_FEATURES: List[str] = [
    "x", "y", "z", "vx", "vy", "vz", "psi", "psi_dot",
    "ax", "ay", "az", "b", "b_dot", "p", "c_id",
    "wx", "wy", "wz", "sigma_gps", "n_nbr",
    "d_min", "t_cpa", "f_avoid",
]
LEAKAGE_FREE_UAV_FEATURES: List[str] = LEGACY_UAV_FEATURES[:20]
LEAKING_FEATURES = ("d_min", "t_cpa", "f_avoid")
LEAKING_RELATIONS = ("conflicts_with",)


def relation_vocab(leakage_free: bool = True) -> Dict[str, int]:
    return dict(LEAKAGE_FREE_RELATION_VOCAB if leakage_free else LEGACY_RELATION_VOCAB)


def uav_feature_names(leakage_free: bool = True) -> List[str]:
    return list(LEAKAGE_FREE_UAV_FEATURES if leakage_free else LEGACY_UAV_FEATURES)


@dataclass
class AirspaceState:
    """Raw (observed) airspace state at a single epoch."""

    uav_positions: np.ndarray       # (N_uav, 3) xyz in meters
    uav_velocities: np.ndarray      # (N_uav, 3)
    uav_headings: np.ndarray        # (N_uav,) radians
    uav_battery: np.ndarray         # (N_uav,) fraction [0,1]
    uav_priority: np.ndarray        # (N_uav,) int class
    uav_avoiding: np.ndarray        # (N_uav,) bool
    sector_occupancy: np.ndarray    # (N_sec, 8)
    weather_cells: np.ndarray       # (N_wx, 12)
    restricted_zones: np.ndarray    # (N_rz, 8)
    corridor_reservations: List[Tuple[int, int, float, float]]
    epoch_time: float
    uav_heading_rates: Optional[np.ndarray] = None    # (N_uav,) rad/s
    uav_accelerations: Optional[np.ndarray] = None    # (N_uav, 3) m/s²
    uav_battery_rates: Optional[np.ndarray] = None    # (N_uav,) discharge rate
    uav_corridor_ids: Optional[np.ndarray] = None     # (N_uav,) corridor assignment
    uav_local_wind: Optional[np.ndarray] = None       # (N_uav, 3) local wind estimate
    uav_gps_dop: Optional[np.ndarray] = None          # (N_uav,) GPS dilution of precision
    # Age-of-information bookkeeping (None => everything is fresh, AoI = 0)
    uav_last_rx_time: Optional[np.ndarray] = None     # (N_uav,) timestamp of freshest report
    env_last_update_time: Optional[Dict[str, float]] = None  # per context source


DELTA_MODES = ("aoi", "legacy")
NEIGHBOR_SEARCH_MODES = ("grid", "bruteforce")
INPUT_SETS = ("full", "telemetry_only")   # telemetry_only: UAV nodes + approaches edges only
# Relations whose (undirected) UAV pairs form the pairwise-scoring candidate set.
CANDIDATE_RELATIONS = ("approaches", "shares_corridor")


@dataclass
class TKGSnapshot:
    """A single temporal knowledge graph snapshot ready for TR-GAT."""

    node_features: torch.Tensor
    node_types: torch.Tensor
    edge_indices: Dict[int, torch.Tensor]
    edge_deltas: Dict[int, torch.Tensor]
    num_uavs: int
    num_nodes: int

    conflict_pairs: Optional[torch.Tensor] = None
    conflict_labels: Optional[torch.Tensor] = None
    conflict_ttc: Optional[torch.Tensor] = None        # (P,) seconds; -1 for negatives
    uav_aoi: Optional[torch.Tensor] = None             # (N_uav,) age of information (s)
    relation_names: Optional[List[str]] = None
    feature_names: Optional[List[str]] = None
    # S4: candidate pairs (i<j) that carry an approaches/shares_corridor edge
    candidate_pairs: Optional[torch.Tensor] = None     # (2, C) long
    # positives that are not in conflict_pairs (counted as misses by metrics)
    num_missed_positives: int = 0
    missed_ttc: Optional[torch.Tensor] = None          # (M,) seconds
    # build statistics
    build_time_ms: float = 0.0
    num_pair_candidates: int = 0                       # pairs that went through the CPA test


class TKGBuilder:
    """Builds TKG snapshots from raw airspace state."""

    def __init__(
        self,
        approach_cpa_h: float = 80.0,
        approach_cpa_v: float = 15.0,
        approach_lookahead: float = 60.0,
        corridor_lookahead: float = 120.0,
        weather_radius: float = 400.0,
        feature_dim: Optional[int] = None,
        leakage_free: bool = True,
        delta_mode: str = "aoi",
        neighbor_search: str = "grid",
        input_set: str = "full",
    ):
        if delta_mode not in DELTA_MODES:
            raise ValueError(f"delta_mode must be one of {DELTA_MODES}, got {delta_mode!r}")
        if neighbor_search not in NEIGHBOR_SEARCH_MODES:
            raise ValueError(
                f"neighbor_search must be one of {NEIGHBOR_SEARCH_MODES}, got {neighbor_search!r}"
            )
        if input_set not in INPUT_SETS:
            raise ValueError(f"input_set must be one of {INPUT_SETS}, got {input_set!r}")
        self.input_set = input_set
        self.telemetry_only = input_set == "telemetry_only"
        self.approach_cpa_h = approach_cpa_h
        self.approach_cpa_v = approach_cpa_v
        self.approach_lookahead = approach_lookahead
        self.corridor_lookahead = corridor_lookahead
        self.weather_radius = weather_radius
        self.leakage_free = leakage_free
        self.delta_mode = delta_mode
        self.neighbor_search = neighbor_search
        self.last_num_candidates = 0

        self.relations = relation_vocab(leakage_free)
        self.relation_names = sorted(self.relations, key=self.relations.get)
        self.num_relations = len(self.relations)
        self.uav_features = uav_feature_names(leakage_free)
        self.feature_dim = feature_dim if feature_dim is not None else len(self.uav_features)
        if self.feature_dim < len(self.uav_features):
            raise ValueError(
                f"feature_dim={self.feature_dim} smaller than UAV feature set "
                f"({len(self.uav_features)})"
            )

        self._last_edge_times: Dict[str, Dict[Tuple[int, int], float]] = {
            r: {} for r in self.relations
        }
        self._aoi_ctx: Tuple[np.ndarray, Dict[str, float]] = (np.zeros(0, np.float32), {})

    # ------------------------------------------------------------------ #
    def build(
        self, state: AirspaceState, device: torch.device = torch.device("cpu")
    ) -> TKGSnapshot:
        """Construct a TKG snapshot from raw airspace state."""
        t_start = time.perf_counter()
        n_uav = state.uav_positions.shape[0]
        if self.telemetry_only:
            n_sec = n_wx = n_rz = 0
        else:
            n_sec = state.sector_occupancy.shape[0]
            n_wx = state.weather_cells.shape[0]
            n_rz = state.restricted_zones.shape[0]
        n_total = n_uav + n_sec + n_wx + n_rz

        node_features = self._build_node_features(state, n_total, n_uav, n_sec, n_wx, n_rz)
        node_types = self._build_node_types(n_uav, n_sec, n_wx, n_rz)

        uav_aoi = self._uav_aoi(state, n_uav)
        env_age = self._env_age(state)
        self._aoi_ctx = (uav_aoi, env_age)
        edge_indices, edge_deltas = self._build_edges(state, n_uav, n_sec, n_wx, n_rz)
        candidate_pairs = self._candidate_pairs(edge_indices)

        ei_tensors = {
            self.relations[r]: torch.tensor(edges, dtype=torch.long, device=device)
            for r, edges in edge_indices.items()
            if len(edges[0]) > 0
        }
        ed_tensors = {
            self.relations[r]: torch.tensor(deltas, dtype=torch.float32, device=device)
            for r, deltas in edge_deltas.items()
            if len(deltas) > 0
        }

        return TKGSnapshot(
            node_features=torch.tensor(node_features, dtype=torch.float32, device=device),
            node_types=torch.tensor(node_types, dtype=torch.long, device=device),
            edge_indices=ei_tensors,
            edge_deltas=ed_tensors,
            num_uavs=n_uav,
            num_nodes=n_total,
            uav_aoi=torch.tensor(uav_aoi, dtype=torch.float32, device=device),
            relation_names=list(self.relation_names),
            feature_names=list(self.uav_features),
            candidate_pairs=torch.tensor(candidate_pairs, dtype=torch.long, device=device),
            build_time_ms=(time.perf_counter() - t_start) * 1000.0,
            num_pair_candidates=self.last_num_candidates,
        )

    @staticmethod
    def _candidate_pairs(edge_indices: Dict[str, Tuple[List, List]]) -> np.ndarray:
        """Unique undirected UAV pairs (i<j) carrying a candidate relation, sorted."""
        src, dst = [], []
        for r in CANDIDATE_RELATIONS:
            if r in edge_indices:
                src.extend(edge_indices[r][0])
                dst.extend(edge_indices[r][1])
        if not src:
            return np.zeros((2, 0), dtype=np.int64)
        a = np.asarray(src, dtype=np.int64)
        b = np.asarray(dst, dtype=np.int64)
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        pairs = np.unique(np.stack([lo, hi], axis=1), axis=0)   # sorted lexicographically
        return pairs.T

    # ------------------------------------------------------------------ #
    # Age of information
    # ------------------------------------------------------------------ #
    @staticmethod
    def _uav_aoi(state: AirspaceState, n_uav: int) -> np.ndarray:
        """AoI_i = t - t_rx(i); zero when the state carries no reception times."""
        if state.uav_last_rx_time is None:
            return np.zeros(n_uav, dtype=np.float32)
        return np.maximum(state.epoch_time - state.uav_last_rx_time[:n_uav], 0.0).astype(np.float32)

    @staticmethod
    def _env_age(state: AirspaceState) -> Dict[str, float]:
        if not state.env_last_update_time:
            return {}
        return {k: max(state.epoch_time - v, 0.0) for k, v in state.env_last_update_time.items()}

    # ------------------------------------------------------------------ #
    def _build_node_features(
        self,
        state: AirspaceState,
        n_total: int,
        n_uav: int,
        n_sec: int,
        n_wx: int,
        n_rz: int,
    ) -> np.ndarray:
        """UAV feature vector (see module docstring for the two feature sets)."""
        feat = np.zeros((n_total, self.feature_dim), dtype=np.float32)

        heading_rates = state.uav_heading_rates if state.uav_heading_rates is not None else np.zeros(n_uav, dtype=np.float32)
        accelerations = state.uav_accelerations if state.uav_accelerations is not None else np.zeros((n_uav, 3), dtype=np.float32)
        battery_rates = state.uav_battery_rates if state.uav_battery_rates is not None else np.full(n_uav, -0.0001, dtype=np.float32)
        corridor_ids = state.uav_corridor_ids if state.uav_corridor_ids is not None else np.zeros(n_uav, dtype=np.float32)
        local_wind = state.uav_local_wind if state.uav_local_wind is not None else np.zeros((n_uav, 3), dtype=np.float32)
        gps_dop = state.uav_gps_dop if state.uav_gps_dop is not None else np.full(n_uav, 2.5, dtype=np.float32)

        if n_uav > 0:
            feat[:n_uav, 0:3] = state.uav_positions[:n_uav]          # x, y, z
            feat[:n_uav, 3:6] = state.uav_velocities[:n_uav]         # vx, vy, vz
            feat[:n_uav, 6] = state.uav_headings[:n_uav]             # ψ
            feat[:n_uav, 7] = heading_rates[:n_uav]                  # ψ̇
            feat[:n_uav, 8:11] = accelerations[:n_uav]               # ax, ay, az
            feat[:n_uav, 11] = state.uav_battery[:n_uav]             # b
            feat[:n_uav, 12] = battery_rates[:n_uav]                 # ḃ
            feat[:n_uav, 13] = state.uav_priority[:n_uav]            # p
            feat[:n_uav, 14] = corridor_ids[:n_uav]                  # c_id
            feat[:n_uav, 15:18] = local_wind[:n_uav]                 # wx, wy, wz
            feat[:n_uav, 18] = gps_dop[:n_uav]                       # σ_gps

            positions = state.uav_positions[:n_uav]
            diffs = positions[None, :, :] - positions[:, None, :]    # (N, N, 3)
            dists = np.sqrt(diffs[..., 0] ** 2 + diffs[..., 1] ** 2 + 1e-12)
            np.fill_diagonal(dists, np.inf)
            feat[:n_uav, 19] = (dists < self.approach_cpa_h).sum(axis=1)   # n_nbr

            if not self.leakage_free:
                nearest = np.argmin(dists, axis=1)
                feat[:n_uav, 20] = dists[np.arange(n_uav), nearest]      # d_min
                dp = diffs[np.arange(n_uav), nearest]                     # (N, 3)
                dv = state.uav_velocities[nearest] - state.uav_velocities[:n_uav]
                dvdv = (dv * dv).sum(axis=1)
                t_cpa = np.where(dvdv > 1e-8, -(dp * dv).sum(axis=1) / np.maximum(dvdv, 1e-8), 0.0)
                feat[:n_uav, 21] = np.maximum(t_cpa, 0.0)                 # t_cpa
                feat[:n_uav, 22] = state.uav_avoiding[:n_uav].astype(np.float32)  # f_avoid

        offset = n_uav
        for i in range(n_sec):
            sec = state.sector_occupancy[i]
            end = min(len(sec), self.feature_dim)
            feat[offset + i, :end] = sec[:end]

        offset += n_sec
        for i in range(n_wx):
            wx = state.weather_cells[i]
            end = min(len(wx), self.feature_dim)
            feat[offset + i, :end] = wx[:end]

        offset += n_wx
        for i in range(n_rz):
            rz = state.restricted_zones[i]
            end = min(len(rz), self.feature_dim)
            feat[offset + i, :end] = rz[:end]

        return feat

    def _build_node_types(
        self, n_uav: int, n_sec: int, n_wx: int, n_rz: int
    ) -> np.ndarray:
        types = np.concatenate([
            np.full(n_uav, ENTITY_TYPES["uav"]),
            np.full(n_sec, ENTITY_TYPES["sector"]),
            np.full(n_wx, ENTITY_TYPES["weather"]),
            np.full(n_rz, ENTITY_TYPES["restricted"]),
        ])
        return types

    # ------------------------------------------------------------------ #
    def _build_edges(
        self,
        state: AirspaceState,
        n_uav: int,
        n_sec: int,
        n_wx: int,
        n_rz: int,
    ) -> Tuple[Dict[str, Tuple[List, List]], Dict[str, List]]:
        edge_indices: Dict[str, Tuple[List, List]] = {r: ([], []) for r in self.relations}
        edge_deltas: Dict[str, List] = {r: [] for r in self.relations}
        t = state.epoch_time

        self._add_approach_edges(state, n_uav, t, edge_indices, edge_deltas)
        if not self.telemetry_only:
            self._add_corridor_edges(state, n_uav, t, edge_indices, edge_deltas)
            self._add_weather_edges(state, n_uav, n_sec, n_wx, t, edge_indices, edge_deltas)
            self._add_restriction_edges(state, n_uav, n_sec, n_wx, n_rz, t, edge_indices, edge_deltas)

        return edge_indices, edge_deltas

    def _edge_delta(self, relation: str, key: Tuple[int, int], t: float) -> float:
        """δ for a UAV–UAV edge.

        aoi    : δ_ij = t - min(t_rx(i), t_rx(j)) = max(AoI_i, AoI_j)
        legacy : elapsed time since this edge was last present in the graph
        """
        if self.delta_mode == "aoi":
            uav_aoi, _ = self._aoi_ctx
            i, j = key
            return float(max(uav_aoi[i], uav_aoi[j]))
        delta = t - self._last_edge_times[relation].get(key, t)
        self._last_edge_times[relation][key] = t
        return delta

    def _env_edge_delta(self, relation: str, key: Tuple[int, int], t: float, source: str) -> float:
        """δ for a UAV–context edge: age of the context source's last publish."""
        if self.delta_mode == "aoi":
            _, env_age = self._aoi_ctx
            return float(env_age.get(source, 0.0))
        delta = t - self._last_edge_times[relation].get(key, t)
        self._last_edge_times[relation][key] = t
        return delta

    # ------------------------------------------------------------------ #
    # approaches edges: candidate generation + vectorised CPA test
    # ------------------------------------------------------------------ #
    def candidate_radius(self, velocities: np.ndarray) -> float:
        """Conservative horizontal radius outside of which no approaches edge
        can exist.

        The gate is cpa_h < D_appr with t_cpa ∈ [0, T].  Since
        cpa_h ≥ h_dist − |dv_h|·t_cpa ≥ h_dist − 2·v_max·T, any pair with
        h_dist ≥ D_appr + 2·v_max·T fails the gate.  v_max is the largest
        observed speed in the current snapshot (the same velocities the gate
        uses), so the bound is exact for that snapshot.
        """
        if velocities.shape[0] == 0:
            return self.approach_cpa_h
        v_max = float(np.sqrt((velocities.astype(np.float64) ** 2).sum(axis=1)).max())
        return self.approach_cpa_h + 2.0 * v_max * self.approach_lookahead

    def pair_candidates(self, positions: np.ndarray, velocities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (ci, cj) with ci < cj, lexicographically sorted, according to
        ``self.neighbor_search``."""
        n = positions.shape[0]
        if n < 2:
            e = np.zeros(0, dtype=np.int64)
            return e, e
        if self.neighbor_search == "bruteforce":
            ci, cj = np.triu_indices(n, k=1)
            return ci.astype(np.int64), cj.astype(np.int64)
        return self._grid_candidates(positions, self.candidate_radius(velocities))

    @staticmethod
    def _grid_candidates(positions: np.ndarray, cell: float) -> Tuple[np.ndarray, np.ndarray]:
        """Spatial hash with cell edge = ``cell``; every pair with horizontal
        distance < cell lies in the same or an 8-adjacent cell."""
        cell = max(float(cell), 1e-3)
        cx = np.floor(positions[:, 0].astype(np.float64) / cell).astype(np.int64)
        cy = np.floor(positions[:, 1].astype(np.float64) / cell).astype(np.int64)
        cx -= cx.min()
        cy -= cy.min()
        ncy = int(cy.max()) + 2
        key = cx * ncy + cy
        order = np.argsort(key, kind="stable")
        key_sorted = key[order]
        uniq, start = np.unique(key_sorted, return_index=True)
        end = np.append(start[1:], len(key_sorted))
        members = {int(k): order[s:e] for k, s, e in zip(uniq, start, end)}

        src: List[np.ndarray] = []
        dst: List[np.ndarray] = []
        # Half-plane of neighbour offsets so each unordered cell pair is visited once.
        offsets = ((1, 0), (1, 1), (0, 1), (-1, 1))
        for k, idx in members.items():
            if idx.size > 1:
                a, b = np.triu_indices(idx.size, k=1)
                src.append(idx[a])
                dst.append(idx[b])
            kx, ky = divmod(k, ncy)
            for ox, oy in offsets:
                nk = (kx + ox) * ncy + (ky + oy)
                other = members.get(nk)
                if other is None or ky + oy < 0 or ky + oy >= ncy:
                    continue
                g = np.meshgrid(idx, other, indexing="ij")
                src.append(g[0].ravel())
                dst.append(g[1].ravel())
        if not src:
            e = np.zeros(0, dtype=np.int64)
            return e, e
        a = np.concatenate(src).astype(np.int64)
        b = np.concatenate(dst).astype(np.int64)
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        n = int(positions.shape[0])
        key = np.sort(lo * n + hi)              # lexicographic (i, j) order, as bruteforce
        return key // n, key % n

    def approach_test(
        self, positions: np.ndarray, velocities: np.ndarray, ci: np.ndarray, cj: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorised linear-CPA gate for candidate pairs.

        Returns (is_approach, cpa_h).  Semantics identical to the original
        per-pair loop: closing pairs are extrapolated to CPA clipped at
        [0, T]; non-closing pairs use their current separation."""
        P = positions.astype(np.float32)
        V = velocities.astype(np.float32)
        dp = P[cj] - P[ci]
        dv = V[cj] - V[ci]
        h_dist = np.sqrt(dp[:, 0] ** 2 + dp[:, 1] ** 2)
        v_dist = np.abs(dp[:, 2])
        speed_close = dp[:, 0] * dv[:, 0] + dp[:, 1] * dv[:, 1]
        dvdv = (dv * dv).sum(axis=1)
        use_cpa = (speed_close < 0) & (dvdv > 1e-8)
        t_cpa = np.where(
            use_cpa,
            np.clip(-(dp * dv).sum(axis=1) / np.where(dvdv > 1e-8, dvdv, 1.0), 0, self.approach_lookahead),
            0.0,
        ).astype(np.float32)
        cpa_pos = dp + dv * t_cpa[:, None]
        cpa_h = np.sqrt(cpa_pos[:, 0] ** 2 + cpa_pos[:, 1] ** 2)
        cpa_h = np.where(use_cpa, cpa_h, h_dist)
        is_approach = (cpa_h < self.approach_cpa_h) & (v_dist < self.approach_cpa_v)
        return is_approach, cpa_h

    def _add_approach_edges(self, state, n_uav, t, edge_indices, edge_deltas):
        r_approach = "approaches"
        r_conflict = "conflicts_with" if not self.leakage_free else None

        P = state.uav_positions[:n_uav]
        V = state.uav_velocities[:n_uav]
        ci, cj = self.pair_candidates(P, V)
        self.last_num_candidates = int(ci.size)
        if ci.size == 0:
            return
        is_approach, cpa_h = self.approach_test(P, V, ci, cj)
        hit = np.nonzero(is_approach)[0]
        for idx in hit:
            i, j = int(ci[idx]), int(cj[idx])
            delta = self._edge_delta(r_approach, (i, j), t)
            for src, dst in [(i, j), (j, i)]:
                edge_indices[r_approach][0].append(src)
                edge_indices[r_approach][1].append(dst)
                edge_deltas[r_approach].append(delta)

            if r_conflict is not None and cpa_h[idx] < self.approach_cpa_h * 0.3:
                delta_c = self._edge_delta(r_conflict, (i, j), t)
                for src, dst in [(i, j), (j, i)]:
                    edge_indices[r_conflict][0].append(src)
                    edge_indices[r_conflict][1].append(dst)
                    edge_deltas[r_conflict].append(delta_c)

    def _add_corridor_edges(self, state, n_uav, t, edge_indices, edge_deltas):
        r = "shares_corridor"
        corridor_map: Dict[int, List[int]] = {}

        for uav_a, uav_b, start_t, end_t in state.corridor_reservations:
            if t <= end_t and (t + self.corridor_lookahead) >= start_t:
                seg_id = hash((min(uav_a, uav_b), max(uav_a, uav_b)))
                corridor_map.setdefault(seg_id, [])
                if uav_a < n_uav:
                    corridor_map[seg_id].append(uav_a)
                if uav_b < n_uav:
                    corridor_map[seg_id].append(uav_b)

        for seg_uavs in corridor_map.values():
            uavs = list(set(seg_uavs))
            for a in range(len(uavs)):
                for b in range(a + 1, len(uavs)):
                    i, j = uavs[a], uavs[b]
                    delta = self._edge_delta(r, (i, j), t)
                    for src, dst in [(i, j), (j, i)]:
                        edge_indices[r][0].append(src)
                        edge_indices[r][1].append(dst)
                        edge_deltas[r].append(delta)

    def _add_weather_edges(self, state, n_uav, n_sec, n_wx, t, edge_indices, edge_deltas):
        r_wind = "is_downwind_of"
        wx_offset = n_uav + n_sec

        if n_wx == 0:
            return

        wx_positions = state.weather_cells[:, :3] if state.weather_cells.shape[1] >= 3 else None
        if wx_positions is None:
            return

        d = np.linalg.norm(
            state.uav_positions[:n_uav, None, :2] - wx_positions[None, :, :2], axis=-1
        )                                                           # (N, n_wx)
        for i, w in zip(*np.nonzero(d < self.weather_radius)):     # row-major: i outer
            wx_node = wx_offset + int(w)
            delta = self._env_edge_delta(r_wind, (int(i), wx_node), t, "weather")
            edge_indices[r_wind][0].append(wx_node)
            edge_indices[r_wind][1].append(int(i))
            edge_deltas[r_wind].append(delta)

    def _add_restriction_edges(self, state, n_uav, n_sec, n_wx, n_rz, t, edge_indices, edge_deltas):
        r = "is_restricted_by"
        rz_offset = n_uav + n_sec + n_wx

        if n_rz == 0 or n_uav == 0:
            return
        rz = state.restricted_zones
        radii = rz[:, 3] if rz.shape[1] > 3 else np.full(n_rz, 200.0, dtype=np.float32)
        d = np.linalg.norm(
            state.uav_positions[:n_uav, None, :2] - rz[None, :, :2], axis=-1
        )                                                           # (N, n_rz)
        for i, z in zip(*np.nonzero(d < radii[None, :] * 1.5)):
            rz_node = rz_offset + int(z)
            delta = self._env_edge_delta(r, (int(i), rz_node), t, "restricted")
            edge_indices[r][0].append(rz_node)
            edge_indices[r][1].append(int(i))
            edge_deltas[r].append(delta)

    def reset(self):
        """Clear cached edge timestamps between scenarios."""
        self._last_edge_times = {r: {} for r in self.relations}
