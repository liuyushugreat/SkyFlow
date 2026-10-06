"""UrbanAir-500: simulation benchmark for dense low-altitude conflict detection.

Pipeline (one scenario):

    run_physics(plans)  ->  TruthLog          ground-truth kinematics for every epoch
    observe(log, e)     ->  AirspaceState     what the edge node sees at epoch e
    label(log, e)       ->  [ConflictEvent]   ground truth derived from the *future* truth

Labels (``label_mode``):
  * ``"lookahead"`` (default): pair (i, j) is positive at epoch e if, at any
    epoch k in [e, e + lookahead_s], the true horizontal separation is below
    ``conflict_h_sep`` AND the true vertical separation is below
    ``conflict_v_sep``.  ``time_to_conflict`` is the first such k minus e.
    The physics is run ``lookahead_s`` beyond the yielded horizon so every
    labelled epoch has a full look-ahead window.
  * ``"instantaneous"`` (legacy): positive iff the separation minima are
    violated at epoch e itself.

Observation (``observation_model``):
  * ``"adsb"`` (default): the edge node sees each UAV through its ADS-B
    reports.  UAV *i* has a per-scenario link latency L_i ~ U(lat_lo, lat_hi);
    the report generated at epoch g arrives at g + L_i and is lost with
    probability ``packet_loss``.  The receiver keeps the freshest received
    report, so the observed kinematics of *i* at epoch e are the true
    kinematics at generation epoch g*(i) = max{g <= e - L_i : not lost},
    plus GPS noise (horizontal sigma = CEP / 1.1774, vertical 1.5x) that is
    fixed per report.  ``uav_last_rx_time[i] = g*(i) * dt`` is the report
    timestamp, so the age of information is t - uav_last_rx_time.  Before
    the first report arrives the filed departure state (epoch 0) is used.
    Context sources are published periodically (``weather_update_s``,
    ``registry_update_s``, ``corridor_update_s``) and expose their last
    publish time via ``env_last_update_time``.
  * ``"legacy"``: observed state == true state at the same epoch (GPS walk
    noise folded into the truth, as in the original implementation).

All observation randomness is pre-drawn into the TruthLog as unit variates,
so ``observe(log, e, params)`` is a pure function and observation conditions
(latency, loss, CEP) can be changed at evaluation time without re-simulating.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from skyflow.data.tkg_builder import AirspaceState, TKGBuilder, TKGSnapshot

SPLIT_IDS = {"train": 0, "val": 1, "test": 2}
LABEL_MODES = ("lookahead", "instantaneous")
OBSERVATION_MODELS = ("adsb", "legacy")
CANDIDATE_MODES = ("proximity", "edges", "all", "sampled")
DENSITY_PRESETS = ("dense", "legacy")
# "dense": layered airspace (4 altitude layers), hub-converging waypoints,
#          tighter start area -> positives become a usable fraction.
# "legacy": original UrbanAir-500 random plans (positives ~0 at 5 km / N=500).
DENSE_ALTITUDE_LAYERS_M = (60.0, 80.0, 100.0, 120.0)
DENSE_LAYER_JITTER_M = 1.0
DENSE_HUB_PROB = 0.7          # probability that the next waypoint is a corridor hub
DENSE_HUB_CHOICES = 3         # pick among the k nearest hubs
DENSE_START_AREA_PER_UAV = 5.0  # effective start square side = N * this (m), capped by grid
CEP_TO_SIGMA = 1.0 / 1.1774          # 2-D Gaussian: CEP = 1.1774 sigma
VERTICAL_SIGMA_FACTOR = 1.5

# --- Conflict causes (synthetic injection; see docs) -------------------------
# Each UAV carries one cause.  A conflict's cause is the highest-ranked cause
# among its two UAVs; two ordinary UAVs give "planned_crossing".
CAUSES = ("planned_crossing", "wind_deviation", "nonconforming", "priority_insertion", "noncooperative")
CAUSE_CODES = {c: i for i, c in enumerate(CAUSES)}
CAUSE_RANK = {"planned_crossing": 0, "wind_deviation": 1, "nonconforming": 2,
              "priority_insertion": 3, "noncooperative": 4}
DEFAULT_CAUSE_MIX: Dict[str, float] = {
    "planned_crossing": 0.80, "nonconforming": 0.08, "wind_deviation": 0.06,
    "priority_insertion": 0.03, "noncooperative": 0.03,
}
WIND_DEVIATION_DRIFT_MPS = (2.0, 4.0)     # persistent cross-track drift of a wind-deviating UAV
NONCONFORMING_OFFSET_M = (100.0, 300.0)   # lateral offset applied to a leg's target
NONCONFORMING_LEG_PROB = 0.6              # probability that a given leg deviates
PRIORITY_INSERTION_START_S = (10.0, 30.0)
PRIORITY_INSERTION_SPEED_MPS = (20.0, 25.0)
NONCOOP_EXTRA_LATENCY_S = 3.0             # surveillance-sensor latency added for non-cooperative UAVs
NONCOOP_NOISE_FACTOR = 4.0                # and coarser position estimates


def _latency_range(value: Union[float, Sequence[float]]) -> Tuple[float, float]:
    if isinstance(value, (int, float)):
        return float(value), float(value)
    lo, hi = value
    return float(lo), float(hi)


@dataclass
class ObservationParams:
    """Observation conditions that can be varied at evaluation time."""
    adsb_latency_s: Union[float, Tuple[float, float]] = (0.5, 1.2)
    packet_loss: float = 0.0
    gps_cep_m: float = 2.5
    weather_update_s: float = 5.0
    registry_update_s: float = 10.0
    corridor_update_s: float = 1.0

    def latency_range(self) -> Tuple[float, float]:
        return _latency_range(self.adsb_latency_s)


def scenario_seed(base_seed: int, split: str, scenario_idx: int) -> int:
    """Deterministic per-scenario seed (replaces the PYTHONHASHSEED-dependent
    ``hash((split, idx))`` used previously)."""
    sid = SPLIT_IDS.get(split, 7)
    return (int(base_seed) * 1_000_003 + sid * 10_007 + int(scenario_idx)) % (2**31 - 1)


@dataclass
class UAVFlightPlan:
    uav_id: int
    waypoints: np.ndarray       # (W, 3) sequence of [x, y, z]  (as filed)
    priority: int               # 0=low, 1=normal, 2=high, 3=emergency
    cruise_speed: float         # m/s
    start_time: float           # seconds
    cause: str = "planned_crossing"
    deviations: Optional[np.ndarray] = None   # (W, 3) flown-target offsets (nonconforming)
    wind_drift: Optional[np.ndarray] = None   # (3,) persistent drift m/s (wind_deviation)
    cooperative: bool = True                  # False -> no ADS-B (noncooperative)


@dataclass
class ConflictEvent:
    uav_i: int
    uav_j: int
    epoch: int
    time_to_conflict: float
    min_separation_h: float
    min_separation_v: float
    cause: str = "planned_crossing"


@dataclass
class TruthLog:
    """Ground-truth kinematic log of one scenario.

    Arrays are indexed ``[epoch, uav, ...]`` and cover ``n_total`` epochs,
    of which only the first ``n_epochs`` are yielded to consumers; the tail
    exists so that look-ahead labels near the end of the horizon are exact.
    """

    dt: float
    n_epochs: int
    n_total: int
    positions: np.ndarray        # (n_total, N, 3)
    velocities: np.ndarray       # (n_total, N, 3)
    headings: np.ndarray         # (n_total, N)
    heading_rates: np.ndarray    # (n_total, N)
    accelerations: np.ndarray    # (n_total, N, 3)
    battery: np.ndarray          # (n_total, N)
    battery_rates: np.ndarray    # (n_total, N)
    priorities: np.ndarray       # (N,)
    wind: np.ndarray             # (n_total, 3)
    local_wind_noise: np.ndarray # (n_total, N, 3)
    gps_dop: np.ndarray          # (n_total, N)
    corridor_pairs: List[Tuple[int, int]]
    weather_gain: Optional[np.ndarray] = None   # (n_total, N_wx) wind scaling per cell
    weather_vis: Optional[np.ndarray] = None    # (n_total, N_wx) visibility (m)
    # Unit variates for the observation layer (pure-function observe()).
    latency_u: Optional[np.ndarray] = None      # (N,) U(0,1) -> per-UAV link latency
    loss_u: Optional[np.ndarray] = None         # (n_total, N) U(0,1) -> lost iff u < p
    gps_noise_unit: Optional[np.ndarray] = None # (n_total, N, 3) N(0,1) per report
    # Static infrastructure of the scenario (restricted zones etc.) so that a
    # log can be observed later without re-generating the simulator state.
    infrastructure: Optional[Dict[str, np.ndarray]] = None
    scenario_id: Optional[Tuple[str, int]] = None   # (split, index)
    causes: Optional[np.ndarray] = None             # (N,) int codes into CAUSES
    cooperative: Optional[np.ndarray] = None        # (N,) bool
    # Filed flight plans (S8d plan context). Waypoints as *filed* (no flown
    # deviations), padded by repeating the last waypoint to a common length.
    plan_waypoints: Optional[np.ndarray] = None     # (N, W_max, 3)
    plan_num_waypoints: Optional[np.ndarray] = None # (N,) int
    plan_cruise_speed: Optional[np.ndarray] = None  # (N,) m/s
    plan_start_time: Optional[np.ndarray] = None    # (N,) filed departure time (s)

    def cause_of_pair(self, i: int, j: int) -> str:
        if self.causes is None:
            return "planned_crossing"
        ci, cj = CAUSES[int(self.causes[i])], CAUSES[int(self.causes[j])]
        return ci if CAUSE_RANK[ci] >= CAUSE_RANK[cj] else cj

    @property
    def num_uavs(self) -> int:
        return self.positions.shape[1]


PLAN_HORIZONS_S: Tuple[float, ...] = (10.0, 20.0, 30.0)


def pack_filed_plans(plans: List["UAVFlightPlan"], n_uav: int):
    """Pad the filed waypoint lists of ``plans`` into (N, W_max, 3) by
    repeating each plan's last waypoint; also return counts, cruise speeds
    and filed departure times."""
    w_max = max(2, max((len(p.waypoints) for p in plans), default=2))
    wps = np.zeros((n_uav, w_max, 3), np.float32)
    counts = np.ones(n_uav, np.int32)
    speeds = np.zeros(n_uav, np.float32)
    starts = np.zeros(n_uav, np.float32)
    for p in plans:
        w = np.asarray(p.waypoints, np.float32)
        k = len(w)
        wps[p.uav_id, :k] = w
        wps[p.uav_id, k:] = w[-1]
        counts[p.uav_id] = k
        speeds[p.uav_id] = p.cruise_speed
        starts[p.uav_id] = p.start_time
    return wps, counts, speeds, starts


def filed_plan_context(
    log: "TruthLog",
    positions: np.ndarray,
    epoch_time: float = 0.0,
    velocities: Optional[np.ndarray] = None,
    horizons_s: Sequence[float] = PLAN_HORIZONS_S,
) -> Optional[np.ndarray]:
    """Decision-time flight-plan context for every UAV (S8d).

    Inputs are only what a UTM node legitimately holds: the *filed* route and
    cruise speed, plus the *observed* position (and heading, used to
    disambiguate the active route segment when a route revisits a hub).  The
    planned path is the polyline [observed position -> next filed waypoint ->
    following waypoints] (the way a waypoint-following controller steers),
    flown at the filed cruise speed.  Returns (N, 3 + 3·len(horizons)) =
    [next waypoint - p (3), planned(+τ_k) - p (3 each)] in metres.
    Non-cooperative UAVs (no filed plan at the UTM node) get all-zero
    context.  No truth state and no labels are used.  (The simulator flies
    every UAV from t = 0; ``plan.start_time`` only gates steering, so no
    departure delay is modelled here.)
    """
    if log.plan_waypoints is None:
        return None
    W = log.plan_waypoints.astype(np.float64)                      # (N, M, 3)
    N, M, _ = W.shape
    p = positions.astype(np.float64)                               # (N, 3)
    a, b = W[:, :-1, :], W[:, 1:, :]                               # segments (N, M-1, 3)
    ab = b - a
    seg_len = np.linalg.norm(ab, axis=-1)                          # (N, M-1)
    valid = seg_len > 1e-6
    ap = p[:, None, :] - a
    t = np.einsum("nmk,nmk->nm", ap, ab) / np.maximum(seg_len ** 2, 1e-9)
    t = np.where(valid, np.clip(t, 0.0, 1.0), 0.0)
    foot = a + t[..., None] * ab                                   # (N, M-1, 3)
    dist = np.linalg.norm(p[:, None, :] - foot, axis=-1)           # distance to each route segment
    dist = np.where(valid, dist, np.inf)
    if velocities is not None:
        # The active segment is the nearest one whose end waypoint lies ahead of
        # the observed heading (a waypoint-following controller flies straight at
        # its target; this also covers overshoot loops back to a missed waypoint
        # and routes that revisit a hub).  Segments whose target is behind the
        # UAV are excluded; a small bias prefers the better-aligned target.
        v = velocities.astype(np.float64)
        vh = np.linalg.norm(v[:, :2], axis=1)
        moving = vh > 0.5
        to_wp = b - p[:, None, :]                                  # (N, M-1, 3) bearing to each segment's target
        d_wp = np.linalg.norm(to_wp[..., :2], axis=-1)
        cos = np.einsum("nk,nmk->nm", v[:, :2], to_wp[..., :2]) / (
            np.maximum(vh, 1e-9)[:, None] * np.maximum(d_wp, 1e-9))
        behind = moving[:, None] & (cos < 0.0) & (d_wp > 5.0)
        dist = np.where(behind, dist + 1e4, dist - 5.0 * np.where(moving[:, None], cos, 0.0))
    no_valid = ~np.isfinite(dist).any(axis=1)
    dist[no_valid, 0] = 0.0
    k0 = np.argmin(dist, axis=1)                                   # active segment (N,)
    rows = np.arange(N)
    nxt = b[rows, k0]                                              # next filed waypoint (N, 3)

    # planned polyline: p -> nxt -> following filed waypoints
    first_len = np.linalg.norm(nxt - p, axis=1)
    rest_len = np.where(np.arange(M - 1)[None, :] > k0[:, None], seg_len, 0.0)     # (N, M-1)
    cum = np.concatenate([np.zeros((N, 1)), first_len[:, None], first_len[:, None] + np.cumsum(rest_len, axis=1)], axis=1)
    total = cum[:, -1]
    speed = log.plan_cruise_speed.astype(np.float64) if log.plan_cruise_speed is not None else np.zeros(N)

    out = np.zeros((N, 3 + 3 * len(horizons_s)), np.float64)
    out[:, 0:3] = nxt - p
    knots = np.concatenate([p[:, None, :], nxt[:, None, :], b], axis=1)          # (N, M+1, 3); knot j -> cum[:, j]
    for h, tau in enumerate(horizons_s):
        s = np.minimum(speed * float(tau), total)
        j = np.clip((s[:, None] >= cum[:, 1:]).sum(axis=1), 0, M - 1)             # knot index of segment start
        seg = knots[rows, j + 1] - knots[rows, j]
        L = cum[rows, j + 1] - cum[rows, j]
        frac = np.where(L > 1e-6, np.clip((s - cum[rows, j]) / np.maximum(L, 1e-9), 0.0, 1.0), 0.0)
        q = knots[rows, j] + frac[:, None] * seg
        out[:, 3 + 3 * h: 6 + 3 * h] = q - p
    if log.cooperative is not None:
        out[~log.cooperative] = 0.0
    return out.astype(np.float32)


class UrbanAir500:
    """Procedural UrbanAir-500 benchmark generator."""

    def __init__(
        self,
        num_uavs: int = 500,
        grid_size: float = 5000.0,
        altitude_range: Tuple[float, float] = (30.0, 150.0),
        num_corridors: int = 24,
        num_sectors: int = 64,
        num_weather_cells: int = 36,
        num_restricted_zones: int = 12,
        dt: float = 0.1,
        wind_std: float = 2.0,
        gps_cep: float = 2.5,
        adsb_latency_range: Tuple[float, float] = (0.5, 1.2),
        conflict_h_sep: float = 10.0,
        conflict_v_sep: float = 3.0,
        seed: int = 42,
        label_mode: str = "lookahead",
        lookahead_s: float = 30.0,
        observation_model: str = "adsb",
        packet_loss: float = 0.0,
        weather_update_s: float = 5.0,
        registry_update_s: float = 10.0,
        corridor_update_s: float = 1.0,
        density_preset: str = "dense",
        cause_mix: Optional[Dict[str, float]] = "default",
    ):
        if label_mode not in LABEL_MODES:
            raise ValueError(f"label_mode must be one of {LABEL_MODES}, got {label_mode!r}")
        if observation_model not in OBSERVATION_MODELS:
            raise ValueError(
                f"observation_model must be one of {OBSERVATION_MODELS}, got {observation_model!r}"
            )
        if density_preset not in DENSITY_PRESETS:
            raise ValueError(f"density_preset must be one of {DENSITY_PRESETS}, got {density_preset!r}")
        self.density_preset = density_preset
        if isinstance(cause_mix, str):
            cause_mix = dict(DEFAULT_CAUSE_MIX) if cause_mix == "default" else None
        self.cause_mix = self._validate_cause_mix(cause_mix)
        self.num_uavs = num_uavs
        self.grid_size = grid_size
        self.altitude_range = altitude_range
        self.num_corridors = num_corridors
        self.num_sectors = num_sectors
        self.num_weather_cells = num_weather_cells
        self.num_restricted_zones = num_restricted_zones
        self.dt = dt
        self.wind_std = wind_std
        self.gps_cep = gps_cep
        self.adsb_latency_range = adsb_latency_range
        self.conflict_h_sep = conflict_h_sep
        self.conflict_v_sep = conflict_v_sep
        self.label_mode = label_mode
        self.lookahead_s = lookahead_s
        self.observation_model = observation_model
        self.base_seed = seed
        self.obs_params = ObservationParams(
            adsb_latency_s=tuple(_latency_range(adsb_latency_range)),
            packet_loss=packet_loss,
            gps_cep_m=gps_cep,
            weather_update_s=weather_update_s,
            registry_update_s=registry_update_s,
            corridor_update_s=corridor_update_s,
        )

        self.rng = np.random.RandomState(seed)
        self._init_infrastructure()

    # ------------------------------------------------------------------ #
    # Static infrastructure
    # ------------------------------------------------------------------ #
    def _init_infrastructure(self):
        n_side = int(math.sqrt(self.num_sectors))
        cell = self.grid_size / n_side
        self.sector_centers = np.array([
            [cell * (i + 0.5), cell * (j + 0.5), 0.0]
            for i in range(n_side) for j in range(n_side)
        ], dtype=np.float32)

        wx_side = int(math.sqrt(self.num_weather_cells))
        wx_cell = self.grid_size / wx_side
        self.weather_positions = np.array([
            [wx_cell * (i + 0.5), wx_cell * (j + 0.5), 90.0]
            for i in range(wx_side) for j in range(wx_side)
        ], dtype=np.float32)

        self.restricted_zones = np.zeros(
            (self.num_restricted_zones, 8), dtype=np.float32
        )
        for z in range(self.num_restricted_zones):
            cx = self.rng.uniform(500, self.grid_size - 500) if self.grid_size > 1000 \
                else self.rng.uniform(0.1 * self.grid_size, 0.9 * self.grid_size)
            cy = self.rng.uniform(500, self.grid_size - 500) if self.grid_size > 1000 \
                else self.rng.uniform(0.1 * self.grid_size, 0.9 * self.grid_size)
            cz = self.rng.uniform(*self.altitude_range)
            radius = self.rng.uniform(100, 300)
            self.restricted_zones[z, :4] = [cx, cy, cz, radius]
            self.restricted_zones[z, 4] = 1.0

        self.corridor_nodes = self._generate_corridor_graph()

    def export_infrastructure(self) -> Dict[str, np.ndarray]:
        return {
            "sector_centers": self.sector_centers.copy(),
            "weather_positions": self.weather_positions.copy(),
            "restricted_zones": self.restricted_zones.copy(),
            "corridor_nodes": self.corridor_nodes.copy(),
        }

    def load_infrastructure(self, infra: Dict[str, np.ndarray]) -> None:
        self.sector_centers = infra["sector_centers"].copy()
        self.weather_positions = infra["weather_positions"].copy()
        self.restricted_zones = infra["restricted_zones"].copy()
        self.corridor_nodes = infra["corridor_nodes"].copy()
        self.num_sectors = self.sector_centers.shape[0]
        self.num_weather_cells = self.weather_positions.shape[0]
        self.num_restricted_zones = self.restricted_zones.shape[0]

    def _generate_corridor_graph(self) -> np.ndarray:
        nodes = []
        lo = min(200.0, 0.1 * self.grid_size)
        for _ in range(self.num_corridors):
            x = self.rng.uniform(lo, self.grid_size - lo)
            y = self.rng.uniform(lo, self.grid_size - lo)
            z = self.rng.uniform(*self.altitude_range)
            nodes.append([x, y, z])
        return np.array(nodes, dtype=np.float32)

    @staticmethod
    def _validate_cause_mix(mix: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
        if mix is None:
            return None
        unknown = set(mix) - set(CAUSES)
        if unknown:
            raise ValueError(f"unknown causes {sorted(unknown)}; allowed: {CAUSES}")
        total = float(sum(mix.values()))
        if total <= 0:
            raise ValueError("cause_mix must have positive total weight")
        return {c: float(mix.get(c, 0.0)) / total for c in CAUSES}

    # ------------------------------------------------------------------ #
    # Cause injection
    # ------------------------------------------------------------------ #
    def _inject_causes(self, plans: List[UAVFlightPlan]) -> None:
        """Assign a cause to every plan and perturb it accordingly.  Called
        *after* all ordinary plan draws so that ``cause_mix=None`` leaves the
        RNG stream (and therefore the plans) untouched."""
        if self.cause_mix is None:
            return
        probs = np.array([self.cause_mix[c] for c in CAUSES])
        codes = self.rng.choice(len(CAUSES), size=len(plans), p=probs)
        hubs = self.corridor_nodes
        center = self.grid_size / 2.0
        for plan, code in zip(plans, codes):
            cause = CAUSES[int(code)]
            plan.cause = cause
            if cause == "wind_deviation":
                ang = self.rng.uniform(0, 2 * np.pi)
                mag = self.rng.uniform(*WIND_DEVIATION_DRIFT_MPS)
                plan.wind_drift = np.array([mag * np.cos(ang), mag * np.sin(ang), 0.0], np.float32)
            elif cause == "nonconforming":
                dev = np.zeros_like(plan.waypoints)
                for w in range(1, len(plan.waypoints)):
                    if self.rng.rand() < NONCONFORMING_LEG_PROB:
                        ang = self.rng.uniform(0, 2 * np.pi)
                        mag = self.rng.uniform(*NONCONFORMING_OFFSET_M)
                        dev[w, :2] = [mag * np.cos(ang), mag * np.sin(ang)]
                plan.deviations = dev
            elif cause == "priority_insertion":
                # emergency flight entering late from the boundary, straight through a hub
                hub = hubs[self.rng.randint(len(hubs))] if len(hubs) else np.array([center, center, 90.0], np.float32)
                ang = self.rng.uniform(0, 2 * np.pi)
                R = 0.5 * self.grid_size
                entry = np.array([center + R * np.cos(ang), center + R * np.sin(ang)], np.float32)
                entry = np.clip(entry, 50.0, self.grid_size - 50.0)
                exit_pt = np.clip(2 * hub[:2] - entry, 50.0, self.grid_size - 50.0)
                z = self._dense_altitude() if self.density_preset == "dense" else self.rng.uniform(*self.altitude_range)
                plan.waypoints = np.array([[*entry, z], [*hub[:2], z], [*exit_pt, z]], np.float32)
                plan.priority = 0
                plan.cruise_speed = float(self.rng.uniform(*PRIORITY_INSERTION_SPEED_MPS))
                plan.start_time = float(self.rng.uniform(*PRIORITY_INSERTION_START_S))
            elif cause == "noncooperative":
                plan.cooperative = False

    def _dense_altitude(self) -> float:
        layer = DENSE_ALTITUDE_LAYERS_M[self.rng.randint(len(DENSE_ALTITUDE_LAYERS_M))]
        z = layer + self.rng.uniform(-DENSE_LAYER_JITTER_M, DENSE_LAYER_JITTER_M)
        return float(np.clip(z, self.altitude_range[0], self.altitude_range[1]))

    def generate_flight_plans(self, num_plans: int = 500) -> List[UAVFlightPlan]:
        plans = []
        dense = self.density_preset == "dense"
        per_uav = DENSE_START_AREA_PER_UAV if dense else 8.0
        effective_grid = min(self.grid_size, self.num_uavs * per_uav)
        center = self.grid_size / 2.0
        margin = min(50.0, 0.02 * self.grid_size)
        hubs = self.corridor_nodes

        for uid in range(num_plans):
            n_wp = self.rng.randint(3, 8)
            waypoints = np.zeros((n_wp, 3), dtype=np.float32)
            waypoints[0] = [
                center + self.rng.uniform(-effective_grid / 2, effective_grid / 2),
                center + self.rng.uniform(-effective_grid / 2, effective_grid / 2),
                self._dense_altitude() if dense else self.rng.uniform(*self.altitude_range),
            ]
            for w in range(1, n_wp):
                if dense and len(hubs) > 0 and self.rng.rand() < DENSE_HUB_PROB:
                    # fly to one of the k nearest corridor hubs (converging traffic),
                    # staying in the current altitude layer
                    d = np.linalg.norm(hubs[:, :2] - waypoints[w - 1, :2], axis=1)
                    k = min(DENSE_HUB_CHOICES, len(hubs))
                    hub = hubs[np.argpartition(d, k - 1)[:k][self.rng.randint(k)]]
                    waypoints[w, :2] = hub[:2] + self.rng.uniform(-30, 30, 2)
                    waypoints[w, 2] = waypoints[w - 1, 2]
                else:
                    dx = self.rng.uniform(-800, 800)
                    dy = self.rng.uniform(-800, 800)
                    dz = 0.0 if dense else self.rng.uniform(-20, 20)
                    waypoints[w] = waypoints[w - 1] + [dx, dy, dz]
                waypoints[w, :2] = np.clip(waypoints[w, :2], margin, self.grid_size - margin)
                waypoints[w, 2] = np.clip(
                    waypoints[w, 2], self.altitude_range[0], self.altitude_range[1]
                )

            priority = self.rng.choice([0, 1, 1, 1, 2, 3], p=[0.1, 0.6, 0.15, 0.05, 0.05, 0.05])
            speed = self.rng.uniform(8.0, 22.0)
            start = self.rng.uniform(0, 30.0)

            plans.append(UAVFlightPlan(
                uav_id=uid,
                waypoints=waypoints,
                priority=priority,
                cruise_speed=speed,
                start_time=start,
            ))
        self._inject_causes(plans)
        return plans

    # ------------------------------------------------------------------ #
    # Physics
    # ------------------------------------------------------------------ #
    def run_physics(
        self,
        plans: List[UAVFlightPlan],
        duration_seconds: float,
        extra_seconds: float = 0.0,
    ) -> TruthLog:
        """Integrate the fleet for ``duration + extra`` seconds and log the truth."""
        N = self.num_uavs
        n_epochs = int(duration_seconds / self.dt)
        n_total = n_epochs + int(round(extra_seconds / self.dt))
        truth_gps_walk = self.observation_model == "legacy"

        positions = np.zeros((N, 3), dtype=np.float32)
        velocities = np.zeros((N, 3), dtype=np.float32)
        prev_velocities = np.zeros((N, 3), dtype=np.float32)
        headings = np.zeros(N, dtype=np.float32)
        prev_headings = np.zeros(N, dtype=np.float32)
        battery = np.ones(N, dtype=np.float32)
        priorities = np.zeros(N, dtype=np.int32)
        wp_idx = np.zeros(N, dtype=np.int32)
        drift = np.zeros((N, 3), dtype=np.float32)
        causes = np.zeros(N, dtype=np.int8)
        cooperative = np.ones(N, dtype=bool)

        for plan in plans:
            uid = plan.uav_id
            positions[uid] = plan.waypoints[0]
            priorities[uid] = plan.priority
            causes[uid] = CAUSE_CODES.get(plan.cause, 0)
            cooperative[uid] = plan.cooperative
            if plan.wind_drift is not None:
                drift[uid] = plan.wind_drift
            if len(plan.waypoints) > 1:
                d = plan.waypoints[1] - plan.waypoints[0]
                dist = np.linalg.norm(d)
                if dist > 1e-6:
                    velocities[uid] = d / dist * plan.cruise_speed
                    headings[uid] = np.arctan2(d[1], d[0])
        has_drift = bool(np.any(drift != 0))

        log = TruthLog(
            dt=self.dt, n_epochs=n_epochs, n_total=n_total,
            positions=np.zeros((n_total, N, 3), np.float32),
            velocities=np.zeros((n_total, N, 3), np.float32),
            headings=np.zeros((n_total, N), np.float32),
            heading_rates=np.zeros((n_total, N), np.float32),
            accelerations=np.zeros((n_total, N, 3), np.float32),
            battery=np.zeros((n_total, N), np.float32),
            battery_rates=np.zeros((n_total, N), np.float32),
            priorities=priorities,
            wind=(self.rng.randn(n_total, 3) * self.wind_std).astype(np.float32),
            local_wind_noise=(self.rng.randn(n_total, N, 3) * 0.3).astype(np.float32),
            gps_dop=(self.gps_cep + self.rng.exponential(0.5, (n_total, N))).astype(np.float32),
            corridor_pairs=self._compute_corridor_pairs(plans),
            weather_gain=(1.0 + 0.1 * self.rng.randn(n_total, self.num_weather_cells)).astype(np.float32),
            weather_vis=self.rng.uniform(5000, 15000, (n_total, self.num_weather_cells)).astype(np.float32),
            latency_u=self.rng.uniform(0.0, 1.0, N).astype(np.float32),
            loss_u=self.rng.uniform(0.0, 1.0, (n_total, N)).astype(np.float32),
            gps_noise_unit=self.rng.randn(n_total, N, 3).astype(np.float32),
            infrastructure=self.export_infrastructure(),
            causes=causes,
            cooperative=cooperative,
        )
        (log.plan_waypoints, log.plan_num_waypoints,
         log.plan_cruise_speed, log.plan_start_time) = pack_filed_plans(plans, N)

        for epoch in range(n_total):
            t = epoch * self.dt
            wind = log.wind[epoch]
            if truth_gps_walk:
                gps_noise = self.rng.randn(N, 3).astype(np.float32) * self.gps_cep * 0.01
            else:
                gps_noise = 0.0

            for uid, plan in enumerate(plans):
                if t < plan.start_time:
                    continue
                wi = wp_idx[uid]
                if wi >= len(plan.waypoints) - 1:
                    velocities[uid] *= 0.95
                    continue

                target = plan.waypoints[wi + 1]
                if plan.deviations is not None:
                    target = target + plan.deviations[wi + 1]      # flown target != filed plan
                to_target = target - positions[uid]
                dist = np.linalg.norm(to_target)

                if dist < 5.0:
                    wp_idx[uid] = min(wi + 1, len(plan.waypoints) - 1)
                    continue

                desired_v = to_target / dist * plan.cruise_speed
                steer = (desired_v - velocities[uid]) * 0.3
                velocities[uid] += steer * self.dt
                velocities[uid] += wind * 0.1 * self.dt
                headings[uid] = np.arctan2(velocities[uid][1], velocities[uid][0])

            positions += velocities * self.dt + gps_noise
            if has_drift:
                positions += drift * self.dt                         # wind pushes off-track; steering re-aims
            positions[:, :2] = np.clip(positions[:, :2], 0, self.grid_size)
            positions[:, 2] = np.clip(
                positions[:, 2], self.altitude_range[0], self.altitude_range[1]
            )
            battery -= self.rng.uniform(0.00001, 0.00005, N).astype(np.float32)
            battery = np.clip(battery, 0, 1)
            battery_discharge = self.rng.uniform(0.00001, 0.00005, N).astype(np.float32)

            log.positions[epoch] = positions
            log.velocities[epoch] = velocities
            log.headings[epoch] = headings
            log.heading_rates[epoch] = (headings - prev_headings) / self.dt
            log.accelerations[epoch] = (velocities - prev_velocities) / self.dt
            log.battery[epoch] = battery
            log.battery_rates[epoch] = -battery_discharge

            prev_velocities[:] = velocities
            prev_headings[:] = headings

        return log

    def _compute_corridor_pairs(self, plans: List[UAVFlightPlan]) -> List[Tuple[int, int]]:
        corridor_users: Dict[int, List[int]] = {}
        for plan in plans:
            for ci, cnode in enumerate(self.corridor_nodes):
                wi = min(1, len(plan.waypoints) - 1)
                wp = plan.waypoints[wi]
                d = np.linalg.norm(wp[:2] - cnode[:2])
                if d < 600:
                    corridor_users.setdefault(ci, []).append(plan.uav_id)

        pairs = []
        for ci, users in corridor_users.items():
            for a in range(len(users)):
                for b in range(a + 1, min(a + 3, len(users))):
                    pairs.append((users[a], users[b]))
        return pairs

    # ------------------------------------------------------------------ #
    # Observation
    # ------------------------------------------------------------------ #
    def observe(
        self,
        log: TruthLog,
        epoch: int,
        params: Optional[ObservationParams] = None,
    ) -> AirspaceState:
        """Return the airspace state as seen by the edge node at ``epoch``.

        ``params`` overrides the simulator's observation conditions (used to
        re-evaluate an existing scenario under different latency / loss / CEP
        without re-running the physics).
        """
        if self.observation_model == "legacy":
            return self._observe_legacy(log, epoch)
        return self._observe_adsb(log, epoch, params or self.obs_params)

    # -- shared helpers -------------------------------------------------- #
    def _corridor_view(self, log: TruthLog, t: float):
        N = log.num_uavs
        corridor_res = [(a, b, t, t + 120.0) for a, b in log.corridor_pairs]
        corridor_ids = np.zeros(N, dtype=np.float32)
        for ua, ub in log.corridor_pairs:
            if ua < N:
                corridor_ids[ua] = 1.0
            if ub < N:
                corridor_ids[ub] = 1.0
        return corridor_res, corridor_ids

    def _weather_view(self, log: TruthLog, epoch: int) -> np.ndarray:
        return self._compute_weather_state(
            epoch * self.dt, log.wind[epoch],
            gain=None if log.weather_gain is None else log.weather_gain[epoch],
            vis=None if log.weather_vis is None else log.weather_vis[epoch],
        )

    # -- legacy: observation == truth ------------------------------------ #
    def _observe_legacy(self, log: TruthLog, epoch: int) -> AirspaceState:
        N = log.num_uavs
        t = epoch * self.dt
        corridor_res, corridor_ids = self._corridor_view(log, t)
        wind = log.wind[epoch]
        local_wind = np.tile(wind, (N, 1)).astype(np.float32) + log.local_wind_noise[epoch]
        positions = log.positions[epoch]

        return AirspaceState(
            uav_positions=positions.copy(),
            uav_velocities=log.velocities[epoch].copy(),
            uav_headings=log.headings[epoch].copy(),
            uav_battery=log.battery[epoch].copy(),
            uav_priority=log.priorities.copy(),
            uav_avoiding=np.zeros(N, dtype=bool),
            sector_occupancy=self._compute_sector_occupancy(positions),
            weather_cells=self._weather_view(log, epoch),
            restricted_zones=self.restricted_zones.copy(),
            corridor_reservations=corridor_res,
            epoch_time=t,
            uav_heading_rates=log.heading_rates[epoch].copy(),
            uav_accelerations=log.accelerations[epoch].copy(),
            uav_battery_rates=log.battery_rates[epoch].copy(),
            uav_corridor_ids=corridor_ids,
            uav_local_wind=local_wind,
            uav_gps_dop=log.gps_dop[epoch].copy(),
            uav_plan_context=filed_plan_context(log, positions, t, log.velocities[epoch]),
        )

    # -- adsb: latency + loss + GPS noise -------------------------------- #
    def reception_epochs(
        self, log: TruthLog, epoch: int, params: Optional[ObservationParams] = None
    ) -> np.ndarray:
        """Generation epoch g*(i) of the freshest ADS-B report received by
        the edge node at ``epoch`` (``-1`` if nothing received yet)."""
        params = params or self.obs_params
        N = log.num_uavs
        lo, hi = params.latency_range()
        lat_u = log.latency_u if log.latency_u is not None else np.zeros(N, np.float32)
        latency = lo + (hi - lo) * lat_u
        if log.cooperative is not None:
            # non-cooperative targets are only seen through slower surveillance
            latency = latency + np.where(log.cooperative, 0.0, NONCOOP_EXTRA_LATENCY_S)
        lat_frames = np.rint(latency / self.dt).astype(np.int64)
        g_max = epoch - lat_frames                                   # (N,)

        if params.packet_loss <= 0.0 or log.loss_u is None:
            return np.where(g_max >= 0, g_max, -1)

        ok = log.loss_u[: epoch + 1] >= params.packet_loss            # (e+1, N)
        idx = np.where(ok, np.arange(epoch + 1)[:, None], -1)
        last_ok = np.maximum.accumulate(idx, axis=0)                 # (e+1, N)
        g_star = np.full(N, -1, dtype=np.int64)
        valid = g_max >= 0
        g_star[valid] = last_ok[g_max[valid], np.nonzero(valid)[0]]
        return g_star

    def _observe_adsb(
        self, log: TruthLog, epoch: int, params: ObservationParams
    ) -> AirspaceState:
        N = log.num_uavs
        t = epoch * self.dt
        g_star = self.reception_epochs(log, epoch, params)
        g_used = np.where(g_star >= 0, g_star, 0)                    # fallback: departure state
        rows = np.arange(N)

        sigma_h = params.gps_cep_m * CEP_TO_SIGMA
        noise = log.gps_noise_unit[g_used, rows] if log.gps_noise_unit is not None \
            else np.zeros((N, 3), np.float32)
        noise = noise * np.array([sigma_h, sigma_h, sigma_h * VERTICAL_SIGMA_FACTOR], np.float32)
        if log.cooperative is not None:
            noise = noise * np.where(log.cooperative, 1.0, NONCOOP_NOISE_FACTOR)[:, None].astype(np.float32)

        positions = log.positions[g_used, rows] + noise
        velocities = log.velocities[g_used, rows]
        last_rx_time = (g_used * self.dt).astype(np.float32)

        corridor_res, corridor_ids = self._corridor_view(log, t)
        wind = log.wind[g_used]                                      # wind as reported
        local_wind = wind + log.local_wind_noise[g_used, rows]

        def _last_publish(period: float) -> float:
            return math.floor(t / period + 1e-9) * period if period > 0 else t

        env_last_update = {
            "weather": _last_publish(params.weather_update_s),
            "restricted": _last_publish(params.registry_update_s),
            "sector": _last_publish(params.corridor_update_s),
            "corridor": _last_publish(params.corridor_update_s),
        }
        wx_epoch = min(int(round(env_last_update["weather"] / self.dt)), log.n_total - 1)

        return AirspaceState(
            uav_positions=positions.astype(np.float32),
            uav_velocities=velocities.copy(),
            uav_headings=log.headings[g_used, rows].copy(),
            uav_battery=log.battery[g_used, rows].copy(),
            uav_priority=log.priorities.copy(),
            uav_avoiding=np.zeros(N, dtype=bool),
            sector_occupancy=self._compute_sector_occupancy(positions),
            weather_cells=self._weather_view(log, wx_epoch),
            restricted_zones=self.restricted_zones.copy(),
            corridor_reservations=corridor_res,
            epoch_time=t,
            uav_heading_rates=log.heading_rates[g_used, rows].copy(),
            uav_accelerations=log.accelerations[g_used, rows].copy(),
            uav_battery_rates=log.battery_rates[g_used, rows].copy(),
            uav_corridor_ids=corridor_ids,
            uav_local_wind=local_wind.astype(np.float32),
            uav_gps_dop=log.gps_dop[g_used, rows].copy(),
            uav_last_rx_time=last_rx_time,
            env_last_update_time=env_last_update,
            uav_plan_context=filed_plan_context(log, positions, t, velocities),   # filed plan + observed state only
        )

    # ------------------------------------------------------------------ #
    # Labels
    # ------------------------------------------------------------------ #
    def label(self, log: TruthLog, epoch: int) -> List[ConflictEvent]:
        """Ground-truth conflict events for ``epoch`` according to ``label_mode``."""
        if self.label_mode == "instantaneous":
            return self._detect_ground_truth_conflicts(
                log.positions[epoch], log.velocities[epoch], epoch, epoch * self.dt
            )
        return self._detect_lookahead_conflicts(log, epoch)

    def _detect_lookahead_conflicts(self, log: TruthLog, epoch: int) -> List[ConflictEvent]:
        """Positive iff separation minima are violated at any epoch in
        [epoch, epoch + lookahead] of the *true* trajectories."""
        L = int(round(self.lookahead_s / self.dt))
        end = min(epoch + L, log.n_total - 1)
        P = log.positions[epoch:end + 1]                  # (T, N, 3)
        T, N = P.shape[0], P.shape[1]
        if N < 2 or T == 0:
            return []

        # Conservative candidate pre-filter: a pair can only violate h_sep
        # within the window if its current horizontal distance is below
        # h_sep + 2 * v_max * window.
        vmax = float(np.linalg.norm(log.velocities[epoch:end + 1], axis=-1).max())
        radius = self.conflict_h_sep + 2.0 * vmax * (T - 1) * self.dt + 1.0
        p0 = P[0]
        ii, jj = np.triu_indices(N, k=1)
        d0 = np.hypot(p0[jj, 0] - p0[ii, 0], p0[jj, 1] - p0[ii, 1])
        keep = d0 <= radius
        ci, cj = ii[keep], jj[keep]

        events: List[ConflictEvent] = []
        CH = 8192
        for s in range(0, len(ci), CH):
            a = ci[s:s + CH]
            b = cj[s:s + CH]
            dp = P[:, b, :] - P[:, a, :]                   # (T, C, 3)
            dh = np.hypot(dp[..., 0], dp[..., 1])
            dv = np.abs(dp[..., 2])
            hit = (dh < self.conflict_h_sep) & (dv < self.conflict_v_sep)
            any_hit = hit.any(axis=0)
            if not any_hit.any():
                continue
            first = hit.argmax(axis=0)
            for idx in np.nonzero(any_hit)[0]:
                k = int(first[idx])
                i, j = int(a[idx]), int(b[idx])
                events.append(ConflictEvent(
                    uav_i=i, uav_j=j, epoch=epoch,
                    time_to_conflict=k * self.dt,
                    min_separation_h=float(dh[k, idx]),
                    min_separation_v=float(dv[k, idx]),
                    cause=log.cause_of_pair(i, j),
                ))
        return events

    def _detect_ground_truth_conflicts(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        epoch: int,
        t: float,
    ) -> List[ConflictEvent]:
        """Legacy instantaneous labels (separation violated *now*)."""
        conflicts = []
        for i in range(self.num_uavs):
            for j in range(i + 1, self.num_uavs):
                dp = positions[j] - positions[i]
                h_dist = np.sqrt(dp[0] ** 2 + dp[1] ** 2)
                v_dist = abs(dp[2])

                if h_dist < self.conflict_h_sep and v_dist < self.conflict_v_sep:
                    dv = velocities[j] - velocities[i]
                    if np.dot(dv, dv) > 1e-8:
                        ttc = -np.dot(dp, dv) / np.dot(dv, dv)
                    else:
                        ttc = 0.0
                    conflicts.append(ConflictEvent(
                        uav_i=i, uav_j=j, epoch=epoch,
                        time_to_conflict=max(0, ttc),
                        min_separation_h=h_dist, min_separation_v=v_dist,
                    ))
        return conflicts

    # ------------------------------------------------------------------ #
    # Scenario driver
    # ------------------------------------------------------------------ #
    def simulate_scenario(
        self,
        duration_seconds: float = 60.0,
        plans: Optional[List[UAVFlightPlan]] = None,
        label_every: int = 1,
        obs_params: Optional[ObservationParams] = None,
        log: Optional[TruthLog] = None,
    ) -> Iterator[Tuple[AirspaceState, List[ConflictEvent]]]:
        """Run a scenario and yield (observed state, conflicts) at each epoch.

        ``label_every``: compute labels only every k-th epoch (others yield
        an empty list).  Labels are comparatively expensive in look-ahead mode.
        ``obs_params``: override observation conditions.  ``log``: reuse an
        existing TruthLog instead of re-running the physics.
        """
        if log is None:
            if plans is None:
                plans = self.generate_flight_plans(self.num_uavs)
            extra = self.lookahead_s if self.label_mode == "lookahead" else 0.0
            log = self.run_physics(plans, duration_seconds, extra_seconds=extra)
        for epoch in range(log.n_epochs):
            state = self.observe(log, epoch, obs_params)
            conflicts = self.label(log, epoch) if epoch % label_every == 0 else []
            yield state, conflicts

    def _compute_sector_occupancy(self, positions: np.ndarray) -> np.ndarray:
        occ = np.zeros((self.num_sectors, 8), dtype=np.float32)
        cell_radius = self.grid_size / (2 * math.sqrt(self.num_sectors))
        for s in range(self.num_sectors):
            center = self.sector_centers[s, :2]
            dists = np.linalg.norm(positions[:, :2] - center, axis=1)
            count = np.sum(dists < cell_radius)
            occ[s, 0] = center[0] / self.grid_size
            occ[s, 1] = center[1] / self.grid_size
            occ[s, 2] = count
            occ[s, 3] = count / max(self.num_uavs * 0.1, 1)
        return occ

    def _compute_weather_state(
        self,
        t: float,
        wind: np.ndarray,
        gain: Optional[np.ndarray] = None,
        vis: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Weather-cell features; per-cell randomness comes from the TruthLog
        (``gain``, ``vis``) so that observation is a pure function of the log."""
        n_wx = self.num_weather_cells
        if gain is None:
            gain = 1.0 + 0.1 * self.rng.randn(n_wx)
        if vis is None:
            vis = self.rng.uniform(5000, 15000, n_wx)
        wx = np.zeros((n_wx, 12), dtype=np.float32)
        wx[:, 0:3] = self.weather_positions
        wx[:, 3:6] = wind[None, :] * np.asarray(gain, np.float32)[:, None]
        wx[:, 6] = np.linalg.norm(wind)
        wx[:, 7] = 1.0
        wx[:, 8] = vis
        return wx

    # ------------------------------------------------------------------ #
    # Dataset
    # ------------------------------------------------------------------ #
    def generate_dataset(
        self,
        split: str = "train",
        num_scenarios: int = 10,
        scenario_duration: float = 60.0,
        device: torch.device = torch.device("cpu"),
        builder: Optional[TKGBuilder] = None,
        obs_params: Optional[ObservationParams] = None,
        candidates: str = "proximity",
        proximity_margin_m: float = 50.0,
    ) -> List[Tuple[TKGSnapshot, torch.Tensor]]:
        """Generate a full dataset split as list of (snapshot, labels).

        Snapshots are taken every ``epoch_step`` epochs (1 Hz).  The scored
        pair set per snapshot is controlled by ``candidates``:

        * ``"proximity"`` (default) – pairs that can physically violate
          separation within the label window given the observed speeds
          (see :meth:`proximity_candidates`); recall 1 by construction up to
          the stated margin.
        * ``"edges"``   – pairs carrying an approaches/shares_corridor edge
          (``snapshot.candidate_pairs``).
        * ``"all"``     – every unordered pair.
        * ``"sampled"`` – legacy: all positives + random negatives (≤ 4N).

        Positives outside the scored set are recorded in
        ``num_missed_positives`` / ``missed_ttc`` and counted as misses by
        the metrics.  ``conflict_ttc`` holds time-to-conflict (s) for
        positives, -1 otherwise.
        """
        logs = self.simulate_logs(split, num_scenarios, scenario_duration)
        return self.dataset_from_logs(
            logs, device=device, builder=builder, obs_params=obs_params,
            candidates=candidates, proximity_margin_m=proximity_margin_m,
        )

    def simulate_logs(
        self, split: str, num_scenarios: int, scenario_duration: float = 60.0
    ) -> List[TruthLog]:
        """Run the physics for ``num_scenarios`` deterministic scenarios and
        return their truth logs (infrastructure embedded).  Observation
        conditions are *not* applied here, so the same logs can be observed
        under different latency / loss / CEP settings."""
        logs = []
        extra = self.lookahead_s if self.label_mode == "lookahead" else 0.0
        for scenario_idx in range(num_scenarios):
            self.rng = np.random.RandomState(scenario_seed(self.base_seed, split, scenario_idx))
            self._init_infrastructure()
            plans = self.generate_flight_plans(self.num_uavs)
            log = self.run_physics(plans, scenario_duration, extra_seconds=extra)
            log.scenario_id = (split, scenario_idx)
            logs.append(log)
        return logs

    def dataset_from_logs(
        self,
        logs: List[TruthLog],
        device: torch.device = torch.device("cpu"),
        builder: Optional[TKGBuilder] = None,
        obs_params: Optional[ObservationParams] = None,
        candidates: str = "proximity",
        proximity_margin_m: float = 50.0,
        epoch_step: int = 10,
    ) -> List[Tuple[TKGSnapshot, torch.Tensor]]:
        """Observe existing truth logs (optionally under new ``obs_params``)
        and build labelled snapshots.  Labels depend only on the truth, so
        they are identical across observation conditions."""
        if candidates not in CANDIDATE_MODES:
            raise ValueError(f"candidates must be one of {CANDIDATE_MODES}, got {candidates!r}")
        builder = builder if builder is not None else TKGBuilder()
        params = obs_params if obs_params is not None else self.obs_params
        dataset = []
        for log in logs:
            if log.infrastructure is not None:
                self.load_infrastructure(log.infrastructure)
            if log.scenario_id is not None:          # deterministic legacy negative sampling
                self.rng = np.random.RandomState(
                    scenario_seed(self.base_seed, log.scenario_id[0], log.scenario_id[1]) + 1
                )
            builder.reset()
            for epoch in range(0, log.n_epochs, epoch_step):
                state = self.observe(log, epoch, params)
                conflicts = self.label(log, epoch)
                snapshot = builder.build(state, device=device)
                self.attach_pair_labels(
                    snapshot, conflicts, candidates, device,
                    obs_params=params, proximity_margin_m=proximity_margin_m,
                )
                dataset.append((snapshot, snapshot.conflict_labels))
        return dataset

    def proximity_candidates(
        self,
        snapshot: TKGSnapshot,
        obs_params: Optional[ObservationParams] = None,
        margin_m: float = 50.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Pairs that can violate (h_sep, v_sep) within the label window.

        With observed speeds bounded by v_max (horizontal) and vz_max
        (vertical), a pair whose observed separation exceeds

            R_h = h_sep + 2·v_max·(W + L_max) + margin
            R_v = v_sep + 2·vz_max·(W + L_max) + margin

        cannot be in conflict during the window W (observations may be up
        to L_max seconds stale).  ``margin`` absorbs GPS noise and speed
        changes.  Candidates are generated with the same grid hash as the
        graph builder, then filtered exactly.  Only applies to lookahead
        labels; for instantaneous labels W = 0.
        """
        n = snapshot.num_uavs
        if n < 2:
            e = np.zeros(0, np.int64)
            return e, e
        params = obs_params if obs_params is not None else self.obs_params
        W = self.lookahead_s if self.label_mode == "lookahead" else 0.0
        if self.observation_model == "adsb":
            W += params.latency_range()[1]
        feats = snapshot.node_features[:n].detach().cpu().numpy()
        P, V = feats[:, 0:3], feats[:, 3:6]
        v_h = float(np.hypot(V[:, 0], V[:, 1]).max())
        v_z = float(np.abs(V[:, 2]).max())
        R_h = self.conflict_h_sep + 2.0 * v_h * W + margin_m
        R_v = self.conflict_v_sep + 2.0 * v_z * W + margin_m

        ci, cj = TKGBuilder._grid_candidates(P, R_h)
        if ci.size == 0:
            return ci, cj
        dh = np.hypot(P[cj, 0] - P[ci, 0], P[cj, 1] - P[ci, 1])
        dz = np.abs(P[cj, 2] - P[ci, 2])
        keep = (dh < R_h) & (dz < R_v)
        return ci[keep], cj[keep]

    def attach_pair_labels(
        self,
        snapshot: TKGSnapshot,
        conflicts: List[ConflictEvent],
        candidates: str = "proximity",
        device: torch.device = torch.device("cpu"),
        obs_params: Optional[ObservationParams] = None,
        proximity_margin_m: float = 50.0,
    ) -> None:
        """Set conflict_pairs / conflict_labels / conflict_ttc (and the missed
        positives bookkeeping) on ``snapshot`` for the chosen candidate set."""
        n = snapshot.num_uavs
        ttc_mat = np.full((n, n), -1.0, dtype=np.float32)
        cause_mat = np.full((n, n), -1, dtype=np.int8)
        for c in conflicts:
            ttc_mat[c.uav_i, c.uav_j] = c.time_to_conflict
            ttc_mat[c.uav_j, c.uav_i] = c.time_to_conflict
            cause_mat[c.uav_i, c.uav_j] = cause_mat[c.uav_j, c.uav_i] = CAUSE_CODES.get(c.cause, 0)

        if candidates == "proximity":
            src, dst = self.proximity_candidates(snapshot, obs_params, proximity_margin_m)
        elif candidates == "edges":
            cand = snapshot.candidate_pairs
            src = cand[0].cpu().numpy() if cand is not None else np.zeros(0, np.int64)
            dst = cand[1].cpu().numpy() if cand is not None else np.zeros(0, np.int64)
        elif candidates == "all":
            src, dst = np.triu_indices(n, k=1)
        else:  # "sampled" (legacy)
            src, dst = self._sample_pairs_legacy(n, ttc_mat)
        src = np.asarray(src, dtype=np.int64)
        dst = np.asarray(dst, dtype=np.int64)

        ttcs = ttc_mat[src, dst] if src.size else np.zeros(0, np.float32)
        labels = (ttcs >= 0).astype(np.float32)

        # positives outside the scored set
        pos_i, pos_j = np.nonzero(np.triu(ttc_mat >= 0, k=1))
        if src.size:
            covered = np.zeros((n, n), dtype=bool)
            covered[src, dst] = True
            covered[dst, src] = True
            missed = ~covered[pos_i, pos_j]
        else:
            missed = np.ones(pos_i.size, dtype=bool)
        missed_ttc = ttc_mat[pos_i[missed], pos_j[missed]]
        missed_cause = cause_mat[pos_i[missed], pos_j[missed]]

        snapshot.conflict_pairs = torch.tensor(np.stack([src, dst]), dtype=torch.long, device=device)
        snapshot.conflict_labels = torch.tensor(labels, dtype=torch.float32, device=device)
        snapshot.conflict_ttc = torch.tensor(ttcs, dtype=torch.float32, device=device)
        snapshot.conflict_cause = torch.tensor(
            cause_mat[src, dst] if src.size else np.zeros(0, np.int8), dtype=torch.int8, device=device
        )
        snapshot.num_missed_positives = int(missed.sum())
        snapshot.missed_ttc = torch.tensor(missed_ttc, dtype=torch.float32, device=device)
        snapshot.missed_cause = torch.tensor(missed_cause, dtype=torch.int8, device=device)

    def _sample_pairs_legacy(self, n: int, ttc_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Legacy sampling: all positives + random negatives (cap 4N, ≥10× positives)."""
        n_sample = min(n * 4, n * (n - 1) // 2)
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        if len(all_pairs) > n_sample:
            pos_pairs = [(i, j) for i, j in all_pairs if ttc_mat[i, j] >= 0]
            neg_pairs = [(i, j) for i, j in all_pairs if ttc_mat[i, j] < 0]
            n_pos = len(pos_pairs)
            n_neg = min(len(neg_pairs), max(n_sample - n_pos, n_pos * 10))
            self.rng.shuffle(neg_pairs)
            sampled = pos_pairs + neg_pairs[:n_neg]
        else:
            sampled = all_pairs
        if not sampled:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        arr = np.asarray(sampled, dtype=np.int64)
        return arr[:, 0], arr[:, 1]
