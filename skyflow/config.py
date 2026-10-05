"""Centralized configuration dataclass matching paper hyperparameters."""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ModelConfig:
    num_layers: int = 4
    embed_dim: int = 128
    num_heads: int = 4
    temporal_dim: int = 32
    recurrent_dim: int = 64
    dropout: float = 0.1
    # Derived from features.leakage_free when None (5 leakage-free / 6 legacy).
    num_relation_types: Optional[int] = None
    # Ablation switches (S7a)
    use_temporal: bool = True       # False -> TR-GAT-NT: φ(δ) removed from attention
    use_gating: bool = True         # False -> uniform relation average instead of g_r(h_i)
    use_gru: bool = True            # False -> per-snapshot projection, no recurrence


@dataclass
class DataConfig:
    num_uavs: int = 500
    num_sectors: int = 64
    num_weather_cells: int = 36
    num_restricted_zones: int = 12
    # Derived from features.leakage_free when None (20 leakage-free / 23 legacy).
    uav_feature_dim: Optional[int] = None
    sector_feature_dim: int = 8
    weather_feature_dim: int = 12
    observation_window: int = 10
    lookahead_seconds: float = 30.0
    conflict_h_sep_m: float = 10.0
    conflict_v_sep_m: float = 3.0
    sim_freq_hz: float = 10.0
    grid_size_m: float = 5000.0
    # Legacy (paper-claimed) volumes; not used by the pipeline.
    scenario_minutes_train: int = 3360
    scenario_minutes_val: int = 720
    scenario_minutes_test: int = 720
    # Dataset actually generated / cached (S7a)
    train_scenarios: int = 40
    val_scenarios: int = 10
    test_scenarios: int = 10
    scenario_duration_s: float = 60.0
    sim_seed: int = 42          # dataset seed: fixed across model seeds (part of the cache key)
    cache_dir: str = "cache"

    def split_scenarios(self, split: str) -> int:
        return {"train": self.train_scenarios, "val": self.val_scenarios,
                "test": self.test_scenarios}[split]


@dataclass
class FeaturesConfig:
    """Input feature pipeline switches (S2)."""
    leakage_free: bool = True       # drop d_min/t_cpa/f_avoid and conflicts_with
    input_set: str = "full"         # "full" | "telemetry_only" (S6)
    normalize_inputs: bool = True   # S8: standardise node features with train-split mean/std (False = raw, legacy)


@dataclass
class LabelsConfig:
    """Ground-truth label definition (authorized change, see docs/repo_map.md §8)."""
    mode: str = "lookahead"         # "lookahead" (30 s window) | "instantaneous" (legacy)


@dataclass
class TemporalConfig:
    """Definition of the edge time offset δ fed to φ(δ) (S3)."""
    delta_mode: str = "aoi"         # "aoi" (age of information) | "legacy" (time since edge last seen)


@dataclass
class GraphConfig:
    """TKG construction (S4)."""
    neighbor_search: str = "grid"   # "grid" (spatial hash) | "bruteforce" (all pairs)


@dataclass
class ScoringConfig:
    """Which UAV pairs the conflict head scores (S4)."""
    # "proximity" (pairs able to violate separation within the label window;
    # recall 1 up to margin) | "edges" (approaches + shares_corridor) | "all" |
    # "sampled" (legacy: positives + random negatives)
    candidates: str = "proximity"
    proximity_margin_m: float = 50.0   # slack for GPS noise / speed changes


@dataclass
class BaselinesConfig:
    """Baseline-specific knobs (S5)."""
    cpa_rule_thresholds: str = "val_search"   # "val_search" (grid-search on val, by F1) | "label" (10 m / 3 m)


@dataclass
class SimConfig:
    """Observation-layer parameters of the UrbanAir-500 simulator (S3/S6).

    Training defaults reproduce the paper's claimed conditions
    (ADS-B latency 0.5–1.2 s, GPS CEP 2.5 m, no packet loss)."""
    observation_model: str = "adsb"           # "adsb" | "legacy" (observation == truth)
    density_preset: str = "dense"             # "dense" (layered airspace, hub-converging plans) | "legacy"
    # Synthetic conflict-cause injection: per-UAV cause shares (normalised).
    # null/None disables injection (all UAVs are ordinary planned flights).
    cause_mix: Optional[Dict[str, float]] = field(default_factory=lambda: {
        "planned_crossing": 0.80, "nonconforming": 0.08, "wind_deviation": 0.06,
        "priority_insertion": 0.03, "noncooperative": 0.03,
    })
    adsb_latency_s: List[float] = field(default_factory=lambda: [0.5, 1.2])  # scalar or [lo, hi]
    packet_loss: float = 0.0                  # per-report loss probability, 0–0.3
    gps_cep_m: float = 2.5
    weather_update_s: float = 5.0
    registry_update_s: float = 10.0
    corridor_update_s: float = 1.0


PAPER_SEEDS = [42, 123, 456, 789, 1024]


@dataclass
class TrainingConfig:
    epochs: int = 150               # max epochs (early stopping may end sooner)
    batch_size: int = 32            # legacy, unused
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    focal_gamma: float = 2.0
    focal_alpha: float = 0.75
    loss: str = "focal"             # "focal" | "bce" (abl_bce)
    conflict_threshold: float = 0.42
    warmup_steps: int = 1000
    seed: int = 42
    num_seeds: int = 5
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1024])
    device: str = "auto"
    # S7a: early stopping on validation F1 (same rule for every method)
    early_stopping_patience: int = 15
    min_epochs: int = 30            # S8: val F1 stays 0 for the first epochs (0.16 % positives)
    threshold_mode: str = "val"     # S8: "val" = select F1-optimal threshold on val, apply to test; "fixed" = conflict_threshold
    eval_every: int = 1
    # S7a: observation windows per optimizer step; OOM halves micro-batch and
    # accumulates gradients to keep this effective batch
    batch_windows: int = 4
    micro_batch_windows: int = 1    # windows per backward (grad accumulation up to batch_windows); memory knob only
    # S7a: numerics
    amp: bool = False
    tf32: bool = True
    # regime boundary on time-to-conflict (s): hard <= boundary < easy
    regime_ttc_boundary_s: float = 15.0


_SECTIONS = ("model", "data", "training", "features", "labels", "temporal", "sim", "graph", "scoring", "baselines")


@dataclass
class SkyFlowConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    labels: LabelsConfig = field(default_factory=LabelsConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    baselines: BaselinesConfig = field(default_factory=BaselinesConfig)
    output_dir: str = "outputs"

    # ------------------------------------------------------------------ #
    # Derived quantities (single source of truth for dimensions)
    # ------------------------------------------------------------------ #
    def leakage_free(self) -> bool:
        feats = getattr(self, "features", None)
        return bool(feats.leakage_free) if feats is not None else False

    def uav_feature_dim(self) -> int:
        from skyflow.data.tkg_builder import uav_feature_names
        explicit = self.data.uav_feature_dim
        return int(explicit) if explicit is not None else len(uav_feature_names(self.leakage_free()))

    def num_relations(self) -> int:
        from skyflow.data.tkg_builder import relation_vocab
        explicit = self.model.num_relation_types
        return int(explicit) if explicit is not None else len(relation_vocab(self.leakage_free()))

    def label_mode(self) -> str:
        labels = getattr(self, "labels", None)
        return labels.mode if labels is not None else "instantaneous"

    def delta_mode(self) -> str:
        temporal = getattr(self, "temporal", None)
        return temporal.delta_mode if temporal is not None else "legacy"

    def candidates(self) -> str:
        scoring = getattr(self, "scoring", None)
        return scoring.candidates if scoring is not None else "sampled"

    def make_builder(self):
        from skyflow.data.tkg_builder import TKGBuilder
        graph = getattr(self, "graph", None) or GraphConfig()
        return TKGBuilder(
            feature_dim=self.data.uav_feature_dim,
            leakage_free=self.leakage_free(),
            delta_mode=self.delta_mode(),
            neighbor_search=graph.neighbor_search,
            input_set=getattr(self.features, "input_set", "full"),
        )

    def dataset_kwargs(self) -> dict:
        """Keyword arguments for ``UrbanAir500.generate_dataset`` derived from
        this config (builder, observation params, candidate set)."""
        scoring = getattr(self, "scoring", None) or ScoringConfig()
        return {
            "builder": self.make_builder(),
            "obs_params": self.observation_params(),
            "candidates": self.candidates(),
            "proximity_margin_m": scoring.proximity_margin_m,
        }

    def observation_params(self):
        from skyflow.data.urbanair500 import ObservationParams
        sim = getattr(self, "sim", None) or SimConfig()
        lat = sim.adsb_latency_s
        return ObservationParams(
            adsb_latency_s=tuple(lat) if isinstance(lat, (list, tuple)) else float(lat),
            packet_loss=sim.packet_loss,
            gps_cep_m=sim.gps_cep_m,
            weather_update_s=sim.weather_update_s,
            registry_update_s=sim.registry_update_s,
            corridor_update_s=sim.corridor_update_s,
        )

    def make_simulator(self, num_uavs: Optional[int] = None, seed: Optional[int] = None):
        from skyflow.data.urbanair500 import UrbanAir500
        sim = getattr(self, "sim", None) or SimConfig()
        obs = self.observation_params()
        return UrbanAir500(
            num_uavs=num_uavs if num_uavs is not None else self.data.num_uavs,
            grid_size=self.data.grid_size_m,
            num_sectors=self.data.num_sectors,
            num_weather_cells=self.data.num_weather_cells,
            num_restricted_zones=self.data.num_restricted_zones,
            dt=1.0 / self.data.sim_freq_hz,
            gps_cep=obs.gps_cep_m,
            adsb_latency_range=obs.latency_range(),
            conflict_h_sep=self.data.conflict_h_sep_m,
            conflict_v_sep=self.data.conflict_v_sep_m,
            seed=seed if seed is not None else self.training.seed,
            label_mode=self.label_mode(),
            lookahead_s=self.data.lookahead_seconds,
            observation_model=sim.observation_model,
            density_preset=getattr(sim, "density_preset", "dense"),
            cause_mix=getattr(sim, "cause_mix", None),
            packet_loss=obs.packet_loss,
            weather_update_s=obs.weather_update_s,
            registry_update_s=obs.registry_update_s,
            corridor_update_s=obs.corridor_update_s,
        )

    # ------------------------------------------------------------------ #
    @classmethod
    def from_yaml(cls, path: str | Path) -> "SkyFlowConfig":
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        cfg = cls()
        for section_name in _SECTIONS:
            if section_name in raw and raw[section_name]:
                section = getattr(cfg, section_name)
                for k, v in raw[section_name].items():
                    if hasattr(section, k):
                        setattr(section, k, v)
        if "output_dir" in raw:
            cfg.output_dir = raw["output_dir"]
        return cfg

    def to_yaml(self, path: str | Path) -> None:
        from dataclasses import asdict
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)
