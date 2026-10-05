"""On-disk cache of labelled TKG snapshots (S7a).

A cache entry is identified by a hash of every configuration section that
influences the data (data, sim, labels, features, temporal, graph, scoring)
plus the split; it lives in ``{cache_dir}/{key}/{split}.pt`` with a
``meta.json`` next to it (config snapshot, git commit, sizes).

Snapshots are stored compactly (int32 pair indices, float16 δ not used —
all floats stay float32 so cached == live exactly) and restored to the exact
in-memory representation produced by ``UrbanAir500.generate_dataset``.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from skyflow.config import SkyFlowConfig
from skyflow.data.tkg_builder import TKGSnapshot

DATA_SECTIONS = ("data", "sim", "labels", "features", "temporal", "graph", "scoring")
_IGNORED_DATA_KEYS = {"cache_dir", "scenario_minutes_train", "scenario_minutes_val",
                      "scenario_minutes_test", "uav_feature_dim"}


def git_commit(repo: Optional[Path] = None) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo or Path(__file__).resolve().parents[2],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def data_signature(cfg: SkyFlowConfig, split: str) -> Dict:
    sig = {}
    for s in DATA_SECTIONS:
        d = asdict(getattr(cfg, s))
        if s == "data":
            d = {k: v for k, v in d.items() if k not in _IGNORED_DATA_KEYS}
            # only the requested split's scenario count matters
            d["num_scenarios"] = cfg.data.split_scenarios(split)
            for k in ("train_scenarios", "val_scenarios", "test_scenarios"):
                d.pop(k, None)
        sig[s] = d
    sig["split"] = split
    return sig


def cache_key(cfg: SkyFlowConfig, split: str) -> str:
    blob = json.dumps(data_signature(cfg, split), sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:16]


def cache_paths(cfg: SkyFlowConfig, split: str, cache_dir: Optional[str] = None) -> Tuple[Path, Path]:
    root = Path(cache_dir or cfg.data.cache_dir) / cache_key(cfg, split)
    return root / f"{split}.pt", root / f"{split}.meta.json"


# ---------------------------------------------------------------- (de)serialise
_TENSOR_FIELDS = ("node_features", "node_types", "conflict_pairs", "conflict_labels",
                  "conflict_ttc", "conflict_cause", "uav_aoi", "candidate_pairs", "missed_ttc", "missed_cause")


def _pack(snapshot: TKGSnapshot) -> Dict:
    d = {}
    for f in _TENSOR_FIELDS:
        v = getattr(snapshot, f)
        if v is None:
            d[f] = None
        elif f in ("conflict_pairs", "candidate_pairs"):
            d[f] = v.to(torch.int32).cpu()
        else:
            d[f] = v.cpu()
    d["edge_indices"] = {r: e.to(torch.int32).cpu() for r, e in snapshot.edge_indices.items()}
    d["edge_deltas"] = {r: e.cpu() for r, e in snapshot.edge_deltas.items()}
    for f in ("num_uavs", "num_nodes", "relation_names", "feature_names",
              "num_missed_positives", "build_time_ms", "num_pair_candidates"):
        d[f] = getattr(snapshot, f)
    return d


def _unpack(d: Dict) -> TKGSnapshot:
    kw = {}
    for f in _TENSOR_FIELDS:
        v = d[f]
        if v is not None and f in ("conflict_pairs", "candidate_pairs"):
            v = v.to(torch.long)
        kw[f] = v
    kw["edge_indices"] = {r: e.to(torch.long) for r, e in d["edge_indices"].items()}
    kw["edge_deltas"] = dict(d["edge_deltas"])
    for f in ("num_uavs", "num_nodes", "relation_names", "feature_names",
              "num_missed_positives", "build_time_ms", "num_pair_candidates"):
        kw[f] = d[f]
    return TKGSnapshot(**kw)


# ---------------------------------------------------------------- public API
def save_dataset(dataset: List[Tuple[TKGSnapshot, torch.Tensor]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save([_pack(s) for s, _ in dataset], path)


def load_dataset(path: Path, device: torch.device = torch.device("cpu")) -> List[Tuple[TKGSnapshot, torch.Tensor]]:
    packed = torch.load(path, map_location="cpu", weights_only=False)
    out = []
    for d in packed:
        s = _unpack(d)
        if device.type != "cpu":
            s = _to_device(s, device)
        out.append((s, s.conflict_labels))
    return out


def _to_device(s: TKGSnapshot, device: torch.device) -> TKGSnapshot:
    for f in _TENSOR_FIELDS:
        v = getattr(s, f)
        if v is not None:
            setattr(s, f, v.to(device))
    s.edge_indices = {r: e.to(device) for r, e in s.edge_indices.items()}
    s.edge_deltas = {r: e.to(device) for r, e in s.edge_deltas.items()}
    return s


def build_split(cfg: SkyFlowConfig, split: str, device: torch.device = torch.device("cpu"),
                num_scenarios: Optional[int] = None, duration: Optional[float] = None):
    sim = cfg.make_simulator(seed=getattr(cfg.data, "sim_seed", cfg.training.seed))
    n = num_scenarios if num_scenarios is not None else cfg.data.split_scenarios(split)
    dur = duration if duration is not None else cfg.data.scenario_duration_s
    return sim.generate_dataset(split, n, dur, device, **cfg.dataset_kwargs())


def get_split(cfg: SkyFlowConfig, split: str, device: torch.device = torch.device("cpu"),
              cache_dir: Optional[str] = None, build_if_missing: bool = True,
              verbose: bool = True) -> List[Tuple[TKGSnapshot, torch.Tensor]]:
    """Load the split from cache, building (and caching) it if missing."""
    pt, meta = cache_paths(cfg, split, cache_dir)
    if pt.exists():
        if verbose:
            print(f"[cache] load {split} from {pt}")
        return load_dataset(pt, device)
    if not build_if_missing:
        raise FileNotFoundError(f"no cache for split={split} at {pt}")
    if verbose:
        print(f"[cache] building {split} (N={cfg.data.num_uavs}, "
              f"{cfg.data.split_scenarios(split)} x {cfg.data.scenario_duration_s:.0f} s) ...")
    t0 = time.perf_counter()
    data = build_split(cfg, split)
    build_s = time.perf_counter() - t0
    save_dataset(data, pt)
    n_pos = int(sum(float(l.sum()) for _, l in data))
    n_missed = int(sum(s.num_missed_positives for s, _ in data))
    n_pairs = int(sum(s.conflict_pairs.shape[1] for s, _ in data))
    meta_d = {
        "split": split,
        "key": cache_key(cfg, split),
        "signature": data_signature(cfg, split),
        "num_snapshots": len(data),
        "num_pairs": n_pairs,
        "num_positives": n_pos,
        "num_missed_positives": n_missed,
        "positive_rate": n_pos / max(n_pairs, 1),
        "build_seconds": build_s,
        "file_bytes": pt.stat().st_size,
        "git_commit": git_commit(),
        "machine": platform.node(),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(meta, "w", encoding="utf-8") as f:
        json.dump(meta_d, f, indent=2)
    if verbose:
        print(f"[cache] saved {pt} ({pt.stat().st_size / 1e6:.1f} MB, {len(data)} snapshots, "
              f"{n_pos} positives, {build_s:.0f} s)")
    if device.type != "cpu":
        data = [(_to_device(s, device), s.conflict_labels) for s, _ in data]
    return data


def cache_size_bytes(cache_dir: str) -> int:
    root = Path(cache_dir)
    if not root.exists():
        return 0
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
