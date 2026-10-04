#!/usr/bin/env python3
"""Leakage-control ablation study.

Retrains TR-GAT with all proximity/CPA-derived input features removed
(n_nbr, d_min, t_cpa, f_avoid; feature indices 19-22 of Eq. 1) and compares
against (a) full TR-GAT and (b) GAT-Static under identical matched conditions:
same data, seeds, optimizer, epochs, threshold.

Scaled-down matched protocol (local RTX 4090): 200 UAVs, dense 500 m grid,
conflict prevalence ~3.7% (paper: 3.1%), 3 seeds.
"""

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Resolve the SkyFlow package root relative to this file.
SKYFLOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKYFLOW_ROOT))

from skyflow.data.urbanair500 import UrbanAir500
from skyflow.models.tr_gat import TRGAT
from skyflow.models.conflict_head import ConflictScoringHead
from skyflow.baselines.gat_static import GATStatic
from skyflow.training.losses import FocalLoss
from skyflow.training.metrics import ConflictMetrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CPA_FEATURE_IDS = [19, 20, 21, 22]   # n_nbr, d_min, t_cpa, f_avoid
THRESHOLD = 0.42
EPOCHS = 80
WARMUP_STEPS = 200
K_WINDOW = 10
SEEDS = [42, 123, 456]
CACHE = Path(__file__).parent / "dataset_cache.pt"
OUT = Path(__file__).parent / "leakage_ablation_results.json"


def gen_data():
    if CACHE.exists():
        print(f"[{time.strftime('%H:%M:%S')}] Loading cached datasets...", flush=True)
        blob = torch.load(CACHE, map_location=DEVICE, weights_only=False)
        return blob["train"], blob["val"], blob["test"]
    print(f"[{time.strftime('%H:%M:%S')}] Generating datasets...", flush=True)
    sim = UrbanAir500(num_uavs=200, grid_size=500.0, altitude_range=(74.0, 78.0), seed=42)
    train = sim.generate_dataset("train", 6, 60.0, DEVICE)
    val = sim.generate_dataset("val", 2, 60.0, DEVICE)
    test = sim.generate_dataset("test", 2, 60.0, DEVICE)
    for name, ds in [("train", train), ("val", val), ("test", test)]:
        pos = sum(d[1].sum().item() for d in ds)
        tot = sum(d[1].numel() for d in ds)
        print(f"  {name}: {len(ds)} snapshots, positives {pos:.0f}/{tot} ({100*pos/tot:.2f}%)", flush=True)
    torch.save({"train": train, "val": val, "test": test}, CACHE)
    return train, val, test


def mask_dataset(data):
    """Zero out CPA-derived features on UAV rows; graph structure unchanged."""
    masked = []
    for snap, labels in data:
        snap2 = copy.copy(snap)
        nf = snap.node_features.clone()
        nf[: snap.num_uavs, CPA_FEATURE_IDS] = 0.0
        snap2.node_features = nf
        masked.append((snap2, labels))
    return masked


def windows(data, K):
    out = []
    for start in range(0, len(data) - K + 1, K):
        out.append(data[start : start + K])
    if len(data) >= K and len(data) % K != 0:
        out.append(data[-K:])
    if not out and data:
        out.append(data)
    return out


def train_trgat(train_data, val_data, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = TRGAT(node_feature_dim=23).to(DEVICE)
    head = ConflictScoringHead().to(DEVICE)
    params = list(model.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=1e-5)
    n_windows = max(len(train_data) // K_WINDOW, 1)
    total_steps = EPOCHS * n_windows
    from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
    warmup = LambdaLR(opt, lr_lambda=lambda s: min(float(s) / WARMUP_STEPS, 1.0))
    cosine = CosineAnnealingLR(opt, T_max=max(total_steps - WARMUP_STEPS, 1))
    sched = SequentialLR(opt, schedulers=[warmup, cosine], milestones=[WARMUP_STEPS])
    crit = FocalLoss(gamma=2.0)

    best_f1, best_state = -1.0, None
    order = list(range(len(train_data)))
    for ep in range(EPOCHS):
        model.train(); head.train()
        wins = windows(train_data, K_WINDOW)
        np.random.shuffle(wins)
        for win in wins:
            opt.zero_grad()
            loss_sum, steps, rec = 0.0, 0, None
            for snap, labels in win:
                emb, rec = model(snap.node_features, snap.edge_indices, snap.edge_deltas, recurrent_state=rec)
                pairs = snap.conflict_pairs
                if pairs is None or pairs.size(1) == 0:
                    rec = rec.detach(); continue
                preds = head(emb[pairs[0]], emb[pairs[1]], rec[pairs[0]], rec[pairs[1]])
                loss_sum = loss_sum + crit(preds, labels)
                steps += 1
                rec = rec.detach()
            if steps:
                (loss_sum / steps).backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
        if (ep + 1) % 5 == 0 or ep == EPOCHS - 1:
            m = eval_trgat(model, head, val_data)
            print(f"    ep {ep+1}: val CDR={m.cdr:.4f} F1={m.f1:.4f} FAR={m.far:.4f}", flush=True)
            if m.f1 > best_f1:
                best_f1 = m.f1
                best_state = (copy.deepcopy(model.state_dict()), copy.deepcopy(head.state_dict()))
    if best_state:
        model.load_state_dict(best_state[0]); head.load_state_dict(best_state[1])
    return model, head


@torch.no_grad()
def eval_trgat(model, head, data):
    model.eval(); head.eval()
    metrics = ConflictMetrics(threshold=THRESHOLD)
    for win in windows(data, K_WINDOW):
        rec = None
        for snap, labels in win:
            emb, rec = model(snap.node_features, snap.edge_indices, snap.edge_deltas, recurrent_state=rec)
            pairs = snap.conflict_pairs
            if pairs is None or pairs.size(1) == 0:
                continue
            preds = head(emb[pairs[0]], emb[pairs[1]], rec[pairs[0]], rec[pairs[1]])
            metrics.update(preds, labels)
    return metrics.compute()


def train_gats(train_data, val_data, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = GATStatic(input_dim=23).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    crit = FocalLoss(gamma=2.0)
    best_f1, best_state = -1.0, None
    for ep in range(EPOCHS):
        model.train()
        idx = np.random.permutation(len(train_data))
        for i in idx:
            snap, labels = train_data[i]
            opt.zero_grad()
            preds = model(snap)
            if preds.numel() == 0:
                continue
            loss = crit(preds, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if (ep + 1) % 5 == 0 or ep == EPOCHS - 1:
            m = eval_gats(model, val_data)
            print(f"    ep {ep+1}: val CDR={m.cdr:.4f} F1={m.f1:.4f} FAR={m.far:.4f}", flush=True)
            if m.f1 > best_f1:
                best_f1 = m.f1
                best_state = copy.deepcopy(model.state_dict())
    if best_state:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def eval_gats(model, data):
    model.eval()
    metrics = ConflictMetrics(threshold=THRESHOLD)
    for snap, labels in data:
        preds = model(snap)
        if preds.numel() == 0:
            continue
        metrics.update(preds, labels)
    return metrics.compute()


def main():
    print(f"Device: {DEVICE}", flush=True)
    train_full, val_full, test_full = gen_data()
    train_mask = mask_dataset(train_full)
    val_mask = mask_dataset(val_full)
    test_mask = mask_dataset(test_full)

    results = {"TR-GAT-full": [], "TR-GAT-noCPA": []}
    for seed in SEEDS:
        print(f"\n=== seed {seed} ===", flush=True)
        print("  TR-GAT full features", flush=True)
        m, h = train_trgat(train_full, val_full, seed)
        r = eval_trgat(m, h, test_full)
        results["TR-GAT-full"].append({"cdr": r.cdr, "far": r.far, "f1": r.f1})
        print(f"  -> test CDR={r.cdr:.4f} F1={r.f1:.4f} FAR={r.far:.4f}", flush=True)

        print("  TR-GAT no CPA features", flush=True)
        m, h = train_trgat(train_mask, val_mask, seed)
        r = eval_trgat(m, h, test_mask)
        results["TR-GAT-noCPA"].append({"cdr": r.cdr, "far": r.far, "f1": r.f1})
        print(f"  -> test CDR={r.cdr:.4f} F1={r.f1:.4f} FAR={r.far:.4f}", flush=True)

        with open(OUT, "w") as f:
            json.dump(results, f, indent=2)

    print("\n===== SUMMARY (mean over seeds) =====", flush=True)
    summary = {}
    for name, runs in results.items():
        cdr = np.mean([x["cdr"] for x in runs]); f1 = np.mean([x["f1"] for x in runs])
        far = np.mean([x["far"] for x in runs])
        cdr_s = np.std([x["cdr"] for x in runs]); f1_s = np.std([x["f1"] for x in runs])
        summary[name] = {"cdr_mean": cdr, "cdr_std": cdr_s, "f1_mean": f1, "f1_std": f1_s, "far_mean": far}
        print(f"{name:16s} CDR={cdr:.4f}±{cdr_s:.4f}  F1={f1:.4f}±{f1_s:.4f}  FAR={far:.4f}", flush=True)

    try:
        from scipy import stats
        for metric in ("cdr", "f1"):
            a = [x[metric] for x in results["TR-GAT-full"]]
            b = [x[metric] for x in results["TR-GAT-noCPA"]]
            t, p = stats.ttest_rel(a, b)
            summary[f"ttest_full_vs_noCPA_{metric}"] = {"t": float(t), "p": float(p)}
            print(f"paired t-test full vs noCPA on {metric}: t={t:.2f}, p={p:.4f}", flush=True)
    except Exception as e:
        print("t-test skipped:", e, flush=True)

    with open(OUT, "w") as f:
        json.dump({"runs": results, "summary": summary}, f, indent=2)
    print(f"Saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
