"""S16: paper figures, generated only from results/ CSV/JSON (no numbers typed by hand).

    python scripts/make_figures.py --results_dir results --out_dir paper/figs

Outputs (PDF, single-column 3.5 in, 8 pt, TrueType fonts - no Type 3):
  fig_robustness.pdf    CDR vs ADS-B latency / packet loss, mean +- std over seeds
  fig_scaling.pdf       log-log P95 latency (total + 3 stages) vs N, fitted alpha, 200 ms budget line
  fig_attention_aoi.pdf in-degree-normalised attention vs AoI delta, one line per relation
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,          # TrueType, never Type 3
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.6, "lines.linewidth": 1.0, "lines.markersize": 3.5,
    "legend.frameon": False,
})
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

COL_W = 3.5           # IEEE single column (in)
BUDGET_MS = 200.0     # edge compute budget used in the paper (design target, excludes link delay)
STYLE = {             # method -> (marker, linestyle); colours left to the default cycle
    "TR-GAT": ("o", "-"), "TR-GAT-NT": ("s", "--"), "GAT-S": ("^", "-."), "CPA-Rule": ("x", ":"),
    "STGCN": ("v", "--"), "LSTM-P": ("D", "-."), "Tfm-P": ("P", ":"), "VO": ("*", ":"),
}


def _style(m):
    return STYLE.get(m, ("o", "-"))


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("written:", path)


# --------------------------------------------------------------------------- robustness
def fig_robustness(lat_csv, loss_csv, methods, out):
    panels = []
    if lat_csv.exists():
        d = pd.read_csv(lat_csv)
        d["x"] = d["value"]                        # nominal latency L; actual draw is U(L(1-j), L(1+j))
        panels.append((d, "ADS-B latency $L$ (s)"))
    if loss_csv.exists():
        d = pd.read_csv(loss_csv)
        d["x"] = d["value"] * 100.0
        panels.append((d, "Packet loss (%)"))
    if not panels:
        print("[skip] no robustness csv"); return None
    fig, axes = plt.subplots(1, len(panels), figsize=(COL_W, 1.55), sharey=True)
    axes = np.atleast_1d(axes)
    meta = {}
    for ax, (d, xlabel) in zip(axes, panels):
        for m in methods:
            dm = d[d["method"] == m]
            if dm.empty:
                continue
            g = dm.groupby("x")["cdr"].agg(["mean", "std", "count"]).reset_index().sort_values("x")
            mk, ls = _style(m)
            ax.errorbar(g["x"], g["mean"], yerr=g["std"].fillna(0.0), marker=mk, ls=ls, capsize=1.5,
                        elinewidth=0.6, label=m)
            meta[f"{xlabel}|{m}"] = g.to_dict("records")
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.3, lw=0.4)
    axes[0].set_ylabel("CDR")
    axes[0].legend(handlelength=1.8, loc="best")
    _save(fig, out)
    return meta


# --------------------------------------------------------------------------- scaling
def fig_scaling(csv, fit_json, out):
    if not csv.exists():
        print("[skip] no scaling csv"); return None
    d = pd.read_csv(csv)
    d = d[d["status"] == "ok"].sort_values("num_uavs")
    fits = json.load(open(fit_json))["fits"] if fit_json.exists() else {}
    fig, ax = plt.subplots(figsize=(COL_W, 1.8))
    series = [("total_p95_ms", "total", "o", "-"), ("graph_build_p95_ms", "graph build", "s", "--"),
              ("gnn_forward_p95_ms", "GNN forward", "^", "-."), ("pair_scoring_p95_ms", "pair scoring", "v", ":")]
    for col, name, mk, ls in series:
        a = fits.get(col, {}).get("alpha")
        lab = f"{name} ($\\alpha$={a:.2f})" if a is not None and np.isfinite(a) else name
        ax.loglog(d["num_uavs"], d[col], marker=mk, ls=ls, label=lab)
    ax.axhline(BUDGET_MS, color="k", ls="--", lw=0.7)
    ax.text(d["num_uavs"].max(), BUDGET_MS * 1.12, f"{BUDGET_MS:.0f} ms budget", fontsize=6.5,
            va="bottom", ha="right")
    ax.set_xlabel("Fleet size $N$")
    ax.set_ylabel("P95 latency (ms)")
    ax.set_xticks(d["num_uavs"].tolist())
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.grid(alpha=0.3, lw=0.4, which="major")
    ax.legend(ncol=2, handlelength=1.8, columnspacing=0.8, loc="best")
    _save(fig, out)
    return {"sizes": d["num_uavs"].tolist(), "alpha": {k: v.get("alpha") for k, v in fits.items()}}


# --------------------------------------------------------------------------- attention vs AoI
def fig_attention(csv, out, layer=None, min_edges=20):
    if not csv.exists():
        print("[skip] no attention csv"); return None
    d = pd.read_csv(csv)
    if layer is None:
        layer = int(d["layer"].max())              # last layer by default
    d = d[(d["layer"] == layer) & (d["n_edges"] >= min_edges)]
    if d.empty:
        print("[skip] attention csv has no rows for layer", layer); return None
    fig, ax = plt.subplots(figsize=(COL_W, 1.7))
    rels = list(dict.fromkeys(d.sort_values("relation")["relation_name"]))
    markers = ["o", "s", "^", "v", "D"]
    for i, r in enumerate(rels):
        dr = d[d["relation_name"] == r].sort_values("delta_bin_lo_s")
        x = 0.5 * (dr["delta_bin_lo_s"] + dr["delta_bin_hi_s"])
        ax.plot(x, dr["attn_norm_mean"], marker=markers[i % len(markers)], label=r.replace("_", " "))
    ax.axhline(1.0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("Age of information $\\delta$ (s)")
    ax.set_ylabel("Attention / uniform")
    ax.grid(alpha=0.3, lw=0.4)
    ax.legend(ncol=2, handlelength=1.8, columnspacing=0.8)
    _save(fig, out)
    return {"layer": layer, "relations": rels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out_dir", default="paper/figs")
    ap.add_argument("--methods", nargs="+", default=["TR-GAT", "TR-GAT-NT", "GAT-S", "CPA-Rule"])
    ap.add_argument("--attention_layer", type=int, default=None)
    args = ap.parse_args()
    R, O = Path(args.results_dir), Path(args.out_dir)
    info = {
        "robustness": fig_robustness(R / "robustness_latency.csv", R / "robustness_loss.csv", args.methods,
                                     O / "fig_robustness.pdf"),
        "scaling": fig_scaling(R / "scaling.csv", R / "scaling_fit.json", O / "fig_scaling.pdf"),
        "attention": fig_attention(R / "attention_vs_aoi.csv", O / "fig_attention_aoi.pdf", args.attention_layer),
    }
    O.mkdir(parents=True, exist_ok=True)
    json.dump({"results_dir": str(R), **info}, open(O / "figures_provenance.json", "w"), indent=2, default=str)


if __name__ == "__main__":
    main()
