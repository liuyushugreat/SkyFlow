"""S16: paper figures, generated only from results/ CSV/JSON (no numbers typed by hand).

    python scripts/make_figures.py --results_dir results --out_dir paper/figs

Outputs (PDF, single-column 3.5 in, 8 pt, TrueType fonts - no Type 3):
  fig_robustness.pdf    CDR vs ADS-B latency / packet loss, mean +- std over seeds
  fig_scaling.pdf       log-log P95 latency (total + 3 stages) vs N, fitted alpha, 200 ms budget line
  fig_attention_aoi.pdf in-degree-normalised attention vs AoI delta, one line per relation
  fig_lead.pdf          fraction of conflict events alerted with >= x s lead (event-level, S8f)
  fig_soc.pdf           SOC curve: event CDR vs false alert episodes per UAV-hour, raw and with the operational layer
  fig_results.pdf       full-width 4-panel figure (SOC | CDR vs latency | CDR vs loss | scaling) for the 4-page budget
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
    "Plan-CPA": ("+", ":"),
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
    panels = _robustness_panels(lat_csv, loss_csv)   # nominal latency L; actual draw is U(L(1-j), L(1+j))
    if not panels:
        print("[skip] no robustness csv"); return None
    fig, axes = plt.subplots(1, len(panels), figsize=(COL_W, 1.55), sharey=True)
    axes = np.atleast_1d(axes)
    meta = {}
    for ax, (d, xlabel) in zip(axes, panels):
        meta.update(_draw_robustness_panel(ax, d, xlabel, methods))
    axes[0].set_ylabel("CDR")
    axes[0].legend(handlelength=1.8, loc="best")
    _save(fig, out)
    return meta


def _robustness_panels(lat_csv, loss_csv):
    panels = []
    if lat_csv.exists():
        d = pd.read_csv(lat_csv)
        d["x"] = d["value"]
        panels.append((d, "ADS-B latency $L$ (s)"))
    if loss_csv.exists():
        d = pd.read_csv(loss_csv)
        d["x"] = d["value"] * 100.0
        panels.append((d, "Packet loss (%)"))
    return panels


def _draw_robustness_panel(ax, d, xlabel, methods):
    meta = {}
    for m in methods:
        dm = d[d["method"] == m]
        if dm.empty:
            continue
        g = dm.groupby("x")["cdr"].agg(["mean", "std", "count"]).reset_index().sort_values("x")
        mk, ls = _style(m)
        ax.errorbar(g["x"], g["mean"], yerr=g["std"].fillna(0.0), marker=mk, ls=ls, capsize=1.5,
                    elinewidth=0.6, label=m)
        meta[f"{xlabel}|{m}"] = g.to_dict("records")
    # S8f: shade the levels beyond the training link mix (extrapolation)
    if "in_train_range" in d.columns and (d["in_train_range"] == 0).any() and (d["in_train_range"] == 1).any():
        x_in = float(d[d["in_train_range"] == 1]["x"].max())
        x_out = sorted(d[d["in_train_range"] == 0]["x"].unique())
        edge = 0.5 * (x_in + x_out[0])
        ax.axvspan(edge, x_out[-1] + 0.5 * (x_out[-1] - x_in) / max(len(x_out), 1), color="0.85", lw=0, zorder=0)
        ax.text(0.98, 0.04, "beyond\ntraining", transform=ax.transAxes, fontsize=6, ha="right", va="bottom",
                color="0.35")
        meta[f"{xlabel}|train_max"] = x_in
    ax.set_xlabel(xlabel)
    ax.grid(alpha=0.3, lw=0.4)
    return meta


# --------------------------------------------------------------------------- scaling
def fig_scaling(csv, fit_json, out):
    if not csv.exists():
        print("[skip] no scaling csv"); return None
    fig, ax = plt.subplots(figsize=(COL_W, 1.8))
    meta = _draw_scaling(ax, csv, fit_json, legend_ncol=2)
    _save(fig, out)
    return meta


def _draw_scaling(ax, csv, fit_json, legend_ncol=2, legend_fontsize=None, legend_loc="best"):
    d = pd.read_csv(csv)
    d = d[d["status"] == "ok"].sort_values("num_uavs")
    fits = json.load(open(fit_json))["fits"] if fit_json.exists() else {}
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
    ax.legend(ncol=legend_ncol, handlelength=1.6, columnspacing=0.6, loc=legend_loc, fontsize=legend_fontsize)
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


# --------------------------------------------------------------------------- lead time (S8f)
def fig_lead(npz, events_csv, methods, out, persistence=1):
    """Fraction of conflict events alerted with at least x seconds of lead
    (per method, lead samples pooled over seeds; undetected events count as
    lead 0, so the curve starts at the event CDR)."""
    if not npz.exists() or not events_csv.exists():
        print("[skip] no events artefacts"); return None
    z = np.load(npz)
    ev = pd.read_csv(events_csv)
    ev = ev[ev["persistence"] == persistence]
    fig, ax = plt.subplots(figsize=(COL_W, 1.7))
    meta = {}
    for m in methods:
        keys = [k for k in z.files if k.startswith(f"{m}/seed") and k.endswith(f"/p{persistence}")]
        if not keys:
            continue
        n_events = float(ev[ev["method"] == m]["n_events"].iloc[0]) if (ev["method"] == m).any() else None
        leads = np.concatenate([z[k] for k in keys])
        if n_events is None or n_events <= 0:
            continue
        per_seed_events = n_events * len(keys)
        xs = np.linspace(0.0, max(float(leads.max()) if leads.size else 1.0, 1.0), 200)
        frac = np.array([(leads >= x).sum() / per_seed_events for x in xs])
        mk, ls = _style(m)
        ax.plot(xs, frac, ls=ls, marker=mk, markevery=25, label=m)
        meta[m] = {"n_seeds": len(keys), "frac_ge_0": float(frac[0]), "lead_median_s": float(np.median(leads)) if leads.size else None}
    ax.set_xlabel("Lead time at first alert (s)")
    ax.set_ylabel("Fraction of events")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3, lw=0.4)
    ax.legend(handlelength=1.8, loc="best")
    _save(fig, out)
    return meta


# --------------------------------------------------------------------------- SOC curve (S8f)
def fig_soc(soc_csv, events_csv, budget_csv, methods, out, layer_method="TR-GAT"):
    """System Operating Characteristic on the test split: event CDR versus
    false alert episodes per UAV-hour (log axis).  Solid: raw scores (alpha=1,
    no hysteresis), seeds averaged per threshold; dotted: the operational
    layer (EMA + hysteresis) of ``layer_method`` with the setting selected on
    validation for the matched budget; markers: validation-F1 operating
    points; rules are single points."""
    if not soc_csv.exists():
        print("[skip] no events_soc.csv"); return None
    fig, ax = plt.subplots(figsize=(COL_W, 1.9))
    meta = _draw_soc(ax, soc_csv, events_csv, budget_csv, methods, layer_method)
    _save(fig, out)
    return meta


def _draw_soc(ax, soc_csv, events_csv, budget_csv, methods, layer_method="TR-GAT", legend_fontsize=None):
    d = pd.read_csv(soc_csv)
    d = d[d["split"] == "test"]
    ev = pd.read_csv(events_csv) if events_csv.exists() else None
    bud = pd.read_csv(budget_csv) if budget_csv.exists() else None
    meta = {}
    for m in methods:
        dm = d[d["method"] == m]
        if dm.empty:
            continue
        mk, ls = _style(m)
        raw = dm[(dm["alpha"] == 1.0) & (dm["hysteresis"] == 0.0)]
        if raw["threshold"].nunique() <= 1:                 # rule: one operating point
            x, y = raw["false_episodes_per_uav_hour"].mean(), raw["event_cdr"].mean()
            ax.plot([x], [y], marker=mk, ls="none", ms=6, label=m)
            meta[m] = {"point": [float(x), float(y)]}
            continue
        g = raw.groupby("threshold")[["false_episodes_per_uav_hour", "event_cdr"]].mean().sort_values("false_episodes_per_uav_hour")
        g = g[g["false_episodes_per_uav_hour"] > 0]
        line, = ax.plot(g["false_episodes_per_uav_hour"], g["event_cdr"], ls=ls, label=m)
        meta[m] = {"raw_points": len(g)}
        if ev is not None:                                  # validation-F1 operating point
            e = ev[(ev["method"] == m) & (ev["persistence"] == 1)]
            if not e.empty:
                ax.plot([e["false_episodes_per_uav_hour"].mean()], [e["event_cdr"].mean()], marker=mk, ls="none",
                        color=line.get_color(), ms=5)
        if m == layer_method and bud is not None:
            b = bud[(bud["method"] == m) & (bud["feasible"] == 1)]
            kinds = sorted(b["budget_kind"].astype(str).unique(),
                           key=lambda k: (not k.startswith("match"), float(k.split("_")[-1]) if k.split("_")[-1].replace(".", "", 1).isdigit() else 0.0))
            b = b[b["budget_kind"].astype(str) == kinds[0]] if kinds else b
            if not b.empty:
                a_sel, h_sel = float(b["alpha"].mode().iloc[0]), float(b["hysteresis"].mode().iloc[0])
                lay = dm[(dm["alpha"] == a_sel) & (dm["hysteresis"] == h_sel)]
                gl = lay.groupby("threshold")[["false_episodes_per_uav_hour", "event_cdr"]].mean().sort_values("false_episodes_per_uav_hour")
                gl = gl[gl["false_episodes_per_uav_hour"] > 0]
                if not gl.empty:
                    ax.plot(gl["false_episodes_per_uav_hour"], gl["event_cdr"], ls=":", color=line.get_color(),
                            label=f"{m} + EMA/hyst.")
                    meta[m]["layer"] = {"alpha": a_sel, "hysteresis": h_sel}
                ax.plot([b["false_episodes_per_uav_hour"].mean()], [b["event_cdr"].mean()], marker="*", ls="none",
                        color=line.get_color(), ms=7)
    ax.set_xscale("log")
    ax.set_xlabel("False alert episodes per UAV-hour")
    ax.set_ylabel("Event CDR")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3, lw=0.4, which="both")
    ax.legend(handlelength=1.8, loc="lower right", ncol=1, fontsize=legend_fontsize)
    return meta


# --------------------------------------------------------------------------- combined results figure (page budget)
TEXT_W = 7.16         # IEEE two-column text width (in)


def fig_results(R, methods, out, layer_method="TR-GAT", rob_legend_loc="lower left", scal_legend_loc="lower right"):
    """One full-width figure* with up to four panels: (a) SOC curve, (b) CDR vs
    latency, (c) CDR vs packet loss, (d) P95 latency scaling.  Same data and
    drawing code as the single-column figures; panels whose CSV is missing are
    skipped so the layout degrades gracefully."""
    panels = []
    if (R / "events_soc.csv").exists():
        panels.append(("soc", None))
    for d, xlabel in _robustness_panels(R / "robustness_latency.csv", R / "robustness_loss.csv"):
        panels.append(("rob", (d, xlabel)))
    if (R / "scaling.csv").exists():
        panels.append(("scal", None))
    if not panels:
        print("[skip] combined results figure: no inputs"); return None
    widths = [1.15 if k == "soc" else (1.0 if k == "scal" else 0.85) for k, _ in panels]
    fig, axes = plt.subplots(1, len(panels), figsize=(TEXT_W, 1.38), gridspec_kw={"width_ratios": widths})
    axes = np.atleast_1d(axes)
    meta, rob_axes = {}, []
    for ax, (kind, payload), letter in zip(axes, panels, "abcdef"):
        if kind == "soc":
            meta["soc"] = _draw_soc(ax, R / "events_soc.csv", R / "events.csv", R / "events_budget.csv", methods,
                                    layer_method, legend_fontsize=6)
        elif kind == "rob":
            d, xlabel = payload
            meta.setdefault("robustness", {}).update(_draw_robustness_panel(ax, d, xlabel, methods))
            rob_axes.append(ax)
        else:
            meta["scaling"] = _draw_scaling(ax, R / "scaling.csv", R / "scaling_fit.json", legend_ncol=1,
                                            legend_fontsize=6, legend_loc=scal_legend_loc)
        ax.set_title(f"({letter})", fontsize=8, loc="left", pad=2)
    if rob_axes:
        rob_axes[0].set_ylabel("CDR")
        # legend in the last robustness panel (loss) - the curves of the latency panel are the ones to read
        rob_axes[-1].legend(handlelength=1.4, loc=rob_legend_loc, fontsize=5.5, labelspacing=0.25)
        for ax in rob_axes[1:]:
            ax.sharey(rob_axes[0])
            ax.tick_params(labelleft=False)
    fig.subplots_adjust(wspace=0.32)
    _save(fig, out)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out_dir", default="paper/figs")
    ap.add_argument("--methods", nargs="+", default=["TR-GAT", "TR-GAT-NT", "GAT-S", "CPA-Rule", "Plan-CPA"])
    ap.add_argument("--attention_layer", type=int, default=None)
    ap.add_argument("--rob_legend_loc", default="lower left", help="legend position in the loss panel of fig_results")
    ap.add_argument("--scal_legend_loc", default="lower right", help="legend position in the scaling panel of fig_results")
    args = ap.parse_args()
    R, O = Path(args.results_dir), Path(args.out_dir)
    info = {
        "robustness": fig_robustness(R / "robustness_latency.csv", R / "robustness_loss.csv", args.methods,
                                     O / "fig_robustness.pdf"),
        "scaling": fig_scaling(R / "scaling.csv", R / "scaling_fit.json", O / "fig_scaling.pdf"),
        "attention": fig_attention(R / "attention_vs_aoi.csv", O / "fig_attention_aoi.pdf", args.attention_layer),
        "lead": fig_lead(R / "events_lead.npz", R / "events.csv", args.methods, O / "fig_lead.pdf"),
        "soc": fig_soc(R / "events_soc.csv", R / "events.csv", R / "events_budget.csv", args.methods, O / "fig_soc.pdf"),
        "results": fig_results(R, args.methods, O / "fig_results.pdf", rob_legend_loc=args.rob_legend_loc,
                               scal_legend_loc=args.scal_legend_loc),
    }
    O.mkdir(parents=True, exist_ok=True)
    json.dump({"results_dir": str(R), **info}, open(O / "figures_provenance.json", "w"), indent=2, default=str)


if __name__ == "__main__":
    main()
