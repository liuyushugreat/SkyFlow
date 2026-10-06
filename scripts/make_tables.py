"""S16: booktabs tables generated only from results/ artefacts and the run config.

    python scripts/make_tables.py --results_dir results --out_dir paper/tables

  tab_main.tex      methods x (CDR, FAR, F1, AUPRC, P95 latency, hard-regime CDR); best per column bold
  tab_ablation.tex  TR-GAT variants x (CDR, FAR, F1, dF1, p)
  tab_events.tex    methods x event-level metrics (event CDR, timely CDR, lead, episode precision, false ep./UAV-h)
  tab_setup.tex     ~10-row experimental setup (dataset, labels, model, training, hardware)
"""

import argparse
import json
import math
import sys
from pathlib import Path

from dataclasses import asdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_common import (ABL_LABEL, ABL_ORDER, MAIN_ORDER, Results, fmt, fmt_pm, tex_escape)  # noqa: E402
from skyflow.config import SkyFlowConfig  # noqa: E402


def _bold_mask(values, higher_better):
    vals = np.array([v if (v is not None and math.isfinite(v)) else np.nan for v in values], dtype=float)
    if np.all(np.isnan(vals)):
        return [False] * len(vals)
    best = np.nanmax(vals) if higher_better else np.nanmin(vals)
    return [bool(np.isfinite(v) and abs(v - best) < 1e-12) for v in vals]


def _b(s, flag):
    return f"\\textbf{{{s}}}" if flag else s


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print("written:", path)


# --------------------------------------------------------------------------- main table
def tab_main(res: Results, out: Path, methods):
    rows = [(m, res.row(m)) for m in methods if res.row(m) is not None]
    if not rows:
        print("[skip] main table: no summary"); return
    use_lat = "latency_p95_ms" in res.summary.columns and res.summary["latency_p95_ms"].notna().any()
    use_auprc = "auprc_mean" in res.summary.columns
    hard = {m: res.regime_cdr(m, "hard") for m, _ in rows}
    cols = {
        "cdr": ([r["cdr_mean"] for _, r in rows], True),
        "far": ([r["far_mean"] for _, r in rows], False),
        "f1": ([r["f1_mean"] for _, r in rows], True),
        "auprc": ([r.get("auprc_mean") for _, r in rows], True),
        "lat": ([r.get("latency_p95_ms") for _, r in rows], False),
        "hard": ([hard[m][0] for m, _ in rows], True),
    }
    bold = {k: _bold_mask(v, hb) for k, (v, hb) in cols.items()}
    n_seeds = sorted({int(r["n_seeds"]) for _, r in rows})
    ref = (res.tests or {}).get("reference", "TR-GAT")

    use_params = "params" in res.summary.columns and res.summary["params"].notna().any()
    # column order: threshold-free AUPRC first, then the operating point, hard regime, cost
    head, spec = ["Method"], "l"
    if use_auprc:
        head.append("AUPRC$\\uparrow$"); spec += "c"
    head += ["CDR$\\uparrow$", "FAR$\\downarrow$", "F1$\\uparrow$", "Hard CDR$\\uparrow$"]; spec += "cccc"
    if use_params:
        head.append("Par.\\,(M)"); spec += "c"
    if use_lat:
        head.append("P95\\,(ms)$\\downarrow$"); spec += "c"
    lines = ["\\begin{tabular}{" + spec + "}", "\\toprule", " & ".join(head) + " \\\\", "\\midrule"]
    for i, (m, r) in enumerate(rows):
        f1 = _b(fmt_pm(r["f1_mean"], r["f1_std"]), bold["f1"][i])
        if m != ref:   # significance marker on F1 (paired t-test vs reference, Bonferroni)
            p = res.p_value("f1", m)
            if p is not None and math.isfinite(p):
                f1 += "$^{*}$" if p < 0.05 else "$^{\\dagger}$"
        cells = [tex_escape(m) if m != ref else f"\\textbf{{{tex_escape(m)}}}"]
        if use_auprc:
            cells.append(_b(fmt_pm(r.get("auprc_mean"), r.get("auprc_std")), bold["auprc"][i]))
        cells += [_b(fmt_pm(r["cdr_mean"], r["cdr_std"]), bold["cdr"][i]),
                  _b(fmt_pm(r["far_mean"], r["far_std"]), bold["far"][i]), f1,
                  _b(fmt_pm(*hard[m]), bold["hard"][i])]
        if use_params:
            pm = r.get("params")
            cells.append(f"{float(pm) / 1e6:.2f}" if pm is not None and math.isfinite(float(pm)) and float(pm) > 0 else "--")
        if use_lat:
            cells.append(_b(fmt(r.get("latency_p95_ms"), 1), bold["lat"][i]))
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]

    env = (res.any_task_metrics() or {}).get("env", {})
    hosts = (res.tests or {}).get("hosts") or []
    gpu = hosts[0][1] if hosts and hosts[0][1] else env.get("gpu_model", "")
    note = (f"% auto-generated from {res.dir}; do not edit\n"
            f"% n_seeds={n_seeds}; latency source={rows[0][1].get('latency_source', 'n/a')}; GPU={gpu}; "
            f"CPU={env.get('cpu_model', '')}; torch={env.get('torch', '')}; git={rows[0][1].get('git_commits', '')}\n"
            f"% * : p<0.05 vs {ref} (paired t-test over seeds, Bonferroni); dagger : not significant\n")
    _write(out, note + "\n".join(lines) + "\n")
    # companion caption fragment as a macro (\input it in the preamble, use \tabMainNote in \caption)
    _write(out.with_name(out.stem + "_note.tex"),
           "\\newcommand{\\tabMainNote}{"
           f"Latency P95 measured on one {tex_escape(gpu)}. "
           f"$^{{*}}$: $p<0.05$ vs.\\ {tex_escape(ref)} (paired $t$-test over seeds, Bonferroni); $^{{\\dagger}}$: n.s.}}\n")


# --------------------------------------------------------------------------- ablation table
def tab_ablation(res: Results, out: Path):
    if res.ablation is None:
        print("[skip] ablation table"); return
    d = res.ablation.set_index("method")
    order = [m for m in ABL_ORDER if m in d.index]
    lines = ["\\begin{tabular}{lcccc}", "\\toprule",
             "Variant & CDR$\\uparrow$ & FAR$\\downarrow$ & F1$\\uparrow$ & $\\Delta$F1 \\\\", "\\midrule"]
    f1s = [d.loc[m, "f1_mean"] for m in order]
    bold = _bold_mask(f1s, True)
    for i, m in enumerate(order):
        r = d.loc[m]
        dF1 = "--" if m == "TR-GAT" else f"{r['d_f1']:+.3f}"
        p = r.get("p_f1_bonf")
        if m != "TR-GAT" and p is not None and math.isfinite(p):
            dF1 += "$^{*}$" if p < 0.05 else "$^{\\dagger}$"
        lines.append(" & ".join([ABL_LABEL.get(m, tex_escape(m)),
                                 fmt_pm(r["cdr_mean"], r["cdr_std"]), fmt_pm(r["far_mean"], r["far_std"]),
                                 _b(fmt_pm(r["f1_mean"], r["f1_std"]), bold[i]), dF1]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    _write(out, f"% auto-generated from {res.dir}; do not edit\n" + "\n".join(lines) + "\n")


# --------------------------------------------------------------------------- event-level table (S8f)
def tab_events(res: Results, out: Path, methods, persistence=1):
    rows = res.events_rows(persistence)
    rows = [(m, rows[m]) for m in methods if m in rows]
    if not rows:
        print("[skip] events table: no events_summary.csv"); return
    ref = (res.events_meta or {}).get("reference", "TR-GAT")
    lead_s = (res.events_meta or {}).get("lead_s", 10)
    # budget-matched block: the matched-rule budget if available, else the first fixed budget
    kinds = res.budget_kinds()
    bkind = kinds[0] if kinds else None
    brows = res.budget_rows(bkind) if bkind else {}
    cols = {
        "ev": ([r["event_cdr_mean"] for _, r in rows], True),
        "tm": ([r["timely_cdr_mean"] for _, r in rows], True),
        "ld": ([r["lead_median_s_mean"] for _, r in rows], True),
        "fa": ([r["false_episodes_per_uav_hour_mean"] for _, r in rows], False),
        "bev": ([brows.get(m, {}).get("event_cdr_mean") for m, _ in rows], True),
        "bld": ([brows.get(m, {}).get("lead_median_s_mean") for m, _ in rows], True),
    }
    bold = {k: _bold_mask(v, hb) for k, (v, hb) in cols.items()}
    head = ["Method", "Event CDR$\\uparrow$", f"Timely$\\uparrow$", "Lead (s)$\\uparrow$", "FA/UAV-h$\\downarrow$"]
    spec = "lcccc"
    if brows:
        head += ["Event CDR$\\uparrow$", "Lead (s)$\\uparrow$"]; spec += "cc"
    lines = ["\\begin{tabular}{" + spec + "}", "\\toprule"]
    if brows:
        lines.append("& \\multicolumn{4}{c}{validation-F1 threshold} & \\multicolumn{2}{c}{matched FA budget} \\\\")
        lines.append("\\cmidrule(lr){2-5}\\cmidrule(lr){6-7}")
    lines += [" & ".join(head) + " \\\\", "\\midrule"]
    for i, (m, r) in enumerate(rows):
        ev = fmt_pm(r["event_cdr_mean"], r["event_cdr_std"])
        if m != ref:
            p = res.events_p_value("event_cdr", m, persistence)
            if p is not None and math.isfinite(p):
                ev += "$^{*}$" if p < 0.05 else "$^{\\dagger}$"
        cells = [tex_escape(m) if m != ref else f"\\textbf{{{tex_escape(m)}}}",
                 _b(ev, bold["ev"][i]),
                 _b(fmt_pm(r["timely_cdr_mean"], r["timely_cdr_std"]), bold["tm"][i]),
                 _b(fmt(r["lead_median_s_mean"], 1), bold["ld"][i]),
                 _b(fmt(r["false_episodes_per_uav_hour_mean"], 1), bold["fa"][i])]
        if brows:
            b = brows.get(m)
            if b is None:
                cells += ["--", "--"]
            else:
                bev = fmt_pm(b["event_cdr_mean"], b["event_cdr_std"])
                if m != ref:
                    p = res.budget_p_value("event_cdr", m, bkind)
                    if p is not None and math.isfinite(p):
                        bev += "$^{*}$" if p < 0.05 else "$^{\\dagger}$"
                cells += [_b(bev, bold["bev"][i]), _b(fmt(b["lead_median_s_mean"], 1), bold["bld"][i])]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    n_ev = rows[0][1].get("n_events")
    bval = ""
    if brows:
        any_b = next(iter(brows.values()))
        bval = f"; budget kind={bkind}, value={any_b.get('budget_mean')}"
    note = (f"% auto-generated from {res.dir}; do not edit\n"
            f"% persistence M={persistence}; events={n_ev}; UAV-hours={rows[0][1].get('uav_hours')}; timely = lead >= {lead_s} s"
            f"{bval}; * : p<0.05 vs {ref} on event CDR (paired t over seeds, Bonferroni)\n")
    _write(out, note + "\n".join(lines) + "\n")


# --------------------------------------------------------------------------- setup table
def tab_setup(res: Results, cfg_path: Path, out: Path):
    cfg = asdict(SkyFlowConfig.from_yaml(str(cfg_path)))     # merged with defaults (partial yaml allowed)
    d, s, t, m = cfg["data"], cfg["sim"], cfg["training"], cfg["model"]
    tm = res.any_task_metrics("TR-GAT") or res.any_task_metrics() or {}
    ds, env, test = tm.get("dataset", {}), tm.get("env", {}), tm.get("test", {})
    pos_rate = (test["num_positives"] / test["num_pairs"] * 100) if test.get("num_pairs") else None
    lat = s["adsb_latency_s"]
    lat_s = f"U({lat[0]}, {lat[1]}) s" if isinstance(lat, list) else f"{lat} s"
    rows = [
        ("Airspace / fleet", f"{d['grid_size_m']/1000:g}\\,km $\\times$ {d['grid_size_m']/1000:g}\\,km, "
                             f"$N={d['num_uavs']}$ UAVs, {d['num_sectors']} sectors, {d['num_restricted_zones']} restricted zones"),
        ("Scenarios (train/val/test)", f"{d['train_scenarios']}/{d['val_scenarios']}/{d['test_scenarios']} "
                                       f"$\\times$ {d['scenario_duration_s']:g}\\,s at {d['sim_freq_hz']:g}\\,Hz"),
        ("Surveillance model", f"ADS-B latency {lat_s}, loss {s['packet_loss']*100:g}\\,\\%, GPS CEP {s['gps_cep_m']}\\,m"),
        ("Conflict label", f"6-DoF look-ahead {d['lookahead_seconds']:g}\\,s, separation "
                           f"{d['conflict_h_sep_m']:g}\\,m H / {d['conflict_v_sep_m']:g}\\,m V"),
        ("Candidate pairs", f"proximity pre-filter (+{cfg['scoring']['proximity_margin_m']:g}\\,m margin); "
                            + (f"positive rate {pos_rate:.2f}\\,\\%" if pos_rate is not None else "")),
        ("Observation window", f"$K={d['observation_window']}$ snapshots; AoI $\\delta$ per edge"),
        ("TR-GAT", f"{m['num_layers']} layers, $d={m['embed_dim']}$, {m['num_heads']} heads, "
                   f"$\\phi(\\delta)\\in\\mathbb{{R}}^{{{m['temporal_dim']}}}$, GRU {m['recurrent_dim']}, dropout {m['dropout']}"),
        ("Training", f"AdamW lr {t['learning_rate']:g}, wd {t['weight_decay']:g}, warm-up {t['warmup_steps']} steps + cosine, "
                     f"focal loss ($\\gamma={t['focal_gamma']:g}$, $\\alpha={t['focal_alpha']:g}$), "
                     f"$\\le${t['epochs']} epochs, patience {t['early_stopping_patience']}"),
        ("Model selection", "F1-optimal threshold on validation, frozen for test" if t.get("threshold_mode", "val") == "val"
                            else f"fixed threshold {t['conflict_threshold']}"),
        ("Hardware", tex_escape(f"{env.get('gpu_model', '')}, {env.get('cpu_model', '')}, PyTorch {env.get('torch', '')}")),
    ]
    lines = ["\\begin{tabular}{@{}p{0.27\\columnwidth}p{0.68\\columnwidth}@{}}", "\\toprule",
             "Item & Setting \\\\", "\\midrule"]
    lines += [f"{k} & {v} \\\\" for k, v in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    _write(out, f"% auto-generated from {cfg_path} and {res.dir}; do not edit\n" + "\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out_dir", default="paper/tables")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--methods", nargs="+", default=MAIN_ORDER)
    args = ap.parse_args()
    res = Results(args.results_dir)
    O = Path(args.out_dir)
    tab_main(res, O / "tab_main.tex", args.methods)
    tab_ablation(res, O / "tab_ablation.tex")
    tab_events(res, O / "tab_events.tex", args.methods)
    tab_setup(res, Path(args.config), O / "tab_setup.tex")


if __name__ == "__main__":
    main()
