"""S16: LaTeX number macros, generated only from results/ artefacts.

    python scripts/make_macros.py --results_dir results --out paper/numbers.tex

Every number quoted in the paper body must come through one of these macros
(or an \\input table), never typed by hand.  Naming: \\<metric><Method>, e.g.
\\cdrTrGat, \\fOneCpaRule, \\latTrGat, \\hardCdrTrGat, \\dFOneTrGatNt (TR-GAT minus variant),
\\pFOneGatS (Bonferroni p), \\alphaTotal, \\cdrLatTrGatMax (CDR at the largest latency), ...
Metrics 3 decimals, latencies 1 decimal, percentages 1-2 decimals.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_common import Results, fmt, fmt_p, macro_name  # noqa: E402

_DIGITS = "Zero One Two Three Four Five Six Seven Eight Nine".split()


def num_word(n):
    """500 -> 'FiveZeroZero' (LaTeX macro names cannot contain digits)."""
    return "".join(_DIGITS[int(c)] for c in str(int(n)))


def _num_ok(x):
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _sci(x):
    """1e-05 -> '$10^{-5}$', 0.0003 -> '0.0003', 3e-4 stays decimal (only pure powers of ten become exponents)."""
    x = float(x)
    if x > 0:
        e = math.log10(x)
        if abs(e - round(e)) < 1e-9 and round(e) <= -3:
            return f"$10^{{{int(round(e))}}}$"
    return f"{x:g}"


def _kind(method):
    try:
        from skyflow.experiments.methods import METHODS
        return METHODS[method].kind if method in METHODS else None
    except Exception:          # noqa: BLE001
        return None


def _group_summaries(M, S):
    """\\learnedFOneMin/Max/SpreadPts, \\learnedFOneBestName, \\learnedAuprcMin/Max, \\ruleFOneBest/BestName,
    \\gainMinLearnedOverRulePts (weakest learned detector minus best rule, in percentage points),
    \\gainBestLearnedOverRulePts, \\nLearned, \\nRules - from the main summary table (methods of the main group)."""
    rows = {m: r for m, r in S.iterrows() if _num_ok(r.get("f1_mean"))}
    learned = {m: r for m, r in rows.items() if _kind(m) in ("trgat", "learned")}
    rules = {m: r for m, r in rows.items() if _kind(m) == "rule"}
    M.add("nLearned", len(learned))
    M.add("nRules", len(rules))
    if learned:
        f = {m: float(r["f1_mean"]) for m, r in learned.items()}
        lo, hi = min(f, key=f.get), max(f, key=f.get)
        M.num("learnedFOneMin", f[lo]); M.num("learnedFOneMax", f[hi])
        M.add("learnedFOneSpreadPts", f"{(f[hi] - f[lo]) * 100:.1f}")
        M.add("learnedFOneBestName", hi.replace("_", "\\_")); M.add("learnedFOneWorstName", lo.replace("_", "\\_"))
        if all("auprc_mean" in r and _num_ok(r["auprc_mean"]) for r in learned.values()):
            a = {m: float(r["auprc_mean"]) for m, r in learned.items()}
            M.num("learnedAuprcMin", min(a.values())); M.num("learnedAuprcMax", max(a.values()))
            M.add("learnedAuprcBestName", max(a, key=a.get).replace("_", "\\_"))
    if rules:
        f = {m: float(r["f1_mean"]) for m, r in rules.items()}
        best = max(f, key=f.get)
        M.num("ruleFOneBest", f[best]); M.add("ruleFOneBestName", best.replace("_", "\\_"))
        if learned:
            fl = [float(r["f1_mean"]) for r in learned.values()]
            M.add("gainMinLearnedOverRulePts", f"{(min(fl) - f[best]) * 100:.1f}")
            M.add("gainBestLearnedOverRulePts", f"{(max(fl) - f[best]) * 100:.1f}")


class Macros:
    def __init__(self):
        self.lines, self.names = [], set()

    def add(self, name, value, comment=None):
        if name in self.names:
            raise ValueError(f"duplicate macro {name}")
        self.names.add(name)
        c = f"  % {comment}" if comment else ""
        self.lines.append(f"\\newcommand{{\\{name}}}{{{value}}}{c}")

    def num(self, name, x, nd=3, comment=None):
        if _num_ok(x):
            self.add(name, fmt(float(x), nd), comment)

    def text(self):
        return "\n".join(self.lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out", default="paper/numbers.tex")
    ap.add_argument("--reference", default="TR-GAT")
    ap.add_argument("--config", default="configs/default.yaml", help="fallback if no task config.yaml is found")
    args = ap.parse_args()
    res = Results(args.results_dir)
    M = Macros()
    ref = args.reference
    # 1 while the numbers come from a smoke / dry-run directory -> the paper prints a watermark
    M.add("numbersAreSmoke", 1 if any(tag in str(res.dir) for tag in ("_smoke", "_dryrun", "_sanity")) else 0)
    M.add("numbersSource", str(res.dir).replace("\\", "/").replace("_", "\\_"))

    # ---- main table -----------------------------------------------------
    if res.summary is not None:
        S = res.summary.set_index("method")
        M.add("nSeeds", int(S["n_seeds"].max()))
        M.add("gitCommits", str(S.loc[ref, "git_commits"]).replace("_", "\\_") if ref in S.index else "")
        for m, r in S.iterrows():
            for key, col, nd in (("cdr", "cdr_mean", 3), ("far", "far_mean", 3), ("fOne", "f1_mean", 3),
                                 ("prec", "precision_mean", 3), ("auprc", "auprc_mean", 3),
                                 ("cdrStd", "cdr_std", 3), ("farStd", "far_std", 3), ("fOneStd", "f1_std", 3),
                                 ("thr", "threshold_mean", 3), ("lat", "latency_p95_ms", 1),
                                 ("latBuild", "graph_build_p95_ms", 1), ("latFwd", "gnn_forward_p95_ms", 1),
                                 ("latScore", "pair_scoring_p95_ms", 1), ("params", "params", 0),
                                 ("epochs", "epochs_run_mean", 0), ("cdrCiLo", "cdr_ci_lo", 3),
                                 ("cdrCiHi", "cdr_ci_hi", 3), ("fOneCiLo", "f1_ci_lo", 3), ("fOneCiHi", "f1_ci_hi", 3)):
                if col in r:
                    M.num(macro_name(key, m), r[col], nd)
            if "params" in r and _num_ok(r["params"]):
                M.add(macro_name("paramsM", m), f"{float(r['params'])/1e6:.2f}")
            hc, _ = res.regime_cdr(m, "hard")
            ec, _ = res.regime_cdr(m, "easy")
            M.num(macro_name("hardCdr", m), hc)
            M.num(macro_name("easyCdr", m), ec)
            if res.per_regime is not None:
                for reg in res.per_regime[res.per_regime["method"] == m]["regime"]:
                    if reg.startswith("cause:"):
                        M.num(macro_name("cdrCause", reg.split(":", 1)[1], m), res.regime_cdr(m, reg)[0])
            if m != ref and ref in S.index:
                for key, col in (("dCdr", "cdr_mean"), ("dFar", "far_mean"), ("dFOne", "f1_mean")):
                    M.add(macro_name(key, m), f"{float(S.loc[ref, col]) - float(r[col]):+.3f}")
                if _num_ok(r["f1_mean"]) and float(r["f1_mean"]) > 0:
                    M.add(macro_name("relFOne", m), f"{(float(S.loc[ref, 'f1_mean']) / float(r['f1_mean']) - 1) * 100:.1f}")
                for met in ("f1", "cdr", "far"):
                    p = res.p_value(met, m)
                    if _num_ok(p):
                        M.add(macro_name("p" + {"f1": "FOne", "cdr": "Cdr", "far": "Far"}[met], m), fmt_p(float(p)))
        # S8g cross-method summaries for the protocol-centred Results text: spread of the learned detectors,
        # best rule, gap of the weakest / best learned detector over the best rule (percentage points).
        _group_summaries(M, S)
        # dataset facts from one metrics.json
        tm = res.any_task_metrics(ref) or res.any_task_metrics() or {}
        ds, test, env = tm.get("dataset", {}), tm.get("test", {}), tm.get("env", {})
        for k, v in ds.items():
            M.add(macro_name("ds", k), v if not isinstance(v, float) else f"{v:g}")
        if test.get("num_pairs"):
            M.add("testPairs", f"{int(test['num_pairs']):,}".replace(",", "\\,"))
            M.add("testPositives", f"{int(test['num_positives']):,}".replace(",", "\\,"))
            M.add("posRatePct", f"{test['num_positives'] / test['num_pairs'] * 100:.2f}")
        for k in ("gpu_model", "cpu_model", "torch", "cuda", "driver"):
            if env.get(k):
                M.add(macro_name("env", k), str(env[k]).replace("_", "\\_"))

    # ---- CPU-only inference latency (S8f; results/eval_cpu/<method>/seed*.json) ----
    for p in sorted((res.dir / "eval_cpu").glob("*/seed*.json")) if (res.dir / "eval_cpu").is_dir() else []:
        j = json.load(open(p, encoding="utf-8"))
        m, lat = j["method"], j.get("latency", {})
        inf, gb = lat.get("inference", {}), lat.get("graph_build", {})
        key = macro_name("latCpu", m)
        if key + "Fwd" in M.names:          # one seed per method is enough
            continue
        M.num(key + "Fwd", inf.get("gnn_forward_p95_ms"), 1)
        M.num(key + "Score", inf.get("pair_scoring_p95_ms"), 1)
        M.num(key + "Build", gb.get("graph_build_p95_ms"), 1)
        M.num(key + "Sum", lat.get("p95_sum_ms"), 1)
        if j.get("env", {}).get("cpu_model"):
            M.add(key + "Host", str(j["env"]["cpu_model"]).replace("_", "\\_"))

    # ---- configuration facts (from the run's own config.yaml, merged with defaults) ----
    cfg_path = next(iter(sorted(res.main_dir.glob(f"{ref}/seed*/config.yaml"))), None) or Path(args.config)
    try:
        from dataclasses import asdict
        from skyflow.config import SkyFlowConfig
        cfg = asdict(SkyFlowConfig.from_yaml(str(cfg_path)))
        d, s, t, m, sc = cfg["data"], cfg["sim"], cfg["training"], cfg["model"], cfg["scoring"]
        lat = s["adsb_latency_s"]
        lat = lat if isinstance(lat, list) else [lat, lat]
        for name, val in (("cfgLookaheadS", f"{d['lookahead_seconds']:g}"), ("cfgSepHM", f"{d['conflict_h_sep_m']:g}"),
                          ("cfgSepVM", f"{d['conflict_v_sep_m']:g}"), ("cfgWindowK", d["observation_window"]),
                          ("cfgSimHz", f"{d['sim_freq_hz']:g}"), ("cfgAdsbLatLo", f"{lat[0]:g}"), ("cfgAdsbLatHi", f"{lat[1]:g}"),
                          ("cfgGpsCepM", f"{s['gps_cep_m']:g}"), ("cfgProximityMarginM", f"{sc['proximity_margin_m']:g}"),
                          ("cfgNumLayers", m["num_layers"]), ("cfgEmbedDim", m["embed_dim"]), ("cfgNumHeads", m["num_heads"]),
                          ("cfgTemporalDim", m["temporal_dim"]), ("cfgRecurrentDim", m["recurrent_dim"]),
                          ("cfgLr", f"{t['learning_rate']:g}"), ("cfgEpochsMax", t["epochs"]), ("cfgPatience", t["early_stopping_patience"]),
                          ("cfgFocalGamma", f"{t['focal_gamma']:g}"), ("cfgFocalAlpha", f"{t['focal_alpha']:g}"),
                          ("cfgRegimeTtcS", f"{t['regime_ttc_boundary_s']:g}"), ("cfgNumSectors", d["num_sectors"]),
                          ("cfgNumZones", d["num_restricted_zones"]),
                          ("cfgWeightDecay", _sci(t["weight_decay"])), ("cfgWarmupSteps", t["warmup_steps"]),
                          ("cfgDropout", f"{m['dropout']:g}")):
            M.add(name, val)
        mix = s.get("cause_mix") or {}
        for k, v in mix.items():
            M.add(macro_name("cfgCausePct", k), f"{float(v) * 100:g}")
    except Exception as e:  # config facts are optional
        print("[warn] config macros skipped:", e)

    # ---- ablation -------------------------------------------------------
    if res.ablation is not None:
        for _, r in res.ablation.iterrows():
            m = r["method"]
            if m == ref:
                continue
            M.num(macro_name("ablCdr", m), r["cdr_mean"])
            M.num(macro_name("ablFar", m), r["far_mean"])
            M.num(macro_name("ablFOne", m), r["f1_mean"])
            if _num_ok(r["d_f1"]):
                M.add(macro_name("ablDFOne", m), f"{float(r['d_f1']):+.3f}")
            if _num_ok(r.get("p_f1_bonf")):
                M.add(macro_name("ablPFOne", m), fmt_p(float(r["p_f1_bonf"])))
        # S8g: largest |dF1| over the ablations (percentage points), its variant, and how many are significant
        ab = res.ablation[(res.ablation["method"] != ref) & res.ablation["d_f1"].apply(_num_ok)]
        if not ab.empty:
            i = ab["d_f1"].abs().idxmax()
            M.add("ablMaxAbsDFOnePts", f"{abs(float(ab.loc[i, 'd_f1'])) * 100:.1f}")
            M.add("ablMaxAbsDFOneName", str(ab.loc[i, "method"]).replace("_", "\\_"))
            M.add("ablMaxAbsDFOneSigned", f"{float(ab.loc[i, 'd_f1']):+.3f}")
            if "p_f1_bonf" in ab.columns:
                sig = ab[ab["p_f1_bonf"].apply(_num_ok) & (ab["p_f1_bonf"].astype(float) < 0.05)]
                M.add("ablNSignificant", len(sig))
                M.add("ablNVariants", len(ab))

    # ---- robustness -----------------------------------------------------
    for tag, df, scale in (("Lat", res.rob_latency, 1.0), ("Loss", res.rob_loss, 100.0)):
        if df is None:
            continue
        xs = sorted(df["value"].unique())
        M.add(f"rob{tag}Levels", ", ".join(f"{x * scale:g}" for x in xs))
        M.add(f"rob{tag}Max", f"{xs[-1] * scale:g}")
        for m in df["method"].unique():
            dm = df[df["method"] == m]
            g = dm.groupby("value")["cdr"].mean()
            lo, hi = float(g.loc[xs[0]]), float(g.loc[xs[-1]])
            M.num(macro_name(f"cdr{tag}", m, "Min"), lo)
            M.num(macro_name(f"cdr{tag}", m, "Max"), hi)
            M.add(macro_name(f"cdr{tag}Drop", m), f"{lo - hi:+.3f}")
            if lo > 0:
                M.add(macro_name(f"cdr{tag}DropPct", m), f"{(lo - hi) / lo * 100:.1f}")
            gf = dm.groupby("value")["f1"].mean()
            M.num(macro_name(f"fOne{tag}", m, "Min"), float(gf.loc[xs[0]]))
            M.num(macro_name(f"fOne{tag}", m, "Max"), float(gf.loc[xs[-1]]))
            if "auprc" in dm.columns:
                ga = dm.groupby("value")["auprc"].mean()
                M.num(macro_name(f"auprc{tag}", m, "Min"), float(ga.loc[xs[0]]))
                M.num(macro_name(f"auprc{tag}", m, "Max"), float(ga.loc[xs[-1]]))
            # S8f: largest level inside the training link mix vs the extrapolation levels
            if "in_train_range" in dm.columns and (dm["in_train_range"] == 1).any() and (dm["in_train_range"] == 0).any():
                x_in = max(dm[dm["in_train_range"] == 1]["value"])
                v_in = float(g.loc[x_in])
                M.num(macro_name(f"cdr{tag}", m, "InMax"), v_in)
                M.add(macro_name(f"cdr{tag}OodDrop", m), f"{v_in - hi:+.3f}")
                if v_in > 0:
                    M.add(macro_name(f"cdr{tag}OodDropPct", m), f"{(v_in - hi) / v_in * 100:.1f}")
                if "auprc" in dm.columns:
                    M.num(macro_name(f"auprc{tag}", m, "InMax"), float(ga.loc[x_in]))
        if "in_train_range" in df.columns and (df["in_train_range"] == 0).any():
            M.add(f"rob{tag}TrainMax", f"{max(df[df['in_train_range'] == 1]['value']) * scale:g}")
            M.add(f"rob{tag}OodLevels", ", ".join(f"{x * scale:g}" for x in sorted(df[df["in_train_range"] == 0]["value"].unique())))
            # S8g: most / least graceful *learned* detector from the largest in-range level to the extreme one
            x_in = max(df[df["in_train_range"] == 1]["value"])
            drops = {}
            for m in df["method"].unique():
                if _kind(m) not in ("trgat", "learned"):
                    continue
                g = df[df["method"] == m].groupby("value")["cdr"].mean()
                if x_in in g.index and xs[-1] in g.index and float(g.loc[x_in]) > 0:
                    drops[m] = (float(g.loc[x_in]) - float(g.loc[xs[-1]])) / float(g.loc[x_in]) * 100
            if len(drops) >= 2:
                worst, best = max(drops, key=drops.get), min(drops, key=drops.get)
                M.add(f"rob{tag}OodDropPctWorstName", worst.replace("_", "\\_")); M.add(f"rob{tag}OodDropPctWorst", f"{drops[worst]:.1f}")
                M.add(f"rob{tag}OodDropPctBestName", best.replace("_", "\\_")); M.add(f"rob{tag}OodDropPctBest", f"{drops[best]:.1f}")

    # ---- scaling --------------------------------------------------------
    if res.scaling_fit is not None:
        fits = res.scaling_fit.get("fits", {})
        for col, key in (("total_p95_ms", "Total"), ("graph_build_p95_ms", "Build"),
                         ("gnn_forward_p95_ms", "Fwd"), ("pair_scoring_p95_ms", "Score"),
                         ("mean_edges", "Edges"), ("mean_candidate_pairs", "Cands")):
            f = fits.get(col, {})
            M.num(f"alpha{key}", f.get("alpha"), 2)
            M.num(f"alpha{key}Se", f.get("alpha_se"), 2)
            M.num(f"alpha{key}Rsq", f.get("r2"), 3)
        M.add("scalingMode", str(res.scaling_fit.get("mode", "")).replace("_", " "))
        ok = res.scaling_fit.get("sizes_ok", [])
        if ok:
            M.add("scalingSizes", ", ".join(str(int(n)) for n in ok))
            M.add("scalingNmax", int(max(ok)))
    if res.scaling is not None:
        d = res.scaling[res.scaling["status"] == "ok"].sort_values("num_uavs")
        for _, r in d.iterrows():
            n = int(r["num_uavs"])
            M.num(f"latTotalN{num_word(n)}", r["total_p95_ms"], 1)
            M.num(f"latBuildN{num_word(n)}", r["graph_build_p95_ms"], 1)
            M.num(f"latFwdN{num_word(n)}", r["gnn_forward_p95_ms"], 1)
        if not d.empty:
            under = d[d["total_p95_ms"] <= 200.0]
            M.add("nMaxUnderBudget", int(under["num_uavs"].max()) if not under.empty else 0)
            if "peak_gpu_mem_gb" in d.columns:
                M.add("peakGpuMemGb", f"{float(d['peak_gpu_mem_gb'].max()):.1f}")

    # ---- attention vs AoI ----------------------------------------------
    if res.attention is not None:
        a = res.attention
        layer = int(a["layer"].max())
        a = a[(a["layer"] == layer) & (a["n_edges"] >= 20)]
        M.add("attnLayer", layer)
        for rel, g in a.groupby("relation_name"):
            g = g.sort_values("delta_bin_lo_s")
            if len(g) >= 2:
                M.num(macro_name("attnFresh", rel), g.iloc[0]["attn_norm_mean"])
                M.num(macro_name("attnStale", rel), g.iloc[-1]["attn_norm_mean"])
                M.add(macro_name("attnRatio", rel), f"{g.iloc[0]['attn_norm_mean'] / max(g.iloc[-1]['attn_norm_mean'], 1e-9):.2f}")
                M.add(macro_name("attnStaleDelta", rel), f"{g.iloc[-1]['delta_bin_hi_s']:g}")

    # ---- event-level metrics (S8f) --------------------------------------
    # \evCdr<M>, \evTimely<M>, \evLeadMed<M>, \evLeadMean<M>, \evPrec<M>, \evFaH<M> (false episodes per UAV-hour),
    # \evPCdr<M> (Bonferroni p vs reference); persistence M>1 adds the suffix P<M>, e.g. \evCdrPThreeTrGat.
    if res.events is not None:
        meta = res.events_meta or {}
        M.add("evLeadS", f"{float(meta.get('lead_s', 10)):g}")
        for P in sorted(res.events["persistence"].unique()):
            P = int(P)
            sfx = "" if P == 1 else "P" + num_word(P)
            rows = res.events_rows(P)
            for m, r in rows.items():
                for key, col, nd in (("evCdr", "event_cdr_mean", 3), ("evCdrStd", "event_cdr_std", 3),
                                     ("evTimely", "timely_cdr_mean", 3), ("evLeadMed", "lead_median_s_mean", 1),
                                     ("evLeadMean", "lead_mean_s_mean", 1), ("evLeadFrac", "lead_frac_mean", 2),
                                     ("evPrec", "episode_precision_mean", 3),
                                     ("evFaH", "false_episodes_per_uav_hour_mean", 1),
                                     ("evFaDur", "false_episode_dur_mean_s_mean", 1)):
                    M.num(macro_name(key + sfx, m), r.get(col), nd)
                if m != ref:
                    p = res.events_p_value("event_cdr", m, P)
                    if _num_ok(p):
                        M.add(macro_name("evPCdr" + sfx, m), fmt_p(float(p)))
                    p = res.events_p_value("false_episodes_per_uav_hour", m, P)
                    if _num_ok(p):
                        M.add(macro_name("evPFaH" + sfx, m), fmt_p(float(p)))
            if P == 1 and rows:
                any_row = next(iter(rows.values()))
                M.add("evNEvents", f"{int(any_row['n_events']):,}".replace(",", "\\,"))
                M.add("evUavHours", f"{float(any_row['uav_hours']):.1f}")
                M.add("evPerUavHour", f"{float(any_row['n_events']) / float(any_row['uav_hours']):.1f}")
        # budget-matched operating points (selected on val): \evB<metric><Method> for the first (matched) budget,
        # \evB<Kind><metric><Method> for every budget kind (Kind = MatchCpaRule / FixedTwoZero ...)
        for k_i, kind in enumerate(res.budget_kinds()):
            ktag = macro_name("".join(num_word(c) if c.isdigit() else c for c in str(kind)).replace("_", " ").replace(".", " "))
            ktag = ktag[:1].upper() + ktag[1:]
            rows_b = res.budget_rows(kind)
            for m, r in rows_b.items():
                for key, col, nd in (("Cdr", "event_cdr_mean", 3), ("CdrStd", "event_cdr_std", 3),
                                     ("Timely", "timely_cdr_mean", 3), ("LeadMed", "lead_median_s_mean", 1),
                                     ("Prec", "episode_precision_mean", 3), ("FaH", "false_episodes_per_uav_hour_mean", 1),
                                     ("Thr", "threshold_mean", 3), ("Alpha", "alpha_mean", 2), ("Hyst", "hysteresis_mean", 2),
                                     ("Budget", "budget_mean", 1)):
                    M.num(macro_name(f"evB{ktag}{key}", m), r.get(col), nd)
                    if k_i == 0:
                        M.num(macro_name(f"evB{key}", m), r.get(col), nd)
                if m != ref:
                    p = res.budget_p_value("event_cdr", m, kind)
                    if _num_ok(p):
                        M.add(macro_name(f"evB{ktag}PCdr", m), fmt_p(float(p)))
                        if k_i == 0:
                            M.add(macro_name("evBPCdr", m), fmt_p(float(p)))
            if k_i == 0:
                M.add("evBudgetKind", str(kind).replace("_", "\\_"))
                # S8g: spread of the learned detectors under the common budget
                for key, col in (("Cdr", "event_cdr_mean"), ("Timely", "timely_cdr_mean"), ("LeadMed", "lead_median_s_mean")):
                    v = {m: float(r[col]) for m, r in rows_b.items()
                         if _kind(m) in ("trgat", "learned") and _num_ok(r.get(col))}
                    if len(v) >= 2:
                        lo, hi = min(v, key=v.get), max(v, key=v.get)
                        nd = 1 if key == "LeadMed" else 3
                        M.num(f"evBLearned{key}Min", v[lo], nd); M.num(f"evBLearned{key}Max", v[hi], nd)
                        M.add(f"evBLearned{key}BestName", hi.replace("_", "\\_"))
                        if key != "LeadMed":
                            M.add(f"evBLearned{key}SpreadPts", f"{(v[hi] - v[lo]) * 100:.1f}")
        # SOC facts: for the reference, false-episode rate of the operational layer at (about) the raw event CDR
        if res.events_soc is not None and ref in set(res.events_soc["method"]):
            soc = res.events_soc[(res.events_soc["method"] == ref) & (res.events_soc["split"] == "test")]
            raw = res.events_rows(1).get(ref)
            if raw is not None and not soc.empty:
                target = float(raw["event_cdr_mean"])
                base_fa = float(raw["false_episodes_per_uav_hour_mean"])
                best = None
                for (a, h), g in soc.groupby(["alpha", "hysteresis"]):
                    if a == 1.0 and h == 0.0:
                        continue
                    g = g.groupby("threshold")[["event_cdr", "false_episodes_per_uav_hour"]].mean().reset_index()
                    ok = g[g["event_cdr"] >= target]
                    if ok.empty:
                        continue
                    fa = float(ok["false_episodes_per_uav_hour"].min())
                    if best is None or fa < best[0]:
                        best = (fa, a, h)
                if best is not None:
                    M.num("socLayerFaHAtRawCdr", best[0], 1)
                    M.add("socLayerFaHReductionPct", f"{(base_fa - best[0]) / base_fa * 100:.0f}" if base_fa > 0 else "")
                    M.add("socLayerAlpha", f"{best[1]:g}")
                    M.add("socLayerHyst", f"{best[2]:g}")
        if res.events_cause is not None:
            c = res.events_cause[res.events_cause["persistence"] == 1] if "persistence" in res.events_cause else res.events_cause
            g = c.groupby(["method", "cause"])[["event_cdr", "timely_cdr", "lead_median_s"]].mean().reset_index()
            for _, r in g.iterrows():
                M.num(macro_name("evCdrCause", r["cause"], r["method"]), r["event_cdr"], 3)
                M.num(macro_name("evLeadCause", r["cause"], r["method"]), r["lead_median_s"], 1)

    # ---- near-miss analysis of false alerts (S8f) -------------------------
    # \nm<Group><Stat><Method> at the validation-F1 operating point, e.g. \nmFalseRowsLtTwoTrGat (fraction of false
    # alert rows with normalised separation < 2), \nmFalseEpisodesMedRhoTrGat, \nmNegativeSampleLtTwoTrGat
    if res.nearmiss is not None:
        nm = res.nearmiss[res.nearmiss["operating_point"] == "val_f1"] if "operating_point" in res.nearmiss else res.nearmiss
        lv_name = {"1": "One", "1.5": "OneHalf", "2": "Two", "3": "Three"}
        for (m, grp), g in nm.groupby(["method", "group"]):
            r = g.iloc[0]
            M.num(macro_name("nm", grp, "MedRho", m), r["rho_median"], 2)
            M.add(macro_name("nm", grp, "N", m), f"{int(r['n']):,}".replace(",", "\\,"))
            for col in g.columns:
                if col.startswith("frac_lt_"):
                    lv = col[len("frac_lt_"):]
                    if _num_ok(r[col]):
                        pct = float(r[col]) * 100
                        # one decimal below 10 % so that rare near-misses among random negatives do not print as 0
                        M.add(macro_name("nm", grp, "Lt" + lv_name.get(lv, lv), m), f"{pct:.1f}" if pct < 10 else f"{pct:.0f}")
            if (_num_ok(r.get("positive_mismatch")) and _num_ok(r.get("negative_mismatch"))
                    and macro_name("nmMismatch", m) not in M.names):
                M.add(macro_name("nmMismatch", m), f"{int(r['positive_mismatch']) + int(r['negative_mismatch'])}")

    # ---- intent-conformance gate (S8e) ----------------------------------
    if res.gate is not None:
        for _, r in res.gate.iterrows():
            g = str(r["group"])
            key = g.replace("pair:cause:", "").replace("pair:", "").replace("uav:", "uav_")
            M.num(macro_name("gateW", key), r["w_mean"], 2)
            M.num(macro_name("gateWMed", key), r["w_median"], 2)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = (f"% auto-generated by scripts/make_macros.py from {res.dir}; do not edit by hand\n"
              f"% {len(M.names)} macros\n")
    out.write_text(header + M.text(), encoding="utf-8")
    print(f"written: {out} ({len(M.names)} macros)")


if __name__ == "__main__":
    main()
