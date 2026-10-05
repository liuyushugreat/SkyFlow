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
    args = ap.parse_args()
    res = Results(args.results_dir)
    M = Macros()
    ref = args.reference

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
        M.add("scalingMode", res.scaling_fit.get("mode", ""))
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

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = (f"% auto-generated by scripts/make_macros.py from {res.dir}; do not edit by hand\n"
              f"% {len(M.names)} macros\n")
    out.write_text(header + M.text(), encoding="utf-8")
    print(f"written: {out} ({len(M.names)} macros)")


if __name__ == "__main__":
    main()
