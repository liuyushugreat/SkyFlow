"""Shared helpers for S16 (tables + macros): load results/ artefacts, format numbers,
sanitise names into LaTeX macro identifiers.  Nothing here invents a number."""

import json
import math
import re
from pathlib import Path

import pandas as pd

# display names and order used in the paper tables
MAIN_ORDER = ["CPA-Rule", "Plan-CPA", "VO", "LSTM-P", "Tfm-P", "GAT-S", "STGCN", "TR-GAT-NT", "TR-GAT"]
ABL_ORDER = ["TR-GAT", "TR-GAT-NT", "abl_no_conf_gate", "abl_no_sync", "abl_no_gru", "abl_tbptt", "abl_no_gating",
             "abl_no_plan", "abl_bce", "abl_telemetry_only"]
ABL_LABEL = {
    "TR-GAT": "TR-GAT (full)", "TR-GAT-NT": "w/o temporal encoding $\\phi(\\delta)$",
    "abl_no_gating": "w/o relation gating", "abl_no_gru": "w/o GRU state",
    "abl_bce": "BCE instead of focal loss", "abl_telemetry_only": "telemetry-only input",
    "abl_no_plan": "w/o filed-plan context",
    "abl_no_conf_gate": "w/o intent-conformance gate", "abl_no_sync": "w/o AoI synchronisation",
    "abl_tbptt": "truncated BPTT (1 step)",
}
ABL_SHORT = {   # compact two-up ablation table (page budget)
    "TR-GAT": "TR-GAT (full)", "TR-GAT-NT": "w/o $\\phi(\\delta)$", "abl_no_gating": "w/o rel. gating",
    "abl_no_gru": "w/o GRU state", "abl_bce": "BCE loss", "abl_telemetry_only": "telemetry only",
    "abl_no_plan": "w/o plan context", "abl_no_conf_gate": "w/o conf. gate",
    "abl_no_sync": "w/o AoI sync.", "abl_tbptt": "TBPTT (1 step)",
}
_CAMEL = {"TR-GAT": "TrGat", "TR-GAT-NT": "TrGatNt", "GAT-S": "GatS", "STGCN": "Stgcn", "LSTM-P": "LstmP",
          "Tfm-P": "TfmP", "CPA-Rule": "CpaRule", "Plan-CPA": "PlanCpa", "VO": "Vo", "abl_no_gating": "AblNoGating",
          "abl_no_gru": "AblNoGru", "abl_bce": "AblBce", "abl_telemetry_only": "AblTelemetry",
          "abl_no_plan": "AblNoPlan", "abl_no_conf_gate": "AblNoConfGate", "abl_no_sync": "AblNoSync",
          "abl_tbptt": "AblTbptt"}


def macro_name(*parts):
    """LaTeX macro names may contain letters only."""
    out = []
    for p in parts:
        p = _CAMEL.get(str(p), str(p))
        p = re.sub(r"[^A-Za-z]", " ", p)
        out.append("".join(w[:1].upper() + w[1:] for w in p.split()))
    s = "".join(out)
    return s[:1].lower() + s[1:]


def fmt(x, nd=3, dash="--"):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return dash
    return f"{float(x):.{nd}f}"


def fmt_pm(mean, std, nd=3):
    if std is None or (isinstance(std, float) and not math.isfinite(std)) or std == 0:
        return fmt(mean, nd)
    return f"{fmt(mean, nd)}$\\pm${fmt(std, nd)}"


def fmt_p(p):
    if p is None or (isinstance(p, float) and not math.isfinite(p)):
        return "--"
    return "$<$0.001" if p < 1e-3 else f"{p:.3f}"


def tex_escape(s):
    return str(s).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


class Results:
    """All S15 artefacts of one results directory (missing files -> None)."""

    def __init__(self, results_dir):
        R = Path(results_dir)
        self.dir = R
        self.summary = self._csv(R / "main_summary.csv")
        self.per_regime = self._csv(R / "main_per_regime.csv")
        self.tests = self._json(R / "main_tests.json")
        self.ablation = self._csv(R / "ablation.csv")
        self.ablation_tests = self._json(R / "ablation_tests.json")
        self.rob_latency = self._csv(R / "robustness_latency.csv")
        self.rob_loss = self._csv(R / "robustness_loss.csv")
        self.scaling = self._csv(R / "scaling.csv")
        self.scaling_fit = self._json(R / "scaling_fit.json")
        self.attention = self._csv(R / "attention_vs_aoi.csv")
        self.gate = self._csv(R / "gate_by_cause.csv")              # S8e intent-conformance gate
        self.events = self._csv(R / "events_summary.csv")           # S8f event-level metrics (per method x persistence)
        self.events_cause = self._csv(R / "events_by_cause.csv")
        self.events_meta = self._json(R / "events.json")
        self.events_soc = self._csv(R / "events_soc.csv")           # SOC curves (val + test)
        self.events_budget = self._csv(R / "events_budget.csv")     # per method x seed x budget
        self.events_budget_summary = self._csv(R / "events_budget_summary.csv")
        self.nearmiss = self._csv(R / "nearmiss.csv")
        self.main_dir = R / "main" if (R / "main").is_dir() else R   # per-task metrics.json live here

    @staticmethod
    def _csv(p):
        return pd.read_csv(p) if p.exists() else None

    @staticmethod
    def _json(p):
        return json.load(open(p, encoding="utf-8")) if p.exists() else None

    # -- convenience -----------------------------------------------------
    def row(self, method):
        if self.summary is None:
            return None
        r = self.summary[self.summary["method"] == method]
        return None if r.empty else r.iloc[0].to_dict()

    def regime_cdr(self, method, regime):
        if self.per_regime is None:
            return None, None
        r = self.per_regime[(self.per_regime["method"] == method) & (self.per_regime["regime"] == regime)]
        return (None, None) if r.empty else (float(r.iloc[0]["cdr_mean"]), float(r.iloc[0]["cdr_std"]))

    def any_task_metrics(self, method=None):
        """First metrics.json found (for dataset / env / config facts)."""
        pats = [f"{method}/seed*/metrics.json"] if method else ["*/seed*/metrics.json"]
        for pat in pats:
            for p in sorted(self.main_dir.glob(pat)):
                return json.load(open(p, encoding="utf-8"))
        return None

    def p_value(self, metric, method):
        if not self.tests:
            return None
        return self.tests.get(f"paired_t_{metric}", {}).get(method, {}).get("p_bonferroni")

    def events_rows(self, persistence=1):
        """Event-level summary rows (dict per method) at one persistence level."""
        if self.events is None:
            return {}
        d = self.events[self.events["persistence"] == persistence]
        return {r["method"]: r.to_dict() for _, r in d.iterrows()}

    def events_p_value(self, metric, method, persistence=1):
        t = (self.events_meta or {}).get("tests", {}).get(f"persistence_{persistence}", {})
        return t.get(f"paired_t_{metric}", {}).get(method, {}).get("p_bonferroni")

    def budget_kinds(self):
        if self.events_budget_summary is None:
            return []
        kinds = [str(k) for k in dict.fromkeys(self.events_budget_summary["budget_kind"])]

        def _val(k):
            tail = k.split("_")[-1]
            return float(tail) if tail.replace(".", "", 1).isdigit() else 0.0
        # matched-rule budget first, then fixed budgets by increasing value
        return sorted(kinds, key=lambda k: (not k.startswith("match"), _val(k)))

    def budget_rows(self, kind):
        """Budget-matched summary rows (dict per method) for one budget kind."""
        if self.events_budget_summary is None:
            return {}
        d = self.events_budget_summary[self.events_budget_summary["budget_kind"] == kind]
        return {r["method"]: r.to_dict() for _, r in d.iterrows()}

    def budget_p_value(self, metric, method, kind):
        t = (self.events_meta or {}).get("budget_tests", {}).get(f"budget_kind_{kind}", {})
        return t.get(f"paired_t_{metric}", {}).get(method, {}).get("p_bonferroni")
