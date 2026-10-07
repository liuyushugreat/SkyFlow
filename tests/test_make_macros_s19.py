"""S19: data-derived wording macros of scripts/make_macros.py.

The Results text ranks methods ("the highest CDR", "inputs matter more than the
encoder", "lowest for non-cooperative aircraft").  These words must come from
results/ like every number, so make_macros derives them; here they are checked
on synthetic summary tables with a known ordering.
"""

import importlib.util
import pathlib

import pandas as pd
import pytest


def _load():
    spec = importlib.util.spec_from_file_location(
        "make_macros", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "make_macros.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


class _FakeRes:
    def __init__(self, p):
        self._p = p

    def p_value(self, metric, method):
        return self._p.get((metric, method))


def _macros(M):
    out = {}
    for line in M.lines:
        name = line.split("{\\", 1)[1].split("}", 1)[0]
        val = line.split("}{", 1)[1].rsplit("}", 1)[0]
        out[name] = val
    return out


def test_names_list():
    mm = _load()
    assert mm._names([]) == "none"
    assert mm._names(["GAT-S"]) == "GAT-S"
    assert mm._names(["GAT-S", "abl_bce"]) == "GAT-S and abl\\_bce"
    assert mm._names(["A", "B", "C"]) == "A, B and C"


def test_group_summaries_ranking_words():
    mm = _load()
    S = pd.DataFrame({
        "method": ["TR-GAT", "GAT-S", "STGCN", "CPA-Rule", "Plan-CPA"],
        "f1_mean": [0.42, 0.41, 0.43, 0.36, 0.38],
        "auprc_mean": [0.40, 0.39, 0.41, 0.30, 0.33],
        "cdr_mean": [0.47, 0.45, 0.50, 0.40, 0.42],
        "far_mean": [0.62, 0.59, 0.65, 0.55, 0.57],
    }).set_index("method")
    res = _FakeRes({("f1", "GAT-S"): 0.2, ("f1", "STGCN"): 0.01})
    M = mm.Macros()
    mm._group_summaries(M, S, res, "TR-GAT")
    m = _macros(M)
    assert m["nLearned"] == "3" and m["nRules"] == "2"
    assert m["learnedFOneBestName"] == "STGCN" and m["learnedFOneWorstName"] == "GAT-S"
    assert m["learnedFOneSpreadPts"] == "2.0"
    assert m["ruleFOneBestName"] == "Plan-CPA" and m["gainMinLearnedOverRulePts"] == "3.0"
    assert m["learnedCdrBestName"] == "STGCN" and m["learnedFarBestName"] == "GAT-S"
    assert m["refCdrRankWord"] == "the second-highest"
    assert m["mainNSignificantFOne"] == "1" and m["mainSignificantFOneNames"] == "STGCN"


def test_ablation_side_split(tmp_path):
    """Input-side vs encoder-side |dF1| means; the comparison word follows the data."""
    mm = _load()
    abl = pd.DataFrame({
        "method": ["TR-GAT", "abl_no_sync", "abl_no_plan", "TR-GAT-NT", "abl_no_gru"],
        "cdr_mean": [0.47] * 5, "far_mean": [0.62] * 5,
        "f1_mean": [0.42, 0.40, 0.41, 0.419, 0.421],
        "d_f1": [float("nan"), 0.02, 0.01, 0.001, -0.001],
        "p_f1_bonf": [float("nan"), 0.001, 0.03, 0.9, 0.9],
    })
    abl.to_csv(tmp_path / "ablation.csv", index=False)
    import subprocess, sys
    out = tmp_path / "numbers.tex"
    script = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "make_macros.py"
    subprocess.run([sys.executable, str(script), "--results_dir", str(tmp_path), "--out", str(out)], check=True)
    txt = out.read_text(encoding="utf-8")
    assert "\\newcommand{\\ablNSignificant}{2}" in txt
    assert "\\newcommand{\\ablSignificantNames}{abl\\_no\\_sync and abl\\_no\\_plan}" in txt
    assert "\\newcommand{\\ablInputsMeanAbsDPts}{1.5}" in txt
    assert "\\newcommand{\\ablEncoderMeanAbsDPts}{0.1}" in txt
    assert "\\newcommand{\\ablInputsVsEncoderWord}{more}" in txt
    assert "\\newcommand{\\ablMaxAbsDFOneName}{abl\\_no\\_sync}" in txt


def test_cause_labels_cover_simulator_causes():
    mm = _load()
    from skyflow.config import SkyFlowConfig
    mix = SkyFlowConfig().sim.cause_mix or {}
    for cause in mix:
        assert cause in mm.CAUSE_LABEL, cause
