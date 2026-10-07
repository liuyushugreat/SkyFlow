"""S19 pre-submission check for the ISCAS PDF (no LaTeX knowledge needed to read the output).

    python scripts/check_paper.py [--paper_dir paper] [--pdf skyflow_iscas2027.pdf] [--max_pages 5] [--body_pages 4]

Checks (each prints PASS / FAIL, exit code 1 if any FAIL):
  fonts        every font embedded, no Type 3 (pdffonts)
  pages        at most --max_pages pages; the References heading starts no later than page --body_pages + 1
               and, if it is on that page, nothing but references is on it (pdftotext)
  macros       the last xelatex log has no "Undefined control sequence", no undefined references / citations,
               and no "??" marker from \\nm{} in the text
  numbers      numbers.tex was generated from the real results directory (numbersAreSmoke = 0)
  check-marks  no %%CHECK comment is left in the .tex
  overfull     overfull boxes wider than --overfull_pt (warning only)
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

TEX_BIN = Path(r"C:\texlive\2025\bin\windows")


def _run(exe, *args):
    p = Path(TEX_BIN / exe) if (TEX_BIN / exe).exists() else exe
    return subprocess.run([str(p), *map(str, args)], capture_output=True, text=True, encoding="utf-8", errors="replace")


def check_fonts(pdf):
    out = _run("pdffonts.exe", pdf).stdout.splitlines()
    rows = [l for l in out[2:] if l.strip()]
    bad = [l for l in rows if "Type 3" in l or re.search(r"\s(no)\s+(yes|no)\s+(yes|no)\s+\d", l)]
    return not bad and bool(rows), ("\n".join(bad) if bad else f"{len(rows)} fonts, all embedded, no Type 3")


def _page_text(pdf, page):
    return _run("pdftotext.exe", "-layout", "-f", page, "-l", page, pdf, "-").stdout


_REF_HEAD = re.compile(r"(?<![A-Za-z])R\s?E\s?F\s?E\s?R\s?E\s?N\s?C\s?E\s?S(?![A-Za-z])")


def check_pages(pdf, max_pages, body_pages):
    """Pages <= max_pages; the References heading no later than page body_pages+1; if it is on that page,
    the left column above it must be empty (pdftotext -layout keeps the two columns side by side)."""
    info = _run("pdfinfo.exe", pdf).stdout
    m = re.search(r"Pages:\s+(\d+)", info)
    n = int(m.group(1)) if m else -1
    msgs = [f"{n} pages"]
    ok = 0 < n <= max_pages
    ref_page, ref_line = None, None
    for p in range(1, n + 1):
        lines = _page_text(pdf, p).splitlines()
        for k, l in enumerate(lines):
            if _REF_HEAD.search(l):
                ref_page, ref_line, ref_lines = p, k, lines
                break
        if ref_page:
            break
    if ref_page is None:
        return False, "References heading not found"
    msgs.append(f"References start on page {ref_page}")
    if ref_page > body_pages + 1:
        ok = False
        msgs.append(f"FAIL: body runs past page {body_pages}")
    elif ref_page == body_pages + 1:
        col = _REF_HEAD.search(ref_lines[ref_line]).start()
        above = [l[:col + 10].strip() for l in ref_lines[:ref_line] if l[:col + 10].strip()]
        if above:
            ok = False
            msgs.append(f"FAIL: body text above the References heading on page {ref_page}: {above[0][:60]!r}")
    return ok, "; ".join(msgs)


def check_log(log_path, overfull_pt):
    if not log_path.exists():
        return False, f"{log_path} missing", []
    log = log_path.read_text(encoding="utf-8", errors="replace")
    problems = []
    for pat, label in ((r"Undefined control sequence", "undefined control sequence"),
                       (r"Reference `[^']+' on page \d+ undefined", "undefined reference"),
                       (r"Citation `[^']+' on page \d+ undefined", "undefined citation"),
                       (r"There were undefined references", "undefined references"),
                       (r"Label\(s\) may have changed", "labels changed (re-run)")):
        k = len(re.findall(pat, log))
        if k:
            problems.append(f"{k} x {label}")
    over = [float(x) for x in re.findall(r"Overfull \\hbox \((\d+\.\d+)pt too wide", log)]
    warn = [f"{x:.1f} pt" for x in over if x > overfull_pt]
    return not problems, ("; ".join(problems) if problems else "no undefined macros / references / citations"), warn


def check_text_markers(pdf):
    txt = "".join(_page_text(pdf, p) for p in range(1, 6))
    hits = re.findall(r"\?\?[A-Za-z]+", txt)
    return not hits, (f"missing-number markers: {sorted(set(hits))[:10]}" if hits else "no ?? markers in the text")


def check_numbers(paper_dir):
    f = paper_dir / "numbers.tex"
    if not f.exists():
        return False, "numbers.tex missing"
    t = f.read_text(encoding="utf-8")
    smoke = re.search(r"\\newcommand\{\\numbersAreSmoke\}\{(\d)\}", t)
    src = re.search(r"\\newcommand\{\\numbersSource\}\{([^}]*)\}", t)
    ok = smoke is not None and smoke.group(1) == "0"
    return ok, f"numbersAreSmoke={smoke.group(1) if smoke else '?'}, source={src.group(1) if src else '?'}"


def check_marks(tex):
    t = tex.read_text(encoding="utf-8")
    k = t.count("%%CHECK")
    return k == 0, (f"{k} %%CHECK comment(s) left" if k else "no %%CHECK comments")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper_dir", default="paper")
    ap.add_argument("--pdf", default="skyflow_iscas2027.pdf")
    ap.add_argument("--max_pages", type=int, default=5)
    ap.add_argument("--body_pages", type=int, default=4)
    ap.add_argument("--overfull_pt", type=float, default=5.0)
    args = ap.parse_args()
    d = Path(args.paper_dir)
    pdf = d / args.pdf
    tex = pdf.with_suffix(".tex")
    log = pdf.with_suffix(".log")
    if not pdf.exists():
        raise SystemExit(f"{pdf} not found")
    results = []
    results.append(("fonts", *check_fonts(pdf)))
    results.append(("pages", *check_pages(pdf, args.max_pages, args.body_pages)))
    ok, msg, warn = check_log(log, args.overfull_pt)
    results.append(("macros", ok, msg))
    results.append(("markers", *check_text_markers(pdf)))
    results.append(("numbers", *check_numbers(d)))
    results.append(("check-marks", *check_marks(tex)))
    failed = False
    for name, ok, msg in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name:12s} {msg}")
        failed |= not ok
    if warn:
        print(f"[WARN] overfull     {len(warn)} box(es) wider than {args.overfull_pt} pt: {', '.join(warn)}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
