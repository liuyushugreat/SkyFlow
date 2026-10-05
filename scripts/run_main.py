#!/usr/bin/env python3
"""Orchestrate the main experiment / ablations as independent subprocesses (S7a).

Each (method, seed) runs ``scripts/run_task.py`` in its own process so that
``--max_concurrent N`` can share one GPU between N tasks.  Tasks that already
have a DONE marker are skipped with ``--resume``.

Examples:
    # smoke: every method, one seed, 2 epochs, tiny scene
    python scripts/run_main.py --config configs/smoke.yaml --results_dir results/_smoke --seeds 42 --epochs 2

    # main experiment, 3 seeds, 2 concurrent tasks
    python scripts/run_main.py --methods TR-GAT TR-GAT-NT GAT-S STGCN LSTM-P Tfm-P CPA-Rule VO \
        --seeds 42 123 456 --max_concurrent 2 --resume

    # ablations
    python scripts/run_main.py --methods abl_no_gating abl_no_gru abl_bce abl_telemetry_only --seeds 42 --resume
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skyflow.experiments.methods import ABLATION_METHODS, MAIN_METHODS, METHODS, is_trained

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--results_dir", default="results/main")
    ap.add_argument("--methods", nargs="+", default=None,
                    help="subset of methods (default: all main methods); 'ablations' expands to the ablation set")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    ap.add_argument("--max_concurrent", type=int, default=1)
    ap.add_argument("--resume", action="store_true", help="skip tasks with a DONE marker")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--dry_run", action="store_true", help="print the task list and exit")
    args = ap.parse_args()

    methods = []
    for m in (args.methods or MAIN_METHODS):
        if m == "ablations":
            methods.extend(ABLATION_METHODS)
        elif m == "main":
            methods.extend(MAIN_METHODS)
        else:
            if m not in METHODS:
                raise SystemExit(f"unknown method {m!r}; available: {sorted(METHODS)}")
            methods.append(m)

    tasks = []
    for m in methods:
        seeds = args.seeds if is_trained(m) else args.seeds[:1]   # deterministic rules: one run
        for s in seeds:
            out = Path(args.results_dir) / m / f"seed{s}"
            if args.resume and (out / "DONE").exists():
                print(f"[skip] {m} seed={s} (DONE)")
                continue
            tasks.append((m, s, out))

    print(f"{len(tasks)} task(s), max_concurrent={args.max_concurrent}, results -> {args.results_dir}")
    for m, s, _ in tasks:
        print(f"  - {m} seed={s}")
    if args.dry_run or not tasks:
        return

    running = []
    failed = []
    t0 = time.perf_counter()
    queue = list(tasks)
    while queue or running:
        while queue and len(running) < args.max_concurrent:
            m, s, out = queue.pop(0)
            out.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, str(ROOT / "scripts" / "run_task.py"), "--method", m, "--seed", str(s),
                   "--config", args.config, "--results_dir", args.results_dir, "--device", args.device]
            if args.epochs is not None:
                cmd += ["--epochs", str(args.epochs)]
            if args.cache_dir:
                cmd += ["--cache_dir", args.cache_dir]
            logf = open(out / "stdout.log", "w", encoding="utf-8")
            p = subprocess.Popen(cmd, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT)
            running.append((p, m, s, out, logf, time.perf_counter()))
            print(f"[start] {m} seed={s} (pid {p.pid})")
        time.sleep(2.0)
        still = []
        for p, m, s, out, logf, ts in running:
            rc = p.poll()
            if rc is None:
                still.append((p, m, s, out, logf, ts))
                continue
            logf.close()
            dt = time.perf_counter() - ts
            if (out / "DONE").exists():
                note = "" if rc == 0 else f" (rc={rc} at teardown, artifacts complete)"
                print(f"[done ] {m} seed={s} in {dt / 60:.1f} min{note}")
            else:
                failed.append((m, s, rc))
                print(f"[FAIL ] {m} seed={s} rc={rc} after {dt / 60:.1f} min -> see {out / 'stdout.log'}")
        running = still

    print(f"finished in {(time.perf_counter() - t0) / 60:.1f} min; {len(tasks) - len(failed)} ok, {len(failed)} failed")
    if failed:
        for m, s, rc in failed:
            print(f"  FAILED: {m} seed={s} rc={rc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
