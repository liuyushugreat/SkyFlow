# Compute plan (S8) — local RTX 4090 vs. cloud

Written 2026-10-05 from measurements on this machine; every number below is
taken from `results/benchmark/*.json` or `results/_dryrun/*/seed42/{metrics,history}.json`
(git `7741ba9` + S8 working tree). Nothing is estimated except where marked **assumption**.

## 1. Environment

| item | value |
|---|---|
| host | DESKTOP-G0C01AN, Windows 11, i9-14900K, 137 GB RAM |
| GPU | NVIDIA GeForce RTX 4090, 24 GB (WDDM; ~1.4 GB used by the desktop) |
| software | Python 3.14.3, torch 2.14.0+cu130, CUDA 13.0, driver 591.86, TF32 on, AMP off |
| data cache | `D:\SkyFlowCache` (env `SKYFLOW_CACHE_DIR`, outside the synced workspace), 4.86 GB for N=500 train/val/test |

Cache build (CPU, numpy): val/test 600 snapshots each in ~6 min; train 2400 snapshots in 46 min
(single process; slower per snapshot than the 1-scenario probe, 0.43 s/snap, because of GC pressure
on the growing list). `abl_telemetry_only` needs its own cache (different `features.input_set`): same cost again.

Dataset (configs/default.yaml): N=500, 5 km, 40/10/10 scenarios × 60 s, 1 Hz snapshots →
2400 / 600 / 600 snapshots; proximity candidates 71k pairs/snapshot; positives 0.160 % (train),
0.153 % (val), 0.178 % (test).

## 2. Measured epoch cost

### 2.1 Memory / concurrency (TR-GAT, 1 epoch, `benchmark_epoch.py`)

| setting | s/epoch (train+val) | peak allocated | reserved | GPU util | file |
|---|---|---|---|---|---|
| micro_batch = 4 windows (old default), 1 process | 345.8 | 19.0 GB | ≈24 GB | 31 % | `DESKTOP-G0C01AN_c1_TR-GAT.json` |
| micro_batch = 1 window, 1 process | **83.7** | 7.75 GB | 12.0 GB | 36 % | `…_c1_mb1_TR-GAT.json` |
| micro_batch = 1 + `expandable_segments`, 1 process | 128.2 | 7.38 GB | 8.2 GB | 30 % | `…_c1_mb1_exp_TR-GAT.json` |
| micro_batch = 1 + `expandable_segments`, **2 processes** | 125.1 each | 7.38 GB each | — | 93 % | `…_c2a_…`, `…_c2b_…` |

Findings:
- Keeping 40 snapshots' graphs alive before `backward()` pushed the allocator to the 24 GB limit and
  made the epoch 4× slower (cudaFree/retry thrash). `training.micro_batch_windows: 1` (gradient
  accumulation, identical maths) is now the default.
- Two processes without `expandable_segments` do not fit (2 × 12 GB reserved → WDDM paging; a 1-epoch
  pair was killed after 25 min). With `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  (now set by `run_task.py`) two tasks run at the single-task speed → **throughput 62.5 s/epoch-equivalent,
  1.34× the best single process**. GPU is then at 93 %; 3 processes would neither fit (3 × 9 GB) nor help.
- **Optimal concurrency on this card: 2.**

### 2.2 Per-method epoch time at concurrency 2 (dry run: every method, 1 seed, 2 epochs, real cache)

`train_s / 2` from `results/_dryrun/<method>/seed42/metrics.json` (train pass + validation):

| method | params | s/epoch | peak GPU | notes |
|---|---|---|---|---|
| TR-GAT | 1.145 M | 123.0 | 8.4 GB | |
| TR-GAT-NT | 1.143 M | 122.7 | 8.3 GB | |
| GAT-S | 0.169 M | 42.6 | 5.0 GB | |
| STGCN | 0.168 M | 34.2 | 5.0 GB | |
| LSTM-P | 0.592 M | 21.4 | 5.0 GB | cuDNN disabled for the RNN |
| Tfm-P | 0.696 M | 27.7 | 5.0 GB | |
| abl_no_gating | 1.132 M | 92.0 | 8.3 GB | |
| abl_no_gru | 1.116 M | 118.6 | 8.4 GB | |
| abl_bce | 1.145 M | 118.6 | 8.4 GB | |
| abl_telemetry_only | 1.145 M | 32.5 | 8.4 GB | UAV nodes + approaches edges only; its own cache took 28 min |
| CPA-Rule | 0 | — | 0.8 GB | eval only, 3.2 min incl. val threshold search |
| VO | 0 | — | 0.8 GB | eval only, 5.8 min (python loop, P95 645 ms) |

Per-task fixed overhead: cache load ≈ 4 s, test evaluation ≈ 10–20 s — negligible.

After 2 epochs every learned model still has val/test F1 = 0 (0.16 % positives, threshold 0.42); this is
why `min_epochs` was raised to 30 and early stopping is disabled while best val F1 == 0.

## 3. Task list and duration

Epoch budget: `training.epochs = 150`, early stopping patience 15 after `min_epochs = 30`.
There is **no historical convergence evidence** (`docs/repo_map.md` §: no per-epoch logs survived), so
per the plan the upper bound uses max_epochs = 150. Early stopping can only shorten it.

Per-run worst case = s/epoch × 150:

| run | h / run (150 ep) | seeds | process-hours |
|---|---|---|---|
| TR-GAT | 5.13 | 3 | 15.4 |
| TR-GAT-NT | 5.11 | 3 | 15.3 |
| GAT-S | 1.78 | 3 | 5.3 |
| STGCN | 1.43 | 3 | 4.3 |
| LSTM-P | 0.89 | 3 | 2.7 |
| Tfm-P | 1.15 | 3 | 3.5 |
| CPA-Rule, VO | 0.15 | 1 | 0.15 |
| **main total** | | | **46.6** |
| abl_no_gating | 3.83 | 3 | 11.5 |
| abl_no_gru | 4.94 | 3 | 14.8 |
| abl_bce | 4.94 | 3 | 14.8 |
| abl_telemetry_only | 1.35 | 3 | 4.1 |
| **ablation total (3 seeds)** | | | **45.2** |
| ablation total (1 seed) | | | 15.1 |

Wall-clock at concurrency 2 (process-hours / 2, both slots busy):

| block | worst case (150 ep) | if runs stop at ~60 epochs |
|---|---|---|
| main (6 methods × 3 seeds + rules) | 23.3 h | 9.4 h |
| ablations, 3 seeds | 22.6 h | 9.0 h |
| ablations, 1 seed | 7.5 h | 3.0 h |
| S15 evaluation (eval_only 1000 epochs × 22 ckpts, robustness 9 conditions incl. 9 test-cache builds ≈ 6 min each, scaling N≤2000, attention) | ≈ 3 h (assumption; cache builds dominate) | ≈ 3 h |
| **total, 3 ablation seeds** | **≈ 49 h** | ≈ 21 h |

**GPU-hours: ≈ 92 process-hours ≈ 46 h of one RTX 4090 at concurrency 2 (worst case).**

## 4. Schedule on the local card only

Start 2026-10-05 ≈ 20:30 (dry run finished 19:36):

- main finished by **2026-10-06 ≈ 20:00** (worst case), ablations (3 seeds) by **2026-10-07 ≈ 18:30**,
  S15 evaluation by **2026-10-07 ≈ 21:30**.
- Deadline for the rent/no-rent rule: all training done before **2026-10-10 20:00** → **≈ 73 h of slack**
  even in the worst case; enough to absorb one full restart of the main block.

Decision rule (from the plan): local card finishes before 10-10 20:00 → **do not rent**.

Cloud cost formula kept for reference: `cost = GPU-hours × hourly price`. With ≈ 46 GPU-h on a
4090-class card the price field is left blank: `46 h × ____ ¥/h = ____ ¥`. Not needed under the decision.

## 5. Recommendations

- Ablation seeds: **3** (time is sufficient; drop to 1 only if the main block overruns 10-07 12:00).
- Run order (`run_main.py --max_concurrent 2 --resume`): main block first, TR-GAT and TR-GAT-NT
  started first so the two long families overlap each other; ablations second.
- Keep the machine awake (no sleep), pause BaiduSyncdisk on the workspace during the run; checkpoints
  are written with retry (`save_with_retry`) but `results/main` itself is inside the synced folder.
- Do not start anything else on the GPU while the two slots are busy (a third CUDA process re-creates
  the WDDM paging collapse measured above).
- `abl_telemetry_only`'s own cache (28 min CPU) was already built by the dry run and is reused.

## 6. Commands

```powershell
# 1. main experiment (3 seeds, 2 concurrent)
python scripts/run_main.py --config configs/default.yaml --results_dir results/main `
    --methods TR-GAT TR-GAT-NT GAT-S STGCN LSTM-P Tfm-P CPA-Rule VO --seeds 42 123 456 --max_concurrent 2 --resume

# 2. ablations (3 seeds)
python scripts/run_main.py --config configs/default.yaml --results_dir results/main `
    --methods abl_no_gating abl_no_gru abl_bce abl_telemetry_only --seeds 42 123 456 --max_concurrent 2 --resume
```

## 7. Restart note (2026-10-05 evening, protocol fix S8b)

The first main launch (20:20) was stopped after 15 TR-GAT epochs: val F1 at the fixed threshold 0.42
stayed at 0 and the "best" checkpoint was frozen at epoch 1. Diagnosis on the 2-epoch dry-run checkpoint
(CPU, 100 val snapshots): predictions compressed into [0.09, 0.21], AUROC 0.72, AUPRC 0.005. Two causes,
both protocol-level and therefore fixed uniformly for every learned method before relaunching:

1. **Raw node features were fed unscaled** (metres up to 5000 next to flags). Fix: `InputStandardizer`
   (per-feature mean/std buffers fitted on the training split, stored in the checkpoint) in TR-GAT and all
   learned baselines; switch `features.normalize_inputs` (default `true`, `false` = legacy raw inputs).
   The switch is excluded from the dataset cache key, so the 5 km cache is reused unchanged.
2. **Model selection at a fixed threshold** is meaningless at a 0.16 % positive rate. Fix:
   `training.threshold_mode: val` (default) selects the F1-optimal threshold on the validation split each
   epoch (AUPRC also logged), early-stops on that best-F1, stores the threshold in the checkpoint and
   applies it unchanged to the test split (and in `eval_only`, robustness and scaling). `fixed` restores the
   old behaviour. Rule baselines are unaffected (binary output).

Verification before relaunch: `pytest -q` 106 passed; 12-method smoke (`configs/smoke.yaml`) + `eval_only`,
`aggregate_main`, `aggregate_ablation`, `run_robustness`, `run_scaling`, `analyze_attention_aoi` all run on
the smoke outputs; 3-epoch TR-GAT sanity on the real cache (`results/_sanity`, see `logs/sanity_trgat.log`).
`results/main` from the aborted launch was deleted. The schedule in §4 shifts by the restart time only
(≈ 1.5 h); the no-rent decision is unchanged.

## 8. Second restart (2026-10-05 night, pair-feature fix S8c)

The S8b relaunch (21:33) trained TR-GAT for 44 epochs (seed 42, `batch_windows: 4`) and plateaued at
val AUPRC ≈ 0.12 / best-F1 ≈ 0.21 (`logs/aborted_main_bw4_trgat_seed42.log`), i.e. **below the CPA
rule** on the same split (test F1 0.353, AUPRC 0.133, `results/_sanity/CPA-Rule/seed42`). A diagnostic
with `batch_windows: 1` (seed 123, `logs/sanity_bw1_trgat_seed123.log`) reached the same plateau by epoch
12-13, so the optimiser budget is not the limit. Two protocol issues were found:

1. **Unequal pair information.** TR-GAT's head received `e_ij = [Δp, Δv, δ]`, while the four learned
   baselines scored `[h_i, h_j]` only - they never saw the relative state. Fix: a single
   `pair_scorer_input` used by every learned method (`features.pair_edge_features`, excluded from the cache
   key). `none` = legacy baselines, `kinematics` = legacy TR-GAT (7-d).
2. **The head had to rediscover CPA geometry** (a division) from raw Δp/Δv under a 0.16 % positive rate.
   Fix: mode `geometry` (default, 12-d) appends `[t_cpa/T, d_cpa_h, |dz_cpa|, range, closing speed]`
   computed from the *observed, delayed* relative state - the same inputs the CPA rule uses, no truth or
   label information. Every learned model thereby contains the rule as a special case; the question the
   experiment asks becomes "what does learning add on top of the rule's decision variables".
3. `training.batch_windows` 4 → 1 (240 optimiser steps/epoch at the same epoch time; same maths).

Verification: `pytest -q` 128 passed (new `tests/test_pair_geometry_s8c.py`); 12-epoch sanity of TR-GAT
and GAT-S with `geometry` on the real cache (`results/_sanity/geo`, `logs/sanity_geo_*.log`) before the
relaunch. Windows "Balanced" power plan throttled the CPU to ~60 % and doubled epoch time after ~22:00;
the "High performance" plan (`powercfg /setactive 8c5e7fda-...`) restores ~150 s/epoch - switch back
afterwards.

## 9. Third restart (2026-10-06 morning, plan context S8d)

The S8c relaunch (00:06) completed TR-GAT, TR-GAT-NT, GAT-S, STGCN and LSTM-P (3 seeds each; archived
under `results/archive/main_noplan_s8c`, not committed). Under the leakage-free protocol **every learned
model tied the CPA rule** (test F1 0.34-0.36 vs 0.353; CDR ≈ 0.32-0.34 for all), and the per-regime
breakdown showed why: CDR ≈ 0.51-0.53 for conflicts with TTC ≤ 15 s but only 0.15-0.19 for TTC > 15 s,
even for *planned* crossings. The labels come from the true 6-DoF future including waypoint turns, while
the 20 observation-only features contain no route information, so conflicts beyond the straight-line
horizon are invisible to rule and model alike - a data ceiling, not a model limit.

Fix (user decision, option A): `features.plan_context: true` adds 12 features of **filed flight-plan
context** per UAV - next filed waypoint and planned positions +10/+20/+30 s along the filed route,
relative to the observed position - computed from the filed plan and the *observed* state only (zero for
non-cooperative UAVs). This is information a UTM node legitimately holds; non-conforming aircraft carry a
plan that disagrees with their telemetry. Two consequences for the protocol: a plan-aware rule baseline
`Plan-CPA` (same interval test on the filed polyline) and an ablation `abl_no_plan` (= the S8c setting).
Implementation notes: the simulator moves every UAV from t = 0 (`start_time` only gates steering), so no
departure delay is modelled; the active route segment is the nearest one whose target waypoint is ahead of
the observed heading (handles overshoot loops at waypoints and hub revisits). On a 200-UAV check the plan
projection error at +30 s is 26 m median / 115 m P90 versus 82 m / 541 m for linear extrapolation.

Cache keys change (features section): train `c3c0ab967d551f33`, val `787283698c56ea4f`, test
`05d90a3003363c93` (`logs/build_cache_s8d_*.log`). `pytest -q` 139 passed (new
`tests/test_plan_context_s8d.py`). Compute: 6 learned methods x 3 seeds + 5 ablations x 3 seeds, two
concurrent on the 4090 at ≈ 110 s/epoch with early stopping around epoch 35-60 → ≈ 20-24 h total.

**Second correction before relaunch (09:07 → 09:35).** The first S8d chain showed TR-GAT val AUPRC 0.168
at epoch 8 - identical to the no-plan run. The plan context entered only as *per-node* features, so the
pair head still had to combine two 12-d route vectors to infer whether the *planned* trajectories cross,
while Plan-CPA gets that quantity directly. New pair mode `features.pair_edge_features: geometry_plan`
(default, 20-d) appends to the S8c geometry the planned horizontal/vertical separation of the pair at
+10/+20/+30 s, the planned minimum horizontal separation and a no-plan flag (observed-velocity fallback for
UAVs without a plan). Same inputs as Plan-CPA, no truth. `abl_no_plan` now switches both `plan_context`
and the pair mode back to the S8c setting. Not in the cache key (pair features are computed on the fly), so
the caches above are reused; `pytest -q` 143 passed; smoke run of TR-GAT, GAT-S and `abl_no_plan` on
`configs/smoke.yaml` OK. The chain was relaunched with `--resume` (CPA-Rule and Plan-CPA seed 42 kept).

## 10. Fourth restart (2026-10-06 afternoon, S8e: sync + gate + heterogeneous links)

Partial S8d-2 results (`results/archive/main_plan_s8d2`, test split): CPA-Rule F1 0.353, Plan-CPA 0.336,
TR-GAT 0.387 ± 0.002 (3 seeds), TR-GAT-NT 0.390 ± 0.001, **GAT-S 0.403 ± 0.003** (0.18 M parameters vs
1.15 M), STGCN seed 42 0.408. Learned models finally beat both rules (AUPRC 0.35–0.39 vs 0.13), but the
temporal machinery of TR-GAT brought nothing. Three verifiable reasons:

1. Reports were used at their *reported* positions: with per-UAV latency 0.5–1.2 s (+3 s non-cooperative)
   and 10–15 m/s, two reports of different age are misaligned by up to ~15 m - more than the 10 m
   threshold. The AoI only entered as one scalar max(a_i, a_j).
2. The GRU state was detached after every snapshot (truncated BPTT of length 1), and the training
   distribution had near-constant AoI and no packet loss, so the temporal path had no signal to learn.
3. 1000 warm-up steps (4 epochs at ~0 lr) for TR-GAT only; baselines had no schedule at all.

Changes (user decision 16:17, "do whatever most improves the ISCAS odds"), all behind config switches:

- `features.pair_edge_features: geometry_plan_sync` (22-d): every report is dead-reckoned to the common
  epoch with its own age (p' = p + v·AoI) before Δp, CPA geometry and plan geometry are formed; both ages
  are given. Both CPA rules apply the same synchronisation (`aoi_sync`). `abl_no_sync` = S8d-2 setting.
- `model.use_conformance_gate: true` (TR-GAT only): w_i = σ(MLP[s_i ‖ r_i]) with r_i the plan/telemetry
  residual; pair geometry on w·plan + (1-w)·extrapolation plus (w_i, w_j) is appended to the head input
  (+9). `abl_no_conf_gate` removes it.
- `training.tbptt_detach: false` (BPTT through the K-snapshot window; `abl_tbptt` = legacy),
  `training.warmup_steps: 200`, `training.scheduler: warmup_cosine` now applied to every learned baseline.
- `sim.link_mix`: every scenario draws packet loss U(0, 0.3) and latency range [U(0.3,0.8), U(1,3)] s
  from its own seed (train/val/test alike; explicit obs_params in the robustness sweeps still override).
  Changes the cache key (sim section).

Verification: `pytest -q` 155 passed (new `tests/test_s8e_sync_gate_links.py`, 12 tests incl. one
end-to-end TR-GAT step with gate + BPTT); smoke run of CPA-Rule, Plan-CPA, TR-GAT, GAT-S, abl_no_conf_gate,
abl_no_plan OK. The S8d-2 chain driver and run_main were stopped (STGCN workers left to finish); their
results are archived, not committed.

Caches (link_mix): train `1f4a3399a8306873` (1018 s, 273,457 positives - labels unchanged), val
`3c42abfaf48df364`, test `66af77676a246eb1` (`logs/build_cache_s8e.log`).

**Validation-only capacity selection (seed 42, `results/_select`, `logs/select*.log`):**

| run | params | val best-F1 (epoch) | val AUPRC | note |
|---|---|---|---|---|
| TR-GAT L=4, d=128 (`configs/trgat_large.yaml`) | 1.16 M | 0.403 @36 | 0.372 | stopped at epoch 36 (still rising slowly) |
| TR-GAT L=2, d=64 (now `default.yaml`) | 0.23 M | **0.426 @46** (0.422 @36) | 0.415 | early stop 61; test F1 0.418 |
| GAT-S (reference) | 0.18 M | 0.419 @27 | 0.414 | test F1 0.416 |

Epoch-1 val AUPRC of TR-GAT went from 0.007 (S8d-2) to 0.175 with the shorter warm-up and BPTT. The
compact TR-GAT was selected on val and its seed-42 run reused as `results/main/TR-GAT/seed42` (identical
config; `metrics.json.config_path` still names the former `trgat_compact.yaml`). Full round launched
18:49 (`logs/chain_s8e.log`): main 9 methods x 3 seeds → 8 ablations x 3 seeds (most important first)
→ S15 → paper build → Balanced power plan. Expected ≈ 25–30 h.

## 11. S8f (10-06 evening): event-level metrics, extrapolating robustness sweep, CPU latency

Evaluation-only additions (no label / model / training change; the S8e chain keeps running):

- `skyflow/experiments/events.py` + `scripts/eval_events.py`: operators reason in conflict *events*
  (maximal run of positive snapshots of a pair), not pair-snapshots. Reported per method x seed:
  event CDR, timely CDR (first alert with ttc >= 10 s, among events where that was possible), lead
  time at first alert, alert-episode precision, false alert episodes per UAV-hour, and the same with a
  3-of-3 persistence filter (`--persistence 1 3`). Paired t-tests vs TR-GAT (Bonferroni) in
  `results/events.json`. Probe on the three finished TR-GAT seeds (`results/_events_probe`, not
  committed): 3,530 events / 83.3 UAV-h; event CDR 0.855±0.001, timely 0.723, lead median 22.0 s,
  episode precision 0.255, 209 false episodes per UAV-h (42.4 true events per UAV-h); persistence 3
  trades event CDR 0.739 for 86.5 false episodes/UAV-h. Alerts flicker (~2 episodes per detected event).
- `scripts/run_robustness.py`: the sweep now switches `sim.link_mix` off (otherwise the per-scenario
  draw silently overrode the fixed level), covers latency {0..5} s and loss {0..0.5} (training mix tops
  out at 3 s / 0.3; rows flagged `in_train_range`), adds STGCN and AUPRC. Figures shade the
  extrapolation region; macros `\cdrLat<M>InMax`, `\cdrLatOodDrop<M>`, ...
- `scripts/run_s15.ps1`: stages `events` and `cpulat` (TR-GAT seed 42 inference on CPU,
  `results/eval_cpu`) added; `make_tables` writes `tab_events.tex`, `make_figures` writes `fig_lead.pdf`,
  `make_macros` adds `\ev...` and `\latCpu...` macros.

Verification: `pytest -q` 164 passed (`tests/test_events_s8f.py`: hand-built event table with exact
counts, oracle / silent detectors, gap and scenario-boundary handling, persistence filter, window grouping
identical to the trainer, TR-GAT per-snapshot scores reproduce `evaluate()` TP/FP/FN, rule + learned
baselines run, robustness condition overrides link_mix / changes the cache key / OOD flag).

### 11.1 Operational layer, SOC curves, matched false-alert budgets, near-miss analysis (10-06 night)

Motivation: 209 false alert episodes per UAV-hour at the validation-F1 threshold is hard to defend
even against 42 true events per UAV-hour. Still evaluation-only (labels, models and training untouched):

- **A — operational layer** (`events.py`): per-pair EMA smoothing of the score along the pair track
  (`smooth_scores`, alpha; reset when the pair re-enters the candidate set), hysteresis alerting
  (`hysteresis_alerts`, on at theta, off below theta-h) and M-of-M persistence. All parameters are
  selected on the validation split only.
- **D — SOC curve + budget-matched operating points** (Kuchar 1996 system operating characteristic):
  `soc_curve` sweeps a 21-point quantile grid of thresholds for every (alpha, hysteresis) in
  {1, 0.5, 0.25} x {0, 0.15}; `select_operating_point` picks, on val, the point with the highest event
  CDR whose false-episode rate is <= a budget, then that fixed (threshold, alpha, h) is evaluated on
  test. Budgets: fixed 20 / 50 / 100 per UAV-h and the CPA-Rule's own val false rate
  (`match_CPA-Rule`, reported first in tables / macros). Outputs `results/events_soc.csv`,
  `events_budget.csv`, `events_budget_summary.csv`; `fig_soc.pdf` (log-x false episodes vs event CDR,
  raw curve per method, dotted TR-GAT + layer, star = matched-budget point); `tab_events.tex` gains a
  "matched FA budget" block; macros `\evB<Kind><Key><M>`, `\evB<Key><M>` (first kind), `\soc...`.
- **E — near-miss analysis** (`skyflow/experiments/nearmiss.py`, `scripts/analyze_nearmiss.py`):
  re-simulates the deterministic truth, checks that it reproduces the stored labels exactly, and
  computes the normalised minimum separation rho = min_k max(dh/10 m, dv/3 m) over the 30 s window for
  false alert rows / false episodes / a 200k random negative sample. Output `results/nearmiss.csv`,
  macros `\nm<Group><Key><M>`, `\nmMismatch<M>`. S15 stage `nearmiss`.

Probe on TR-GAT seed 42 (`results/_events_probe`, not committed; test split, N=500):
hysteresis h=0.15 alone keeps event CDR 0.858 and cuts false episodes to 118/UAV-h (-43 %);
alpha=0.5, h=0.15 reaches CDR 0.854 at 100/UAV-h (-52 %) and ~0.94 at the raw false rate (~207).
Val-selected budget points evaluated on test: 20/UAV-h -> CDR 0.548, lead 8.9 s (17.0 false);
50 -> 0.769, 19.4 s (60.8); 100 -> 0.881, 22.9 s (117). Near-miss: 0 label mismatches
(78,749 positives + 200k negatives re-simulated); false rows median rho 2.33 (24 % < 1.5, 42 % < 2,
65 % < 3), false episodes median 2.70, random negatives median 84.6 (0.2 % < 2): the false alerts are
near-misses, not random pairs. Runtime ~65 s per checkpoint after pruning / radix-sort optimisation.

Verification: `pytest -q` 171 passed (new: sorted table / positions, EMA vs naive loop, hysteresis vs
naive loop, SOC curve monotone CDR and pruning exactness, budget selection, synthetic rho, re-simulated
truth reproduces labels). Dry runs of `make_tables` / `make_macros` / `make_figures` on the probe
directory produce `tab_events.tex`, 131 macros and `fig_soc.pdf` (TrueType embedded only).

Deferred (decide after the chain): **B** — pair-level temporal memory inside TR-GAT (a model change,
needs retraining of all seeds). **C** (temporal consistency loss) not recommended.

### 11.2 Paper skeleton and 4-page budget (10-06 night, while the chain runs)

Done without any result number (all sentences reference macros; comparative wording is marked `%%CHECK`
in the tex and must be confirmed against `results/` before submission):

- `paper/skyflow_iscas2027.tex`: Results rewritten as Pair-snapshot / Event-level + false-alert budget /
  Beyond training link conditions / Edge latency and scaling / Ablation; abstract and conclusion updated;
  setup table removed (its unique rows now in the "Training and protocol" paragraph with new macros
  `\cfgWeightDecay`, `\cfgWarmupSteps`, `\cfgDropout`); the three single-column result figures replaced by one
  full-width `fig_results.pdf` (SOC | CDR vs latency | CDR vs loss | scaling; `make_figures.fig_results`, legend
  positions via `--rob_legend_loc/--scal_legend_loc`); attention-vs-AoI figure dropped (one optional clause in
  the ablation paragraph); compact author block; gate equation inlined (equations now 1-4).
- `paper/figs/arch.tex`: adds the AoI-synchronised pair geometry box (shared with baselines), the
  intent-conformance gate box, BPTT note, operational layer note; equation references updated.
- Tables fit one column: `tab_main` shows mean+-std on F1 only (other columns mean; `\tabMainNote` reports
  their max std), `tab_events` mean+-std on the two event-CDR columns (`\tabEventsNote`), `tab_ablation`
  two variants per row (`--ablation_wide` restores the old layout); `\setlength{\tabcolsep}{2.5pt}`.
- README rewritten for the current pipeline (no numbers; points to `results/` files); `run.sh` is now the bash
  twin of `scripts/run_s15.ps1`; legacy `reproduce_table*.sh` headers no longer carry typed numbers.

Page estimate (temp build with partial results, tables padded to 9 methods, dummy robustness/scaling CSVs
for layout only): body ends ~0.4 page into page 5. Remaining cuts, in order, once the real numbers are in:
Results prose to the confirmed claims only (-0.2 col), drop the per-cause sentence, `\vspace` around floats,
shorten the robustness sentence to TR-GAT / TR-GAT-NT / CPA-Rule, Eq. (2) to one line; last resort: drop
the hard-regime sentence or two ablation rows from the prose (the table keeps them).

## 12. S8g: what the finished runs say, and the state-carry variant (10-07 morning)

Main comparison finished 10-07 02:32 (`results/main`, 3 seeds each; `stdout.log` TEST lines):

| method | test F1 (3 seeds) | CDR | FAR |
|---|---|---|---|
| TR-GAT | 0.4164 +- 0.0015 | 0.469 | 0.626 |
| TR-GAT-NT | 0.4165 +- 0.0004 | 0.469 | 0.625 |
| GAT-S | 0.4149 +- 0.0008 | 0.424 | 0.593 |
| STGCN | 0.4203 +- 0.0017 | 0.439 | 0.597 |
| LSTM-P | 0.4261 +- 0.0007 | 0.456 | 0.600 |
| Tfm-P | 0.4229 +- 0.0006 | 0.464 | 0.612 |
| Plan-CPA / CPA-Rule / VO (1 run) | 0.382 / 0.363 / 0.165 | | |

Partial S15 on the first three learned methods (`results/_events_partial`, `results/_robust_partial`, evaluation
only, no training): at a fixed validation budget the event-level CDR of TR-GAT, GAT-S and STGCN agree within
1 pt; TR-GAT leads only in timely CDR at 50 false episodes/UAV-h (0.551 +- 0.011 vs 0.518 / 0.515) with a longer
median lead (19.4 s vs 16 s).  Beyond the training latency range TR-GAT degrades *faster* than GAT-S/STGCN
(F1 0.194 vs 0.250 at 3.0 s); packet loss up to 0.5 costs every method < 0.3 pt.

Reading: all learned pair scorers sit on the same plateau (0.415-0.426); the architecture contributes nothing
measurable on top of the shared e_ij, and the recurrence over a 10-snapshot (10 s) window does not help
(TR-GAT-NT = TR-GAT; abl_no_gru pending).  The user chose to try one model change (option B) before deciding
the paper's framing.

### 12.1 Variant: recurrent state carried over the scenario (`training.state_carry`)

The GRU state was reset at every K=10 window in training *and* evaluation, i.e. 10 s of memory.  S8g adds
`training.state_carry = "window" | "scenario"`: in "scenario" mode the windows of a scenario are processed in
time order, the state is carried across them (detached, truncated BPTT per window), and reset only at the first
snapshot of a scenario - the same schedule in `SkyFlowTrainer.train/evaluate`, `events.iter_scores` and the two
analysis scripts.  Default stays "window" (reproduces every S8e run bit-for-bit, including the RNG draws of the
window shuffle); method `TR-GAT-SC` switches it on.  Tests: `tests/test_state_carry_s8g.py` (11).

Cost: none at inference (same per-snapshot work), no cache change.  Expected effect: honest uncertainty - longer
memory can only help the conformance gate and the per-UAV state (wind drift / non-conforming trends accumulate
over tens of seconds); the OOD-latency degradation may get worse.

### 12.2 Chain restructuring

The S8e driver was stopped at 07:03 (its two `abl_no_gru` children s42/s456 were left running); the first version
of the S8g edit was imported by `abl_no_gru/seed123` before the trainer import line landed -> NameError at start;
that seed is re-queued.  `chain_s8g.ps1` (Temp) waits for the two children, then runs
`TR-GAT-SC` (3 seeds, first in the queue) + the remaining ablations with `--resume`, then S15, paper build,
Balanced power plan.  A first 3-concurrent probe was killed: at 23.9/24.5 GB the epoch time went from ~90 s to
590 s.

Decision rule for B (after `TR-GAT-SC/seed42`): compare val best-F1 / test F1 / event-level timely CDR with
`TR-GAT/seed42`.  If not clearly better (> 1 pt F1 or > 3 pt timely CDR, consistent on seeds 123/456), keep the
S8e model as TR-GAT and reframe the paper around the leakage-free benchmark + event-level/budget protocol +
operational layer; otherwise make "scenario" the default, re-run the ablations on it and treat the "window"
runs as the ablation `abl_state_window`.

### 12.3 Outcome of B (10-07 10:25) - null result, option closed

| run | best val F1 (epoch) | test F1 | test CDR / FAR |
|---|---|---|---|
| TR-GAT seed42 (window) | 0.4262 (46) | 0.4180 | 0.468 / 0.622 |
| TR-GAT-SC seed42 (scenario) | 0.4228 (46) | 0.4172 | 0.446 / 0.608 |
| TR-GAT seed123 | 0.4211 (41) | 0.4151 | 0.471 / 0.629 |
| TR-GAT-SC seed123 | 0.4219 (48) | 0.4195 | 0.461 / 0.615 |

The learning curves coincide epoch by epoch; the differences (-0.1 / +0.4 pt) are inside the seed spread, and
`abl_no_gru` (0.4161 / 0.4176 on s42 / s456) shows the recurrence carries no signal at all.  Seed 456 of
TR-GAT-SC was stopped at epoch 5 to free the GPU slot; `training.state_carry` stays "window" by default and
TR-GAT-SC is kept in `results/main` as the recorded negative (group "variant", not in MAIN_METHODS, so it does
not enter the paper tables).

Consequence, agreed with the user: the paper is reframed (commit d13e4be) - title "What does learning add to
rule-based UAV conflict detection under stale surveillance at the edge?", contributions = leakage-free protocol
+ AoI-synchronised shared pair geometry, event-level evaluation with budget / operational layer / near-miss
analysis, and a controlled comparison in which TR-GAT is the graph-based reference rather than the claimed
winner.  New macros for that text: `\learnedFOne{Min,Max,SpreadPts,BestName}`, `\ruleFOneBest{,Name}`,
`\gain{Min,Best}LearnedOverRulePts`, `\evBLearned{Cdr,Timely,LeadMed}{Min,Max,BestName,SpreadPts}`,
`\ablMaxAbsDFOne{Pts,Name,Signed}`, `\ablNSignificant`, `\ablNVariants`, `\rob{Lat,Loss}OodDropPct{Worst,Best}{,Name}`.
`scripts/check_paper.py` is the S19 gate (fonts, page rule, undefined macros, `??` markers, smoke flag, `%%CHECK`).

## Conclusion

**Do not rent: the local 4090 (2 concurrent tasks) finishes the main experiment and 3-seed ablations by about 10-07 evening even if every run goes to 150 epochs.**
