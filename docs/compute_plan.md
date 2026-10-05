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

## Conclusion

**Do not rent: the local 4090 (2 concurrent tasks) finishes the main experiment and 3-seed ablations by about 10-07 evening even if every run goes to 150 epochs.**
