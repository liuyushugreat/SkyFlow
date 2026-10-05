# SkyFlow 仓库摸底报告（S1，只读分析）

- 分析对象：`pressRequire/SkyFlow/SkyFlowCode`，git `46b2300`（唯一一次 commit，工作区干净）
- 分析日期：2026-10-05
- 分析方法：通读全部源码 + 运行现有 `pytest -q`（23 passed）+ 一个临时计时探针（N=100/500，10 s 场景，探针文件未入库）

> 结论先行：**仓库当前实现与论文描述有多处根本性不一致**（见 §5），其中最严重的是：
> (a) 冲突标签不是“30 s 前视 + 6-DoF 积分”，而是“当前时刻水平 <10 m 且垂直 <3 m”；
> (b) 默认 5 km 场景下正样本几乎为零（N=500 时每快照 0.00 个真值冲突）；
> (c) ADS-B 延迟/丢包完全没有注入，观测 = 真值；
> (d) 论文 Table 3–8 的数字在仓库中没有任何可追溯产物，`outputs/` 里只有一次 `--quick` 运行（50 UAV，F1≈0.002）。
> 这些直接影响 S2–S6 的设计，需要在 S2 开始前由你拍板（见文末“待决策”）。

---

## 1. 目录树（3 层）与作用

```
SkyFlowCode/
├── configs/default.yaml          # 全部超参（model/data/training）
├── skyflow/                      # 核心包
│   ├── config.py                 # dataclass 配置 + YAML 加载
│   ├── data/
│   │   ├── urbanair500.py        # 仿真器：航迹、风、GPS 噪声、真值标签、数据集采样
│   │   ├── tkg_builder.py        # TKG 构建：节点特征 (23 维)、6 类关系词表、δ
│   │   └── sdd_adapter.py        # SDD 适配（本次改投不用）
│   ├── models/
│   │   ├── tr_gat.py             # TRGATLayer (Eq.3–5) + TRGAT (input_proj → L 层 → GRUCell)
│   │   ├── temporal_encoding.py  # φ(δ) 正弦编码，T=300 s
│   │   ├── conflict_head.py      # 成对打分 MLP (Eq.6)，含 no_edge_token
│   │   └── resolution.py         # PGD 避让（本次改投不写入论文）
│   ├── baselines/                # VO / LSTM-P / Tfm-P / STGCN / GAT-S
│   ├── training/
│   │   ├── trainer.py            # 训练循环、K=10 窗口、评估、多 seed
│   │   ├── losses.py             # FocalLoss γ=2 α=0.75
│   │   └── metrics.py            # CDR/FAR/F1、RegimeMetrics(未被调用)、t 检验、LatencyTimer
│   └── utils/visualization.py    # 画图
├── scripts/
│   ├── train.py / evaluate.py / run_baselines.py / eval_scalability.py
│   ├── reproduce_paper.py        # 一键流程（数据→TR-GAT 多 seed→NT 消融→基线→扩展性→t 检验→图）
│   ├── eval_sdd_transfer.py      # SDD（弃用）
│   └── *.sh                      # 包装脚本
├── supplementary_experiments/    # rebuttal 期补做的两个实验及其 JSON 结果
├── tests/                        # 23 个单元测试（tkg_builder / tr_gat / metrics）
├── outputs/                      # 未入库(.gitignore)，含一次 --quick 运行的结果与 best_model.pt
└── docs/                         # 本报告 + SKYFLOW_PLAN.md
```

---

## 2. 关键组件定位

### 2.1 UrbanAir-500 仿真器（`skyflow/data/urbanair500.py`）

| 组件 | 位置 | 现状 |
|---|---|---|
| 航迹生成 | `generate_flight_plans` L114；`simulate_scenario` L150 | 3–7 个随机航点，一阶速度跟踪 `steer=(v_des−v)·0.3`，巡航 8–22 m/s。**不是 6-DoF 旋翼动力学**，是质点一阶跟踪。 |
| 风场 | L181 `wind_field = randn(n_epochs,3)*wind_std` | 每个 epoch **一个全局风矢量**（白噪声，σ=2 m/s），对所有 UAV 相同，`v += wind*0.1*dt`。没有空间风场，没有“历史测量”。 |
| GPS 噪声 | L187 `gps_noise = randn*gps_cep*0.01` | σ=0.025 m，而且是**直接加到真实位置上**（L211 `positions += v·dt + gps_noise`），即修改了真值而非生成观测。观测 = 真值。 |
| ADS-B 延迟 | `adsb_latency_range=(0.5,1.2)` 仅存为属性 | **从未使用**。没有延迟、没有丢包。 |
| 冲突标签 | `_detect_ground_truth_conflicts` L262 | 对每对 (i,j)：**当前时刻** `h_dist<10 ∧ v_dist<3` 才记为冲突。**没有前视窗口、没有轨迹积分**。`time_to_conflict` 仅作为附带字段计算，不参与判定。纯 Python 双循环 O(N²)。 |
| 内置避让 / f_avoid 来源 | `avoiding = zeros(bool)` L166 | **没有任何避让逻辑**，`avoiding` 永远全 False → 特征 `f_avoid≡0`。 |
| 走廊预约 | `_compute_corridor_reservations` L314 | 以航点 1 到 24 个走廊节点距离 <600 m 判定“使用该走廊”，同一走廊内相邻用户两两生成预约 (a,b,t,t+120)。每个 epoch 重算。 |
| 数据集采样 | `generate_dataset` L335 | 每 10 个 epoch 取 1 个快照（1 Hz）。每快照采样 `n·4` 对：全部正样本 + 随机负样本（上限 `max(n·4−n_pos, 10·n_pos)`）。**测试集同样是采样对**，不是全对。 |

### 2.2 TKG 构建（`skyflow/data/tkg_builder.py`，对应论文 Algorithm 1）

- 词表 `RELATION_VOCAB`：0 approaches、1 conflicts_with、2 shares_corridor、3 is_downwind_of、4 has_reserved、5 is_restricted_by。
- **`has_reserved`（UAV→sector）从未被构建**——没有对应的 `_add_*` 方法。实际被填充的只有 5 类。
- `approaches`（L243）：全对 O(N²) Python 循环；若 closing 则用线性外推 CPA（前视 60 s，**不是论文/配置的 30 s**），阈值 80 m 水平 / 15 m 垂直。
- `conflicts_with`（L274）：在 approaches 判定内，`cpa_h < 0.3·80 = 24 m` 时额外连边。**纯几何规则，不读真值标签**，但它是“更紧的 CPA 阈值”，与标签高度相关 → 审稿人质疑合理。
- δ 的计算（所有关系统一）：`δ = t − last_edge_time[(i,j)]`，首次出现时 δ=0；由于 1 Hz 采样，持续存在的边 δ≡1.0 s。探针实测：δ 均值 0.89、P95 1.00（N=100 与 N=500 相同）。**δ 不携带任何通信延迟信息**，仅表示“这条边上次被看到是多久前”。
- 环境边：weather `d<400 m`（N×36 循环），restricted `d<1.5·R`（N×12 循环），均为 Python 双循环。
- 探针实测建图耗时：N=100 → 24 ms/快照；**N=500 → 526 ms/快照**（论文称 <12 ms）。

### 2.3 TR-GAT（`skyflow/models/tr_gat.py`）

- `TRGATLayer.forward`：按关系 r 分别做 `W_Q[r], W_K[r], W_V[r]`（每关系独立 Linear），注意力输入 `[q ‖ k ‖ φ(δ)]` 与 `attn_vectors[r]` 点积 → LeakyReLU → scatter-softmax（按 dst）→ 聚合；`gate_proj: Linear(out_dim·R → R)` + softmax 融合；`LayerNorm(residual + dropout(fused))`。与 Eq.3–5 一致（是 post-norm，不是论文写的 pre-norm）。
- `TRGAT`：`input_proj: Linear(23→128)` → 4 层 → `GRUCell(128→64)`。`num_relations` 决定 Q/K/V 与 gate 的维度，改 5 类时**自动适配**（只需传参）。
- 参数量实测：TR-GAT 1,248,664 + 打分头 102,929 = **1,351,593 ≈ 1.35M**。论文写 4.2M，**不一致**。
- `ConflictScoringHead`（`conflict_head.py`）：输入 `[h_i‖h_j‖s_i‖s_j‖e_ij]`，`e_ij` 维度 16；**训练与评估中从未传入 `edge_feat`**，始终用 `no_edge_token` → 论文的 e_ij 实际上不存在。

### 2.4 基线（`skyflow/baselines/`）与各自输入

| 基线 | 文件 | 输入特征 | 是否用到泄漏特征 |
|---|---|---|---|
| VO | `velocity_obstacle.py` | 节点特征列 0:3 位置、3:6 速度 | 否（只用运动学），但其本身就是 CPA 规则 |
| LSTM-P | `lstm_pair.py` | **全部 23 维**（`node_features[:n_uav]`，历史长度 T=1，即没有历史） | **是**（含 d_min/t_cpa/f_avoid） |
| Tfm-P | `transformer_pair.py` | 同上 23 维，T=1 | **是** |
| STGCN | `stgcn.py` | 23 维 + 合并所有关系的 UAV–UAV 边（含 conflicts_with） | **是** |
| GAT-S | `gat_static.py` | 23 维 + 合并所有关系边（含 conflicts_with 与环境节点） | **是** |
| TR-GAT-NT | `reproduce_paper.py` L123 `temporal_dim=2` | 同 TR-GAT；“NT”实现为把 φ(δ) 维度从 32 降到 2，**并未真正移除** | **是** |

> S2 结论预告：LSTM-P、Tfm-P 都直接读 23 维特征 → **必须重训**，不能 `--reuse`。另外 LSTM-P/Tfm-P 的 `history` 始终为 None，序列长度 1，它们事实上是 MLP。

### 2.5 训练 / 评估入口与指标

- 入口：`scripts/train.py`（单/多 seed）、`scripts/reproduce_paper.py`（一键全流程）、`scripts/run_baselines.py`、`scripts/evaluate.py`、`scripts/eval_scalability.py`。
- 配置：`configs/default.yaml` → `SkyFlowConfig`。`training.batch_size=32` **未被使用**（每个 K=10 窗口即一次优化步）。
- 训练循环 `trainer.py::train`：AdamW + warmup(1000) + cosine；窗口内 10 个快照共享 GRU 状态，loss 取窗口均值后一次 backward；**验证只在 epoch 1 和每 10 个 epoch 做一次**，按 val F1 保存 `output_dir/best_model.pt`（**多 seed 时同一文件被覆盖**）。没有 early stopping。
- 指标 `metrics.py`：`ConflictMetrics`（CDR=recall、FAR=FP/(TP+FP)、F1、P95 延迟）、`RegimeMetrics`（**定义了但全仓库没有调用**，论文 Table 4 per-regime 数字无生成代码）、`bonferroni_ttest`、`LatencyTimer`（有 cuda.synchronize）。
- 计时：`trainer.evaluate` 只计 forward+head；`eval_scalability.py` 分 graph/forward 两段；均**没有 pair_scoring 单独一段**，也没有 1000 epoch 的 P95（scalability 默认 `n_epochs=1000` 但受场景数限制）。
- seed：`torch.manual_seed` + `np.random.seed`。**数据集生成用 `hash((split, idx))`**（`urbanair500.py` L349）——Python 对 str 的 hash 每次进程启动随机（PYTHONHASHSEED），因此**数据集跨进程不可复现**，这是必须在 S7a 缓存前修掉的 bug。

---

## 3. UAV 23 维特征的实际顺序（`tkg_builder.py::_build_node_features`）

| idx | 名称 | 来源 | 备注 |
|---|---|---|---|
| 0–2 | x,y,z | `uav_positions` | 真值位置（无观测噪声） |
| 3–5 | vx,vy,vz | `uav_velocities` | 真值 |
| 6 | ψ | `uav_headings` | |
| 7 | ψ̇ | `(ψ−ψ_prev)/dt` | 未做角度 wrap，跨 ±π 会跳变 |
| 8–10 | ax,ay,az | `(v−v_prev)/dt` | |
| 11 | b | battery | |
| 12 | ḃ | 随机放电率 | |
| 13 | p | priority 0–3 | |
| 14 | c_id | 0/1：本 epoch 是否在任何走廊预约里 | 不是走廊编号 |
| 15–17 | wx,wy,wz | 全局风 + N(0,0.3) | |
| 18 | σ_gps | 2.5 + Exp(0.5) | |
| 19 | **n_nbr** | 水平距离 <80 m 的邻居数 | 由全对距离矩阵算出 |
| 20 | **d_min** | 到最近邻的**水平**距离 | **与标签直接相关**：标签就是 d_h<10 ∧ d_v<3 |
| 21 | **t_cpa** | 与最近邻的线性 CPA 时间 | |
| 22 | **f_avoid** | `uav_avoiding` | 恒为 0 |

- 环境节点（sector 8 维、weather 12 维、restricted 8 维）填入同一 23 维矩阵的前几列，其余补零，与 UAV 列语义不同但共用 `input_proj`。
- 由于当前标签定义是“此刻 d_h<10 m”，而 `d_min` 就是“此刻到最近邻的 d_h”，**d_min 对正样本几乎是标签本身**（唯一差别是垂直约束与“最近邻”未必是冲突对象）。

## 4. `conflicts_with` 的 "conflict score" 来自哪里

来自**规则**：`tkg_builder.py` L274，`cpa_h < 0.3·approach_cpa_h (=24 m)` 且已满足 approaches 条件。不是真值、不是模型输出。论文 §3.1 写的 “active conflict score ≥ τ” 与实现不符（实现里没有 τ 参与）。

---

## 5. 代码与论文不一致清单

| # | 论文说法 | 代码实际 | 严重度 |
|---|---|---|---|
| 1 | 标签由 6-DoF 轨迹积分、30 s 前视、10 m/3 m 判定 | 当前时刻 10 m/3 m 判定，无前视、无积分 | **致命**：任务从“预测 30 s 内冲突”变为“检测此刻已失去间隔”；与 PLAN 规则 2 的前提矛盾 |
| 2 | ADS-B 延迟 0.5–1.2 s、丢包 | 完全未注入；观测=真值 | **致命**：鲁棒性实验(S6/S7b)无从谈起，需要新实现 |
| 3 | 3.1% 正样本、1.47M 训练对 | 默认 5 km 场景 N=500 每快照 **0.00** 个真值冲突；rebuttal 实验把网格缩到 500 m×500 m、高度 74–78 m 才得到 ~4% | **致命**：默认配置训不出任何东西（`outputs/` 的 quick 结果 F1≈0.002 即此） |
| 4 | Table 3–8 全部数字 | 仓库无任何对应产物；`outputs/all_results.json` 是 50 UAV quick run，基线全 0 | **致命**：改投论文的所有数字都必须重新产生 |
| 5 | 6 类关系 | `has_reserved` 从未构建，实际 5 类；去掉 conflicts_with 后实际 4 类 | 高 |
| 6 | 参数量 4.2M，基线匹配 ±5% | TR-GAT+head 1.35M；GAT-S/STGCN/LSTM-P 各自独立设定 | 高 |
| 7 | 建图 <12 ms；O(N²) 只是“略超线性” | N=500 实测 526 ms/快照（纯 Python） | 高（S4 要解决） |
| 8 | e_ij 直接邻近边特征 | 从未传入，恒为 no_edge_token | 中（S2 要补） |
| 9 | 风场“历史测量的随机场” | 单一全局白噪声风矢量 | 中（写作时如实描述） |
| 10 | 前视窗口 Δ=30 s | approaches 边用 60 s，VO 用 60 s | 中 |
| 11 | pre-norm | post-norm | 低（改论文措辞） |
| 12 | per-regime (easy/hard) 结果 | `RegimeMetrics` 无调用，无 is_hard 标注 | 高（S7a 需实现 regime 标注） |
| 13 | 95th 延迟 1000 epoch | 评估阶段只计 forward；无 pair_scoring 段 | 中（S4/S7b） |
| 14 | 5 seeds | 多 seed 共用一个 `best_model.pt` 文件被覆盖 | 中（S7a 改输出目录） |
| 15 | 数据可复现 | `hash(str)` 随机 → 数据集不可复现 | 高（S7a 修） |
| 16 | TR-GAT-NT = 去掉 φ(δ) | 实为 `temporal_dim=2` | 中（S7a 改为真正移除） |

---

## 6. 运行环境、数据与旧日志

- 本机：Windows 10/11，Python **3.14.3**，PyTorch **2.14.0+cu130**（CUDA 13.0 可用），NumPy 2.4.2，pytest 9.0.2；GPU **RTX 4090 24 GB**，驱动 591.86。
- 仓库声明：Python ≥3.10、torch ≥2.2（README 称在 A100 / CUDA 12.1 上测试过）。`requirements.txt` 无版本上限，本机已满足。
- 测试命令：`cd SkyFlowCode; python -m pytest -q` → **23 passed, 1.8 s**。
- 数据集：**不落盘**，每次运行时现场生成（`generate_dataset`）。没有缓存目录，没有数据文件。
- 旧训练日志：**没有**。`outputs/` 只有 `--quick` 一次运行的 JSON 与两个 `best_model.pt`（5.4 MB，对应 50 UAV）。无法读出“验证 F1 在第几个 epoch 后不再提升”。
- 唯一有参考价值的真实训练记录：`supplementary_experiments/leakage_ablation_results.json`（200 UAV、500 m 网格、80 epoch、3 seed、每 5 epoch 验证）。其脚本注释写明“在本地 RTX 4090 上跑”；结果 TR-GAT-full CDR 0.993 / F1 0.846 / FAR 0.262；去 CPA 特征后 CDR 0.999 / F1 0.816 / FAR 0.306（F1 配对 p=0.49）。**没有保存 per-epoch 验证曲线**，收敛点未知。

## 7. 耗时估算

依据：探针实测（10 s 场景）+ 代码结构。所有数字为本机 CPU 现场生成，无 GPU 训练实测。

| 项目 | 依据 | 估算 |
|---|---|---|
| 仿真 1 个 epoch（N=500） | 实测 115 ms（其中 O(N²) 真值循环占大头） | 60 s 场景 = 600 epoch ≈ **69 s/场景** |
| 建图 1 个快照（N=500，现有 O(N²) Python） | 实测 526 ms | 60 快照/场景 ≈ **32 s/场景** |
| 生成正式数据（20+5+5 场景 × 60 s） | 上两行 | ≈ 30 × 100 s ≈ **50 min**（每次运行都要重做，S7a 缓存后一次性） |
| 单 epoch 训练（1200 训练快照 = 120 窗口，N=500，4 层，GPU） | 无实测；quick run 无日志 | **未知**，S8 实测 |
| 完整训练一次（150 epoch） | 无依据 | **未知**（论文称 A100 11 h，但该数字无产物支撑） |
| 一次测试集评估（300 快照） | 无实测 | **未知**，S8 实测 |
| 旧 N=500 收敛 epoch | 无日志 | **未知**；只能参考 rebuttal 实验的 80 epoch 设置 |

---

## 8. 待决策（S2 之前必须由你确认）

PLAN 的规则 2 假设仓库已实现“6-DoF 积分 + 30 s 前视”标签；**实际没有**。如果保持现状（“此刻失去间隔”标签），那么：
- 去掉 d_min 后模型仍可从 approaches 边的 Δp 直接算出距离，任务是平凡的几何判断，CPA-Rule（S5）会接近 100%，论文失去意义；
- S5 的 CPA-Rule（30 s 窗口）和 S6 的“冲突成因”都与标签定义不匹配。

建议方案（需你明确说“同意”才会实施，因为涉及规则 2）：
- **A（推荐）**：在仿真器中新增 `labels.mode: "lookahead"（默认）| "instantaneous"（旧）`。lookahead 模式下，用仿真器自身的航点跟踪动力学（无风、无噪声、按当前航点计划）向前积分 30 s，若任一时刻 d_h<10 ∧ d_v<3 则标记为正样本；同时记录 `time_to_conflict`，供 easy/hard 分档。这就是论文声称的标签定义。
- **B**：保持现状，论文改口为“loss-of-separation detection”，并删除所有“anticipatory / 30 s look-ahead”表述。不推荐。

同样需要确认的两项（属 S6 范围，但影响 S2 设计）：
- 正样本稀疏：默认 5 km 网格下几乎没有冲突。是否接受把训练/评测场景改为更密集的配置（例如 rebuttal 用的 500 m 网格 + 窄高度带，或 5 km 网格 + 走廊汇聚航点），并在论文 Setup 中如实写明？
- ADS-B 延迟/丢包需从零实现（观测状态 = 真值状态延迟 L 帧 + 丢包冻结），这是 S3/S6 的前提，工作量比 PLAN 预估大。

## 9. 其他需要知道的事实

- `docs/SKYFLOW_PLAN.md` 已按 PLAN §0 复制进仓库，但其中含作者邮箱与投稿策略；**公开推送前请确认是否保留**（S19 会再提醒）。
- `.gitignore` 忽略 `outputs/` 与 `*.pt`，PLAN 要求的 `results/` 目录尚不存在，未被忽略；后续需决定 results 是否入库（建议入库 CSV/JSON，忽略 checkpoint）。
- `scripts/eval_sdd_transfer.py`、`skyflow/data/sdd_adapter.py`、`skyflow/models/resolution.py` 本次改投不再使用，但按规则 5 不删除。
