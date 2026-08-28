# TuFT Framework Mismatch / Corrector 研究档案

> **文档定位**：写论文期间的唯一事实来源（single source of truth）。收录问题定义、机理推导、全部实验与真实数字、方法演进过程（含负面结果）、代码与数据路径、以及已知的内部不一致与待补实验。
> **论文摆位**：TuFT 系统论文（NSDI 方向）的 **Corrector 章节**，定位为 sampling 侧的 reliability layer，约占 1/4 篇幅，与 scheduling 共同构成系统贡献。
> **整理日期**：2026-08-07
> **事实来源**：`/Users/yanghanzhang/work/research/` 下的代码、数据、checkpoint，以及 2026-05-14 至 2026-07-01 的研究日志。所有数字均从源文件核出，凡是无法核实的一律在第 11 节标注。

---

## 0. 符号与约定（先统一，否则后面全乱）

| 符号 | 定义 | 备注 |
|---|---|---|
| `s_t` | sampling 引擎（vLLM）在采样时记录的 logprob | 必须是 **post-temperature** |
| `p_t` | training 引擎（HF/FSDP/Tinker）forward 重算的 logprob | 必须用同一个 temperature |
| **`Δ_t = s_t − p_t`** | **代码与本文档统一采用的定义**（sampling 减 training） | ⚠️ 见下方警告 |
| `c` | per-token systematic bias，即 `E[Δ_t]` | BF16 约 0.002–0.014，任务相关 |
| `log w_seq` | 序列级 log IS weight = `Σ_t Δ_t`（或其反号，取决于 IS 方向） | 核心受害量 |
| `T` | response 长度（token 数） | |
| `σ_seq` | `Σ_t Δ_t` 在序列间的标准差 | 实测 0.25–0.42 |
| `τ` | clip 阈值的 log 形式 | ε=0.2 → `log 1.2 = 0.1823` |
| `clip02` | `P(|Σ_t Δ_t| 超出 [log 0.8, log 1.2])` | **序列级**违反率，诊断量 |
| `clip01` | 同上，阈值 `[log 0.9, log 1.1]` | |

> ⚠️ **符号警告（必须在论文里统一一次）**
> 代码库（`verify_mismatch.py`、`predictor/README.md`、`losses.py`）全部用 `Δ_t = s_t − p_t`。
> 而之前若干轮讨论中用的是 `δ_t = training − sampling`，**符号相反**。
> 本文档统一用代码的定义。写论文时任选其一，但要与图、表、公式全部对齐。
>
> 另一个易错点：`losses.py` 里 clip 阈值是 **非对称**的 —— `(log 1.1, log 0.9) = (+0.0953, −0.1054)`，不是 `±log(1.1)`。clip02 同理是 `(+0.1823, −0.2231)`。

---

## 1. 问题定义

### 1.1 两条 logprob 路径

在 disaggregated RL fine-tuning 里，每个 response token 有两个 logprob：

- `s_t`：vLLM rollout 时顺手记录的采样 logprob；
- `p_t`：训练引擎对同一条序列做一次 full-sequence forward 得到的 logprob。

理论上两者应该相等（同一份 weight、同一个 token、同一个前缀）。实际上不等，来源是**两个引擎的实现路径不同**：

- kernel 实现不同（PagedAttention vs 标准 attention）；
- 数值精度不同（vLLM 可能 FP8 权重 / 不同累加顺序，training 侧 BF16/FP32）；
- batching 与 padding 位置不同（prefill/decode 流水 vs full-sequence teacher forcing）；
- CUDA graph、chunked prefill 等图级优化。

这不是 bug，是 disaggregation 的**结构性代价**：为了拿到 vLLM 的吞吐，就必须接受两个引擎的数值不一致。

### 1.2 为什么这会伤害训练

RL 的 importance sampling ratio 是

```
w_t = π_train(a_t | s_t) / π_behavior(a_t | s_t) = exp(p_t − s_t) = exp(−Δ_t)
```

分母本应是"真实采样分布"。当 `s_t` 带有系统性偏差时，IS ratio 里混入了一个与 reward、与策略距离**都无关**的伪权重。序列级尤其致命，因为 IS 是乘性的：

```
w_seq = exp(Σ_t Δ_t) ≈ exp(T·c + noise)
```

**per-token 看起来无害的 c，按序列长度线性累积后变成序列级的巨大偏移。**

### 1.3 关键区分：framework mismatch vs staleness

两者都会让 `π_behavior ≠ π_train`，但性质完全不同：

| | framework mismatch | staleness |
|---|---|---|
| 成因 | 两个引擎的数值实现差异 | 采样用的是旧版本 weight |
| 是否"真实策略距离" | **否**，是 artifact | **是**，是真实的 off-policy 距离 |
| 是否该被 IS 修正 | 不该 —— 应当消除 | 该 —— 这正是 IS 的用途 |
| sync on-policy 下 | 仍然存在 | 为 0 |

**这个区分是整个 Corrector 章节的立论基础**：TuFT 要做的不是"消除 IS weight"，而是**从 IS weight 里剥离掉 framework 成分、保留 staleness 成分**。

一个用户在讨论中提出的关键论断（正确，且应写进论文）：

> **在严格 sync on-policy 下，vLLM 与 HF 之间的"策略差距" 100% 等于 framework mismatch**，因为两者用的是同一份 weight，不存在版本差。此时 `Δ_t` 是纯 artifact。

这给了一个非常干净的数据采集协议：**只在 `sync_mode: true`、`staleness == 0` 的 rollout 上采集 `(s_t, p_t)` 对**，采到的就是纯 framework mismatch，不混 staleness。代码里 `filter_clean_records` 正是这么做的。

---

## 2. 机理：per-token 微小偏差如何变成 sequence-level 灾难

这一节是 Motivation 的核心，全部结论都有实验支撑。

### 2.1 长度累积（length bias）

设 per-token bias 为 `c`，则序列级 log IS 的期望是 `T·c`。用实测的 BF16 bias `c = 0.00229`：

| T | `T·c` | 伪权重 `exp(T·c)` |
|---|---|---|
| 200 | 0.458 | **1.58×** |
| 500 | 1.145 | **3.14×** |

FP8 下 bias 放大到 `c = 0.00519`（2.3×）：

| T | `T·c` | 伪权重 |
|---|---|---|
| 200 | 1.038 | **2.82×** |
| 500 | 2.595 | **13.4×** |

**含义**：一条长 response 相比短 response，仅仅因为长，就被赋予 3× 甚至 13× 的梯度权重，而这与它的 reward 好坏毫无关系。这是一种**对长序列的系统性歧视（或偏袒，取决于符号）**。

在真实数据上更夸张。`bench_merged_r8_t07.jsonl`（1280 条 GSM8K 序列）里，`Σ_t Δ_t` 的最大值达到 **75.49**，对应 IS weight `exp(75.49) ≈ 6.1×10³²`。predictor README 里也记录了 `exp(Σ_t Δ_t) ≈ 10^14` 量级的观测。

### 2.2 对不同 RL 算法的影响路径

这是导师要求"从公式角度推导实际意义"时整理出来的三段式，可直接进论文：

| 算法 | 受影响路径 | 严重程度 |
|---|---|---|
| **REINFORCE / RLOO** | 序列级 IS 直接乘 `exp(T·c)`，产生与 reward 无关的长度伪权重 | **最严重** |
| **off-policy GRPO** | 序列级 IS 被污染 → 把 fresh 样本误判为 stale（或反之），trust region 判据失真 | **严重** |
| **on-policy GRPO / PPO（token-level clip）** | 单 token 的 `c ≈ 0.002` 被 `ε = 0.1~0.2` 的容错空间吞掉，影响 < 0.2% | 直接影响可忽略 |
| **PPO 的 KL 早停** | total KL = `Σ_t` 被 `T·c` 系统性高估 → 长序列过早触发早停 | **间接但真实** |
| **clip 方向对称性** | `c` 恒为正 → upward clip 概率 > downward clip 概率，clip 变成单侧 | **公平性问题** |

**结论**：不能笼统说"mismatch 影响所有 RL"。要精确说：**它影响任何依赖 sequence-level importance weight 的算法**，而 token-level clip 的主流实现受影响很小。这个精确 scope 比夸大 claim 更能扛 review。

### 2.3 为什么 clip 率降不下来（数学必然，不是 bug）

这是一个反复被误解的点，必须在论文里主动交代。

`clip 率 = P(|Σ_t Δ_t| > τ)`。实测：

- `σ_seq`（序列间标准差）= **0.25 ~ 0.42**
- `τ`（clip 阈值）= **0.10 ~ 0.22**

当 `σ_seq ≫ τ` 时，**对整个分布做 centering（减去均值）不改变 `P(|x| > τ)` 的总量**，只改变它的构成（对称性）。

`predictor/README.md` §13.3 给了同样的推导：

```
Var(Σ_t residual_t) = T · Var(residual_t) ≈ 600 · 0.025² ≈ 0.375
std ≈ 0.61  →  exp(±0.61) = [0.54, 1.84]
```

即使 bias 被完全消除，仍有 60%+ 的序列天然落在 `[0.9, 1.1]` 之外。

**因此 bias correction 的价值定位必须是 correctness，不是 efficiency：**

> 它改变的是**哪些序列被 clip**（消除长度歧视、恢复 clip 的双侧对称性），而不是**多少序列被 clip**。

要真正降低 clip 率，必须降低 per-token residual 的**方差**，而这（见第 4 节）被证明不可学。

### 2.4 量化（FP8）放大问题 —— 趋势论据

| 配置 | per-token bias | 相对 BF16 |
|---|---|---|
| BF16 | 0.00229 | 1.0× |
| FP8 | 0.00519 | **2.3×** |

更重要的是，FP8 下的偏差**不是纯常数**：

- FP8 mlp 层：`med|log_is|` 改善 **16.9%**（纯常数偏移不会改善这个指标）
- FP8 transformer：`med|log_is|` 改善 **11%**

这说明 FP8 引入了 **layer-dependent 的量化偏差结构**，是可学的信号，而不只是一个全局标量。

量化的极端后果见 §3.3 的 severity 表：**Self-hosted + Quant 的 p99 IS weight 达到 7.25×10⁷**。

**论文叙事价值**：推理侧量化是不可逆的行业趋势（FP8/INT8 已是 serving 标配）。mismatch 问题只会越来越严重，而 Corrector 的成本是零。这是一个很强的"面向未来"论据。

---

## 3. 测量与刻画（Motivation 实验）

### 3.1 实验一：跨引擎 mismatch 微基准

**脚本**：`code/experiments/TuFT/experiments/01_mismatch_motivation/verify_mismatch.py`（481 行）

**协议**：固定 prompt、从初始权重采样一次 response 后**全程固定**，此后每一轮只做 (a) 两条路径重算 logprob、(b) 一步 IS 训练。这样把 framework mismatch 与数据分布漂移彻底隔离。

**配置**（从 argparse 默认 + README + 数据几何反推，三方交叉验证）：

| 参数 | 值 |
|---|---|
| base model | Qwen3-4B |
| LoRA rank | 16 |
| temperature | 0.7 |
| precision | BF16 |
| rounds | 30（→ 31 条记录，round 0–30） |
| prompts × samples | 5 × 4 = **20 序列/轮** |
| max_tokens | 64（1280 tokens / 20 seqs） |
| learning rate | 5e-5 |
| seed | 42 |

**注**：数据集是脚本内置的 5 条固定 QA prompt（"What is the capital of France?" 等），**不是公开 benchmark**。论文里要如实说明这是 microbenchmark。

**产出数据**：
- `data/microbench/01_framework_mismatch/tinker_mismatch_results.json`（31 条）
- `data/microbench/01_framework_mismatch/tuft_mismatch_results.json`（31 条）
- 两份文件在 `paper_writing/experiments/04_mismatch_moti/data/` 和 `code/.../01_mismatch_motivation/{tinker,tuft}_results/` 下有 **byte-identical 副本**（md5 已核）

**每条记录 18 个字段**：`mean_diff, mean_abs_diff, max_abs_diff, std_diff, num_tokens, mean_cum_diff, max_cum_diff, std_cum_diff, mean_sampling_logprob, mean_training_logprob, mean_is_weight, std_is_weight, min_is_weight, max_is_weight, p99_is_weight, p_out_clip_02, p_out_clip_01, round`

### 3.2 主结果表（31 轮聚合，真实数字）

| 指标 | **Tinker**（商业 disagg 服务） | **Self-hosted**（本地 vLLM+FSDP） |
|---|---|---|
| `mean_abs_diff` 均值 | **0.015618** | **0.011334** |
| `mean_abs_diff` 范围 | 0.011175 – 0.021589 | 0.009261 – 0.015086 |
| `max_abs_diff` 均值 / 峰值 | 0.770137 / **1.549732** | 0.453810 / 0.870706 |
| `std_diff` 均值 | 0.055071 | 0.036012 |
| `max_cum_diff` 均值 / 峰值 | 1.215252 / **2.309009** | 0.783809 / 1.375944 |
| `std_cum_diff` 均值 | 0.432868 | 0.281882 |
| `mean_is_weight` | 1.094967 | 1.125846 |
| `std_is_weight` 均值 / 峰值 | 0.467651 / **1.275122**（round 18） | 0.355486 / 0.706369 |
| `min_is_weight` 最小 | **0.099360** | 0.459338 |
| `max_is_weight` 最大 | **6.644358** | 3.958812 |
| `p_out_clip_02` 均值 | **52.42%** | **40.65%** |
| `p_out_clip_01` 均值 | **71.94%** | 67.90% |

**可写进论文的三句话**：
1. 即使在同一份权重上，两个引擎的 per-token logprob 平均相差 1.1%–1.6%，最大单 token 差异超过 1.5 nats。
2. 商业 disaggregated 服务（Tinker）的 mismatch 比自建部署**更严重**（52.4% vs 40.7% 的序列超出 ε=0.2 clip 带），而且**不透明、不可控**——这直接支撑 TuFT 的 "reproducibility gap" 立论。
3. IS weight 的方差会间歇性爆发（Tinker round 18 的 `std_is_weight` 从 ~0.4 飙到 1.275，`max_is_weight` 达 6.64），不是均匀恶化。

### 3.3 Severity 三方对比（含量化臂）—— 最强的单张表

**脚本**：`paper_writing/experiments/04_mismatch_moti/code/plot_mismatch_severity.py`
**产出**：`paper_writing/experiments/04_mismatch_moti/figures/mismatch_severity_summary.csv`
**口径**：只取三个 series 的 **9 个公共 round（1–9）**

| series | mean `mean_abs_diff` | mean `max_abs_diff` | median `p99_is_weight` | mean `clip01` |
|---|---|---|---|---|
| Tinker | 0.012598 | 0.370899 | 1.5956 | 0.6778 |
| Self-hosted | 0.010619 | 0.276847 | 1.3873 | 0.6444 |
| **Self-hosted + Quant** | **0.026780** | **17.448** | **7.2474×10⁷** | **0.8671** |

量化把 `max_abs_diff` 推高 **63×**，把 p99 IS weight 推高 **7 个数量级**。这一行是整个 Motivation 里冲击力最强的证据。

配套图（同一脚本产出，figsize 4.2×2.35，serif，`pdf.fonttype 42`，配色 `#DBE2EF / #3F72AF / #112D4E`）：
- `mismatch_severity_comparison.{pdf,png}` — y 轴 Mean |Δ logprob|
- `mismatch_sequence_cumulative_comparison.{pdf,png}` — y 轴 |Mean cumulative Δ logprob|

### 3.4 任务异质性（论证"全局常数不够用"）

**数据**：`05_rlhf_bench/simulator/logprobs/multitask_logprobs.jsonl`，7680 序列 = 6 任务 × 1280，rank {8,16}，staleness 0。

| task | mean Δ | token MAE | cum mean | cum std | clip02 | 平均长度 | temp |
|---|---|---|---|---|---|---|---|
| countdown | **0.013889** | 0.06337 | 2.8266 | 5.7804 | 0.9234 | 203.5 | 0.9 |
| math | 0.011341 | 0.02851 | 4.9161 | **21.6452** | 0.9141 | 433.5 | 0.7 |
| ifeval（holdout） | 0.008394 | 0.04337 | **4.2855** | 11.0601 | 0.9289 | **510.6** | 0.7 |
| mbpp | 0.006130 | 0.02907 | 1.5081 | 3.3822 | 0.8906 | 246.0 | 0.7 |
| gsm8k | 0.004078 | 0.02454 | 0.8681 | 1.7756 | 0.8805 | 212.8 | 0.7 |
| humaneval | **0.002753** | 0.01686 | 0.6313 | 2.0835 | **0.8086** | 229.3 | 0.7 |

**跨任务的 mean Δ 相差 5 倍**（0.002753 → 0.013889）。5 个训练任务 pooled 后 mean Δ = **0.008113**，与 launch script 里硬编码的 `GLOBAL_MEAN_BIAS=0.008113` 精确吻合（已独立复算验证）。

**这张表是"为什么需要 learned predictor 而不是一个全局常数"的直接量化依据。**

### 3.5 Staleness 量化（作为对照，说明两种偏差量级可比）

来源：`docs/TuFT_Research_Proposal_v4_EN.md` §4.2.2，6400 条固定 prompt、20 个 gradient step。

| 指标 | Adjacent (v_n → v_{n+1}) | Cumulative (v_0 → v_n) |
|---|---|---|
| IS weight 范围 | [0.3402, 1.1875] | [0.0108, 1.5640] |
| μ(w) | 0.983 | 0.868 |
| P(\|w−1\| > 0.2) | **1.31%**（MODERATE） | **26.78%**（SIGNIFICANT） |

**第 8 步后出现相变**：σ(w) 从 0.024 跃升到 0.263，`w_max > 8`。

**对照价值**：单步 staleness（1.31%）远小于 framework mismatch（40.65%）。也就是说，**在 multi-step async 场景下，IS weight 里的 framework 成分很可能压过 staleness 成分**——这正是需要剥离的理由。这个对比应该在 Motivation 里显式给出。

---

## 4. 方法演进史（含全部负面结果）

这一节记录了走过的弯路。**强烈建议在论文里保留一个精简版**，因为负面结果本身构成了"为什么最终设计是这样"的论证链条，比直接给出结论更有说服力。

### 4.1 v0：早期 predictor "有效"，实为 temperature bug

最初版本的 ML4Sys predictor 显示出很好的可学性。**根因排查后发现是 bug**：training 侧 forward 时**漏传 temperature**，导致 `Δ_t` 里混入了一个确定性的温度变换差。那当然是可学的——但学的是 bug，不是 mismatch。

**修复方式**（已固化进代码）：
- sampling 侧必须返回 **post-temperature** logprob（IS 的分母是实际采样分布）；
- training 侧通过 `loss_fn_config["temperature"]` 把同一个 T 传下去，让 backend 在 `log_softmax` 前做 `logits /= temperature`；
- rollout 时**关闭 top-p / top-k**（否则采样分布不是 softmax，logprob 无意义）。

代码里保留了明确注释：

```python
# Pass temperature via loss_fn_config so the training backend applies
# logits /= temperature before computing log_softmax.  This aligns with
# the sampling-side logprobs which are computed after temperature scaling.
```

> **这是一条值得写进论文的工程教训**：跨引擎 logprob 对齐的第一要务是确认两侧的 temperature/采样截断完全一致，否则测到的"mismatch"是假的。

### 4.2 v1：修正后 per-token 完全不可学

扩大数据集重测（465K tokens / 2240 序列）：

- val **R² = −0.0012**，test **R² = 0.0000**（后续更大数据集 R² = 0.0035）
- test token MAE = 0.013（1.3%），修正后反而更差 **−1.1%**
- 唯一改善的是 bias / mean_log_is（+70%），但 `med|log_is|` 和 clip 率**纹丝不动**

**结论**：模型只学会了一个 scalar offset。`self-normalized IS`、`running mean subtraction`、`per-batch centering` 一行代码即可等效。

**当时的判断**：ML4Sys predictor 作为独立 contribution 死亡，mismatch 降级为 problem statement。

### 4.3 v2：三条独立证据证明 per-token 不可学

**脚本**：`03_bias_predictor/per_token_impossibility.py`
**结果**：`05_rlhf_bench/simulator/results/per_token_impossibility/results.json`
**数据**：`tp2_fsdp1_id0_noquant.jsonl` 的 test split，2240 序列 / 465,637 tokens

**证据 1 — ANOVA 方差分解**

```json
"total_var":   0.012090466147618058,
"var_between": 0.000096683083031400,   // 0.7997%
"var_within":  0.011993783064586656,   // 99.2003%
```

分解是精确的（`9.668e-5 + 0.0119938 = 0.0120905`）。

> ⚠️ **重要限定（必须写清楚，否则自相矛盾）**：这个 0.80% 的上界约束的是「**序列内输出恒定**的预测器」，**不是任意 per-token 预测器**。它不与后面 per-token 模型拿到 ΔR² = 0.23–0.52 冲突，因为后者用了 token identity 和 `sampling_lp` 的逐位置信息。论文里必须显式调和这两个数字，否则 reviewer 一定会打。

**证据 2 — 自相关**

```json
"mean_acf_lag1":     -0.006659821236407658,
"max_abs_mean_acf":   0.028286435356905150,
"ci_mean":            0.009508962812535503,
"is_white_noise":     false
```

`is_white_noise` 为 false 是因为判据是 `max_abs < 2 × ci_mean = 0.019018`，而实测 0.028286 略微超过。**但绝对量级极小**（|ACF| ≤ 0.028 ⇒ 每个 lag 解释 ≤ 0.08% 的方差）。正确表述是「存在统计上可检出但量级可忽略的时序结构」，**不要写成"是白噪声"**——脚本自己都判 false。

**证据 3 — IS weight 重建**

```json
"is_weight_error_raw":              0.7388354833286007,
"is_weight_error_per_seq_oracle":   0.0,
"is_weight_error_per_token_oracle": 0.0,
"resid_mae_ratio":                  1.1666755462216805
```

两个 oracle 都把 IS 误差打到 0。**per-token 精度对 IS weight 毫无额外贡献**，因为下游量是一个纯粹的求和。

更有意思的是 `resid_mae_ratio = 1.1667`：减去 per-sequence mean 会让 token 级 MAE 从 0.015820 **升高到** 0.018457（+16.7%）。

> **这是全部材料里对 token-level 与 sequence-level 目标错位最锋利的一句话**：一个在序列级完美的 oracle 修正，在 token 级是有害的。它同时证明了 (a) token MAE 是错误的优化目标，(b) 正确目标是 per-sequence bias。

### 4.4 v3：确立最终设计

综合以上，最终架构决策：

> **predictor 学的是 per-sequence 的 scalar bias，而不是 per-token 的 delta。**

实现上有两种形态，代码里两种都有：
1. **per-token 输出 → 取序列均值**（当前 RL loop 实际使用的，`PredictorCorrector.correct_batch`）
2. **直接输出 per-seq scalar**（`ScalarTransformerPredictor` / `seq_bias_predictor.py`，形态更干净，但见 §11 未跑通）

代码注释写得很清楚：

```python
"""Per-token predictor，实际学到的是 per-sequence scalar bias（within-seq 是白噪声）。"""
# 取序列均值作为 per-sequence scalar（避免 per-token 噪声）
c_hat = float(delta_hat.mean().item())
corrected_lps = [lp - c_hat for lp in samp_lps]
```

### 4.5 一个反直觉的观察：noise 是有益的

`global mean subtraction` 曾出现**反向超调**（clip ratio 打到 0.13×），说明 bias **不是纯常数，而是 weakly input-dependent**。

进一步的洞察（2026-06-27 记录）：

> **per-token noise 起到 implicit regularization 的作用**（它触发 clip）；predictor 的价值在于「**去 bias、保 noise**」，这是一个 sweet spot。

这条洞察在解释「为什么 predictor 优于把 IS 直接设成 1 的 oracle」时是关键（详见第 9 节）。

### 4.6 相关工作定位（已核查，空白确认）

系统调研结论（2026-06-03 记录）：

- **统计 IS 族**：`Diagnosing T-I Mismatch`（arXiv 2605.14220）survey 了 TIS / RS / masking，原文确认**无 learned model**
- **数值 fix 族**：`Defeating via FP16`（2510.26788）、`FP8-RL`（2601.18150）、NeurIPS'25 `Rollout-Training Mismatch`（deterministic kernel）、Slime `Truly-On-Policy`
- **最接近的 learned 邻居**：`DPE`（arXiv 2308.14897）训练辅助模型估计 behavior policy 做 sequence-generation off-policy IS —— **但不是 framework-delta predictor**
- 6 个 Tinker 系统（SkyRL / MinT / TinkerCloud-Slime / OpenTinker / twinkle / mlx-tinker）**全部用 IS/clip 处理 mismatch，无一做 ML 预测**
- `Punica`（MLSys'24）只做 multi-LoRA serving，不碰 training，因此结构性地没有 mismatch 问题

**空白是真实的，但是双刃剑**：既是新颖性，也会招来 MARLaaS 式的"ML4Sys 预测不可行"质疑。建议在 related work 里把 DPE 作为唯一的 learned-correction 邻居做区分定位。

---

## 5. Corrector（Predictor）设计

### 5.1 学习目标与零初始化

学的是**残差** `Δ_t`，不是直接学 `p_t`。两个好处（README §1.2）：

1. 信号集中：`Δ_t` 均值约 0.013，远比 `p_t`（均值 −2、方差几十）易学；
2. **零初始化等价 baseline**：输出 head 用 `nn.init.zeros_`，训练前 `Δ̂_t ≡ 0`，等价于"不做修正"，给训练一个安全起点。

第 2 点在日志里得到了**逐比特验证**：所有 epoch-0 的 `corrected_*` 与 `baseline_*` 完全相同。这是一个很漂亮的设计性质，值得在论文里点一句。

### 5.2 输入特征（完整枚举）

`data.py::LogprobMismatchDataset.__getitem__` 是全部的特征计算：

| # | 特征 | 形状 | 计算方式 |
|---|---|---|---|
| 1 | `token_ids` | `[B,T]` | 原始 `response_tokens`，截断到 2048，padding=0（被 mask） |
| 2 | `position` | 隐式 `[T]` | 模型内部 `arange(T)` → sinusoidal 编码 |
| 3 | `sampling_lps` | `[B,T]` | vLLM logprob 原值（同时也是 target 的一半） |
| 4 | `n_prompt_tokens` | `[B]` | 标量 → sinusoidal 编码 |
| 5 | `temperature` | `[B]` | 原值（观测到 0.7 / 0.9） |
| 6 | `lora_rank` | `[B]` | `rank / 64.0` → 观测到 0.125（r8）/ 0.25（r16） |
| 7 | `mask` | `[B,T]` | 有效位 |
| — | **target** | `[B,T]` | `Δ_t = sampling_lps − training_lps` |

**显式不使用**：`prompt_tokens`（只用其长度）、`task`、`tenant_id`、`reward`、`advantage`、`step`、`sample_weight_version`。

`model.py` 的 docstring 明确写了这个设计决策：

```
Design notes:
  • No task_id / tenant_id: mismatch is user-agnostic; the above continuous
    features already capture all relevant adapter-config information.
  • lora_rank is normalized (÷ 64) so the model sees a [0, 1]-range scalar;
    this avoids OOV issues and generalises to unseen rank values.
```

> **这是一次重要的架构演进**：早期（0603 代）用的是 `task_idx` / `lora_rank_idx` 的 categorical embedding；现在改成**用户无关 + 连续特征**。这个改动的动机（泛化到未见过的 task 和 rank）应该写进论文，因为它直接对应多租户部署场景。

### 5.3 三种架构

**MLPPredictor（Tier-0），4,891,393 参数**

```python
self.token_emb = nn.Embedding(vocab_size, token_emb_dim)   # 152064 × 32 = 4,866,048
in_dim = token_emb_dim + pos_dim + pos_dim + 1 + 1 + 1      # 32+16+16+1+1+1 = 67
self.mlp = nn.Sequential(
    nn.Linear(in_dim, hidden), nn.GELU(),                   # 67→128
    nn.Linear(hidden, hidden), nn.GELU(),                   # 128→128
    nn.Linear(hidden, 1),                                   # 128→1
)
nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)
```

Embedding 占 **99.48%** 的参数。每个 token 独立处理，无上下文。

**TransformerPredictor（Tier-1），5,284,225 参数**

`d_model=128, n_heads=4, n_layers=2, ffn_mult=4 (→512), dropout=0.0, norm_first=True, max_position=4096`，每层 198,272 参数，embedding 占 92.1%。

前向是「per-token 流 + 广播的 per-sequence 条件向量」：

```python
x = self.token_proj(self.token_emb(token_ids))          # [B,T,128]
x = x + self.position_table[:T].unsqueeze(0)
x = x + self.lp_proj(sampling_lps.unsqueeze(-1))
prefix_enc = sinusoidal_encode(n_prompt_tokens, self.d_model)
seq_feat = (self.prefix_proj(prefix_enc)
            + self.temp_proj(temperature.unsqueeze(-1))
            + self.lora_proj(lora_rank.unsqueeze(-1))).unsqueeze(1)
x = x + seq_feat
x = self.encoder(x, src_key_padding_mask=~mask)
delta = self.head(x).squeeze(-1)
```

**ScalarTransformerPredictor（Tier-2）**：同 trunk，head 前做 masked mean pooling，直接输出 per-seq scalar。**从未训练**（见 §11.5 的 dispatch bug）。

### 5.4 损失函数

```
L = L_token + λ_seq · L_seq + λ_bias · L_bias
```

记 `d_bt = (Δ̂_bt − Δ_bt) · m_bt`，`N = Σ m_bt`：

| 项 | 公式 | 作用 |
|---|---|---|
| `L_token` | `(1/N) Σ_{b,t} \|d_bt\|` | token 级 L1 |
| `L_seq` | `(1/B) Σ_b \|Σ_t d_bt\| / T_b` | **长度归一化**的序列累积残差绝对值 |
| `L_bias` | `\|Σ_{b,t} d_bt\| / N` | batch 级平均残差绝对值 |

实际使用：`λ_seq = 1.0`，`λ_bias = 2.0`（代码默认是 5.0，六个 run 全部用的 2.0）。

> **设计缺陷（可作为 future work）**：`L_seq` 除以了 `T_b`，这恰好**抵消掉了 IS 爆炸真正依赖的长度加权**。既然问题的本质是「长序列累积更多 bias」，损失函数不应该把长度归一化掉。改成不除 `T_b`（或除 `sqrt(T_b)`）可能直接改善下游 clip 指标。

### 5.5 训练配置（六个 run 实际使用）

```
optimizer     AdamW, lr=3e-4, weight_decay=1e-4, grad_clip=1.0
schedule      3 epoch 线性 warmup → cosine 退火到 0（每 batch step）
batch_size    32
epochs        20
seed          0
device        cpu        ← MLP 19–36 s/epoch，Transformer 388–585 s/epoch
lambda_seq    1.0
lambda_bias   2.0
max_seq_len   2048
split         70 / 15 / 15
```

**模型选择**：`best_metric = corrected_token_mae`（scalar 模型则用 `corrected_token_bias`）。best checkpoint 会在最后被重新载入再做一次 test 评估。

> ⚠️ **这是一个明确的方法论问题**：用 `corrected_token_mae` 做模型选择，与下游 RL 目标（序列级 bias）不一致。最直接的证据是 `transformer_task_id_tenant_0603` —— 选中的 epoch 12 恰好是该 run 里 `corrected_token_bias` 最差（0.010691）、`mean_log_is` 最差（+3.54）的一个 epoch。**建议改用 `corrected_mean_log_is` 或 `corrected_token_bias` 做选择，并在论文里说明。**

### 5.6 数据划分策略（防泄漏）

`data.py` 提供两种，README 给出了不能用随机划分的理由：

> ❌ **不能用随机划分**：同一个 `sample_weight_version` 下的 token 既会出现在 train 又会出现在 test，模型可能记住"这个版本下的偏置"，test 指标虚高。

**A. `split_by_weight_version`（时间顺序）**：按 `(sample_weight_version, tenant_id, item_idx)` 排序后 70/15/15 切片。评估的是"在没见过的 weight 上"的泛化 —— 这正是 RL 推理时 predictor 面对的场景。

**B. `split_by_tenant`（跨租户）**：按 tenant 划分，评估跨租户泛化。

在 `22_agents_50_step_results.logprobs.jsonl`（36,499 条，36,153 条 clean）上：

| split | train | val | test |
|---|---|---|---|
| weight_version | 25,307 | 5,422 | 5,424 |
| tenant | 23,584 | 4,608（3 租户） | 7,961（4 租户） |

数据集特征：**22 租户、11 个 task**（apibank 6400 / toolbench 4352 / mbpp 3840 / triviaqa 3731 / humaneval 3328 / countdown 3584 / math_agent 3200 / hotpotqa 2560 / gsm8k 2240 / ifeval 1856 / math 1408）、`lora_rank ∈ {8,16}`、`temperature ∈ {0.7,0.9}`、50 个 weight version。

### 5.7 部署形态与三个 invariance

Corrector 的部署假设（moti 里 claim、body 里推导、eval 里兑现）：

| | invariance | 含义 | 决定的部署性质 |
|---|---|---|---|
| **I1** | 时间稳定 | mismatch 在一次训练过程中不随 weight 版本剧烈漂移 | 可以 **offline profile** 一次，不必在线重训 |
| **I2** | knob 可组合 | 对 rank / temperature / 长度的依赖是可参数化的 | knob 作为 **输入特征**，而非每个配置训一个模型 |
| **I3** | 跨租户 | mismatch 是 user-agnostic 的 | **per-base-model 一个 Corrector**，多租户共享 |

**部署结论**：每个 base model + 每条 sampling path（DP group）离线 profile 一个 Corrector，用少量 `sync_mode=true` 的 rollout 做 calibration probe。predictor 只有约 5M 参数，README 称 10 分钟即可训完。

**架构位置（MEMORY.md 已锁定）**：Corrector 运行在 **sampling 侧**，修正每条 rollout **返回的 logprob**（不是 token，不是 training 侧修正），使租户看到的 IS ratio `π_train/π_behavior` 是准确的。它与 scheduling **解耦**。

### 5.8 集成进 RL loop

```python
corrected_sampling_lp = sampling_lps - delta_hat
log_is_ratio = training_lps - corrected_sampling_lp.detach()   # PPO 用这个
ratio = torch.exp(log_is_ratio).clamp(1 - eps, 1 + eps)
```

**关键**：predictor 输出必须 `.detach()`，避免梯度回流到 predictor。

实际 `rlhf_bench.py` 里的四臂修正逻辑：

```python
if correction == "baseline":
    return sampling_lps_list, {}
elif correction == "oracle":
    # 用 training logprobs 直接替换 sampling logprobs，IS ratio=1，完全无偏
    return training_lps, {}
elif correction == "global_mean":
    return [[lp - global_mean_bias for lp in lps] for lps in sampling_lps_list], ...
elif correction == "predictor":
    return predictor.correct_batch(...)
```

等效 per-token 权重：`w_t = exp(p_t − (s_t − ĉ))`，其中 `ĉ` 分别为 0 / 全局常数 / predictor 的 per-seq 均值 / （oracle 时 `s_t ← p_t` 使 `w_t ≡ 1`）。

---

## 6. 离线评测结果

### 6.1 六个 checkpoint 的 test 结果（真实数字）

数据：`22_agents_50_step_results.logprobs.jsonl`

| 指标 | mlp_task_id_tenant_0603 | mlp_tenant_0604 | **mlp_weight_0604** | transformer_task_id_tenant_0603 | **transformer_tenant_0604** |
|---|---|---|---|---|---|
| split | tenant | tenant | weight_version | tenant | tenant |
| n_tokens / n_seqs | 1,500,389 / 7,961 | 同左 | 705,309 / 5,424 | 同左 | 同左 |
| baseline token MAE | 0.048474 | 0.048474 | 0.038147 | 0.048474 | 0.048474 |
| **corrected token MAE** | 0.036349 | 0.032872 | **0.026145** | 0.029206 | **0.024828** |
| baseline token bias | 0.014625 | 0.014625 | 0.012837 | 0.014625 | 0.014625 |
| **corrected token bias** | 0.005653 | 0.001011 | 0.000584 | 0.012683 | **0.000136** |
| baseline `mean_log_is` | 2.756327 | 2.756327 | 1.669271 | 2.756327 | 2.756327 |
| **corrected `mean_log_is`** | −1.065410 | +0.190530 | +0.075913 | +2.390351 | **+0.025705** |
| baseline `med\|log_is\|` | 2.017975 | 2.017975 | 0.791143 | 2.017975 | 2.017975 |
| corrected `med\|log_is\|` | 1.281166 | 0.972661 | 0.373429 | 1.583258 | **0.610109** |
| baseline clip01 / clip02 | 0.962568 / 0.935435 | 同左 | 0.857485 / 0.797198 | 同左 | 同左 |
| **corrected clip01** | 0.950760 | 0.926266 | 0.786873 | 0.961814 | **0.881296** |
| **corrected clip02** | 0.903530 | 0.857807 | 0.645649 | 0.929155 | **0.780555** |
| **ΔR²** | 0.463062 | 0.497052 | 0.233853 | 0.490778 | **0.521713** |

**最佳模型 `transformer_tenant_0604` 的改善幅度**：
- token MAE **−48.8%**（0.048474 → 0.024828）
- token bias **−99.07%**（0.014625 → 0.000136）
- `mean_log_is` **2.756 → +0.026**（−99.07%）
- `med|log_is|` **−69.8%**（2.018 → 0.610）
- clip01 −8.1pp，clip02 −15.5pp
- **ΔR² = 0.5217**

**时间顺序划分（更难）的 `mlp_weight_0604`**：MAE −31.5%，bias −95.5%，clip02 −19.0%，但 ΔR² 只有 **0.2339**。

> ⚠️ **必须主动交代的问题**：`transformer_tenant_0604` 的 **val ΔR² 只有 0.138，而 test ΔR² 是 0.522**，差 3.8 倍。原因是 tenant split 的 val（math-shared / mbpp-A / mbpp-shared）与 test（toolbench×2 / triviaqa×2）的 mismatch 统计差异很大。**任何"R²=0.52"的 claim 都必须限定说明模型选择是在一个只测到 0.138 的集合上做的。** 跨租户泛化是高度异质的——这本身也是一个可报告的发现。

### 6.2 训练轨迹要点

**`mlp_weight_0604`**（时间顺序 split 上的最佳泛化，单调无过拟合）：

| epoch | val MAE | val bias | mean_log_is | med\|is\| | clip02 | ΔR² |
|---|---|---|---|---|---|---|
| 1 | 0.040601 | 0.000409 | +0.0814 | 1.1380 | 0.8857 | −0.0061 |
| 6 | 0.036218 | 0.000574 | +0.1144 | 0.8768 | 0.8314 | 0.0038 |
| 8 | 0.028574 | 0.001724 | +0.3432 | 0.6611 | 0.8098 | 0.2057 |
| 12 | 0.025023 | 0.001265 | +0.2518 | 0.6134 | 0.7872 | 0.2525 |
| **20** | **0.024287** | **0.0000749** | **+0.0149** | 0.6243 | 0.7790 | **0.2708** |

注意 epoch 6→8 之间 ΔR² 有一个台阶（0.004 → 0.206）。

**`transformer_tenant_0604`**：val ΔR² 在 0.138 见顶；MLP 在 ΔR² 0.24–0.28 平台，且 `mean_log_is` 有两个 epoch 周期的震荡（−1.156 → −0.170 → −1.421），说明 bias 项未完全收敛。

### 6.3 非学习型 baseline（`simple_corrections.py`）

| 方法 | 公式 | 需要什么 |
|---|---|---|
| `baseline` | `r_t = Δ_t` | — |
| `global_mean_sub(train)` | 减去 train 上的全局均值 | 一次 calibration run |
| `global_mean_sub(oracle)` | 减去 test 上的全局均值 | oracle |
| `per_seq_mean_sub` | 减去每条序列自己的均值 | **oracle**（需要两条 logprob 路径） |
| `self_norm_IS` | `r_t = Δ_t − mean_j(log w_j)/T_i` | 只需当前 batch |

**要点**：`per_seq_mean_sub` 是 **oracle 上界**（它把序列级误差打到 0，但需要 training logprob，推理时拿不到）。predictor 的任务本质上就是**在只有 sampling 侧信息的条件下逼近这个 oracle**。这个 framing 非常干净，建议直接用在论文的 Design 小节开头。

`self-normalized IS` 不能完全消除 length bias：它只减掉 batch 均值 `T̄·c`，残留 `(T_i − T̄)·c`。正确做法是 token-level 减 `c`。

> ⚠️ 这两个脚本（`simple_corrections.py`、`seq_bias_predictor.py`）**都没有保存输出**，只打印到 stdout。要进论文表格必须重跑并落盘。

---

## 7. 端到端 RL 实验

### 7.1 实验设置

**驱动**：`05_rlhf_bench/rlhf_bench.py`（38K）

| 项 | 值 |
|---|---|
| base model | Qwen3-4B |
| LoRA rank | 8（`train_mlp/attn/unembed` 全开） |
| learning rate | 1e-4，Adam(0.9, 0.95, eps 1e-8) |
| buffer_size / grpo_g | 8 / 8 |
| weight sync | **每步同步** → nominal staleness = 0，残差是纯 framework mismatch |
| backend | Tinker service @ localhost:10610 |
| 训练 loss | `importance_sampling` |
| eval | temperature **0.01**（近 greedy），每问题 1 个样本，滚动**非重叠** test 切片，`reward > 0.5` 判对 |
| 七个 arm | reinforce/grpo × baseline/predictor/global_mean + grpo_oracle |

**实际训练 loss（`src/tuft/loss_fn/importance_sampling.py`）**：

```python
prob_ratio = torch.exp(target_logprobs - sampling_logprobs)
loss = -(prob_ratio * advantages).sum()
```

> 🔴 **这是一个必须正视的事实：loss 里没有任何 clipping。** `clip02` 只是一个记录下来的**诊断量**，不是作用在梯度上的算子。
>
> **两个后果**：
> 1. **好消息**：无 clip 意味着 bias **直接线性放大梯度**，端到端效应比有 clip 时更纯粹、更易解释。
> 2. **坏消息**：之前"oracle（IS≡1）失败是因为 clip 永不触发、GRPO 失去 trust region"的解释，**与这份代码不符**——这里根本没有 clip 可失去。见 §9.4 和 §11.1。

### 7.2 GSM8K（`results/bench/`，50 步，eval_n=30，seed 42）

| 指标 | reinforce_baseline | reinforce_predictor | grpo_baseline | grpo_predictor |
|---|---|---|---|---|
| **final test_acc** | 0.867 | **0.933** | 0.933 | **0.967** |
| mean test_acc（10 次 eval） | 0.7367 | **0.8033** | 0.8567 | **0.8700** |
| mean_reward（last10） | 0.8250 | **0.9000** | 0.8875 | **0.9500** |
| corrected mean_diff（ALL） | 0.00528 | **0.00381** | 0.00454 | **−0.00023** |
| corrected std_cum（ALL） | 1.5845 | 2.1580 | 1.3351 | 1.0146 |

逐次 eval 对比（REINFORCE，10 次）：predictor **6 胜 3 平 1 负**，唯一落后的是 step 20（0.700 vs 0.833）。后半程（step 35 起）优势持续扩大：0.867/0.833/0.933/0.933 vs 0.733/0.667/0.667/0.867。

> ⚠️ 研究日志里曾记录「REINFORCE+pred 全程稳定领先（step 10 起）」，**与数据不符**（step 20 是反例）。论文里请用上面这个精确表述。

### 7.3 MATH（`results/bench_math/`，60 步，eval_n=50，seed 42）

| 指标 | reinforce_baseline | reinforce_predictor | grpo_baseline | grpo_predictor |
|---|---|---|---|---|
| **final test_acc** | 0.740 | **0.780** | **0.600** | **0.860** |
| best test_acc | 0.740 | 0.800 | 0.740 | **0.880** |
| mean_reward（last10） | 0.4500 | 0.5500 | 0.3250 | **0.6125** |
| raw std_cum（last10） | 7.1965 | 10.2739 | **30.0967** | 9.5097 |
| raw clip02（last10） | 0.9375 | 0.9375 | **1.0000** | 0.9625 |

逐 eval 轨迹（step 10→60）：
- `grpo_baseline`: 0.640, 0.580, 0.580, 0.620, 0.740, **0.600**
- `grpo_predictor`: 0.620, 0.780, 0.740, 0.840, 0.880, **0.860**

**这是最有戏剧性的一组**：MATH + GRPO 的 baseline **崩溃**（final 0.600，last10 `std_cum` = 30.10，clip02 = **1.000** 即每条序列都超出 clip 带，step 54 的 `std_cum` 峰值达 **213.7**），而 predictor 组维持在 0.860 且 `std_cum` 压到 9.51。差距 **26pp**。

**崩溃是间歇性爆发（5 → 49 → 43）而非均匀恶化**，这支持一个 positive feedback loop 的机制假说：长序列 → 更大累积 bias → 更大伪权重 → 更倾向生成长序列。

### 7.4 MATH 多 seed（42/43/44，GRPO 三臂）

`global_mean_bias = 0.00360880373475033`

| arm | final acc | reward(last5) | median\|corrected bias\| | median corrected std_cum | n_explode（std_cum>10 的步数/60） |
|---|---|---|---|---|---|
| baseline | **0.7400 ± 0.0748** | 0.4000 ± 0.2354 | 0.00498 | 1.8909 | 3.67 |
| global_mean | **0.7600 ± 0.0327** | 0.4917 ± 0.1830 | 0.00176 | 1.8048 | 3.33 |
| **predictor** | **0.8067 ± 0.0471** | **0.5333 ± 0.0514** | **0.00129** | **1.5966** | **5.33** |

**逐 seed 明细**：

| arm | seed 42 | seed 43 | seed 44 |
|---|---|---|---|
| baseline | 0.640 | 0.760 | **0.820** |
| global_mean | 0.800 | 0.720 | 0.760 |
| predictor | **0.840** | **0.840** | 0.740 |

**可以 claim 的**：
- predictor 相对 baseline **+6.67pp 绝对 / +9.0% 相对**；global_mean 只有 +2.0pp
- predictor 把 **seed 间 final reward 的方差压缩 4.6×**（±0.2354 → ±0.0514），同时均值更高
- median |corrected bias| 降 **74%**，median corrected std_cum 降 **16%**

**必须同时交代的**：
- 🔴 **seed 44 是反例**：baseline 0.820 > predictor 0.740。逐 seed 配对看是 +0.20 / +0.08 / **−0.08**，**2/3 seed 支持 predictor**
- 🔴 n=3，predictor 优势约 1.4σ，**未达常规显著性阈值**
- 🔴 **predictor 的 n_explode 反而更高**（5.33 vs 3.67）。原因是 predictor 组生成更长的 response（528/484/522 vs baseline 366/481/403），长序列累积更多 `Σ_t Δ_t`
- 🔴 **三臂的 mean corrected clip02 几乎相同**（0.9076 / 0.9021 / 0.9021）。**不要 claim 降低 clip 率**——这与 §2.3 的数学预测完全一致，win 在 bias 和 std_cum

### 7.5 Countdown（seed 42，T=0.9，max_tok 256）

| 指标 | baseline | predictor | global_mean |
|---|---|---|---|
| **final test_acc** | 0.220 | **0.320** | 0.240 |
| mean_reward(last10) | **0.2794** | 0.2305 | 0.1850 |
| raw std_cum | **0.2042** | 5.9828 | 6.0254 |
| raw clip02 | **0.3250** | 0.9250 | 0.9375 |
| **平均 response 长度** | **74.1** | **208.6** | 202.8 |

> 🔴 **这组数据极易被误读，必须与长度曲线一起呈现。**
>
> baseline 那些"漂亮"的 mismatch 指标（std_cum 0.204、clip02 0.325）**不是稳定性的证据，而是长度坍缩的 artifact**。baseline 的逐步长度轨迹：
>
> ```
> 185 238 256 146 211 237 169 191 205 198 134 185 180 108  69 154 199 148  73 168
>  79  66  53  22 100  31  37  17  15  28  14  18  15  31  15  16  17  22  14  17
>  36  17  28  17  15  15  17  15  15  17  14  15  15  15  17  15  16  13  15  17
> ```
>
> 从第 ~28 步起 baseline 只输出约 15 个 token。`T=15` 时 `Σ_t Δ_t` 当然很小。predictor 组全程维持 ~200 token 且 final acc 更高（0.320 vs 0.220）。
>
> **任何只报 clip 率不报长度的表格都会被误读。**

### 7.6 跨数据集分层结果（机制特异性）

| 数据集 | predictor 收益 | 解释 |
|---|---|---|
| **Countdown** | **+10pp** | 长 reasoning，最受益 |
| **MATH** | **+6 ~ +9pp**（单 seed 最高 +26pp） | 长 reasoning + 高难度 |
| **GSM8K** | **+3pp** | 中等长度 |
| HumanEval | ~0 | 接近 ceiling |
| **IFEval** | **−6pp**（0.480 → 0.420） | 短序列 + binary 约束 reward，无 length bias 可修 |
| MBPP / TriviaQA / HotpotQA | 无有效信号 | — |

> **这张表是论文里最有说服力的一张。** IFEval 的负结果不是退化，而是**机制特异性的反向验证**：reward 由 binary 约束驱动、response 短 → `T·c` 累积小 → predictor 修复空间小。
>
> **叙事建议**：用分层表证明「我们理解它什么时候有效、为什么有效」，比「全 benchmark 都赢」更有说服力，也更抗 review。明确写出适用条件：**任务难度高 + 序列长 + 序列级 IS 主导**。

---

## 8. 论文里的定位与 claim 边界

### 8.1 最终定位（已锁定）

> **Corrector = reliability layer，不是 peak optimizer。**

它不提升 SOTA 精度上限，它**消除 disaggregation 引入的 correctness 缺陷**，让训练在长序列、高难度任务上不崩。这个定位有三个好处：

1. 与系统论文的气质一致（系统论文卖 reliability 和 efficiency，不卖 SOTA）；
2. 天然解释了为什么在 HumanEval/IFEval 上没有收益（本来就没坏，不需要修）；
3. 避开了"为什么不和最新 RL 算法比"的攻击。

### 8.2 可以 claim 的

- 跨引擎 logprob 存在 1.1%–1.6% 的系统性差异，商业服务（52.4%）比自建（40.7%）更严重且不透明
- 量化把 p99 IS weight 推高 7 个数量级
- per-token 残差的 99.2% 是序列内不可约噪声；唯一可学的信号是 per-sequence 均值
- learned per-seq predictor 把 token bias 降 **99%**、`mean_log_is` 从 2.756 降到 0.026、`med|log_is|` 降 **70%**
- 端到端：MATH GRPO 3 seed 上 **+6.7pp 绝对精度**，seed 间 reward 方差压缩 **4.6×**
- 收益与「任务难度 × 序列长度」强相关，机制清晰可预测

### 8.3 不能 claim 的（会被打）

| 不要说 | 因为 |
|---|---|
| "降低 clip 率" | 三臂 clip02 几乎相同（0.9076/0.9021/0.9021），且 §2.3 证明这是数学必然 |
| "全 benchmark 提升" | IFEval −6pp，MBPP/TriviaQA/HotpotQA 无信号 |
| "统计显著" | n=3，1.4σ，seed 44 是反例 |
| "训练更稳定"（笼统） | predictor 的 n_explode 反而更高（5.33 vs 3.67） |
| "Countdown 上 baseline 更不稳" | baseline 的好指标是长度坍缩 artifact |
| "ΔR²=0.52 的泛化能力" | 模型选择集上只有 0.138 |
| "per-token 完全不可学" | ANOVA 的 0.8% 只约束序列内恒定预测器 |

---

## 9. Oracle 与 baseline grid 的概念澄清

这一节是最近几轮讨论的核心产出，直接决定 evaluation 表怎么设计。

### 9.1 两种"training 重算 sampling logprob"

| | **做法 A：Disaggregated** | **做法 B：Co-location** |
|---|---|---|
| 采样引擎 | vLLM | training 引擎自己 autoregressive |
| training_lps | HF/FSDP forward | 同一次生成时记录 |
| sampling_lps | vLLM 记录 | 同一引擎记录 |
| `Δ_t` | **真实有差** | **恒为 0** |
| IS weight | 真实且有意义 | ≡ 1（sync）或纯 staleness（async） |
| forward 次数 | 快采样 + 1 次 full-seq forward | T 次 autoregressive forward |
| 吞吐 | 高 | 低（常见 5–10× 退化） |

**关键澄清**：做法 A **不是把同一次 forward 抓两份**，而是**两次独立的 forward**，走完全不同的实现路径。数学上求值的是同一个函数，实现上结果不同 —— 这就是 mismatch 的来源。

### 9.2 sync on-policy 下的重要性质

用户提出并且正确的论断：

> sync on-policy 下，vLLM 与 HF 的"策略差距" **100% 是 framework mismatch**，因为 weight 是同一份，不存在 staleness 成分。

**两个推论**：

1. **数据采集**：只在 sync 模式下采 `(s_t, p_t)`，得到的就是纯 mismatch —— 代码已经这么做了（`staleness == 0` 过滤）。
2. **Scope 限定**：sync on-policy 下真实策略距离为 0，**本来就不需要 IS**，也就不需要 Corrector。**TuFT 的 Corrector 只在 async / multi-step 场景下才有价值**，因为只有那时：(a) 需要 IS 处理 staleness，(b) IS 又被 mismatch 污染，(c) 才需要把两者剥离。

**论文必须显式写出这个 scope**，否则 reviewer 会问"on-policy 下你这套有什么用"。建议措辞：

> TuFT's Corrector addresses framework mismatch in **async / multi-step disaggregated** RL, where importance weights are necessary to handle staleness but are contaminated by framework-induced bias. Under strictly synchronous on-policy training, importance weights are unnecessary and so is the Corrector.

### 9.3 Oracle（IS ≡ 1）为什么不是合法 baseline

Oracle 的做法是用 `training_lps` 直接替换 `sampling_lps`，使 IS ratio ≡ 1。

**问题**：采样**仍然来自 vLLM**（真实行为分布是 `π_vllm`），但强行令 ratio = 1，**等价于在错误的分布下做 on-policy 假设**。它不只消除了 mismatch artifact，还消除了 vLLM ↔ HF 之间真实存在的分布差异。

**在 async 场景下更糟**：oracle 同时杀死了 mismatch artifact **和** staleness signal。

**因此 oracle 不应进入 evaluation 表。** 它不是一个合法的训练配置，只是一个 thought experiment，而这个 thought experiment 因为语义退化而不 informative。

### 9.4 Reframe：把 oracle 的失败变成支持 TuFT 的证据

不要说"oracle 失败说明 mismatch 不重要"——这会让 reviewer 质疑整个方法。要说：

> Oracle (IS ≡ 1) fails because it simultaneously removes both the staleness signal and the mismatch artifact, degenerating the off-policy correction. This validates that **importance weighting is necessary**, and motivates **separating staleness from mismatch within the importance weight** — which is exactly what the Corrector does.

这样 oracle 的失败就从"质疑 TuFT"变成"证明 TuFT 必要"。

> ⚠️ **但注意 §11.1**：本地**没有任何 oracle 运行结果**，而且当前 loss 里**没有 clip**，所以旧的"clip 永不触发→失去 trust region"这一具体机制解释**与代码不符**。要么补跑 oracle 并重新解释机制，要么在论文里干脆不提 oracle。

### 9.5 建议的 baseline grid（五行，oracle 不入表）

| Setting | Sampling 引擎 | training_lps 来源 | `Δ_t` | 用途 |
|---|---|---|---|---|
| **Sequential (UB)** | training 引擎 | 同引擎（sync） | ≡ 0，IS ≡ 1 | accuracy 上界 |
| **Async co-location** | training 引擎（旧 weight） | 同引擎 | ≡ 0，IS = 纯 staleness | **真正的 mismatch=0 对照** |
| **Naive Disagg** | vLLM | HF | 真实有差，不处理 | throughput 上界，acc 低 |
| **Off-policy IS only** | vLLM | HF | 用 raw `Δ_t` 算 IS | 只处理 staleness，不处理 mismatch |
| **TuFT (Corrector)** | vLLM | HF | predictor 校正 | full method |

**第 2 行（async co-location）是最重要的新增**：它与 TuFT 有**同样的 staleness、但没有 mismatch**，能干净地隔离出 mismatch 单独造成多少精度损失。可行性上，HF/FSDP 做 autoregressive generation 很慢，但只需要跑几个 seed 拿 accuracy 数字当 baseline，不追求吞吐。

**另外建议的 sanity check**：sync on-policy 的 disagg，对比 raw vs Corrector。这是**没有 staleness 干扰的纯净环境**，如果仅 mismatch 就能让 raw 组训练变差而 Corrector 修复它，这是 Corrector 价值最干净的证据。

### 9.6 与 MARLaaS 的关系（重要）

MEMORY.md 已核实：MARLaaS 是**严格 on-policy** 的（只在最新 committed 版本上训练，无 staleness）。这意味着：

- MARLaaS 结构上**无法使用 TuFT 的 delay lever**；
- 但按 §9.2，它同样**结构上不需要 Corrector**（on-policy 下 IS 不必要）；
- 因此需要加一个 **TuFT-onpolicy (W=0) variant**，让 reviewer 不能把增益归因于放松 on-policy 约束。

---

## 10. 资产清单（代码 / 数据 / 图 / checkpoint）

根目录 `/Users/yanghanzhang/work/research/`

### 10.1 代码

| 路径 | 作用 |
|---|---|
| `code/experiments/TuFT/experiments/01_mismatch_motivation/verify_mismatch.py` | 微基准生成器（31 轮 JSON） |
| `.../01_mismatch_motivation/plot_comparison.py` | 2×2 对比图（v3 图的祖先） |
| `paper_writing/experiments/04_mismatch_moti/code/plot_mismatch_severity.py` | severity CSV + 两张论文图 |
| `.../03_bias_predictor/{data,model,losses,train,predict,plot}.py` | Predictor 主体 |
| `.../03_bias_predictor/simple_corrections.py` | 非学习型 baseline |
| `.../03_bias_predictor/seq_bias_predictor.py` | sklearn per-seq 回归（11 特征） |
| `.../03_bias_predictor/length_clip_analysis.py` | 长度分桶 / clip 非对称性 / 长度×clip 交叉 |
| `.../03_bias_predictor/per_token_impossibility.py` | ANOVA / ACF / IS 重建三实验 |
| `.../05_rlhf_bench/rlhf_bench.py` | 端到端 RL 主驱动（7 arm） |
| `.../05_rlhf_bench/plot_bench.py`、`plot_multi_seed.py` | 训练曲线 / 多 seed 聚合 |
| `.../05_rlhf_bench/predictor/correction_comparison{,_v2}.py` | 7/8 方案离线对比 |
| `.../05_rlhf_bench/launch_*.sh`（15 个） | 逐数据集 / 逐 seed 启动脚本 |
| `src/tuft/loss_fn/importance_sampling.py` | 实际训练 loss（**无 clip**） |

`05_rlhf_bench/predictor/` 是 `03_bias_predictor/` 的**较新超集**（多了 `ScalarMLPPredictor`、`build_multitask_v{2,3}.py`、dispatch bug 修复）。写论文时以 `05_rlhf_bench` 版本为准。

### 10.2 数据

| 路径 | 内容 |
|---|---|
| `data/microbench/01_framework_mismatch/{tinker,tuft}_mismatch_results.json` | 各 31 轮微基准 |
| `paper_writing/experiments/04_mismatch_moti/data/tp2_fsdp1_id0_quant.jsonl` | 量化臂原始数据，113.6 MB，8480 行，step 1–9 |
| `.../05_rlhf_bench/simulator/logprobs/bench_merged_r8_t07.jsonl` | 1280 序列 GSM8K r8 T0.7 |
| `.../simulator/logprobs/multitask_logprobs.jsonl` | 7680 序列 = 6 任务 × 1280，107 MB |
| `.../simulator/logprobs/tp2_fsdp1_id0_noquant.jsonl` | impossibility 分析用，209 MB |
| `.../04_simulator/22_agents_50_step_results.logprobs.jsonl` | predictor 训练集，403 MB，36,499 条 |
| `data/gpu_stats_{async,sync_long}.csv` | GPU 利用率（scheduling 侧，非 mismatch） |

### 10.3 图

| 路径 | 内容 |
|---|---|
| `docs/figures/framework_mismatch_comparison_v3.{png,pdf}` | 2×2：per-token 幅度 / 序列累积（含 within-clip 带）/ IS 方差 / clip 违反率 |
| `paper_writing/experiments/04_mismatch_moti/figures/mismatch_severity_{comparison,sequence_cumulative_comparison}.{pdf,png}` | 三方 severity 对比 |
| `.../05_rlhf_bench/simulator/results/per_token_impossibility/exp{1,2,3}_*.png` | ANOVA / ACF / IS 重建 |
| `.../simulator/results/length_clip_analysis{,_tau041}/exp{1,2,3}_*.png` | 长度 → 累积 bias → clip 机制 |

### 10.4 Checkpoint

`03_bias_predictor/predictor/checkpoints/` 下六个：`mlp_task_id_tenant_0603`、`mlp_tenant_0604`、`mlp_weight_0604`、`transformer_task_id_tenant_0603`、`transformer_tenant_0604`、`transformer_weight_0604`（未跑完）。

**本地缺失**（端到端实验实际用的就是它们）：`bench_v1`、`multitask_transformer_v1`、`scalar_bench_v1`、`mlp_bench_v1`。原始位置：`/mnt/nas/hanzhang.yhz/evaluation/predictor/checkpoints/`。

### 10.5 相关论文（`papers/`）

`MARLAAS.pdf`、`TIS.pdf`、`flexllm.pdf`、`HybridFlow_Verl.pdf`、`opentinker.pdf`、`roll.pdf`、`m-lora.pdf`、`LoRAFusion.pdf`、`阿里_stable_rl.pdf`、`Your Efficient RL Framework Secretly Brings You Off-Policy RL Training`、`Training sampling scheduling.pdf`（自己的思路文档）。

---

## 11. 已知不一致与待补实验

按严重程度排序。**投稿前必须逐条处理。**

### 🔴 P0 —— 会动摇 claim

**11.1 Oracle 组本地无任何结果，且机制解释与代码矛盾**
- `grpo_oracle` 在 `group_spec` 里、`launch_math_oracle_seed42.sh` 等三个脚本存在，但 `results/bench_math_oracle_seed{42,43,44}/` **不存在**
- 更严重：训练 loss `importance_sampling` **没有 clipping**，所以"oracle 失败是因为 clip 永不触发、GRPO 失去 trust region"这一解释**在这份代码里不成立**
- **处理**：要么补跑 oracle 并重新给出机制解释，要么按 §9.3 干脆不把 oracle 放进论文。若保留，必须核对当时跑 oracle 用的是不是另一个带 clip 的 loss

**11.2 Countdown baseline 的好指标是长度坍缩 artifact**
- baseline 从 step ~28 起只输出 ~15 token（overall mean 74.1 vs predictor 208.6）
- `std_cum` 0.204、clip02 0.325 全是长度导致
- **处理**：任何 Countdown 表格必须并排给出 response 长度列，或在 caption 里明说

**11.3 统计功效不足**
- n=3 seed，predictor 优势约 1.4σ，seed 44 是反例（baseline 0.820 > predictor 0.740）
- **处理**：按之前的决策，**优先增加 seed 到 5–10**（而不是换更大模型）。`launch_math_baseline_verify_seed45.sh` 已存在但未跑

**11.4 三臂 clip02 几乎相同**
- 0.9076 / 0.9021 / 0.9021 —— **不能 claim 降 clip 率**
- 与 §2.3 的数学预测一致，是预期行为，但必须主动说明而不是回避

**11.5 val ΔR² 0.138 vs test ΔR² 0.522**
- 模型选择在一个只测到 0.138 的集合上做的
- **处理**：报告时必须给出 val/test 两个数，并解释跨租户异质性

**11.6 predictor 的 n_explode 反而更高**
- 5.33 vs baseline 3.67（60 步中 `std_cum > 10` 的步数）
- 原因是 predictor 组生成更长 response
- **处理**：主动解释为"predictor 阻止了长度坍缩，代价是更多长序列进入统计"，并配长度曲线

**11.7 ANOVA 的 0.8% 上界与 per-token 模型 ΔR² 0.23–0.52 表面矛盾**
- 0.8% 只约束「序列内输出恒定」的预测器
- **处理**：论文里必须显式调和，或在 22_agents 数据上重跑 ANOVA

### 🟠 P1 —— 数字对不上，必须订正

**11.8 proposal 与实际数据不符（三处）**

| proposal 写的 | 实际数据 |
|---|---|
| Tinker 53% / 本地 38% | **52.42% / 40.65%** |
| σ(IS) 峰值在 Round 17 | **round 18** |
| 固定 10 个 prompt | **5 prompt × 4 samples = 20 序列** |

**11.9 predictor README 的结果表对应不上任何本地 checkpoint**
- README 称 70,115 条 / 48,656-10,426-10,427 划分；实际文件 36,499 条 → 36,153 clean → 25,307/5,422/5,424
- README 称 R² 0.514 / MAE 0.0251 / clip01 0.862；最接近的 `transformer_tenant_0604` 是 0.5217 / 0.024828 / 0.881296
- **处理**：README 描述的是一个不在本地的 run。论文只能引用**能核实的 checkpoint 数字**

**11.10 README 与代码的其他 drift**
- README 说 12 tasks / `lora_rank ∈ {4,8,16}`；实际 **11 tasks / {8,16}**
- README 说 `Var(Δ_t) ≈ 0.001`；从 epoch-0 R² 反推实际为 **0.0382（weight split）/ 0.0905（tenant split）**，impossibility 数据集上 0.0121
- README 描述的 `task_idx` / `lora_rank_idx` categorical embedding 已被移除（见 §5.2）
- **处理**：把 README 更新到与 0604 代码一致，或在论文里只引用代码

**11.11 `delta_r2` 的实现与 README 公式不一致**
- `SS_tot` 做了 mean-centering，`SS_res` 没有
- 后果：零预测器给出的不是 0 而是 `−mean(Δ)²/Var(Δ)`（日志里 epoch-0 恒为 −0.00254 / −0.00340）
- **处理**：论文里如实写成"fraction of variance explained"，或改代码统一口径

**11.12 global_mean 常数存在两个版本**
- 脚本里 `0.00360880373475033`（GSM8K r8 T0.7），独立复算得 `0.003473`（差异可能来自 `MAX_SEQ_LEN=512` 截断）
- 新脚本用 `0.008113`（5 个 multitask 任务 pooled，已精确复算验证）
- **处理**：论文里说清楚每个 global_mean 数字的拟合来源

**11.12b 研究日志的一条结论与数据不符**
- 日志（2026-06-21）记录「GSM8K: REINFORCE+pred 全程稳定领先（step 10 起），非仅后期拉开」
- 实测：10 次 eval 中 predictor **6 胜 3 平 1 负**，step 20 为反例（0.700 vs baseline 0.833）
- **处理**：已在 §7.2 订正。**凡是只出现在研究日志、未在本地文件中核实的结论，进论文前都要重新核一遍**

### 🟡 P2 —— 缺失的实验 / 产物

**11.13 五个数据集只有脚本没有结果**
HumanEval、MBPP、IFEval、HotpotQA、TriviaQA 的 `launch_*.sh` 都在，`results/` 下**没有对应目录**。§7.6 的跨数据集表里这几行的来源是研究日志而非本地文件，**必须重跑或从 NAS 取回后才能进论文**。

**11.14 GSM8K 60 步三臂 run 在第 9 步崩溃**
`results/bench_gsm8k_seed42/` 只有 9 行 `grpo_baseline`，未到第一次 eval。不可用。

**11.15 "newpred"（multitask_transformer_v1）臂无结果**
`launch_math_new_pred_seed42.sh` 注释称该 predictor 有 **−83% bias**，但无运行结果，checkpoint 也不在本地。

**11.16 离线 7/8 方案对比无法复现**
`correction_comparison{,_v2}.py` 需要 `bench_v1` / `multitask_{transformer,scalar,mlp}_v1`，全部缺失。v2 脚本设计了 in-distribution 与 **OOD holdout（ifeval）** 双评估，这是一个很好的泛化性证据，值得补回来。

**11.17 所有 RL 训练曲线图未落盘**
`*.png` 在 `.gitignore` 里。需要重跑：
```bash
python plot_bench.py --input results/bench_math_seed42 --output results/bench_math_seed42
python plot_multi_seed.py --base results --seeds 42 43 44 --output results/bench_math_multiseed
```

**11.18 `plot_bench.py` 的图标题硬编码为 GSM8K**
标题写死 `"Test Accuracy on GSM8K Held-out Set"`，但同一脚本也用于 MATH / Countdown。**进 figure caption 前必须改**。

**11.19 `framework_mismatch_comparison_v3` 无生成脚本**
只有祖先版 `plot_comparison.py`（缺 (b) 面板的 mean+max 叠加和 clip 阴影带）。可复现性缺口。

**11.20 `simple_corrections.py` / `seq_bias_predictor.py` 无保存输出**
只打印 stdout。要进论文表格必须加 `--output` 重跑。

### 🟢 P3 —— 代码缺陷（不影响已有结论，但影响后续实验）

**11.21 `train.py::build()` 无法构造 `scalar_transformer`**
```python
if args.model == "transformer": return build_model("transformer", ...)
return build_model("mlp", ...)          # ← scalar_transformer 静默变成 MLP
```
`--model scalar_transformer` 会切换 loss 和选择指标，但建出来的是 MLP。**`05_rlhf_bench/predictor/train.py` 里已修复**。这就是"下一版 predictor 直接输出 per-seq scalar"的架构决策**至今没有实验数据**的原因。

**11.22 模型选择指标与下游目标不一致**
用 `corrected_token_mae` 选，导致 `transformer_task_id_tenant_0603` 选中了 bias 最差的 epoch。建议改成 `corrected_mean_log_is`。

**11.23 `L_seq` 除以 `T_b` 抵消了长度加权**
见 §5.4。这与问题本质（长序列累积更多 bias）相悖，是一个有价值的改进方向。

**11.24 `filter_clean_records` 默认值与 README 相反**
代码 `r.get("staleness", 1.0)` → 缺字段的记录被**丢弃**；README 写的是默认 0.0（保留）。

**11.25 clip 阈值非对称**
`(log 1.1, log 0.9) = (+0.0953, −0.1054)`，不是 `±log(1.1)`。`median_abs` 用的是排序后 `len//2` 的上中位数，无插值，与 `simple_corrections.py` 里的 `np.median`（有插值）口径不同。

**11.26 `median_per_seq_within_ratio = 1.0` 是恒等式 artifact**
```python
seq_var_within = float(np.var(d - d.mean()))   # ≡ np.var(d)
```
恒等于 1.0。**不要引用这个数字。**

**11.27 `transformer_weight_0604` 未跑完**
20 epoch 只跑了 10，无 `test_summary.json`。但它有**全部 run 里最低的 val MAE（0.021029）**，值得跑完。

**11.28 `data/gpu_stats_sync_long.csv` 是孤儿文件**
没有任何脚本读它；绘图脚本读的是另一个 tab 分隔、更短、且表头单位被破坏的 `gpu_stats_sync.csv`。

---

## 12. 下一步建议（按投入产出比排序）

| 优先级 | 动作 | 理由 |
|---|---|---|
| 1 | **MATH GRPO 补到 5–10 个 seed** | 直接解决 P0 的统计功效问题，成本最低 |
| 2 | **跑 async co-location baseline** | §9.5 表里唯一缺失的关键行，是"mismatch=0"的合法对照 |
| 3 | **重跑 HumanEval/MBPP/IFEval/HotpotQA/TriviaQA** | §7.6 分层表是最有说服力的一张，但现在数据不在本地 |
| 4 | **sync on-policy disagg：raw vs Corrector** | 纯净环境下的 sanity check，Corrector 价值最干净的证据 |
| 5 | **训练并评估 per-seq scalar predictor** | 架构决策已定 2 个月，但因 dispatch bug 一直没数据 |
| 6 | **dose-response 注入实验**（`c_inject = 0.01/0.02/0.05`） | 替代 oracle 做 mismatch severity 分析，语义干净 |
| 7 | 补 response length distribution 分析 | 验证 length bias 假说，同时解决 P0 的 Countdown 与 n_explode 解释 |
| 8 | 订正 proposal / README 的全部数字 | P1 全部条目 |
| 9 | 改 `L_seq` 去掉长度归一化后重训 | 可能直接改善下游 clip 指标 |
| 10 | 模型选择指标改为 `corrected_mean_log_is` 后重训 | 低成本，直接对齐下游目标 |

---

## 13. Corrector 章节骨架建议

```
§X  Sampling-Side Corrector
 X.1  Why disaggregation introduces logprob mismatch
      - 两条 forward 路径（§1.1）
      - framework mismatch vs staleness 的区分（§1.3）
      - scope 声明：async/multi-step（§9.2）
 X.2  How a tiny per-token bias becomes a sequence-level failure
      - T·c 累积表（§2.1）
      - 按算法分类的影响路径（§2.2）
      - clip 率不可降的数学论证（§2.3）
      → Fig: framework_mismatch_comparison_v3
 X.3  Measurement
      - 微基准协议（§3.1）
      - Tinker vs self-hosted vs quant（§3.2/3.3）
      - 任务异质性（§3.4）→ 论证需要 learned model
      → Fig: mismatch_severity_comparison
 X.4  What is learnable and what is not
      - ANOVA / ACF / IS 重建三证据（§4.3）
      - per-seq oracle 作为上界的 framing（§6.3）
      → Fig: exp1_variance_decomposition
 X.5  Design
      - 学残差 + 零初始化（§5.1）
      - 特征与架构（§5.2/5.3）
      - 三个 invariance → offline profile 部署形态（§5.7）
      - 集成进 sampling 路径（§5.8）
 X.6  Evaluation
      - 离线：bias/mean_log_is/med|log_is|/ΔR²（§6.1）
      - 端到端：MATH 多 seed（§7.4）
      - 分层表：任务难度 × 长度（§7.6）
      - 与 global-mean 的 ablation（§7.4）
      → Fig: multiseed_final_acc_bars
 X.7  Limitations（主动写，别等 reviewer 问）
      - clip 率不降（数学必然）
      - 短序列 / binary reward 任务无收益
      - 依赖 sync-mode calibration probe
```

---
