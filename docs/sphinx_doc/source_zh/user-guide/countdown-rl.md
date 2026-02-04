# Countdown 强化学习（RL）

本指南演示如何使用运行中的 TuFT 服务器在 **Countdown** 数据集上进行**强化学习（RL）**微调。完整的可运行代码在 `examples/countdown_rl/countdown_rl.ipynb` notebook 中。虽然这是一个通用的 RL 指南，但它也记录了用户在使用 TuFT 进行 RL 时可能遇到的常见问题，并提供逐步指导帮助他们成功完成端到端运行。

---

## 您将学到

1. 如何加载和拆分 **Countdown** 任务并将其转换为**提示词风格**的问题
2. 如何设计**基于规则的奖励函数**（格式 + 有效性 + 正确性 + 可选塑形）
3. 如何在 TuFT 中运行最小的 **GRPO 风格** RL 循环（组采样 + 归一化优势 + 重要性采样损失）
4. 如何使用奖励曲线选择和调优 **LoRA rank** 和**学习率**

---

## 目录
1. [数据集](#数据集)
2. [最小训练示例（RL）](#最小训练示例rl)
3. [关键概念](#关键概念)
   - [可验证输出的提示词设计](#可验证输出的提示词设计)
   - [奖励设计：格式、有效性、正确性、塑形](#奖励设计格式有效性正确性塑形)
   - [组采样和归一化优势](#组采样和归一化优势)
   - [RL 的 Datum 格式](#rl-的-datum-格式)
4. [参数选择](#参数选择)
5. [常见问题](#常见问题)

---

## 数据集

本指南使用 **[`Jiayi-Pan/Countdown-Tasks-3to4`](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4)**。

| 数据集 | 来源 | 典型样本字段 | 用例 |
|---|---|---|---|
| Countdown-Tasks-3to4 | `Jiayi-Pan/Countdown-Tasks-3to4` | `nums`（列表）、`target`（整数） | 可验证的算术表达式生成 |

**拆分策略**
- 测试集：前 `TEST_SIZE` 行
- 训练集：剩余行，使用 `SEED` 打乱

这使得运行可重现，并避免需要预定义的"测试"拆分。

---

## 最小训练示例（RL）

与 Tinker 不同，TuFT 可以在本地 GPU 上运行；以下实验在本地 2× NVIDIA A100-SXM4-80GB 设置上进行（驱动 550.54.15，CUDA 12.9）。运行示例前，请按照[安装指南](../getting-started/installation.md)在本地启动 TuFT 服务器。

关键 TuFT 调用（完整代码在 `examples/countdown_rl/countdown_rl.ipynb`）：
```python
import tinker
from tinker import types

service_client = tinker.ServiceClient(base_url=TINKER_BASE_URL, api_key=TINKER_API_KEY)

training_client = service_client.create_lora_training_client(
    base_model=BASE_MODEL,
    rank=LORA_RANK,
    train_mlp=True,
    train_attn=True,
    train_unembed=True,
)

# RL 更新使用重要性采样风格的目标：
training_client.forward_backward(datums, loss_fn="importance_sampling").result()
training_client.optim_step(types.AdamParams(learning_rate=LEARNING_RATE)).result()
```

这个 RL 工作流的一个实际细节：每个训练步骤导出一个采样器兼容的检查点，然后使用采样客户端为目标生成 rollout 和 logprobs。

---

## 关键概念

### 可验证输出的提示词设计

当奖励信号可靠时，RL 效果最佳。对于 Countdown，我们强制执行**可验证的输出契约**：

- 模型必须**仅**在以下标记内输出最终表达式：
  ```text
  <answer> ... </answer>
  ```
- 停止序列 `</answer>` 提前截断生成，减少无用 token 和奖励噪声。

预先添加一个小的 **few-shot** 示例以提高早期训练稳定性（模型更快学习格式）。
```python
COUNTDOWN_FEWSHOT = (
    "Q: Using the numbers 2, 3, 7, reach the target number 13. "
    "You may use +, -, *, / and parentheses, and each number can only be used once. "
    "Put ONLY the final expression inside <answer>...</answer>. "
    "Example: <answer>(1+2)/3</answer>."
    "A: <answer>(2*3)+7</answer>"
)
```

---

### 奖励设计：格式、有效性、正确性、塑形

奖励被有意**分解**为多个阶段：

1) **格式奖励**：输出必须包含 `<answer>...</answer>`
2) **有效性奖励**：表达式必须**精确**使用提供的数字（多重集匹配）
3) **安全评估**：表达式必须在受限的语法/字符集下可解析
4) **正确性**：评估的数值结果必须匹配 `target`

常见的 RL 实践是包含**奖励塑形**以减少稀疏性：

- 如果精确匹配：reward = `1.0`
- 如果不精确：
  - 要么只给一个小常数 `FORMAT_SCORE`（稀疏）
  - 要么使用**连续塑形**，如：$r = r_{\mathrm{fmt}} + \left(1 - r_{\mathrm{fmt}}\right)\frac{1}{1 + \left|y - \mathrm{target}\right|}$

这在模型"接近但不正确"时也提供梯度。

**为什么常数 `FORMAT_SCORE` 重要**
- 它防止早期的"全有或全无"学习。
- 它鼓励策略至少满足格式/有效性约束，然后才能可靠地解决数学问题。

```python
def compute_reward(
    response_text: str,
    target: int,
    nums: list[int],
    format_score: float,
    use_continuous_shaping: bool,
) -> float:
    equation = extract_solution(response_text)
    if equation is None:
        return 0.0

    if not validate_equation(equation, nums):
        return float(format_score)

    result = evaluate_equation(equation)
    if result is None:
        return float(format_score)

    err = abs(result - target)
    if err < 1e-5:
        return 1.0

    if not use_continuous_shaping:
        return float(format_score)

    shaped = format_score + (1.0 - format_score) * (1.0 / (1.0 + err))
    return float(shaped)
```

---

### 组采样和归一化优势

我们不是每个提示词采样单个完成，而是采样一**组**完成：

- 对于每个提示词（问题），采样 `GROUP_SIZE = G` 个 rollout。
- 计算每个 rollout 的奖励。

然后计算**组内**统计：

- 组内平均奖励：$\mu$
- 组内奖励标准差：$\sigma$

优势在**同一组内**归一化：

- 对于样本 $i$：$A_i=\frac{r_i-\mu}{\sigma+\varepsilon}$

这在精神上类似于 GRPO：

- 它从同一提示词的样本之间的**相对质量**中学习（组内比较/排名）。
- 它减少了对学习的价值函数的需求。
- 它对奖励方差敏感：如果 $\sigma$ 约为 0，则跳过该提示词，因为没有有用的学习信号。

**直觉**
- 鼓励模型增加比组平均更好的 rollout 的概率，减少更差的 rollout 的概率。

```python
# 采样 GROUP_SIZE 个完成 -> 计算奖励 -> 在组内归一化优势
res = sampling_client.sample(prompt=prompt, num_samples=GROUP_SIZE,
                             sampling_params=sampling_params_train).result()

rewards, toks_list, lps_list = [], [], []
for seq in res.sequences:
    toks = list(seq.tokens)
    lps  = list(seq.logprobs)  # 必须由采样器返回
    text = tokenizer.decode(toks, skip_special_tokens=True)

    r = compute_reward(text, target=prob.target, nums=prob.nums,
                       format_score=FORMAT_SCORE,
                       use_continuous_shaping=USE_CONTINUOUS_SHAPING)

    rewards.append(float(r))
    toks_list.append(toks); lps_list.append(lps)

mu = sum(rewards) / len(rewards)
sigma = (sum((r - mu) ** 2 for r in rewards) / len(rewards)) ** 0.5
if sigma < 1e-8:
    skipped_problems += 1
    continue

advantages = [(r - mu) / (sigma + 1e-6) for r in rewards]
```

---

### RL 的 Datum 格式

每个采样的 rollout 被转换为包含以下内容的 `Datum`：

- **`model_input`**：提示词 token + 生成的 token（用作下一个 token 预测的输入序列）
- **`loss_fn_inputs`**（每个 token 对齐的额外张量）：
  - **`target_tokens`**：模型应该预测的下一个 token
  - **`logprobs`**：行为策略（采样时）的对数概率，用于重要性采样
  - **`advantages`**：每个 token 的权重（通常在提示词上为 **0**，在生成区域为常数优势）

这使得 token 空间中的**重要性采样风格目标**成为可能：
- 将当前策略似然与行为策略似然（来自采样时）进行比较
- 按优势加权更新，使高奖励 rollout 得到强化

```python
ob_len = prompt.length - 1 

for toks, lps, adv in zip(tokens_G_T, logprobs_G_T, advantages_G):
    model_input = prompt.append(types.EncodedTextChunk(tokens=toks[:-1]))
    target_tokens = [0] * ob_len + toks
    padded_sampling_logprobs = [0.0] * ob_len + lps
    padded_advantages = [0.0] * ob_len + [adv] * (model_input.length - ob_len)

    datums_D.append(
        types.Datum(
            model_input=model_input,
            loss_fn_inputs={
                "target_tokens": TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.long)),
                "logprobs": TensorData.from_torch(torch.tensor(padded_sampling_logprobs, dtype=torch.float32)),
                "advantages": TensorData.from_torch(torch.tensor(padded_advantages, dtype=torch.float32)),
            },
        )
    )
```

---

## 参数选择

本节解释如何为 Countdown RL 任务选择 `lora_rank` 和 `learning_rate`，并根据提供的实验结果总结结论。本文档基于 [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)。

### `lora_rank` 和 `learning_rate` 的作用

**`lora_rank`（LoRA 适配器秩）**控制适配器容量：
- 更高的秩 → 更多可训练参数 → 潜在更高的上限，但更多计算/内存且在 RL 中可能不太稳定
- 更低的秩 → 更便宜且通常更稳定；通常足以进行策略/行为塑形

**`learning_rate`**控制策略更新的步长：
- 太低 → 改进缓慢（更新不足）
- 适中 → 快速且稳定的奖励增益
- 太高 → 训练不稳定/奖励崩溃（过度更新），这在 RL 微调中很常见

---

### 图表的实验结论

基于**图 1**（`lora_rank ∈ {8, 32, 128}` 的最终奖励 vs. 学习率）：

1) **奖励随着 LR 从 `1e-6` 增长到约 `1e-4` 而增加。**
   所有秩在这个范围内都呈上升趋势，表明优化器需要足够大的 LR 才能取得有意义的策略进展。

2) **最佳性能区域在 `5e-5` 到 `1e-4` 附近。**
   最终奖励在 `1e-4` 附近达到峰值（在 `5e-5` 时已经很强），跨所有秩。这个范围是稳定学习 + 良好性能的实用"甜蜜点"。

3) **太大的 LR（`5e-4`）导致奖励崩溃，尤其是对于更高的秩。**
   在 `5e-4` 时，`lora_rank=32` 和 `128` 急剧下降（接近失败），而 `rank=8` 下降但明显更好。这表明在过于激进的 LR 下更新不稳定。

4) **秩的收益递减；更大的秩并不总是更好。**
   在最优 LR 区域（`5e-5`–`1e-4`），秩 `8/32/128` 相对接近。在这个设置中，**LR 是主导因素**，增加秩超过中等值并不能可靠地提高奖励。

5) **较小的秩在激进学习率下更宽容。**
   当 LR 推得太高时，`rank=8` 比 `rank=32/128` 下降得更少，表明对大更新有更好的鲁棒性。

```{figure} ../../_static/images/countdown_rl.png
:alt: Countdown RL 最终奖励 vs. 学习率
:width: 720px
:align: center

**图 1. 不同 LoRA 秩下 Countdown RL 最终奖励 vs. 学习率**
```

### 实用建议

**强默认值（推荐起点）**
- `lora_rank = 8 或 32`
- `learning_rate = 1e-4`

**如果训练不稳定（奖励先升后降/崩溃）**
- 首先降低 LR：`1e-4 → 5e-5 → 1e-5`
- 如果仍不稳定，降低秩：`32/128 → 8`
- 对于这个设置避免 `5e-4`（图 1 显示高崩溃风险，尤其是 `rank ≥ 32`）

**如果学习太慢/奖励停滞**
- 逐渐增加 LR 到 `5e-5` 或 `1e-4`
- 优先更多训练步骤（和/或更强的稳定化，如果适用）然后再增加秩

---

## 常见问题

您可以参考 [Chat SFT 指南](chat-sft.md)中的常见问题部分。我们将来也会添加更多与 RL 相关的问答。
