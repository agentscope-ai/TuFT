# 在策略蒸馏（OPD）

本指南演示如何在**运行中的 TuFT 服务器**上完成**在策略蒸馏（On-Policy Distillation，OPD）**——让*学生*模型在**自己采样的轨迹上**逐 token 对齐*教师*模型的输出分布。完整可运行代码位于 [`examples/on_policy_distillation/`](https://github.com/agentscope-ai/TuFT/tree/main/examples/on_policy_distillation)（`train.py`、`dataset.py`、`config.yaml`）。这既是一份通用的 OPD 指南，也记录了在 TuFT 上端到端跑通所需的实践细节（教师对数概率、反向 KL 优势、单卡与多卡部署）。

OPD 采用 Thinking Machines 博客 [*On-Policy Distillation*](https://thinkingmachines.ai/blog/on-policy-distillation/) 中的配方。它兼具蒸馏的**逐 token 密集监督**（每个 token 都有训练信号，而 RL 整条轨迹只有一个标量奖励）与**在策略采样**（学生在自己的轨迹上学习，避免了离策略 SFT 的误差累积）。在 TuFT 上，它复用**与 RL 完全相同的机制**：`importance_sampling` 损失，只是把逐 token 优势设为教师与学生的对数概率之差。

---

## 您将学到

1. 什么是**在策略蒸馏**，以及什么情况下应选择它而非 **SFT** 或 **RL**
2. 如何用 `compute_logprobs` 获取**教师模型**对学生 token 的**逐 token 对数概率**
3. **逐 token 反向 KL 优势**（`teacher_logprob − student_logprob`）如何对接 `importance_sampling` 损失
4. 如何在 TuFT 上运行端到端的 OPD 循环（采样 → 打分 → 计算优势 → `forward_backward` → `optim_step`）
5. 如何部署教师和学生，以及为什么 OPD 的收益来自真实的**能力差距**，而不只是模型规模

---

## 目录
1. [何时选择 OPD 而非 SFT 或 RL](#何时选择-opd-而非-sft-或-rl)
2. [任务设置](#任务设置)
3. [最小训练示例（OPD）](#最小训练示例opd)
4. [关键概念](#关键概念)
   - [OPD 训练循环](#opd-训练循环)
   - [通过 compute_logprobs 获取教师对数概率](#通过-compute_logprobs-获取教师对数概率)
   - [逐 token 反向 KL 优势](#逐-token-反向-kl-优势)
   - [OPD 的 Datum 格式](#opd-的-datum-格式)
   - [部署方式](#部署方式)
5. [实验结果](#实验结果)
6. [参数选择](#参数选择)
7. [常见问题](#常见问题)

---

## 何时选择 OPD 而非 SFT 或 RL

| 对比维度 | SFT | RL | 在策略蒸馏 |
|---|---|---|---|
| 训练信号 | 标准答案 token | 每条轨迹一个标量奖励 | 教师在**每个** token 上的完整分布 |
| 数据来源 | 固定数据集（离策略） | 学生自身采样（在策略） | 学生自身采样（在策略） |
| 前置条件 | 精心整理的答案 | 奖励函数 / 验证器 | 一个更强（或提示词更充分）的**教师** |
| 每条序列的反馈量 | O(N)，但离策略 | O(1) | **O(N)，且在策略** |
| 规避的失败模式 | — | 稀疏、缓慢的信用分配 | SFT 在未见状态上的误差累积 |

**经验法则：** 当你手上有一个**信得过的教师**（更大的模型，或同一模型配上更充分的提示词），并希望学生**低成本地**内化其行为时，就该考虑 OPD——据报告，OPD 能以远低于 RL 的算力达到 RL 级别的效果，因为每个 token 都携带训练信号。它天然适合放在 SFT *之后*使用；在教师可用时，也是 RL 的低成本替代方案。

---

## 任务设置

我们使用 **[GSM8K](https://huggingface.co/datasets/openai/gsm8k)** 小学数学应用题。

| 角色 | 看到的提示词 | 行为 |
|---|---|---|
| **教师** | 系统指令 **+ 4 个带完整解答的 few-shot 示例** + 题目 | 逐步推理，以 `#### <数字>` 结尾 |
| **学生** | 仅题目本身 | 通过 OPD 训练以对齐教师 |

在默认配置中，教师**就是学生的基础模型**——它唯一的优势是 few-shot 提示词。OPD 把这份*上下文*蒸馏进学生的 LoRA：训练完成后，学生**不需要提示词里的任何示例**，就能以教师级别的质量和简洁度进行推理。这就是「上下文蒸馏」（context distillation）；由于全程只有一份基础模型权重，**一张普通 GPU** 就能装下。

两份提示词唯一的差别是 few-shot 部分（`dataset.py`）：

```python
def student_prompt_ids(tokenizer, question):           # 仅题目
    messages = [{"role": "system", "content": STUDENT_SYSTEM},
                {"role": "user", "content": question}]
    return _render_ids(tokenizer, messages)

def teacher_prompt_ids(tokenizer, question):           # 附加 few-shot 完整解答示例
    messages = [{"role": "system", "content": TEACHER_SYSTEM}]
    for q, a in FEWSHOT:
        messages += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    messages.append({"role": "user", "content": question})
    return _render_ids(tokenizer, messages)
```

---

## 最小训练示例（OPD）

以下实验在 [Modal](../deployment/modal.md) 上的**单张 NVIDIA A100-40GB** 上完成（教师与学生共用基础模型，`colocate: true`；24GB 的 L4 也能跑，只需调小 batch/group）。运行前，先用 [`examples/on_policy_distillation/config.yaml`](https://github.com/agentscope-ai/TuFT/blob/main/examples/on_policy_distillation/config.yaml) 启动一台服务器。

整个循环只用到三个你在 RL 指南中已经熟悉的 TuFT 调用——`sample`、`compute_logprobs` 和 `forward_backward(..., loss_fn="importance_sampling")`：

```python
import tinker
from tinker import types

client = tinker.ServiceClient(base_url="http://localhost:10610", api_key=TINKER_API_KEY)

# 教师 = 基础模型 + few-shot 提示词（冻结；只负责打分）。
teacher = client.create_sampling_client(base_model=BASE_MODEL)

# 学生 = 我们要训练的 LoRA。
training = client.create_lora_training_client(
    base_model=BASE_MODEL, rank=LORA_RANK,
    train_mlp=True, train_attn=True, train_unembed=False,
)
```

---

## 关键概念

### OPD 训练循环

每一步都完全**在策略**：先把最新的学生权重同步到采样器，让学生用不含示例的提示词自行作答，再让教师对这些一模一样的 token 打分。

```python
for step in range(NUM_STEPS):
    # 1. 在策略：从当前学生采样。
    student_path = training.save_weights_for_sampler(name=f"opd-{step}").result().path
    student_sampler = client.create_sampling_client(model_path=student_path)

    datums = []
    for problem in batch:
        student_prompt = types.ModelInput.from_ints(student_prompt_ids(tok, problem.question))
        res = student_sampler.sample(prompt=student_prompt, num_samples=GROUP,
                                     sampling_params=types.SamplingParams(max_tokens=256, temperature=1.0)).result()

        teacher_ids = teacher_prompt_ids(tok, problem.question)
        for seq in res.sequences:
            answer, student_lp = list(seq.tokens), list(seq.logprobs)

            # 2. 教师对学生采出的原样 token 打分（见下一节）。
            teacher_input = types.ModelInput.from_ints(teacher_ids + answer)
            tlp = teacher.compute_logprobs(teacher_input).result()
            p = len(teacher_ids)

            # 3. 逐 token 反向 KL 优势。
            adv = [float(tlp[p + j]) - student_lp[j] for j in range(len(answer))]

            datums.append(build_distillation_datum(
                prompt=student_prompt, answer_tokens=answer,
                sampling_logprobs=student_lp, advantages=adv))

    # 4. 一次在策略更新。
    training.forward_backward(datums, loss_fn="importance_sampling").result()
    training.optim_step(types.AdamParams(learning_rate=1e-4)).result()
```

### 通过 compute_logprobs 获取教师对数概率

`compute_logprobs(prompt)` 为提示词中的每个 token 返回一个该模型下的对数概率——`lp[i] = log P(token_i | token_0..i-1)`。要让**教师**给**学生**的回答打分，只需把教师的上下文与学生采出的 token ID 拼接起来（直接复用 ID，不重新分词），再切出回答区间：

```text
teacher_input = [ 教师上下文（P 个 token） ][ 学生回答（T 个 token） ]
                                             ^
teacher_logprob[j] = compute_logprobs(teacher_input)[P + j]      # 与 student_logprob[j] 一一对应
```

位置 `P + j` 上的值，就是教师在自己的 few-shot 上下文下、对学生第 `j` 个回答 token 给出的对数概率——它与采样时返回的 `seq.logprobs[j]`（学生自己对同一 token 的对数概率）逐位对齐。

### 逐 token 反向 KL 优势

令 `advantage[t] = teacher_logprob[t] − student_logprob[t]`，这正是**负的逐 token 反向 KL** `−KL(student ‖ teacher)` 的单样本估计：在学生采样分布的期望下，它等于该位置上 `teacher − student` 的平均，即 `−D_KL`。把它代入 `importance_sampling` 损失

```text
loss = − Σ_t  exp(target_logprob_t − sampling_logprob_t) · advantage_t
```

其梯度（刚采样完就训练时，比值 ≈ 1）就是 REINFORCE 更新 `−Σ_t advantage_t · ∇ log π_student(a_t)`：教师比学生更青睐的 token，概率被**推高**；其余 token 被**压低**——学生因此逐步向教师靠拢。这与 [Countdown RL 指南](countdown-rl.md)用的是同一个损失，唯一的区别在于优势的定义（逐 token 的 KL，而非奖励）。

### OPD 的 Datum 格式

`Datum` 的构造与 RL 示例完全一致，只是 `advantages` 是**逐 token 的列表**（KL 信号），而不是每条轨迹一个标量。提示词区间的优势置零，只有回答 token 参与训练（`dataset.py: build_distillation_datum`）：

```python
model_input = prompt.append(types.EncodedTextChunk(tokens=answer_tokens[:-1]))
ob_len = prompt.length - 1                       # 提示词区间：不参与损失

loss_fn_inputs = {
    "target_tokens": [0]*ob_len + answer_tokens,           # 学生需要预测的 token
    "logprobs":      [0.0]*ob_len + sampling_logprobs,     # 学生采样时自己的对数概率
    "advantages":    [0.0]*ob_len + advantages,            # 逐 token 的 teacher_lp − student_lp
}
```

服务器负责计算 `target_tokens` 在当前策略下的可微对数概率；`logprobs` 是固定不变的采样对数概率；`advantages` 承载蒸馏信号。

### 部署方式

由于教师与学生共用**同一个基础模型**，用 `colocate: true`（训练与 vLLM 共卡）就能在**一张 GPU** 上跑完全程。若教师是*另一个*更大的模型，可以放在独立的 TuFT 服务器上——用 `--teacher-model` / `--teacher-base-url` / `--teacher-api-key` 把 `train.py` 指向它。

不过在换更大的教师之前，值得先了解：**OPD 迁移的是能力本身，而不只是风格**——真正带来收益的，是教师与学生之间实打实的*逐 token* 能力差距，而不是模型大小。任务不难时，小幅加大教师意义有限（同一家族的两个强模型逐 token 的预测大多一致；我们在 GSM8K 上试过 8B→1.7B 的师生组合，效果并不比这里的同模型 few-shot 方案好）。只有当差距既大又与能力直接相关时，更大的教师才有帮助——比如学生弱得多，或者任务换成 MATH/AIME 这类小模型真正吃力的难题。而本文的同模型 few-shot 方案，正是一种可靠又便宜的「制造差距」方式。

---

## 实验结果

Qwen3-1.7B 学生模型，单张 A100-40GB，**16 步**（batch 8 × group 4），`lr=1e-4`，rank 16，在 100 道留出的 GSM8K 题上贪心解码评测：

| | 准确率 | 提示词内容 | 推理风格 |
|---|---|---|---|
| 训练前的学生（无示例） | **63%** | 仅题目 | 冗长、爱用标题、很少有干净的结论行 |
| OPD 之后（仍无示例） | **72%** | 仅题目 | 教师式的简洁分步推理 |
| few-shot 教师（上限） | 72% | 附 4 个完整示例 | 简洁分步，以 `#### <数字>` 结尾 |

变化一目了然：在**提示词不含任何 few-shot 示例**的情况下，学生的留出集准确率从 **63% 跃升至 72%，追平了 few-shot 教师**——教师的推理方式已经写进了学生的 LoRA。输出文本也明显更简洁（贴近教师的解题风格），不再像原始模型那样绕来绕去。整次训练大约只花 **1 美元的 A100-40GB 机时**。

下面是一道留出题：训练前的学生答**错**，训练后的学生答**对**——同样的无示例提示词，贪心解码：

```text
Susan earns $5 every 10 minutes for an online task she does. If she works between 8 a.m. and 11 a.m. and
pauses in between for half an hour, how much money does she earn?  (answer: $75)
```

**OPD 之前**——学生把半小时的暂停*列了出来*，随后却忘了减掉，自信地给出错误总额：

```text
### Given:
- Susan earns $5 every 10 minutes.
- She works between 8 a.m. and 11 a.m.
- She pauses for half an hour in between.       ← 列出来了，之后再没用上

### Step 1: total time = 11 − 8 = 3 hours = 180 minutes
### Step 2: 180 / 10 = 18 intervals
### Step 3: 18 × $5 = $90

Final Answer: 90
```

**OPD 之后**——它把暂停算了进去，用教师式的简洁风格得出正确答案：

```text
From 8 a.m. to 11 a.m. is 3 hours. She pauses half an hour, so she works 3 − 0.5 = 2.5 hours = 150 minutes.
In 10 minutes she earns $5, so 150 / 10 = 15 intervals.
15 × $5 = $75.
```

这并非个例：在整个留出集上，从错变对的题远多于从对变错的（本次运行为 **13 比 6**，净差即上表的准确率提升）。原始模型的典型错法正如上例——铺排一大段格式，最后丢掉一个约束，或在推导中途耗尽 token 预算——OPD 让它学会教师那种简短而完整的算术。

即便是原始模型**本来就能做对的题**，输出同样被收紧——只是不再啰嗦。相同提示词、相同答案（142），训练前后对比：

```text
Ricardo grows tomatoes and eggplants in his garden. Each tomato plant yields 22 tomatoes while each plant
of eggplant yields 4 eggplants. He planted 5 tomato plants and 8 plants of eggplant. How many fruits can
Ricardo get from his plants?  (answer: 142)
```

**OPD 之前**——先是 markdown 标题和一堆套话；256 个 token 过去了，还停留在*铺垫*阶段：

```text
Let's break down the problem step by step:

### Given:
- Each **tomato plant** yields **22 tomatoes**.
- Each **eggplant plant** yields **4 eggplants**.
- Ricardo planted:
  - 5 **tomato plants**
  - 8 **eggplant plants**

---

### Step 1:                       ←（到这里还只是在复述题目）
```

**OPD 之后**——直奔算术，风格与教师一致：

```text
Ricardo's tomato plants: 5 plants × 22 tomatoes = 110 tomatoes
Ricardo's eggplant plants: 8 plants × 4 eggplants = 32 eggplants
Total fruits: 110 + 32 = 142

Ricardo can get 142 fruits from his plants.
```

更多真实输出（包括其余从错变对的题）见 [`examples/on_policy_distillation/sample_outputs.md`](https://github.com/agentscope-ai/TuFT/blob/main/examples/on_policy_distillation/sample_outputs.md)。

OPD 见效**很快**——得益于逐 token 的密集信号，大部分提升在最初几步内就已完成。反向 KL 具有模式寻找（mode-seeking）特性，训练过久会过头（几十步之后准确率反而回落），所以整个运行保持短小；`--num-steps 16` 是不错的默认值（参见[参数选择](#参数选择)）。

---

## 参数选择

- **`learning_rate`** —— `1e-4` 是稳妥的默认值（与 RL 示例相同）。反向 KL 比交叉熵更敏感；若损失出现尖峰或学生输出退化，将其减半。
- **`kl_coef` / `adv_clip`** —— 分别对逐 token 优势做缩放和裁剪。保持 `kl_coef=1.0` 即可；`adv_clip`（默认 `10`）只是为了防范偶发的巨大对数概率差。
- **`group_size`** —— 每道题采样的学生回答数。样本越多，每步的信号越密集、方差越低（教师打分的调用也越多）。`4` 是不错的平衡点；追求速度可降到 `1–2`。
- **`temperature`** —— rollout 采样用 `~1.0`，让学生充分探索，教师才有纠正的空间；评测时用贪心（`0.0`）。
- **`lora_rank`** —— 对这类行为迁移来说 `16` 已绰绰有余；只有面对更困难的能力差距时才考虑加到 `32–64`。
- **`num_steps`** —— OPD 见效快，大部分提升集中在前几步。反向 KL 训练过久会过头（继续训练准确率反而回落），因此保持短小——`16` 是不错的默认值；脚本会按 `--eval-every` 周期性打印准确率，方便你观察峰值。
- **教师选择** —— 收益取决于*逐 token 的能力差距*，而非模型大小（参见[部署方式](#部署方式)）。

---

## 常见问题

### （1）为什么用 `importance_sampling`，而不是专门的「蒸馏」损失？

因为 OPD *本质上就是*以逐 token KL 为「奖励」的策略梯度。`importance_sampling` 已经在计算 `−Σ exp(target_lp − sampling_lp) · advantage`；令 `advantage = teacher_lp − student_lp`，它就成了在策略蒸馏更新。不需要任何新的损失函数——而且当采样器与训练器的对数概率出现漂移时，你还免费获得 RL 式的重要性修正。

### （2）教师和学生的提示词不同——KL 还成立吗？

成立。学生和教师各自以自己的上下文为条件，在**同一段回答 token** 上定义了两个分布。反向 KL 衡量的正是每个回答位置上这两个条件分布的差异；将其最小化，就是让无示例提示的学生表现得像有 few-shot 提示的教师——这恰恰是上下文蒸馏的目标。

### （3）准确率上去了，但学生还是不输出教师的 `####` 结尾行——为什么？

在策略蒸馏只能调整学生**实际采到的** token 的概率。无示例的 Qwen3 学生推理得不错，却从不探索字面的 `####` 标记，教师因此没有机会在学生的轨迹上为它打分（并强化它）。真正被迁移的是教师在学生已产出 token 上的**推理质量与简洁度**——这正是准确率提升的来源。这是在策略方法的一般性质：它们锐化并重新加权已探索的行为，不会凭空移植未探索的 token。（若想同时迁移表面格式，可以把该格式写进学生提示词促使其被探索，或先用少量 SFT 预热。）

### （4）`compute_logprobs` 在某些位置返回 `None`

第一个 token 没有前文，对数概率为 `None`；示例代码把所有 `None` 当作该 token 的零优势处理。由于我们始终切取的是**回答**区间（`P + j` 且 `P ≥ 1`），这只是防御性处理。

### （5）数据集下载失败（无法访问 huggingface.co）

在导入 `datasets` 之前设置镜像：
```python
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
```

### （6）小显存 GPU 上 OOM 或速度过慢

按顺序调低 `--max-new-tokens`、`--group-size`、`--batch-size`，并/或调小 `config.yaml` 中的 `sampling_memory_fraction` / `sampling_max_model_len`。24GB 的 L4 用 `--batch-size 4 --group-size 2` 即可运行；让教师与学生共用基础模型（默认配置），GPU 上就只需常驻一份模型。
