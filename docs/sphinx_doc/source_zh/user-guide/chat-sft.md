# Chat 监督微调（SFT）

本指南演示如何使用运行中的 TuFT 服务器对**聊天格式数据**进行**监督微调（SFT）**。完整的可运行代码在 `examples/chat_sft/chat_sft.ipynb` notebook 中。虽然这是一个通用的 SFT 指南，但它也记录了用户在使用 TuFT 进行 SFT 时可能遇到的常见问题，并提供逐步指导帮助他们成功完成端到端运行。

---

## 您将学到

1. 如何从 HuggingFace 加载**聊天数据集**并提取多轮 `messages`
2. 如何使用**模型聊天模板**格式化对话（`apply_chat_template`）
3. 如何实现**仅助手损失掩码**并计算用于评估的掩码负对数似然
4. 如何构建 `Datum` 对象并通过 TuFT 服务器运行端到端的 **LoRA SFT** 循环
5. 如何根据训练/测试曲线选择和调优 **LoRA rank** 和**学习率**

---

## 目录
1. [何时使用 SFT vs. RL](#何时使用-sft-vs-rl)
2. [数据集](#数据集)
3. [最小训练示例（SFT）](#最小训练示例sft)
4. [关键概念](#关键概念)
   - [聊天格式化与模板](#聊天格式化与模板)
   - [损失掩码（仅助手）](#损失掩码仅助手)
   - [Datum 格式](#datum-格式)
   - [损失函数](#损失函数)
5. [参数选择](#参数选择)
6. [常见问题](#常见问题)

---

## 何时使用 SFT vs. RL

### SFT vs. RL（高层对比）

| 主题 | SFT（监督微调） | RL（强化学习） |
|---|---|---|
| 训练信号 | 示范（目标响应） | 奖励/偏好（标量或排名） |
| 适用场景 | 风格、格式、指令遵循、基于精选答案的领域行为 | 将行为与偏好/约束对齐、安全策略、多目标权衡 |
| 所需数据 | 高质量的助手响应 | 奖励模型、偏好对或评估器 |
| 典型工作流 | 通常是第一阶段 | 通常在 SFT 之后（SFT → RL） |
| 训练数据/信号示例 | 输入输出对，如 prompt："改写成礼貌的邮件..." → target："尊敬的...敬上..." | LLM 评判：对 A vs B 排名或打分 |

**经验法则**
- 当您能提供好的"黄金"助手响应并希望模型模仿明确的目标输出时，使用 **SFT**。
- 当没有单一正确答案，但您可以通过奖励或偏好信号定义什么是"更好"时，使用 **RL**，通常基于任务要求如有用性、安全性、风格、格式或工具使用行为。

---

## 数据集

本指南使用 **[`no_robots`](https://huggingface.co/datasets/HuggingFaceH4/no_robots)**。

| 数据集 | 来源 | 大小 | 训练目标 | 用例 |
|---|---|---|---|---|
| `no_robots` | `HuggingFaceH4/no_robots` | 约 9.5K 训练 + 500 测试 | 所有助手消息（掩码） | 快速实验 |

最小加载器：
```python
from datasets import load_dataset

ds = load_dataset("HuggingFaceH4/no_robots")
train_data = [row["messages"] for row in ds["train"]]
test_data  = [row["messages"] for row in ds["test"]]
```

每个样本是一个聊天消息列表：
```text
{"role": "user" | "assistant", "content": "..."}
```

---

## 最小训练示例（SFT）

**TuFT**（Tenant-unified FineTuning，租户统一微调）是一个多租户系统，为大语言模型的微调提供统一的服务 API。它支持 Tinker 服务 API，可以与 Tinker SDK 一起使用。与 Tinker 不同，TuFT 可以在本地 GPU 上运行；以下实验在本地 **2× NVIDIA A100-SXM4-80GB** 设置上进行（驱动 550.54.15，CUDA 12.9）。运行示例前，请按照[安装指南](../getting-started/installation.md)在本地启动 TuFT 服务器。

关键 TuFT 调用（完整代码在 `examples/chat_sft/chat_sft.ipynb`）：
```python
import tinker
from tinker import types

service_client = tinker.ServiceClient(base_url="http://localhost:10610", api_key=TINKER_API_KEY)

training_client = service_client.create_lora_training_client(
    base_model=BASE_MODEL,
    rank=LORA_RANK,
    train_mlp=True,
    train_attn=True,
    train_unembed=True,
)

fwdbwd = training_client.forward_backward(datums, loss_fn="cross_entropy").result()
training_client.optim_step(types.AdamParams(learning_rate=LEARNING_RATE)).result()
```

---

## 关键概念

### 聊天格式化与模板

我们使用基础模型的聊天模板，使提示词遵循训练时看到的相同角色/标记格式。

```python
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
)
tokens = tokenizer.encode(text, add_special_tokens=False)
```

- **`tokenize=False`**：返回渲染的**文本**（字符串），而不是 token ID；我们在下一行显式进行分词。
- **`add_generation_prompt=False`**：不追加最终的"助手开始"标记；对于训练/编码现有轮次很有用。（对于推理，通常设置为 `True` 以提示模型生成下一个助手回复。）
- **`add_special_tokens=False`**：避免重复特殊 token，因为聊天模板已经包含了所需的标记。

### 损失掩码（仅助手）

对于聊天 SFT，我们通常希望模型学习生成**助手响应**，而不是预测**用户提示词**。因此我们构建每个 token 的权重：

- **助手**轮次的 token → `weight = 1.0`
- **用户**轮次的 token → `weight = 0.0`

因为训练是**下一个 token 预测**，掩码必须与**目标 token**（被预测的 token）对齐。如果我们为原始 token 流 `tokens[0..N-1]` 构建权重，那么步骤 `t` 的损失预测 `tokens[t+1]`，所以我们使用 `weights[1:]` 与 `target_tokens = tokens[1:]` 对齐。

```python
def build_sft_example(messages, tokenizer, max_length=2048):
    # 构建 token 流 + 每个 token 的权重（assistant=1, user=0）
    tokens, weights = [], []
    for msg in messages:
        turn_tokens = tokenizer.encode(msg["content"], add_special_tokens=False)
        tokens += turn_tokens
        weights += [1.0 if msg["role"] == "assistant" else 0.0] * len(turn_tokens)

    # 可选截断
    tokens, weights = tokens[:max_length], weights[:max_length]

    # 下一个 token 预测：input[t] -> target[t] = tokens[t+1]
    input_tokens  = tokens[:-1]
    target_tokens = tokens[1:]

    # 将掩码对齐到目标（被预测的 token）
    target_weights = weights[1:]

    return input_tokens, target_tokens, target_weights
```

### Datum 格式

每个对话被转换为下一个 token 预测样本：

- `model_input`：token `[0..T-2]`
- `target_tokens`：token `[1..T-1]`
- `weights`：应用于目标的掩码（仅助手）

示例：
```python
from tinker import types

datum = types.Datum(
    model_input=types.ModelInput.from_ints(input_tokens),
    loss_fn_inputs={
        "target_tokens": list(target_tokens),
        "weights": target_weights.tolist(),
    },
)
```

### 损失函数

训练使用以下损失函数：
- `loss_fn="cross_entropy"`

TuFT 返回每个 token 的对数概率（`logprobs`）。本指南计算**掩码负对数似然（NLL）**：

$$
\mathrm{NLL}=\frac{\sum_{t}\bigl(-\log p(y_t)\bigr)\,w_t}{\sum_{t} w_t}
$$

最小计算：
```python
def masked_nll(loss_fn_outputs, datums):
    total_loss, total_w = 0.0, 0.0
    for out, d in zip(loss_fn_outputs, datums):
        for lp, w in zip(out["logprobs"], d.loss_fn_inputs["weights"]):
            total_loss += -lp * w
            total_w += w
    return total_loss / max(total_w, 1.0)
```

---

## 参数选择

本节解释如何选择 `lora_rank` 和 `learning_rate`，并总结实验结果的结论。本文档基于 [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)。

### `lora_rank` 和 `learning_rate` 的作用

**`lora_rank`（LoRA 适配器秩）**控制适配器容量：
- 更高的秩 = 更多可训练参数 → 可能更好的拟合，更多计算/内存，更高的过拟合风险
- 更低的秩 = 更便宜，通常足以进行风格/小行为变化

**`learning_rate`**控制更新步长：
- 太高（如 `1e-3`）：快速但可能不稳定/过拟合
- 太低（如 `1e-5`）：稳定但慢
- 中等（如 `1e-4`）：LoRA SFT 的常见默认值

### 图表的实验结论

基于**图 1**（测试 NLL）和**图 2**（训练平均 NLL）：

1) 非常低的 LR（`1e-5`）收敛慢得多
2) `1e-4` 和 `1e-3` 早期改进很快
3) 秩的收益递减
4) 最佳测试损失通常聚集在中等秩 + 中等/高 LR 附近

> 注意：确切的"最佳"取决于停止步骤和下游生成质量（不仅是 NLL）。

```{figure} ../../_static/images/test_nll_sft.png
:alt: 测试 NLL
:width: 720px
:align: center

**图 1. 测试 NLL**
```

```{figure} ../../_static/images/train_mean_nll_sft.png
:alt: 训练平均 NLL
:width: 720px
:align: center

**图 2. 训练平均 NLL**
```

### 实用建议

- 强默认值：`lora_rank = 8 或 32`，`learning_rate = 1e-4`
- 更快的早期进展（风险更高）：`lora_rank = 8 或 32`，`learning_rate = 1e-3`
- 如果不稳定/过拟合：降低 LR（`1e-4 → 5e-5 → 1e-5`）或降低秩（`32 → 8`）
- 如果任务更难：先尝试 `32` 再尝试 `128`，保持 LR `1e-4`，尽可能先增加步数再增加秩。"更难"意味着学习问题本身更困难（更复杂的输入→输出映射），如更严格的输出约束/格式、更长的上下文、更多推理步骤或更高的输出多样性/歧义性。它不简单意味着"更多数据"；更多数据通常只需要更多训练步骤，而不是更高的 LoRA 秩。

---

## 常见问题

### (1) 由于访问 huggingface.co 的网络问题导致数据集下载失败

如果您看到类似错误：
```
MaxRetryError('HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded ...
(Caused by NewConnectionError(... [Errno 101] Network is unreachable))')
```

对于 Jupyter notebook 用户，在**第一个单元格的最顶部**添加以下内容：
```python
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
```

然后**重启内核并清除所有输出**。

---

### (2) `invalid Api_key`

在 Tinker SDK 中，环境变量 `TINKER_API_KEY` 优先于这里传递的 `api_key=` 参数：
```python
service_client = tinker.ServiceClient(base_url=TINKER_BASE_URL, api_key=TINKER_API_KEY)
```

所以如果您的代码传递了正确的密钥但仍然得到 `invalid api_key`，您需要设置正确的环境变量（通过 `export TINKER_API_KEY=...`）或清除它并依赖 `api_key=` 参数：
```bash
unset TINKER_API_KEY
```

---

### (3) Jupyter 警告：`TqdmWarning: IProgress not found...`

如果您看到：
```
TqdmWarning: IProgress not found. Please update jupyter and ipywidgets.
```

**选项 A（推荐）：安装/升级 Jupyter widgets**
```bash
pip install -U ipywidgets jupyter
```
然后重启内核。

**选项 B：避免在 notebook 中使用基于 widget 的 tqdm**
使用标准的 `tqdm` 进度条而不是 `tqdm.auto` / `tqdm.notebook`：
```python
from tqdm import tqdm
```

---

### (4) OOM 或训练缓慢

如果遇到内存不足（OOM）错误或训练太慢，减少以下一个或多个：
- `MAX_LENGTH`
- `BATCH_SIZE`
- `LORA_RANK`

在大多数情况下，降低 `MAX_LENGTH` 带来最大的内存/速度改进，其次是 `BATCH_SIZE`，然后是 `LORA_RANK`。

### (5) 将虚拟环境添加到 Jupyter（注册新内核）

如果您在远程服务器上工作，将现有的虚拟环境（virtualenv/venv）添加为可选的 Jupyter 内核通常很方便。

1) **激活虚拟环境**
```bash
source /path/to/venv/bin/activate
```

2) **在环境中安装 `ipykernel`**
```bash
pip install ipykernel
```

3) **将环境注册为 Jupyter 内核**
```bash
python -m ipykernel install --user --name=myproject --display-name "Python (myproject)"
```

4) **在 Jupyter 中选择内核**
- 在 Jupyter Notebook/Lab 中：**Kernel → Change Kernel → Python (myproject)**
