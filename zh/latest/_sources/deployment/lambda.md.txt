# 在 Lambda Cloud 上部署

[Lambda Cloud](https://lambda.ai) 提供普通的按需 GPU 虚拟机，按分钟计费，直到你终止它们——没有编排层，也没有抢占。本指南将带你完成端到端的流程——配置一个 TuFT 服务器，在 Lambda GPU 上启动它，在本地（笔记本）上基于 `Qwen/Qwen3-0.6B` 训练一个"像 Yoda 一样说话"的 LoRA，并下载该适配器——**无需本地 GPU**。

```{admonition} 你将构建的内容
:class: note

一个运行在 Lambda GPU 虚拟机上的 TuFT 服务器（通过 Docker 自动置备并自我引导），你可以通过 SSH 隧道连接它，并使用 [`examples/personality_sft/`](https://github.com/agentscope-ai/TuFT/tree/main/examples/personality_sft) 中的可运行代码在本地（笔记本）上对其进行微调。
```

```{admonition} Lambda 没有缩容至零
:class: warning

与 Modal 不同，Lambda 虚拟机会**持续计费，直到你终止它**——没有空闲/自动停止。完成后务必销毁（终止）实例（参见 [第 5 步](#step-5-tear-down)）。
```

## 准备工作

1. **一个 Lambda Cloud 账户、API key 和 SSH key。** 在
   [Lambda Cloud 控制台](https://cloud.lambda.ai)中，生成一个 API key
   （[API keys](https://cloud.lambda.ai/api-keys)）并注册一个 SSH 公钥
   （[SSH keys](https://cloud.lambda.ai/ssh-keys)）——你将使用匹配的私钥来连接服务器。详情请参阅 Lambda 的[文档](https://docs.lambda.ai)。导出 API key，以便启动器能够找到它：

   ```bash
   export LAMBDA_API_KEY=secret_...
   ```

2. **TuFT 仓库。** 部署辅助脚本与 Lambda 的 HTTP API 通信；它在本地只需要 `pyyaml`（所有 GPU 依赖都在虚拟机上的容器内运行）：

   ```bash
   git clone https://github.com/agentscope-ai/TuFT
   cd TuFT
   pip install pyyaml
   ```

## 第 1 步 —— 配置服务器

部署辅助脚本 [`deploy/lambda/launch.py`](https://github.com/agentscope-ai/TuFT/tree/main/deploy/lambda/launch.py)
是**配置文件驱动**的，并与 Modal 启动器保持一致：你编辑一个标准的
`tuft_config.yaml` 并运行该脚本。Lambda 基础设施配置放在一个可选的 `lambda:` 小节中，该小节会在服务器看到之前被剥离。

将以下内容保存为 `yoda_lambda.yaml`：

```yaml
checkpoint_dir: ~/.cache/tuft/checkpoints   # mapped to the VM's /data; see Step 4 about durability
model_owner: cloud-user

supported_models:
  - model_name: Qwen/Qwen3-0.6B
    model_path: Qwen/Qwen3-0.6B            # HF id (downloaded on first launch) or a local path
    max_model_len: 4096
    tensor_parallel_size: 1
    colocate: true                         # single GPU: training + vLLM sampling share it
    sampling_memory_fraction: 0.4
    max_lora_rank: 16
    max_loras: 2

authorized_users:
  tml-REPLACE_WITH_A_STRONG_KEY: cloud-user  # clients send this as the X-API-Key header

persistence:
  mode: DISABLE

telemetry:
  enabled: false

# Lambda Cloud infra for deploy/lambda/launch.py (TuFT ignores this; it's stripped before the server sees it):
lambda:
  gpu: a100              # family hint; auto-pick prefers a100
  name: tuft-yoda
  # ssh_key: my-key      # default: your account's sole registered key
  # filesystem: tuft     # Lambda persistent filesystem for durable checkpoints (else ephemeral root disk)
```

为 `authorized_users` 生成一个真实的 API key（它**必须**以 `tml-` 开头）：

```bash
python -c "import secrets; print('tml-' + secrets.token_urlsafe(24))"
```

```{admonition} 训练请选择 a100，而不是 a10
:class: warning

最便宜的 Lambda GPU（`gpu_1x_a10`，sm_86）在当前的 TuFT 镜像中有一个已知问题：
**服务可以正常工作，但训练会返回 null logprobs**。因此自动选择会**优先选择
a100**，仅在万不得已时才使用 a10。对于本训练示例，请坚持使用 a100。
```

## 第 2 步 —— 启动 GPU 服务器

运行启动器。在未指定实例的情况下，它会自动选择与你的 `gpu:` 提示相匹配、可用且最便宜的 GPU，对其进行置备，并通过 cloud-init 在 Docker 中自我引导 TuFT（无需手动 SSH）：

```bash
python deploy/lambda/launch.py --config yoda_lambda.yaml
```

它会打印所选择的实例，并在虚拟机启动后打印一条连接横幅，其中包含一条 SSH 隧道命令和实例 id：

```text
[launch] gpu_1x_a100_sxm4 in us-east-1 (~$1.99/hr), ssh_key=my-key, name=tuft-yoda
[launch] provisioning instance abcd...  (this takes a minute)
[launch] instance abcd1234... is active at 203.0.113.45

Connect securely over an SSH tunnel (recommended; keeps :10610 off the public net):
    ssh -N -L 10610:localhost:10610 ubuntu@203.0.113.45
```

```{tip}
你可以随时使用 `python deploy/lambda/launch.py --status` 检查或列出实例。要复用某个现有实例而不是启动新实例，请传入 `--instance-id <id>`。
```

## 第 3 步 —— 在本地（笔记本）上训练 Yoda LoRA

在一个终端中打开上面打印的 SSH 隧道（首次启动会拉取镜像并加载 vLLM，这可能需要几分钟）：

```bash
ssh -N -L 10610:localhost:10610 ubuntu@203.0.113.45
```

```{note}
训练仍然运行**在你的笔记本上**——`train.py` 是一个仅使用 CPU 的客户端，它通过 HTTP 驱动训练循环。`-L 10610:localhost:10610` 标志会把你笔记本的 `10610` 端口转发到虚拟机上服务器的 `10610` 端口，因此 `http://localhost:10610`（下面会用到）只是隧道的*本地端*：每个请求都会通过 SSH 传送到**远程 GPU**，训练和采样实际就在那里运行。请在整个运行过程中保持这个 SSH 终端打开。（不想使用隧道？你也可以直接将 `--base-url` 指向 `http://<vm-ip>:10610`，但这会把 API 暴露在公共互联网上。）
```

在第二个终端中，通过隧道确认服务器健康：

```bash
curl http://localhost:10610/api/v1/healthz
# {"status":"ok"}
```

训练脚本 [`examples/personality_sft/train.py`](https://github.com/agentscope-ai/TuFT/tree/main/examples/personality_sft/train.py)
是一个**客户端**，它通过 Tinker SDK 经由 HTTP 驱动训练循环——它只需要 CPU 端的依赖：

```bash
pip install tinker transformers
```

数据集是
[`dataset.py`](https://github.com/agentscope-ai/TuFT/tree/main/examples/personality_sft/dataset.py)
中约 50 对手工编写的 `(user, assistant-in-Yoda-voice)` 配对。只有**助手**的 token 获得损失权重（提示词被掩码），因此模型学习的是*语气*，而不是问题本身。这里给出几个示例配对：

```python
YODA_PAIRS = [
    (
        "How do I stay motivated?",
        "Motivated, you wish to stay? Hmm. Small steps each day, take you must. "
        "Begin, and momentum follows. Do or do not — there is no try.",
    ),
    (
        "What's the best way to learn programming?",
        "Patient you must be, young one. Write code daily, you should. "
        "Fail often, learn faster you will. Much to learn, you still have.",
    ),
    # ... ~50 pairs total
]
```

将脚本指向隧道（`http://localhost:10610`），传入 `tml-` key，并选择模型（它必须与服务器上的 `supported_models` 条目相匹配）：

```bash
python examples/personality_sft/train.py \
    --base-url http://localhost:10610 \
    --api-key tml-REPLACE_WITH_A_STRONG_KEY \
    --model Qwen/Qwen3-0.6B
```

脚本会在训练**之前**对基础模型采样，运行 60 个 LoRA 步骤，然后在训练**之后**对训练好的适配器采样，这样你就能看到个性逐渐显现：

```text
[before] sampling base model on held-out prompts
   [base] Give me advice on becoming a better person.
      -> Focus on small consistent habits: be honest, listen to others, and learn from your mistakes.
   [base] What's the key to writing good software?
      -> Keep it readable and well-tested, prefer small functions, and document your intent.

[train] 60 steps, batch 8, lr 0.0001, rank 16
   step   0  loss=2.3987
   step  10  loss=1.7765
   step  50  loss=1.2410
   step  59  loss=1.1702

[save] sampler=tinker://abcd1234.../yoda-sampler
[save] run_id=abcd1234-...

[after] sampling the trained adapter
   [yoda] Give me advice on becoming a better person.
      -> Small steps each day, take you must. Honest, be. Listen more than speak, you should. Grow, you will, hmm.
   [yoda] What's the key to writing good software?
      -> Simple, keep it, you must. Read aloud your code, you should. Fail often, learn faster you will, young one.
```

```{tip}
可调参数：`--lora-rank`、`--num-steps`、`--batch-size`、`--learning-rate`、`--max-length`。
传入 `--no-before` 可跳过基础模型采样。编辑 `dataset.py` 即可换成不同的角色。
```

## 第 4 步 —— 下载适配器

训练会向服务器的 `checkpoint_dir` 写入一个标准的 [PEFT](https://huggingface.co/docs/peft) LoRA 适配器，在虚拟机上它位于 `/home/ubuntu/tuft-data/checkpoints` 下。脚本会打印一个 `run_id`——用 `scp` 把适配器从实例上拷贝下来：

```bash
scp -r ubuntu@203.0.113.45:/home/ubuntu/tuft-data/checkpoints/<run_id> ./weights/
# -> ./weights/<run_id>/yoda-final/adapter/{adapter_config.json, adapter_model.safetensors}
```

```{admonition} 在终止之前先下载
:class: warning

默认情况下，检查点位于实例的**临时根磁盘**上——终止虚拟机（第 5 步）会销毁它们。请**先下载**，或者使用持久化的 `filesystem:`（一个 [Lambda filesystem](https://docs.lambda.ai)）启动，这样检查点就能在终止后保留下来。
```

可选：将适配器合并到完整的模型权重中（需要 `torch`、`peft`、`transformers`）：

```python
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype=torch.bfloat16)
merged = PeftModel.from_pretrained(base, "./weights/<run_id>/yoda-final/adapter").merge_and_unload()
merged.save_pretrained("./yoda-merged")   # standard HF model dir, servable by vLLM
```

(step-5-tear-down)=
## 第 5 步 —— 销毁（终止）

Lambda 会一直计费直到你终止，所以完成后请务必关闭实例：

```bash
python deploy/lambda/launch.py --down --instance-id abcd1234...
# or by name:
python deploy/lambda/launch.py --down --name tuft-yoda
```

确认没有任何东西仍在运行：

```bash
python deploy/lambda/launch.py --status
```

```{tip}
Lambda 会持续对 GPU 计费，因此单个由笔记本驱动的运行**会为大量空闲时间付费**。由于 TuFT 是多租户的，你可以将**多个并发作业或用户**指向同一个实例（每个用户在 `authorized_users` 下都有一个 key；提高 `max_loras` 可同时支持更多适配器），以保持 GPU 繁忙并分摊成本。参见
{ref}`保持 GPU 繁忙 <keeping-the-gpu-busy>`。
```

## 后续步骤

- 通过更改配置中的两处名称来尝试更大的模型（例如 `Qwen/Qwen3-4B`）；自动选择的 a100（80 GB）有充裕的空间。
- 更喜欢缩容至零的方案？参见 [Modal 指南](modal.md)。
- 浏览 [Lambda 的文档](https://docs.lambda.ai)，了解实例类型、区域和文件系统。
