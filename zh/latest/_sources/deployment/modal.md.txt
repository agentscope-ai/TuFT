# 在 Modal 上部署

[Modal](https://modal.com) 是一个无服务器（Serverless）GPU 平台：你部署一个容器，它仅在有请求处理时才运行在 GPU 上，空闲时缩容至零。本指南将带你从头到尾完成整个流程——配置一个 TuFT 服务器、将其部署到 Modal、在本地（笔记本）驱动训练，在 `Qwen/Qwen3-0.6B` 上训练一个"像尤达大师那样说话"的 LoRA，并下载该适配器——而且**无需本地 GPU**。

```{admonition} 你将构建的内容
:class: note

一个运行在 Modal L4 GPU 上的 TuFT 服务器，可通过一个公开的 `…modal.run` URL 访问；你将使用
[`examples/personality_sft/`](https://github.com/agentscope-ai/TuFT/tree/main/examples/personality_sft)
中的可运行代码在本地（笔记本）对其进行微调。
```

## 准备工作

1. **一个 Modal 账户 + CLI。** 安装客户端并完成身份验证（这会打开浏览器以创建一个 API token——参见 Modal 的
   [Getting started](https://modal.com/docs/guide) 和
   [`modal token`](https://modal.com/docs/reference/cli/token) 文档）：

   ```bash
   pip install modal
   modal token new
   ```

2. **TuFT 仓库。** 部署辅助脚本和训练示例都位于该仓库中；启动器本身在本地只需要 `modal` + `pyyaml`（繁重的 GPU 依赖运行在 Modal 的容器内部，而不是你的机器上）：

   ```bash
   git clone https://github.com/agentscope-ai/TuFT
   cd TuFT
   pip install modal pyyaml
   ```

`Qwen/Qwen3-0.6B` 是可公开下载的，因此不需要 Hugging Face token。对于**受限（gated）**模型，请创建一个保存有你的
`HF_TOKEN` 的 [Modal Secret](https://modal.com/docs/guide/secrets)，并向启动器传入 `--hf-secret <secret-name>`。

## 第 1 步 —— 配置服务器

部署辅助脚本 [`deploy/modal/launch.py`](https://github.com/agentscope-ai/TuFT/tree/main/deploy/modal/launch.py)
是**由配置文件驱动**的：你编辑一个标准的 `tuft_config.yaml`（即 `tuft launch --config` 所使用的同一个文件）并运行该脚本。Modal 基础设施配置放在一个可选的 `modal:`
区段中，该区段在服务器看到配置之前会被剥离掉。

将以下内容保存为 `yoda_modal.yaml`：

```yaml
checkpoint_dir: ~/.cache/tuft/checkpoints   # on Modal, launch.py pins this to a Volume automatically
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

# Modal infra for deploy/modal/launch.py (TuFT ignores this; it's stripped before the server sees it):
modal:
  gpu: L4
  name: tuft-yoda
  proxy_auth: false      # the tml- API key is the auth; the Tinker SDK can't send Modal gateway headers
  min_containers: 0      # scale to zero when idle
```

为 `authorized_users` 生成一个真实的 API key（它**必须**以 `tml-` 开头）：

```bash
python -c "import secrets; print('tml-' + secrets.token_urlsafe(24))"
```

```{tip}
一个包含所有可用选项的现成示例位于
[`deploy/modal/tuft_config.example.yaml`](https://github.com/agentscope-ai/TuFT/tree/main/deploy/modal/tuft_config.example.yaml)，
而下文使用的示例自带其
[`config.yaml`](https://github.com/agentscope-ai/TuFT/tree/main/examples/personality_sft/config.yaml)。
```

## 第 2 步 —— 部署到 Modal

在**前台**运行启动器，这样当你按下 `Ctrl-C` 时服务器会被自动销毁（终止）：

```bash
python deploy/modal/launch.py --config yoda_modal.yaml --foreground
```

该脚本会生成一个自包含的 Modal 应用（一个
[`@modal.web_server`](https://modal.com/docs/guide/webhooks)）并将其提供出去。它会打印一个公开的 URL——把它复制下来：

```text
✓ Created web endpoint => https://<your-workspace>--tuft-yoda-tuftserver-serve.modal.run
```

确认服务器已启动（健康检查路由无需鉴权；首次调用会冷启动镜像并加载 vLLM，这可能需要几分钟）：

```bash
curl https://<your-workspace>--tuft-yoda-tuftserver-serve.modal.run/api/v1/healthz
# {"status":"ok"}
```

**部署模式。** `--foreground`（此处使用）封装了 `modal serve`——适合交互式运行；按 `Ctrl-C` 即可停止。省略它则为**分离式（detached）**部署（`modal deploy`），它会一直运行，直到你用 `--down` 将其停止：

```bash
python deploy/modal/launch.py --config yoda_modal.yaml          # detached
python deploy/modal/launch.py --down --name tuft-yoda           # stop it later
```

`checkpoint_dir` 会被自动绑定到一个 [Modal Volume](https://modal.com/docs/guide/volumes)
（`tuft-checkpoints`），这样你的适配器在容器关闭后仍能保留下来。

## 第 3 步 —— 在本地（笔记本）训练 Yoda LoRA

训练脚本 [`examples/personality_sft/train.py`](https://github.com/agentscope-ai/TuFT/tree/main/examples/personality_sft/train.py)
是一个**客户端**，它通过 Tinker SDK 经由 HTTP 连接到运行中的服务器——只需要 CPU 侧的依赖：

```bash
pip install tinker transformers
```

数据集是约 50 对手工编写的 `(user, assistant-in-Yoda-voice)` 配对，位于
[`dataset.py`](https://github.com/agentscope-ai/TuFT/tree/main/examples/personality_sft/dataset.py)
中。只有**助手**的 token 会获得损失权重（提示词被掩码），因此模型学习的是*语气*，而非问题本身。下面是几个示例配对：

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

将脚本指向你的 Modal URL，传入 `tml-` key，并选择模型（它必须与服务器上的 `supported_models` 条目匹配）：

```bash
python examples/personality_sft/train.py \
    --base-url https://<your-workspace>--tuft-yoda-tuftserver-serve.modal.run \
    --api-key tml-REPLACE_WITH_A_STRONG_KEY \
    --model Qwen/Qwen3-0.6B
```

该脚本会在训练**之前**对基础模型进行采样，运行 60 个 LoRA 步骤，然后在训练**之后**对训练好的适配器进行采样，这样你就能看到这种个性逐渐浮现：

```text
[before] sampling base model on held-out prompts
   [base] How should I spend my weekend?
      -> You could relax, catch up on friends, or work on a hobby. Try to balance rest with something fun.
   [base] I'm nervous about starting a new job.
      -> That's normal. Prepare a little, get good sleep, and remember they hired you for a reason.

[train] 60 steps, batch 8, lr 0.0001, rank 16
   step   0  loss=2.4131
   step  10  loss=1.8027
   step  50  loss=1.2563
   step  59  loss=1.1894

[save] sampler=tinker://abcd1234.../yoda-sampler
[save] run_id=abcd1234-...

[after] sampling the trained adapter
   [yoda] How should I spend my weekend?
      -> Rest and adventure, balance them you must, hmm. Outside, go — the Force in nature lives. Restored, you will be.
   [yoda] I'm nervous about starting a new job.
      -> Nervous, you are? Natural, this fear is. Breathe, you must. Chosen, you were — trust yourself, young one.
```

```{tip}
可调参数：`--lora-rank`、`--num-steps`、`--batch-size`、`--learning-rate`、`--max-length`。
传入 `--no-before` 可跳过对基础模型的采样。编辑 `dataset.py` 即可换成另一个角色。
```

## 第 4 步 —— 下载适配器

训练会将一个标准的 [PEFT](https://huggingface.co/docs/peft) LoRA 适配器写入服务器的 `checkpoint_dir`，在 Modal 上即
`tuft-checkpoints` [Volume](https://modal.com/docs/guide/volumes)。该脚本会打印一个 `run_id`——用它通过
[`modal volume get`](https://modal.com/docs/reference/cli/volume) 下载适配器：

```bash
modal volume get tuft-checkpoints <run_id> ./weights/
# -> ./weights/<run_id>/yoda-final/adapter/{adapter_config.json, adapter_model.safetensors}
```

你也可以选择将适配器合并进完整的模型权重中（需要 `torch`、`peft`、`transformers`）：

```python
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype=torch.bfloat16)
merged = PeftModel.from_pretrained(base, "./weights/<run_id>/yoda-final/adapter").merge_and_unload()
merged.save_pretrained("./yoda-merged")   # standard HF model dir, servable by vLLM
```

## 第 5 步 —— 销毁（终止）

如果你是在前台部署的，只需按 `Ctrl-C` 即可。对于分离式部署：

```bash
python deploy/modal/launch.py --down --name tuft-yoda
```

由于设置了 `min_containers: 0`，一个空闲的 Modal 部署本就不产生任何费用，但 `--down` 会将该应用彻底移除。你的适配器会一直保留在 `tuft-checkpoints` Volume 中，直到你将其删除。

```{tip}
即便采用了缩容至零，在一次活跃会话期间 GPU 仍然**利用率不足**——客户端在两次突发请求之间处于等待状态。由于 TuFT 是多租户的，你可以通过将**多个并发客户端**指向同一个热部署来让它保持繁忙（无论是你自己的任务还是其他用户的任务，每个都在 `authorized_users` 下持有一个 key；提高 `max_loras` 以同时容纳更多适配器）。参见
{ref}`保持 GPU 繁忙 <keeping-the-gpu-busy>`。
```

## 后续步骤

- 尝试一个更大的模型（例如 `Qwen/Qwen3-4B`），方法是同时更改配置中的两个名称，并将
  `modal.gpu` 设为 `H100`（或传入 `--gpu H100`）。
- 如果你想要一个专用的按需 GPU 虚拟机，请参见 [Lambda Cloud 指南](lambda.md)。
- 浏览 [Modal 的文档](https://modal.com/docs) 以了解 GPU 类型、Volume 和 Secret。
