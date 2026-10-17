# 在 Hugging Face Jobs 上部署

[Hugging Face Jobs](https://huggingface.co/docs/hub/jobs) 在 Hub 托管的硬件上运行一个 Docker 镜像，按分钟计费，并且可以通过一个需要身份验证的代理暴露容器端口。本指南将带你从头到尾完成整个流程——配置一个 TuFT 服务器、将其作为 Job 启动、在本地（笔记本）驱动训练，在 `Qwen/Qwen3-1.7B` 上训练一个"像尤达大师那样说话"的 LoRA，并下载该适配器——而且**无需本地 GPU**。

```{admonition} 你将构建的内容
:class: note

一个运行在 HF Jobs L4 GPU 上的 TuFT 服务器，可通过 `https://<job_id>--10610.hf.jobs` URL 访问；你将使用
[`examples/personality_sft/`](https://github.com/agentscope-ai/TuFT/tree/main/examples/personality_sft)
中的可运行代码在本地（笔记本）对其进行微调。检查点保存在你账户下的一个私有
[Storage Bucket](https://huggingface.co/docs/hub/storage-buckets) 中。
```

## 准备工作

1. **一个可以运行 Jobs 的 Hugging Face 账户**（参见
   [Jobs 定价](https://huggingface.co/docs/hub/jobs-pricing)），并使用一个有权启动 Jobs 的 token 登录：

   ```bash
   pip install -U huggingface_hub pyyaml
   hf auth login
   ```

2. **TuFT 仓库。** 启动器在本地只需要 `huggingface_hub` + `pyyaml`；GPU 依赖都在 Job 内运行的官方 TuFT 镜像中：

   ```bash
   git clone https://github.com/agentscope-ai/TuFT
   cd TuFT
   ```

以 HF 仓库 id 指定的模型会**以只读方式从 Hub 挂载**，而不是下载，因此服务器启动时无需下载步骤。受限（gated）模型在你的账户获得访问权限后也以同样方式挂载；若没有访问权限，Job 会在提交时被拒绝，不会产生费用。如需改为在 Job 内下载，请传入 `--no-mount-models --hf-token`。

## 第 1 步 —— 配置服务器

部署辅助脚本 [`deploy/hf_jobs/launch.py`](https://github.com/agentscope-ai/TuFT/tree/main/deploy/hf_jobs/launch.py)
是**由配置文件驱动**的：你编辑一个标准的 `tuft_config.yaml`（与 `tuft launch --config` 使用的文件相同）并运行该脚本。Jobs 的基础设施配置放在一个可选的 `hf_jobs:` 段中，它会在服务器读取配置之前被剥离。

将以下内容保存为 `yoda_hf_jobs.yaml`：

```yaml
checkpoint_dir: ~/.cache/tuft/checkpoints   # on HF Jobs, launch.py pins this to a Storage Bucket
model_owner: cloud-user

supported_models:
  - model_name: Qwen/Qwen3-1.7B
    model_path: Qwen/Qwen3-1.7B            # HF id (mounted read-only from the Hub) or a local path
    max_model_len: 4096
    tensor_parallel_size: 1
    colocate: true                         # single GPU: training + vLLM sampling share it
    sampling_memory_fraction: 0.3
    max_lora_rank: 16
    max_loras: 2

authorized_users:
  tml-REPLACE_WITH_A_STRONG_KEY: cloud-user  # clients send this as the X-API-Key header

persistence:
  mode: DISABLE

telemetry:
  enabled: false

# HF Jobs infra for deploy/hf_jobs/launch.py (TuFT ignores this; it's stripped before the server sees it):
hf_jobs:
  flavor: l4x1           # `hf jobs hardware` lists flavors and prices
  name: tuft-yoda
  timeout: 1h            # hard stop: Jobs don't scale to zero
```

为 `authorized_users` 生成一个真实的 API 密钥（它**必须**以 `tml-` 开头）：

```bash
python -c "import secrets; print('tml-' + secrets.token_urlsafe(24))"
```

启动器会把配置作为加密的
[secret](https://huggingface.co/docs/hub/jobs-configuration#environment-variables-and-secrets)
传给 Job，因此你的密钥不会出现在 Job 的命令中，也不会出现在 `hf jobs inspect` 的输出中。

```{tip}
包含所有可用选项的现成示例见
[`deploy/hf_jobs/tuft_config.example.yaml`](https://github.com/agentscope-ai/TuFT/tree/main/deploy/hf_jobs/tuft_config.example.yaml)。
```

## 第 2 步 —— 启动 Job

```bash
python deploy/hf_jobs/launch.py --config yoda_hf_jobs.yaml
```

启动器会在需要时创建检查点 bucket，使用官方 `ghcr.io/agentscope-ai/tuft` 镜像提交 Job，等待镜像拉取完成以及 `/api/v1/healthz` 就绪，然后打印 URL 和一段客户端代码（在 L4 上启动大约需要 5 分钟）：

```text
[job] submitted 6ad308a7c662881452265aa5: https://huggingface.co/jobs/<you>/6ad308a7c662881452265aa5
[wait] stage=SCHEDULING
[wait] stage=RUNNING

✅ TuFT is up: https://6ad308a7c662881452265aa5--10610.hf.jobs
```

**两层身份验证。** 暴露的端口位于 Jobs 代理之后，代理要求以 `Authorization: Bearer` 的形式提供一个对该 Job 命名空间有读取权限的 HF token。TuFT 的 `tml-` 密钥则单独通过 `X-API-Key` 发送，因此官方 Tinker SDK 只需额外添加一个请求头：

```python
import tinker

client = tinker.ServiceClient(
    base_url="https://<job_id>--10610.hf.jobs",
    api_key="tml-REPLACE_WITH_A_STRONG_KEY",
    default_headers={"Authorization": f"Bearer {HF_TOKEN}"},
)
```

传入 `--public` 可以去掉 HF token 这一层，仅依靠 `tml-` 密钥。

**存储。** `checkpoint_dir` 会被固定到私有 bucket `<you>/tuft-checkpoints`（可用 `--ckpt-bucket` 修改），因此适配器在 Job 结束后依然保留，其他 Job 也可以挂载它们。

**私有网络。** 使用 `--network-group <name>` 时，同一组内的其他 Job 可以通过 `http://${HF_NETWORK_GROUP_PREFIX}tuft:10610` 直接访问服务器，无需经过代理——当训练循环本身也作为 Job 运行时很有用。

## 第 3 步 —— 在本地（笔记本）训练 Yoda LoRA

训练脚本 [`examples/personality_sft/train.py`](https://github.com/agentscope-ai/TuFT/tree/main/examples/personality_sft/train.py)
是一个**客户端**，通过 Tinker SDK 以 HTTP 方式连接到正在运行的服务器。当 URL 是 `*.hf.jobs` 地址时，它会自动附加你的 HF token（来自 `hf auth login` 或 `HF_TOKEN`）；使用 `--public` 启动的 Job 只需要 `tml-` 密钥。聊天模板需要 `jinja2`：

```bash
pip install tinker transformers jinja2

python examples/personality_sft/train.py \
    --base-url https://<job_id>--10610.hf.jobs \
    --api-key tml-REPLACE_WITH_A_STRONG_KEY
```

脚本会在训练**之前**对基础模型采样，运行 LoRA 训练步骤，然后在训练**之后**对训练好的适配器采样（以下输出来自在 `l4x1` 上运行 30 步）：

```text
[train] 30 steps, batch 8, lr 0.0001, rank 16
   step   0  loss=2631.0788
   step  29  loss=773.8658
[save] run_id=daedfac5-1336-495e-a82e-93617e7a7e64

[after] sampling the trained adapter
   [yoda] How should I spend my weekend?
      -> Rest. You must. Enjoy it. Go out, you should. Nature, the best gift. Walk, laugh, and be you.
   [yoda] What's the key to writing good software?
      -> Code you must write, read it, you must. Always. Listen, learn. Good it becomes.
```

## 第 4 步 —— 下载适配器

训练会将一个标准的 [PEFT](https://huggingface.co/docs/peft) LoRA 适配器写入检查点 bucket。使用打印出的 `run_id` 将其同步到本地：

```bash
hf buckets sync hf://buckets/<you>/tuft-checkpoints/<run_id> ./weights/<run_id>
# -> ./weights/<run_id>/yoda-final/adapter/{adapter_config.json, adapter_model.safetensors}
```

如需将其合并为完整模型权重，请参阅 [Modal 指南的第 4 步](modal.md)。

## 第 5 步 —— 销毁（终止）

HF Jobs **不会缩容至零**：Job 会一直计费，直到达到其 `timeout` 或被你取消。

```bash
python deploy/hf_jobs/launch.py --status --config yoda_hf_jobs.yaml   # list TuFT Jobs
python deploy/hf_jobs/launch.py --down --config yoda_hf_jobs.yaml     # or --job-id <id> --namespace <ns>
```

传入 `--config` 时，`--down` 和 `--status` 会使用其中的 `hf_jobs.name` 和 `hf_jobs.namespace`，因此在组织下启动的 Job 也能被找到。你的适配器会保留在 bucket 中，直到你将其删除。

## 多 GPU 训练（FSDP）

在多 GPU 规格上，请将 GPU 分配给 FSDP 训练和 vLLM 采样，而不是让二者共置（colocate）。以下布局使用 `l4x4` 的全部四块 GPU：

```yaml
supported_models:
  - model_name: Qwen/Qwen3-4B
    model_path: Qwen/Qwen3-4B
    max_model_len: 4096
    colocate: false
    tensor_parallel_size: 2      # vLLM sampling on 2 GPUs
    training_backend: fsdp
    fsdp_num_gpus: 2             # FSDP training on the other 2
    max_lora_rank: 16
    max_loras: 8
```

```bash
python deploy/hf_jobs/launch.py --config fsdp.yaml --flavor l4x4
```

在该布局下，vLLM 运行在 GPU 0–1 上，FSDP worker 运行在 GPU 2–3 上。FSDP 适配器以相同的目录结构写入同一个 bucket，并可用标准 PEFT 加载。L4 之间通过 PCIe 通信，因此在 `l4x4` 上对 `Qwen/Qwen3-4B` 执行一个 SFT 步骤需要数秒；如需更高吞吐量，请选择更大的规格。

## 后续步骤

- 尝试更大的模型（例如 `Qwen/Qwen3-4B`）：修改配置中的两个名称，并将 `hf_jobs.flavor` 设置为 `a100-large`（或传入 `--flavor a100-large`）。
- 需要缩容至零请参阅 [Modal 指南](modal.md)，需要专用虚拟机请参阅 [Lambda Cloud](lambda.md)。
- 浏览 [HF Jobs 文档](https://huggingface.co/docs/hub/jobs) 了解硬件规格、卷与网络组。
