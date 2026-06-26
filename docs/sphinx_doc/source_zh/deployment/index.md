# 部署

```{admonition} 🚀 没有 GPU？没问题！
:class: tip

运行 TuFT 并不需要你自己拥有 GPU。这些指南将展示如何在**按量付费的云服务商**上搭建一个 TuFT
服务器——按需租用 GPU，用完即释放。只需将
[Tinker](https://github.com/thinking-machines-lab/tinker) SDK 指向云端服务器，即可在本地（笔记本，无本地 GPU）驱动训练。
```

TuFT 是一个单一、标准的服务器（`tuft launch`）。
[`deploy/`](https://github.com/agentscope-ai/TuFT/tree/main/deploy) 目录下的部署辅助工具只是
在租用的算力上运行这个完全相同的服务器，并为你接好存储、端口和密钥——
它们不会改变产品本身的任何东西。请选择适合你工作流程的后端：

```{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Modal
:link: modal
:link-type: doc
:shadow: none

**无服务器（Serverless），缩容至零。** 部署一个 Web 端点，空闲时缩容至零，
按秒计费。适合突发或间歇性的使用场景。
:::

:::{grid-item-card} Lambda Cloud
:link: lambda
:link-type: doc
:shadow: none

**一台普通的按需 GPU 虚拟机**，按分钟计费，直到你将其终止为止。没有编排层；
实例会一直运行，直到你将它停止。
:::
```

## 我应该选择哪一个？

| 如果你想要…… | 使用 | 说明 |
|---|---|---|
| 空闲时缩容至零 | **[Modal](modal.md)** | 无服务器（Serverless）；缩容至零、按秒计费、托管代理与卷（Volume）。 |
| 一个单独的专用 GPU 实例 | **[Lambda Cloud](lambda.md)** | 按需、按分钟计费，直到你将其终止；没有编排层，也没有抢占。 |

(keeping-the-gpu-busy)=
## 保持 GPU 繁忙

单个由笔记本驱动的训练运行会让租用的 GPU **利用率不足**：客户端要做数据分词、
构建批次，并在多次 GPU 计算突发之间等待 HTTP 往返，因此 GPU 在大部分运行时间里都处于空闲状态。由于 TuFT 是
**多租户**的，一个已部署的服务器可以在同一块 GPU 上承载**多个并发的任务或用户**——
在 `authorized_users` 下为每个人分配各自的密钥，并调高 `max_loras`，使多个 LoRA 适配器可以同时训练。以这种方式共享
能提升利用率并分摊成本。这在 **Lambda** 上最为重要（无论 GPU 是否在工作都会持续计费），
在活跃的 **Modal** 会话期间也很重要（容器处于热状态，但 GPU 的使用是突发的）。

两份指南都会带你走完**同一个端到端示例**：配置服务器、
部署服务器、在本地（笔记本）上为 `Qwen/Qwen3-0.6B` 训练一个“像 Yoda 一样说话”的 LoRA，并
下载训练好的适配器——全部复用
[`examples/personality_sft/`](https://github.com/agentscope-ai/TuFT/tree/main/examples/personality_sft) 中可直接运行的代码。

```{toctree}
:maxdepth: 1
:hidden:

modal
lambda
```
