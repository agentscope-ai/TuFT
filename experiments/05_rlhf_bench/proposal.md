# TuFT：面向多租户在线 RL 微调的分离式训练系统

## 摘要

随着大语言模型（LLM）在 agentic 任务中的应用日益广泛，在线强化学习（RL）微调成为提升模型能力的关键手段。然而，现有的 RLHF 系统大多面向单租户、固定 batch 的离线训练场景设计，无法高效支持多租户、高并发的在线 RL 微调需求。本文提出 TuFT，一个面向多租户独立 LoRA 微调的在线 RL 训练系统。TuFT 采用 training-sampling 分离架构，通过 adapter-aware affinity scheduling 和动态资源管理，在维持高采样吞吐的同时，系统性地量化和修正了分离架构引入的 training-inference mismatch。实验表明，在多 adapter 混合负载下，TuFT 的 affinity scheduling 相比 vLLM 的 FCFS 调度提升吞吐 **1.45x-2.1x**；TuFT 的 Multi-Tenant Importance Sampling（MT-IS）算法能够有效修正 framework mismatch 和 statelessness mismatch，保证训练稳定性。

（TBD：和主线无关，不一定会加入）
此外，TuFT 支持 GPU 在 training 与 sampling pool 间的弹性资源管理。

---

## 1. Introduction

### 1.1 背景：LLM 微调与在线 RL

大语言模型（LLM）通过预训练获得了广泛的知识，但要在特定任务上取得优异表现，通常需要经过微调（Fine-Tuning）。近年来，基于人类反馈的强化学习（RLHF）和直接偏好优化（DPO）等方法成为提升模型对齐能力的主流手段。这些方法的核心循环是：采样（rollout）→ 评估（reward）→ 训练（gradient update）。

传统的 RLHF 训练通常是**离线批处理**的——先收集一批数据，再统一训练。然而，随着 agentic AI 的兴起，一种**在线、持续**的微调模式正在变得重要：

- **Agent 在环境中实时交互**，每次交互后立即生成训练数据；
- **模型参数持续更新**，策略在训练过程中不断进化；
- **多租户并发**，不同用户/应用同时训练各自的 adapter。

这种在线 RL 微调模式对系统提出了全新的挑战：采样和训练需要同时进行，且多个租户的训练任务需要共享 GPU 资源。

### 1.2 Agentic RL 的两种形态

在线 RL 微调在 agentic 场景下呈现两种不同的形态：

**形态一：单用户优化独立模型（Per-User RL）**

每个用户拥有独立的 agent 和独立的 LoRA adapter。用户的 agent 在特定环境中交互（如个人助手、代码补全工具），交互数据仅用于优化该用户自己的 adapter。这种场景的特点是：
- 每个 adapter 独立训练，不共享梯度；
- 用户数量多（数十到数千），但每个用户的请求量相对稀疏；
- 对延迟敏感，要求实时响应。

**形态二：多用户协作优化共享 Adapter（Per-Base-Model RL）**

多个用户共享同一个基础模型和同一个 LoRA adapter，所有用户的交互数据共同用于优化这个共享的 adapter（如针对私域特殊知识和功能要求的微调，对大模型泛化能力的微调）。在这种形态下，同一时刻可能有大量属于同一个 adapter 的请求到达 sampling pool，为 affinity scheduling 提供了优化空间。这种场景的特点是：
- 多个用户同时向同一个 adapter 贡献训练数据；
- 请求天然按 adapter 聚合，适合 batch 优化；
- 对训练吞吐要求高，因为数据量巨大。

TuFT 的设计目标同时覆盖这两种形态：支持**独立 LoRA adapter**的并行训练，同时通过**资源共享**最大化 GPU 利用率。

### 1.3 现有系统的局限性

现有工作可以从三个互补维度进行梳理，但均未完全覆盖 TuFT 所针对的多租户在线 RL 微调场景：

**（1）单租户全参数微调框架。** 当前主流的 RLHF 训练框架——如 verl、OpenRLHF、trl 等——主要面向单租户、全参数、固定 batch 的离线训练设计。这些系统在调度、资源分配和状态管理上均假设单一模型、单一训练任务，在扩展到多租户独立 LoRA 微调时面临根本性挑战：不同租户的 adapter 需要独立的梯度流和版本管理，而现有框架并未为此提供原生的系统支持。

**（2）多 LoRA 高效训练与推理系统。** 在推理侧，S-LoRA 和 Punica 通过 BGMV/SGMV 算子实现了高效的多 LoRA 并行推理；在训练侧，LoRA-Fusion 和 M-LoRA 等方案优化了多 adapter 的梯度计算效率。然而，这些工作大多仅针对**单一应用场景**——要么只做推理，要么只做训练——并未同时考虑 agentic RL 中采样（rollout）与训练（gradient update）交替进行、数据流相互依赖的端到端需求。

**（3）架构设计：Co-location 与 Disaggregation 的两难。** 在 training-sampling 的部署模式上，现有工作形成了两条对立路线：
- **Co-location 方案**（如 FlexLLM、Loquetier）将两者部署在同一进程或同一运行时中，通过共享参数和一致的计算后端消除了 **Framework Mismatch**（即因 sampling 与 training 后端框架不同导致的 logProb 差异）。但 co-location 无法消除 batch size 不同和 prompt position 不同造成的 mismatch（同一后端在不同 batch 组成下也会输出不同的 logProb），也无法消除 statelessness mismatch（sampling 用旧版本参数，training 用新版本重新计算 logProb）。此外，co-location 付出了部署灵活度受限（backend 无法独立升级、资源无法独立扩缩容）和运行效率降低（training 阻塞 sampling）的代价。
- **Disaggregation 方案**（如 Tinker）采用 training 与 sampling 分离部署，提供灵活的资源管理和独立的 backend 演进能力。但分离架构天然引入 Framework Mismatch，且 Tinker 作为闭源商业服务，其内部的 mismatch 处理机制不可知，无法为学术研究提供可复现、可验证的基线。

**综上**，现有工作要么局限于单租户场景，要么局限于训练/推理的单一任务类型，要么通过 co-location 回避 mismatch 但牺牲灵活性，要么通过 disaggregation 获得灵活性但引入 mismatch。TuFT 首次针对多租户在线 RL 微调的端到端需求，在 disaggregation 架构下系统性地量化并修正 mismatch，同时兼顾高吞吐采样与高效训练。

### 1.4 TuFT 的核心贡献

本文提出 TuFT，一个面向多租户在线 RL 微调的分离式训练系统，核心贡献包括：

1. **分离式架构设计**：独立的 training pool 和 sampling pool，支持灵活的资源管理和 backend 升级；
2. **Adapter-Aware Affinity Scheduling**：针对多 adapter 混合负载的调度优化，实验验证吞吐提升 **1.45x-2.1x**；
3. **Training-Inference Mismatch 量化**：首次系统性地量化了分离架构下的 framework mismatch（受 batch size、量化精度、并行策略等因素影响）和 statelessness mismatch（staleness + observation bias）；
4. **系统和算法结合解决 mismatch 问题**：在系统层面引入 ML-based bias estimator 对 framework mismatch 进行输出矫正，同时引入 Multi-Tenant Importance Sampling 在算法层面修正 statelessness mismatch，保证训练稳定性；
5. **动态资源管理**：带开销感知的弹性 GPU 分配，支持 training 与 sampling pool 间的资源调度。（[TBD] 资源调度与扩缩容与 mismatch 主线关联较弱，待后续完善或移至附录。）

---

## 2. Motivation

### 2.1 Rollout 效率瓶颈

在在线 RL 训练中，采样（rollout）占据了绝大部分计算时间。以典型的 PPO 训练为例，一个训练 step 包括：
- **Rollout**：使用当前策略生成 responses（占 80-90% 时间）；
- **Reward 计算**：评估生成质量（占 5-10% 时间）；
- **Gradient Update**：更新模型参数（占 5-10% 时间）。

在多租户场景下，rollout 效率问题被进一步放大：
- 每个租户使用不同的 LoRA adapter，导致 GPU 上需要频繁切换 adapter weights；
- 请求混合到达，batch 组成不可预测，无法充分利用计算并行性；
- Agent 的工具调用等待期间 GPU 空闲，造成资源浪费。

### 2.2 Training-Sampling 部署模式对比

在线 RL 要求 training 和 sampling 同时进行，现有工作根据资源分配方式形成了三种部署模式：

| 维度 | Co-location（同地部署） | Temporal Sharing（时序共享） | Spatial Disaggregation（空间分离） |
|------|------------------------|---------------------------|----------------------------------|
| **部署方式** | 同一进程/GPU，紧密耦合 | 同一套 GPU，按时间片交替执行 | 独立的 GPU pool，完全并行 |
| **代表工作** | FlexLLM, Loquetier | verl, OpenRLHF, trl | TuFT, Tinker, Ape-X/IMPALA |
| **优点** | ① 消除 Framework Mismatch<br>② 无参数冗余<br>③ 零跨节点通信开销 | ① 无需维护两套参数副本<br>② 工程实现成熟<br>③ 各自时隙内可用较优并行策略 | ① 完全并行无阻塞<br>② 各自最优部署和并行策略<br>③ 独立扩缩容，backend 独立升级<br>④ 适合多租户在线 serving |
| **缺点** | ① 缺乏灵活度，backend 无法独立升级<br>② training 阻塞 sampling<br>③ 资源配比固定，无法针对负载独立优化<br>④ 单点故障风险<br>⑤ 仍存在 batch size/position 引起的 mismatch | ① training 与 sampling 互相阻塞<br>② 存在切换开销和 GPU 空闲等待<br>③ 仍存在 Framework Mismatch（sampling 用 vLLM、training 用 FSDP）<br>④ batch size 受限于总资源<br>⑤ 不适合在线多租户 serving | ① 存在 Framework Mismatch<br>② 需要参数同步机制<br>③ 参数冗余（两份副本）<br>④ 需要额外的 mismatch 修正机制 |

**三类模式的核心差异**：

**（1）Co-location（同地部署）** 如 FlexLLM、Loquetier 将 training 和 sampling 融合在同一进程或单一运行时中，通过共享参数和一致的计算后端消除了 **Framework Mismatch**。但代价是部署灵活度受限——backend 无法独立升级、资源无法独立扩缩容——且 training 对 sampling 的阻塞会显著降低运行效率。此外，co-location 无法消除 batch size 不同和 prompt position 不同造成的 mismatch（同一后端在不同 batch 组成下也会输出不同的 logProb），也无法消除 statelessness mismatch（sampling 使用旧版本参数，training 重新计算 logProb）。

**（2）Temporal Sharing（时序共享）** 如 verl、OpenRLHF、trl 等主流 RLHF 框架，training 和 sampling 共享同一套 GPU 资源，按 step 交替执行（先 rollout，再 gradient update）。这种模式的工程实现最为成熟，无需维护两套参数副本。但 sampling 和 training 在时序上互相阻塞，GPU 在切换阶段存在空闲等待；且这些框架通常 sampling 用 vLLM、training 用 FSDP，**仍然存在 Framework Mismatch**，只是没有跨节点的网络开销。此外，batch size 受限于总 GPU 资源，无法同时针对采样高峰和训练高峰独立优化，因此不适合在线多租户 serving 场景。

**（3）Spatial Disaggregation（空间分离）** 如 TuFT、Tinker 以及分布式 RL 中的 Ape-X/IMPALA（actor-learner 分离），training 和 sampling 拥有独立的 GPU pool 和 backend，完全并行运行。这种模式消除了互相阻塞，允许各自采用最优的并行策略和部署配置，支持独立扩缩容和 backend 独立升级，天然适合多租户在线 serving。但分离架构引入了 **Framework Mismatch**（不同后端产生的 logProb 差异），需要通过参数同步和算法层修正（如 MT-IS）来保证训练稳定性。Relax 提出的 prefill-decode disaggregation 思想与此相通，但其针对的是推理阶段内部的分离，而非 training-sampling 分离。

TuFT 选择 **Spatial Disaggregation** 路线，因为在线多租户 RL 微调的核心需求是**持续并行的高吞吐采样**和**独立扩展的高效训练**，这只有在完全分离的架构下才能实现。TuFT 通过系统性的 mismatch 量化和 MT-IS 修正机制，在保留灵活性的同时确保训练稳定性。

### 2.3 Mismatch 问题

在分离架构下，sampling 端和 training 端使用不同的后端（如 vLLM vs FSDP），同一输入在两个端会产生不同的 logProb。TuFT 将 mismatch 来源归纳为两个核心层面（详见 §4.1）：

**Framework Mismatch**：vLLM 和 FSDP 使用不同的 kernel 实现、并行策略和计算路径。即使输入和模型权重相同，两边的配置差异（batch size、量化精度、TP/PP size 等）也会导致 logProb 不同。这是分离式架构最基础、最核心的 mismatch 来源。预实验表明：(1) 不同 adapter 的请求组成不会影响目标 request 的输出（adapter 间互相独立）；(2) batch size 会影响 logProb 输出；(3) prompt 在 batch 内的位置也会影响输出。

**Statelessness Mismatch**：training 算法假设 sampling 和 training 是"无状态"的独立操作，即 training 时根据当前 policy 重新计算 logProb，而不使用 sampling 时实际记录的 logProb。在 agentic RL 场景下，模型更新时 sampling pool 仍在持续产生数据，这些数据基于旧版本模型参数。如果直接用于新模型的计算，会产生 training 和 sampling 结果的 mismatch。

现有工作（FlexLLM、Loquetier）选择用 co-location / 单一运行时来回避 Framework Mismatch，但这牺牲了分离架构的灵活性和可扩展性，且无法消除 batch size/position 引起的 mismatch 和 Statelessness Mismatch。TuFT 则选择在分离架构下，通过系统层 ML-based bias estimator 与算法层 MT-IS 的协同，分层处理两类 mismatch。

### 2.4 Dynamic Resource Scaling 挑战 [TBD]

> **[TBD] 本节内容与 mismatch 主线关联较弱。待后续完善或移至附录。**

分离架构的另一个核心挑战是 **training 与 sampling 资源的动态伸缩**。在线 RL 的负载天然具有时变性：

- **采样高峰期**：大量 agent 同时交互，sampling queue 堆积，需要更多 sampling GPU；
- **训练高峰期**：某几个 adapter 的 training backlog 激增，需要更多 training GPU；
- **资源竞争**：固定比例的静态划分（如 7:3）在非峰值时段造成资源闲置，在峰值时段又成为瓶颈。

动态伸缩允许将 GPU 在 training pool 和 sampling pool 之间灵活迁移，但引入了三个子问题：

1. **迁移决策**：何时将 GPU 从 sampling pool 迁移到 training pool（或反之）？需要平衡 queue length、backlog 和迁移开销；
2. **迁移开销**：GPU 状态（KV cache、adapter weights）的迁移和 warmup 需要时间，频繁迁移反而降低有效吞吐；
3. **性能抖动**：pool 规模变化导致 batch size 分布和 staleness 分布同时变化，可能加剧 mismatch 效应。

TuFT 的设计通过**带开销感知的弹性伸缩策略**解决这一问题，在最大化资源利用率的同时，将迁移对 mismatch 的影响控制在可接受范围内。

---

## 3. TuFT 系统设计

### 3.1 分离式架构

TuFT 采用 **Training Pool + Sampling Pool** 的分离式架构：

```
┌────────────────────────────────────────────┐
│              TuFT Controller               │
│  ┌─────────────┐        ┌─────────────┐    │
│  │ Sampling    │Adapter │ Training    │    │
│  │ Pool        │◄──►    │ Pool        │    │
│  │ (vLLM)      │        │ (FSDP)      │    │
│  └─────────────┘        └─────────────┘    │
│         ▲                       ▲          │
│         │                       │          │
│  ┌──────┴──────┐         ┌──────┴──────┐   │
│  │ Adapter     │         │ Adapter     │   │
│  │ Store       │         │ Sync        │   │
│  └─────────────┘         └─────────────┘   │
└────────────────────────────────────────────┘
```

**Sampling Pool**：
- 基于 vLLM 构建，负责多租户请求的采样（rollout）；
- 每个请求附带 `adapter_id` 和 `version_tag`；
- 持续运行，不受 training 阻塞。

**Training Pool**：
- 基于 FSDP 构建，负责 adapter 的梯度更新；
- 每个 adapter 独立训练，不共享梯度；
- 训练完成后通过 Adapter Sync 机制通知 sampling pool 更新版本。

**Adapter Store**：
- 维护所有 adapter 的版本历史和参数；
- 支持快速的版本切换和回滚。

分离式架构支持在线agentic rl持续流水线sampling和training，sampling和training互不阻塞

### 3.2 Adapter-Aware Affinity Scheduling

#### 3.2.1 问题：Adapter Churn

在多 adapter 混合负载下，vLLM 的 FCFS 调度导致每个 batch 包含多个不同 adapter，频繁切换 adapter weights。我们在 vLLM 上进行了预实验，对比了随机混合（random/interleaved）与按 adapter 分组（uniform）两种负载模式下的吞吐和延迟。

| 对比维度 | batch_size=64 | batch_size=256 |
|---------|--------------|----------------|
| uniform vs random 加速比 | **2.13x** | **1.45x** |
| uniform vs random latency 降低 | **53%** | **31%** |
| uniform vs random-batch-uniform 差异 | <0.2% | <0.04% |

**关键发现**：(1) 性能瓶颈是 batch 内的 adapter 数量，与 request 提交顺序无关；(2) 按 adapter 分组比随机快 **1.45x-2.13x**；(3) 即使 batch_size=256，uniform 仍比 random 快 45%，说明大 batch 摊薄但无法消除 adapter churn。

#### 3.2.2 方案：Affinity Scheduling

基于实验数据，TuFT 设计了 Adapter-Aware Affinity Scheduling：

1. **Waiting Window**：设置短暂等待窗口（5-20ms），窗口内到达的请求按 adapter 分组；
2. **Batch 组成优化**：从 waiting queue 中选择请求时，优先选择与当前 batch 中已有请求相同 adapter 的请求；
3. **Fallback**：若无匹配 adapter，fallback 到 FCFS，避免过度延迟。

对于 LoRA inference，核心计算过程为：

W X + (α/r) · B A X

其中 W 代表 base model，A 为 down-projection、B 为 up-projection，计算顺序为先 A X 再左乘 B，α/r 为 LoRA 缩放因子。

以下优化机会为 affinity scheduling 提供了收益：

1. 相同 adapter 不同 input：将属于同一个 LoRA adapter 的 request 汇聚到一起能够使用更高效的并行计算（B A X 可以 batched GEMM）。在 vLLM 等框架中基本使用 FCFS 策略，并不会在内部对不同的 LoRA adapter 的 request 进行重组；
2. 不同 adapter 相同 input：不同 adapter 共享 prompt 时，base model 的 W X 部分及其 KV cache 可以复用（如 S-LoRA 等工作所示），而 (α/r)·BAX 部分因 adapter 不同仍需单独计算；这一优化的收益需结合实际负载占比进行评估。

预实验验证：改变同一 batch 中 filler requests 的组成（adapter 数量、每个 adapter 的请求数）不会影响目标 adapter request 的输出 logProb，说明不同 adapter 之间互相独立、无干扰。因此 affinity scheduling 是纯粹的性能优化手段，与 mismatch 修正解耦。


### 3.3 资源管理 [TBD]

> **[TBD] 本节内容与 mismatch 主线关联较弱。待后续完善或移至附录。放在§3末尾简述。**

TuFT 支持静态划分与动态弹性两种资源分配模式。动态弹性根据 `sampling_queue_length`、`training_backlog`、`gpu_utilization` 等指标，采用带滞回的阈值策略在 pool 间迁移 GPU，同时通过最小迁移粒度、渐进式迁移和 warmup 缓冲控制迁移开销。动态伸缩会间接影响 staleness 分布，TuFT 通过 ML-based bias estimator 和 MT-IS 的 adaptive clipping 进行协同修正。

---

## 4. Training-Inference Mismatch 量化与修正

### 4.1 Mismatch 来源分析

TuFT 的分离架构引入了两类 mismatch 来源：Framework Mismatch（框架实现差异）和 Statelessness Mismatch（算法设计与系统实现的语义鸿沟）。

#### Framework Mismatch

Framework Mismatch 来源于 training 和 sampling 后端在硬件/框架层面的固有差异。vLLM 和 FSDP 使用不同的 kernel 实现、并行策略和计算路径，导致即使输入和模型权重相同，两边的 logProb 也存在系统性偏差。

**影响因素**：

| 因子 | Sampling 端 (vLLM) | Training 端 (FSDP) | 是否可控 |
|------|-------------------|-------------------|---------|
| Batch size | 动态变化 | 固定 | 不可控（两端独立运行） |
| Quant precision | 可配置（FP8/INT8/BF16） | 通常 BF16/FP32 | 可控（对齐即可消除） |
| Tensor Parallelism size | 可配置 | 可配置 | 可控（对齐即可消除） |
| Pipeline Parallelism size | 无 | 可配置 | 不可控（sampling 端没有 PP） |
| Kernel 实现 | vLLM 自定义 | PyTorch 原生 | 不可控（框架固有） |
| 计算顺序 | vLLM 的并行切分 | FSDP 的并行切分 | 不可控（框架固有） |
| Prompt position | 动态变化 | 动态变化 | 不可控 |

**预实验验证**：
1. **Adapter 独立性**：改变同一 batch 中 filler requests 的组成（adapter 数量、每个 adapter 的请求数）不会影响目标 adapter request 的输出 logProb，说明不同 adapter 之间互相独立、无干扰。
2. **Batch size 影响**：对于同一 adapter 及其 requests，修改 batch size 会改变 logProb 输出。
3. **Prompt position 影响**：对于同一 adapter 及其 requests，修改 batch 内 prompt 组成位置会改变 logProb 输出。

由于用户训练时 prompt position 是动态变化的、无法预测，ML bias estimator 不将其作为输入特征，其影响可被 batch size 和 seq_len 吸收。

**Framework Mismatch 的特点**：偏差模式相对稳定（给定相同配置可复现），理论上可通过系统层校准消除。

#### Statelessness Mismatch

Statelessness Mismatch 由训练算法的设计引入，源于算法假设与系统实现之间的脱节。

在 agentic RL 场景下，模型更新时 sampling pool 仍在持续产生数据。这部分数据基于旧版本的模型参数，如果直接用于新模型的计算，会产生 training 和 sampling 结果的 mismatch。具体表现为：

- **Staleness**：sampling 使用版本 v_n 的参数生成 rollout 数据，training 使用版本 v_{n+k} 计算梯度。版本差异导致 policy 漂移。
- **Observation bias**：即使使用同一版本，training 端（FSDP）重新计算的 logProb 与 sampling 端（vLLM）记录的值存在系统性偏差，导致 advantage 估计和 policy gradient 方向混入噪声。

**Statelessness Mismatch 的特点**：偏差方向取决于具体实现，需要从算法设计层面进行修正。

---

**两类 mismatch 的关联性**：Framework Mismatch 是分离架构的"基线"偏差；Statelessness Mismatch 则反映了算法设计与系统实现之间的语义鸿沟。TuFT 的修正策略：对 Framework Mismatch 优先在系统层消除，对 Statelessness Mismatch 从算法层进行兜底修正。

### 4.2 Mismatch 实验量化分析

#### 4.2.1 Framework Mismatch 基线测量

固定 LoRA adapter、输入序列、batch size=1、量化精度=BF16，在 vLLM 和 FSDP 两端同时计算 logProb，测量 per-token diff 和 cum_diff 的分布。

| 指标 | 数值 |
|------|------|
| Per-token mean | ~0.010-0.013 |
| Cumulative diff (mean) | 0.5-1.5 |
| IS weight range | [0.34, 1.56] |
| P(\|w-1\| > 0.2) | ~38% |

这表明 Framework Mismatch 是分离架构下最基础、最显著的 mismatch 来源，单个 token 层面就已存在系统性偏差。

#### 4.2.2 Staleness：实验量化

用 6,400 条固定 prompts 连续执行 20 步梯度更新，每步记录相邻版本（v_n → v_{n+1}）和累积版本（v_0 → v_n）的 IS 权重分布：

| 指标 | Adjacent (v_n → v_{n+1}) | Cumulative (v_0 → v_n) |
|------|--------------------------|------------------------|
| IS weight 范围 | [0.3402, 1.1875] | [0.0108, 1.5640] |
| μ(w) | ~0.983 | 0.868 |
| P(\|w-1\| > 0.2) | 1.31% | 26.78% |
| **判定** | **MODERATE** | **SIGNIFICANT** |

相邻版本间 98.7% 的 IS 权重落在 PPO clip 边界 [0.8, 1.2] 内，但累积 20 步后 26.8% 的样本超出该范围，且 cum_μ(w) 从 1.0 单调下降到 0.868，揭示系统性 policy 漂移。第 8 步后出现相变：σ(w) 从 0.024 跃升至 0.263，极端异常值 w_max > 8，说明异步流水线中过时的 rollout 数据会快速退化。

#### 4.2.3 Framework Mismatch 的外部验证：Tinker 基线

上述 staleness 实验均在受控的本地环境下进行。为了验证**真实商业分离式部署**中 Framework Mismatch 的普遍性，我们额外测量了一个典型的外部多租户推理服务（Tinker）在 30 轮在线 RL 训练中的 mismatch 程度。Tinker 采用 training-sampling 分离部署，其 sampling 后端与本地 training 后端（FSDP）属于不同框架，天然存在 Framework Mismatch。

**实验设置**：固定 10 条 prompts，每轮先用 Tinker 的 sampling 引擎采样，再用本地 FSDP 计算同一样本的 logProb，记录 per-token diff、sequence-level cum_diff 和 IS weight。作为对照，我们在相同的本地 vLLM+sampling / FSDP+training 架构上也采集了 30 轮数据（记为"本地基线"）。

![Framework Mismatch: 本地基线 vs. Tinker](figures/framework_mismatch_comparison_v3.png)

**图 1 四个子图的详细解释**：

**(a) Per-Token Mismatch Magnitude**：Y 轴为 `mean_abs_diff`。Tinker（红线）的 per-token mismatch 在 0.014-0.020 之间波动，本地基线（蓝线）在 0.010-0.013 之间。两者均显示 Framework Mismatch 在单个 token 层面就已存在。

**(b) Sequence-Level Cumulative Diff**：Y 轴为 `cum_diff`。实线为 mean，虚线为 max。Tinker 的 mean（红实线）在后期明显偏高，大量典型序列的 cum_diff 超出 PPO clip 安全区间（绿色区域）；其 max（红虚线）频繁突破 2.0。本地基线（蓝线）整体较低，但虚线显示最坏情况仍超出安全区间。

**(c) IS Weight Variance**：Y 轴为 `std_is_weight`。Tinker 的方差在 Round 17 飙升到 1.28，意味着训练信号极其不稳定。本地基线也有波动，但幅度较小。

**(d) PPO Clip Violation Rate**：Y 轴为超出 clip 范围 [0.8, 1.2]（ε=0.2）的样本比例。Tinker 平均有 53% 的样本被 clip，本地基线平均 38%。

**核心结论**：

1. **Framework Mismatch 是真实分离式部署中的普遍问题**。Tinker 作为成熟的商业服务，其 Framework Mismatch 程度（53% clip 违反率）甚至高于本地基线（38%），说明 mismatch 不是 TuFT 独有的缺陷，而是所有 disaggregation 架构共同面临的挑战。
2. **Mismatch 随训练轮次演化而非恒定**。图 1 显示 mismatch 幅度在训练过程中动态变化，不存在简单的静态校准方案。
3. **该实验服务于 motivation，而非系统评估**。本地基线数据用于与 Tinker 做横向对比，证明 mismatch 问题的普遍性；后续 §5 的系统评估将在完整的 TuFT 实现上进行端到端训练对比。



### 4.3 Solve mismatch problem

（TBD 需要实验验证不同方法的效果）

TuFT 的 mismatch 修正遵循"**系统层优先，算法层兜底**"的分层策略：

#### 4.3.1 系统层无偏输出（首选）

对于 Framework Mismatch，优先在系统层面消除偏差根源，使 sampling 端直接输出"尽量无偏"的 logProb，减少后续算法的修正负担。

**ML-based Bias Estimator**：训练一个轻量级模型来预测 bias = logProb_vLLM - logProb_FSDP。

输入特征包括：

| 特征 | 说明 |
|------|------|
| batch_size_vllm | vLLM sampling 时的 batch size |
| batch_size_fsdp | FSDP training 时的 batch size |
| quant_precision_vllm | vLLM 量化精度（FP8/INT8/BF16） |
| quant_precision_fsdp | FSDP 量化精度 |
| tp_size_vllm | vLLM tensor parallelism size |
| tp_size_fsdp | FSDP tensor parallelism size |
| pp_size_fsdp | FSDP pipeline parallelism size（vLLM 无，固定为 1） |
| seq_len | 输入序列长度 |
| adapter_id | 可选，如果不同 adapter 有不同 bias pattern |

排除 prompt position，因为用户训练时 position 是动态变化的、无法预测，且预实验已证明 adapter 之间互相独立，其影响可被 batch size 和 seq_len 吸收。

训练方式：
- 离线阶段：收集 N 条样本的 `(vLLM_logProb, FSDP_logProb, features)` 对
- 在线阶段：bias_estimator 实时推理，开销很小（MLP 或轻量树模型）
- 持续更新：定期重新训练，适应模型版本变化

补偿公式：`logProb_corrected = logProb_vLLM - bias_estimator.predict(features)`

> **TBD**：系统层 ML-based 校准器的具体设计（模型结构、训练数据收集方式、在线推理开销）待进一步探索。

#### 4.3.2 算法层修正（兜底）MT-IS (multi-tenant importance sampling)

对于 Statelessness Mismatch，系统层难以完全消除（因为版本差异是动态的），此时通过算法层进行补偿。TuFT 的 MT-IS（Multi-Tenant Importance Sampling）和 Adaptive Clipping 作为兜底机制：

**符号约定。** 记 token 序列 τ = (s₁, a₁, …, s_T, a_T)；记 adapter i 在版本 v 下对动作 a_t 的策略为 π_{i,v}(a_t | s_t)；记 sampling 时的版本为 v_{samp}、training 时的版本为 v_{train}。

**Staleness 项**。每个采样输出附带 `sample_version` v_{samp}，训练时计算：

```
w_stale(t) = π_{i, v_train}(a_t | s_t) / π_{i, v_samp}(a_t | s_t)
```

及其 token-level clip 转换：`w_stale_clip(t) = clip(w_stale(t), 1-ε_s, 1+ε_s)`。

**Unified MT-IS weight**。如果系统层 bias estimator 已对 Framework Mismatch 进行补偿，则 MT-IS 主要处理 staleness：

```
w_MT-IS(t) = w_stale(t)
w_MT-IS_clip(t) = clip(w_MT-IS(t), 1-ε(t), 1+ε(t))
```

其中 Adaptive Clipping 范围 ε(t) 根据当前 mismatch 强度动态调整：

```
ε(t) = ε₀ · (1 + α · σ_hat(w_MT-IS))
```

σ_hat(w_MT-IS) 为近期窗口内 IS 权重的滚动标准差，α 为灵敏度超参。

**带 MT-IS 的 PPO 目标**。在 PPO/GRPO 的 surrogate loss 中将原始的 ratio 换成 MT-IS 权重：

```
L_MT-IS(θ) = E_t [ min( w_MT-IS_clip(t) · A_t,
                      clip(w_MT-IS(t), 1-ε(t), 1+ε(t)) · A_t ) ]
```

算法层修正的优势在于不依赖系统层的完美对齐，对动态变化的 mismatch 具有更强的适应性；缺点是在 mismatch 严重时可能牺牲收敛速度或最终性能。

---

## 5. Evaluation 计划

本节描述 TuFT 的端到端评估方案。由于完整系统实现仍在进行中，以下给出实验设计、baseline、数据集构建方法和待评估指标。

### 5.1 实验设置

| 配置项 | 设置 |
|--------|------|
| 硬件 | 8x A100 80GB |
| 基础模型 | LLaMA-3-8B-Instruct / Qwen2.5-7B-Instruct |
| Adapter | LoRA rank=16, alpha=32 |
| Sampling Backend | vLLM 0.4.0+ |
| Training Backend | FSDP (PyTorch) |
| RL 算法 | PPO / GRPO |

### 5.2 Baseline 对比

| Baseline | 类型 | 对比维度 |
|----------|------|---------|
| **vLLM FCFS + OpenRLHF** | Temporal Sharing 代表 | 单租户 RLHF 流程，无多租户优化，无 mismatch 修正 |
| **FlexLLM / Loquetier** | Co-location 代表 | 消除 Framework Mismatch，但牺牲灵活性和扩展性 |
| **TuFT w/o Affinity** | Ablation | 移除 affinity scheduling，验证 adapter churn 的影响 |
| **TuFT w/o MT-IS** | Ablation | 移除 mismatch 修正，验证 training 稳定性退化程度 |
| **TuFT w/o Bias Estimator** | Ablation | 移除系统层 Framework Mismatch 补偿 |
| **Tinker** | 外部商业服务 | 真实分离式部署的 mismatch 程度参考（非公平系统对比） |

### 5.3 数据集构建

#### 5.3.1 固定 Prompts（用于 Mismatch 量化）
- 从公开数据集中抽取 1,000-10,000 条代表性 prompts，覆盖不同长度分布（32-512 tokens）；
- 每条 prompt 在 sampling 端和 training 端同时计算 logProb，形成 per-token diff 和 cum_diff 的 ground truth；
- 用于 L1/L2/L3 各层 mismatch 的离线测量和校准器训练。

#### 5.3.2 多租户在线 RL 训练数据

**Adapter 任务设计**：
- **代码生成**：HumanEval / MBPP，reward 基于单元测试通过率；
- **数学推理**：GSM8K / MATH，reward 基于答案正确性；
- **指令遵循**：Alpaca / LIMA，reward 基于 LLM-as-a-Judge（如 GPT-4 打分）；
- **Agent 工具调用**：自定义环境（如 WebShop、ToolBench），reward 基于任务完成度。

**多租户负载模拟**：
- 租户数量：8-32 个独立 adapter；
- 请求到达模式：Poisson 过程，峰值/谷值负载比 3:1；
- 每个租户的训练数据独立生成，adapter 间不共享梯度。

#### 5.3.3 在线交互数据（Agentic 场景）
- 部署 agent 在模拟环境中持续交互（如个人助手、代码补全工具）；
- 交互日志实时送入 sampling pool 生成 rollout，经 reward model 打分后送入 training pool；
- 记录每个 adapter 的采样-训练延迟、staleness 分布和 mismatch 演化。

### 5.4 评估章节设计

#### 5.4.1 系统性能
- **吞吐与延迟**：在 8/16/32-adapter 混合负载下测量 sampling pool 的 tok/s 和 P99 latency；
- **Affinity Scheduling 增益**：TuFT Affinity vs vLLM FCFS 的吞吐比和延迟降低；
- **扩展性**：adapter 数量从 8 增加到 32 时，系统吞吐的衰减曲线。

#### 5.4.2 Mismatch 量化
- **Framework Mismatch**：固定输入、BS=1、BF16，测量 vLLM vs FSDP 的 per-token logProb diff 分布；
- **Framework Mismatch 影响因素**：变化 batch size（1/4/8/16/32）、量化精度（FP8/BF16）、TP size，测量各自对 mismatch 幅度的影响；
- **Statelessness Mismatch**：对比"使用 sampling logProb" vs "training 重新计算 logProb"的 advantage 估计偏差和梯度方向一致性；
- **Staleness**：不同版本间隔下的 IS 权重分布和 policy 漂移。

#### 5.4.3 训练稳定性
- **Reward Curve**：各 baseline 和 ablation 的 reward 随训练 step 的变化；
- **KL Divergence**：policy 与参考模型的 KL 散度，验证 mismatch 修正是否防止过度偏离；
- **收敛速度与最终性能**：达到目标 reward 阈值所需的 step 数，以及最终任务准确率；
- **Clip 违反率**：训练过程中 IS weight 超出 [0.8, 1.2] 的比例随时间演化。

#### 5.4.4 消融实验
- 逐个移除系统组件（Affinity Scheduling / Bias Estimator / MT-IS / Adaptive Clipping），量化每项对训练稳定性和最终性能的贡献；
- 绘制"组件贡献饼图"或"性能衰减柱状图"。

#### 5.4.5 多租户公平性与隔离性
- 不同 adapter 的训练进度差异（落后比例）；
- 高负载租户对低负载租户的性能影响；
- Resource contention 下的调度公平性指标。

---

## 6. Related Work

### 6.1 单租户 RLHF 训练框架

当前主流的 RLHF 训练框架——包括 verl、OpenRLHF 和 trl——主要面向单租户、全参数、固定 batch 的离线训练设计。这些框架在调度、资源分配和状态管理上均假设单一模型、单一训练任务，在扩展到多租户独立 LoRA 微调时面临根本性挑战：不同租户的 adapter 需要独立的梯度流和版本管理，而现有框架并未为此提供原生的系统支持。此外，这些框架通常假设 training 和 sampling 在同一进程内串行执行，未考虑在线 RL 中两者需要持续并行运行的需求。

### 6.2 多 LoRA 高效训练与推理系统

在推理侧，S-LoRA 和 Punica 通过 BGMV/SGMV 算子实现了高效的多 LoRA 并行推理，但其调度策略（如 FCFS）未针对多 adapter 混合负载优化，实验表明在多 adapter 场景下导致 1.45x-2.1x 的性能损失。在训练侧，LoRA-Fusion 和 M-LoRA 等方案优化了多 adapter 的梯度计算效率。然而，这些工作大多仅针对**单一应用场景**——要么只做推理，要么只做训练——并未同时考虑 agentic RL 中采样（rollout）与训练（gradient update）交替进行、数据流相互依赖的端到端需求。Tinker 提供了多租户 LoRA 微调服务，但其为闭源商业服务，内部的 mismatch 处理机制不可知，无法作为学术研究的可复现基线。

### 6.3 Co-location 与 Disaggregation 架构

在 training-sampling 的部署模式上，现有工作形成了两条对立路线。**Co-location 方案**（如 FlexLLM、Loquetier）将两者部署在同一进程或同一运行时中，通过共享参数和一致的计算后端消除了 **Framework Mismatch**（即因 sampling 与 training 后端框架不同导致的 logProb 差异）。但 co-location 无法消除 batch size 不同和 prompt position 不同造成的 mismatch（同一后端在不同 batch 组成下也会输出不同的 logProb），也无法消除 statelessness mismatch（sampling 使用旧版本参数，training 重新计算 logProb），且付出了部署灵活度受限（backend 无法独立升级、资源无法独立扩缩容）和运行效率降低（training 阻塞 sampling）的代价。**Disaggregation 方案**（如本文对比的 Tinker）采用分离部署，提供灵活的资源管理和独立的 backend 演进能力，但天然引入 Framework Mismatch。TuFT 选择 disaggregation 路线，并通过系统性的 mismatch 量化和修正机制，在保留灵活性的同时确保训练稳定性。

### 6.4 Training-Inference Mismatch 修正

TIS 和阿里 Stable RL 提出了 token-level importance sampling 来修正单租户场景下的 mismatch。但这些工作假设 fixed batch size 和 static adapter，未覆盖多租户动态采样场景。TuFT 的 MT-IS 首次将 IS 扩展到多租户动态环境，处理 statelessness mismatch（staleness + observation bias）。

### 6.5 分布式 RL

Ape-X 和 IMPALA 处理了分布式 RL 中的 staleness，但针对的是**分布式环境并行**（多个 actor 并行采样），而非多租户异构 staleness（不同 adapter 有不同的采样-训练间隔）。在多租户场景下，每个 adapter 的 staleness 分布独立且动态变化，需要 per-adapter 的 mismatch 修正策略，这是分布式 RL 框架未覆盖的。

---

## 7. Discarded Directions（已排除的方向）

在研究过程中，我们评估了多个潜在方向，最终因与主线关联不强或已被现有工作覆盖而排除：

| 方向 | 排除原因 |
|------|---------|
| Adapter-switching-noise | 实验验证 diff=0，adapter 加载是确定性操作 |
| Hybrid adapter requests mismatch | 实验验证改变 filler request 的组成（adapter 数量、每 adapter 请求数）不影响目标 request 的输出 logProb |
| 隐私保护（DP-LoRA / 安全聚合 / GPU 隔离） | 导师要求后续再议，当前聚焦 mismatch 主线 |
| Agent 长轨迹 1-batch 瓶颈优化 | General agent 推理问题，vLLM/SGLang 已持续优化 |
| KV Cache 碎片化优化 | vLLM/SGLang 已覆盖，非多租户特有 |
| 请求调度的公平性 | 通用调度问题，非 mismatch 特有 |
| 大规模 Adapter 存储层次优化 | 工程量大，与 mismatch 主线关联弱 |
| 联邦学习 / 跨租户知识迁移 | 与 mismatch 主线关联不强 |

这些方向可以作为 TuFT 的后续扩展工作。

---

## 8. Conclusion

本文提出 TuFT，一个面向多租户在线 RL 微调的分离式训练系统。TuFT 通过分离式架构实现了 training 和 sampling 的独立扩展和灵活管理，同时通过 adapter-aware affinity scheduling 和 MT-IS 算法系统性地解决了分离架构引入的 mismatch 问题。

**实验验证**：
- Adapter-aware affinity scheduling 在多 adapter 混合负载下提升吞吐 **1.45x-2.1x**；
- Framework Mismatch 已被实验量化（per-token diff ~0.010-0.013，38% clip 违反率），ML-based bias estimator 用于系统层补偿；
- Staleness 被实验量化为 MODERATE（单步）到 SIGNIFICANT（累积），MT-IS 算法可有效修正；
- 动态资源管理支持 training 与 sampling pool 间的 GPU 弹性迁移。（[TBD] 资源调度与扩缩容与 mismatch 主线关联较弱，待后续完善。）

TuFT 的设计同时适用于单用户优化独立模型和多用户协作优化共享 adapter 两种 agentic RL 场景，为在线 RL 微调的系统支持提供了新的思路。

---

## 9. Remaining Work（待完成工作）

以下是在形成完整论文前仍需完成的关键工作：

1. **构建端到端 evaluation 数据**：用于后续 adapter affinity scheduler 效果验证，mismatch 问题解决的效果验证。

2. **实现 adapter affinity scheduler**：在完整 vLLM 栈中实现 request reorganization 并测量吞吐增益。

3. **跨框架 batch-size-variance 验证（实验 A）**：测量 vLLM(BS=x) vs FSDP 的 mismatch 是否显著大于 vLLM(BS=1) vs FSDP，以决定 batch size 是否在 ML bias estimator 中作为独立特征。

4. **系统层 ML-based 校准器原型**。§4.3.1 提出通过 learned bias estimator 预测并补偿 Framework Mismatch，但具体模型结构、训练数据收集和在线推理开销仍需探索。

5. **算法层 MT-IS 算法设计和效果验证**。设计并实现 MT-IS 的 staleness term 和 adaptive clipping，在端到端训练中验证其对训练稳定性的提升。

6. **端到端训练稳定性实验**。§5.4.3 规划的 reward curve 和 KL divergence 对比实验需要在完整 TuFT 系统上进行，当前仅完成了离线 mismatch 量化。

7. **资源调度与扩缩容的独立评估**。§3.3 标记为 TBD 的资源管理章节，需在 mismatch 主线论文完成后，单独评估其有效性和开销。

---

*文档生成时间：2026-05-13*
*版本：v4.1（论文结构版）*
