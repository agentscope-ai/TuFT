# FlexBackend 技术细节报告：Torch-TP Training Backend 与 vLLM Sampling Backend 的 Zero-Copy 双向模式切换

本文档面向论文写作，系统描述 FlexBackend 的架构设计与实现细节，包括：training backend 的形态、sampling backend 的形态、以及两者之间基于 CUDA IPC 的 zero-copy 双向转换机制（哪些内存被卸载、哪些被清除、哪些被重新映射）。所有数据均来自真实 GPU 实测（Qwen3-4B TP=2，A100-80GB）。

---

## 1. 系统总览

FlexBackend 是一个独立于具体训练/推理引擎的**模式转换层**，其核心语义是：

- 同一组 GPU 上在 **training 模式**与 **sampling 模式**之间切换；
- 切换意味着上一阶段已经完成，源端 runtime 不保留（或仅保留可休眠的轻量 shell）；
- 转换的对象是 **base model 本身**：将 base model 从源端形态快速转换为目标端可用的形态；
- 全程 **zero-copy**：base model 权重不发生任何物理拷贝，也不经过 CPU 中转；
- 核心不变式：**显存中任意时刻只存在一份 base model 权重**（由 training 侧持有，sampling 侧通过 IPC 别名引用）。

切换的两条路径对应两种底层机制：

| 切换方向 | 机制 | 关键操作 |
|---|---|---|
| training → sampling | CUDA IPC 跨进程别名注入 | 生成 IPC descriptor → 注入 vLLM worker → 释放 training runtime |
| sampling → training | meta skeleton + storage 别名重建 | 构建空壳 training model → 参数 storage 绑定到 keepalive → 释放 vLLM runtime |

可选的 **keep-runtime** 策略进一步把切换开销压缩到毫秒级：不销毁 vLLM runtime，而是通过 CuMem 虚拟内存 unmap/remap 释放/恢复 KV cache 等显存（见第 5 节）。

---

## 2. Training Backend（fused PyTorch Tensor Parallel）

### 2.1 模型形态

Training backend 基于 **PyTorch 原生张量并行**（`torch.distributed.tensor.parallel`），但对 Qwen 类模型的线性层做了 **vLLM 兼容的 fused 重排**：

- `q_proj / k_proj / v_proj` → 融合为单个 `qkv_proj`（`FusedQKVLinear`）；
- `gate_proj / up_proj` → 融合为单个 `gate_up_proj`（`FusedGateUpLinear`）；
- 融合权重按 **rank-major** 方式打包，使得 TP=r 时第 i 个 rank 的本地分片恰好等于 vLLM 在 tensor_parallel_rank=i 上的本地分片。

这一重排是 zero-copy 可行的前提：training 侧每个 rank 的本地 weight shard 与 vLLM 对应 rank 的 weight shard 在布局和形状上完全一致，无需任何转换即可直接 alias。

### 2.2 并行切分

使用 `parallelize_module` + `init_device_mesh` 施加 `ColwiseParallel` / `RowwiseParallel`：

```
qkv_proj, gate_up_proj  -> ColwiseParallel（按输出维切分）
o_proj, down_proj       -> RowwiseParallel（按输入维切分，输出 all-reduce）
```

TP=1 时跳过全部并行化与 `torch.distributed` 初始化，直接单卡运行（支持单卡 RL 训练）。

### 2.3 参数与 LoRA 结构

- **base 参数全部冻结**（`requires_grad=False`）：base 权重在 RL 训练中不变，这是 descriptor 可跨切换复用的理论基础；
- LoRA 以 **wrapper 容器**形式叠加在 fused 线性层上（`FusedQKVLoRA` / `RowwiseLoRALinear`）：
  - 每个 wrapper 内部维护 adapter registry（ModuleDict），支持多个不同 rank 的 adapter 共存；
  - 同一时刻仅一个 active adapter 参与 forward；
  - 重复 `create_adapter` 只注册新 adapter，不会嵌套 wrapper；
- optimizer（AdamW）仅覆盖 LoRA 参数，base 权重不产生 optimizer state。

### 2.4 显存布局

training 阶段每卡显存 ≈ 一份 base shard（4B TP=2 约 4.9 GB）+ LoRA 参数 + 激活 + AdamW state。
**training 侧持有的 base storage 是整个系统的唯一权重副本**，sampling 侧通过 IPC 引用它。

---

## 3. Sampling Backend（vLLM + CUDA IPC 别名）

### 3.1 引擎形态

Sampling backend 使用 vLLM（V1 engine），关键配置：

- `load_format="dummy"`：vLLM 不从磁盘加载权重，而是在 GPU 上**分配 dummy 权重**并走完完整的引擎初始化流程（KV cache profiling、CUDA Graph capture 等）；
- `enable_sleep_mode=True`：启用 vLLM 的 CuMem 可插拔分配器（`CuMemAllocator`），使 weights/KV cache 从支持 unmap/remap 的虚拟内存池分配；
- `worker_cls=TuftFlexGPUWorker`：自定义 worker，在 CUDA Graph capture 前注入 IPC 别名（见 4.3）。

### 3.2 dummy 权重与内存标签

在 `enable_sleep_mode` 下，vLLM 的所有显存分配带有标签（tag）：

| tag | 内容 | 说明 |
|---|---|---|
| `weights` | dummy 模型权重 | 将被 IPC 别名替换，随后 retag 为 `discarded_weights` |
| `kv_cache` | KV cache（多块分配） | 4B TP=2 为 37 块共 35.3 GB；32B TP=4 为 65 块共 31.4 GB |
| `discarded_weights` | 被别名替换后的旧 dummy 权重页 | FlexBackend 引入的自定义标签 |

CuMemAllocator 的关键能力：每个分配记录 `(handle, tag)`，`handle` 保存虚拟地址映射信息。`unmap_and_release(handle)` 释放物理页但**保留虚拟地址记录**；`create_and_map(handle)` 用同一 handle 重新分配物理页并映射回**原虚拟地址**。这是 CUDA Graph 在 sleep/wake 后仍然有效的根本原因。

### 3.3 参数对象缓存

注入前需要知道 vLLM 模型中每个参数对象的引用。`prepare_cuda_ipc_alias_cache` 遍历 worker 的 model，把 `named_parameters` 按 vLLM state_dict 键名缓存到 worker 对象上（`_tuft_flex_object_cache`），后续每次注入直接复用该缓存。

---

## 4. training → sampling：CUDA IPC 别名注入

### 4.1 阶段分解

```
(1) state_dict 构建：从 training model 提取 vLLM 兼容的 fused state dict（每个 key -> 本地 weight shard）
(2) IPC descriptor 生成：对每个 shard 调用 reduce_tensor（torch.multiprocessing.reductions），
    提取 (cudaIpcMemHandle, storage_offset, shape, stride, dtype, device_index)
(3) all_gather_object：收集全部 rank 的 descriptor（每个 vLLM worker 只消费自己 rank 的）
(4) keepalive 保存：持有全部 base storage 的 Python 引用（producer storage 生命周期保证）
(5) vLLM 引擎创建：dummy load + KV cache 初始化 +（可选 pre-capture 注入）+ CUDA Graph capture
(6) 别名注入：collective_rpc(inject_cuda_ipc_alias)
(7) training runtime 释放：model / optimizer / LoRA / tp_group 全部释放 + 一次性 empty_cache
```

### 4.2 注入的实现（inject_cuda_ipc_alias）

对每个 worker（按自己的 TP rank 选取 descriptor）逐参数执行：

```python
shared = rebuild_cuda_tensor(*descriptors[key]["ipc"])   # cudaIpcOpenMemHandle，打开 producer 的物理页
obj.data = shared                                        # Parameter.data 重指向共享 storage
```

- `rebuild_cuda_tensor` 内部调用 `cudaIpcOpenMemHandle`，在 vLLM 进程的地址空间中映射 training 进程已有的物理页 —— **不发生任何数据拷贝**；
- 被替换的旧 dummy 权重的 CuMem 分配被 **retag** 为 `discarded_weights`（记录旧 `data_ptr()` → `pointer_to_data[ptr].tag = "discarded_weights"`），使其在后续 sleep 中被丢弃、wake 时不参与大块 remap；
- FP8 量化 scale 键（`*_q_scale` 等）没有对应 training 权重，直接填 1.0（非量化路径不使用）；
- 形状不匹配的参数跳过并记录（例如 embedding 的 vocab shard 与全词表差异，走 skip 策略）；
- 首次注入后执行一次性 `gc.collect() + empty_cache()`，归还 dummy 权重占用的 allocator 缓存；此后不再重复。

### 4.3 Pre-capture 注入时机（CUDA Graph 正确性）

`enforce_eager=False` 时 vLLM 会 capture CUDA Graph，图内记录的是权重张量的**设备指针**。若在 capture 之后才替换权重，图仍引用旧 dummy storage，导致非法访存或 `RuntimeError: cancelled`。

解决方案：自定义 worker `TuftFlexGPUWorker` 覆写 `compile_or_warm_up_model()`：

```
vLLM 初始化序列：
  load_model(dummy) → initialize_kv_cache → compile_or_warm_up_model()
                                              ├─ [TuFT hook] 读取 IPC descriptor 文件并注入别名
                                              └─ super()：torch.compile + CUDA Graph capture
```

即：**CUDA Graph capture 发生在别名注入之后**，图捕获的就是 IPC 别名后的真实 base storage 地址。descriptor 通过临时 pickle 文件 + 环境变量（`TUFT_FLEX_PRECAPTURE_IPC_PATH`）传递给 spawn 出的 worker 子进程。

### 4.4 稳态优化：descriptor 复用

base storage 在切换间保持稳定（keepalive 持续持有，且 base 权重冻结不变），因此第二次及之后的 t→s 切换**直接复用已缓存的 IPC descriptor**，跳过 state_dict 构建与 descriptor 生成（32B 实测该阶段约 354 ms → ≈0）。由 `sampling_reuse_ipc_descriptors`（默认 True）控制，`force=True` 可强制重建。实测 descriptor 生成本身仅 ~3 ms（4B），首次一次性初始化 ~100 ms。

---

## 5. sampling → training：skeleton 重建与 runtime 休眠

sampling → training 有两条路径。

### 5.1 路径 A：销毁-重建（release 语义）

```
(1) sampling runtime 释放：vLLM engine 销毁（KV cache / graph / dummy 页全部释放）
(2) training skeleton 构建：在 meta device 上用 init_empty_weights 构建 fused 模型空壳（不加载任何权重）
(3) 参数 storage 别名：把 keepalive 中保存的 base storage 直接绑定到 skeleton 参数
    （DTensor 参数替换其 _local_tensor；普通参数重建 Parameter）
(4) 残余 meta 张量物化：未被别名覆盖的 meta 张量（norm / embedding 等）以零初始化 CUDA 张量补齐
(5) LoRA 重新施加：在重建的模型上重新注册 adapter
```

关键点：**不加载第二份 base model**。skeleton 不含任何权重数据，参数直接指向切换前已存在的 base storage（即 training 端最初持有的那份），全程满足"只有一份 base model"约束。

### 5.2 路径 B：keep-runtime 休眠（sleep/wake 语义，推荐）

不销毁 vLLM runtime（保留 CUDA Graph），只通过 CuMem 虚拟内存机制释放/恢复物理页。

**Flex sleep（sampling → training，`flex_sleep_vllm_worker`）**：

| 内存对象 | 处理 | 说明 |
|---|---|---|
| dummy 权重页（`discarded_weights`） | `unmap_and_release`（清除物理页） | 已被 IPC 别名替换，无需保留；retag 为 `discarded_weights_sleeping` |
| KV cache（`kv_cache`，多块） | `unmap_and_release`（清除物理页） | 支持 partial：仅 unmap 指定 GB，其余 KV 保持常驻 |
| 真实 base 权重 | **不动** | 它是 training 进程的 IPC 物理页，不属于 vLLM 分配器 |
| CUDA Graph / graph pool / workspace | **保持常驻** | 占用小（4B 约 0.6 GB）、重建昂贵 |
| CPU offload | **不发生** | Flex 路径调用 `allocator.sleep(offload_tags=())` 等价语义，无任何 CPU 备份 |

实测 4B TP=2：sleep 释放 39.1 GB/卡（dummy 3.8 GB + KV 35.3 GB），耗时 96 ms。

**Flex wake（training → sampling，`flex_wake_vllm_worker`）**：

| 内存对象 | 处理 | 说明 |
|---|---|---|
| `discarded_weights_sleeping` 页 | `create_and_map`（重新映射） | 恢复虚拟地址→新物理页映射，页内容为空（dummy 权重，永不被读取） |
| `kv_cache_sleeping` 页 | `create_and_map`（重新映射） | 同上；内容无需恢复（推理时重写） |
| 真实 base 权重 | **不动** | IPC alias 全程有效，无需任何恢复 |
| CUDA Graph | **无需重新 capture** | 虚拟地址未变，图直接可用 |
| CPU reload | **不发生** | remap 只是建立页表映射，不传输数据 |

wake 后调用 `model_runner.post_kv_cache_wake_up()` 重置 KV cache 元数据。实测 4B TP=2：wake 29 ms（partial KV 12 GB 配置下 32B TP=4 为 732 ms）。

**为什么必须同时 remap `weights`（discarded）和 `kv_cache` 两类页**：只 remap KV 而保持 weights-tag 页 unmap 会触发 illegal memory access —— vLLM 的非参数 buffer / attention workspace 中仍存在对 weights 池虚拟地址的引用。remap 这些页只恢复映射不传数据，开销与页大小成正比但与数据无关。

### 5.3 两条路径对比

| 维度 | 路径 A（release） | 路径 B（keep-runtime sleep/wake） |
|---|---|---|
| vLLM runtime | 销毁，下次重建 | 常驻（休眠） |
| 常驻显存 | 0 | graph pool + 保留的 KV 页（GB 级可调） |
| s→t 开销 | skeleton 重建 + alias（稳态 ~127–185 ms） | sleep 96 ms |
| t→s 开销 | vLLM 重新初始化（一次性分钟级） | wake 29 ms |
| 适用场景 | 显存极度紧张 | RL 高频交替切换（推荐） |

---

## 6. 内存生命周期总表

以一次完整 RL 交替为例（keep-runtime 路径，4B TP=2）：

| 时刻 | 事件 | base 权重 | dummy 权重页 | KV cache | CUDA Graph |
|---|---|---|---|---|---|
| T0 | training 阶段 | training 进程持有（唯一副本） | 不存在 | 不存在 | 不存在 |
| T1 | t→s：IPC 注入 | vLLM 参数 data_ptr 指向 training 物理页 | 被替换，retag `discarded_weights` | 分配（满配置） | capture（捕获别名后地址） |
| T2 | sampling 阶段 | 同 T1，zero-copy 读取 | 常驻（已无用） | 使用中 | 回放中 |
| T3 | s→t：flex sleep | 不动（IPC 页仍映射） | 物理页 unmap+release | 物理页 unmap+release | 常驻 |
| T4 | training 阶段 | 同 T0 的同一份 storage | 仅虚拟地址记录 | 仅虚拟地址记录 | 常驻 |
| T5 | t→s：flex wake | 不动 | 重新映射（空页） | 重新映射（空页） | 无需重建 |

关键结论：
- **被卸载（offload 到 CPU）的内容：无**。整个系统在稳态切换中不发生任何 base 权重的 CPU 中转；
- **被清除（物理页释放）的内容**：dummy 权重页（`discarded_weights`）、KV cache 页（可 partial）；
- **被重新映射（remap）的内容**：上述被清除页的虚拟地址（wake 时），映射回新分配的物理页，无数据传输；
- **始终不变的**：唯一一份 base 权重（IPC 共享物理页）与 CUDA Graph。

---

## 7. 实测性能（Qwen3-4B，TP=2，A100-80GB）

统一 workload：training batch=2、seq=256；sampling 16 prompts × 128 tokens；性能口径关闭 verify-inject。

### 7.1 训练与推理

| 指标 | eager=True | eager=False + CUDA Graph |
|---|---|---|
| mean_train_ms | 274.0 | 281.6 |
| sampling_throughput | 752.5 tok/s | **2317.8 tok/s（3.1×）** |
| sampling_latency | 2661 ms | 866 ms |

### 7.2 切换开销

| 指标 | eager=True（release 路径） | eager=False（sleep/wake 路径） |
|---|---|---|
| 稳态 t→s | 70.2 ms | wake 28.6 ms |
| 稳态 s→t | 127–185 ms（skeleton 复建） | sleep 96.2 ms |
| 重复 IPC inject | 20.5 ms | 0（pre-capture 一次完成） |
| 一次性开销 | vLLM dummy load ~140 s；skeleton 首建 5.4 s | vLLM 初始化（含 compile/capture）~180 s |

### 7.3 多轮 round-trip 正确性与稳定性（3 轮）

- 每轮 `storage_stable=True`、`readback_exact=True`、IPC `mismatch=0`；
- sampling 吞吐 735.8 → 743.8 → 771.3 tok/s，无衰减；
- 切换后 training 侧 allocated 稳定在 4.9 GB（单份 base），free ≈ 72.5 GB，无泄漏。

### 7.4 规模扩展（32B，TP=4）

- retag dummy 权重 15.48 GB/rank；KV 为 65 块共 31.38 GB/rank；
- partial KV（12 GB）配置：sleep 144 ms，wake 732 ms，双向均 < 1 s；
- 训练 554 ms/step，sampling 859.7 tok/s（`disable_custom_all_reduce=True` 口径）。

---

## 8. 已知限制

1. **custom all-reduce 不兼容**：IPC 别名权重下 vLLM custom all-reduce 的 CUDA Graph capture 失败（`RuntimeError: cancelled`），稳定路径需 `disable_custom_all_reduce=True`；采样性能对齐的是该口径下的纯 vLLM baseline；
2. **vocab 分片参数**：embedding/lm_head 在 vLLM 侧为 vocab shard，与 training 全词表不一致，注入时 skip（base 推理不受影响，但训练回切后 embedding 为 keepalive 原始全词表）；
3. **`sleep(level=1)` 不可用**：vLLM 原生 sleep 会把 weights offload 到 CPU，与 zero-copy 语义冲突，故 FlexBackend 实现独立的 flex sleep/wake（直接操作 CuMemAllocator，手动 tag 跟踪，避免原生 API 二次 sleep 的 invalid argument 问题）；
4. **partial KV 权衡**：释放更多 KV 给 training 显存 ↔ wake 更慢，需按训练显存需求选择最小释放量。

---

## 9. 关键实现位置

| 模块 | 文件 | 内容 |
|---|---|---|
| 状态机基类 | `src/tuft/backends/flex/flex_backend.py` | FlexBackendMode / TransformResult / transform 状态机 |
| 具体 backend | `src/tuft/backends/flex/torchtp.py` | t→s/s→t 实现、sleep/wake 接入、训练 API |
| zero-copy 原语 | `src/tuft/backends/flex/torchtp_zero_copy.py` | descriptor 工厂、inject、retag、flex sleep/wake、partial KV |
| 训练运行时 | `src/tuft/backends/flex/torchtp_training.py` | fused 模块、TP 加载、LoRA 容器、skeleton 构建 |
| pre-capture hook | `src/tuft/backends/flex/vllm_worker.py` | TuftFlexGPUWorker（capture 前注入） |
| 配置 | `src/tuft/config.py` | `sampling_*` 系列开关（enforce_eager / sleep_mode / pre_capture_alias / partial_kv / reuse_descriptors） |

---

## 10. 2026-08 更新：训练正确性修复、FixedSamplingBackend 与 Router

本节补充原报告之后完成的关键架构与正确性修复。此前文档主要覆盖 zero-copy mode switch 与性能路径；后续实验发现 FlexBackend 的训练曲线一度弱于 HF+vLLM baseline，根因不是 TorchTP forward/backward 本身，而是 vLLM 动态 LoRA reload 的缓存语义。

### 10.1 FlexBackend 训练正确性修复

后续通过 0.6B / 4B countdown RL 与逐步 logprob ratio 诊断定位到：

- sampling 端 vLLM 动态 LoRA 在 sleep/wake fast path 下可能复用旧 adapter cache；
- 仅更换 `lora_int_id` 不足以强制刷新，`lora_name` 仍固定时仍可能命中旧 cache；
- 训练侧 forward 重新计算的 target logprobs 与 sampling 侧 logprobs 因 stale LoRA policy 不一致，导致 importance ratio 爆炸和 reward 下降。

最终修复方式：

```text
external lora_id 保持稳定（例如 countdown_rl）
internal vLLM lora_name 每个导出快照唯一化（例如 countdown_rl__snapshot_101）
```

这样既保持 TuFT/FlexBackend 外部多 LoRA 语义不变，又避免 vLLM 内部按 name 复用旧 adapter。修复后：

- 0.6B fast path ratio 从 `log_abs_mean≈0.79, ratio_max≈55` 恢复到 `log_abs_mean≈0.02, ratio_max≈1.8` 量级；
- 4B countdown 50-step 真实训练中，FlexBackend reward 从 `0.1004` 提升到 `0.3720`，HF+vLLM baseline 同配置最终约 `0.2773`；
- FlexBackend fast path 默认恢复，不再把 release/rebuild 作为最终性能方案。

### 10.2 默认高性能路径

当前推荐路径为：

```text
sampling_pre_capture_alias=True
sampling_enforce_eager=False
sampling_keep_runtime_on_training=True
sampling_rebuild_runtime_for_lora=False
unique internal vLLM lora_name per adapter snapshot
```

该路径保持 CUDA Graph 性能，并通过 unique `lora_name` 保证 LoRA reload 正确性。`sampling_rebuild_runtime_for_lora=True` 仅作为兼容未来 vLLM 动态 LoRA 回归时的 correctness fallback。

实测验证：

- 4B switch-only：t→s / s→t 均 < 1s；
- 4B CUDA Graph batch sampling：32 prompts × 128 tokens，约 3724 tok/s；
- 4B countdown RL 10-step/50-step 均显示 reward 正常提升。

### 10.3 FixedSamplingBackend 与量化语义

Flex sampling 路径不能启用 vLLM base quantization，因为 zero-copy 要求 training bf16 storage 与 vLLM dummy 参数保持 shape/dtype/layout 一致。量化会改变 vLLM 侧权重表示，破坏 alias 语义。

因此新增独立的 `FixedSamplingBackend`：

```text
Flex sampling runtime:
  bf16, dummy load, CUDA IPC alias, no quantization

FixedSamplingBackend:
  independent vLLM runtime, loads its own base model, may enable quantization
```

`FixedSamplingBackend` 继承/复用 `VLLMSamplingBackend` 的 vLLM engine 创建逻辑，但强制与 Flex zero-copy 解耦，并通过 `fixed_sampling_quantization` 配置独立启用 fp8/awq/gptq/bitsandbytes 等量化方式。GPU2 上已验证 0.6B + fp8：vLLM 日志显示 `quantization=fp8`，并选择 `MarlinFP8ScaledMMLinearKernel`，采样输出正确。

### 10.4 SamplingRuntimeRouter

新增 `SamplingRuntimeRouter` 用于在 Flex sampling 与 Fixed sampling 间路由：

- `active_backend ∈ {flex, fixed}`；
- `switch_to_fixed()`：先将 Flex 收缩到 training/minimal footprint，再初始化或 wake Fixed；
- `switch_to_flex()`：先将 Fixed `sleep(level=1)`，释放 Fixed 的 base/KV 大块显存，再切回 Flex sampling；
- 通过 `_fixed_asleep` 防止对 vLLM 原生 sleep 连续重复调用，规避已知 CUDA invalid argument 风险；
- 通过 adapter path replay 将已知 LoRA adapter 同步到 Fixed runtime。

0.6B 实测：

| 状态 | GPU used |
|---|---:|
| Flex sampling warm (`sampling_memory_fraction=0.20`) | 18.85 GiB |
| Fixed active (fp8, standalone) | 75.25 GiB |
| Fixed asleep + Flex active | 20.20 GiB |

Fixed `sleep(level=1)` 每轮释放约 `70.26 GiB`，保留约 `4.99 GiB` 的 runtime / CUDA Graph / allocator residual。Fixed→Flex 约 0.95s，Flex→Fixed 约 3.8s（主要为 Fixed wake）。

注意：当 Flex sampling 自身也以高 `sampling_memory_fraction=0.90` 占满 KV cache 时，Flex-only active 已实测约 `74.36 GiB`，此时再叠加 Fixed asleep residual 可能接近 80GB 上限。因此 Flex/Fixed 共存的 active KV budget 必须为 inactive residual 预留安全余量。

### 10.5 SwitchScheduler 与 simulator-side switch_tinker

为了支持 baseline 3/4 的快速 evaluation，新增通用 `SwitchScheduler` 核心：

```text
1. drain training queue
2. switch to sampling
3. sampling 至少持续 sampling_min_window_s
4. 窗口内 train 到达不抢占
5. 窗口后若 train pending 切回 training，否则继续 sampling
```

支持两种策略：

- `fixed`：固定 sampling window；
- `adaptive`：若最近窗口内 `total_switch_time / elapsed_time > max_switch_overhead_fraction`（默认 10%），按 `adaptive_growth_factor` 放大 `sampling_min_window_s`。

为最快跑论文实验，先在 `/mnt/nas/hanzhang.yhz/evaluation/simulator` 侧实现 `SwitchScheduledTinkerBackend`（`backend.type: switch_tinker`），作为 TinkerBackend 的 client-side wrapper：

```text
simulator tenant -> switch_tinker queue -> TinkerBackend -> TuFT server
```

这不改变 TuFT server 的 FutureStore / Controller 请求路径，适合快速评估 fixed/adaptive scheduling 对 multi-tenant workload 的影响。真实 GPU smoke 已验证：`switch_tinker` 能对 TuFT server 执行 create_adapter → sync_weights → sample → train_step → sample 的完整链路。后续如需严格 server-side baseline，可将同一 SwitchScheduler 接入 ServerState/FutureStore，并把 `switch_to_training/switch_to_sampling` 回调绑定到真实 backend mode switch 或 vLLM sleep/wake endpoint。
