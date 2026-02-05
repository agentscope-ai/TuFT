# 快速开始示例

本示例演示如何使用 [Tinker SDK](https://pypi.org/project/tinker/) 通过 TuFT 进行训练和采样。

在运行代码之前，请确保服务器在端口 10610 上运行。有关启动服务器的说明，请参阅[安装](installation.md)部分。

## 1. 数据准备

按照 TuFT 期望的格式准备训练数据：

```python
import tinker
from tinker import types

# 连接到运行中的 TuFT 服务器
client = tinker.ServiceClient(base_url="http://localhost:10610", api_key="local-dev-key")

# 发现可用的基础模型
capabilities = client.get_server_capabilities()
base_model = capabilities.supported_models[0].model_name

print("支持的模型：")
for model in capabilities.supported_models:
    print("-", model.model_name or "(未知)")

# 准备训练数据
# 实际使用中，您会使用 tokenizer：
# tokenizer = training.get_tokenizer()
# prompt_tokens = tokenizer.encode("Hello from TuFT")
# target_tokens = tokenizer.encode(" Generalizing beyond the prompt")

# 本示例使用模拟的 token ID
prompt_tokens = [101, 42, 37, 102]
target_tokens = [101, 99, 73, 102]

datum = types.Datum(
    model_input=types.ModelInput.from_ints(prompt_tokens),
    loss_fn_inputs={
        "target_tokens": types.TensorData(
            data=target_tokens, 
            dtype="int64", 
            shape=[len(target_tokens)]
        ),
        "weights": types.TensorData(data=[1.0, 1.0, 1.0, 1.0], dtype="float32", shape=[4])
    },
)
```

**示例输出：**
```
支持的模型：
- Qwen/Qwen3-4B
- Qwen/Qwen3-8B
```

## 2. 训练

创建 LoRA 训练客户端并执行前向/反向传播和优化器步骤：

```python
# 创建 LoRA 训练客户端
training = client.create_lora_training_client(base_model=base_model, rank=8)

# 运行前向/反向传播
fwdbwd = training.forward_backward([datum], "cross_entropy").result(timeout=30)
print("损失指标：", fwdbwd.metrics)

# 应用优化器更新
optim = training.optim_step(types.AdamParams(learning_rate=1e-4)).result(timeout=30)
print("优化器指标：", optim.metrics)
```

**示例输出：**
```
损失指标：{'loss:sum': 2.345, 'step:max': 0.0, 'grad_norm:mean': 0.123}
优化器指标：{'learning_rate:mean': 0.0001, 'step:max': 1.0, 'update_norm:mean': 0.045}
```

## 3. 保存检查点

保存训练好的模型检查点和采样器权重：

```python
# 保存检查点用于恢复训练
checkpoint = training.save_state("demo-checkpoint").result(timeout=60)
print("检查点保存到：", checkpoint.path)

# 保存采样器权重用于推理
sampler_weights = training.save_weights_for_sampler("demo-sampler").result(timeout=60)
print("采样器权重保存到：", sampler_weights.path)

# 查看会话信息
rest = client.create_rest_client()
session_id = client.holder.get_session_id()
session_info = rest.get_session(session_id).result(timeout=30)
print("会话包含的训练运行：", session_info.training_run_ids)
```

**示例输出：**
```
检查点保存到：tinker://550e8400-e29b-41d4-a716-446655440000/weights/checkpoint-001
采样器权重保存到：tinker://550e8400-e29b-41d4-a716-446655440000/sampler_weights/sampler-001
会话包含的训练运行：['550e8400-e29b-41d4-a716-446655440000']
```

## 4. 采样

加载保存的权重并生成 token：

```python
# 使用保存的权重创建采样客户端
sampling = client.create_sampling_client(model_path=sampler_weights.path)

# 准备采样的提示词
# sample_prompt = tokenizer.encode("Tell me something inspiring.")
sample_prompt = [101, 57, 12, 7, 102]

# 生成 token
sample = sampling.sample(
    prompt=types.ModelInput.from_ints(sample_prompt),
    num_samples=1,
    sampling_params=types.SamplingParams(max_tokens=5, temperature=0.5),
).result(timeout=30)

if sample.sequences:
    print("采样 token：", sample.sequences[0].tokens)
    # 将 token 解码为文本：
    # sample_text = tokenizer.decode(sample.sequences[0].tokens)
    # print("生成的文本：", sample_text)
```

**示例输出：**
```
采样 token：[101, 57, 12, 7, 42, 102]
```

> **注意**：当您本地有可用的 tokenizer 时，请将模拟的 token ID 替换为实际的 tokenizer 调用。

## 下一步

- 了解 [Chat SFT](../user-guide/chat-sft.md) 进行聊天数据的监督微调
- 探索 [Countdown RL](../user-guide/countdown-rl.md) 了解强化学习示例
- 配置[持久化](../user-guide/persistence.md)实现崩溃恢复
- 设置[可观测性](../user-guide/telemetry.md)进行监控
