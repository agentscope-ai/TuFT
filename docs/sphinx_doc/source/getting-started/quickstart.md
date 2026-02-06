# Quickstart

This example demonstrates how to use TuFT for training and sampling with the [Tinker SDK](https://pypi.org/project/tinker/).

Make sure the server is running on port 10610 before running the code. See the [Installation](installation.md) section for instructions on starting the server.

## 1. Data Preparation

Prepare your training data in the format expected by TuFT:

```python
import tinker
from tinker import types

# Connect to the running TuFT server
client = tinker.ServiceClient(base_url="http://localhost:10610", api_key="local-dev-key")

# Discover available base models
capabilities = client.get_server_capabilities()
base_model = capabilities.supported_models[0].model_name

print("Supported models:")
for model in capabilities.supported_models:
    print("-", model.model_name or "(unknown)")

# Prepare training data
# In practice, you would use a tokenizer:
# tokenizer = training.get_tokenizer()
# prompt_tokens = tokenizer.encode("Hello from TuFT")
# target_tokens = tokenizer.encode(" Generalizing beyond the prompt")

# For this example, we use fake token IDs
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

**Example Output:**
```
Supported models:
- Qwen/Qwen3-4B
- Qwen/Qwen3-8B
```

## 2. Training

Create a LoRA training client and perform forward/backward passes with optimizer steps:

```python
# Create a LoRA training client
training = client.create_lora_training_client(base_model=base_model, rank=8)

# Run forward/backward pass
fwdbwd = training.forward_backward([datum], "cross_entropy").result(timeout=30)
print("Loss metrics:", fwdbwd.metrics)

# Apply optimizer update
optim = training.optim_step(types.AdamParams(learning_rate=1e-4)).result(timeout=30)
print("Optimizer metrics:", optim.metrics)
```

**Example Output:**
```
Loss metrics: {'loss:sum': 2.345, 'step:max': 0.0, 'grad_norm:mean': 0.123}
Optimizer metrics: {'learning_rate:mean': 0.0001, 'step:max': 1.0, 'update_norm:mean': 0.045}
```

## 3. Save Checkpoint

Save the trained model checkpoint and sampler weights:

```python
# Save checkpoint for training resumption
checkpoint = training.save_state("demo-checkpoint").result(timeout=60)
print("Checkpoint saved to:", checkpoint.path)

# Save sampler weights for inference
sampler_weights = training.save_weights_for_sampler("demo-sampler").result(timeout=60)
print("Sampler weights saved to:", sampler_weights.path)

# Inspect session information
rest = client.create_rest_client()
session_id = client.holder.get_session_id()
session_info = rest.get_session(session_id).result(timeout=30)
print("Session contains training runs:", session_info.training_run_ids)
```

**Example Output:**
```
Checkpoint saved to: tinker://550e8400-e29b-41d4-a716-446655440000/weights/checkpoint-001
Sampler weights saved to: tinker://550e8400-e29b-41d4-a716-446655440000/sampler_weights/sampler-001
Session contains training runs: ['550e8400-e29b-41d4-a716-446655440000']
```

## 4. Sampling

Load the saved weights and generate tokens:

```python
# Create a sampling client with saved weights
sampling = client.create_sampling_client(model_path=sampler_weights.path)

# Prepare prompt for sampling
# sample_prompt = tokenizer.encode("Tell me something inspiring.")
sample_prompt = [101, 57, 12, 7, 102]

# Generate tokens
sample = sampling.sample(
    prompt=types.ModelInput.from_ints(sample_prompt),
    num_samples=1,
    sampling_params=types.SamplingParams(max_tokens=5, temperature=0.5),
).result(timeout=30)

if sample.sequences:
    print("Sample tokens:", sample.sequences[0].tokens)
    # Decode tokens to text:
    # sample_text = tokenizer.decode(sample.sequences[0].tokens)
    # print("Generated text:", sample_text)
```

**Example Output:**
```
Sample tokens: [101, 57, 12, 7, 42, 102]
```

> **Note**: Replace fake token IDs with actual tokenizer calls when you have a tokenizer available locally.

## Next Steps

- Learn about [Chat SFT](../user-guide/chat-sft.md) for supervised fine-tuning on chat data
- Explore [Countdown RL](../user-guide/countdown-rl.md) for reinforcement learning examples
- Configure [Persistence](../user-guide/persistence.md) for crash recovery
- Set up [Observability](../user-guide/telemetry.md) for monitoring
