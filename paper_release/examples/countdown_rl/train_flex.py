"""Countdown RL training using FlexBackend directly (single-backend mode).

This demonstrates the FlexBackend single-backend RL loop:
  training -> sampling -> training -> ...

No server required. Uses zero-copy IPC alias for weight sync between modes.

Usage:
  TUFT_FLEX_COUNTDOWN_GPU=1 python examples/countdown_rl/train_flex.py \
      --num-steps 10 --batch-size 2 --group-size 4
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

import torch
from tinker import types
from tinker.types.tensor_data import TensorData

# Add examples dir to path for env import
sys.path.insert(0, str(Path(__file__).parent))
from env import (
    COUNTDOWN_FEWSHOT,
    CountdownDatasetLoader,
    compute_reward,
    make_prompt_model_input,
)


def parse_args():
    p = argparse.ArgumentParser(description="Countdown RL with FlexBackend")
    p.add_argument("--model-name", default="qwen3-0.6b")
    p.add_argument("--num-steps", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--tp-size", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--test-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--format-score", type=float, default=0.1)
    p.add_argument("--fast-lora-switch", action="store_true", help="Deprecated: fast LoRA switch is now the default.")
    p.add_argument("--rebuild-lora-runtime", action="store_true")
    p.add_argument("--native-sleep", action="store_true")
    p.add_argument("--diagnose-log-ratio", action="store_true")
    p.add_argument("--sampling-enforce-eager", action="store_true")
    p.add_argument("--dataset", default="Jiayi-Pan/Countdown-Tasks-3to4")
    return p.parse_args()


def build_importance_sampling_datum(
    *,
    prompt: types.ModelInput,
    ob_len: int,
    toks: list[int],
    lps: list[float],
    adv: float,
) -> types.Datum:
    """Build datum for importance sampling loss (same as original example)."""
    model_input = prompt.append(types.EncodedTextChunk(tokens=toks[:-1]))
    target_tokens = [0] * ob_len + toks
    padded_sampling_logprobs = [0.0] * ob_len + lps
    padded_advantages = [0.0] * ob_len + [adv] * (model_input.length - ob_len)

    if not (
        model_input.length
        == len(target_tokens)
        == len(padded_sampling_logprobs)
        == len(padded_advantages)
    ):
        raise RuntimeError("Length mismatch in datum construction")

    return types.Datum(
        model_input=model_input,
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.long)),
            "logprobs": TensorData.from_torch(
                torch.tensor(padded_sampling_logprobs, dtype=torch.float32)
            ),
            "advantages": TensorData.from_torch(
                torch.tensor(padded_advantages, dtype=torch.float32)
            ),
        },
    )


def _log_ratio_stats(datums: list[types.Datum], outputs: list[dict]) -> dict[str, float]:
    values: list[float] = []
    ratios: list[float] = []
    for datum, output in zip(datums, outputs, strict=True):
        target_logprobs = output["logprobs"].to_torch().float()
        sampling_logprobs = datum.loss_fn_inputs["logprobs"].to_torch().float()
        advantages = datum.loss_fn_inputs["advantages"].to_torch().float()
        mask = advantages != 0
        diff = (target_logprobs[mask] - sampling_logprobs[mask]).detach().cpu()
        values.extend(float(x) for x in diff.tolist())
        ratios.extend(float(torch.exp(x).item()) for x in diff)
    if not values:
        return {
            "log_abs_mean": 0.0,
            "log_min": 0.0,
            "log_max": 0.0,
            "ratio_max": 0.0,
            "ratio_mean": 0.0,
        }
    abs_values = [abs(x) for x in values]
    return {
        "log_abs_mean": sum(abs_values) / len(abs_values),
        "log_min": min(values),
        "log_max": max(values),
        "ratio_max": max(ratios),
        "ratio_mean": sum(ratios) / len(ratios),
    }


async def main():
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Import FlexBackend after env setup
    from tuft.backends.flex.torchtp import FusedTorchTPVLLMFlexBackend
    from tuft.backends.flex.torchtp_training import load_tokenizer_for_spec, resolve_flex_model_spec
    from tuft.config import ModelConfig

    spec = resolve_flex_model_spec(args.model_name)
    tokenizer = load_tokenizer_for_spec(spec)
    print(f"[init] model={args.model_name} path={spec.path}")
    print(f"[init] tokenizer={type(tokenizer).__name__} vocab={spec.vocab_size}")

    config = ModelConfig(
        model_name=args.model_name,
        model_path=Path(spec.path),
        max_model_len=1024,
        tensor_parallel_size=args.tp_size,
        training_backend="flex",
        sampling_memory_fraction=0.40,
        sampling_enforce_eager=args.sampling_enforce_eager,
        sampling_max_model_len=1024,
        sampling_rebuild_runtime_for_lora=args.rebuild_lora_runtime and not args.fast_lora_switch,
        sampling_flex_kv_cache_only_sleep=not args.native_sleep,
        micro_batch_size=1,
    )

    # Initialize FlexBackend
    print("[init] Loading FlexBackend...")
    t0 = time.perf_counter()
    backend = FusedTorchTPVLLMFlexBackend(config)
    await backend.async_init()
    await backend.create_adapter("countdown_rl", types.LoraConfig(rank=args.lora_rank, seed=args.seed))
    print(f"[init] FlexBackend ready in {time.perf_counter() - t0:.1f}s")

    # Load dataset
    print(f"[data] Loading {args.dataset}...")
    dataset = CountdownDatasetLoader(args.dataset, args.test_size, args.seed)
    print(f"[data] train={len(dataset.train)} test={len(dataset.test)}")

    sampling_params = types.SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature)
    adam_params = types.AdamParams(
        learning_rate=args.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
    )

    print(f"\n[rl] Starting {args.num_steps} RL steps "
          f"(batch={args.batch_size}, group={args.group_size}, lr={args.learning_rate})")
    print("=" * 70)

    metrics_history = []

    for step in range(args.num_steps):
        step_start = time.perf_counter()
        problems = dataset.get_batch(args.batch_size, split="train")

        # === Phase 1: Sampling (generate rollouts) ===
        result = await backend.transform_to_sampling()
        assert result.supported, f"transform_to_sampling failed: {result.message}"

        datums = []
        mean_rewards = []
        kept_rollouts = 0
        total_correct = 0
        total_samples = 0

        for prob in problems:
            prompt_text = COUNTDOWN_FEWSHOT + prob.question
            prompt = make_prompt_model_input(tokenizer, prompt_text)

            sample_res = await backend.sample(
                prompt=prompt,
                num_samples=args.group_size,
                sampling_params=sampling_params,
                lora_id="countdown_rl",
            )

            rewards_g = []
            tokens_g = []
            logprobs_g = []

            for seq in sample_res.sequences:
                toks = list(seq.tokens)
                lps = seq.logprobs
                if lps is None:
                    raise RuntimeError("Sampling did not return logprobs.")
                lps = list(lps)

                resp_text = tokenizer.decode(toks, skip_special_tokens=True)
                r = compute_reward(
                    response_text=resp_text,
                    target=prob.target,
                    nums=prob.nums,
                    format_score=args.format_score,
                    use_continuous_shaping=True,
                )
                rewards_g.append(float(r))
                tokens_g.append(toks)
                logprobs_g.append(lps)
                total_samples += 1
                if r >= 1.0:
                    total_correct += 1

            mean_r = sum(rewards_g) / len(rewards_g)
            mean_rewards.append(mean_r)

            # Skip if no variance (all same reward)
            var_r = sum((r - mean_r) ** 2 for r in rewards_g) / max(1, len(rewards_g))
            std_r = var_r**0.5
            if std_r < 1e-8:
                continue

            advantages = [(r - mean_r) / (std_r + 1e-6) for r in rewards_g]
            ob_len = prompt.length - 1

            for toks, lps, adv in zip(tokens_g, logprobs_g, advantages, strict=True):
                datums.append(
                    build_importance_sampling_datum(
                        prompt=prompt, ob_len=ob_len, toks=toks, lps=lps, adv=adv
                    )
                )
                kept_rollouts += 1

        train_mean_reward = sum(mean_rewards) / max(1, len(mean_rewards))
        accuracy = total_correct / max(1, total_samples)

        # === Phase 2: Training (importance sampling update) ===
        result = await backend.transform_to_training()
        assert result.supported, f"transform_to_training failed: {result.message}"

        # Re-create adapter if needed (released during transform). Use
        # _adapter_configs because in worker-pool mode the parent never holds
        # optimizer objects.
        if "countdown_rl" not in backend._adapter_configs:
            await backend.create_adapter(
                "countdown_rl", types.LoraConfig(rank=args.lora_rank, seed=args.seed)
            )

        if datums:
            if args.diagnose_log_ratio:
                probe = await backend.forward(
                    data=datums,
                    lora_id="countdown_rl",
                    loss_fn="importance_sampling",
                    loss_fn_config=None,
                    backward=False,
                )
                ratio_stats = _log_ratio_stats(datums, probe.loss_fn_outputs)
                print(
                    "  ratio | "
                    f"log_abs_mean={ratio_stats['log_abs_mean']:.4f} "
                    f"log_min={ratio_stats['log_min']:.4f} "
                    f"log_max={ratio_stats['log_max']:.4f} "
                    f"ratio_max={ratio_stats['ratio_max']:.2f} "
                    f"ratio_mean={ratio_stats['ratio_mean']:.4f}"
                )
            output = await backend.forward(
                data=datums,
                lora_id="countdown_rl",
                loss_fn="importance_sampling",
                loss_fn_config=None,
                backward=True,
            )
            step_result = await backend.optim_step(adam_params, lora_id="countdown_rl")
            loss_val = output.metrics.get("loss:sum", 0.0)
        else:
            loss_val = 0.0

        step_ms = (time.perf_counter() - step_start) * 1000
        metrics = {
            "step": step,
            "train_mean_reward": train_mean_reward,
            "accuracy": accuracy,
            "kept_rollouts": kept_rollouts,
            "loss": loss_val,
            "step_ms": step_ms,
        }
        metrics_history.append(metrics)

        print(
            f"  step {step:3d} | reward={train_mean_reward:.4f} "
            f"acc={accuracy:.3f} rollouts={kept_rollouts} "
            f"loss={loss_val:.4f} ({step_ms:.0f}ms)"
        )

    # Summary
    print("\n" + "=" * 70)
    print("[summary] Countdown RL with FlexBackend completed")
    if len(metrics_history) >= 2:
        first = metrics_history[0]
        last = metrics_history[-1]
        print(f"  reward: {first['train_mean_reward']:.4f} -> {last['train_mean_reward']:.4f}")
        print(f"  accuracy: {first['accuracy']:.3f} -> {last['accuracy']:.3f}")
        reward_improved = last["train_mean_reward"] > first["train_mean_reward"]
        acc_improved = last["accuracy"] >= first["accuracy"]
        print(f"  reward_improved={reward_improved}, acc_improved={acc_improved}")


if __name__ == "__main__":
    import asyncio

    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    asyncio.run(main())
