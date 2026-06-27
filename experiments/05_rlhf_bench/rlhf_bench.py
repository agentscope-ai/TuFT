"""
RLHF Algorithm Benchmark with Bias Correction
==============================================
对比两种主流 RL 算法（IS-REINFORCE、GRPO）在 predictor 校正前后的训练效果：

  算法维度：
    - IS-REINFORCE : 每步 B 个不同 prompt，各 1 个响应，batch 内归一化 advantage
    - GRPO         : 每步 1 个 prompt，G 个响应，组内归一化 advantage

  校正维度：
    - baseline     : 不做任何校正
    - predictor    : 用训练好的 predictor 估计 per-sequence bias 并减去

  评估：
    - 每 eval_interval 步在 GSM8K held-out test set 上做 greedy 解码，计算 exact match acc
    - 记录 training reward / approx_loss / mismatch 指标随步数变化

使用方式:
  python evaluation/rlhf_bench.py \\
    --groups all \\
    --num_steps 50 \\
    --eval_interval 5 --eval_n 30 \\
    --grpo_g 8 --buffer_size 8 \\
    --predictor_ckpt evaluation/predictor/checkpoints/bench_v1/best.pt \\
    --output evaluation/results/bench
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


# ── 路径设置 ──────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent
_SIM_DIR = _ROOT / "simulator"
sys.path.insert(0, str(_SIM_DIR))
sys.path.insert(0, str(_SIM_DIR / "simulator"))
_PRED_DIR = _ROOT / "predictor"
sys.path.insert(0, str(_PRED_DIR))

from simulator.backend.base import AdapterConfig
from simulator.backend.tinker_backend import TinkerBackend
from simulator.config import BackendConfig
from simulator.tasks.agent_task import AgentTask
from simulator.tasks.countdown import CountdownTask
from simulator.tasks.gsm8k import GSM8KTask
from simulator.tasks.hotpotqa import HotpotQATask
from simulator.tasks.humaneval import HumanEvalTask
from simulator.tasks.ifeval import IFEvalTask
from simulator.tasks.math_agent import MathAgentTask
from simulator.tasks.math_task import MATHTask
from simulator.tasks.mbpp import MBPPTask
from simulator.tasks.triviaqa import TriviaQATask


def build_task(task_name: str, seed: int = 42):
    """Build a task instance by name."""
    if task_name == "gsm8k":
        return GSM8KTask(seed=seed)
    elif task_name == "math":
        return MATHTask(seed=seed)
    elif task_name == "mbpp":
        return MBPPTask(seed=seed)
    elif task_name == "countdown":
        return CountdownTask(seed=seed)
    elif task_name == "humaneval":
        return HumanEvalTask(seed=seed)
    elif task_name == "ifeval":
        return IFEvalTask(seed=seed)
    elif task_name == "hotpotqa":
        return HotpotQATask(seed=seed)
    elif task_name == "triviaqa":
        return TriviaQATask(seed=seed)
    elif task_name == "math_agent":
        return MathAgentTask(seed=seed)
    else:
        raise ValueError(f"Unknown task: {task_name}")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rlhf_bench")


# ─────────────────────────────────────────────────────────────────────────────
# Predictor
# ─────────────────────────────────────────────────────────────────────────────
class PredictorCorrector:
    """Per-token predictor，实际学到的是 per-sequence scalar bias（within-seq 是白噪声）。"""

    def __init__(self, ckpt_path: str, device: str = "cpu"):
        from model import build_model

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        args = ckpt["args"]
        vocab_size = ckpt["vocab_size"]

        model_type = args.get("model", "transformer")
        if model_type == "transformer":
            self.model = build_model(
                "transformer",
                vocab_size=vocab_size,
                d_model=args.get("d_model", 128),
                token_emb_dim=args.get("token_emb_dim", 32),
                n_heads=args.get("n_heads", 4),
                n_layers=args.get("n_layers", 2),
            )
        else:
            self.model = build_model(
                "mlp",
                vocab_size=vocab_size,
                token_emb_dim=args.get("token_emb_dim", 32),
                hidden=args.get("hidden", 128),
            )
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.model.to(device)
        self.device = device
        self.max_seq_len = args.get("max_seq_len", 2048)
        logger.info(
            f"[predictor] Loaded {model_type} from {ckpt_path} "
            f"(epoch={ckpt['epoch']}, vocab={vocab_size})"
        )

    @torch.no_grad()
    def correct_batch(
        self,
        response_tokens_list: List[List[int]],
        sampling_lps_list: List[List[float]],
        n_prompt_tokens_list: List[int],
        temperature: float,
        lora_rank: int,
    ) -> List[List[float]]:
        """
        Predictor 输出 delta_hat[t]，由于 within-seq 是白噪声，
        模型实际学到的是 per-sequence mean bias c_hat。
        我们使用 delta_hat 的序列均值作为 scalar correction，
        避免引入额外 per-token 噪声。
        """
        corrected = []
        for resp_toks, samp_lps, n_prompt in zip(
            response_tokens_list, sampling_lps_list, n_prompt_tokens_list
        ):
            T = min(len(resp_toks), len(samp_lps), self.max_seq_len)
            if T == 0:
                corrected.append(samp_lps)
                continue

            token_ids = torch.tensor([resp_toks[:T]], dtype=torch.long, device=self.device)
            s_lps = torch.tensor([samp_lps[:T]], dtype=torch.float32, device=self.device)
            mask = torch.ones(1, T, dtype=torch.bool, device=self.device)
            n_prompt_t = torch.tensor([float(n_prompt)], device=self.device)
            temp_t = torch.tensor([temperature], device=self.device)
            lora_t = torch.tensor([lora_rank / 64.0], device=self.device)

            delta_hat = self.model(token_ids, s_lps, mask, n_prompt_t, temp_t, lora_t)
            # 取序列均值作为 per-sequence scalar（避免 per-token 噪声）
            c_hat = float(delta_hat.mean().item())

            corrected_lps = [lp - c_hat for lp in samp_lps]
            corrected.append(corrected_lps)
        return corrected


# ─────────────────────────────────────────────────────────────────────────────
# Mismatch metrics
# ─────────────────────────────────────────────────────────────────────────────
def compute_mismatch_metrics(
    sampling_lps: List[List[float]],
    training_lps: List[List[float]],
) -> Dict[str, float]:
    all_deltas: List[float] = []
    seq_cum_diffs: List[float] = []
    for s_lps, t_lps in zip(sampling_lps, training_lps):
        length = min(len(s_lps), len(t_lps))
        if length == 0:
            continue
        deltas = [s - t for s, t in zip(s_lps[:length], t_lps[:length])]
        all_deltas.extend(deltas)
        seq_cum_diffs.append(sum(deltas))
    if not all_deltas:
        return {"mean_abs_diff": 0.0, "mean_diff": 0.0, "std_cum_diff": 0.0, "clip02": 0.0}
    arr = np.array(all_deltas)
    seq_arr = np.array(seq_cum_diffs)
    clip02_hi, clip02_lo = math.log(1.2), math.log(0.8)
    clip02 = float(np.mean((seq_arr > clip02_hi) | (seq_arr < clip02_lo)))
    return {
        "mean_abs_diff": float(np.mean(np.abs(arr))),
        "mean_diff": float(np.mean(arr)),
        "std_cum_diff": float(np.std(seq_arr)),
        "clip02": clip02,
        "n_seqs": len(seq_cum_diffs),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm implementations
# ─────────────────────────────────────────────────────────────────────────────


def apply_correction(
    correction: str,
    sampling_lps_list: List[List[float]],
    response_tokens_list: List[List[int]],
    n_prompt_tokens_list: List[int],
    temperature: float,
    lora_rank: int,
    predictor: Optional[PredictorCorrector],
    global_mean_bias: float,
    training_lps: Optional[List[List[float]]] = None,
) -> Tuple[List[List[float]], Dict[str, Any]]:
    """根据 correction 类型应用 sampling_logprob 校正。"""
    if correction == "baseline":
        return sampling_lps_list, {}
    elif correction == "oracle":
        # 用 training logprobs 直接替换 sampling logprobs，IS ratio=1，完全无偏
        assert training_lps is not None, "oracle correction requires training_lps"
        return training_lps, {}
    elif correction == "global_mean":
        return [[lp - global_mean_bias for lp in lps] for lps in sampling_lps_list], {
            "global_mean_bias": global_mean_bias
        }
    elif correction == "predictor":
        assert predictor is not None
        return predictor.correct_batch(
            response_tokens_list=response_tokens_list,
            sampling_lps_list=sampling_lps_list,
            n_prompt_tokens_list=n_prompt_tokens_list,
            temperature=temperature,
            lora_rank=lora_rank,
        ), {}
    else:
        raise ValueError(f"Unknown correction: {correction}")


def response_length_stats(lengths: List[int]) -> Dict[str, float]:
    """计算 response 长度统计量。"""
    if not lengths:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0, "max": 0}
    arr = np.array(lengths)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": int(np.min(arr)),
        "max": int(np.max(arr)),
    }


async def step_reinforce(
    backend: TinkerBackend,
    adapter_id: str,
    task: GSM8KTask,
    tokenizer,
    buffer_size: int,
    temperature: float,
    max_tokens: int,
    lora_rank: int,
    correction: str,
    predictor: Optional[PredictorCorrector],
    global_mean_bias: float,
) -> Dict[str, Any]:
    """
    IS-REINFORCE: buffer_size 个不同 prompt 各 1 个响应，
    batch 内归一化 advantage，用校正后 sampling_logprobs 训练。
    """
    samples = []
    for _ in range(buffer_size):
        prompt = task.get_prompt()
        prompt_tokens = tokenizer.encode(prompt.text, add_special_tokens=False)
        try:
            results = await backend.sample(
                adapter_id=adapter_id,
                prompt_tokens=prompt_tokens,
                num_samples=1,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as e:
            logger.warning(f"sample failed: {e}")
            continue
        result = results[0]
        try:
            reward = task.compute_reward(result.text, prompt.metadata)
        except Exception:
            reward = 0.0
        samples.append(
            {
                "prompt_tokens": prompt_tokens,
                "response_tokens": result.tokens,
                "sampling_logprobs": result.logprobs,
                "reward": reward,
            }
        )

    if not samples:
        return {}

    rewards = [s["reward"] for s in samples]
    mean_r = float(np.mean(rewards))
    std_r = float(np.std(rewards))
    if std_r < 1e-8:
        advantages = [0.0] * len(samples)
    else:
        advantages = [(r - mean_r) / (std_r + 1e-6) for r in rewards]

    response_lengths = [len(s["response_tokens"]) for s in samples]
    n_prompt_tokens_list = [len(s["prompt_tokens"]) for s in samples]

    # 获取 training logprobs 用于 mismatch 计算
    try:
        training_lps = await backend.compute_training_logprobs(
            adapter_id,
            [
                {"prompt_tokens": s["prompt_tokens"], "response_tokens": s["response_tokens"]}
                for s in samples
            ],
            temperature=temperature,
        )
    except Exception:
        training_lps = []

    raw_metrics = (
        compute_mismatch_metrics([s["sampling_logprobs"] for s in samples], training_lps)
        if training_lps
        else {}
    )

    # 应用校正
    corrected_lps, corr_info = apply_correction(
        correction=correction,
        sampling_lps_list=[s["sampling_logprobs"] for s in samples],
        response_tokens_list=[s["response_tokens"] for s in samples],
        n_prompt_tokens_list=n_prompt_tokens_list,
        temperature=temperature,
        lora_rank=lora_rank,
        predictor=predictor,
        global_mean_bias=global_mean_bias,
        training_lps=training_lps if training_lps else None,
    )

    corr_metrics = compute_mismatch_metrics(corrected_lps, training_lps) if training_lps else {}

    # 训练
    training_datums = [
        {
            "prompt_tokens": s["prompt_tokens"],
            "response_tokens": s["response_tokens"],
            "sampling_logprobs": clps,
            "advantage": adv,
        }
        for s, clps, adv in zip(samples, corrected_lps, advantages)
    ]
    await backend.train_step(adapter_id, training_datums)

    return {
        "mean_reward": mean_r,
        "reward_std": std_r,
        "n_samples": len(samples),
        "raw": raw_metrics,
        "corrected": corr_metrics,
        "response_length_stats": response_length_stats(response_lengths),
        "correction_info": corr_info,
    }


async def step_grpo(
    backend: TinkerBackend,
    adapter_id: str,
    task: GSM8KTask,
    tokenizer,
    grpo_g: int,
    temperature: float,
    max_tokens: int,
    lora_rank: int,
    correction: str,
    predictor: Optional[PredictorCorrector],
    global_mean_bias: float,
) -> Dict[str, Any]:
    """
    GRPO: 1 个 prompt，采 G 个响应，组内归一化 advantage。
    长序列因 T×c 累积偏移比短序列获得更大 IS weight，
    predictor 校正能让 IS weight 回归由真实 policy divergence 决定。
    """
    prompt = task.get_prompt()
    prompt_tokens = tokenizer.encode(prompt.text, add_special_tokens=False)

    try:
        results = await backend.sample(
            adapter_id=adapter_id,
            prompt_tokens=prompt_tokens,
            num_samples=grpo_g,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception as e:
        logger.warning(f"grpo sample failed: {e}")
        return {}

    if not results:
        return {}

    rewards = []
    for r in results:
        try:
            rw = task.compute_reward(r.text, prompt.metadata)
        except Exception:
            rw = 0.0
        rewards.append(rw)

    mean_r = float(np.mean(rewards))
    std_r = float(np.std(rewards))
    if std_r < 1e-8:
        advantages = [0.0] * len(results)
    else:
        advantages = [(r - mean_r) / (std_r + 1e-6) for r in rewards]

    sampling_lps_list = [r.logprobs for r in results]
    response_tokens_list = [r.tokens for r in results]
    n_prompt_tokens_list = [len(prompt_tokens)] * len(results)
    response_lengths = [len(r.tokens) for r in results]

    # 获取 training logprobs
    try:
        training_lps = await backend.compute_training_logprobs(
            adapter_id,
            [{"prompt_tokens": prompt_tokens, "response_tokens": r.tokens} for r in results],
            temperature=temperature,
        )
    except Exception:
        training_lps = []

    raw_metrics = compute_mismatch_metrics(sampling_lps_list, training_lps) if training_lps else {}

    # 应用校正
    corrected_lps, corr_info = apply_correction(
        correction=correction,
        sampling_lps_list=sampling_lps_list,
        response_tokens_list=response_tokens_list,
        n_prompt_tokens_list=n_prompt_tokens_list,
        temperature=temperature,
        lora_rank=lora_rank,
        predictor=predictor,
        global_mean_bias=global_mean_bias,
        training_lps=training_lps if training_lps else None,
    )

    corr_metrics = compute_mismatch_metrics(corrected_lps, training_lps) if training_lps else {}

    training_datums = [
        {
            "prompt_tokens": prompt_tokens,
            "response_tokens": r.tokens,
            "sampling_logprobs": clps,
            "advantage": adv,
        }
        for r, clps, adv in zip(results, corrected_lps, advantages)
    ]
    await backend.train_step(adapter_id, training_datums)

    return {
        "mean_reward": mean_r,
        "reward_std": std_r,
        "n_samples": len(results),
        "raw": raw_metrics,
        "corrected": corr_metrics,
        "response_length_stats": response_length_stats(response_lengths),
        "correction_info": corr_info,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Agent GRPO step (multi-turn episodes)
# ─────────────────────────────────────────────────────────────────────────────
async def step_grpo_agent(
    backend: TinkerBackend,
    adapter_id: str,
    task: AgentTask,
    tokenizer,
    grpo_g: int,
    temperature: float,
    max_tokens: int,
    lora_rank: int,
    correction: str,
    predictor: Optional[PredictorCorrector],
    global_mean_bias: float,
) -> Dict[str, Any]:
    """
    Agent GRPO: run grpo_g independent episodes on the same question,
    compute group-normalized advantage, train on all turns.
    Each turn within an episode shares the episode reward as advantage.
    """
    import copy

    episodes: List[Dict] = []

    for _ in range(grpo_g):
        task_copy = copy.copy(task)
        episode_prompt = task_copy.reset_episode()
        conversation = episode_prompt.text
        metadata = episode_prompt.metadata
        turns = []
        step_results = []

        for _turn in range(10):  # max turns per episode
            prompt_tokens = tokenizer.encode(conversation, add_special_tokens=False)
            try:
                results = await backend.sample(
                    adapter_id=adapter_id,
                    prompt_tokens=prompt_tokens,
                    num_samples=1,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except Exception as e:
                logger.warning(f"agent sample failed: {e}")
                break
            if not results:
                break
            r = results[0]
            turns.append(
                {
                    "prompt_tokens": prompt_tokens,
                    "response_tokens": r.tokens,
                    "sampling_lps": r.logprobs,
                }
            )
            try:
                step_result = task_copy.step(r.text, metadata)
            except Exception as e:
                logger.warning(f"agent env step failed: {e}")
                break
            step_results.append(step_result)
            conversation = conversation + r.text + task_copy.format_observation(step_result)
            if step_result.done:
                break

        try:
            episode_reward = task_copy.compute_episode_reward(step_results, metadata)
        except Exception:
            episode_reward = 0.0

        if turns:
            episodes.append({"turns": turns, "reward": episode_reward})

    if not episodes:
        return {}

    rewards = [ep["reward"] for ep in episodes]
    mean_r = float(np.mean(rewards))
    std_r = float(np.std(rewards))
    advantages = (
        [(r - mean_r) / (std_r + 1e-6) for r in rewards] if std_r > 1e-8 else [0.0] * len(rewards)
    )

    all_sampling_lps = [t["sampling_lps"] for ep in episodes for t in ep["turns"]]
    all_response_tokens = [t["response_tokens"] for ep in episodes for t in ep["turns"]]
    all_prompt_tokens = [t["prompt_tokens"] for ep in episodes for t in ep["turns"]]

    try:
        training_lps = await backend.compute_training_logprobs(
            adapter_id,
            [
                {"prompt_tokens": t["prompt_tokens"], "response_tokens": t["response_tokens"]}
                for ep in episodes
                for t in ep["turns"]
            ],
            temperature=temperature,
        )
    except Exception:
        training_lps = []

    raw_metrics = compute_mismatch_metrics(all_sampling_lps, training_lps) if training_lps else {}

    corrected_lps, corr_info = apply_correction(
        correction=correction,
        sampling_lps_list=all_sampling_lps,
        response_tokens_list=all_response_tokens,
        n_prompt_tokens_list=[len(p) for p in all_prompt_tokens],
        temperature=temperature,
        lora_rank=lora_rank,
        predictor=predictor,
        global_mean_bias=global_mean_bias,
        training_lps=training_lps if training_lps else None,
    )
    corr_metrics = compute_mismatch_metrics(corrected_lps, training_lps) if training_lps else {}

    training_datums = []
    clp_idx = 0
    for ep, adv in zip(episodes, advantages):
        for turn in ep["turns"]:
            training_datums.append(
                {
                    "prompt_tokens": turn["prompt_tokens"],
                    "response_tokens": turn["response_tokens"],
                    "sampling_logprobs": corrected_lps[clp_idx],
                    "advantage": adv,
                }
            )
            clp_idx += 1

    if training_datums:
        await backend.train_step(adapter_id, training_datums)

    response_lengths = [len(t["response_tokens"]) for ep in episodes for t in ep["turns"]]
    return {
        "mean_reward": mean_r,
        "reward_std": std_r,
        "n_samples": len(episodes),
        "raw": raw_metrics,
        "corrected": corr_metrics,
        "response_length_stats": response_length_stats(response_lengths),
        "correction_info": corr_info,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Test accuracy evaluation
# ─────────────────────────────────────────────────────────────────────────────
async def evaluate_test_accuracy(
    backend: TinkerBackend,
    adapter_id: str,
    task: GSM8KTask,
    tokenizer,
    n: int,
    max_tokens: int,
) -> float:
    """在 held-out test set 上做 greedy 解码，计算 exact match accuracy。"""
    prompts = task.get_eval_prompts(n)
    correct = 0
    for prompt in prompts:
        prompt_tokens = tokenizer.encode(prompt.text, add_special_tokens=False)
        try:
            results = await backend.sample(
                adapter_id=adapter_id,
                prompt_tokens=prompt_tokens,
                num_samples=1,
                max_tokens=max_tokens,
                temperature=0.01,  # greedy
            )
            reward = task.compute_reward(results[0].text, prompt.metadata)
            correct += int(reward > 0.5)
        except Exception:
            pass
    return correct / max(n, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Single group runner
# ─────────────────────────────────────────────────────────────────────────────
async def run_group(
    group_name: str,
    algo: str,  # "reinforce" | "grpo"
    correction: str,  # "baseline" | "predictor" | "global_mean"
    args: argparse.Namespace,
    predictor: Optional[PredictorCorrector],
    global_mean_bias: float,
    output_dir: Path,
) -> List[Dict[str, Any]]:
    logger.info(f"\n{'=' * 65}")
    logger.info(f"  Group: {group_name}  (algo={algo}, correction={correction})")
    logger.info(f"{'=' * 65}\n")

    # 初始化 backend
    backend_cfg = BackendConfig(
        type="tinker",
        base_url=args.base_url,
        api_key=args.api_key,
        base_model=args.base_model,
    )
    backend = TinkerBackend()
    await backend.initialize(backend_cfg.to_dict())

    adapter_id = f"bench_{group_name}_{int(time.time())}"
    adapter_cfg = AdapterConfig(lora_rank=args.lora_rank, learning_rate=args.learning_rate)
    await backend.create_adapter(adapter_id, adapter_cfg)
    logger.info(f"[{group_name}] Created adapter: {adapter_id}")

    task = build_task(args.task, seed=args.seed)
    task.load_dataset()
    tokenizer = backend.get_tokenizer()

    predictor_for_group = predictor if correction == "predictor" else None

    log_path = output_dir / f"{group_name}_log.jsonl"
    records: List[Dict[str, Any]] = []
    weight_version = await backend.sync_weights(adapter_id)

    with open(log_path, "w") as log_fp:
        for step in range(1, args.num_steps + 1):
            t0 = time.time()

            is_agent = isinstance(task, AgentTask)
            if algo == "reinforce" and not is_agent:
                step_result = await step_reinforce(
                    backend=backend,
                    adapter_id=adapter_id,
                    task=task,
                    tokenizer=tokenizer,
                    buffer_size=args.buffer_size,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    lora_rank=args.lora_rank,
                    correction=correction,
                    predictor=predictor_for_group,
                    global_mean_bias=global_mean_bias,
                )
            elif is_agent:  # agent tasks: always use episode-based GRPO
                step_result = await step_grpo_agent(
                    backend=backend,
                    adapter_id=adapter_id,
                    task=task,
                    tokenizer=tokenizer,
                    grpo_g=args.grpo_g,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    lora_rank=args.lora_rank,
                    correction=correction,
                    predictor=predictor_for_group,
                    global_mean_bias=global_mean_bias,
                )
            else:  # grpo, non-agent
                step_result = await step_grpo(
                    backend=backend,
                    adapter_id=adapter_id,
                    task=task,
                    tokenizer=tokenizer,
                    grpo_g=args.grpo_g,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    lora_rank=args.lora_rank,
                    correction=correction,
                    predictor=predictor_for_group,
                    global_mean_bias=global_mean_bias,
                )

            if not step_result:
                continue

            weight_version = await backend.sync_weights(adapter_id)

            # 周期性 test accuracy
            test_acc = None
            length_stats = step_result.get("response_length_stats", {})
            len_str = f"len_mean={length_stats.get('mean', 0):.1f}"
            if args.eval_interval > 0 and step % args.eval_interval == 0:
                test_acc = await evaluate_test_accuracy(
                    backend=backend,
                    adapter_id=adapter_id,
                    task=task,
                    tokenizer=tokenizer,
                    n=args.eval_n,
                    max_tokens=args.max_tokens,
                )
                logger.info(
                    f"[{group_name}] step={step}/{args.num_steps} "
                    f"reward={step_result['mean_reward']:.3f} "
                    f"test_acc={test_acc:.3f} "
                    f"corr_bias={step_result.get('corrected', {}).get('mean_diff', float('nan')):.5f} "
                    f"{len_str}"
                )
            else:
                logger.info(
                    f"[{group_name}] step={step}/{args.num_steps} "
                    f"reward={step_result['mean_reward']:.3f} "
                    f"raw_bias={step_result.get('raw', {}).get('mean_diff', float('nan')):.5f} "
                    f"corr_bias={step_result.get('corrected', {}).get('mean_diff', float('nan')):.5f} "
                    f"std_cum={step_result.get('raw', {}).get('std_cum_diff', float('nan')):.4f} "
                    f"{len_str}"
                )

            record = {
                "step": step,
                "group": group_name,
                "algo": algo,
                "correction": correction,
                "mean_reward": step_result["mean_reward"],
                "reward_std": step_result.get("reward_std", 0.0),
                "test_acc": test_acc,
                "raw": step_result.get("raw", {}),
                "corrected": step_result.get("corrected", {}),
                "response_length_stats": length_stats,
                "correction_info": step_result.get("correction_info", {}),
                "step_time_s": time.time() - t0,
            }
            records.append(record)
            log_fp.write(json.dumps(record) + "\n")
            log_fp.flush()

    try:
        await backend.cleanup(adapter_id)
        logger.info(f"[{group_name}] Cleaned up adapter {adapter_id}")
    except Exception as e:
        logger.warning(f"[{group_name}] Cleanup failed: {e}")

    logger.info(f"[{group_name}] Done. {len(records)} steps logged to {log_path}")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
def print_summary(all_results: Dict[str, List[Dict]], last_n: int = 10) -> None:
    print(f"\n{'=' * 75}")
    print("  BENCHMARK SUMMARY")
    print(f"{'=' * 75}")

    def last_n_mean(records, *keys):
        vals = []
        for r in records[-last_n:]:
            try:
                v = r
                for k in keys:
                    v = v[k]
                if v is not None and not math.isnan(float(v)):
                    vals.append(float(v))
            except Exception:
                pass
        return float(np.mean(vals)) if vals else float("nan")

    groups = list(all_results.keys())
    hdr = f"  {'metric':<28}" + "".join(f"{g:>18}" for g in groups)
    print(hdr)
    print("-" * len(hdr))

    metrics = [
        ("mean_reward (last10)", ("mean_reward",)),
        ("test_acc (last eval)", ("test_acc",)),
        ("raw bias mean_diff", ("raw", "mean_diff")),
        ("corr bias mean_diff", ("corrected", "mean_diff")),
        ("raw std_cum_diff", ("raw", "std_cum_diff")),
        ("corr std_cum_diff", ("corrected", "std_cum_diff")),
        ("raw clip02", ("raw", "clip02")),
        ("corr clip02", ("corrected", "clip02")),
    ]

    for label, keys in metrics:
        row = f"  {label:<28}"
        for g in groups:
            v = last_n_mean(all_results[g], *keys)
            row += f"{v:>18.4f}"
        print(row)

    # test_acc 中只取有值的步骤
    print()
    print("  Test accuracy at each eval step:")
    for g, records in all_results.items():
        eval_steps = [(r["step"], r["test_acc"]) for r in records if r.get("test_acc") is not None]
        if eval_steps:
            accs = [f"step{s}={a:.3f}" for s, a in eval_steps]
            print(f"    {g}: {', '.join(accs)}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="RLHF Benchmark with Bias Correction")
    p.add_argument(
        "--groups",
        nargs="+",
        choices=[
            "reinforce_baseline",
            "reinforce_predictor",
            "reinforce_global_mean",
            "grpo_baseline",
            "grpo_predictor",
            "grpo_global_mean",
            "grpo_oracle",
            "all",
        ],
        default=["all"],
    )

    p.add_argument("--base_url", default="http://localhost:10610")
    p.add_argument("--api_key", default="tml-test-key")
    p.add_argument("--base_model", default="Qwen/Qwen3-4B")

    p.add_argument(
        "--task",
        default="gsm8k",
        choices=[
            "gsm8k",
            "math",
            "mbpp",
            "countdown",
            "humaneval",
            "ifeval",
            "hotpotqa",
            "triviaqa",
            "math_agent",
        ],
        help="task to train and evaluate on",
    )
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--num_steps", type=int, default=50)

    # IS-REINFORCE: buffer_size 个 prompt per step
    p.add_argument("--buffer_size", type=int, default=8, help="prompts per IS-REINFORCE step")
    # GRPO: G responses per prompt per step
    p.add_argument("--grpo_g", type=int, default=8, help="responses per prompt in GRPO")

    # Test accuracy evaluation
    p.add_argument(
        "--eval_interval", type=int, default=5, help="evaluate test accuracy every N steps (0=off)"
    )
    p.add_argument(
        "--eval_n", type=int, default=30, help="number of held-out test problems to evaluate"
    )

    p.add_argument("--predictor_ckpt", default="evaluation/predictor/checkpoints/bench_v1/best.pt")
    p.add_argument("--predictor_device", default="cpu")
    p.add_argument(
        "--global_mean_bias",
        type=float,
        default=float("nan"),
        help="constant bias subtracted from all sampling logprobs in global_mean group",
    )

    p.add_argument("--output", default="evaluation/results/bench")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载 predictor
    predictor: Optional[PredictorCorrector] = None
    if os.path.exists(args.predictor_ckpt):
        try:
            predictor = PredictorCorrector(args.predictor_ckpt, args.predictor_device)
        except Exception as e:
            logger.error(f"Failed to load predictor: {e}")
            raise
    else:
        logger.warning(f"Predictor ckpt not found: {args.predictor_ckpt}")

    # 决定运行哪些组
    group_spec = [
        ("reinforce_baseline", "reinforce", "baseline"),
        ("reinforce_predictor", "reinforce", "predictor"),
        ("reinforce_global_mean", "reinforce", "global_mean"),
        ("grpo_baseline", "grpo", "baseline"),
        ("grpo_predictor", "grpo", "predictor"),
        ("grpo_global_mean", "grpo", "global_mean"),
        ("grpo_oracle", "grpo", "oracle"),
    ]

    groups_input = args.groups
    if "all" in groups_input:
        selected = group_spec
    else:
        selected = [(n, a, c) for n, a, c in group_spec if n in groups_input]

    # 保存实验配置
    with open(output_dir / "bench_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # 确定 global_mean_bias
    global_mean_bias = args.global_mean_bias
    if math.isnan(global_mean_bias):
        global_mean_bias = 0.0
        logger.warning("--global_mean_bias not set; using 0.0 for global_mean group")

    all_results: Dict[str, List[Dict]] = {}

    for group_name, algo, correction in selected:
        if correction == "predictor" and predictor is None:
            logger.warning(f"Skipping {group_name}: predictor not available")
            continue

        records = asyncio.run(
            run_group(
                group_name=group_name,
                algo=algo,
                correction=correction,
                args=args,
                predictor=predictor,
                global_mean_bias=global_mean_bias,
                output_dir=output_dir,
            )
        )
        all_results[group_name] = records

        # 中间保存
        with open(output_dir / "all_results.json", "w") as f:
            json.dump(all_results, f, indent=2)

    print_summary(all_results)
    print(f"\n  Results: {output_dir}/")


if __name__ == "__main__":
    main()
