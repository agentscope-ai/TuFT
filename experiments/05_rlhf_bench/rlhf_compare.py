"""
RLHF Correction Comparison Experiment
======================================
三组对照实验，验证mismatch校正对RLHF训练效果的实际提升：

  Group A (baseline):   纯IS-GRPO，无任何校正
  Group B (global):     Global mean subtraction（从calibration数据计算常量）
  Group C (predictor):  MLP/Transformer predictor在线逐token校正

每轮记录：
  - mean_reward       当批奖励均值
  - train_loss        IS loss（近似，通过sampling/training logprob差异计算）
  - clip02            IS权重超出[0.8,1.2]的序列比例
  - std_cum_diff      per-seq累积logprob差异标准差（bias异质性）
  - mean_abs_diff     token级MAE

运行方式：
  # 单组运行（可并行）：
  python evaluation/rlhf_compare.py --group baseline --output results/baseline
  python evaluation/rlhf_compare.py --group global   --output results/global
  python evaluation/rlhf_compare.py --group predictor --predictor_ckpt \
      evaluation/predictor/checkpoints/transformer_22agents_v1/best.pt \
      --output results/predictor

  # 全部串行运行：
  python evaluation/rlhf_compare.py --run_all --output results/compare
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
from typing import Any, Dict, List, Optional

import numpy as np
import torch


# ── 路径设置：使用simulator的backend和tasks ──────────────────────────────────
_SIM_DIR = Path(__file__).parent / "simulator"
sys.path.insert(0, str(_SIM_DIR))
sys.path.insert(0, str(_SIM_DIR / "simulator"))

from simulator.backend.base import AdapterConfig  # noqa: E402
from simulator.backend.tinker_backend import TinkerBackend  # noqa: E402
from simulator.config import BackendConfig, TenantConfig  # noqa: E402
from simulator.tasks.gsm8k import GSM8KTask  # noqa: E402


# ── predictor路径 ─────────────────────────────────────────────────────────────
_PRED_DIR = Path(__file__).parent / "predictor"
sys.path.insert(0, str(_PRED_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rlhf_compare")


# ─────────────────────────────────────────────────────────────────────────────
# Mismatch metrics
# ─────────────────────────────────────────────────────────────────────────────
def compute_mismatch_metrics(
    sampling_lps: List[List[float]],
    training_lps: List[List[float]],
) -> Dict[str, float]:
    """Compute mismatch metrics between sampling and training logprobs."""
    all_deltas: List[float] = []
    seq_cum_diffs: List[float] = []

    for s_lps, t_lps in zip(sampling_lps, training_lps, strict=False):
        length = min(len(s_lps), len(t_lps))
        if length == 0:
            continue
        deltas = [s - t for s, t in zip(s_lps[:length], t_lps[:length], strict=False)]
        all_deltas.extend(deltas)
        seq_cum_diffs.append(sum(deltas))

    if not all_deltas:
        return {
            "mean_abs_diff": 0.0,
            "mean_diff": 0.0,
            "std_cum_diff": 0.0,
            "clip01": 0.0,
            "clip02": 0.0,
            "mean_cum_diff": 0.0,
        }

    arr = np.array(all_deltas)
    seq_arr = np.array(seq_cum_diffs)

    clip01_hi, clip01_lo = math.log(1.1), math.log(0.9)
    clip02_hi, clip02_lo = math.log(1.2), math.log(0.8)

    clip01 = float(np.mean((seq_arr > clip01_hi) | (seq_arr < clip01_lo)))
    clip02 = float(np.mean((seq_arr > clip02_hi) | (seq_arr < clip02_lo)))

    return {
        "mean_abs_diff": float(np.mean(np.abs(arr))),
        "mean_diff": float(np.mean(arr)),
        "std_cum_diff": float(np.std(seq_arr)),
        "mean_cum_diff": float(np.mean(seq_arr)),
        "clip01": clip01,
        "clip02": clip02,
        "n_seqs": len(seq_cum_diffs),
        "n_tokens": len(all_deltas),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Predictor wrapper
# ─────────────────────────────────────────────────────────────────────────────
class PredictorCorrector:
    """Loads a trained predictor and applies per-token correction."""

    def __init__(self, ckpt_path: str, device: str = "cpu"):
        from model import build_model

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        args = ckpt["args"]
        vocab_size = ckpt["vocab_size"]

        if args["model"] == "transformer":
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
            f"[predictor] Loaded {args['model']} from {ckpt_path} "
            f"(epoch={ckpt['epoch']}, vocab_size={vocab_size})"
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
        """返回校正后的 sampling_logprobs（逐 token 减去预测的 delta_hat）。"""
        corrected = []
        for resp_toks, samp_lps, n_prompt in zip(
            response_tokens_list, sampling_lps_list, n_prompt_tokens_list, strict=False
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
            delta_hat = delta_hat.squeeze(0).cpu().tolist()  # [T]

            # 校正后 logprob = sampling_lp - delta_hat
            corrected_lps = [s - d for s, d in zip(samp_lps[:T], delta_hat, strict=False)] + list(
                samp_lps[T:]
            )  # 超出max_seq_len的部分不校正
            corrected.append(corrected_lps)
        return corrected


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment runner
# ─────────────────────────────────────────────────────────────────────────────
class RLHFExperiment:
    def __init__(
        self,
        group: str,
        backend: TinkerBackend,
        task: GSM8KTask,
        adapter_id: str,
        config: TenantConfig,
        output_dir: Path,
        predictor: Optional[PredictorCorrector] = None,
        calibration_bias: float = 0.0,
    ):
        self.group = group
        self.backend = backend
        self.task = task
        self.adapter_id = adapter_id
        self.config = config
        self.output_dir = output_dir
        self.predictor = predictor
        self.calibration_bias = calibration_bias

        self.train_steps = 0
        self.log_records: List[Dict[str, Any]] = []
        self._tokenizer = backend.get_tokenizer()

    def _apply_correction(
        self,
        response_tokens_list: List[List[int]],
        sampling_lps_list: List[List[float]],
        n_prompt_tokens_list: List[int],
    ) -> List[List[float]]:
        """根据group类型对sampling logprobs应用校正。"""
        if self.group == "baseline":
            return sampling_lps_list

        elif self.group == "global":
            # 减去全局常量 calibration_bias
            return [[lp - self.calibration_bias for lp in lps] for lps in sampling_lps_list]

        elif self.group == "predictor":
            assert self.predictor is not None
            return self.predictor.correct_batch(
                response_tokens_list=response_tokens_list,
                sampling_lps_list=sampling_lps_list,
                n_prompt_tokens_list=n_prompt_tokens_list,
                temperature=self.config.temperature,
                lora_rank=self.config.lora_rank,
            )

        return sampling_lps_list

    async def run(self, num_steps: int, buffer_size: int) -> List[Dict[str, Any]]:
        """运行完整实验，返回每步的指标记录。"""
        logger.info(f"[{self.group}] Starting experiment: steps={num_steps}, buffer={buffer_size}")
        buffer: List[Dict[str, Any]] = []
        weight_version = await self.backend.sync_weights(self.adapter_id)

        log_path = self.output_dir / f"{self.group}_log.jsonl"
        log_fp = open(log_path, "w")

        while self.train_steps < num_steps:
            # ── 1. 采样 ──────────────────────────────────────────────────────
            prompt = self.task.get_prompt()
            prompt_tokens = self._tokenizer.encode(prompt.text, add_special_tokens=False)

            try:
                results = await self.backend.sample(
                    adapter_id=self.adapter_id,
                    prompt_tokens=prompt_tokens,
                    num_samples=1,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                )
            except Exception as e:
                logger.warning(f"[{self.group}] Sample failed: {e}")
                continue

            result = results[0]
            try:
                reward = self.task.compute_reward(result.text, prompt.metadata)
            except Exception:
                reward = 0.0

            buffer.append(
                {
                    "prompt_tokens": prompt_tokens,
                    "response_tokens": result.tokens,
                    "sampling_logprobs": result.logprobs,
                    "reward": reward,
                    "weight_version": weight_version,
                }
            )

            # ── 2. 缓冲区满时训练 ─────────────────────────────────────────
            if len(buffer) < buffer_size:
                continue

            batch = buffer[:buffer_size]
            buffer = buffer[buffer_size:]

            # ── 3. 计算训练侧 logprobs（用于测量mismatch） ────────────────
            training_lps_raw: List[List[float]] = []
            try:
                training_lps_raw = await self.backend.compute_training_logprobs(
                    self.adapter_id,
                    [
                        {
                            "prompt_tokens": item["prompt_tokens"],
                            "response_tokens": item["response_tokens"],
                        }
                        for item in batch
                    ],
                    temperature=self.config.temperature,
                )
            except Exception as e:
                logger.warning(f"[{self.group}] compute_training_logprobs failed: {e}")

            # ── 4. 计算原始 mismatch metrics（校正前，三组统一基准） ───────
            raw_metrics = {}
            if training_lps_raw:
                raw_metrics = compute_mismatch_metrics(
                    [item["sampling_logprobs"] for item in batch],
                    training_lps_raw,
                )

            # ── 5. 应用校正到 sampling_logprobs ──────────────────────────
            corrected_slps = self._apply_correction(
                response_tokens_list=[item["response_tokens"] for item in batch],
                sampling_lps_list=[item["sampling_logprobs"] for item in batch],
                n_prompt_tokens_list=[len(item["prompt_tokens"]) for item in batch],
            )

            # ── 6. 计算校正后 mismatch metrics ───────────────────────────
            corrected_metrics = {}
            if training_lps_raw:
                corrected_metrics = compute_mismatch_metrics(
                    corrected_slps,
                    training_lps_raw,
                )

            # ── 7. 计算 advantages ────────────────────────────────────────
            rewards = [item["reward"] for item in batch]
            mean_r = sum(rewards) / len(rewards)
            var_r = sum((r - mean_r) ** 2 for r in rewards) / max(len(rewards), 1)
            std_r = var_r**0.5
            if std_r < 1e-8:
                advantages = [0.0] * len(batch)
            else:
                advantages = [(r - mean_r) / (std_r + 1e-6) for r in rewards]

            # ── 8. 构建训练datum（使用校正后的logprobs） ──────────────────
            training_datums = [
                {
                    "prompt_tokens": item["prompt_tokens"],
                    "response_tokens": item["response_tokens"],
                    "sampling_logprobs": corr_lps,
                    "advantage": adv,
                }
                for item, corr_lps, adv in zip(batch, corrected_slps, advantages, strict=False)
            ]

            # ── 9. 执行训练步 ─────────────────────────────────────────────
            t0 = time.time()
            try:
                await self.backend.train_step(self.adapter_id, training_datums)
                train_duration = time.time() - t0
            except Exception as e:
                logger.error(f"[{self.group}] train_step failed: {e}")
                continue

            # ── 10. 同步权重 ──────────────────────────────────────────────
            weight_version = await self.backend.sync_weights(self.adapter_id)
            self.train_steps += 1

            # ── 11. 计算approximate IS loss（衡量训练质量） ───────────────
            # IS loss ≈ -mean( advantage * clip(w, 1-ε, 1+ε) )
            # 简化：记录corrected logprob与training logprob的均方差作为代理loss
            approx_loss = corrected_metrics.get("mean_abs_diff", float("nan"))

            # ── 12. 记录指标 ──────────────────────────────────────────────
            record = {
                "step": self.train_steps,
                "group": self.group,
                "mean_reward": mean_r,
                "reward_std": std_r,
                "train_duration_s": train_duration,
                # mismatch metrics (raw, 校正前，三组可比基准)
                "raw": {
                    "mean_abs_diff": raw_metrics.get("mean_abs_diff", float("nan")),
                    "mean_diff": raw_metrics.get("mean_diff", float("nan")),
                    "std_cum_diff": raw_metrics.get("std_cum_diff", float("nan")),
                    "clip01": raw_metrics.get("clip01", float("nan")),
                    "clip02": raw_metrics.get("clip02", float("nan")),
                },
                # mismatch metrics (corrected, 体现校正效果)
                "corrected": {
                    "mean_abs_diff": corrected_metrics.get("mean_abs_diff", float("nan")),
                    "mean_diff": corrected_metrics.get("mean_diff", float("nan")),
                    "std_cum_diff": corrected_metrics.get("std_cum_diff", float("nan")),
                    "clip01": corrected_metrics.get("clip01", float("nan")),
                    "clip02": corrected_metrics.get("clip02", float("nan")),
                },
                "approx_loss": approx_loss,
            }

            self.log_records.append(record)
            log_fp.write(json.dumps(record) + "\n")
            log_fp.flush()

            logger.info(
                f"[{self.group}] step={self.train_steps}/{num_steps} "
                f"reward={mean_r:.3f} "
                f"clip02_raw={raw_metrics.get('clip02', float('nan')):.3f} "
                f"clip02_corr={corrected_metrics.get('clip02', float('nan')):.3f} "
                f"std_cum={raw_metrics.get('std_cum_diff', float('nan')):.4f}"
            )

        log_fp.close()
        logger.info(
            f"[{self.group}] Experiment complete. {self.train_steps} steps logged to {log_path}"
        )
        return self.log_records


# ─────────────────────────────────────────────────────────────────────────────
# Calibration: compute global mean bias from existing logprob data
# ─────────────────────────────────────────────────────────────────────────────
def compute_calibration_bias(data_path: str, max_records: int = 5000) -> float:
    """从已有logprob数据文件中计算全局 mean(sampling_lp - training_lp)。"""
    all_deltas = []
    with open(data_path) as f:
        for i, line in enumerate(f):
            if i >= max_records:
                break
            try:
                rec = json.loads(line.strip())
                # 只用 staleness==0 的记录（framework mismatch干净）
                if rec.get("staleness", 0) != 0:
                    continue
                s_lps = rec.get("sampling_logprobs", [])
                t_lps = rec.get("training_logprobs", [])
                T = min(len(s_lps), len(t_lps))
                if T == 0:
                    continue
                for s, t in zip(s_lps[:T], t_lps[:T], strict=False):
                    all_deltas.append(s - t)
            except Exception:
                continue

    if not all_deltas:
        logger.warning("[calibration] No valid records found, using bias=0.0")
        return 0.0

    bias = float(np.mean(all_deltas))
    logger.info(
        f"[calibration] Computed global bias={bias:.6f} "
        f"from {len(all_deltas):,} tokens ({max_records} records limit)"
    )
    return bias


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="RLHF Correction Comparison")
    p.add_argument(
        "--group",
        choices=["baseline", "global", "predictor", "all"],
        default="all",
        help="which group to run (or 'all' for sequential)",
    )
    p.add_argument(
        "--output",
        default="evaluation/results/compare",
        help="output directory for logs and results",
    )

    # TuFT connection
    p.add_argument("--base_url", default="http://localhost:10610")
    p.add_argument("--api_key", default="tml-test-key")
    p.add_argument("--base_model", default="Qwen/Qwen3-4B")

    # Experiment config
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--num_steps", type=int, default=80, help="number of training steps per group")
    p.add_argument("--buffer_size", type=int, default=8, help="samples per training step")

    # Predictor
    p.add_argument(
        "--predictor_ckpt",
        default="evaluation/predictor/checkpoints/transformer_22agents_v1/best.pt",
        help="path to trained predictor checkpoint",
    )
    p.add_argument("--predictor_device", default="cpu", help="device for predictor inference")

    # Calibration data
    p.add_argument(
        "--calibration_data",
        default="evaluation/simulator/logprobs/tp2_fsdp1_id0_quant.jsonl",
        help="logprob data file for computing global bias",
    )

    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


async def run_single_group(
    group: str,
    args: argparse.Namespace,
    predictor: Optional[PredictorCorrector],
    calibration_bias: float,
    output_dir: Path,
) -> List[Dict[str, Any]]:
    """Run one group's experiment."""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"  Starting group: {group.upper()}")
    if group == "global":
        logger.info(f"  calibration_bias = {calibration_bias:.6f}")
    logger.info(f"{'=' * 60}\n")

    # ── 初始化 backend ────────────────────────────────────────────────────
    backend_cfg = BackendConfig(
        type="tinker",
        base_url=args.base_url,
        api_key=args.api_key,
        base_model=args.base_model,
    )
    backend = TinkerBackend()
    await backend.initialize(backend_cfg.to_dict())

    # ── 创建 adapter ──────────────────────────────────────────────────────
    adapter_id = f"compare_{group}_{int(time.time())}"
    adapter_cfg = AdapterConfig(
        lora_rank=args.lora_rank,
        learning_rate=args.learning_rate,
    )
    await backend.create_adapter(adapter_id, adapter_cfg)
    logger.info(f"[{group}] Created adapter: {adapter_id}")

    # ── 加载任务 ──────────────────────────────────────────────────────────
    task = GSM8KTask(seed=args.seed)
    task.load_dataset()

    # ── TenantConfig（仅用于存参数）────────────────────────────────────
    tenant_cfg = TenantConfig(
        id=adapter_id,
        task="gsm8k",
        request_rate=2.0,
        buffer_size=args.buffer_size,
        num_train_steps=args.num_steps,
        lora_rank=args.lora_rank,
        learning_rate=args.learning_rate,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        sync_mode=True,
    )

    # ── 创建实验实例并运行 ────────────────────────────────────────────────
    exp = RLHFExperiment(
        group=group,
        backend=backend,
        task=task,
        adapter_id=adapter_id,
        config=tenant_cfg,
        output_dir=output_dir,
        predictor=predictor if group == "predictor" else None,
        calibration_bias=calibration_bias if group == "global" else 0.0,
    )

    try:
        records = await exp.run(
            num_steps=args.num_steps,
            buffer_size=args.buffer_size,
        )
    finally:
        try:
            await backend.cleanup(adapter_id)
            logger.info(f"[{group}] Cleaned up adapter {adapter_id}")
        except Exception as e:
            logger.warning(f"[{group}] Cleanup failed: {e}")

    return records


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 计算 calibration bias（供 global 组使用） ─────────────────────────
    calibration_bias = 0.0
    if os.path.exists(args.calibration_data):
        calibration_bias = compute_calibration_bias(args.calibration_data)
    else:
        logger.warning(f"Calibration data not found: {args.calibration_data}, using bias=0.0")

    # ── 加载 predictor（供 predictor 组使用） ──────────────────────────────
    predictor: Optional[PredictorCorrector] = None
    if os.path.exists(args.predictor_ckpt):
        try:
            predictor = PredictorCorrector(args.predictor_ckpt, args.predictor_device)
        except Exception as e:
            logger.error(f"Failed to load predictor: {e}")
            if args.group in ("predictor", "all"):
                raise
    else:
        logger.warning(f"Predictor checkpoint not found: {args.predictor_ckpt}")

    # ── 保存实验配置 ──────────────────────────────────────────────────────
    exp_config = vars(args)
    exp_config["calibration_bias"] = calibration_bias
    with open(output_dir / "experiment_config.json", "w") as f:
        json.dump(exp_config, f, indent=2)

    # ── 决定运行哪些组 ────────────────────────────────────────────────────
    groups_to_run = ["baseline", "global", "predictor"] if args.group == "all" else [args.group]

    all_results: Dict[str, List[Dict]] = {}

    for group in groups_to_run:
        if group == "predictor" and predictor is None:
            logger.warning("Skipping predictor group: no checkpoint available")
            continue

        records = asyncio.run(
            run_single_group(
                group=group,
                args=args,
                predictor=predictor,
                calibration_bias=calibration_bias,
                output_dir=output_dir,
            )
        )
        all_results[group] = records

        # 每组结束后保存中间结果
        with open(output_dir / "all_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"[{group}] Saved results to {output_dir}/all_results.json")

    # ── 打印汇总对比 ──────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  EXPERIMENT SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Steps: {args.num_steps}  Buffer: {args.buffer_size}  Model: {args.base_model}")
    print(f"  Calibration bias: {calibration_bias:.6f}")
    print()

    header = f"{'metric':<25}" + "".join(f"{g:>15}" for g in groups_to_run if g in all_results)
    print(header)
    print("-" * len(header))

    def last_n_mean(records, key, path=None, n=10):
        vals = []
        for r in records[-n:]:
            try:
                v = r[path][key] if path else r[key]
                if not math.isnan(v):
                    vals.append(v)
            except Exception:
                pass
        return float(np.mean(vals)) if vals else float("nan")

    metrics_to_show = [
        ("mean_reward (last10)", "mean_reward", None),
        ("raw clip02 (last10)", "clip02", "raw"),
        ("corr clip02 (last10)", "clip02", "corrected"),
        ("raw std_cum (last10)", "std_cum_diff", "raw"),
        ("corr std_cum (last10)", "std_cum_diff", "corrected"),
        ("raw mean_abs (last10)", "mean_abs_diff", "raw"),
        ("corr mean_abs (last10)", "mean_abs_diff", "corrected"),
    ]

    for label, key, path in metrics_to_show:
        row = f"  {label:<23}"
        for g in groups_to_run:
            if g not in all_results:
                row += f"{'n/a':>15}"
                continue
            v = last_n_mean(all_results[g], key, path)
            row += f"{v:>15.4f}"
        print(row)

    print(f"\n  Full results saved to: {output_dir}/all_results.json")
    print(f"  Per-group logs:        {output_dir}/<group>_log.jsonl")


if __name__ == "__main__":
    main()
