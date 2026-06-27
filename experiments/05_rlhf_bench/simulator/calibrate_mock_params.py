"""Calibrate MockBackendParams from a live TuFT service.

Measures latency for:
  S1 - Sampling: prefill + decode coefficients (max_tokens × concurrency sweep)
  S2 - Sampling: multi-adapter batch overhead (n_distinct_adapters sweep)
  S3 - Sampling: batch-size non-linearity (dense concurrency sweep)
  T1 - Training: forward+backward latency (total_tokens × lora_rank sweep)
  T2 - Training: optim step latency (lora_rank sweep)
  T3 - Training: weight sync full pipeline (lora_rank sweep)

Each probe is dumped to JSONL immediately to survive interruptions.
Re-run with --resume to re-fit from existing data without new probes.

Usage
-----
    cd /mnt/nas/hanzhang.yhz/evaluation
    .venv/bin/python simulator/calibrate_mock_params.py \\
        --base-url  http://localhost:10610 \\
        --api-key   tml-test-key \\
        --base-model Qwen/Qwen3-4B \\
        --output    calibration_results.jsonl \\
        --repeats   3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Path setup so we can import simulator modules
# ---------------------------------------------------------------------------
_SIM_DIR = Path(__file__).parent
sys.path.insert(0, str(_SIM_DIR))

from simulator.backend.base import AdapterConfig
from simulator.backend.tinker_backend import TinkerBackend
from simulator.tasks import create_task


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("calibrate")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ProbeResult:
    """One measured data point.  `kind` identifies the sweep it belongs to."""

    kind: str  # "S1" | "S2" | "S3" | "T1" | "T2" | "T3"
    probe_id: int

    # ---- Sampling fields (S1/S2/S3) ----
    concurrency: int  # number of concurrent sample() calls
    total_prefill_tokens: int  # sum of prompt lengths across all requests
    max_tokens: int  # max_tokens passed to sample()
    n_distinct_adapters: int  # how many different adapter_ids in the batch
    actual_output_tokens: int  # mean actual output tokens (across requests)

    # ---- Training fields (T1/T2/T3) ----
    lora_rank: int  # LoRA rank of the adapter used
    total_train_tokens: int  # total tokens in the training batch (T1)

    # ---- Timing ----
    wall_latency_s: float  # measured wall-clock seconds
    ts: float  # unix timestamp


# ---------------------------------------------------------------------------
# Calibrator
# ---------------------------------------------------------------------------

# Number of adapters pre-created for multi-adapter sweep (S2)
_N_MULTI_ADAPTERS = 8

# Tasks to pull real prompts from
_TASK_NAMES = ["gsm8k", "countdown", "math"]
_PROMPTS_PER_TASK = 5  # prompts kept in memory


class Calibrator:
    # ---- Sweep configs ----

    # S1: prefill + decode coefficients
    # max_tokens extended to 1024/2048 to cover real-world long-generation scenarios
    S1_MAX_TOKENS = [64, 128, 256, 512, 1024, 2048]
    S1_CONCURRENCY = [1, 2, 4, 8, 16]

    # S2: multi-adapter overhead  (total concurrency fixed = 8)
    S2_N_ADAPTERS = [1, 2, 4, 8]
    S2_CONCURRENCY = 8
    S2_MAX_TOKENS = 512  # use a longer generation for more stable signal

    # S3: batch-size non-linearity (max_tokens fixed = 512)
    S3_CONCURRENCY = [1, 2, 4, 8, 16, 24, 32]
    S3_MAX_TOKENS = 512

    # T1: fwd+bwd (total_tokens × rank)
    T1_TOTAL_TOKENS = [256, 512, 1024, 2048, 4096]
    T1_RANKS = [8, 16]
    # Number of warm-up calls before recording T1 (discards JIT/cache warm-up latency)
    T1_WARMUP = 1

    # T2: optim step (rank)
    T2_RANKS = [8, 16]

    # T3: weight sync (rank)
    T3_RANKS = [8, 16]
    # Discard first T3 call per rank (first sync always slower due to file-system cold cache)
    T3_WARMUP = 1

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str],
        base_model: str,
        output_path: Path,
        repeats: int = 3,
        temperature: float = 0.7,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.base_model = base_model
        self.output_path = output_path
        self.repeats = repeats
        self.temperature = temperature

        self._backend = TinkerBackend()
        self._tokenizer: Any = None

        # adapter_id -> lora_rank mapping
        # "probe_r8"  = rank-8  base adapter used for S1/S3/T1/T2/T3
        # "probe_r16" = rank-16 base adapter used for T1/T2/T3
        # "probe_s2_0" ... "probe_s2_7" = rank-8 adapters for S2 sweep
        self._adapters: Dict[str, int] = {}  # adapter_id -> rank

        self._probe_counter = 0
        self._results: List[ProbeResult] = []

        # Flat list of (task_name, prompt_text, token_ids) loaded once
        self._prompts: List[Tuple[str, str, List[int]]] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def setup(self) -> None:
        logger.info("Connecting to %s ...", self.base_url)
        await self._backend.initialize(
            {
                "base_url": self.base_url,
                "api_key": self.api_key,
                "base_model": self.base_model,
            }
        )

        # Create base adapters for S1/S3/T* sweeps
        for rank in [8, 16]:
            aid = f"probe_r{rank}"
            logger.info("Creating adapter %s (rank=%d) ...", aid, rank)
            await self._backend.create_adapter(
                aid, AdapterConfig(lora_rank=rank, learning_rate=1e-4)
            )
            self._adapters[aid] = rank

        # Sync rank-8 adapter so S1/S2/S3 have a sampling client immediately
        logger.info("Syncing probe_r8 weights ...")
        await self._backend.sync_weights("probe_r8")

        # Create multi-adapter pool for S2 sweep (all rank-8, max_loras=16 so plenty of room)
        for i in range(_N_MULTI_ADAPTERS):
            aid = f"probe_s2_{i}"
            logger.info("Creating S2 adapter %s ...", aid)
            await self._backend.create_adapter(aid, AdapterConfig(lora_rank=8, learning_rate=1e-4))
            await self._backend.sync_weights(aid)
            self._adapters[aid] = 8

        self._tokenizer = self._backend.get_tokenizer()

        # Load real prompts
        logger.info("Loading real prompts from tasks ...")
        self._load_prompts()

        logger.info(
            "Setup complete. %d adapters, %d prompts loaded.",
            len(self._adapters),
            len(self._prompts),
        )

    async def teardown(self) -> None:
        for aid in list(self._adapters):
            try:
                await self._backend.cleanup(aid)
            except Exception as e:
                logger.warning("Cleanup failed for %s: %s", aid, e)

    # ------------------------------------------------------------------
    # Prompt loading
    # ------------------------------------------------------------------

    def _load_prompts(self) -> None:
        for task_name in _TASK_NAMES:
            try:
                task = create_task(task_name)
                for _ in range(_PROMPTS_PER_TASK):
                    p = task.get_prompt()
                    tokens = self._tokenizer.encode(p.text, add_special_tokens=False)
                    self._prompts.append((task_name, p.text, tokens))
            except Exception as e:
                logger.warning("Could not load task '%s': %s", task_name, e)

        if not self._prompts:
            raise RuntimeError("No prompts loaded; check dataset / HF_TOKEN.")

        self._prompts.sort(key=lambda x: len(x[2]))
        token_lens = [len(t) for _, _, t in self._prompts]
        logger.info(
            "Prompts: %d total, token len range %d–%d",
            len(self._prompts),
            min(token_lens),
            max(token_lens),
        )

    def _pick_prompts(self, n: int) -> List[Tuple[str, List[int]]]:
        """Round-robin pick n (task_name, tokens) pairs."""
        return [
            (self._prompts[i % len(self._prompts)][0], self._prompts[i % len(self._prompts)][2])
            for i in range(n)
        ]

    # ------------------------------------------------------------------
    # Low-level fire helpers
    # ------------------------------------------------------------------

    async def _fire_sampling_batch(
        self,
        prompt_tokens_list: List[List[int]],
        adapter_ids: List[str],
        max_tokens: int,
    ) -> Tuple[float, int]:
        """Concurrent sample(), return (wall_s, mean_actual_output_tokens)."""
        t0 = time.perf_counter()

        async def _one(pt: List[int], aid: str):
            return await self._backend.sample(
                adapter_id=aid,
                prompt_tokens=pt,
                num_samples=1,
                max_tokens=max_tokens,
                temperature=self.temperature,
            )

        results = await asyncio.gather(
            *[_one(pt, aid) for pt, aid in zip(prompt_tokens_list, adapter_ids)]
        )
        wall = time.perf_counter() - t0
        mean_out = sum(len(r[0].tokens) for r in results) / len(results)
        return wall, int(round(mean_out))

    async def _fire_train_step(
        self,
        adapter_id: str,
        total_tokens: int,
    ) -> float:
        """Run one train_step with synthetic data, return wall_s."""
        datums = self._build_synthetic_datums(total_tokens)
        t0 = time.perf_counter()
        await self._backend.train_step(adapter_id, datums)
        return time.perf_counter() - t0

    async def _fire_sync_weights(self, adapter_id: str) -> float:
        """Run sync_weights, return wall_s (full pipeline: save + vLLM hot-load)."""
        t0 = time.perf_counter()
        await self._backend.sync_weights(adapter_id)
        return time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Synthetic datum builder (for training probes, no real dataset needed)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_synthetic_datums(total_tokens: int) -> List[Dict[str, Any]]:
        """Build a list of training datums whose token count sums to total_tokens.

        We use a single datum with prompt:response split 50/50 to keep it simple.
        Tokens are random integers in vocab range [1, 32000).
        """
        half = total_tokens // 2
        prompt_tokens = [random.randint(1, 31999) for _ in range(half)]
        response_tokens = [random.randint(1, 31999) for _ in range(total_tokens - half)]
        logprobs = [random.gauss(-2.0, 0.5) for _ in response_tokens]
        return [
            {
                "prompt_tokens": prompt_tokens,
                "response_tokens": response_tokens,
                "sampling_logprobs": logprobs,
                "advantage": random.gauss(0.0, 1.0),
            }
        ]

    # ------------------------------------------------------------------
    # Dump helper
    # ------------------------------------------------------------------

    def _record(self, result: ProbeResult) -> None:
        self._results.append(result)
        with open(self.output_path, "a") as f:
            f.write(json.dumps(asdict(result)) + "\n")
        self._probe_counter += 1

    def _blank_sampling_fields(self) -> Dict:
        return dict(
            concurrency=0,
            total_prefill_tokens=0,
            max_tokens=0,
            n_distinct_adapters=0,
            actual_output_tokens=0,
        )

    def _blank_training_fields(self) -> Dict:
        return dict(lora_rank=0, total_train_tokens=0)

    # ------------------------------------------------------------------
    # Sweep S1: prefill + decode coefficients
    # ------------------------------------------------------------------

    async def sweep_S1(self) -> None:
        logger.info("=== Sweep S1: prefill+decode coefficients ===")
        aid = "probe_r8"
        total = len(self.S1_MAX_TOKENS) * len(self.S1_CONCURRENCY) * self.repeats
        done = 0
        for max_tokens in self.S1_MAX_TOKENS:
            for concurrency in self.S1_CONCURRENCY:
                for rep in range(self.repeats):
                    picked = self._pick_prompts(concurrency)
                    pts = [t for _, t in picked]
                    total_prefill = sum(len(t) for t in pts)
                    aids = [aid] * concurrency
                    try:
                        wall, mean_out = await self._fire_sampling_batch(pts, aids, max_tokens)
                    except Exception as e:
                        logger.warning("S1 probe failed: %s", e)
                        done += 1
                        continue
                    self._record(
                        ProbeResult(
                            kind="S1",
                            probe_id=self._probe_counter,
                            concurrency=concurrency,
                            total_prefill_tokens=total_prefill,
                            max_tokens=max_tokens,
                            n_distinct_adapters=1,
                            actual_output_tokens=mean_out,
                            **self._blank_training_fields(),
                            wall_latency_s=wall,
                            ts=time.time(),
                        )
                    )
                    done += 1
                    logger.info(
                        "S1 [%3d/%d] max_tokens=%4d concurrency=%2d rep=%d "
                        "prefill=%5d latency=%.3fs",
                        done,
                        total,
                        max_tokens,
                        concurrency,
                        rep,
                        total_prefill,
                        wall,
                    )

    # ------------------------------------------------------------------
    # Sweep S2: multi-adapter batch overhead
    # ------------------------------------------------------------------

    async def sweep_S2(self) -> None:
        logger.info("=== Sweep S2: multi-adapter overhead ===")
        total = len(self.S2_N_ADAPTERS) * self.repeats
        done = 0
        for n_adapters in self.S2_N_ADAPTERS:
            # Distribute S2_CONCURRENCY=8 requests across n_adapters round-robin
            aids = [f"probe_s2_{i % n_adapters}" for i in range(self.S2_CONCURRENCY)]
            for rep in range(self.repeats):
                picked = self._pick_prompts(self.S2_CONCURRENCY)
                pts = [t for _, t in picked]
                total_prefill = sum(len(t) for t in pts)
                try:
                    wall, mean_out = await self._fire_sampling_batch(pts, aids, self.S2_MAX_TOKENS)
                except Exception as e:
                    logger.warning("S2 probe failed: %s", e)
                    done += 1
                    continue
                self._record(
                    ProbeResult(
                        kind="S2",
                        probe_id=self._probe_counter,
                        concurrency=self.S2_CONCURRENCY,
                        total_prefill_tokens=total_prefill,
                        max_tokens=self.S2_MAX_TOKENS,
                        n_distinct_adapters=n_adapters,
                        actual_output_tokens=mean_out,
                        **self._blank_training_fields(),
                        wall_latency_s=wall,
                        ts=time.time(),
                    )
                )
                done += 1
                logger.info(
                    "S2 [%3d/%d] n_adapters=%d rep=%d prefill=%5d latency=%.3fs",
                    done,
                    total,
                    n_adapters,
                    rep,
                    total_prefill,
                    wall,
                )

    # ------------------------------------------------------------------
    # Sweep S3: batch-size non-linearity
    # ------------------------------------------------------------------

    async def sweep_S3(self) -> None:
        logger.info("=== Sweep S3: batch-size non-linearity ===")
        aid = "probe_r8"
        total = len(self.S3_CONCURRENCY) * self.repeats
        done = 0
        for concurrency in self.S3_CONCURRENCY:
            for rep in range(self.repeats):
                picked = self._pick_prompts(concurrency)
                pts = [t for _, t in picked]
                total_prefill = sum(len(t) for t in pts)
                aids = [aid] * concurrency
                try:
                    wall, mean_out = await self._fire_sampling_batch(pts, aids, self.S3_MAX_TOKENS)
                except Exception as e:
                    logger.warning("S3 probe failed: %s", e)
                    done += 1
                    continue
                self._record(
                    ProbeResult(
                        kind="S3",
                        probe_id=self._probe_counter,
                        concurrency=concurrency,
                        total_prefill_tokens=total_prefill,
                        max_tokens=self.S3_MAX_TOKENS,
                        n_distinct_adapters=1,
                        actual_output_tokens=mean_out,
                        **self._blank_training_fields(),
                        wall_latency_s=wall,
                        ts=time.time(),
                    )
                )
                done += 1
                logger.info(
                    "S3 [%3d/%d] concurrency=%2d rep=%d prefill=%5d latency=%.3fs",
                    done,
                    total,
                    concurrency,
                    rep,
                    total_prefill,
                    wall,
                )

    # ------------------------------------------------------------------
    # Sweep T1: fwd+bwd latency
    # ------------------------------------------------------------------

    async def sweep_T1(self) -> None:
        logger.info("=== Sweep T1: training fwd+bwd ===")
        total = len(self.T1_TOTAL_TOKENS) * len(self.T1_RANKS) * self.repeats
        done = 0
        for rank in self.T1_RANKS:
            aid = f"probe_r{rank}"
            # Warm-up: discard first T1_WARMUP calls to avoid JIT/cache effects
            for wu in range(self.T1_WARMUP):
                try:
                    await self._fire_train_step(aid, self.T1_TOTAL_TOKENS[0])
                    logger.info("T1 warm-up rank=%d wu=%d done", rank, wu)
                except Exception as e:
                    logger.warning("T1 warm-up failed: %s", e)
            for total_tokens in self.T1_TOTAL_TOKENS:
                for rep in range(self.repeats):
                    try:
                        wall = await self._fire_train_step(aid, total_tokens)
                    except Exception as e:
                        logger.warning("T1 probe failed: %s", e)
                        done += 1
                        continue
                    self._record(
                        ProbeResult(
                            kind="T1",
                            probe_id=self._probe_counter,
                            **self._blank_sampling_fields(),
                            lora_rank=rank,
                            total_train_tokens=total_tokens,
                            wall_latency_s=wall,
                            ts=time.time(),
                        )
                    )
                    done += 1
                    logger.info(
                        "T1 [%3d/%d] rank=%2d total_tokens=%5d rep=%d latency=%.3fs",
                        done,
                        total,
                        rank,
                        total_tokens,
                        rep,
                        wall,
                    )

    # ------------------------------------------------------------------
    # Sweep T2: optim step latency
    # ------------------------------------------------------------------

    async def sweep_T2(self) -> None:
        """Isolate optim step: measure train_step (fwd+bwd+optim) minus fwd-only.

        TinkerBackend.train_step always includes optim; we can't call them
        separately via the public SDK.  So we take two consecutive measurements:
          - train_step (full): fwd + bwd + optim
          - We already have T1 data for fwd+bwd at a fixed token count.
        Here we just record repeated full train_step at a SMALL token count
        (256) so the fwd+bwd contribution is minimal and the optim fixed cost
        dominates.  The fwd coefficient from T1 can then be subtracted.
        """
        logger.info("=== Sweep T2: optim step (via small-batch train_step) ===")
        SMALL_TOKENS = 128  # tiny batch → optim dominates
        total = len(self.T2_RANKS) * self.repeats
        done = 0
        for rank in self.T2_RANKS:
            aid = f"probe_r{rank}"
            for rep in range(self.repeats):
                try:
                    wall = await self._fire_train_step(aid, SMALL_TOKENS)
                except Exception as e:
                    logger.warning("T2 probe failed: %s", e)
                    done += 1
                    continue
                self._record(
                    ProbeResult(
                        kind="T2",
                        probe_id=self._probe_counter,
                        **self._blank_sampling_fields(),
                        lora_rank=rank,
                        total_train_tokens=SMALL_TOKENS,
                        wall_latency_s=wall,
                        ts=time.time(),
                    )
                )
                done += 1
                logger.info(
                    "T2 [%3d/%d] rank=%2d rep=%d latency=%.3fs",
                    done,
                    total,
                    rank,
                    rep,
                    wall,
                )

    # ------------------------------------------------------------------
    # Sweep T3: weight sync full pipeline
    # ------------------------------------------------------------------

    async def sweep_T3(self) -> None:
        logger.info("=== Sweep T3: weight sync full pipeline ===")
        total = len(self.T3_RANKS) * self.repeats
        done = 0
        for rank in self.T3_RANKS:
            aid = f"probe_r{rank}"
            # Warm-up: discard first T3_WARMUP calls (filesystem cold-cache effect)
            for wu in range(self.T3_WARMUP):
                try:
                    await self._fire_sync_weights(aid)
                    logger.info("T3 warm-up rank=%d wu=%d done", rank, wu)
                except Exception as e:
                    logger.warning("T3 warm-up failed: %s", e)
            for rep in range(self.repeats):
                try:
                    wall = await self._fire_sync_weights(aid)
                except Exception as e:
                    logger.warning("T3 probe failed: %s", e)
                    done += 1
                    continue
                self._record(
                    ProbeResult(
                        kind="T3",
                        probe_id=self._probe_counter,
                        **self._blank_sampling_fields(),
                        lora_rank=rank,
                        total_train_tokens=0,
                        wall_latency_s=wall,
                        ts=time.time(),
                    )
                )
                done += 1
                logger.info(
                    "T3 [%3d/%d] rank=%2d rep=%d latency=%.3fs",
                    done,
                    total,
                    rank,
                    rep,
                    wall,
                )

    # ------------------------------------------------------------------
    # Run all sweeps
    # ------------------------------------------------------------------

    async def run(self) -> List[ProbeResult]:
        await self.sweep_S1()
        await self.sweep_S2()
        await self.sweep_S3()
        await self.sweep_T1()
        await self.sweep_T2()
        await self.sweep_T3()
        logger.info("All sweeps complete. %d probes recorded.", len(self._results))
        return self._results


# ---------------------------------------------------------------------------
# OLS fitting
# ---------------------------------------------------------------------------


def fit_params(results: List[ProbeResult]) -> Dict[str, Any]:
    try:
        import numpy as np
    except ImportError:
        logger.error("numpy not available.")
        return {}

    out: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Sampling fit (S1 + S2 + S3 combined)
    # Strategy (Plan A): fit only on rows where actual_output_tokens <= 512.
    # Beyond that, vLLM's KV-cache pressure causes super-linear latency that
    # a linear mock cannot express — those points would corrupt the coefficients.
    # The mock scheduler handles long-generation slowdown implicitly via
    # accumulated token counts rather than a per-call correction.
    #
    # Model: latency_us = A*total_prefill + B*actual_output_tokens + C*n_adapters + D
    # ------------------------------------------------------------------
    _S_TOKEN_CUTOFF = 512
    s_all = [r for r in results if r.kind in ("S1", "S2", "S3")]
    s_rows = [r for r in s_all if r.actual_output_tokens <= _S_TOKEN_CUTOFF]
    n_dropped_s = len(s_all) - len(s_rows)
    if n_dropped_s:
        logger.info(
            "Sampling fit: dropped %d rows with actual_output_tokens > %d (super-linear region)",
            n_dropped_s,
            _S_TOKEN_CUTOFF,
        )
    if len(s_rows) >= 4:
        X = np.array(
            [
                [r.total_prefill_tokens, r.actual_output_tokens, r.n_distinct_adapters, 1.0]
                for r in s_rows
            ]
        )
        y = np.array([r.wall_latency_s * 1e6 for r in s_rows])
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        prefill_us, decode_us, adapter_switch_us, intercept_us = beta
        y_pred = X @ beta
        r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)

        # Non-linearity: check if adding log(concurrency) helps
        conc = np.array([r.concurrency for r in s_rows], dtype=float)
        X_nl = np.column_stack([X, np.log1p(conc)])
        beta_nl, _, _, _ = np.linalg.lstsq(X_nl, y, rcond=None)
        y_pred_nl = X_nl @ beta_nl
        r2_nl = 1 - np.sum((y - y_pred_nl) ** 2) / np.sum((y - np.mean(y)) ** 2)

        logger.info("--- Sampling OLS (actual_output_tokens <= %d) ---", _S_TOKEN_CUTOFF)
        logger.info("  n_rows (after filter)      = %d / %d", len(s_rows), len(s_all))
        logger.info("  prefill_us_per_token       = %.3f", prefill_us)
        logger.info("  decode_us_per_actual_token = %.3f", decode_us)
        logger.info("  adapter_switch_ms          = %.3f", adapter_switch_us / 1000)
        logger.info("  intercept_us               = %.1f", intercept_us)
        logger.info("  R² (linear)                = %.4f", r2)
        logger.info("  R² (+ log concurrency)     = %.4f  (delta=%.4f)", r2_nl, r2_nl - r2)
        if r2_nl - r2 > 0.02:
            logger.info(
                "  >> Non-linearity detected (ΔR²>0.02); log(concurrency) term is significant"
            )
        if prefill_us < 0 or decode_us < 0:
            logger.warning(
                "  >> Negative coefficient(s)! prefill=%.3f decode=%.3f"
                " — may need more varied data.",
                prefill_us,
                decode_us,
            )

        out["sampling_prefill_us_per_token"] = max(float(prefill_us), 0.1)
        out["sampling_decode_us_per_token"] = max(float(decode_us), 0.1)
        out["sampling_adapter_switch_ms"] = max(float(adapter_switch_us) / 1000, 0.0)
        out["sampling_intercept_us"] = float(intercept_us)
        out["sampling_r2_linear"] = float(r2)
        out["sampling_r2_nonlinear"] = float(r2_nl)
        out["sampling_n_samples"] = len(s_rows)
        out["sampling_n_total"] = len(s_all)
        out["sampling_token_cutoff"] = _S_TOKEN_CUTOFF
    else:
        logger.warning("Not enough sampling probes after filtering (need ≥4, got %d).", len(s_rows))

    # ------------------------------------------------------------------
    # Training fit (T1): latency_us = A * total_tokens + B (per rank)
    # Robust: drop top-1 outlier per (rank, token_count) group before fitting
    # to eliminate JIT/thermal spikes that inflate variance.
    # ------------------------------------------------------------------
    for rank in [8, 16]:
        t1_rows = [r for r in results if r.kind == "T1" and r.lora_rank == rank]
        if len(t1_rows) >= 3:
            # Per-token_count group: drop the single highest latency point
            from itertools import groupby

            robust_rows = []
            t1_sorted = sorted(t1_rows, key=lambda r: r.total_train_tokens)
            for _tok, grp in groupby(t1_sorted, key=lambda r: r.total_train_tokens):
                grp_list = list(grp)
                if len(grp_list) > 2:
                    # Remove single max outlier
                    grp_list = sorted(grp_list, key=lambda r: r.wall_latency_s)[:-1]
                robust_rows.extend(grp_list)
            n_dropped = len(t1_rows) - len(robust_rows)
            if n_dropped:
                logger.info("T1 rank=%d: dropped %d outlier(s)", rank, n_dropped)
            X = np.array([[r.total_train_tokens, 1.0] for r in robust_rows])
            y = np.array([r.wall_latency_s * 1e6 for r in robust_rows])
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            fwd_bwd_us, fixed_us = beta
            y_pred = X @ beta
            r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
            logger.info(
                "--- Training OLS (rank=%d, n=%d after outlier removal) ---", rank, len(robust_rows)
            )
            logger.info("  fwd_bwd_us_per_token = %.3f", fwd_bwd_us)
            logger.info("  fixed_overhead_us    = %.1f", fixed_us)
            logger.info("  R²                   = %.4f", r2)
            out[f"training_fwd_bwd_us_per_token_r{rank}"] = max(float(fwd_bwd_us), 1.0)
            out[f"training_fixed_overhead_us_r{rank}"] = float(fixed_us)
            out[f"training_r2_r{rank}"] = float(r2)
        else:
            logger.warning("Not enough T1 probes for rank=%d.", rank)

    # ------------------------------------------------------------------
    # T2: optim step median per rank
    # ------------------------------------------------------------------
    for rank in [8, 16]:
        t2_rows = [r for r in results if r.kind == "T2" and r.lora_rank == rank]
        if t2_rows:
            vals = sorted(r.wall_latency_s for r in t2_rows)
            median_ms = vals[len(vals) // 2] * 1000
            logger.info("--- Optim step (rank=%d) median=%.1f ms ---", rank, median_ms)
            out[f"training_optim_ms_r{rank}"] = median_ms
        else:
            logger.warning("No T2 probes for rank=%d.", rank)

    # ------------------------------------------------------------------
    # T3: weight sync median per rank
    # ------------------------------------------------------------------
    for rank in [8, 16]:
        t3_rows = [r for r in results if r.kind == "T3" and r.lora_rank == rank]
        if t3_rows:
            vals = sorted(r.wall_latency_s for r in t3_rows)
            median_ms = vals[len(vals) // 2] * 1000
            logger.info("--- Weight sync (rank=%d) median=%.1f ms ---", rank, median_ms)
            out[f"training_save_weights_ms_r{rank}"] = median_ms
        else:
            logger.warning("No T3 probes for rank=%d.", rank)

    return out


# ---------------------------------------------------------------------------
# YAML output
# ---------------------------------------------------------------------------


def print_yaml_snippet(params: Dict[str, Any], note: str = "") -> None:
    # Use rank-8 as default; print rank-16 values as comments
    fwd_r8 = params.get("training_fwd_bwd_us_per_token_r8", 500.0)
    fwd_r16 = params.get("training_fwd_bwd_us_per_token_r16", fwd_r8 * 2)
    opt_r8 = params.get("training_optim_ms_r8", 20.0)
    opt_r16 = params.get("training_optim_ms_r16", opt_r8)
    sync_r8 = params.get("training_save_weights_ms_r8", 80.0)
    sync_r16 = params.get("training_save_weights_ms_r16", sync_r8 * 2)

    nl_note = ""
    if params.get("sampling_r2_nonlinear", 0) - params.get("sampling_r2_linear", 0) > 0.02:
        nl_note = "  # WARNING: non-linearity detected in batch; linear model is an approximation"

    print("\n" + "=" * 65)
    print("# MockBackendParams — calibrated from live TuFT service")
    if note:
        print(f"# {note}")
    print(
        f"# Sampling R²={params.get('sampling_r2_linear', float('nan')):.3f}  "
        f"n={params.get('sampling_n_samples', 0)}"
    )
    print(
        f"# Training R² r8={params.get('training_r2_r8', float('nan')):.3f}  "
        f"r16={params.get('training_r2_r16', float('nan')):.3f}"
    )
    print()
    print("mock_params:  # paste under supported_models entry")
    print("  virtual_time_scale: 100.0")
    print(
        f"  sampling_prefill_us_per_token: {params.get('sampling_prefill_us_per_token', 10.0):.2f}"
    )
    print(
        f"  sampling_decode_us_per_token:  {params.get('sampling_decode_us_per_token', 30.0):.2f}"
    )
    print(f"  sampling_adapter_switch_ms:    {params.get('sampling_adapter_switch_ms', 5.0):.2f}")
    print(f"  sampling_batch_accum_ms:       0.5{nl_note}")
    print(f"  training_fwd_bwd_us_per_token: {fwd_r8:.2f}  # rank-16: {fwd_r16:.2f}")
    print(f"  training_optim_ms:             {opt_r8:.1f}  # rank-16: {opt_r16:.1f}")
    print(f"  training_save_weights_ms:      {sync_r8:.1f}  # rank-16: {sync_r16:.1f}")
    print("  training_gpu_slots: 1")
    print("=" * 65 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


async def _main(args: argparse.Namespace) -> None:
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.resume:
        # Load existing JSONL and re-fit only
        results: List[ProbeResult] = []
        if output_path.exists():
            with open(output_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            results.append(ProbeResult(**json.loads(line)))
                        except Exception:
                            pass
        logger.info("Resume mode: loaded %d results from %s", len(results), output_path)
        params = fit_params(results)
        print_yaml_snippet(params, f"resume from {output_path}")
        fit_path = output_path.with_suffix(".fit.json")
        with open(fit_path, "w") as f:
            json.dump(params, f, indent=2)
        logger.info("Fit saved to %s", fit_path)
        return

    # Fresh run
    if output_path.exists():
        output_path.unlink()
        logger.info("Cleared existing output file.")

    calibrator = Calibrator(
        base_url=args.base_url,
        api_key=args.api_key,
        base_model=args.base_model,
        output_path=output_path,
        repeats=args.repeats,
        temperature=args.temperature,
    )

    try:
        await calibrator.setup()
        results = await calibrator.run()
    finally:
        await calibrator.teardown()

    params = fit_params(results)
    note = (
        f"base_url={args.base_url}  model={args.base_model}  "
        f"repeats={args.repeats}  n_probes={len(results)}"
    )
    print_yaml_snippet(params, note)

    fit_path = output_path.with_suffix(".fit.json")
    with open(fit_path, "w") as f:
        json.dump(params, f, indent=2)
    logger.info("Raw probes: %s", output_path)
    logger.info("Fit params: %s", fit_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate MockBackendParams from a live TuFT service."
    )
    parser.add_argument("--base-url", default="http://localhost:10610")
    parser.add_argument("--api-key", default="tml-test-key")
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B")
    parser.add_argument(
        "--output",
        default="calibration_results.jsonl",
        help="JSONL output path (appended per probe; safe to interrupt)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Repeat measurements per cell",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip probes; load existing JSONL and re-fit only",
    )
    args = parser.parse_args()
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
