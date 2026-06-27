"""Tenant actor: manages the sampling loop, buffer, and training triggers."""

from __future__ import annotations

import asyncio
import copy
import logging
import time
from dataclasses import dataclass
from typing import Any, List

from .backend.base import AdapterConfig, TrainingBackend
from .config import EvaluationConfig, TenantConfig
from .metrics import TenantMetrics
from .tasks.agent_task import AgentTask, StepResult
from .tasks.base import Task


logger = logging.getLogger(__name__)


@dataclass
class BufferItem:
    """A single item in the replay buffer."""

    prompt_tokens: List[int]
    response_tokens: List[int]
    logprobs: List[float]
    reward: float
    weight_version: int
    latency_ms: float


class Tenant:
    """A single tenant/adapter that continuously samples and triggers training."""

    def __init__(
        self,
        tenant_id: str,
        backend: TrainingBackend,
        task: Task,
        config: TenantConfig,
        eval_config: EvaluationConfig,
        collect_logprobs: bool = False,
        logprob_dump_writer: Any = None,
    ):
        self.id = tenant_id
        self.backend = backend
        self.task = task
        self.config = config
        self.eval_config = eval_config
        # Whether to compute & record sampling-vs-training logprob summaries.
        self._collect_logprobs = collect_logprobs
        # Optional callable: writer(record_dict) -> None. When provided, the
        # tenant will dump per-step per-item per-token logprob pairs to it.
        self._logprob_dump_writer = logprob_dump_writer

        # State
        self.current_weight_version: int = 0
        self.train_steps_completed: int = 0
        self.training_in_progress: bool = False
        self.buffer: List[BufferItem] = []

        # Metrics
        self.metrics = TenantMetrics(tenant_id=tenant_id)

        # Tokenizer (set during run)
        self._tokenizer: Any = None

    async def run(self) -> None:
        """Full lifecycle of this tenant."""
        logger.info(
            f"[{self.id}] Starting tenant (task={self.config.task}, "
            f"steps={self.config.num_train_steps})"
        )

        # Create adapter
        adapter_config = AdapterConfig(
            lora_rank=self.config.lora_rank,
            learning_rate=self.config.learning_rate,
        )
        await self.backend.create_adapter(self.id, adapter_config)

        # Get tokenizer
        self._tokenizer = self.backend.get_tokenizer()

        # Sync initial weights
        self.current_weight_version = await self.backend.sync_weights(self.id)

        # Load task dataset
        try:
            self.task.load_dataset()
        except Exception as e:
            logger.error(f"[{self.id}] Failed to load dataset for task '{self.config.task}': {e}")
            return

        # Start sampling loop
        await self._sampling_loop()

        # Final eval after all training steps complete (if not already evaluated at last step)
        if self.eval_config.eval_interval_steps > 0 and self.train_steps_completed > 0:
            last_eval_step = (
                self.train_steps_completed // self.eval_config.eval_interval_steps
            ) * self.eval_config.eval_interval_steps
            if last_eval_step < self.train_steps_completed:
                logger.info(f"[{self.id}] Running final eval at step {self.train_steps_completed}")
                await self._do_eval()

        logger.info(
            f"[{self.id}] Tenant finished. steps={self.train_steps_completed}, "
            f"samples={self.metrics.total_samples}"
        )

    async def _sampling_loop(self) -> None:
        """Continuously sample until num_train_steps is reached."""
        is_agent_task = isinstance(self.task, AgentTask)

        if self.config.async_sampling:
            await self._sampling_loop_async()
            return

        while self.train_steps_completed < self.config.num_train_steps:
            if is_agent_task:
                await self._run_agent_episode()
            else:
                await self._run_single_turn_sample()

            # Check buffer full -> trigger training
            if len(self.buffer) >= self.config.buffer_size and not self.training_in_progress:
                self.training_in_progress = True
                batch = self.buffer[: self.config.buffer_size]
                self.buffer = self.buffer[self.config.buffer_size :]
                if self.config.sync_mode:
                    # Synchronous: pause sampling, run training to completion,
                    # then resume. Guarantees staleness == 0 for every item
                    # in this batch (and all subsequent batches too, because
                    # no sampling happens while train is in flight).
                    await self._do_train(batch)
                else:
                    # Asynchronous (default): training runs concurrently with
                    # continued sampling, naturally producing staleness > 0
                    # for in-flight samples.
                    asyncio.create_task(self._do_train(batch))

        # Wait for any in-progress training to finish
        while self.training_in_progress:
            await asyncio.sleep(0.1)

    async def _sampling_loop_async(self) -> None:
        """Async-sampling loop: fire up to `async_sampling_concurrency` requests
        concurrently without waiting for the previous to complete.  Models a
        shared tenant served by multiple concurrent users.

        For single-turn tasks, each concurrent request is fully independent.
        For agent tasks, each concurrent episode uses a shallow-copied task
        instance so per-episode state (e.g. _current_paragraphs, _call_history)
        does not collide across episodes.  The underlying dataset lists are
        shared (not deep-copied) to avoid memory overhead.

        A Semaphore caps the number of in-flight requests/episodes so the
        server is not flooded.  Each completed sample/episode adds items to the
        buffer immediately, and training is triggered whenever the buffer fills.
        """
        is_agent_task = isinstance(self.task, AgentTask)
        sem = asyncio.Semaphore(self.config.async_sampling_concurrency)
        pending_tasks: set[asyncio.Task] = set()

        def _get_task_instance():
            """Return a task instance safe for one concurrent episode.

            Agent tasks carry per-episode mutable state on the instance
            (_current_paragraphs, _call_history, etc.).  A shallow copy
            gives each concurrent episode its own copy of those attributes
            while sharing the read-only dataset lists.
            """
            if is_agent_task:
                return copy.copy(self.task)
            return self.task

        async def _one_sample() -> None:
            """Single-turn sample (non-agent)."""
            await asyncio.sleep(1.0 / self.config.request_rate)
            prompt = self.task.get_prompt()
            prompt_tokens = self._tokenizer.encode(prompt.text, add_special_tokens=False)
            t0 = time.time()
            try:
                results = await self.backend.sample(
                    adapter_id=self.id,
                    prompt_tokens=prompt_tokens,
                    num_samples=1,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                )
            except Exception as e:
                logger.warning(f"[{self.id}] Async sample failed: {e}")
                return
            latency = time.time() - t0
            result = results[0]
            try:
                reward = self.task.compute_reward(result.text, prompt.metadata)
            except Exception as e:
                logger.warning(f"[{self.id}] compute_reward failed: {e}")
                reward = 0.0
            self._add_to_buffer(
                BufferItem(
                    prompt_tokens=prompt_tokens,
                    response_tokens=result.tokens,
                    logprobs=result.logprobs,
                    reward=reward,
                    weight_version=self.current_weight_version,
                    latency_ms=latency * 1000,
                )
            )
            self.metrics.record_sample(reward, latency)

        async def _one_agent_episode() -> None:
            """One full multi-turn agent episode using an isolated task copy."""
            await asyncio.sleep(1.0 / self.config.request_rate)
            task_copy = _get_task_instance()
            max_turns = self.config.max_turns if self.config.max_turns > 0 else 8

            initial_prompt = task_copy.reset_episode()
            conversation_text = initial_prompt.text
            metadata = initial_prompt.metadata

            episode_turns: List[dict] = []
            step_results: List[StepResult] = []
            episode_t0 = time.time()

            for turn_idx in range(max_turns):
                prompt_tokens = self._tokenizer.encode(conversation_text, add_special_tokens=False)
                prompt_tokens = self._truncate_prompt_tokens(prompt_tokens)
                t0 = time.time()
                try:
                    results = await self.backend.sample(
                        adapter_id=self.id,
                        prompt_tokens=prompt_tokens,
                        num_samples=1,
                        max_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                    )
                except Exception as e:
                    logger.warning(f"[{self.id}] Async agent sample failed at turn {turn_idx}: {e}")
                    break
                latency = time.time() - t0
                result = results[0]
                episode_turns.append(
                    {
                        "prompt_tokens": prompt_tokens,
                        "response_tokens": result.tokens,
                        "logprobs": result.logprobs,
                        "latency": latency,
                    }
                )
                try:
                    step_result = task_copy.step(result.text, metadata)
                except Exception as e:
                    logger.warning(f"[{self.id}] Environment step failed: {e}")
                    step_result = StepResult(observation=f"Error: {e}", reward=0.0, done=True)
                step_results.append(step_result)
                conversation_text += result.text + task_copy.format_observation(step_result)
                if step_result.done:
                    break

            try:
                episode_reward = task_copy.compute_episode_reward(step_results, metadata)
            except Exception as e:
                logger.warning(f"[{self.id}] compute_episode_reward failed: {e}")
                episode_reward = 0.0

            episode_latency = time.time() - episode_t0
            for turn_data in episode_turns:
                self._add_to_buffer(
                    BufferItem(
                        prompt_tokens=turn_data["prompt_tokens"],
                        response_tokens=turn_data["response_tokens"],
                        logprobs=turn_data["logprobs"],
                        reward=episode_reward,
                        weight_version=self.current_weight_version,
                        latency_ms=turn_data["latency"] * 1000,
                    )
                )
            self.metrics.record_sample(episode_reward, episode_latency)

        work_fn = _one_agent_episode if is_agent_task else _one_sample

        while self.train_steps_completed < self.config.num_train_steps:
            # Stop dispatching once we have already collected enough samples
            # (committed to training + sitting in buffer).  This avoids firing
            # inference requests whose results will only be discarded.
            committed = self.train_steps_completed * self.config.buffer_size
            if committed + len(self.buffer) >= self._target_samples:
                break
            # Acquire semaphore BEFORE creating the task so we only have
            # at most `async_sampling_concurrency` requests in-flight.
            await sem.acquire()
            if self.train_steps_completed >= self.config.num_train_steps:
                sem.release()
                break

            async def _run_with_release() -> None:
                try:
                    await work_fn()
                finally:
                    sem.release()

            task = asyncio.create_task(_run_with_release())
            pending_tasks.add(task)
            task.add_done_callback(pending_tasks.discard)
            # Yield once so the event loop can schedule the new task
            await asyncio.sleep(0)

        # Drain remaining in-flight requests (already dispatched before the
        # quota check fired; their results are valid and should be used).
        if pending_tasks:
            await asyncio.gather(*pending_tasks, return_exceptions=True)

        # Drain any remaining buffer into training.  In shared/async mode a
        # single tenant can overshoot the per-tenant sample quota because
        # episodes produce multiple turns; those turns are valid training
        # data and should not be discarded.
        while self.buffer and self.train_steps_completed < self.config.num_train_steps:
            if not self.training_in_progress:
                self.training_in_progress = True
                batch = self.buffer[: self.config.buffer_size]
                self.buffer = self.buffer[self.config.buffer_size :]
                asyncio.create_task(self._do_train(batch))
            else:
                await asyncio.sleep(0.05)

        # Wait for any in-progress training to finish
        while self.training_in_progress:
            await asyncio.sleep(0.1)

    async def _run_single_turn_sample(self) -> None:
        """Execute a single-turn sample (original logic)."""
        # Rate limiting
        await asyncio.sleep(1.0 / self.config.request_rate)

        # Get prompt
        prompt = self.task.get_prompt()
        prompt_tokens = self._tokenizer.encode(prompt.text, add_special_tokens=False)

        # Sample
        t0 = time.time()
        try:
            results = await self.backend.sample(
                adapter_id=self.id,
                prompt_tokens=prompt_tokens,
                num_samples=1,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
        except Exception as e:
            logger.warning(f"[{self.id}] Sample failed: {e}")
            return
        latency = time.time() - t0

        result = results[0]

        # Compute reward
        try:
            reward = self.task.compute_reward(result.text, prompt.metadata)
        except Exception as e:
            logger.warning(f"[{self.id}] compute_reward failed: {e}")
            reward = 0.0

        # Add to buffer
        self.buffer.append(
            BufferItem(
                prompt_tokens=prompt_tokens,
                response_tokens=result.tokens,
                logprobs=result.logprobs,
                reward=reward,
                weight_version=self.current_weight_version,
                latency_ms=latency * 1000,
            )
        )

        # Record metrics
        self.metrics.record_sample(reward, latency)

    # Maximum prompt length to prevent OOM on serving side.
    # Reserve space for max_tokens generation within the model's context window.
    _MAX_CONTEXT_TOKENS = 7168

    def _truncate_prompt_tokens(self, prompt_tokens: List[int]) -> List[int]:
        """Truncate prompt tokens to fit within context budget.

        Keeps the beginning (system prompt) and end (recent turns) if too long.
        """
        max_prompt_len = self._MAX_CONTEXT_TOKENS - self.config.max_tokens
        if len(prompt_tokens) <= max_prompt_len:
            return prompt_tokens

        # Keep first 30% (system prompt + initial question) and last 70% (recent context)
        keep_start = int(max_prompt_len * 0.3)
        keep_end = max_prompt_len - keep_start
        truncated = prompt_tokens[:keep_start] + prompt_tokens[-keep_end:]
        logger.debug(
            f"[{self.id}] Truncated prompt from {len(prompt_tokens)} to {len(truncated)} tokens"
        )
        return truncated

    async def _run_agent_episode(self) -> None:
        """Execute a multi-turn agent episode.

        The agent interacts with the environment over multiple turns:
        1. Get initial prompt from task
        2. Agent generates response (parsed for action)
        3. Environment executes action and returns observation
        4. Observation is appended to conversation context
        5. Repeat until done or max_turns reached
        6. Episode reward is assigned to all turns for training
        """
        assert isinstance(self.task, AgentTask)
        max_turns = self.config.max_turns if self.config.max_turns > 0 else 8

        # Reset episode
        initial_prompt = self.task.reset_episode()
        conversation_text = initial_prompt.text
        metadata = initial_prompt.metadata

        # Track all turns for this episode
        episode_turns: List[dict] = []  # [{prompt_tokens, response_tokens, logprobs, latency}]
        step_results: List[StepResult] = []
        episode_t0 = time.time()

        for turn_idx in range(max_turns):
            # Rate limiting per turn
            await asyncio.sleep(1.0 / self.config.request_rate)

            # Encode current conversation as prompt
            prompt_tokens = self._tokenizer.encode(conversation_text, add_special_tokens=False)
            # Truncate to avoid exceeding server's max context length
            prompt_tokens = self._truncate_prompt_tokens(prompt_tokens)

            # Sample agent response
            t0 = time.time()
            try:
                results = await self.backend.sample(
                    adapter_id=self.id,
                    prompt_tokens=prompt_tokens,
                    num_samples=1,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                )
            except Exception as e:
                logger.warning(f"[{self.id}] Agent sample failed at turn {turn_idx}: {e}")
                break
            latency = time.time() - t0

            result = results[0]
            agent_response = result.text

            # Record this turn's sampling data
            episode_turns.append(
                {
                    "prompt_tokens": prompt_tokens,
                    "response_tokens": result.tokens,
                    "logprobs": result.logprobs,
                    "latency": latency,
                }
            )

            # Execute action in environment
            try:
                step_result = self.task.step(agent_response, metadata)
            except Exception as e:
                logger.warning(f"[{self.id}] Environment step failed: {e}")
                step_result = StepResult(observation=f"Error: {e}", reward=0.0, done=True)

            step_results.append(step_result)

            # Append agent response and observation to conversation
            observation_text = self.task.format_observation(step_result)
            conversation_text += agent_response + observation_text

            if step_result.done:
                break

        # Compute episode-level reward
        try:
            episode_reward = self.task.compute_episode_reward(step_results, metadata)
        except Exception as e:
            logger.warning(f"[{self.id}] compute_episode_reward failed: {e}")
            episode_reward = 0.0

        episode_latency = time.time() - episode_t0

        # Add all turns to buffer with episode-level reward
        # Each turn gets the same episode reward (credit assignment handled by advantage normalization)
        for turn_data in episode_turns:
            self.buffer.append(
                BufferItem(
                    prompt_tokens=turn_data["prompt_tokens"],
                    response_tokens=turn_data["response_tokens"],
                    logprobs=turn_data["logprobs"],
                    reward=episode_reward,
                    weight_version=self.current_weight_version,
                    latency_ms=turn_data["latency"] * 1000,
                )
            )

        # Record metrics (one sample record per episode with total latency)
        self.metrics.record_sample(episode_reward, episode_latency)

        num_turns = len(episode_turns)
        finished = any(s.done for s in step_results)
        logger.debug(
            f"[{self.id}] Episode done: turns={num_turns}, "
            f"reward={episode_reward:.3f}, finished={finished}"
        )

    @property
    def _target_samples(self) -> int:
        """Total samples needed to complete all training steps."""
        return self.config.num_train_steps * self.config.buffer_size

    def _add_to_buffer(self, item: BufferItem) -> None:
        """Asyncio-safe buffer append with training trigger."""
        self.buffer.append(item)
        if len(self.buffer) >= self.config.buffer_size and not self.training_in_progress:
            self.training_in_progress = True
            batch = self.buffer[: self.config.buffer_size]
            self.buffer = self.buffer[self.config.buffer_size :]
            asyncio.create_task(self._do_train(batch))

    async def _do_train(self, batch: List[BufferItem]) -> None:
        """Execute one training step from buffer contents."""
        try:
            # Compute advantages (normalize rewards within batch)
            rewards = [item.reward for item in batch]
            mean_r = sum(rewards) / len(rewards)
            var_r = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
            std_r = var_r**0.5

            if std_r < 1e-8:
                advantages = [0.0] * len(batch)
            else:
                advantages = [(r - mean_r) / (std_r + 1e-6) for r in rewards]

            # Compute staleness for this batch
            stalenesses = [
                float(self.train_steps_completed - item.weight_version) for item in batch
            ]
            self.metrics.record_staleness(self.train_steps_completed + 1, stalenesses)

            # ---- Logprob mismatch collection -----------------------------
            # Compute training-side logprobs on the CURRENT trainable weights
            # (the same weights that the upcoming forward_backward will use).
            # This is paired with the sampling_logprobs already stored in each
            # BufferItem (taken at sample-time on weights `item.weight_version`).
            training_lps: List[List[float]] = []
            if self._collect_logprobs:
                try:
                    training_lps = await self.backend.compute_training_logprobs(
                        self.id,
                        [
                            {
                                "prompt_tokens": item.prompt_tokens,
                                "response_tokens": item.response_tokens,
                            }
                            for item in batch
                        ],
                        temperature=self.config.temperature,
                    )
                except Exception as e:
                    logger.warning(f"[{self.id}] compute_training_logprobs failed: {e}")
                    training_lps = []

            # Build training datums
            datums = []
            for item, adv in zip(batch, advantages):
                datums.append(
                    {
                        "prompt_tokens": item.prompt_tokens,
                        "response_tokens": item.response_tokens,
                        "sampling_logprobs": item.logprobs,
                        "advantage": adv,
                    }
                )

            # Execute training step (timed separately)
            t_train = time.time()
            await self.backend.train_step(self.id, datums)
            train_duration = time.time() - t_train
            self.train_steps_completed += 1

            # Record logprob mismatch summary (after step counter increment so
            # `step` matches downstream metrics).
            if training_lps and len(training_lps) == len(batch):
                sampling_lps = [item.logprobs for item in batch]
                summary = self.metrics.record_logprob_mismatch(
                    step=self.train_steps_completed,
                    weight_version=self.current_weight_version,
                    sampling_logprobs=sampling_lps,
                    training_logprobs=training_lps,
                    stalenesses=stalenesses,
                )
                logger.info(
                    f"[{self.id}] step={self.train_steps_completed} "
                    f"logprob_mismatch: mean_abs={summary['mean_abs_diff']:.4f} "
                    f"mean_diff={summary['mean_diff']:.4f} "
                    f"is_w={summary['mean_is_weight']:.3f} "
                    f"clip01={summary['p_out_clip_01']:.2f} "
                    f"staleness={summary['mean_staleness']:.2f}"
                )
                # Optional per-token dump
                if self._logprob_dump_writer is not None:
                    for idx, (item, samp, trn) in enumerate(zip(batch, sampling_lps, training_lps)):
                        record = {
                            # ── tenant / task static features ──
                            "tenant_id": self.id,
                            "task": self.config.task,
                            "lora_rank": self.config.lora_rank,
                            "learning_rate": self.config.learning_rate,
                            "temperature": self.config.temperature,
                            "max_tokens": self.config.max_tokens,
                            "sync_mode": self.config.sync_mode,
                            # ── timing / versioning ──
                            "step": self.train_steps_completed,
                            "item_idx": idx,
                            "sample_weight_version": item.weight_version,
                            "train_weight_version": self.current_weight_version,
                            "staleness": float(
                                self.train_steps_completed - 1 - item.weight_version
                            ),
                            # ── RL signals (informational, not used by predictor) ──
                            "reward": item.reward,
                            "advantage": advantages[idx],
                            # ── token-level features (required by predictor) ──
                            "n_prompt_tokens": len(item.prompt_tokens),
                            "n_response_tokens": len(item.response_tokens),
                            "prompt_tokens": list(item.prompt_tokens),
                            "response_tokens": list(item.response_tokens),
                            # ── logprobs ──
                            "sampling_logprobs": list(samp),
                            "training_logprobs": list(trn),
                        }
                        try:
                            self._logprob_dump_writer(record)
                        except Exception as e:
                            logger.warning(f"[{self.id}] logprob dump failed: {e}")

            # Sync new weights (timed separately)
            t_sync = time.time()
            self.current_weight_version = await self.backend.sync_weights(self.id)
            sync_duration = time.time() - t_sync

            # Record training metrics
            self.metrics.record_train_step(
                self.train_steps_completed,
                mean_r,
                train_duration_s=train_duration,
                sync_duration_s=sync_duration,
            )

            logger.info(
                f"[{self.id}] Train step {self.train_steps_completed}/"
                f"{self.config.num_train_steps}, "
                f"mean_reward={mean_r:.4f}, mean_staleness={sum(stalenesses) / len(stalenesses):.2f}"
            )

            # Evaluation check
            if (
                self.eval_config.eval_interval_steps > 0
                and self.train_steps_completed % self.eval_config.eval_interval_steps == 0
            ):
                await self._do_eval()

        except Exception as e:
            logger.error(f"[{self.id}] Training step failed: {e}", exc_info=True)
        finally:
            self.training_in_progress = False

    async def _do_eval(self) -> None:
        """Run evaluation on current model weights."""
        is_agent_task = isinstance(self.task, AgentTask)
        eval_prompts = self.task.get_eval_prompts(self.eval_config.eval_sample_size)
        correct = 0

        for prompt in eval_prompts:
            if is_agent_task:
                reward = await self._eval_agent_episode(prompt)
            else:
                tokens = self._tokenizer.encode(prompt.text, add_special_tokens=False)
                try:
                    results = await self.backend.sample(
                        adapter_id=self.id,
                        prompt_tokens=tokens,
                        num_samples=1,
                        max_tokens=self.config.max_tokens,
                        temperature=self.eval_config.eval_temperature,
                    )
                    reward = self.task.compute_reward(results[0].text, prompt.metadata)
                except Exception as e:
                    logger.warning(f"[{self.id}] Eval sample failed: {e}")
                    continue

            if reward >= 1.0:
                correct += 1

        accuracy = correct / len(eval_prompts) if eval_prompts else 0.0
        self.metrics.record_eval(self.train_steps_completed, accuracy)
        logger.info(
            f"[{self.id}] Eval at step {self.train_steps_completed}: "
            f"accuracy={accuracy:.4f} ({correct}/{len(eval_prompts)})"
        )

    async def _eval_agent_episode(self, initial_prompt: "TaskPrompt") -> float:
        """Run a single evaluation episode for an agent task."""
        assert isinstance(self.task, AgentTask)
        max_turns = self.config.max_turns if self.config.max_turns > 0 else 8

        conversation_text = initial_prompt.text
        metadata = initial_prompt.metadata
        # Restore environment state for this evaluation episode
        self.task._current_paragraphs = metadata.get("paragraphs", {})
        self.task._current_page = None

        step_results: List[StepResult] = []

        for turn_idx in range(max_turns):
            tokens = self._tokenizer.encode(conversation_text, add_special_tokens=False)
            tokens = self._truncate_prompt_tokens(tokens)
            try:
                results = await self.backend.sample(
                    adapter_id=self.id,
                    prompt_tokens=tokens,
                    num_samples=1,
                    max_tokens=self.config.max_tokens,
                    temperature=self.eval_config.eval_temperature,
                )
            except Exception as e:
                logger.warning(f"[{self.id}] Eval agent sample failed: {e}")
                break

            agent_response = results[0].text

            try:
                step_result = self.task.step(agent_response, metadata)
            except Exception as e:
                step_result = StepResult(observation=f"Error: {e}", reward=0.0, done=True)

            step_results.append(step_result)
            observation_text = self.task.format_observation(step_result)
            conversation_text += agent_response + observation_text

            if step_result.done:
                break

        try:
            return self.task.compute_episode_reward(step_results, metadata)
        except Exception:
            return 0.0
