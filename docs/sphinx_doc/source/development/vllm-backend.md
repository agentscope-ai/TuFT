# Direct vLLM Integration

TuFT's sampling backend talks to vLLM directly. It previously went through
[Trinity-RFT](https://github.com/agentscope-ai/Trinity-RFT)'s
`vLLMRolloutModel` actor; that dependency was removed after the analysis
summarized in this document. This page records **why**, **what replaced it**,
and — most importantly — **what to watch when maintaining the vLLM
integration going forward**.

## Why the Trinity dependency was removed

The coupling was always narrow: three deferred imports in
`src/tuft/backends/sampling_backend.py` (`InferenceModelConfig`,
`vLLMRolloutModel`) plus one `pyproject.toml` line. TuFT used exactly five
methods of Trinity's actor — `prepare()`, `get_api_server_url()`,
`_generate_internal()`, `add_lora_adapter()`, `remove_lora_adapter()` — and
none of Trinity's RL machinery (explorer/trainer/buffer, weight sync,
multimodal, `sample()`/`chat()`/`logprobs()`).

In fact TuFT had already been forced to bypass Trinity's public API:
trinity-rft 0.5.1's `sample()` constructed tinker types with tinker-0.7-era
constructor keywords, crashing under tinker 0.18.2. The fix (PR #111) called
Trinity's *private* `_generate_internal()` and built `SampleResponse` from
the raw vLLM `RequestOutput` in TuFT — meaning the sampling data path was
already "direct vLLM"; only the actor shell was Trinity's. Trinity 0.6.0
changed the tinker constructor API a third time (`tokens_np=`, requiring
tinker >= 0.20), so upgrading Trinity would not have allowed deleting the
bypass.

The version-coupling costs were concrete:

- **Resolver deadlocks.** Before TuFT moved to transformers 5, trinity 0.6.0
  (`transformers>=5.12.1`) was uninstallable next to TuFT's
  `transformers<5` pin, silently holding TuFT on trinity 0.5.1 — which pins
  `verl==0.7.0` exactly and caps `vllm<=0.15.1`. Trinity effectively
  dictated TuFT's training *and* inference engine versions.
- **Dependency drag.** trinity-rft carries ~29 mandatory dependencies of
  which ~18 are unused by TuFT (wandb, sqlalchemy, psycopg2-binary,
  streamlit, flask, tensorboard, matplotlib, networkx, math_verify, ...):
  roughly 40–60 extra transitive wheels and 300–500 MB of install weight in
  every image build.
- **Runtime-only breakage class.** Because TuFT installed `trinity[vllm]`
  (not `trinity[tinker]`), Trinity↔tinker incompatibilities never surfaced
  at resolve time — only as `TypeError` in production sampling paths.

Training was never affected: `fsdp_training_backend.py` imports verl
directly and has no Trinity dependency.

## What replaced it

Three TuFT-owned modules under `src/tuft/backends/`, together ~600 lines
(≈350 of which are adapted from Trinity's own Apache-2.0 code, with
attribution headers):

| Module | Role | Origin |
|---|---|---|
| `vllm_engine.py` | `VLLMEngineConfig` + `VLLMEngine` Ray actor: env setup, `AsyncEngineArgs` → `AsyncLLMEngine.from_engine_args()`, `generate()` returning raw `RequestOutput`, `add_lora`/`remove_lora`, in-process API server startup, shutdown | New (mirrors trinity 0.6.0's `vLLMRolloutModel` minus weight sync, multimodal, multi-node) |
| `vllm_api_server.py` | Runs vLLM's own OpenAI-compatible server **in the actor process**, sharing the already-created engine | Adapted from trinity `api_patch_v17.py` |
| `vllm_worker.py` | `TuFTGPUWorker` (custom `worker_cls`) applying the prompt-logprobs temperature-scaling patch inside each worker process | Adapted from trinity `worker_patch.py` + `vllm_worker.py` |

Unchanged by design (they only consume the backend's public contract or
vLLM-native HTTP endpoints):

- `sampling_backend.py`'s `_build_sample_response` — the single adapter from
  vLLM `RequestOutput` to tinker 0.18.2 dataclasses (kept verbatim).
- The OpenAI proxy (`tuft.oai`) — it already loaded trained LoRA adapters via
  vLLM's native `/v1/load_lora_adapter` endpoint and proxied
  `/v1/completions`, `/v1/chat/completions`, `/v1/models`.
- Weight delivery — LoRA adapters saved to disk by the training backend and
  registered via `add_lora`; there is no NCCL/collective weight sync.
- Colocate mode — plain Ray fractional-GPU placement plus
  `gpu_memory_utilization`; no engine sleep/wake involved.

Deleted along with the dependency: the 100-line trinity-workaround comment
blocks, the dead `_normalize_sample_response()` helper, and Trinity's
in-worker imports of verl and its checkpoint weight-transfer engine.
The swap also fixed a latent bug: `remove_adapter` previously passed the
string adapter name to vLLM's `remove_lora`, which expects the integer
adapter id, so engine-side removal could never succeed.

## Behavioral notes (read before touching logprobs)

- **Sampled-token logprobs** are temperature-scaled natively by the
  `logprobs_mode="processed_logprobs"` engine argument. Do not remove it:
  RL importance ratios require logprobs of the *actual* sampling
  distribution.
- **Prompt logprobs** are *not* temperature-scaled by upstream vLLM (still
  unfixed in 0.24.0). `vllm_worker.py` patches
  `GPUModelRunner._get_prompt_logprobs_dict` to add the missing
  `logits /= temperature` step. Removing this patch silently changes prompt
  logprob values whenever temperature ≠ 1.0, which corrupts
  importance-sampling losses that consume them through the Tinker API.
- **`skip_reading_prefix_cache`** (set in `sample()` when prompt logprobs are
  requested) is a native vLLM `SamplingParams` field since 0.12, *not* a
  Trinity patch; newer vLLM sets it automatically for prompt-logprob
  requests. It prevents cached prompt chunks from skipping logit
  computation.
- **Engine seed** is fixed at 42 (matching the behavior TuFT always had via
  Trinity's default); per-request determinism comes from
  `SamplingParams.seed`. `ModelConfig.seed` is intentionally not wired to
  the engine to avoid a behavior change — wire it deliberately if desired.
- **Eager execution** defaults to enabled for the embedded engine. vLLM 0.24's
  default TorchInductor/CUDA-graph warmup did not finish within several
  minutes on an L4, both standalone and colocated. Set
  ``sampling_enforce_eager`` to ``false`` only after validating startup
  latency, prompt logprobs, LoRA, and colocation on the target hardware.

## Maintenance guide

The vLLM API surface TuFT uses splits into two stability classes:

**Stable (public engine-level surface):**
`AsyncEngineArgs`, `AsyncLLMEngine.from_engine_args`, `generate()`,
`add_lora`/`remove_lora`, `SamplingParams` fields (`n`, `seed`, `top_k`,
`top_p`, `temperature`, `logprobs`, `prompt_logprobs`, `stop`,
`stop_token_ids`).

**Churn-prone (the in-process OpenAI server recipe in
`vllm_api_server.py`):** vLLM's stock `run_server` cannot serve a
pre-existing engine, so TuFT reuses vLLM's building blocks
(`make_arg_parser`, `build_app`, `init_app_state`, `serve_http`). These
internals moved or changed signature at vLLM 0.12, 0.13, ~0.17, 0.22 and
0.23 — historically a 10–50 line adaptation every 2–4 minor releases.
Trinity absorbed exactly this churn with three version-dispatched patch
files; TuFT carries only the variant for its pinned range.

When **bumping the vLLM pin** in `pyproject.toml`, check:

1. `vllm_api_server.py` imports still resolve (`build_app`,
   `init_app_state`, `create_server_socket`, `serve_http`,
   `make_arg_parser`, `FlexibleArgumentParser`, utils modules) and
   `init_app_state(engine, app.state, args)` signature is unchanged.
2. `vllm_worker.py`: diff the patched `_get_prompt_logprobs_dict` against
   upstream `gpu_model_runner.py` for the new version; re-vendor with the
   `PATCH START/END` block if upstream changed, and check whether upstream
   has finally added temperature scaling (then delete the patch entirely).
3. Env vars in `VLLMEngine.__init__` (e.g. `VLLM_USE_V2_MODEL_RUNNER`,
   `VLLM_ENABLE_V1_MULTIPROCESSING`) — review against the new version's
   defaults.
4. Align vLLM's exact PyTorch ABI requirement and its FastAPI/Transformers
   bounds in `pyproject.toml`, then resolve the dependency graph with uv.
5. Run `scripts/verify_runtime_versions.py` in packaging/release-image paths
   and run the GPU verification below.
6. Medium-term watch item: vLLM's experimental Rust frontend (RFC #40846)
   runs the API server out-of-process and cannot share an in-process
   engine. The Python server remains the default with no deprecation
   signal, but if that changes, the fallback is running `vllm serve` as a
   subprocess and using HTTP for everything (including a
   `/v1/completions`-based sampling path or `engine.generate` kept local).

**GPU verification checklist** (after any vLLM bump or edit to these
modules):

```bash
# Sampling backend + OpenAI API integration tests on a GPU machine
TUFT_TEST_MODEL=Qwen/Qwen3-0.6B uv run pytest tests/test_sampling_backend.py -m gpu --gpu
TUFT_TEST_MODEL=Qwen/Qwen3-0.6B uv run pytest tests/test_openai_api.py --gpu
```

The current project pins vLLM 0.24.0 and its required PyTorch 2.11.0 ABI.
The release Dockerfile verifies those installed versions against
`pyproject.toml` before an image can be published.

Also confirm manually: (a) prompt logprobs at `temperature=0.7` with and
without prefix cache agree, (b) a trained LoRA is servable both through
tinker `sample()` and through the OpenAI proxy by name, and (c) colocate
mode still fits alongside the training actor at the configured
`sampling_memory_fraction`.
