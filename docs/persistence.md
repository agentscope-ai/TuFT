# Persistence

TuFT supports **optional persistence** for server state. When enabled, TuFT can recover key runtime metadata (sessions, training runs, sampling sessions, futures) after a server restart, and then **reconstruct model runtime state from checkpoints on disk**.

This document explains both **how to use persistence** and **how it works internally** (what is stored, key layout, restore semantics, and safety checks).

---

## Table of Contents

- [Goals and non-goals](#goals-and-non-goals)
- [Persistence backends](#persistence-backends)
- [What is persisted](#what-is-persisted)
- [Redis key design](#redis-key-design)
- [Startup restore semantics](#startup-restore-semantics)
- [Safety checks](#safety-checks)
- [Configuration](#configuration)
- [Operational workflows](#operational-workflows)
- [Troubleshooting](#troubleshooting)

---

## Goals and non-goals

### Goals

- **Crash/restart recovery** of server state metadata so users can:
  - list sessions / training runs / sampling sessions after restart
  - retrieve completed futures after restart (within TTL)
  - continue training **from the latest checkpoint** for each training run
- **Safety-first restore**: prevent silent corruption when server configuration changes.

### Non-goals

- Persistence **does not** snapshot live GPU memory / in-flight model execution state.
- TuFT **does not** re-run pending tasks after a crash. Pending work is treated as unsafe and must be retried.
- Persistence **does not** store model weight blobs in Redis; weight artifacts live on disk under `checkpoint_dir` (and can be archived via the API).

---

## Persistence backends

TuFT exposes three modes via `persistence.mode`:

- `DISABLE`: in-memory only; everything is lost on restart.
- `REDIS`: external Redis via `redis-py` (recommended for production).
- `FILE`: a file-backed Redis-like store (intended for demos/tests; uses a JSON file and is not optimized for concurrency/performance).

Internally, all records are stored as **JSON-serialized Pydantic models**, one record per key.

---

## What is persisted

Persistence is implemented at the level of **major server subsystems** (not every tiny field/variable), specifically:

- **Sessions** (`SessionManager`)
  - session metadata, tags, `user_id`, heartbeat timestamp
  - stored as permanent records (no TTL)

- **Training runs** (`TrainingController`)
  - training run metadata (`training_run_id`, `base_model`, `lora_rank`, `model_owner`, `next_seq_id`, etc.)
  - **checkpoint records (metadata only)** (training checkpoints and sampler checkpoints) are stored under separate keys  
    (the actual checkpoint weight artifacts live on disk under `checkpoint_dir`, not in Redis)
  - stored as permanent records (no TTL)

- **Sampling sessions** (`SamplingController`)
  - sampling session metadata + **sampling history** (seq ids + prompt hashes)
  - stored as permanent records (no TTL)

- **Futures** (`FutureStore`)
  - request lifecycle records: `pending` / `ready` / `failed`
  - includes `operation_type`, `operation_args`, `future_id`, payload or error
  - stored with a **TTL** (default: 1 day) to prevent unbounded growth

- **Configuration signature** (`ConfigSignature`)
  - a snapshot of selected `AppConfig` fields for restore safety

---

## Redis key design

All keys are prefixed by a configurable `namespace` (default: `persistence-tuft-server`) and use `::` as the separator:

- Top-level records:  
  `"{namespace}::{type}::{id}"`
- Nested records:  
  `"{namespace}::{type}::{parent_id}::{nested_type}::{nested_id}"`

> Note: To avoid ambiguity, any literal `::` inside parts is escaped internally.

### Key families (high-level)

With the default namespace, TuFT uses these major key families:

- **Sessions**  
  `persistence-tuft-server::session::{session_id}`

- **Training runs**  
  `persistence-tuft-server::training_run::{training_run_id}`

- **Training checkpoints metadata** (nested under training runs)  
  `persistence-tuft-server::training_run::{training_run_id}::ckpt::{checkpoint_id}`

- **Sampler checkpoints metadata** (nested under training runs)  
  `persistence-tuft-server::training_run::{training_run_id}::sampler_ckpt::{checkpoint_id}`

- **Sampling sessions**  
  `persistence-tuft-server::sampling_session::{sampling_session_id}`

- **Futures** (TTL-based)  
  `persistence-tuft-server::future::{request_id}`

- **Config signature**  
  `persistence-tuft-server::config_signature`

---

## Startup restore semantics

Restore has **one preflight step** plus **two restore phases**:

### Phase 0: configuration validation (before server starts)

When persistence is enabled, TuFT validates the current `AppConfig` against the stored configuration signature **before** launching the server. This is designed to prevent a restart from silently interpreting old state with a new incompatible config.

If a mismatch is detected, TuFT aborts startup with a fatal error and shows a diff.

### Phase 1: in-memory restore from persistence backend (controller construction)

On process start, these components restore their in-memory registries by scanning their key prefixes and deserializing records:

- `SessionManager` restores sessions.
- `TrainingController` restores training runs + checkpoint records.
  - If a training run references a `base_model` not present in the current config, it is marked **corrupted**.
- `SamplingController` restores sampling sessions.
  - Sampling sessions whose `base_model` is no longer supported are deleted from storage.
- `FutureStore` restores futures.
  - Completed futures (`ready` / `failed`) are immediately marked as completed.
  - Restored futures also rebuild `future_id` allocation state to keep ordering monotonic.

At the end of Phase 1, TuFT has restored *metadata*, but model runtime state (adapters/weights in GPU memory) is not yet reconstructed.

### Phase 2: checkpoint-based recovery (async init)

After controller restore, TuFT performs checkpoint-based recovery:

- For each training run that is not corrupted and has a usable backend:
  - load the **latest checkpoint** on disk (and recreate adapter state)
  - treat that checkpoint as the server’s recovery boundary
- Futures are reconciled against this boundary:
  - if a training run has a valid latest checkpoint with `future_id = F`, **all futures for that run with `future_id > F` are marked failed** (and must be retried)
  - if no checkpoint exists, **all futures for that run are marked failed**

This means TuFT guarantees:

- **Training can continue from the latest checkpoint**, but
- any operations after that checkpoint are considered unsafe and require retries.

> Important: sequence IDs remain **monotonically increasing** even across restarts. TuFT does not “rewind” `next_seq_id` to the checkpoint boundary, by design.

---

## Safety checks

TuFT includes a restart-safety check to prevent silent corruption when configuration changes across restarts.

### Config signature validation (restart safety)

TuFT stores a `ConfigSignature` derived from `AppConfig.get_config_for_persistence()` (notably excluding the persistence config itself).

On startup, TuFT compares selected fields (default: `SUPPORTED_MODELS`) and can be configured to check additional fields:

- `SUPPORTED_MODELS` (always checked; mandatory)
- `CHECKPOINT_DIR`
- `MODEL_OWNER`
- `TOY_BACKEND_SEED`
- `AUTHORIZED_USERS`
- `TELEMETRY`

If validation fails, startup aborts and prints a diff. This avoids cases where an old state references models no longer configured, or a new deployment accidentally points at an old namespace.

---

## Configuration

### Install optional dependency

```bash
uv pip install "tuft[persistence]"
```

### Enable persistence

Add a `persistence` section to your `tuft_config.yaml`:

```yaml
persistence:
  mode: REDIS
  redis_url: "redis://localhost:6379/0"
  namespace: "persistence-tuft-server"
  check_fields:
    - SUPPORTED_MODELS
```

#### File-backed store (demos/tests)

```yaml
persistence:
  mode: FILE
  file_path: "~/.cache/tuft/file_redis.json"
  namespace: "persistence-tuft-server"
```

---

## Operational workflows

### Clearing persistence state

If you intentionally changed config and want to start fresh, clear persistence data:

```bash
tuft clear persistence --config /path/to/tuft_config.yaml
```

This removes keys under the configured `namespace`. It does **not** delete checkpoint files on disk.

### Changing config safely

Recommended workflow when changing any field that affects restore safety:

- deploy with a **new namespace**, or
- clear the old namespace explicitly before restart.

---

## Troubleshooting

### Startup fails with “Configuration Mismatch”

- **Cause**: you restarted TuFT with a config whose signature differs from the stored signature in the same namespace.
- **Fix**:
  - either revert the config change,
  - or clear persistence state (destructive) and restart,
  - or switch to a new `persistence.namespace` for the new deployment.

### After restart, some results are marked failed

This is expected if those futures were created **after** the latest recovered checkpoint (or when no checkpoint existed). Re-run those operations from the client.

### Redis grows indefinitely

Long-lived records (sessions, training runs, sampling sessions, checkpoints metadata) do not expire. Futures do expire by TTL, but you should also set a namespace per deployment and clear unused namespaces.

