# Observability (OpenTelemetry)

TuFT includes optional **OpenTelemetry** integration for:

- **Traces** (distributed tracing across HTTP → controllers → Ray actors)
- **Metrics** (request/workflow + resource metrics)
- **Logs** (Python logging bridged into the OpenTelemetry log pipeline)

This document explains both **how to enable telemetry** and **how TuFT wires it internally** (what gets recorded, correlation keys, and how to extend instrumentation).

---

## Table of Contents

- [Quick start](#quick-start)
- [Exporters and configuration](#exporters-and-configuration)
- [What TuFT records](#what-tuft-records)
  - [Traces](#traces)
  - [Metrics](#metrics)
  - [Logs](#logs)
- [Correlation / share keys](#correlation--share-keys)
- [Context propagation (Ray actors)](#context-propagation-ray-actors)
- [Developer guide](#developer-guide)
- [Recommended collector configuration](#recommended-collector-configuration)
- [Troubleshooting](#troubleshooting)

---

## Quick start

Enable telemetry in `tuft_config.yaml`:

```yaml
telemetry:
  enabled: true
  service_name: tuft
  otlp_endpoint: http://localhost:4317
  resource_attributes:
    deployment.environment: production
    service.version: 1.0.0
```

> **Note**: `resource_attributes` allows you to attach custom metadata to all telemetry data. Common attributes include:
> - `deployment.environment`: e.g., `production`, `staging`, `development`
> - `service.version`: your application version
> - `service.namespace`: logical grouping of services
> - `host.name`: hostname of the machine
> - Custom attributes for your organization (e.g., `team`, `region`)

Alternatively set the endpoint via env var:

```bash
export TUFT_OTLP_ENDPOINT=http://localhost:4317
```

Debug mode (prints to console exporters):

```bash
export TUFT_OTEL_DEBUG=1
```

---

## Exporters and configuration

TuFT initializes OpenTelemetry providers at CLI startup and uses one of two exporter modes:

- **Console exporters** if `TUFT_OTEL_DEBUG=1`
- **OTLP gRPC exporters** otherwise

The endpoint resolution order is:

1. `telemetry.otlp_endpoint` from config
2. `TUFT_OTLP_ENDPOINT` environment variable
3. OpenTelemetry exporter default endpoint (if neither is provided)

TuFT exports:

- traces via `OTLPSpanExporter` (gRPC)
- metrics via `OTLPMetricExporter` (gRPC)
- logs via `OTLPLogExporter` (gRPC)

Resource metadata:

- `service.name` is set to `telemetry.service_name`
- additional resource attributes can be attached via `telemetry.resource_attributes`

---

## What TuFT records

### Traces

TuFT produces spans at several layers:

- **HTTP server spans**: FastAPI is instrumented when telemetry is enabled.
- **Controller spans**: training/sampling operations create spans such as:
  - `training_controller.create_model`
  - `training_controller.run_forward`
  - `training_controller.run_forward_backward`
  - `training_controller.run_optim_step`
  - `training_controller.save_checkpoint`
  - `sampling_controller.create_sampling_session`
  - `sampling_controller.run_sample`
  - `future_store.execute_operation`
- **Persistence spans**: Redis operations create spans such as:
  - `redis.SET`, `redis.GET`, `redis.DEL`, `redis.SCAN`
- **Backend / actor spans** (when running with Ray):
  - spans inside model actors (e.g. HuggingFace backend) are linked via propagated context
  - examples include `hf_model.forward_backward`, `hf_model.optim_step`, `hf_model.save_state`, etc.

TuFT attaches key identifiers as span attributes (see [Correlation / share keys](#correlation--share-keys)).

### Metrics

TuFT emits two broad categories of metrics:

#### 1) Workflow metrics (counters/histograms)

Examples (names are stable and can be used as dashboards/alerts):

- Training:
  - `tuft.training.models.active` (up/down counter)
  - `tuft.training.tokens_per_second` (histogram)
  - `tuft.training.checkpoints.saved` (counter)
  - `tuft.training.checkpoint.size_bytes` (histogram)
- Sampling:
  - `tuft.sampling.sessions.active` (up/down counter)
  - `tuft.sampling.requests` (counter)
  - `tuft.sampling.duration` (histogram, seconds)
  - `tuft.sampling.tokens_per_second` (histogram)
  - `tuft.sampling.output_tokens` (histogram)
- Futures:
  - `tuft.futures.queue_length` (up/down counter)
  - `tuft.futures.created` (counter)
  - `tuft.futures.completed` (counter)
  - `tuft.futures.wait_time` (histogram)
  - `tuft.futures.execution_time` (histogram)
- Persistence:
  - `tuft.redis.operation.duration` (histogram)

Many metrics include attributes such as `base_model`, `operation_type`, `model_id`, or `queue_state` to support filtering and grouping.

#### 2) Resource metrics (observable gauges)

When telemetry is enabled, TuFT starts a background resource collector that exports observable gauges such as:

- CPU:
  - `tuft.resource.cpu.utilization_percent`
- Memory:
  - `tuft.resource.memory.used_bytes`
  - `tuft.resource.memory.total_bytes`
  - `tuft.resource.memory.utilization_percent`
- GPU (if NVML is available):
  - `tuft.resource.gpu.utilization_percent` (with attribute `gpu_id`)
  - `tuft.resource.gpu.memory_used_bytes` (with attribute `gpu_id`)
  - `tuft.resource.gpu.memory_total_bytes` (with attribute `gpu_id`)
- Process:
  - `tuft.resource.process.memory_used_bytes`

### Logs

TuFT bridges Python logging into OpenTelemetry:

- log records are exported via the configured OTel log provider
- existing logging handlers are not removed (OTel handler is added)

---

## Correlation / share keys

TuFT is multi-tenant and multi-client. To make traces searchable and linkable across components, TuFT sets a small set of **high-signal attributes** on spans.

You can treat these as "share keys" in your observability backend:

- `tuft.session_id`
  - links a user session across training + sampling activity
- `tuft.training_run_id`
  - links all spans for a specific training run (LoRA adapter lifecycle)
- `tuft.sampling_session_id`
  - links all spans for a specific sampling session
- `tuft.request_id`
  - links FutureStore execution spans with the async future lifecycle
- `tuft.operation_type`
  - categorizes futures (e.g. `forward_backward`, `optim_step`, `sample`)
- (additional common attributes)
  - `tuft.base_model`
  - `tuft.lora_rank`
  - `tuft.backward`

### How to use these keys in practice

- Given a **session_id** from the client, filter traces by `tuft.session_id = <id>`.
- Given a **training_run_id**, filter by `tuft.training_run_id = <id>`.
- Given a **sampling_session_id**, filter by `tuft.sampling_session_id = <id>`.
- Given an async **request_id** (future), filter by `tuft.request_id = <id>`.

This is the recommended way to trace activity across:

- session → training client → training controller → backend execution
- session → sampling client → sampling controller → backend execution
- API call → future enqueue → future execution → result retrieval

---

## Context propagation (Ray actors)

When TuFT executes work inside Ray actors, it propagates the OpenTelemetry context explicitly:

- At the controller/backend boundary, TuFT **injects** the current trace context into a serializable dict.
- The Ray actor side **extracts** that context and starts child spans under that context.

This provides end-to-end traces that cross process boundaries even when Ray scheduling changes the execution location.

---

## Developer guide

### Creating spans

Use `get_tracer()` and standard OpenTelemetry span APIs:

```python
from tuft.telemetry.tracing import get_tracer

tracer = get_tracer("tuft.my_module")
with tracer.start_as_current_span("my_operation") as span:
    span.set_attribute("tuft.session_id", session_id)
    span.set_attribute("tuft.training_run_id", training_run_id)
    # ... your logic ...
```

### Recording metrics

Use the centralized metrics registry:

```python
from tuft.telemetry.metrics import get_metrics

metrics = get_metrics()
metrics.futures_created.add(1, {"operation_type": "forward_backward"})
metrics.futures_execution_time.record(0.123, {"operation_type": "forward_backward"})
```

### Choosing attributes

Prefer:

- stable identifiers (`session_id`, `training_run_id`, `sampling_session_id`, `request_id`)
- low-cardinality labels for metrics (e.g. `base_model`, `operation_type`, `queue_state`)

Avoid putting user prompts or raw payloads into span attributes.

---

## Recommended collector configuration

TuFT exports OTLP over **gRPC** by default. We recommend running an OpenTelemetry Collector (or an observability backend that provides an OTLP receiver) listening on the configured `otlp_endpoint` (commonly `4317`).

At minimum, ensure:

- the collector has an **OTLP receiver** enabled (gRPC)
- traces/metrics/logs pipelines are configured to accept OTLP and forward to your backend (Jaeger/Tempo, Prometheus, SigNoz, etc.)

TuFT-specific knobs:

- set `telemetry.service_name` to a stable value (e.g. `tuft-prod`)
- use `telemetry.resource_attributes` to tag deployments (e.g. environment, region, version)

---

## Troubleshooting

### No telemetry shows up

- Ensure `telemetry.enabled: true`.
- Ensure the OTLP endpoint is reachable from the TuFT server process.
- Try debug mode to confirm instrumentation works:

```bash
export TUFT_OTEL_DEBUG=1
```

### GPU metrics missing

- GPU gauges require NVML (`pynvml`) and a working NVML environment. TuFT will gracefully skip GPU metrics if unavailable.
