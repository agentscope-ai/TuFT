#!/usr/bin/env python3
"""Summarize 4-GPU campaign outputs into plotting CSV files."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path('/mnt/nas/hanzhang.yhz/flex_backend/TuFT/exp/results/campaign')
OUT = Path('/mnt/nas/hanzhang.yhz/flex_backend/TuFT/exp/results/campaign_summary')
OUT.mkdir(parents=True, exist_ok=True)
SYSTEM_MAP = {
    'serial_async': 'Serial-Async',
    'unified_engine': 'Unified-Engine',
    'colocate_2copies': 'Colocate-2Copies',
    'static_disagg': 'Static-Disagg',
    'optimal': 'TuFT',
}
MODES = ['serial_async', 'unified_engine', 'colocate_2copies', 'static_disagg', 'optimal']
TENANTS = [1, 2, 4, 8, 16, 24]

def load_json(path: Path) -> dict:
    return json.loads(path.read_text())

def iter_runs():
    for mode in MODES:
        for n in TENANTS:
            run_dir = ROOT / f'{mode}_t{n}' / mode
            sim_path = run_dir / 'simulator_results.json'
            if not sim_path.exists():
                continue
            yield mode, SYSTEM_MAP[mode], n, run_dir, load_json(sim_path)

# Per-run summary: throughput, latency, freshness, accuracy, GPU stats, switch stats.
rows = []
for mode, system, tenants, run_dir, sim in iter_runs():
    per = sim.get('per_tenant', {})
    total_steps = sum(float(t.get('steps', 0)) for t in per.values())
    total_samples = sum(float(t.get('samples', 0)) for t in per.values())
    wall = float(sim.get('total_wall_clock_seconds', 0) or 0)
    metrics = load_json(run_dir / 'tuft_eval_metrics.json') if (run_dir / 'tuft_eval_metrics.json').exists() else {}
    gpu = metrics.get('gpu_utilization', {})
    ev = metrics.get('evaluation', {})
    br = ev.get('iteration_breakdown', {}) if isinstance(ev, dict) else {}
    rows.append({
        'system': system,
        'mode': mode,
        'tenants': tenants,
        'wall_clock_s': f'{wall:.3f}',
        'completed_steps': f'{total_steps:.0f}',
        'samples': f'{total_samples:.0f}',
        'steps_per_hour': f'{(total_steps / wall * 3600) if wall else 0:.6f}',
        'mean_accuracy': f'{(sum(float(t.get("accuracy", 0)) for t in per.values()) / len(per)) if per else 0:.6f}',
        'mean_staleness': f'{(sum(float(t.get("staleness", 0)) for t in per.values()) / len(per)) if per else 0:.6f}',
        'sampling_s_total': f'{sum(float(t.get("sampling", 0)) for t in per.values()):.3f}',
        'training_s_total': f'{sum(float(t.get("training", 0)) for t in per.values()):.3f}',
        'sync_weights_s_total': f'{sum(float(t.get("sync_weights", 0)) for t in per.values()):.3f}',
        'gpu_util_avg': f'{float(gpu.get("gpu_util_avg", 0)):.6f}',
        'gpu_util_active_avg': f'{float(gpu.get("gpu_util_active_avg", 0)):.6f}',
        'gpu_util_p90': f'{float(gpu.get("gpu_util_p90", 0)):.6f}',
        'gpu_mem_peak_mb': f'{float(gpu.get("gpu_mem_peak_mb", 0)):.1f}',
        'gate_wait_training_s': f'{float(br.get("gate_wait_training_s", 0)):.6f}',
        'gate_wait_sampling_s': f'{float(br.get("gate_wait_sampling_s", 0)):.6f}',
        'flex_switches': f'{float(br.get("flex_switches", 0)):.0f}',
        'flex_switch_total_ms': f'{float(br.get("flex_switch_total_ms", 0)):.3f}',
    })
with open(OUT / 'campaign_summary.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

# Request latency rows from simulator request event logs.
latency_rows = []
for mode, system, tenants, run_dir, _sim in iter_runs():
    req_path = run_dir / 'simulator_results.requests.jsonl'
    if not req_path.exists():
        continue
    arrivals: dict[tuple[str, str, int], int] = {}
    with open(req_path) as f:
        for line in f:
            ev = json.loads(line)
            event = ev.get('event')
            key = (ev.get('tenant_id', ''), ev.get('operation_type', ''), int(ev.get('train_step', -1)))
            if event == 'request_arrival':
                arrivals[key] = int(ev.get('monotonic_ns', 0))
            elif event in ('request_complete', 'request_completed', 'training_complete', 'sample_complete'):
                start = arrivals.get(key)
                if start:
                    latency_rows.append({
                        'system': system,
                        'mode': mode,
                        'tenants': tenants,
                        'tenant_id': key[0],
                        'operation_type': key[1],
                        'train_step': key[2],
                        'latency_s': f'{(int(ev.get("monotonic_ns", 0)) - start) / 1e9:.6f}',
                    })
if latency_rows:
    with open(OUT / 'request_latencies.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(latency_rows[0].keys()))
        w.writeheader(); w.writerows(latency_rows)

# GPU trace: average per timestamp across GPUs and emit one row per system/run.
gpu_rows = []
for mode, system, tenants, run_dir, _sim in iter_runs():
    trace = run_dir / 'gpu_trace.csv'
    if not trace.exists():
        continue
    buckets: dict[float, list[float]] = {}
    with open(trace) as f:
        r = csv.DictReader(f)
        for row in r:
            t = round(float(row['time_s']))
            buckets.setdefault(t, []).append(float(row['gpu_util']))
    for t in sorted(buckets):
        vals = buckets[t]
        gpu_rows.append({
            'time_s': f'{t:.0f}',
            'system': system,
            'mode': mode,
            'tenants': tenants,
            'gpu_util': f'{sum(vals)/len(vals):.6f}',
        })
if gpu_rows:
    with open(OUT / 'gpu_utilization_traces.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(gpu_rows[0].keys()))
        w.writeheader(); w.writerows(gpu_rows)

print(OUT / 'campaign_summary.csv')
print(f'runs={len(rows)}')
print(f'latency_rows={len(latency_rows)} gpu_rows={len(gpu_rows)}')
