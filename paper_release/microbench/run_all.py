#!/usr/bin/env python
"""Main orchestrator: runs all mode-switch benchmark experiments.

Iterates over all (model, tp_config, approach) combinations,
runs each benchmark in a subprocess for GPU memory isolation,
collects results into a unified JSON report.

Usage:
    # Run all experiments
    python run_all.py

    # Run specific model only
    python run_all.py --model qwen3-4b

    # Run specific approach only
    python run_all.py --approach flexgpu

    # Run specific config
    python run_all.py --model qwen3-4b --tp-size 4 --approach flexgpu

    # Quick test (reduced rounds)
    python run_all.py --quick
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

EXP_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = EXP_ROOT / "results"

# Use TuFT venv Python for correct dependency versions
PYTHON = os.environ.get(
    "BENCH_PYTHON",
    "/mnt/nas/hanzhang.yhz/flex_backend/TuFT/.venv/bin/python",
)

sys.path.insert(0, str(EXP_ROOT))
from config import APPROACHES, EXPERIMENTS, MODELS, WORKLOADS

# ─── Benchmark Scripts ────────────────────────────────────────────────────────

APPROACH_SCRIPTS = {
    "coexistence": EXP_ROOT / "bench_approach1_coexist.py",
    "process_switch": EXP_ROOT / "bench_approach2_process.py",
    "sleep_wake": EXP_ROOT / "bench_approach3_sleepwake.py",
    "flexgpu": EXP_ROOT / "bench_approach4_flexgpu_roundtrip.py",
}


def run_single_benchmark(
    approach: str,
    model: str,
    tp_size: int,
    warmup_rounds: int = 1,
    measure_rounds: int = 3,
    timeout_s: int = 3600,
) -> dict[str, Any] | None:
    """Run a single benchmark in a subprocess.

    Returns parsed result dict or None on failure.
    """
    script = APPROACH_SCRIPTS[approach]
    if not script.exists():
        print(f"    [SKIP] Script not found: {script}")
        return None

    cmd = [
        PYTHON, str(script),
        "--model", model,
        "--tp-size", str(tp_size),
        "--warmup-rounds", str(warmup_rounds),
        "--measure-rounds", str(measure_rounds),
        "--output-dir", str(RESULTS_DIR),
    ]

    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    env.setdefault("CUDA_HOME", "/usr/local/cuda-12.9")
    env["PATH"] = "/usr/local/cuda-12.9/bin:" + env.get("PATH", "")

    print(f"    Running: {' '.join(cmd[-8:])}")
    start = time.perf_counter()

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
            cwd=str(EXP_ROOT),
        )
    except subprocess.TimeoutExpired:
        print(f"    [TIMEOUT] Exceeded {timeout_s}s")
        return {"error": f"timeout after {timeout_s}s", "approach": approach,
                "model": model, "tp_size": tp_size}

    elapsed = time.perf_counter() - start
    output = proc.stdout + "\n" + proc.stderr

    if proc.returncode != 0:
        print(f"    [FAILED] returncode={proc.returncode} ({elapsed:.0f}s)")
        # Print last few lines for debugging
        lines = output.strip().split("\n")
        for line in lines[-10:]:
            print(f"      {line}")
        return {"error": output[-2000:], "approach": approach,
                "model": model, "tp_size": tp_size, "returncode": proc.returncode}

    # Parse JSON result from output
    result = None
    for line in output.split("\n"):
        if "__RESULT_JSON__" in line:
            json_str = line.split("__RESULT_JSON__")[1].strip()
            try:
                result = json.loads(json_str)
            except json.JSONDecodeError:
                pass

    if result is None:
        # Try to find JSON in output
        for line in reversed(output.split("\n")):
            line = line.strip()
            if line.startswith("{") and '"approach"' in line:
                try:
                    result = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue

    if result:
        print(f"    [OK] {elapsed:.0f}s | "
              f"samp={result.get('sampling_throughput_toks', 0):.0f} tok/s | "
              f"train={result.get('training_step_ms', 0):.0f}ms | "
              f"switch={result.get('switch_t2s_ms', 0):.0f}ms | "
              f"oom={result.get('oom', False)}")
    else:
        print(f"    [WARN] No JSON result found in output ({elapsed:.0f}s)")

    return result


def format_summary_table(results: list[dict[str, Any]]) -> str:
    """Format results as a paper-friendly summary table with cold/hot."""
    lines = []
    lines.append("")
    lines.append("=" * 140)
    lines.append("  MODE-SWITCH MICRO-BENCHMARK RESULTS (Cold / Hot)")
    lines.append("=" * 140)
    lines.append(
        f"{'Model':<14} {'TP':>3} {'DP':>3} {'Approach':<22} "
        f"{'T→S_C':>9} {'T→S_H':>9} {'S→T_C':>9} {'S→T_H':>9} "
        f"{'Samp':>8} {'Train':>8} {'MemT':>7} {'MemS':>7} "
        f"{'FreeT':>7} {'FreeS':>7} {'Status':>8}"
    )
    lines.append("-" * 140)

    for r in results:
        if "error" in r and "approach" not in r:
            continue
        model = r.get("model", "?")[:13]
        tp = r.get("tp_size", 0)
        dp = r.get("dp_size", 1)
        approach = r.get("approach_name", r.get("approach", "?"))[:21]
        t2s_c = r.get("cold_switch_t2s_ms", 0)
        t2s_h = r.get("hot_switch_t2s_ms", r.get("switch_t2s_ms", 0))
        s2t_c = r.get("cold_switch_s2t_ms", 0)
        s2t_h = r.get("hot_switch_s2t_ms", r.get("switch_s2t_ms", 0))
        # Mean sampling throughput and training step time
        samp = r.get("sampling_throughput_toks", 0)
        train = r.get("training_step_ms", 0)
        mem_t = r.get("peak_memory_training_gb", 0)
        mem_s = r.get("peak_memory_sampling_gb", 0)
        free_t = r.get("free_memory_after_t2s_release_gb", 0)
        free_s = r.get("free_memory_after_s2t_release_gb", 0)
        oom = r.get("oom", False)
        oom_phase = r.get("oom_phase", "")

        if oom:
            status = f"OOM({oom_phase[:4]})"
        elif "error" in r:
            status = "ERROR"
        else:
            status = "OK"

        lines.append(
            f"{model:<14} {tp:>3} {dp:>3} {approach:<22} "
            f"{t2s_c:>9.1f} {t2s_h:>9.1f} {s2t_c:>9.1f} {s2t_h:>9.1f} "
            f"{samp:>8.0f} {train:>8.0f} {mem_t:>7.1f} {mem_s:>7.1f} "
            f"{free_t:>7.1f} {free_s:>7.1f} {status:>8}"
        )

    lines.append("=" * 140)
    lines.append("  T→S_C/H = Cold/Hot training→sampling switch latency (ms)")
    lines.append("  S→T_C/H = Cold/Hot sampling→training switch latency (ms)")
    lines.append("  Samp = mean sampling throughput (tok/s), Train = mean training step time (ms)")
    lines.append("  MemT/MemS = peak training/sampling GPU memory (GB); FreeT/FreeS = free GPU memory after T→S/S→T release when available")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all mode-switch benchmarks")
    parser.add_argument("--model", type=str, default="",
                        help="Run only this model (qwen3-4b or qwen3-32b)")
    parser.add_argument("--approach", type=str, default="",
                        help="Run only this approach")
    parser.add_argument("--tp-size", type=int, default=0,
                        help="Run only this TP size")
    parser.add_argument("--warmup-rounds", type=int, default=0)
    parser.add_argument("--measure-rounds", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=3600,
                        help="Per-benchmark timeout in seconds")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 1 warmup, 2 measure rounds")
    args = parser.parse_args()

    if args.quick:
        args.warmup_rounds = 1
        args.measure_rounds = 2

    # Determine which experiments to run
    models_to_run = [args.model] if args.model else list(EXPERIMENTS.keys())
    approaches_to_run = [args.approach] if args.approach else list(APPROACHES.keys())

    print("=" * 70)
    print("  MODE-SWITCH MICRO-BENCHMARK")
    print(f"  Time: {datetime.now().isoformat()}")
    print(f"  Models: {models_to_run}")
    print(f"  Approaches: {approaches_to_run}")
    print(f"  Rounds: warmup={args.warmup_rounds}, measure={args.measure_rounds}")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results: list[dict[str, Any]] = []
    total_start = time.perf_counter()

    for model_key in models_to_run:
        if model_key not in EXPERIMENTS:
            print(f"\n  [SKIP] Unknown model: {model_key}")
            continue

        model_cfg = MODELS[model_key]
        parallel_configs = EXPERIMENTS[model_key]

        for pconfig in parallel_configs:
            if args.tp_size > 0 and pconfig.tp_size != args.tp_size:
                continue

            print(f"\n{'─'*70}")
            print(f"  Model: {model_cfg['name']} | TP={pconfig.tp_size} DP={pconfig.dp_size}")
            print(f"{'─'*70}")

            for approach_key in approaches_to_run:
                if approach_key not in APPROACHES:
                    print(f"  [SKIP] Unknown approach: {approach_key}")
                    continue

                approach_info = APPROACHES[approach_key]
                print(f"\n  [{approach_info['id']}/4] {approach_info['name']}")

                result = run_single_benchmark(
                    approach=approach_key,
                    model=model_key,
                    tp_size=pconfig.tp_size,
                    warmup_rounds=args.warmup_rounds,
                    measure_rounds=args.measure_rounds,
                    timeout_s=args.timeout,
                )

                if result is None:
                    result = {
                        "model": model_cfg["name"],
                        "tp_size": pconfig.tp_size,
                        "dp_size": pconfig.dp_size,
                        "approach": approach_key,
                        "approach_name": approach_info["name"],
                        "error": "no result",
                    }
                else:
                    # Ensure dp_size is set
                    result["dp_size"] = pconfig.dp_size

                all_results.append(result)

                # Brief pause between benchmarks for GPU cleanup
                time.sleep(3)

    total_elapsed = time.perf_counter() - total_start

    # ─── Save unified results ─────────────────────────────────────────────────
    report = {
        "experiment": "mode_switch_micro_benchmark",
        "timestamp": datetime.now().isoformat(),
        "hardware": "4x NVIDIA A100-SXM4-80GB",
        "total_elapsed_s": total_elapsed,
        "config": {
            "warmup_rounds": args.warmup_rounds,
            "measure_rounds": args.measure_rounds,
        },
        "results": all_results,
    }

    report_path = RESULTS_DIR / f"benchmark_report_{datetime.now():%Y%m%d_%H%M%S}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n  Report saved: {report_path}")

    # ─── Print summary table ──────────────────────────────────────────────────
    print(format_summary_table(all_results))

    # ─── Print DP=2 throughput estimates ──────────────────────────────────────
    print("\n  DP=2 Throughput Estimates (multiply single-group by 2):")
    print(f"  {'Model':<14} {'TP':>3} {'Approach':<22} {'Samp×2(tok/s)':>14} {'Train×2(tok/s)':>15}")
    print(f"  {'-'*70}")
    for r in all_results:
        if r.get("dp_size", 1) == 2 and not r.get("oom", False) and "error" not in r:
            samp2 = r.get("sampling_throughput_toks", 0) * 2
            train_toks = r.get("training_throughput_toks", 0) * 2
            print(f"  {r.get('model', '?'):<14} {r.get('tp_size', 0):>3} "
                  f"{r.get('approach_name', '?'):<22} {samp2:>14.1f} {train_toks:>15.1f}")

    print(f"\n  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
    print("DONE")


if __name__ == "__main__":
    main()
