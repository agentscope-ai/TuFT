#!/usr/bin/env python
"""Run the five-mode evaluation matrix against the same fixed-seed workload.

For each mode this script:
1. Validates the TuFT deployment config (always, even in --dry-run).
2. Launches `tuft launch --config ...` on the given port and GPU subset.
3. Waits for /api/v1/healthz.
4. Runs the simulator (`run.py --config <workload> --output <results>`), which
   talks to TuFT through the Tinker SDK and is completely unaware of the
   deployment mode underneath.
5. Fetches /api/v1/evaluation_metrics and saves it alongside the results.
6. Shuts the server down.

Nothing in the simulator is mode-specific; all differentiation lives in the
TuFT configs under config/.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import yaml


TUFT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = TUFT_ROOT / "config"
DEFAULT_SIMULATOR_DIR = Path("/mnt/nas/hanzhang.yhz/evaluation/simulator")

MODES = {
    "serial_async": "tuft_config_eval_serial_async.yaml",
    "unified_engine": "tuft_config_eval_unified_engine.yaml",
    "colocate_2copies": "tuft_config_eval_colocate_2copies.yaml",
    "static_disagg": "tuft_config_eval_static_disagg.yaml",
    "optimal": "tuft_config_eval_optimal.yaml",
    "optimal_share": "tuft_config_eval_optimal_share.yaml",
    "optimal_duty": "tuft_config_eval_optimal_duty.yaml",
    "optimal_reactive": "tuft_config_eval_optimal_reactive.yaml",
    "optimal_l2": "tuft_config_eval_optimal_l2.yaml",
    "optimal_fl_noholds": "tuft_config_eval_optimal_fl_noholds.yaml",
    "optimal_fl_partial": "tuft_config_eval_optimal_fl_partial.yaml",
    "optimal_l2off": "tuft_config_eval_optimal_l2off.yaml",
    "optimal_l2_300ms": "tuft_config_eval_optimal_l2_300ms.yaml",
    "optimal_quantized": "tuft_config_eval_optimal_quantized.yaml",
    "optimal_dutycycle": "tuft_config_eval_optimal_dutycycle.yaml",
    "optimal_gapfill": "tuft_config_eval_optimal_gapfill.yaml",
    "optimal_gapfill3": "tuft_config_eval_optimal_gapfill3.yaml",
    "optimal_gapfill_l3": "tuft_config_eval_optimal_gapfill_l3.yaml",
    "optimal_window_l3": "tuft_config_eval_optimal_window_l3.yaml",
    "optimal_window_grad": "tuft_config_eval_optimal_window_grad.yaml",
    "optimal_window_full": "tuft_config_eval_optimal_window_full.yaml",
    "optimal_window_l3_p240": "tuft_config_eval_optimal_window_l3_p240.yaml",
    "optimal_window_p40t12": "tuft_config_eval_optimal_window_p40t12.yaml",
    "optimal_window_p60t20": "tuft_config_eval_optimal_window_p60t20.yaml",
    "optimal_window_p90t30": "tuft_config_eval_optimal_window_p90t30.yaml",
    "optimal_window_p12t4": "tuft_config_eval_optimal_window_p12t4.yaml",
    "optimal_window_p16t6": "tuft_config_eval_optimal_window_p16t6.yaml",
    "optimal_window_p20t7": "tuft_config_eval_optimal_window_p20t7.yaml",
    # ── ratio experiment configs ──
    "static_disagg_1_1": "tuft_config_eval_static_disagg_1_1.yaml",
    "serial_async_1_1": "tuft_config_eval_serial_async_1_1.yaml",
    "optimal_1_1": "tuft_config_eval_optimal_1_1.yaml",
    # ── 8-GPU paper evaluation configs ──
    "serial_async_8gpu": "tuft_config_eval_serial_async_8gpu.yaml",
    "unified_engine_8gpu": "tuft_config_eval_unified_engine_8gpu.yaml",
    "colocate_2copies_8gpu": "tuft_config_eval_colocate_2copies_8gpu.yaml",
    "static_disagg_8gpu": "tuft_config_eval_static_disagg_8gpu.yaml",
    "optimal_8gpu": "tuft_config_eval_optimal_8gpu.yaml",
    "ablation_nosched_8gpu": "tuft_config_eval_ablation_nosched_8gpu.yaml",
    "ablation_fastloop_8gpu": "tuft_config_eval_ablation_fastloop_8gpu.yaml",
    "ablation_sampling_8gpu": "tuft_config_eval_ablation_sampling_8gpu.yaml",
    "ablation_sampling_tuned_8gpu": "tuft_config_eval_ablation_sampling_tuned_8gpu.yaml",
    "ablation_reactive_8gpu": "tuft_config_eval_ablation_reactive_8gpu.yaml",
    "ablation_slowloop_8gpu": "tuft_config_eval_ablation_slowloop_8gpu.yaml",
    "slowloop_static_1p3_8gpu": "tuft_config_eval_slowloop_static_1p3_8gpu.yaml",
    "slowloop_grow_1p3_8gpu": "tuft_config_eval_slowloop_grow_1p3_8gpu.yaml",
    "ablation_sb_base_8gpu": "tuft_config_eval_ablation_sb_base_8gpu.yaml",
    "ablation_sb_fastloop_8gpu": "tuft_config_eval_ablation_sb_fastloop_8gpu.yaml",
    "ablation_sb_sampling_8gpu": "tuft_config_eval_ablation_sb_sampling_8gpu.yaml",
    "ablation_sb_reactive_8gpu": "tuft_config_eval_ablation_sb_reactive_8gpu.yaml",
    "ablation_sb_reactive_capped_8gpu":
        "tuft_config_eval_ablation_sb_reactive_capped_8gpu.yaml",
    "ablation_sb_reactive_uncapped_8gpu":
        "tuft_config_eval_ablation_sb_reactive_uncapped_8gpu.yaml",
    "ablation_f7_base_8gpu": "tuft_config_eval_ablation_f7_base_8gpu.yaml",
    "ablation_f7_fastloop_8gpu": "tuft_config_eval_ablation_f7_fastloop_8gpu.yaml",
    "ablation_f7_sampling_8gpu": "tuft_config_eval_ablation_f7_sampling_8gpu.yaml",
    "ablation_f7_duty_8gpu": "tuft_config_eval_ablation_f7_duty_8gpu.yaml",
    "ablation_f7_slowloop_8gpu": "tuft_config_eval_ablation_f7_slowloop_8gpu.yaml",
    "ablation_sb_reactive_longwin_8gpu":
        "tuft_config_eval_ablation_sb_reactive_longwin_8gpu.yaml",
    "ablation_sb_reactive_fill_8gpu":
        "tuft_config_eval_ablation_sb_reactive_fill_8gpu.yaml",
    "ablation_f7_duty_fixed_8gpu":
        "tuft_config_eval_ablation_f7_duty_fixed_8gpu.yaml",
    "paper26_static_disagg_4b": "tuft_config_paper26_static_disagg_4b.yaml",
    "paper26_serial_async_4b": "tuft_config_paper26_serial_async_4b.yaml",
    "paper26_unified_engine_4b": "tuft_config_paper26_unified_engine_4b.yaml",
    "paper26_colocate_2copies_4b": "tuft_config_paper26_colocate_2copies_4b.yaml",
    "paper26_optimal_4b": "tuft_config_paper26_optimal_4b.yaml",
    "paper26_static_disagg_32b": "tuft_config_paper26_static_disagg_32b.yaml",
    "paper26_serial_async_32b": "tuft_config_paper26_serial_async_32b.yaml",
    "paper26_unified_engine_32b": "tuft_config_paper26_unified_engine_32b.yaml",
    "paper26_colocate_2copies_32b": "tuft_config_paper26_colocate_2copies_32b.yaml",
    "paper26_optimal_32b": "tuft_config_paper26_optimal_32b.yaml",
    "paper27_static_32b_memfit": "tuft_config_paper27_static_32b_memfit.yaml",
    "paper27_optimal_2to6_4b": "tuft_config_paper27_optimal_2to6_4b.yaml",
    "paper28_static_1to1_4b": "tuft_config_paper28_static_1to1_4b.yaml",
    "sbound_base_2gpu": "tuft_config_eval_sbound_base_2gpu.yaml",
    "sbound_flexcap_2gpu": "tuft_config_eval_sbound_flexcap_2gpu.yaml",
    "sbound_duty_2gpu": "tuft_config_eval_sbound_duty_2gpu.yaml",
    "starved_base_8gpu": "tuft_config_eval_starved_base_8gpu.yaml",
    "starved_reactive_capped_8gpu":
        "tuft_config_eval_starved_reactive_capped_8gpu.yaml",
    "ablation_sb_slowloop_8gpu": "tuft_config_eval_ablation_sb_slowloop_8gpu.yaml",
    "optimal_fl_noholds_8gpu": "tuft_config_eval_optimal_fl_noholds_8gpu.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=list(MODES),
        choices=list(MODES),
        help="Subset of modes to run",
    )
    parser.add_argument("--workload", required=True, help="Fixed-seed simulator workload YAML")
    parser.add_argument("--results-dir", default="exp/results/eval_matrix")
    parser.add_argument("--simulator-dir", default=str(DEFAULT_SIMULATOR_DIR))
    parser.add_argument(
        "--simulator-python",
        default=None,
        help="Python interpreter for the simulator (defaults to its .venv or sys.executable)",
    )
    parser.add_argument("--gpu-subset", default="0", help="CUDA_VISIBLE_DEVICES for the server")
    parser.add_argument(
        "--gpu-map",
        default="",
        help=(
            "Per-mode GPU override, e.g. 'serial_async=0,1;static_disagg=0,1'. "
            "Coexistence modes (serial_async/static_disagg) need separate GPUs "
            "for training and sampling because standalone vLLM reserves its GPU."
        ),
    )
    parser.add_argument("--port", type=int, default=10610)
    parser.add_argument("--health-timeout-s", type=int, default=600)
    parser.add_argument(
        "--sim-timeout-s",
        type=int,
        default=2700,
        help="Hard cap for one mode's simulator run; on expiry the mode is "
        "recorded as timed out and the matrix moves on.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Only validate configs, do not launch"
    )
    return parser.parse_args()


def validate_tuft_config(config_path: Path) -> None:
    """Parse the YAML through TuFT's own config model to catch schema errors."""
    sys.path.insert(0, str(TUFT_ROOT / "src"))
    from tuft.config import AppConfig

    with open(config_path) as f:
        raw = yaml.safe_load(f)
    config = AppConfig.model_validate(raw)
    config.check_validity()


class GpuUtilSampler:
    """Background nvidia-smi sampler for GPU utilization during a mode run."""

    def __init__(self, gpu_indices: list[int], interval_s: float = 1.0) -> None:
        self.gpu_indices = gpu_indices
        self.interval_s = interval_s
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "GpuUtilSampler":
        if self.gpu_indices:
            self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)

    def _run(self) -> None:
        idx = ",".join(str(i) for i in self.gpu_indices)
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                        "-i",
                        idx,
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                    timeout=10,
                ).strip()
                for line in out.splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) == 3:
                        self.samples.append(
                            {
                                "t": time.time(),
                                "gpu": int(parts[0]),
                                "util": float(parts[1]),
                                "mem_mb": float(parts[2]),
                            }
                        )
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def trace_offset_s(self) -> float | None:
        """Seconds from the first sample to now, i.e. an offset into the trace."""
        if not self.samples:
            return None
        return time.time() - self.samples[0]["t"]

    def trace_epoch_unix_s(self) -> float | None:
        """Unix time of trace t=0, so event logs can be mapped into the trace."""
        if not self.samples:
            return None
        return self.samples[0]["t"]

    def summary(self) -> dict:
        if not self.samples:
            return {"gpu_samples": 0}
        utils = [s["util"] for s in self.samples]
        active = [u for u in utils if u > 0]
        sorted_utils = sorted(utils)
        p90 = sorted_utils[min(len(sorted_utils) - 1, int(0.9 * (len(sorted_utils) - 1)))]
        return {
            "gpu_samples": len(self.samples),
            "gpu_util_avg": sum(utils) / len(utils),
            "gpu_util_active_avg": sum(active) / len(active) if active else 0.0,
            "gpu_util_p90": p90,
            "gpu_util_max": max(utils),
            "gpu_mem_peak_mb": max(s["mem_mb"] for s in self.samples),
        }

    def write_trace(self, path) -> None:
        """Dump the per-sample time series for utilization plots.

        Columns: time_s (relative to the first sample), gpu_index, gpu_util,
        mem_mb. Merge scripts add a ``system`` column and average across GPUs
        to produce the (time_s, system, gpu_util) CSV the figure consumes.
        """
        if not self.samples:
            return
        t0 = self.samples[0]["t"]
        with open(path, "w") as f:
            f.write("time_s,gpu_index,gpu_util,mem_mb\n")
            for s in self.samples:
                f.write(f"{s['t'] - t0:.2f},{s['gpu']},{s['util']:.0f},{s['mem_mb']:.0f}\n")


def workload_window_s(requests_path, trace_epoch_unix_s, sim_end_s):
    """Trace offsets of the phase where the workload really drives the GPUs.

    The simulator spends its first minutes creating adapters, initializing
    tokenizers and building datasets on the CPU; every GPU is idle then, so
    including that head deflates occupancy by tens of points and by a different
    amount per tenant count. The window therefore opens at the first sampling
    request arrival and runs to the simulator exit, which keeps the training
    drain tail (real GPU work) inside it.
    """
    if trace_epoch_unix_s is None or not requests_path.exists():
        return {}
    first_ns = None
    with requests_path.open() as fh:
        for line in fh:
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if ev.get("request_kind") != "sampling":
                continue
            ts = ev.get("timestamp_ns")
            if ts is not None and (first_ns is None or ts < first_ns):
                first_ns = ts
    if first_ns is None:
        return {}
    start_s = first_ns / 1e9 - trace_epoch_unix_s
    if sim_end_s is not None and start_s >= sim_end_s:
        return {}
    return {"workload_start_s": start_s, "workload_end_s": sim_end_s}


def port_already_serving(base_url: str) -> bool:
    """True if a TuFT server is already answering on the port.

    `tuft launch` cannot bind a busy port, but `wait_for_health` would happily
    pass against the *foreign* server and the simulator would then drive
    somebody else's deployment: two workloads on one server, and the scraped
    metrics belong to neither run. Measured that way once (a 24-tenant run
    landed on a concurrent 8-tenant arm's server and both results were junk),
    so the port is checked before the launch and ownership after it.
    """
    try:
        with urllib.request.urlopen(f"{base_url}/api/v1/healthz", timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def wait_for_health(base_url: str, timeout_s: int) -> bool:
    deadline = time.monotonic() + timeout_s
    url = f"{base_url}/api/v1/healthz"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(3)
    return False


def fetch_evaluation_metrics(base_url: str) -> dict:
    url = f"{base_url}/api/v1/evaluation_metrics"
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def server_is_ours(base_url: str, server: subprocess.Popen) -> tuple[bool, str]:
    """Check the server answering the port is the one we launched.

    Watching our own process is not enough. `tuft launch` can lose the GPU race
    to a concurrent campaign, have uvicorn report "Application startup failed"
    and yet linger (Ray actors keep the process alive), so `poll()` stays None
    while the port is answered by the other campaign's server. That is exactly
    how a sat-probe arm produced a full result set measuring a window-mode arm
    it never launched. The server reports its own pid, so ownership is checked
    directly instead of inferred.
    """
    try:
        pid = fetch_evaluation_metrics(base_url).get("server_pid")
    except Exception as e:  # pylint: disable=broad-except
        return False, f"cannot read evaluation metrics: {e}"
    if pid is None:
        # Older server without the marker: fall back to liveness only.
        return server.poll() is None, "server does not report server_pid"
    if int(pid) != server.pid:
        return False, f"port answered by pid {pid}, our server is pid {server.pid}"
    return True, "ok"


def parse_gpu_map(raw: str) -> dict:
    gpu_map = {}
    for item in raw.split(";"):
        item = item.strip()
        if not item or "=" not in item:
            continue
        mode, gpus = item.split("=", 1)
        gpu_map[mode.strip()] = gpus.strip()
    return gpu_map


def _kill_process_group(proc: subprocess.Popen) -> None:
    """Terminate a subprocess and its whole process group (no orphans).

    Orphaned simulator processes keep retrying against the port with stale
    sessions and poison subsequent mode runs, so cleanup must be group-wide.
    """
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
    except (ProcessLookupError, PermissionError):
        pass


def wait_for_simulator(
    sim_proc: subprocess.Popen, server: subprocess.Popen, timeout_s: int
) -> tuple[int, bool]:
    """Wait for the simulator, aborting if our own server dies first.

    The pre-launch port check only sees a *healthy* foreign server. A concurrent
    campaign whose server is still initializing does not answer /healthz yet, so
    the check passes, our `tuft launch` then loses the GPU race and exits -- and
    from that moment the port is answered by the other campaign's server while
    our simulator keeps happily driving it. That produced a full 24-tenant
    result set belonging to nobody (quarantined in
    exp/results/_invalid_concurrent_port/). Checking server liveness once before
    the simulator starts is not enough; it has to be watched throughout.

    Returns (simulator_rc, server_died).
    """
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            return sim_proc.wait(timeout=5), False
        except subprocess.TimeoutExpired:
            pass
        if server.poll() is not None:
            return 3, True
        if time.monotonic() >= deadline:
            raise subprocess.TimeoutExpired(sim_proc.args, timeout_s)


def run_mode(args: argparse.Namespace, mode: str, config_name: str) -> int:
    config_path = CONFIG_DIR / config_name
    print(f"[{mode}] validating {config_path}")
    validate_tuft_config(config_path)
    if args.dry_run:
        print(f"[{mode}] dry-run OK")
        return 0

    results_dir = Path(args.results_dir) / mode
    results_dir.mkdir(parents=True, exist_ok=True)
    base_url = f"http://127.0.0.1:{args.port}"
    server_log = open(results_dir / "tuft_server.log", "w")

    gpu_subset = parse_gpu_map(args.gpu_map).get(mode, args.gpu_subset)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_subset
    physical = [
        int(part) for part in gpu_subset.split(",") if part.strip().lstrip("-").isdigit()
    ]
    # CUDA_VISIBLE_DEVICES renumbers devices to 0..n-1 inside the child, so the
    # indices the backends call set_device() with must be logical, not physical.
    # Using the physical ids here worked only because every run so far started at
    # GPU 0; a run pinned to 4,5,6,7 died with "CUDA error: invalid device
    # ordinal" because set_device(4) has no device 4 to find.
    gpu_indices = list(range(len(physical)))

    tuft_bin = shutil.which("tuft") or str(TUFT_ROOT / ".venv" / "bin" / "tuft")
    if port_already_serving(base_url):
        print(
            f"[{mode}] port {args.port} is already serving a TuFT server; refusing to run "
            "(another campaign owns the GPUs -- kill it or pass a free --port)",
            file=sys.stderr,
        )
        return 3
    with GpuUtilSampler(gpu_indices) as gpu_sampler:
        server = subprocess.Popen(
            [tuft_bin, "launch", "--config", str(config_path), "--port", str(args.port)],
            env=env,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            cwd=str(TUFT_ROOT),
        )
        try:
            if not wait_for_health(base_url, args.health_timeout_s):
                print(f"[{mode}] server failed to become healthy", file=sys.stderr)
                return 2
            if server.poll() is not None:
                print(
                    f"[{mode}] our server exited (rc={server.returncode}) yet port "
                    f"{args.port} answers: the healthy server is somebody else's",
                    file=sys.stderr,
                )
                return 3
            owned, why = server_is_ours(base_url, server)
            if not owned:
                print(
                    f"[{mode}] refusing to run: {why} -- the healthy server on port "
                    f"{args.port} is not the one we launched",
                    file=sys.stderr,
                )
                return 3

            simulator_dir = Path(args.simulator_dir)
            sim_python = args.simulator_python or str(simulator_dir / ".venv" / "bin" / "python")
            if not Path(sim_python).exists():
                sim_python = sys.executable
            # Run the simulator in its own process group so it can be killed as a
            # whole: orphaned simulator processes keep hammering the port with
            # stale sessions and poison subsequent mode runs.
            sim_proc = subprocess.Popen(
                [
                    sim_python,
                    str(simulator_dir / "run.py"),
                    "--config",
                    args.workload,
                    "--output",
                    str(results_dir / "simulator_results.json"),
                ],
                cwd=str(simulator_dir),
                start_new_session=True,
            )
            sim_window_start_s = gpu_sampler.trace_offset_s()
            server_died = False
            try:
                completed_rc, server_died = wait_for_simulator(
                    sim_proc, server, args.sim_timeout_s
                )
            except subprocess.TimeoutExpired:
                print(
                    f"[{mode}] simulator timed out after {args.sim_timeout_s}s; moving on",
                    file=sys.stderr,
                )
                completed_rc = 124
            finally:
                sim_window_end_s = gpu_sampler.trace_offset_s()
                _kill_process_group(sim_proc)
            if server_died:
                print(
                    f"[{mode}] our server exited (rc={server.returncode}) mid-run: the "
                    "simulator was driving another campaign's deployment; discarding "
                    "this arm instead of saving metrics",
                    file=sys.stderr,
                )
                return 3
            if completed_rc != 0:
                print(f"[{mode}] simulator exited with {completed_rc}", file=sys.stderr)

            owned, why = server_is_ours(base_url, server)
            if not owned:
                print(
                    f"[{mode}] discarding this arm: {why} -- the simulator was driving "
                    "another campaign's deployment",
                    file=sys.stderr,
                )
                return 3

            try:
                metrics = fetch_evaluation_metrics(base_url)
            except Exception as e:
                print(f"[{mode}] failed to fetch evaluation metrics: {e}", file=sys.stderr)
                metrics = {}
            metrics["gpu_utilization"] = gpu_sampler.summary()
            # gpu_trace.csv starts before the server launch, so the raw trace
            # includes model load / CUDA-graph capture. Occupancy has to be read
            # over this window or startup silently deflates every busy fraction
            # (it differs by minutes between modes).
            metrics["gpu_trace_window_s"] = {
                "sim_start_s": sim_window_start_s,
                "sim_end_s": sim_window_end_s,
                "trace_epoch_unix_s": gpu_sampler.trace_epoch_unix_s(),
            }
            metrics["gpu_trace_window_s"].update(
                workload_window_s(
                    results_dir / "simulator_results.requests.jsonl",
                    gpu_sampler.trace_epoch_unix_s(),
                    sim_window_end_s,
                )
            )
            gpu_sampler.write_trace(results_dir / "gpu_trace.csv")
            with open(results_dir / "tuft_eval_metrics.json", "w") as f:
                json.dump(metrics, f, indent=2, default=str)
            print(f"[{mode}] metrics saved to {results_dir / 'tuft_eval_metrics.json'}")
            return completed_rc
        finally:
            server.send_signal(signal.SIGINT)
            try:
                server.wait(timeout=60)
            except subprocess.TimeoutExpired:
                server.kill()
            server_log.close()


def main() -> int:
    args = parse_args()
    # The simulator subprocess runs from its own directory; keep paths absolute.
    args.workload = str(Path(args.workload).resolve())
    args.results_dir = str(Path(args.results_dir).resolve())
    if not Path(args.workload).exists() and not args.dry_run:
        print(f"workload file not found: {args.workload}", file=sys.stderr)
        return 1
    rc = 0
    for mode in args.modes:
        try:
            mode_rc = run_mode(args, mode, MODES[mode])
        except Exception as e:
            print(f"[{mode}] failed: {e}", file=sys.stderr)
            mode_rc = 1
        rc = rc or mode_rc
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
