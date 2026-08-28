"""Mock evaluation figures for the TuFT paper.

Each ``#%%`` cell can be run independently after the setup cell.  The numbers
below are placeholders chosen to match the evaluation narrative in ``main.tex``;
replace the mock dictionaries/arrays with measured data when experiments finish.
"""

#%% Setup
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

try:
    import pandas as pd
except ImportError:  # The GPU-utilization mock path works without pandas.
    pd = None

sysname = "TuFT"
baselines = ["Serial-Async", "Unified-Engine", "Colocate-2Copies", "Static-Disagg", sysname]
short_names = ["Serial", "Unified", "Colocate", "Static", sysname]

PALETTE = {
    "paper": "#F9F7F7",
    "mist": "#DBE2EF",
    "blue": "#3F72AF",
    "navy": "#112D4E",
    "steel": "#6096B4",
    "cyan": "#93BFCF",
    "pale": "#BDCDD6",
    "sand": "#EEE9DA",
    "tuft": "#AA96DA",
}
SYSTEM_COLORS = {
    "Serial-Async": PALETTE["pale"],
    "Unified-Engine": PALETTE["sand"],
    "Colocate-2Copies": PALETTE["cyan"],
    "Static-Disagg": PALETTE["steel"],
    sysname: PALETTE["tuft"],
}
SYSTEM_MARKERS = {
    "Serial-Async": "o",
    "Unified-Engine": "s",
    "Colocate-2Copies": "^",
    "Static-Disagg": "D",
    sysname: "*",
}
BASELINE_COLORS = [SYSTEM_COLORS[name] for name in baselines]

ROOT_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT_DIR / "figures" / "exp_figure"
OUT_DIR.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10.5,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3.2,
        "ytick.major.size": 3.2,
        "lines.linewidth": 1.35,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def style_axis(ax: plt.Axes, *, grid_axis: str | None = "y") -> None:
    ax.set_facecolor("white")
    if grid_axis is not None:
        ax.grid(axis=grid_axis, linestyle="--", linewidth=0.45, alpha=0.35, color=PALETTE["navy"])
    else:
        ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PALETTE["navy"])
    ax.spines["bottom"].set_color(PALETTE["navy"])
    ax.tick_params(colors=PALETTE["navy"])
    ax.xaxis.label.set_color(PALETTE["navy"])
    ax.yaxis.label.set_color(PALETTE["navy"])


def save_figure(fig: plt.Figure, stem: str) -> None:
    pdf_path = OUT_DIR / f"{stem}.pdf"
    png_path = OUT_DIR / f"{stem}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    print(pdf_path)


def annotate_bars(ax: plt.Axes, bars, fmt: str = "{:.0f}", dy: float = 2.0) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + dy,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=9.5,
            color=PALETTE["navy"],
        )


#%% Throughput under 4/8 GPU budget
# Unit: completed RL training steps per hour after warm-up.
throughtput_4_GPUs = np.array([15.8, 11.7, 20.1, 18.4, 27.6])
throughtput_8_GPUs = np.array([31.5, 23.8, 39.2, 36.1, 57.8])
gpu_utilization_8 = np.array([54.0, 48.0, 66.0, 61.0, 83.0])

fig, ax = plt.subplots(figsize=(3.05, 2.55))
gpu_groups = ["GPU=4", "GPU=8"]
throughput_by_gpu = np.vstack([throughtput_4_GPUs, throughtput_8_GPUs])
group_x = np.arange(len(gpu_groups))
width = 0.13
offsets = (np.arange(len(baselines)) - (len(baselines) - 1) / 2) * width
for idx, name in enumerate(baselines):
    bars = ax.bar(
        group_x + offsets[idx],
        throughput_by_gpu[:, idx],
        width,
        label=name,
        color=SYSTEM_COLORS[name],
        edgecolor=PALETTE["paper"],
        linewidth=0.8,
        zorder=3,
    )
    if name == sysname:
        annotate_bars(ax, bars, fmt="{:.1f}", dy=1.0)
ax.set_xticks(group_x)
ax.set_xticklabels(gpu_groups)
ax.set_ylabel("Training steps / hour")
ax.set_ylim(0, 66)
ax.legend(frameon=False, ncol=1, loc="upper left", fontsize=8.8, handlelength=1.0)
strongest = throughtput_8_GPUs[:-1].max()
improvement = throughtput_8_GPUs[-1] / strongest
ax.text(
    group_x[-1] + offsets[-1],
    throughtput_8_GPUs[-1] + 6.0,
    f"{improvement:.2f}×",
    ha="center",
    va="bottom",
    fontsize=10,
    color=PALETTE["tuft"],
    fontweight="bold",
)
style_axis(ax)
fig.tight_layout()
save_figure(fig, "eval_throughput_gpu_budget")


#%% Scaling with tenant counts
tenants = np.array([1, 2, 4, 8, 16, 32])
# Unit: aggregate completed RL training steps/hour. Serial-Async is time-shared
# after four tenants because isolated copies do not fit beyond that point.
scaling_throughput = {
    "Serial-Async": np.array([8.8, 15.9, 22.4, 22.0, 18.2, 13.7]),
    "Unified-Engine": np.array([6.9, 12.6, 18.1, 20.4, 19.2, 15.6]),
    "Colocate-2Copies": np.array([9.1, 17.4, 29.0, 38.2, 34.5, 25.8]),
    "Static-Disagg": np.array([8.4, 16.8, 28.9, 36.1, 35.4, 28.3]),
    sysname: np.array([8.7, 17.9, 32.5, 49.4, 57.8, 55.2]),
}
# Unit: minutes to finish a fixed 3,200-step evaluation window.
finished_time = {name: 3200.0 / values * 60.0 for name, values in scaling_throughput.items()}

fig, ax = plt.subplots(figsize=(3.05, 2.55))
for idx, name in enumerate(baselines):
    ax.plot(
        tenants,
        scaling_throughput[name],
        marker=SYSTEM_MARKERS[name],
        markersize=6.4 if name == sysname else 4.0,
        color=SYSTEM_COLORS[name],
        linewidth=2.2 if name == sysname else 1.2,
        label=name,
    )
ax.axvspan(4, 8, color=PALETTE["sand"], alpha=0.32, linewidth=0)
ax.axvline(16, color=PALETTE["tuft"], linestyle="--", linewidth=1.0, alpha=0.85)
ax.text(4.25, 62.0, "baseline knee", fontsize=9.0, color=PALETTE["navy"])
ax.text(16.5, 62.0, "TuFT knee", fontsize=9.0, color=PALETTE["tuft"], fontweight="bold")
ax.set_xscale("log", base=2)
ax.set_xticks(tenants)
ax.set_xticklabels([str(t) for t in tenants])
ax.set_xlabel("Concurrent tenants")
ax.set_ylabel("Training steps / hour")
ax.set_ylim(0, 66)
ax.legend(frameon=False, ncol=1, loc="lower right", fontsize=8.6, handlelength=1.2)
style_axis(ax)
fig.tight_layout()
save_figure(fig, "eval_scaling_tenants")


#%% Response time
# Unit: seconds.  One grouped boxplot shows the two tenant-visible delays.
response_median_p95 = {
    "Training": {
        "median": np.array([97.0, 18.5, 33.0, 52.0, 16.0]),
        "p95": np.array([245.0, 43.0, 86.0, 138.0, 35.0]),
        "slo": 60.0,
    },
    "Sampling": {
        "median": np.array([12.5, 31.0, 17.0, 21.5, 14.2]),
        "p95": np.array([28.0, 74.0, 42.0, 59.0, 31.0]),
        "slo": 45.0,
    },
}

rng = np.random.default_rng(23)
box_data: list[np.ndarray] = []
positions: list[float] = []
centers = [1.0, 2.55]
inner_offsets = np.linspace(-0.34, 0.34, len(baselines))
for group_idx, metric in enumerate(["Training", "Sampling"]):
    for baseline_idx, name in enumerate(baselines):
        median = response_median_p95[metric]["median"][baseline_idx]
        p95 = response_median_p95[metric]["p95"][baseline_idx]
        sigma = max((p95 - median) / 1.65, median * 0.08)
        samples = np.clip(rng.normal(median, sigma, 180), 0.5, None)
        box_data.append(samples)
        positions.append(centers[group_idx] + inner_offsets[baseline_idx])

fig, ax = plt.subplots(figsize=(3.25, 2.45))
box = ax.boxplot(
    box_data,
    positions=positions,
    widths=0.12,
    patch_artist=True,
    showfliers=False,
    medianprops={"color": PALETTE["navy"], "linewidth": 0.9},
    whiskerprops={"color": PALETTE["navy"], "linewidth": 0.7},
    capprops={"color": PALETTE["navy"], "linewidth": 0.7},
)
for idx, patch in enumerate(box["boxes"]):
    name = baselines[idx % len(baselines)]
    patch.set_facecolor(SYSTEM_COLORS[name])
    patch.set_edgecolor(PALETTE["paper"])
    patch.set_linewidth(0.8)
ax.axhline(response_median_p95["Training"]["slo"], linestyle="--", linewidth=0.85, color=PALETTE["navy"], alpha=0.65)
ax.axhline(response_median_p95["Sampling"]["slo"], linestyle=":", linewidth=0.95, color=PALETTE["navy"], alpha=0.65)
ax.set_xticks(centers)
ax.set_xticklabels(["Training", "Sampling"])
ax.set_ylabel("Latency (s)")
ax.set_xlim(0.35, 3.2)
ax.set_ylim(0, 260)
legend_handles = [Patch(facecolor=SYSTEM_COLORS[name], edgecolor=PALETTE["paper"], label=name) for name in baselines]
ax.legend(handles=legend_handles, frameon=False, ncol=1, loc="upper right", fontsize=8.4)
style_axis(ax)
fig.tight_layout()
save_figure(fig, "eval_response_time")


#%% Fidelity
# Same-token average absolute log-probability mismatch for each system.
rng = np.random.default_rng(7)
token_count = 1200
shared_token_difficulty = rng.gamma(shape=1.3, scale=0.004, size=token_count)
average_prob_mismatch = {
    "Serial-Async": np.clip(shared_token_difficulty + rng.normal(0.026, 0.006, token_count), 0, None),
    "Unified-Engine": np.zeros(token_count),
    "Colocate-2Copies": np.clip(shared_token_difficulty + rng.normal(0.018, 0.005, token_count), 0, None),
    "Static-Disagg": np.clip(shared_token_difficulty + rng.normal(0.021, 0.006, token_count), 0, None),
    sysname: np.clip(shared_token_difficulty * 0.45 + rng.normal(0.004, 0.002, token_count), 0, None),
}
mean_mismatch = np.array([average_prob_mismatch[name].mean() for name in baselines])
p95_mismatch = np.array([np.percentile(average_prob_mismatch[name], 95) for name in baselines])
err = np.vstack([mean_mismatch - np.zeros_like(mean_mismatch), p95_mismatch - mean_mismatch])

fig, ax = plt.subplots(figsize=(3.15, 2.45))
x = np.arange(len(baselines))
ax.errorbar(
    x,
    mean_mismatch,
    yerr=err,
    fmt="none",
    ecolor=PALETTE["navy"],
    elinewidth=0.8,
    capsize=2.5,
    alpha=0.75,
    zorder=2,
)
ax.scatter(
    x,
    mean_mismatch,
    s=[68 if name == sysname else 44 for name in baselines],
    marker="o",
    color=BASELINE_COLORS,
    edgecolor=PALETTE["paper"],
    linewidth=0.8,
    zorder=3,
)
for idx, value in enumerate(mean_mismatch):
    ax.text(idx, value + 0.0035, f"{value:.3f}", ha="center", va="bottom", fontsize=9.0, color=PALETTE["navy"])
ax.set_xticks(x)
ax.set_xticklabels(short_names, rotation=24, ha="right")
ax.set_ylabel("Mean |log-prob mismatch|")
ax.set_ylim(0, max(p95_mismatch) * 1.2)
style_axis(ax)
fig.tight_layout()
save_figure(fig, "eval_fidelity_mean_logprob_mismatch")


#%% Freshness
# Unit: average policy update version observed by each tenant during training.
tenant_ids = np.arange(1, 13)
average_tenant_update_version = {
    "Serial-Async": np.array([2.1, 2.6, 3.2, 3.5, 3.8, 4.1, 4.4, 4.6, 4.7, 4.9, 5.0, 5.2]),
    "Unified-Engine": np.array([2.8, 3.0, 3.2, 3.4, 3.5, 3.7, 3.9, 4.1, 4.2, 4.3, 4.5, 4.6]),
    "Colocate-2Copies": np.array([2.4, 2.8, 3.3, 3.7, 4.0, 4.3, 4.5, 4.8, 5.0, 5.1, 5.4, 5.6]),
    "Static-Disagg": np.array([2.0, 2.7, 3.1, 3.8, 4.2, 4.6, 4.9, 5.3, 5.6, 5.8, 6.0, 6.3]),
    sysname: np.array([3.0, 3.2, 3.4, 3.6, 3.7, 3.9, 4.0, 4.1, 4.3, 4.4, 4.5, 4.6]),
}

fig, ax = plt.subplots(figsize=(4.8, 3.075))
for name in baselines:
    ax.plot(
        tenant_ids,
        average_tenant_update_version[name],
        marker=SYSTEM_MARKERS[name],
        markersize=6.2 if name == sysname else 3.7,
        color=SYSTEM_COLORS[name],
        linewidth=2.1 if name == sysname else 1.15,
        label=name,
    )
ax.set_xlabel("Tenant id")
ax.set_ylabel("Avg. parameter update version")
ax.set_xticks([1, 4, 8, 12])
ax.set_ylim(1.6, 6.7)
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.30), fontsize=8.8)
style_axis(ax, grid_axis="both")
fig.tight_layout()
save_figure(fig, "eval_freshness")


#%% Training performance, sampling performance and switching overhead
# Unit: milliseconds per RL iteration segment in a single 8-GPU node.
backend_components = ["train compute", "sample decode", "queue wait", "switch"]
backend_breakdown_ms = {
    "Unified-Engine": [920, 2310, 180, 0],
    "Colocate-2Copies": [780, 980, 430, 140],
    "Static-Disagg": [810, 760, 620, 0],
    sysname: [735, 690, 180, 42],
}
component_colors = [PALETTE["blue"], PALETTE["cyan"], PALETTE["sand"], PALETTE["navy"]]

fig, ax = plt.subplots(figsize=(5.2, 2.55))
y = np.arange(len(backend_breakdown_ms))
left = np.zeros(len(backend_breakdown_ms))
for comp_idx, component in enumerate(backend_components):
    values = np.array([backend_breakdown_ms[name][comp_idx] for name in backend_breakdown_ms])
    ax.barh(
        y,
        values,
        left=left,
        height=0.52,
        color=component_colors[comp_idx],
        edgecolor=PALETTE["paper"],
        linewidth=0.8,
        label=component,
    )
    left += values
for idx, total in enumerate(left):
    ax.text(total + 55, idx, f"{total/1000:.2f}s", va="center", fontsize=10, color=PALETTE["navy"])
ax.set_yticks(y)
ax.set_yticklabels(list(backend_breakdown_ms.keys()))
ax.invert_yaxis()
ax.set_xlabel("Per-iteration segment time (ms)")
ax.set_xlim(0, max(left) * 1.22)
ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.02))
style_axis(ax, grid_axis="x")
fig.tight_layout()
save_figure(fig, "eval_backend_breakdown")


#%% Corrector performance
# Unit: average per-generation sequence-bias reduction relative to uncorrected
# two-engine log-probabilities. Higher is better.
generations = np.array([1, 2, 3, 4, 5, 6])
average_sequence_bias_reduction = {
    "Token correction": np.array([18.0, 23.5, 28.0, 31.2, 33.4, 35.1]),
    "Sequence correction": np.array([26.0, 34.0, 41.5, 47.0, 50.8, 53.2]),
    "Adaptive correction": np.array([31.0, 42.5, 51.0, 58.0, 63.5, 67.0]),
}

fig, ax = plt.subplots(figsize=(4.8, 3.225))
line_colors = [PALETTE["steel"], PALETTE["blue"], PALETTE["tuft"]]
for idx, (name, reduction) in enumerate(average_sequence_bias_reduction.items()):
    ax.plot(
        generations,
        reduction,
        marker=["o", "s", "*"][idx],
        markersize=7.8 if name == "Adaptive correction" else 5.2,
        color=line_colors[idx],
        linewidth=2.4 if name == "Adaptive correction" else 1.7,
        label=name,
    )
ax.set_xlabel("Generation")
ax.set_ylabel("Sequence bias reduction (%)")
ax.set_ylim(0, 75)
ax.set_xticks(generations)
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.32), fontsize=9.0)
style_axis(ax, grid_axis="both")
fig.tight_layout()
save_figure(fig, "eval_corrector_performance")


#%% GPU utilization of baselines and TuFT
# Future data format: a CSV with columns ``time_s``, ``system`` and ``gpu_util``.
# If the file does not exist, this cell generates a mock trace with the same shape.
def load_gpu_utilization_or_mock(csv_path: Path | None = None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if csv_path is not None and csv_path.exists():
        if pd is None:
            raise RuntimeError("pandas is required to read GPU utilization CSV files")
        df = pd.read_csv(csv_path)
        required = {"time_s", "system", "gpu_util"}
        if not required.issubset(df.columns):
            raise ValueError(f"GPU CSV must contain columns: {sorted(required)}")
        traces = {}
        for system, group in df.groupby("system"):
            group = group.sort_values("time_s")
            traces[str(system)] = (group["time_s"].to_numpy(), group["gpu_util"].to_numpy())
        return traces

    time_s = np.arange(0, 900, 5)
    rng = np.random.default_rng(17)
    traces = {}
    patterns = {
        "Serial-Async": (52, 18, 180),
        "Unified-Engine": (49, 10, 120),
        "Colocate-2Copies": (65, 14, 150),
        "Static-Disagg": (61, 12, 165),
        sysname: (83, 7, 210),
    }
    for name, (mean, amp, period) in patterns.items():
        phase = 0.7 if name == sysname else 0.0
        util = mean + amp * np.sin(2 * np.pi * time_s / period + phase) + rng.normal(0, 3.0, len(time_s))
        if name in {"Serial-Async", "Static-Disagg"}:
            # Periodic valleys represent idle windows during static ownership changes.
            util -= 14 * ((time_s // 150) % 2 == 1)
        traces[name] = (time_s, np.clip(util, 8, 96))
    return traces


gpu_traces = load_gpu_utilization_or_mock()

fig, ax = plt.subplots(figsize=(4.8, 3.15))
for idx, name in enumerate(baselines):
    time_s, util = gpu_traces[name]
    ax.plot(
        time_s / 60.0,
        util,
        color=SYSTEM_COLORS[name],
        linewidth=2.2 if name == sysname else 1.0,
        alpha=0.95 if name == sysname else 0.78,
        label=name,
    )
ax.set_xlabel("Time (min)")
ax.set_ylabel("GPU utilization (%)")
ax.set_ylim(0, 100)
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.32), fontsize=8.8)
style_axis(ax, grid_axis="both")
fig.tight_layout()
save_figure(fig, "eval_gpu_utilization")


#%% Ablation Study, 8 GPUs, compare throughput and training quality.
# Throughput is completed RL training steps/hour; quality is final eval reward.
quality_throughput_points = {
    "Serial-Async": (0.58, 31.5),
    "Unified-Engine": (0.66, 23.8),
    "Colocate-2Copies": (0.57, 39.2),
    "Static-Disagg": (0.55, 36.1),
    "TuFT base": (0.49, 23.8),
    "+ torch-tp": (0.51, 31.6),
    "+ zero-copy": (0.54, 39.4),
    "+ request sched.": (0.58, 46.8),
    "+ loop sched.": (0.62, 52.1),
    sysname: (0.65, 57.8),
}
tuFT_path = ["TuFT base", "+ torch-tp", "+ zero-copy", "+ request sched.", "+ loop sched.", sysname]

fig, ax = plt.subplots(figsize=(4.8, 2.2))
for name in baselines[:-1]:
    quality, throughput = quality_throughput_points[name]
    ax.scatter(
        quality,
        throughput,
        s=70,
        color=SYSTEM_COLORS[name],
        marker=SYSTEM_MARKERS[name],
        edgecolor=PALETTE["paper"],
        linewidth=0.8,
        label=name,
        zorder=3,
    )
path_quality = np.array([quality_throughput_points[name][0] for name in tuFT_path])
path_throughput = np.array([quality_throughput_points[name][1] for name in tuFT_path])
ax.scatter(
    path_quality,
    path_throughput,
    s=[48, 48, 48, 48, 48, 105],
    color=PALETTE["tuft"],
    marker="*",
    edgecolor=PALETTE["paper"],
    linewidth=0.8,
    label="TuFT variants",
    zorder=4,
)
for name in baselines[:-1] + [sysname]:
    quality, throughput = quality_throughput_points[name]
    ax.text(quality + 0.004, throughput + 0.5, short_names[baselines.index(name)] if name in baselines else name, fontsize=8.8, color=PALETTE["navy"])
for label in ["base", "+tp", "+zc", "+sched"]:
    idx = ["base", "+tp", "+zc", "+sched"].index(label)
    ax.text(path_quality[idx] - 0.018, path_throughput[idx] - 2.0, label, fontsize=8.2, color=PALETTE["tuft"])
ax.set_xlabel("Training quality (avg. reward)")
ax.set_ylabel("Throughput (steps / hour)")
ax.set_xlim(0.47, 0.69)
ax.set_ylim(20, 62)
ax.legend(frameon=False, loc="lower right", fontsize=8.6)
style_axis(ax, grid_axis=None)
fig.tight_layout()
save_figure(fig, "eval_ablation_quality_throughput")

# %%
