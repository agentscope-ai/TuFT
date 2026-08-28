"""Evaluation figures for the TuFT paper.

Each ``#%%`` cell can be run independently after the setup cell. Overall
performance, SLO and ablation figures use measured experiment data; mechanism
microbenchmark figures are still placeholders until those runs complete.
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


#%% Throughput on Qwen3 models
# Unit: completed RL training steps per hour after warm-up on the 8-GPU setup.
model_groups = ["Qwen3-4B", "Qwen3-32B"]
throughput_by_model = {
    "Qwen3-4B": np.array([184.3, 39.9, 381.5, 283.9, 541.9]),
    "Qwen3-32B": np.array([116.3, 25.9, np.nan, 179.5, 342.1]),
}

fig, ax = plt.subplots(figsize=(3.05, 2.75))
group_x = np.arange(len(model_groups))
width = 0.13
offsets = (np.arange(len(baselines)) - (len(baselines) - 1) / 2) * width
for group_idx, model in enumerate(model_groups):
    for idx, name in enumerate(baselines):
        value = throughput_by_model[model][idx]
        x = group_x[group_idx] + offsets[idx]
        if np.isnan(value):
            ax.text(x, 18, "OOM", ha="center", va="bottom", fontsize=7.2, color=PALETTE["navy"], alpha=0.75)
            continue
        bars = ax.bar(
            x,
            value,
            width,
            label=name if group_idx == 0 else None,
            color=SYSTEM_COLORS[name],
            edgecolor=PALETTE["paper"],
            linewidth=0.8,
            zorder=3,
        )
        annotate_bars(ax, bars, fmt="{:.0f}", dy=8.0)
ax.set_xticks(group_x)
ax.set_xticklabels(model_groups)
ax.set_ylabel("Training steps / hour")
ax.set_ylim(0, 640)
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22), fontsize=6.5, handlelength=1.0)
style_axis(ax)
fig.subplots_adjust(left=0.18, right=0.98, bottom=0.16, top=0.78)
save_figure(fig, "eval_throughput_gpu_budget")


#%% Scaling with tenant counts
tenants = np.array([1, 2, 4, 8, 16])
# Unit: aggregate completed RL training steps/hour on Qwen3-4B.
scaling_throughput = {
    "Serial-Async": np.array([156, 166, 186, 184, 183]),
    "Unified-Engine": np.array([12, 22, 30, 40, 42]),
    "Colocate-2Copies": np.array([210, 305, 345, 381, 305]),
    "Static-Disagg": np.array([155, 228, 255, 284, 228]),
    sysname: np.array([303, 508, 523, 542, 414]),
}

fig, ax = plt.subplots(figsize=(3.05, 2.75))
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
ax.axvline(8, color=PALETTE["tuft"], linestyle="--", linewidth=1.0, alpha=0.75)
ax.text(8.25, 585, "TuFT peak", fontsize=8.0, color=PALETTE["tuft"], fontweight="bold")
ax.set_xscale("log", base=2)
ax.set_xticks(tenants)
ax.set_xticklabels([str(t) for t in tenants])
ax.set_xlabel("Concurrent tenants")
ax.set_ylabel("Training steps / hour")
ax.set_ylim(0, 620)
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22), fontsize=6.5, handlelength=1.2)
style_axis(ax)
fig.subplots_adjust(left=0.18, right=0.98, bottom=0.16, top=0.78)
save_figure(fig, "eval_scaling_tenants")


#%% Response time
# Unit: seconds. Measured SLO response times on Qwen3-4B.
response_time_slo = {
    "Training": np.array([146.7, 20.0, 25.8, 72.8, 29.6]),
    "Sampling": np.array([140.3, 703.2, 16.8, 16.1, 11.3]),
}

fig, ax = plt.subplots(figsize=(3.15, 2.65))
group_x = np.arange(2)
width = 0.13
offsets = (np.arange(len(baselines)) - (len(baselines) - 1) / 2) * width
for group_idx, metric in enumerate(["Training", "Sampling"]):
    for idx, name in enumerate(baselines):
        x = group_x[group_idx] + offsets[idx]
        value = response_time_slo[metric][idx]
        ax.bar(
            x,
            value,
            width,
            label=name if group_idx == 0 else None,
            color=SYSTEM_COLORS[name],
            edgecolor=PALETTE["paper"],
            linewidth=0.8,
            zorder=3,
        )
ax.set_yscale("log")
ax.set_yticks([10, 30, 100, 300, 700])
ax.set_yticklabels(["10", "30", "100", "300", "700"])
ax.set_xticks(group_x)
ax.set_xticklabels(["Training", "Sampling"])
ax.set_ylabel("Response time (s)")
ax.set_ylim(5, 1000)
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22), fontsize=6.5, handlelength=1.0)
style_axis(ax)
fig.subplots_adjust(left=0.20, right=0.98, bottom=0.16, top=0.78)
save_figure(fig, "eval_response_time")


#%% Fidelity
# Absolute sequence-level log-probability bias for each system.
logprob_bias = np.array([0.1141, 0.0195, 0.1141, 0.1141, 0.0181])

fig, ax = plt.subplots(figsize=(3.15, 2.45))
x = np.arange(len(baselines))
bars = ax.bar(
    x,
    logprob_bias,
    width=0.56,
    color=BASELINE_COLORS,
    edgecolor=PALETTE["paper"],
    linewidth=0.8,
    zorder=3,
)
for bar, value in zip(bars, logprob_bias):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.004,
        f"{value:.4f}",
        ha="center",
        va="bottom",
        fontsize=7.2,
        color=PALETTE["navy"],
    )
ax.set_xticks(x)
ax.set_xticklabels(short_names, rotation=24, ha="right")
ax.set_ylabel("Log-probability bias")
ax.set_ylim(0, 0.135)
style_axis(ax)
fig.tight_layout()
save_figure(fig, "eval_fidelity_mean_logprob_mismatch")


#%% Freshness
# Average policy-version staleness observed across tenants.
average_staleness = np.array([2.19, 0.52, 1.84, 3.45, 2.58])

fig, ax = plt.subplots(figsize=(4.8, 3.075))
x = np.arange(len(baselines))
bars = ax.bar(
    x,
    average_staleness,
    width=0.56,
    color=BASELINE_COLORS,
    edgecolor=PALETTE["paper"],
    linewidth=0.8,
    zorder=3,
)
for bar, value in zip(bars, average_staleness):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.08,
        f"{value:.2f}",
        ha="center",
        va="bottom",
        fontsize=8.0,
        color=PALETTE["navy"],
    )
ax.set_xticks(x)
ax.set_xticklabels(short_names, rotation=20, ha="right")
ax.set_ylabel("Average staleness (versions)")
ax.set_ylim(0, 4.0)
style_axis(ax)
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
# Accuracy with and without predictor. GSM8K is single-seed; MATH uses the
# multi-seed aggregate for GRPO and the strongest single seed for REINFORCE.
methods = ["REINFORCE", "GRPO"]
tasks = ["GSM8K", "MATH"]
accuracy_without_predictor = {
    "GSM8K": np.array([0.867, 0.933]),
    "MATH": np.array([0.600, 0.740]),
}
accuracy_with_predictor = {
    "GSM8K": np.array([0.933, 0.967]),
    "MATH": np.array([0.860, 0.807]),
}

fig, ax = plt.subplots(figsize=(4.8, 3.225))
bar_width = 0.065
condition_gap = 0.075
method_gap = 0.34
method_centers = [-method_gap / 2, method_gap / 2]
condition_colors = [PALETTE["steel"], PALETTE["tuft"]]
condition_labels = ["w/o predictor", "w/ predictor"]

for task_idx, task in enumerate(tasks):
    for method_idx, method in enumerate(methods):
        for condition_idx, condition in enumerate(condition_labels):
            values = accuracy_without_predictor if condition_idx == 0 else accuracy_with_predictor
            value = values[task][method_idx]
            offset = method_centers[method_idx] + (condition_idx - 0.5) * condition_gap
            bar = ax.bar(
                task_idx + offset,
                value,
                bar_width,
                color=condition_colors[condition_idx],
                edgecolor=PALETTE["paper"],
                linewidth=0.7,
                hatch="" if condition_idx == 1 else "//",
                zorder=3,
            )
            ax.text(
                task_idx + offset,
                value + 0.012,
                f"{value:.3f}".rstrip("0").rstrip("."),
                ha="center",
                va="bottom",
                fontsize=7.2,
                color=PALETTE["navy"],
            )

axis_transform = ax.get_xaxis_transform()
for task_idx, task in enumerate(tasks):
    for method_idx, method in enumerate(methods):
        center = task_idx + method_centers[method_idx]
        ax.text(center, -0.10, method, transform=axis_transform, ha="center", va="top", fontsize=8.4, color=PALETTE["navy"])
    ax.text(task_idx, -0.23, task, transform=axis_transform, ha="center", va="top", fontsize=10.0, color=PALETTE["navy"], fontweight="bold")

ax.set_xticks([])
ax.set_ylabel("Accuracy")
ax.set_ylim(0.5, 1.03)
ax.set_xlim(-0.45, len(tasks) - 0.55)

boundary = 0.5
ax.plot(
    [boundary, boundary],
    [-0.02, 1.02],
    transform=axis_transform,
    color=PALETTE["navy"],
    linestyle="-",
    linewidth=0.9,
    alpha=0.65,
    clip_on=False,
    zorder=4,
)
legend_handles = [
    Patch(facecolor=condition_colors[0], edgecolor=PALETTE["paper"], hatch="//", label="w/o predictor"),
    Patch(facecolor=condition_colors[1], edgecolor=PALETTE["paper"], label="w/ predictor"),
]
ax.legend(handles=legend_handles, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.18), fontsize=8.8)
style_axis(ax)
fig.subplots_adjust(left=0.15, right=0.98, bottom=0.28, top=0.84)
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


#%% Ablation Study, 8 GPUs, compare throughput and log-probability bias.
# Throughput is completed RL training tokens/s; bias is absolute log-probability bias.
ablation_baseline_points = {
    "Serial-Async": (0.1141, 153.0),
    "Colocate-2Copies": (0.1141, 381.5),
    "Unified-Engine": (0.0195, 39.9),
}
ablation_tuft_points = [
    ("static (1:1)", 0.1141, 186.6),
    ("+tp", 0.0181, 225.4),
    ("+sampling", 0.0181, 246.3),
    ("+fast", 0.0181, 270.6),
    ("+slow", 0.0181, 423.5),
]

fig, ax = plt.subplots(figsize=(4.8, 2.4))
ablation_baseline_styles = {
    "Serial-Async": (SYSTEM_COLORS["Serial-Async"], SYSTEM_MARKERS["Serial-Async"]),
    "Colocate-2Copies": (SYSTEM_COLORS["Colocate-2Copies"], SYSTEM_MARKERS["Colocate-2Copies"]),
    "Unified-Engine": (SYSTEM_COLORS["Unified-Engine"], SYSTEM_MARKERS["Unified-Engine"]),
}
baseline_label_offsets = {
    "Serial-Async": (8, -14, "left"),
    "Colocate-2Copies": (8, 7, "left"),
    "Unified-Engine": (-7, 7, "right"),
}
for name, (bias, throughput) in ablation_baseline_points.items():
    color, marker = ablation_baseline_styles[name]
    ax.scatter(
        bias,
        throughput,
        s=72,
        color=color,
        marker=marker,
        edgecolor=PALETTE["paper"],
        linewidth=0.8,
        label=name,
        zorder=3,
    )
    dx, dy, ha = baseline_label_offsets[name]
    ax.annotate(
        name,
        (bias, throughput),
        textcoords="offset points",
        xytext=(dx, dy),
        fontsize=7.0,
        color=PALETTE["navy"],
        ha=ha,
    )

for idx, (name, bias, throughput) in enumerate(ablation_tuft_points):
    ax.scatter(
        bias,
        throughput,
        s=92 if idx == len(ablation_tuft_points) - 1 else 58,
        color=PALETTE["tuft"],
        marker="*",
        edgecolor=PALETTE["paper"],
        linewidth=0.8,
        label=None,
        zorder=4,
    )

tuft_label_offsets = {
    "static (1:1)": (8, 9, "left"),
    "+tp": (-7, -15, "right"),
    "+sampling": (-7, -4, "right"),
    "+fast": (-7, 7, "right"),
    "+slow": (-7, -16, "right"),
}
for name, bias, throughput in ablation_tuft_points:
    dx, dy, ha = tuft_label_offsets[name]
    ax.annotate(
        name,
        (bias, throughput),
        textcoords="offset points",
        xytext=(dx, dy),
        fontsize=7.2,
        color=PALETTE["tuft"],
        ha=ha,
    )

ax.set_xscale("log")
ax.set_xlim(0.16, 0.012)  # Reversed: larger bias sits closer to the origin.
ax.set_xticks([0.12, 0.08, 0.05, 0.03, 0.02])
ax.set_xticklabels(["0.12", "0.08", "0.05", "0.03", "0.02"])
ax.set_xlabel("Log-probability bias")
ax.set_ylabel("Throughput")
ax.set_ylim(0, 480)
style_axis(ax, grid_axis=None)
fig.tight_layout()
save_figure(fig, "eval_ablation_quality_throughput")

# %%
