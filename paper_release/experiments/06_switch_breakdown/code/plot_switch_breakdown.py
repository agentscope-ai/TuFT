"""Generate switch breakdown characterization figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

# Unit: milliseconds.  Each method has two directions; each direction is shown
# as a stacked breakdown bar.
methods = ["Process", "Framework", "Runtime"]
components = ["Teardown/offload", "Runtime restore", "Metadata remap"]
colors = ["#DBE2EF", "#3F72AF", "#112D4E"]

breakdown_ms = {
    "Process": {
        "T→S": [3162.0, 216587.9, 0.0],
        "S→T": [2096.1, 78200.7, 0.0],
    },
    "Framework": {
        "T→S": [541.0, 5550.7, 0.0],
        "S→T": [5215.1, 109.9, 0.0],
    },
    "Runtime": {
        "T→S": [308.1, 3.9, 58.3],
        "S→T": [0.0, 201.0, 4.5],
    },
}

fig, ax = plt.subplots(figsize=(6.9, 2.75))

bar_h = 0.28
group_gap = 0.92
y_positions: list[float] = []
y_labels: list[str] = []
method_centers: list[float] = []

for method_idx, method in enumerate(methods):
    base_y = method_idx * group_gap
    method_centers.append(base_y + bar_h / 2)
    for direction_idx, direction in enumerate(["T→S", "S→T"]):
        y = base_y + (0 if direction_idx == 0 else bar_h + 0.08)
        y_positions.append(y)
        y_labels.append(direction)

        left = 0.0
        total = sum(breakdown_ms[method][direction])
        for comp_idx, value in enumerate(breakdown_ms[method][direction]):
            if value <= 0:
                continue
            ax.barh(
                y,
                value / 1000.0,
                left=left / 1000.0,
                height=bar_h,
                color=colors[comp_idx],
                edgecolor="#F9F7F7",
                linewidth=0.7,
            )
            left += value

        label = f"{total / 1000.0:.2f}s" if total >= 1000 else f"{total:.0f}ms"
        ax.text(
            total / 1000.0 * 1.12,
            y,
            label,
            va="center",
            ha="left",
            fontsize=8.2,
            color="#112D4E",
        )

# Direction labels on y-axis.
ax.set_yticks(y_positions)
ax.set_yticklabels(y_labels, fontsize=8.5)
ax.invert_yaxis()

# Method labels as group headers on the left side.
for center, method in zip(method_centers, methods):
    ax.text(
        -0.105,
        center,
        method,
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="center",
        fontsize=8.7,
        fontweight="bold" if method == "Runtime" else "normal",
    )

# Group separators.
for method_idx in range(1, len(methods)):
    sep_y = method_idx * group_gap - 0.23
    ax.axhline(sep_y, color="#DDDDDD", linewidth=0.7, zorder=0)

ax.set_xscale("log")
ax.set_xlim(0.12, 500)
ax.set_xlabel("Switch latency breakdown (s, log scale)", fontsize=9)
ax.grid(axis="x", which="major", linestyle="--", linewidth=0.5, alpha=0.45)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#112D4E")
ax.spines["bottom"].set_color("#112D4E")
ax.tick_params(axis="x", colors="#112D4E")
ax.tick_params(axis="y", colors="#112D4E")
ax.tick_params(axis="y", pad=8)

legend_handles = [Patch(facecolor=c, edgecolor="#F9F7F7", label=label) for c, label in zip(colors, components)]
ax.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.01),
    ncol=3,
    frameon=False,
    fontsize=8,
    columnspacing=1.2,
    handlelength=1.1,
    handletextpad=0.4,
)

fig.savefig(OUT_DIR / "switch_breakdown_grouped.pdf", bbox_inches="tight")
fig.savefig(OUT_DIR / "switch_breakdown_grouped.png", dpi=300, bbox_inches="tight")
print(OUT_DIR / "switch_breakdown_grouped.pdf")
