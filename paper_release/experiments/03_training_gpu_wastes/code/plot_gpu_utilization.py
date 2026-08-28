"""
Plot GPU utilization over time for async and sync training experiments.
Generates publication-quality figures for the paper.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# ============================================================
# Scientific figure style configuration
# ============================================================
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "lines.linewidth": 1.2,
    "axes.grid": True,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

# Color palette (user-specified)
COLORS = {
    "training": "#F4D35E",  # warm yellow
    "sampling": "#457B9D",  # steel blue
}

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_async_data(filepath: Path) -> pd.DataFrame:
    """Load async GPU stats CSV (comma-separated, full datetime timestamps)."""
    df = pd.read_csv(filepath, skipinitialspace=True, dtype=str)
    df.columns = [c.strip() for c in df.columns]

    df["timestamp"] = pd.to_datetime(df["timestamp"].str.strip())
    df["index"] = df["index"].astype(int)

    util_col = [c for c in df.columns if "utilization.gpu" in c][0]
    df["gpu_util"] = df[util_col].str.replace("%", "").str.strip().astype(float)

    t0 = df["timestamp"].min()
    df["elapsed_min"] = (df["timestamp"] - t0).dt.total_seconds() / 60.0

    return df[["elapsed_min", "index", "gpu_util"]].copy()


def load_sync_data(filepath: Path) -> pd.DataFrame:
    """Load sync GPU stats CSV (tab-separated, MM:SS.s timestamps)."""
    df = pd.read_csv(filepath, sep="\t", skipinitialspace=True, dtype=str)
    df.columns = [c.strip() for c in df.columns]

    # Parse MM:SS.s timestamp -> minutes
    def parse_mmss(ts: str) -> float:
        ts = ts.strip()
        parts = ts.split(":")
        return int(parts[0]) + float(parts[1]) / 60.0

    df["raw_min"] = df["timestamp"].apply(parse_mmss)

    # Handle timestamp wrap-around (59:59 -> 00:00)
    # Track cumulative offset per GPU
    df["elapsed_min"] = np.nan
    for gpu_id in df["index"].unique():
        mask = df["index"] == gpu_id
        raw = df.loc[mask, "raw_min"].values
        offsets = np.zeros(len(raw))
        cumulative = 0.0
        for i in range(1, len(raw)):
            if raw[i] < raw[i - 1] - 30:  # significant drop = wrap
                cumulative += 60.0
            offsets[i] = cumulative
        df.loc[mask, "elapsed_min"] = raw + offsets

    # Shift to start from 0
    df["elapsed_min"] -= df["elapsed_min"].min()

    df["index"] = df["index"].astype(int)

    util_col = [c for c in df.columns if "utilization.gpu" in c][0]
    df["gpu_util"] = df[util_col].str.replace("%", "").str.strip().astype(float)

    return df[["elapsed_min", "index", "gpu_util"]].copy()


def smooth(y: np.ndarray, window: int = 60) -> np.ndarray:
    """Moving average smoothing."""
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def main():
    # ----------------------------------------------------------
    # Load data
    # ----------------------------------------------------------
    async_df = load_async_data(DATA_DIR / "gpu_stats_async.csv")
    sync_df = load_sync_data(DATA_DIR / "gpu_stats_sync.csv")

    # ----------------------------------------------------------
    # Print average utilization for 3 active GPUs
    # ----------------------------------------------------------
    print("=" * 60)
    print("Average GPU Utilization (Async Training)")
    print("=" * 60)
    for gpu_id in [0, 1, 2]:
        gpu_data = async_df[async_df["index"] == gpu_id]["gpu_util"]
        print(f"  GPU {gpu_id}: {gpu_data.mean():.2f}% ± {gpu_data.std():.2f}%")

    print()
    print("=" * 60)
    print("Average GPU Utilization (Sync Training)")
    print("=" * 60)
    for gpu_id in [0, 1, 2]:
        gpu_data = sync_df[sync_df["index"] == gpu_id]["gpu_util"]
        print(f"  GPU {gpu_id}: {gpu_data.mean():.2f}% ± {gpu_data.std():.2f}%")
    print("=" * 60)

    # ----------------------------------------------------------
    # Figure: GPU utilization over time (2 panels, GPU 0 & 1)
    # ----------------------------------------------------------
    fig, axes = plt.subplots(
        2, 1,
        figsize=(7.0, 3.5),
        sharex=False,
        gridspec_kw={"hspace": 0.40},
    )
    fig.patch.set_facecolor("white")

    # --- Panel (a): Async ---
    ax = axes[0]

    for gpu_id, color, label in [
        (0, COLORS["training"], "Training"),
        (1, COLORS["sampling"], "Sampling"),
    ]:
        gpu_data = async_df[async_df["index"] == gpu_id].sort_values("elapsed_min")
        x = gpu_data["elapsed_min"].values
        y = gpu_data["gpu_util"].values
        ax.plot(x, smooth(y), color=color, label=label, alpha=0.95)

    ax.set_ylabel("GPU Util. (%)")
    ax.set_xlabel("Elapsed Time (min)")
    ax.set_ylim(-2, 105)
    ax.set_xlim(left=0)
    ax.legend(loc="lower right", framealpha=0.9, edgecolor="none")
    ax.text(-0.08, 1.05, "(a)", transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom")
    ax.set_title("Asynchronous Training", fontsize=10, loc="left", pad=8)

    # --- Panel (b): Sync ---
    ax = axes[1]

    for gpu_id, color, label in [
        (0, COLORS["training"], "Training"),
        (1, COLORS["sampling"], "Sampling"),
    ]:
        gpu_data = sync_df[sync_df["index"] == gpu_id].sort_values("elapsed_min")
        x = gpu_data["elapsed_min"].values
        y = gpu_data["gpu_util"].values
        ax.plot(x, smooth(y), color=color, label=label, alpha=0.95)

    ax.set_ylabel("GPU Util. (%)")
    ax.set_xlabel("Elapsed Time (min)")
    ax.set_ylim(-2, 105)
    ax.set_xlim(left=0)
    ax.legend(loc="lower right", framealpha=0.9, edgecolor="none")
    ax.text(-0.08, 1.05, "(b)", transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom")
    ax.set_title("Synchronous Training", fontsize=10, loc="left", pad=8)

    # Save
    out_path = OUT_DIR / "gpu_utilization_over_time.pdf"
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"))
    print(f"\nFigure saved to: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
