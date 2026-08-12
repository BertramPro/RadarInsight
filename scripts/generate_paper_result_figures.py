"""Generate publication figures for the recognition-result section."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "artifacts" / "fusion_202608122222" / "test_trajectory_metrics.json"
OUTPUT_DIR = ROOT / "artifacts" / "paper_figures_v1"

CLASS_LABELS = ["无人机", "飞鸟", "空飘球", "杂波", "未知目标"]


def configure_matplotlib() -> None:
    font_path = Path(r"C:\Windows\Fonts\simsun.ttc")
    if not font_path.exists():
        font_path = Path(r"C:\Windows\Fonts\simsunb.ttf")
    font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
    matplotlib.rcParams.update(
        {
            "font.family": font_name,
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def normalize_rows(matrix: list[list[int]]) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    totals = values.sum(axis=1, keepdims=True)
    return np.divide(values, totals, out=np.zeros_like(values), where=totals != 0)


def draw_confusion_matrices(metrics: dict) -> None:
    panels = [
        ("航迹分支", normalize_rows(metrics["tr_branch"]["confusion_matrix"])),
        ("RD分支", normalize_rows(metrics["rd_branch"]["confusion_matrix"])),
        ("质量感知门控", normalize_rows(metrics["soft_cascade"]["confusion_matrix"])),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.85), constrained_layout=True)
    image = None
    for index, (title, matrix) in enumerate(panels):
        ax = axes[index]
        image = ax.imshow(matrix, cmap="Greys", vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_title(f"（{chr(97 + index)}）{title}", fontsize=9.5, pad=6)
        ax.set_xticks(range(len(CLASS_LABELS)), labels=CLASS_LABELS)
        ax.set_yticks(range(len(CLASS_LABELS)), labels=CLASS_LABELS)
        ax.tick_params(axis="x", rotation=0, length=0, pad=5)
        ax.tick_params(axis="y", length=0, pad=4)
        ax.set_xlabel("预测类别", labelpad=6)
        if index == 0:
            ax.set_ylabel("真实类别", labelpad=7)

        ax.set_xticks(np.arange(-0.5, len(CLASS_LABELS), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(CLASS_LABELS), 1), minor=True)
        ax.grid(which="minor", color="#d0d0d0", linewidth=0.55)
        ax.tick_params(which="minor", bottom=False, left=False)

        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = matrix[row, col]
                color = "white" if value >= 0.55 else "black"
                ax.text(col, row, f"{value:.2f}", ha="center", va="center", color=color, fontsize=7.3)

        for spine in ax.spines.values():
            spine.set_linewidth(0.7)
            spine.set_color("#666666")

    colorbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.018, ticks=np.linspace(0, 1, 6))
    colorbar.ax.set_yticklabels([f"{value:.1f}" for value in np.linspace(0, 1, 6)])
    colorbar.outline.set_linewidth(0.7)
    colorbar.set_label("识别比例", rotation=270, labelpad=14)

    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT_DIR / f"recognition_confusion_matrices_grayscale.{suffix}", dpi=300)
    fig.savefig(
        OUTPUT_DIR / "recognition_confusion_matrices_grayscale.jpg",
        dpi=220,
        format="jpg",
        pil_kwargs={"quality": 94, "optimize": True},
    )
    plt.close(fig)


def main() -> None:
    configure_matplotlib()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    draw_confusion_matrices(metrics)


if __name__ == "__main__":
    main()
