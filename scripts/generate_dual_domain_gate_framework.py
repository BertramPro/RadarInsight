"""Render the paper's trajectory-RD quality-aware gated-fusion framework."""

from pathlib import Path

import matplotlib
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "paper_figures_v1"
OUTPUT_PNG = OUTPUT / "trajectory_rd_gated_fusion_framework.png"
OUTPUT_PDF = OUTPUT / "trajectory_rd_gated_fusion_framework.pdf"


def chinese_font() -> font_manager.FontProperties:
    for candidate in (r"C:\Windows\Fonts\Deng.ttf", r"C:\Windows\Fonts\simhei.ttf"):
        if Path(candidate).is_file():
            return font_manager.FontProperties(fname=candidate)
    return font_manager.FontProperties(family="sans-serif")


FONT = chinese_font()
COLORS = {
    "tr": ("#EAF3FA", "#1774A6"),
    "rd": ("#FFF2E5", "#D46A00"),
    "gate": ("#E7F5F0", "#007F6E"),
    "output": ("#F2F4F6", "#58636E"),
    "neutral": ("#F7F8FA", "#68737D"),
}


def add_box(axis, x, y, width, height, text, family, *, fontsize=10.2, weight="normal"):
    fill, edge = COLORS[family]
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.025,rounding_size=0.08",
        linewidth=1.25,
        edgecolor=edge,
        facecolor=fill,
        zorder=2,
    )
    axis.add_patch(patch)
    axis.text(
        x + width / 2, y + height / 2, text,
        ha="center", va="center", fontsize=fontsize,
        fontproperties=FONT, fontweight=weight, color="#16222C", zorder=3,
        linespacing=1.35,
    )
    return patch


def add_arrow(axis, start, end, *, color="#596670", label=None, label_offset=(0, 0)):
    arrow = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=12,
        linewidth=1.25, color=color, connectionstyle="arc3,rad=0", zorder=1,
    )
    axis.add_patch(arrow)
    if label:
        axis.text(
            (start[0] + end[0]) / 2 + label_offset[0],
            (start[1] + end[1]) / 2 + label_offset[1],
            label, ha="center", va="center", fontsize=8.8,
            fontproperties=FONT, color="#4C5863", zorder=3,
        )


def draw_framework() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    figure, axis = plt.subplots(figsize=(14.5, 6.6), dpi=300)
    figure.patch.set_facecolor("white")
    axis.set_xlim(0, 15.0)
    axis.set_ylim(0, 7.1)
    axis.axis("off")

    axis.text(0.35, 6.82, "航迹运动信息", fontsize=11.5, fontproperties=FONT, fontweight="bold", color="#1774A6")
    axis.text(0.35, 3.08, "距离-速度结构信息", fontsize=11.5, fontproperties=FONT, fontweight="bold", color="#D46A00")

    add_box(axis, 0.35, 5.25, 1.78, 0.82, "[T,14]\n航迹序列", "tr", fontsize=10.8, weight="bold")
    add_box(axis, 0.35, 4.08, 1.78, 0.82, "17 维\n航迹统计特征", "tr", fontsize=10.3)
    add_box(axis, 0.35, 1.36, 1.78, 1.02, "统一速度坐标\nRD 图序列", "rd", fontsize=10.6, weight="bold")

    add_box(axis, 3.05, 4.55, 2.16, 1.12, "航迹时序表征\n与统计特征联合编码", "tr", fontsize=10.4)
    add_box(axis, 5.92, 4.55, 1.88, 1.12, "整条航迹\n类别概率", "tr", fontsize=10.8, weight="bold")

    add_box(axis, 3.05, 1.30, 2.16, 1.12, "RD 结构识别\n逐幅提取距离-速度响应", "rd", fontsize=10.2)
    add_box(axis, 5.92, 1.30, 1.88, 1.12, "单幅 RD 图\n类别概率", "rd", fontsize=10.8, weight="bold")
    add_box(axis, 5.92, 2.90, 1.88, 0.92, "同航迹概率聚合\n一致性与图数", "rd", fontsize=9.5)

    add_box(axis, 8.53, 2.23, 3.22, 2.55, "质量感知类别门控\n\nTR/RD 类别概率\n预测熵与类别间隔\nRD 观测间一致性与有效图数\n\n五类自适应权重  α_c", "gate", fontsize=10.1, weight="bold")
    axis.text(10.14, 1.70, r"$p_{F,c}=\alpha_c p_{RD,c}+(1-\alpha_c)p_{TR,c}$", ha="center", va="center", fontsize=10.5, color="#006B5D")

    add_box(axis, 12.46, 4.73, 2.12, 0.88, "可追溯判断记录\nTR、RD 与融合", "output", fontsize=10.3)
    add_box(axis, 12.46, 2.55, 2.12, 1.06, "门控融合判断\n与类别权重记录", "output", fontsize=10.2, weight="bold")
    add_box(axis, 12.46, 0.83, 2.12, 0.86, "五类低空目标\n识别结果", "output", fontsize=10.6, weight="bold")

    add_arrow(axis, (2.13, 5.66), (3.05, 5.15), color="#1774A6")
    add_arrow(axis, (2.13, 4.49), (3.05, 5.05), color="#1774A6")
    add_arrow(axis, (5.21, 5.11), (5.92, 5.11), color="#1774A6")
    add_arrow(axis, (2.13, 1.87), (3.05, 1.87), color="#D46A00")
    add_arrow(axis, (5.21, 1.86), (5.92, 1.86), color="#D46A00")
    add_arrow(axis, (6.86, 2.42), (6.86, 2.90), color="#D46A00", label="同一航迹", label_offset=(0.53, 0.0))
    add_arrow(axis, (7.80, 5.11), (8.53, 4.28), color="#1774A6", label="航迹证据", label_offset=(0.18, 0.28))
    add_arrow(axis, (7.80, 3.36), (8.53, 3.36), color="#D46A00", label="航迹证据", label_offset=(0.02, -0.30))
    add_arrow(axis, (7.80, 1.86), (8.53, 2.76), color="#D46A00", label="质量量", label_offset=(0.13, -0.25))
    add_arrow(axis, (11.75, 4.28), (12.46, 5.17), color="#1774A6")
    add_arrow(axis, (11.75, 3.36), (12.46, 3.08), color="#007F6E", label="融合概率", label_offset=(0.00, -0.26))
    add_arrow(axis, (13.52, 2.55), (13.52, 1.69), color="#596670")

    axis.text(7.80, 6.18, "统一航迹编号，以整条航迹作为判断单位", ha="center", va="center", fontsize=10.0, fontproperties=FONT, color="#374550", fontweight="bold")
    figure.subplots_adjust(left=0.015, right=0.995, top=0.97, bottom=0.05)
    figure.savefig(OUTPUT_PNG, dpi=300, facecolor="white")
    figure.savefig(OUTPUT_PDF, facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    draw_framework()
    print(OUTPUT_PNG)
    print(OUTPUT_PDF)
