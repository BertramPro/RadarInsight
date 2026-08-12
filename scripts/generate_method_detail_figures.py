"""Render the method-detail flowcharts used in Section 2 of the paper."""

from pathlib import Path

import matplotlib
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "paper_figures_v1"
FONT_PATHS = (r"C:\Windows\Fonts\Deng.ttf", r"C:\Windows\Fonts\simhei.ttf")


def chinese_font() -> font_manager.FontProperties:
    for candidate in FONT_PATHS:
        if Path(candidate).is_file():
            return font_manager.FontProperties(fname=candidate)
    return font_manager.FontProperties(family="sans-serif")


FONT = chinese_font()
COLORS = {
    "tr": ("#EAF3FA", "#1774A6"),
    "rd": ("#FFF2E5", "#D46A00"),
    "gate": ("#E7F5F0", "#007F6E"),
    "neutral": ("#F2F4F6", "#58636E"),
}


def box(axis, x, y, width, height, text, family, *, fontsize=11.0, weight="normal"):
    fill, edge = COLORS[family]
    axis.add_patch(FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.025,rounding_size=0.08",
        linewidth=1.25, edgecolor=edge, facecolor=fill, zorder=2,
    ))
    axis.text(
        x + width / 2, y + height / 2, text,
        ha="center", va="center", fontsize=fontsize, fontproperties=FONT,
        fontweight=weight, color="#16222C", linespacing=1.35, zorder=3,
    )


def arrow(axis, start, end, *, color="#596670", label=None, label_offset=(0.0, 0.0)):
    axis.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=12, linewidth=1.25,
        color=color, connectionstyle="arc3,rad=0", zorder=1,
    ))
    if label:
        axis.text(
            (start[0] + end[0]) / 2 + label_offset[0],
            (start[1] + end[1]) / 2 + label_offset[1], label,
            ha="center", va="center", fontsize=9.4, fontproperties=FONT,
            color="#4C5863", zorder=3,
        )


def setup(title: str, xlim, ylim, *, figsize=(13.8, 5.2)):
    figure, axis = plt.subplots(figsize=figsize, dpi=300)
    figure.patch.set_facecolor("white")
    axis.set_xlim(*xlim)
    axis.set_ylim(*ylim)
    axis.axis("off")
    axis.text(0.35, ylim[1] - 0.24, title, fontsize=13.5, fontproperties=FONT,
              fontweight="bold", color="#263744")
    return figure, axis


def draw_tr() -> None:
    figure, axis = setup("TR 航迹运动特征识别分支的处理流程", (0, 14.6), (0, 5.25))
    box(axis, 0.35, 2.95, 1.9, 1.0, "[T,14]\n逐点航迹序列", "tr", weight="bold")
    box(axis, 0.35, 1.10, 1.9, 1.0, "17 维\n航迹统计特征", "tr", weight="bold")
    box(axis, 3.05, 2.95, 2.25, 1.0, "有效点掩码约束\n进行时序编码", "tr")
    box(axis, 5.95, 2.95, 2.15, 1.0, "注意力聚合\n得到 z_seq", "tr")
    box(axis, 3.05, 1.10, 2.25, 1.0, "统计量归一化与\n物理特征编码", "tr")
    box(axis, 5.95, 1.10, 2.15, 1.0, "得到 z_phy", "tr")
    box(axis, 8.85, 2.00, 2.25, 1.0, "[z_seq; z_phy]\n联合表征", "neutral", weight="bold")
    box(axis, 11.75, 2.00, 2.35, 1.0, "整条航迹\n五类概率 p_TR", "tr", weight="bold")
    arrow(axis, (2.25, 3.45), (3.05, 3.45), color=COLORS["tr"][1])
    arrow(axis, (2.25, 1.60), (3.05, 1.60), color=COLORS["tr"][1])
    arrow(axis, (5.30, 3.45), (5.95, 3.45), color=COLORS["tr"][1])
    arrow(axis, (5.30, 1.60), (5.95, 1.60), color=COLORS["tr"][1])
    arrow(axis, (8.10, 3.45), (8.85, 2.72), color=COLORS["tr"][1], label="时序证据")
    arrow(axis, (8.10, 1.60), (8.85, 2.28), color=COLORS["tr"][1], label="统计证据")
    arrow(axis, (11.10, 2.50), (11.75, 2.50), color=COLORS["tr"][1])
    axis.text(7.20, 4.45, "同一航迹内保留局部变化与全程运动规律", ha="center", va="center",
              fontsize=10.2, fontproperties=FONT, color="#52616D")
    figure.subplots_adjust(left=0.015, right=0.995, top=0.93, bottom=0.08)
    figure.savefig(OUTPUT / "tr_branch_processing_flow.png", dpi=300, facecolor="white")
    figure.savefig(OUTPUT / "tr_branch_processing_flow.pdf", facecolor="white")
    plt.close(figure)


def draw_rd_gate() -> None:
    figure, axis = setup(
        "RD 卷积识别、结果汇集与质量感知门控流程", (0, 26.7), (0, 5.25), figsize=(20.2, 5.5)
    )
    box(axis, 0.30, 3.05, 1.85, 1.0, "统一 RD 图\n31×900", "rd", weight="bold")
    box(axis, 2.55, 3.05, 2.05, 1.0, "双通道构造\n主通道 + 局部速度对比", "rd")
    box(axis, 5.00, 3.05, 2.15, 1.0, "卷积模块 1\n2→32，3×3\n最大池化", "rd")
    box(axis, 7.55, 3.05, 2.15, 1.0, "卷积模块 2\n32→64，3×3\n最大池化", "rd")
    box(axis, 10.10, 3.05, 2.15, 1.0, "卷积模块 3\n64→128，3×3", "rd")
    box(axis, 12.65, 3.05, 2.15, 1.0, "自适应平均池化\n128 维表征", "rd")
    box(axis, 15.20, 3.05, 1.95, 1.0, "线性分类层\n五类概率 p_RD,c^(k)", "rd", weight="bold")
    box(axis, 17.60, 3.05, 2.10, 1.0, "同一目标平均汇集\n类别概率 p_RD,c", "rd")
    box(axis, 17.60, 1.08, 2.10, 1.0, "观测间一致性 ρ\n与有效图数 M", "rd")
    box(axis, 20.35, 4.15, 2.10, 0.70, "TR 五类概率 p_TR", "tr", weight="bold")
    box(axis, 20.35, 1.95, 2.35, 1.65, "门控输入\np_TR、p_RD、熵、间隔\nρ、M", "gate", weight="bold")
    box(axis, 23.25, 2.55, 1.15, 1.0, "门控\n权重", "gate", weight="bold")
    box(axis, 24.95, 2.55, 1.25, 1.0, "融合概率\n最终类别", "neutral", weight="bold")
    for start_x, end_x in ((2.15, 2.55), (4.60, 5.00), (7.15, 7.55), (9.70, 10.10),
                           (12.25, 12.65), (14.80, 15.20), (17.15, 17.60)):
        arrow(axis, (start_x, 3.55), (end_x, 3.55), color=COLORS["rd"][1])
    arrow(axis, (17.15, 3.35), (17.60, 1.78), color=COLORS["rd"][1], label="同一目标")
    arrow(axis, (19.70, 3.55), (20.35, 3.15), color=COLORS["rd"][1], label="RD 概率")
    arrow(axis, (19.70, 1.58), (20.35, 2.38), color=COLORS["rd"][1], label="质量量")
    arrow(axis, (21.40, 4.15), (21.40, 3.60), color=COLORS["tr"][1])
    arrow(axis, (22.70, 2.78), (23.25, 3.05), color=COLORS["gate"][1])
    arrow(axis, (24.40, 3.05), (24.95, 3.05), color=COLORS["gate"][1])
    axis.text(11.0, 1.0, "每幅 RD 图独立经过同一 CNN；同一目标的全部结果再用于形成分类概率和观测质量量",
              ha="center", va="center", fontsize=9.8, fontproperties=FONT, color="#52616D")
    axis.text(24.70, 1.25, r"$p_{F,c}=\alpha_c p_{RD,c}+(1-\alpha_c)p_{TR,c}$",
              ha="center", va="center", fontsize=10.2, color="#006B5D")
    figure.subplots_adjust(left=0.015, right=0.995, top=0.93, bottom=0.07)
    figure.savefig(OUTPUT / "rd_trajectory_gate_processing_flow.png", dpi=300, facecolor="white")
    figure.savefig(OUTPUT / "rd_trajectory_gate_processing_flow.pdf", facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    OUTPUT.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    draw_tr()
    draw_rd_gate()
    print(OUTPUT / "tr_branch_processing_flow.png")
    print(OUTPUT / "rd_trajectory_gate_processing_flow.png")
