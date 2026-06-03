import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# 自定义工具箱
import ToolBoxes.Tool_PlotConfig as TPC
FIGFMT = TPC.FIGFMT
DPI    = TPC.DPI_medium
BASEDATA = TPC.BASEDATA

def plot_circular_ring(
    df: pd.DataFrame,
    label_col: str,
    percent_col: str,
    color_map: dict = None,      # {label: color}
    title: str = None,
    savepath: str = None,
    colors: list = None,         # 若不传 color_map，可传一个颜色列表，按行顺序使用
    startangle: float = 90,
    ring_width: float = 0.32,    # 圆环厚度，越大越厚
    show_pct_on_wedge: bool = False,
    pct_format: str = "{:.1f}%", # 扇区上显示的百分比格式
    legend: bool = False,
    legend_loc: str = "center left",
    legend_bbox: tuple = (1.02, 0.5),
    normalize_if_not_100: bool = True,  # 若总和不是100，是否自动归一化
):
    """
    用 df 画圆环百分比图（所有类别加起来=100%）
    df[label_col]  : 类别名
    df[percent_col]: 占比（建议已是百分数 0~100；也支持传比例 0~1，但你要自己保证一致）
    """
    if label_col not in df.columns or percent_col not in df.columns:
        raise ValueError(f"df 必须包含列: {label_col}, {percent_col}")

    d = df[[label_col, percent_col, 'Area']].copy()
    d = d.dropna(subset=[label_col, percent_col, 'Area'])
    d[percent_col] = pd.to_numeric(d[percent_col], errors="coerce")
    d = d.dropna(subset=[percent_col])

    labels = d[label_col].astype(str).tolist()
    values = d[percent_col].to_numpy(dtype=float)

    total = float(np.nansum(values))
    if total <= 0:
        raise ValueError("percent 总和必须 > 0")

    # 检查是否≈100
    if not np.isclose(total, 100.0, atol=1e-6):
        if normalize_if_not_100:
            values = values / total * 100.0
            total = 100.0
        else:
            raise ValueError(f"percent 总和应为 100，但当前为 {total}")

    # --- 颜色分配 ---
    if color_map is not None:
        # 确保每个 label 都能找到颜色
        missing = [lab for lab in labels if lab not in color_map]
        if missing:
            raise ValueError(f"color_map 缺少这些 label 的颜色: {missing}")
        wedge_colors = [color_map[lab] for lab in labels]
    elif colors is not None:
        if len(colors) < len(labels):
            raise ValueError("colors 列表长度不足以覆盖所有类别")
        wedge_colors = colors[:len(labels)]
    else:
        # 不指定的话：用 matplotlib 默认色循环
        wedge_colors = [None] * len(labels)

    fig, ax = plt.subplots(figsize=(5, 5))

    # 扇区百分比文字
    def _autopct(pct):
        # pct 是 matplotlib 按总和=100 计算的百分比
        return pct_format.format(pct) if show_pct_on_wedge else ""

    wedges, texts, autotexts = ax.pie(
        values,
        labels=None,  # 通常 donut 不把 label 直接贴在扇区上（容易挤），建议用 legend
        startangle=startangle,
        colors=wedge_colors,
        autopct=_autopct,
        pctdistance=0.85,
        wedgeprops=dict(width=ring_width, edgecolor="white"),
    )

    # 中间挖空后的“中心文字”（可按需改成别的，比如总和、样本数等）
    label = "Up"
    color = color_map.get(label, "black") if color_map else "black"
    value = d.loc[d[label_col] == label, percent_col].values[0]
    value_str = f"Increase: {value:.2f}%"
    area = d.loc[d[label_col] == label, 'Area'].values[0]
    area_str = f"Area: {area:.2f} x10⁴ km²"
    ax.text(0, 0.3, value_str, ha="center", va="center", fontsize=18, fontweight="bold", color=color)
    ax.text(0, 0.15, area_str, ha="center", va="center", fontsize=14, fontweight="bold", color=color)
    label = "Down"
    color = color_map.get(label, "black") if color_map else "black"
    value = d.loc[d[label_col] == label, percent_col].values[0]
    area = d.loc[d[label_col] == label, 'Area'].values[0]
    value_str = f"Decrease: {value:.2f}%"
    area_str = f"Area: {area:.2f} x10⁴ km²"
    ax.text(0, -0.15, value_str, ha="center", va="center", fontsize=18, fontweight="bold", color=color)
    ax.text(0, -0.3, area_str, ha="center", va="center", fontsize=14, fontweight="bold", color=color)

    # 1) 中间分割线（建议略上移一点）
    inner_r = 1 - ring_width          # donut 中间空心半径
    L = inner_r * 0.85                # 线的半长度（0.6~0.9都好看）
    ax.plot(
        [-L, L], [0, 0],              # 一条水平线
        linewidth=1.4,
        alpha=0.7,
        color="0.4",                  # 灰色（不抢视觉）
        solid_capstyle="round",       # 圆角端点更精致
        zorder=10
    )

    if title:
        print(title)
        ax.text(0, 1.13, title, ha="center", va="center", fontsize=22, fontweight="bold", color="k")

    ax.set(aspect="equal")

    # 图例（label + 数值）
    if legend:
        handles = [
            Patch(facecolor=wedges[i].get_facecolor(), edgecolor="none",
                  label=f"{labels[i]}  ({values[i]:.1f}%)")
            for i in range(len(labels))
        ]
        ax.legend(handles=handles, loc=legend_loc, bbox_to_anchor=legend_bbox, frameon=False)

    plt.tight_layout()

    if savepath:
        fig.savefig(savepath, bbox_inches="tight", format=FIGFMT, dpi=DPI)
    return fig, ax



def plot_circular_ring_liang(
    df: pd.DataFrame,
    label_col: str,
    percent_col: str,
    color_map: dict = None,      # {label: color}
    title: str = None,
    savepath: str = None,
    colors: list = None,         # 若不传 color_map，可传一个颜色列表，按行顺序使用
    startangle: float = 90,
    ring_width: float = 0.32,    # 圆环厚度，越大越厚
    show_pct_on_wedge: bool = False,
    pct_format: str = "{:.1f}%", # 扇区上显示的百分比格式
    legend: bool = False,
    legend_loc: str = "center left",
    legend_bbox: tuple = (1.02, 0.5),
    normalize_if_not_100: bool = True,  # 若总和不是100，是否自动归一化
):
    """
    用 df 画圆环百分比图（所有类别加起来=100%）
    df[label_col]  : 类别名
    df[percent_col]: 占比（建议已是百分数 0~100；也支持传比例 0~1，但你要自己保证一致）
    """
    if label_col not in df.columns or percent_col not in df.columns:
        raise ValueError(f"df 必须包含列: {label_col}, {percent_col}")

    d = df[[label_col, percent_col, 'Area']].copy()
    d = d.dropna(subset=[label_col, percent_col, 'Area'])
    d[percent_col] = pd.to_numeric(d[percent_col], errors="coerce")
    d = d.dropna(subset=[percent_col])

    labels = d[label_col].astype(str).tolist()
    values = d[percent_col].to_numpy(dtype=float)

    total = float(np.nansum(values))
    if total <= 0:
        raise ValueError("percent 总和必须 > 0")

    # 检查是否≈100
    if not np.isclose(total, 100.0, atol=1e-6):
        if normalize_if_not_100:
            values = values / total * 100.0
            total = 100.0
        else:
            raise ValueError(f"percent 总和应为 100，但当前为 {total}")

    # --- 颜色分配 ---
    if color_map is not None:
        # 确保每个 label 都能找到颜色
        missing = [lab for lab in labels if lab not in color_map]
        if missing:
            raise ValueError(f"color_map 缺少这些 label 的颜色: {missing}")
        wedge_colors = [color_map[lab] for lab in labels]
    elif colors is not None:
        if len(colors) < len(labels):
            raise ValueError("colors 列表长度不足以覆盖所有类别")
        wedge_colors = colors[:len(labels)]
    else:
        # 不指定的话：用 matplotlib 默认色循环
        wedge_colors = [None] * len(labels)

    fig, ax = plt.subplots(figsize=(5, 5))

    # 扇区百分比文字
    def _autopct(pct):
        # pct 是 matplotlib 按总和=100 计算的百分比
        return pct_format.format(pct) if show_pct_on_wedge else ""

    wedges, texts, autotexts = ax.pie(
        values,
        labels=None,  # 通常 donut 不把 label 直接贴在扇区上（容易挤），建议用 legend
        startangle=startangle,
        colors=wedge_colors,
        autopct=_autopct,
        pctdistance=0.85,
        wedgeprops=dict(width=ring_width, edgecolor="white"),
    )

    # # 中间挖空后的“中心文字”（可按需改成别的，比如总和、样本数等）
    # label = "Up"
    # color = color_map.get(label, "black") if color_map else "black"
    # value = d.loc[d[label_col] == label, percent_col].values[0]
    # value_str = f"Increase: {value:.2f}%"
    # area = d.loc[d[label_col] == label, 'Area'].values[0]
    # area_str = f"Area: {area:.2f} x10⁴ km²"
    # # ax.text(0, 0.3, value_str, ha="center", va="center", fontsize=18, fontweight="bold", color=color)
    # # ax.text(0, 0.15, area_str, ha="center", va="center", fontsize=14, fontweight="bold", color=color)
    # label = "Down"
    # color = color_map.get(label, "black") if color_map else "black"
    # value = d.loc[d[label_col] == label, percent_col].values[0]
    # area = d.loc[d[label_col] == label, 'Area'].values[0]
    # value_str = f"Decrease: {value:.2f}%"
    # area_str = f"Area: {area:.2f} x10⁴ km²"
    # # ax.text(0, -0.15, value_str, ha="center", va="center", fontsize=18, fontweight="bold", color=color)
    # # ax.text(0, -0.3, area_str, ha="center", va="center", fontsize=14, fontweight="bold", color=color)

    # 1) 中间分割线（建议略上移一点）
    inner_r = 1 - ring_width          # donut 中间空心半径
    L = inner_r * 0.85                # 线的半长度（0.6~0.9都好看）
    ax.plot(
        [-L, L], [0, 0],              # 一条水平线
        linewidth=1.4,
        alpha=0.7,
        color="0.4",                  # 灰色（不抢视觉）
        solid_capstyle="round",       # 圆角端点更精致
        zorder=10
    )

    if title:
        print(title)
        ax.text(0, 1.13, title, ha="center", va="center", fontsize=22, fontweight="bold", color="k")

    ax.set(aspect="equal")

    # 图例（label + 数值）
    if legend:
        handles = [
            Patch(facecolor=wedges[i].get_facecolor(), edgecolor="none",
                  label=f"{labels[i]}  ({values[i]:.1f}%)")
            for i in range(len(labels))
        ]
        ax.legend(handles=handles, loc=legend_loc, bbox_to_anchor=legend_bbox, frameon=False)

    plt.tight_layout()

    if savepath:
        fig.savefig(savepath, bbox_inches="tight", format=FIGFMT, dpi=DPI)
    return fig, ax
