import os
import time
import cmaps
import warnings
import numpy as np
import xarray as xr
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import font_manager as fm
from matplotlib.colorbar import ColorbarBase
from matplotlib.patches import Rectangle
# 自定义工具箱
import ToolBoxes.Utils as TU
import ToolBoxes.Tool_PlotConfig as TPC
import ToolBoxes.Tool_YangtzeColorMap as TYCM
FIGFMT = TPC.FIGFMT
DPI    = TPC.DPI_medium

mpl.use('Agg')  # 不显示图，只保存
# =============================================================================
# 【核心修改 1】全局字体配置：强制全粗体 (包括数学公式)
# =============================================================================
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Noto Sans', 'Arial']
# 开启自定义数学字体模式
mpl.rcParams['mathtext.fontset'] = 'custom'
# 关键：将数学公式的 常规(rm)、斜体(it)、粗体(bf) 全部映射到 "Noto Sans:bold"
# 这样即使公式里没写 \bf，渲染出来也是粗体
mpl.rcParams['mathtext.rm'] = 'Noto Sans:bold'
mpl.rcParams['mathtext.it'] = 'Noto Sans:bold:italic'
mpl.rcParams['mathtext.bf'] = 'Noto Sans:bold'
# 让数学公式默认使用我们定义的 'rm' (即上面的 Noto Sans:bold)
mpl.rcParams['mathtext.default'] = 'rm'
warnings.filterwarnings("ignore", category=RuntimeWarning)


def plot_spatial_cbar_core_V(mapcfg, Length, label, savepath, tickfz=None, labelfz=None, 
                             thickness_ratio=0.08, extend_type='neither', fmt=None):
    """
    纵向色条 (Vertical)
    :param Length: 色条的总高度 (英寸)
    :param thickness_ratio: 色条宽度相对于高度的比例
    :param extend_type: 扩展类型，可选 'both', 'max', 'min', 'neither'
    """
    diffcm         = mapcfg.cmap
    diff_maplevs   = mapcfg.levs[0]
    ndiff_maplevs  = mapcfg.levs[1]
    
    # 动态计算字体：随 Length 自动缩放
    LABEL_FZ = labelfz if labelfz is not None else int(Length * 2.2)
    TICK_FZ  = tickfz if tickfz is not None else int(Length * 1.8)
    TICK_FMT = fmt if fmt is not None else '{:.2f}'

    # 1. 计算画布尺寸
    fig_h_in = Length
    fig_w_in = Length * thickness_ratio * 4  # 宽度预留给刻度和标签
    
    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=DPI)
    
    # 2. 指定轴位置 [left, bottom, width, height]
    # bottom 和 height 留出 0.05 的边距，防止 extend 的尖角超出边界
    cbar_width_pct = 0.25 
    ax = fig.add_axes([0.1, 0.05, cbar_width_pct, 0.9]) 

    ticks = np.linspace(diff_maplevs[0], diff_maplevs[-1], ndiff_maplevs)
    norm  = mcolors.Normalize(vmin=diff_maplevs[0], vmax=diff_maplevs[-1])
    
    # 添加 extend 参数
    cbar = ColorbarBase(ax, cmap=diffcm, norm=norm, orientation='vertical',
                        ticks=ticks, spacing='uniform', extend=extend_type)

    cbar.set_label(label, fontsize=LABEL_FZ, labelpad=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=TICK_FZ, length=Length*1.5) 
    cbar.ax.set_yticklabels([TICK_FMT.format(t) for t in ticks])

    # 强制更新，确保文本不会被裁剪
    fig.savefig(savepath, dpi=DPI, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    return 1



def plot_spatial_cbar_core_H(mapcfg, Length, label, savepath, tickfz=None, labelfz=None, 
                             thickness_ratio=0.06, extend_type='neither', fmt=None):
    """
    横向色条 (Horizontal)
    :param Length: 色条的总宽度 (英寸)
    :param thickness_ratio: 色条高度相对于宽度的比例
    :param extend_type: 扩展类型，可选 'both', 'max', 'min', 'neither'
    """
    diffcm         = mapcfg.cmap
    diff_maplevs   = mapcfg.levs[0]
    ndiff_maplevs  = mapcfg.levs[1]

    # 动态字体大小
    LABEL_FZ = labelfz if labelfz is not None else int(Length * 2.2)
    TICK_FZ  = tickfz if tickfz is not None else int(Length * 1.8)
    TICK_FMT = fmt if fmt is not None else '{:.2f}'


    # 1. 计算画布尺寸
    fig_w_in = Length
    fig_h_in = Length * thickness_ratio * 4 

    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=DPI)
    
    # 2. 设置轴位置 [left, bottom, width, height]
    # left 和 width 留出边距给横向的 extend 尖角
    cbar_height_pct = 0.25
    ax = fig.add_axes([0.05, 0.4, 0.9, cbar_height_pct])

    ticks = np.linspace(diff_maplevs[0], diff_maplevs[-1], ndiff_maplevs)
    norm  = mcolors.Normalize(vmin=diff_maplevs[0], vmax=diff_maplevs[-1])
    
    # 添加 extend 参数
    cbar = ColorbarBase(ax, cmap=diffcm, norm=norm, orientation='horizontal',
                        ticks=ticks, spacing='uniform', extend=extend_type)

    cbar.ax.xaxis.set_label_position('bottom')
    cbar.set_label(label, fontsize=LABEL_FZ, labelpad=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=TICK_FZ, length=Length)
    cbar.ax.set_xticklabels([TICK_FMT.format(t) for t in ticks])

    fig.savefig(savepath, dpi=DPI, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    return 1




def plot_hot_wet_coupling_colorbar(cmap, tick_labels, colorbar_label, savepath, font_size: int = 12):
    """
    Draw a custom colorbar with three color blocks and labels, and save it as a PNG.
    
    Parameters:
    - cmap: List of colors for the three states [Down, None, Up]
    - tick_labels: Custom tick labels for the colorbar
    - colorbar_label: Label for the colorbar
    - savepath: Path to save the colorbar PNG
    - font_size: Font size for the colorbar label and ticks (default: 12)
    """
    # Create the figure and axis for the colorbar
    fig, ax = plt.subplots(figsize=(6, 0.5))  # Adjust the size to fit the colorbar horizontally

    # Get the number of colors in the colormap
    num_colors = len(cmap.colors)  # Now cmap is assumed to be a list of colors
    
    # Define the width of the color blocks
    block_width = 1 / num_colors
    
    # Create the color blocks as rectangles
    for i, color in enumerate(cmap.colors):
        ax.add_patch(Rectangle((i * block_width, 0), block_width, 1, color=color))
    
    # Set the ticks and labels
    for i, label in enumerate(tick_labels):
        ax.text(i * block_width + block_width / 2, 1.075, label, ha='center', va='bottom', fontsize=14)
    
    # Set the colorbar label
    ax.text(0.5, -0.3, colorbar_label, ha='center', va='top', fontsize=16, fontweight='bold')

    # Remove axis and ticks
    ax.set_axis_off()
    
    # Set the aspect ratio to make the blocks even
    ax.set_aspect('auto')

    # Save the colorbar as PNG
    plt.savefig(savepath, dpi=DPI, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)



def plot_discrete_cbar_core_H(
        mapcfg,
        Width: float,
        label: str,
        savepath: str,
        ticklabels: list[str] = None,
        tickfz=None, labelfz=None
    ):
    """
    横向“离散分类”色条（用于分类标签 / 分档色带）。

    参数
    ----
    mapcfg.cmap : 离散 colormap（比如 N 个类别的颜色）
    mapcfg.levs[0] : boundaries（断点数组），长度 = n_boundaries
                     - 若是分类分档，一般是 [b0, b1, ..., bN]
                     - N 个颜色，对应 N 个区间
    Width : 图的总宽度（英寸）
    label : 色标标题
    savepath : 输出路径
    ticklabels : list[str] 或 None
        - 若为 None：刻度直接显示数值（levels 或 midpoints）
        - 若长度 = len(levels)-1：认为是每个区间的标签，在区间中点处打刻度
    """
    cmap   = mapcfg.cmap
    bounds = np.asarray(mapcfg.levs[0], dtype=float)  # 断点，例如 [-1.5, -0.5, 0.5, 1.5]
    n_bounds = bounds.size
    if n_bounds < 2:
        raise ValueError(f"离散色标需要至少 2 个断点，目前 levels={bounds}")

    # ========= 几何常量（英寸，与原函数保持一致） =========
    BAR_H_IN = 0.30     # 色条本体高度
    PAD_L_IN = 0.20     # 左边
    PAD_R_IN = 0.20     # 右边
    PAD_T_IN = 0.10     # 上边（给 label）
    PAD_B_IN = 0.90     # 下边（给刻度）
    LABEL_FZ = labelfz if labelfz is not None else 16
    TICK_FZ  = tickfz if tickfz is not None else 12

    def make_axes(fig_w_in, fig_h_in):
        fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=DPI)

        left   = PAD_L_IN / fig_w_in
        bottom = PAD_B_IN / fig_h_in
        width  = 1.0 - (PAD_L_IN + PAD_R_IN) / fig_w_in
        height = BAR_H_IN / fig_h_in

        ax = fig.add_axes([left, bottom, width, height])
        return fig, ax

    def save_h_cbar_discrete(fig_w_in, cmap, bounds, label, savepath, ticklabels):
        fig_h_in = PAD_T_IN + BAR_H_IN + PAD_B_IN
        fig, ax = make_axes(fig_w_in, fig_h_in)

        # 使用 BoundaryNorm，与分类地图保持一致
        norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=cmap.N)

        # ---- 计算 ticks & ticklabels ----
        if ticklabels is not None and len(ticklabels) == (len(bounds) - 1):
            # 每个区间一个标签：刻度放在中点
            ticks = 0.5 * (bounds[:-1] + bounds[1:])
            tick_text = ticklabels
        else:
            # 直接用边界值做刻度，但去除边界的刻度（即第一个和最后一个）
            ticks = bounds[1:-1]  # 去除第一个和最后一个边界的刻度
            tick_text = ticklabels[1:-1] if ticklabels is not None else [str(v) for v in ticks]

        cbar = ColorbarBase(
            ax,
            cmap=cmap,
            norm=norm,
            boundaries=bounds,
            ticks=ticks,
            orientation='horizontal',
            spacing='uniform',   # 每个类别等宽显示
        )

        cbar.ax.xaxis.tick_bottom()
        cbar.ax.xaxis.set_label_position('bottom')
        cbar.set_label(label, fontsize=LABEL_FZ, labelpad=8, fontweight='bold')

        cbar.ax.tick_params(labelsize=TICK_FZ)
        
        # 设置刻度标签
        if tick_text:
            cbar.ax.set_xticklabels(tick_text)
        else:
            # 如果没有刻度标签，隐藏刻度
            cbar.ax.set_xticklabels([])
            cbar.ax.tick_params(length=0)  # 隐藏刻度线

        # 隐藏所有的小刻度（包括边界上的）
        cbar.ax.tick_params(which='both', length=0)
        
        # 重新设置主要刻度，只显示我们指定的刻度
        cbar.ax.xaxis.set_ticks(ticks)
        
        # 确保不显示第一个和最后一个边界上的刻度
        cbar.ax.xaxis.set_ticks_position('none')  # 不显示刻度线
        cbar.ax.tick_params(axis='x', which='major', length=0)  # 主要刻度长度为0

        cbar.ax.xaxis.label.set_clip_on(False)
        for t in cbar.ax.get_xticklabels():
            t.set_clip_on(False)

        fig.savefig(savepath, dpi=DPI, bbox_inches='tight')
        plt.close(fig)

    save_h_cbar_discrete(Width, cmap, bounds, label, savepath, ticklabels)
    return 1






def plot_discrete_cbar_core_V(
        mapcfg,
        Height: float,
        label: str,
        savepath: str,
        ticklabels: list[str] = None,
        tickfz=None, labelfz=None
    ):
    """
    竖直“离散分类”色条（用于分类 / 分档标签）。

    参数
    ----
    mapcfg.cmap      : colormap（通常是离散的）
    mapcfg.levs[0]   : boundaries（断点数组），长度 = n_boundaries
                       N 个颜色 → N 个区间，bounds 形如 [b0, b1, ..., bN]
    Height           : 整个图的总高度（英寸）
    label            : 色标标题
    savepath         : 输出路径
    ticklabels       : list[str] 或 None
                       - 若为 None：刻度直接显示数值（bounds 或 midpoints）
                       - 若长度 = len(bounds)-1：每个区间一个标签，在中点处打刻度
    """
    cmap   = mapcfg.cmap
    bounds = np.asarray(mapcfg.levs[0], dtype=float)
    n_bounds = bounds.size
    if n_bounds < 2:
        raise ValueError(f"离散色标需要至少 2 个断点，目前 levels={bounds}")

    # ========= 几何常量（英寸，与连续版保持一致） =========
    BAR_W_IN = 0.40     # 色条本体宽度
    PAD_L_IN = 0.10     # 左边（很小即可）
    PAD_R_IN = 0.90     # 右边（为 ticks/label 预留足够空间）
    PAD_T_IN = 0.20     # 上边
    PAD_B_IN = 0.20     # 下边
    LABEL_FZ = labelfz if labelfz is not None else 16
    TICK_FZ  = tickfz if tickfz is not None else 12

    def make_axes(fig_w_in, fig_h_in):
        fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=DPI)
        left   = PAD_L_IN / fig_w_in
        bottom = PAD_B_IN / fig_h_in
        width  = BAR_W_IN / fig_w_in
        height = 1.0 - (PAD_T_IN + PAD_B_IN) / fig_h_in
        ax = fig.add_axes([left, bottom, width, height])
        return fig, ax

    def save_v_cbar_discrete(fig_h_in, cmap, bounds, label, savepath, ticklabels):
        # 宽度由条宽 + padding 决定
        fig_w_in = PAD_L_IN + BAR_W_IN + PAD_R_IN
        fig, ax  = make_axes(fig_w_in, fig_h_in)

        # 和地图一致的 BoundaryNorm
        norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=cmap.N)

        # ---- 计算 ticks & ticklabels ----
        if ticklabels is not None and len(ticklabels) == (len(bounds) - 1):
            # 每个区间一个标签：刻度放在中点
            ticks = 0.5 * (bounds[:-1] + bounds[1:])
            tick_text = ticklabels
        else:
            # 直接用边界值做刻度，但去除边界的刻度（即第一个和最后一个）
            ticks = bounds[1:-1]  # 去除第一个和最后一个边界的刻度
            tick_text = ticklabels[1:-1] if ticklabels is not None else [str(v) for v in ticks]

        cbar  = ColorbarBase(
            ax,
            cmap=cmap,
            norm=norm,
            boundaries=bounds,
            ticks=ticks,
            orientation='vertical',
            spacing='uniform',   # 每个类别等宽
        )

        cbar.set_label(label, fontsize=LABEL_FZ, labelpad=8, fontweight='bold')
        cbar.ax.tick_params(labelsize=TICK_FZ)
        cbar.ax.set_yticklabels(tick_text)

        cbar.ax.yaxis.label.set_clip_on(False)
        for t in cbar.ax.get_yticklabels():
            t.set_clip_on(False)

        fig.savefig(savepath, dpi=DPI, bbox_inches='tight')
        plt.close(fig)

    save_v_cbar_discrete(Height, cmap, bounds, label, savepath, ticklabels)
    return 1




def plot_spatial_cbar_core_heatmap(
    mapcfg, Length, label, savepath,
    tickfz=None, labelfz=None,
    thickness_ratio=0.08,
    extend_type='neither',
    fmt=None,
    end_text=("Decrease", "Increase"),   # (bottom, top)
    endfz=None,                          # 端点文字字号（不传则随 Length 缩放）
    endpad=0.03,                         # 端点文字离色条端部的相对距离（Axes坐标）
):
    """
    纵向色条 (Vertical)

    :param Length: 色条总高度(英寸)
    :param thickness_ratio: 色条宽度相对于高度的比例
    :param extend_type: 'both','max','min','neither'
    :param end_text: (bottom_text, top_text) e.g. ("Decrease","Increase")
    :param endpad: 端点文字与色条端点的间距（axes fraction）
    """

    diffcm        = mapcfg.cmap
    diff_maplevs  = mapcfg.levs[0]
    ndiff_maplevs = mapcfg.levs[1]

    # 动态计算字体：随 Length 自动缩放
    LABEL_FZ = labelfz if labelfz is not None else int(Length * 2.2)
    TICK_FZ  = tickfz  if tickfz  is not None else int(Length * 1.8)
    END_FZ   = endfz   if endfz   is not None else int(Length * 2.0)
    TICK_FMT = fmt if fmt is not None else '{:.2f}'

    # 1) 计算画布尺寸
    fig_h_in = Length
    fig_w_in = Length * thickness_ratio * 4  # 宽度预留给刻度和标签

    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=DPI)

    # 2) 指定轴位置 [left, bottom, width, height]
    cbar_width_pct = 0.25
    ax = fig.add_axes([0.1, 0.05, cbar_width_pct, 0.9])

    ticks = np.linspace(diff_maplevs[0], diff_maplevs[-1], ndiff_maplevs)
    norm  = mcolors.Normalize(vmin=diff_maplevs[0], vmax=diff_maplevs[-1])

    cbar = ColorbarBase(
        ax, cmap=diffcm, norm=norm, orientation='vertical',
        ticks=ticks, spacing='uniform', extend=extend_type
    )

    cbar.set_label(label, fontsize=LABEL_FZ, labelpad=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=TICK_FZ, length=Length*1.5)
    cbar.ax.set_yticklabels([TICK_FMT.format(abs(t)) for t in ticks])

    # --- 关键新增：上下端文本 ---
    bottom_text, top_text = end_text

    # 放在色条轴的上方/下方（Axes坐标系：y=1 是顶部，y=0 是底部）
    # x 取 0.5 让它居中对齐色条
    ax.text(
        0.5, 1.0 + endpad, top_text,
        transform=ax.transAxes,
        ha='left', va='bottom',
        fontsize=END_FZ, fontweight='bold'
    )
    ax.text(
        0.5, 0.0 - endpad, bottom_text,
        transform=ax.transAxes,
        ha='left', va='top',
        fontsize=END_FZ, fontweight='bold'
    )

    fig.savefig(savepath, dpi=DPI, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    return 1




def plot_spatial_cbar_core_heatmap_new(
    mapcfg, Length, label, savepath,
    tickfz=None, labelfz=None,
    thickness_ratio=0.08,
    extend_type='neither',
    fmt=None,
    end_text=("Decrease", "Increase"),   # (bottom, top)
    endfz=None,                          # 端点文字字号（不传则随 Length 缩放）
    endpad=0.03,                         # 端点文字离色条端部的相对距离（Axes坐标）
):
    """
    纵向色条 (Vertical)

    :param Length: 色条总高度(英寸)
    :param thickness_ratio: 色条宽度相对于高度的比例
    :param extend_type: 'both','max','min','neither'
    :param end_text: (bottom_text, top_text) e.g. ("Decrease","Increase")
    :param endpad: 端点文字与色条端点的间距（axes fraction）
    """

    diffcm        = mapcfg.cmap
    diff_maplevs  = mapcfg.levs[0]
    ndiff_maplevs = mapcfg.levs[1]

    # 动态计算字体：随 Length 自动缩放
    LABEL_FZ = labelfz if labelfz is not None else int(Length * 2.2)
    TICK_FZ  = tickfz  if tickfz  is not None else int(Length * 1.8)
    END_FZ   = endfz   if endfz   is not None else int(Length * 2.0)
    TICK_FMT = fmt if fmt is not None else '{:.2f}'

    # 1) 计算画布尺寸
    fig_h_in = Length
    fig_w_in = Length * thickness_ratio * 4  # 宽度预留给刻度和标签

    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=DPI)

    # 2) 指定轴位置 [left, bottom, width, height]
    cbar_width_pct = 0.25
    ax = fig.add_axes([0.1, 0.05, cbar_width_pct, 0.9])

    ticks = np.linspace(diff_maplevs[0], diff_maplevs[-1], ndiff_maplevs)
    norm  = mcolors.Normalize(vmin=diff_maplevs[0], vmax=diff_maplevs[-1])

    cbar = ColorbarBase(
        ax, cmap=diffcm, norm=norm, orientation='vertical',
        ticks=ticks, spacing='uniform', extend=extend_type
    )

    # cbar.set_label(label, fontsize=LABEL_FZ, labelpad=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=TICK_FZ, length=Length*1.5)
    cbar.ax.set_yticklabels([TICK_FMT.format(abs(t)) for t in ticks])

    # --- 关键新增：上下端文本 ---
    bottom_text, top_text = end_text

    # 放在色条轴的上方/下方（Axes坐标系：y=1 是顶部，y=0 是底部）
    # x 取 0.5 让它居中对齐色条
    ax.text(
        3.2, 0.8, top_text,
        transform=ax.transAxes,
        ha='center', va='center', rotation=90,
        fontsize=END_FZ, fontweight='bold'
    )
    ax.text(
        3.2, 0.2, bottom_text,
        transform=ax.transAxes,
        ha='center', va='center', rotation=90,
        fontsize=END_FZ, fontweight='bold'
    )

    ax.text(
        3.2, 0.5, "No Change",
        transform=ax.transAxes,
        ha='center', va='center', rotation=90,
        fontsize=END_FZ, fontweight='bold'
    )

    fig.savefig(savepath, dpi=DPI, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    return 1