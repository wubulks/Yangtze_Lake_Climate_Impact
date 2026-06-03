import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# 自定义工具箱 (保持原样)
import ToolBoxes.Tool_PlotConfig as TPC
FIGFMT = TPC.FIGFMT
DPI    = TPC.DPI_medium
BASEDATA = TPC.BASEDATA

def plot_heatmap(
    df: pd.DataFrame,
    label_col: str,
    percent_col: str,
    area_col: str,
    area_pct_col: str,
    title: str = None,
    color_map: str = 'coolwarm',
    savepath: str = None,
    figsize: tuple = (6.5, 6) # 稍微加宽加高，以容纳更多文字
):
    # --- 1. 数据映射与含义定义 ---
    # 构建 2x2 索引映射，同时定义每个格子的语义标签
    mapping = {
        (0, 0): {'key': 'Lake-Up',   'trend': 'Increase', 'desc': 'Local'},
        (0, 1): {'key': 'Land-Up',   'trend': 'Increase', 'desc': 'Non-local'},
        (1, 0): {'key': 'Lake-Down', 'trend': 'Decrease', 'desc': 'Local'},
        (1, 1): {'key': 'Land-Down', 'trend': 'Decrease', 'desc': 'Non-local'}
    }
    
    val_matrix = np.zeros((2, 2))
    for (r, c), info in mapping.items():
        row = df[df[label_col] == info['key']]
        if not row.empty:
            val_matrix[r, c] = row[percent_col].values[0]

    fig, ax = plt.subplots(figsize=figsize)
    
    # --- 核心修改：设置 vmin, vmax 和 center ---
    sns.heatmap(
        val_matrix, 
        cmap=color_map, 
        vmin=-100,      # 设置最小值
        vmax=100,       # 设置最大值
        center=0,       # 确保 0 是颜色的分界点
        cbar=False, 
        linewidths=0, 
        ax=ax
    )

    # --- 2. 核心修改：分层标注，明确语义与基准 ---
    for (i, j), info in mapping.items():
        row_data = df[df[label_col] == info['key']]
        if row_data.empty: continue
        
        p = row_data[percent_col].values[0]
        a = row_data[area_col].values[0]
        ap = row_data[area_pct_col].values[0]

        if abs(p) > 60:
            incolor = 'white'
        else:
            incolor = 'black'

        # A. 顶部语义标签 (Increase/Decrease & Local/Non-local)
        # 放在格子顶部 i + 0.15 位置
        ax.text(j + 0.05, i + 0.1, f"{info['trend']}", 
                ha='left', va='center', fontsize=18, fontweight='bold', 
                color=incolor)
        
        ax.text(j + 0.95, i + 0.1, f"{info['desc']}", 
                ha='right', va='center', fontsize=18, fontweight='bold', 
                color=incolor)

        # B. 中间行：占显著区域的比例 (分母: Significant Areas)
        # 放置在中心偏上位置，字体最大，作为核心指标
        ax.text(j + 0.5, i + 0.4, f"{abs(p):.2f}%", 
                ha='center', va='center', fontsize=24, fontweight='black', color=incolor)
        # ax.text(j + 0.5, i + 0.5, "of Sig. Area", 
        #         ha='center', va='center', fontsize=16, fontweight='bold', color=incolor)
        
        # 视觉分割线：区分统计占比与物理覆盖
        ax.plot([j + 0.15, j + 0.85], [i + 0.55, i + 0.55], color=incolor, lw=1.0)

        # C. 底部行：占整个研究区的物理属性 (分母: Total Study Area)
        # 放置在中心偏下位置
        area_info = f"Area: {abs(a):.2f} ×10⁴ km²"
        ax.text(j + 0.5, i + 0.7, area_info, 
                ha='center', va='center', fontsize=17, color=incolor)
        
        total_pct_info = f"({abs(ap):.2f}% of study area)"
        ax.text(j + 0.5, i + 0.8, total_pct_info, 
                ha='center', va='center', fontsize=17, color=incolor, fontstyle='italic')

    # --- 3. 边框与轴样式 ---
    line_width = 1 # 稍微加粗边框，增加质感
    for k in range(3):
        ax.hlines(k, 0, 2, colors='black', lw=line_width)
        ax.vlines(k, 0, 2, colors='black', lw=line_width)

    # --- X轴和Y轴：彻底移除刻度与标签（只保留主图） ---
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # 保险起见：把 seaborn/matplotlib 可能残留的 tick 也关掉
    ax.tick_params(axis='both', which='both',
                   bottom=False, top=False, left=False, right=False,
                   labelbottom=False, labeltop=False, labelleft=False, labelright=False,
                   length=0)

    # 不需要把 x tick 放到顶部了（因为我们已移除）
    # ax.xaxis.tick_top()
    # ax.xaxis.set_label_position('top')

    # 可选：隐藏外框（你已经自己画了粗边框线）
    for spine in ax.spines.values():
        spine.set_visible(True)

    if title:
        # 增加 pad 以防标题和上方标签重合
        ax.set_title(title, fontsize=26, pad=10, fontweight='bold')

    plt.tight_layout()
    if savepath:
        # 确保 DPI 使用之前的全局变量或默认值
        fig.savefig(savepath, bbox_inches="tight", dpi=DPI, format=FIGFMT)
    
    return fig, ax




def plot_heatmap_ticks(
    df: pd.DataFrame,
    label_col: str,
    percent_col: str,
    area_col: str,
    area_pct_col: str,
    title: str = None,
    color_map: str = 'coolwarm',
    show_axis_labels: str = "xy",
    savepath: str = None,
    figsize: tuple = (6.5, 6),

    # ✅ 新增：控制只在上/左添加哪些 tick 标签："" / "x" / "y" / "xy"

    # ✅ 你要显示的轴标签文本
    x_tick_labels: tuple = ("Local", "Non-local"),
    y_tick_labels: tuple = ("Increase", "Decrease"),

    # ✅ 边框
    grid_lw: float = 1.0,
    tick_fz: int = 20,
    title_fz: int = 26,
):
    """
    2x2 heatmap:
      columns: Local, Non-Local   (top xticks only)
      rows   : Increase, Decrease (left yticks only)

    show_axis_labels:
      "x"  -> only top xticks
      "y"  -> only left yticks
      "xy" -> both
      ""   -> none
    """

    # --- 1) 映射（注意：这里的矩阵行顺序与你的 y_tick_labels 对齐） ---
    # 我这里按 y_tick_labels=("Increase","Decrease") 定义行 0=Increase, 1=Decrease
    # 列 0=Local, 1=Non-Local
    mapping = {
        (0, 0): {'key': 'Lake-Up'},     # Increase Local
        (0, 1): {'key': 'Land-Up'},     # Increase Non-Local
        (1, 0): {'key': 'Lake-Down'},   # Decrease Local
        (1, 1): {'key': 'Land-Down'}    # Decrease Non-Local
    }

    val_matrix = np.full((2, 2), np.nan)
    for (r, c), info in mapping.items():
        row = df.loc[df[label_col] == info['key']]
        if not row.empty:
            val_matrix[r, c] = row[percent_col].values[0]

    fig, ax = plt.subplots(figsize=figsize)

    # --- 核心修改：设置 vmin, vmax 和 center ---
    sns.heatmap(
        val_matrix, 
        cmap=color_map, 
        vmin=-100,      # 设置最小值
        vmax=100,       # 设置最大值
        center=0,       # 确保 0 是颜色的分界点
        cbar=False, 
        linewidths=0, 
        ax=ax
    )

    # --- 2. 核心修改：分层标注，明确语义与基准 ---
    for (i, j), info in mapping.items():
        row_data = df[df[label_col] == info['key']]
        if row_data.empty: continue
        
        p = row_data[percent_col].values[0]
        a = row_data[area_col].values[0]
        ap = row_data[area_pct_col].values[0]

        if abs(p) > 60:
            incolor = 'white'
        else:
            incolor = 'black'

        # B. 中间行：占显著区域的比例 (分母: Significant Areas)
        # 放置在中心偏上位置，字体最大，作为核心指标
        ax.text(j + 0.5, i + 0.2, f"{abs(p):.2f}%", 
                ha='center', va='center', fontsize=24, fontweight='black', color=incolor)
        ax.text(j + 0.5, i + 0.35, "of Sig. change area", 
                ha='center', va='center', fontsize=16, fontweight='bold', color=incolor)
        
        # 视觉分割线：区分统计占比与物理覆盖
        ax.plot([j + 0.15, j + 0.85], [i + 0.5, i + 0.5], color=incolor, lw=1.0)

        # C. 底部行：占整个研究区的物理属性 (分母: Total Study Area)
        # 放置在中心偏下位置
        area_info = f"Area: {abs(a):.2f} ×10⁴ km²"
        ax.text(j + 0.5, i + 0.65, area_info, 
                ha='center', va='center', fontsize=17, color=incolor)
        
        total_pct_info = f"({abs(ap):.2f}% of study area)"
        ax.text(j + 0.5, i + 0.8, total_pct_info, 
                ha='center', va='center', fontsize=17, color=incolor)

    # --- 4) 画黑色粗边框（2x2） ---
    for k in range(3):
        ax.hlines(k, 0, 2, colors='black', lw=grid_lw)
        ax.vlines(k, 0, 2, colors='black', lw=grid_lw)

    # --- 5) 只在上侧/左侧添加 ticks（由 show_axis_labels 控制） ---
    s = (show_axis_labels or "").lower()

    # 先把所有 tick 清空（避免 seaborn 默认残留）
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # x：上侧
    if "x" in s:
        ax.set_xticks([0.5, 1.5])
        ax.set_xticklabels(list(x_tick_labels), fontsize=tick_fz, fontweight='bold')
        ax.xaxis.tick_top()
        ax.tick_params(axis='x', top=True, bottom=False, labeltop=True, labelbottom=False, length=0, pad=8)

    # y：左侧
    if "y" in s:
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(list(y_tick_labels), fontsize=tick_fz, fontweight='bold', rotation=90)
        ax.tick_params(axis='y', left=True, right=False, labelleft=True, labelright=False, length=0, pad=8)

    # 确保右侧/下侧不显示任何东西
    ax.tick_params(axis='both', which='both', right=False, bottom=False)

    # 可选：隐藏外框（你已经手动画了边框线）
    for spine in ax.spines.values():
        spine.set_visible(True)

    if title:
        ax.set_title(title, fontsize=title_fz, fontweight='bold', pad=10)

    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches="tight", dpi=DPI, format=FIGFMT)
    return fig, ax