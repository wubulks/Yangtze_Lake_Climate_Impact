import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import squarify 
import warnings


# 假设您的环境中已定义这些变量（示例值）
FIGFMT = 'png'
DPI    = 300

mpl.use('Agg')  # 不显示图，只保存
mpl.rcParams['font.family'] = 'Noto Sans'
warnings.filterwarnings("ignore", category=RuntimeWarning)


def plot_Treemap(
        df_in,
        path, 
        values,
        title,
        savepath,
        cfg,
        color_col: str = None,
    ):
    """
    【最终修正版本】
    - **数据过滤：只保留原始数值 (values) >= 0.5M 的数据。**
    - 面积决定：Treemap 面积按分级数量平均分配给每个分级。
    - 块内分配：块内面积按元素数量均匀分配。
    - 颜色由原始 'values' 决定。
    - 完全移除图例。
    """
    
    # 确定用于计算面积和分箱的列
    value_col = values
    color_target_col = color_col if color_col is not None else value_col
    
    # --- 过滤数据 ---
    # 1. 过滤掉 area/color 值为 NaN 的行
    df = df_in.dropna(subset=[color_target_col]).copy()
    
    # 2. **新增过滤条件：只保留 value_col >= 0.1M 的行**
    MIN_VALUE_THRESHOLD = 200000.0
    df = df[df[value_col] >= MIN_VALUE_THRESHOLD].copy()
    
    if df.empty:
        print(f"警告：Treemap '{title}' 在过滤 (>= {MIN_VALUE_THRESHOLD/1e6:.1f}M) 后没有有效数据。跳过保存到 {savepath}。")
        return

    # ===== 1) 规范化 boundaries (不变) =====
    levels = cfg.levs
    cmap = cfg.cmap
    lev_arr = np.asarray(levels, dtype=object)

    if lev_arr.ndim >= 1 and isinstance(lev_arr[0], (list, tuple, np.ndarray)):
        boundaries = np.asarray(lev_arr[0], dtype=float)
    else:
        boundaries = np.asarray(lev_arr, dtype=float)

    boundaries = np.ravel(boundaries).astype(float)

    if boundaries.size < 2:
        raise ValueError(f"Boundary 数量太少，必须至少有两个断点，现在 boundaries={boundaries}")

    # ===== 2) 构造离散 colormap 映射和标签 (不变) =====
    n_bins = boundaries.size - 1
    
    colors_list = [
        mcolors.to_hex(cmap((i + 0.5) / n_bins))
        for i in range(n_bins)
    ]

    labels = []
    for i in range(n_bins):
        lo = boundaries[i]
        hi = boundaries[i + 1]
        if hi >= 1e6:
            labels.append(f"{lo/1e6:.1f}–{hi/1e6:.1f}M")
        elif hi >= 1e3:
            labels.append(f"{lo/1e3:.0f}–{hi/1e3:.0f}k")
        else:
            labels.append(f"{lo:.0f}–{hi:.0f}")

    # ===== 3) 按 levels 分箱，分配「类别」和「颜色」 (不变) =====
    val_arr = df[color_target_col].to_numpy(dtype=float)

    bin_idx = np.digitize(val_arr, boundaries) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    df["_bin_idx"] = bin_idx
    df["_bin_label"] = df["_bin_idx"].map({i: labels[i] for i in range(n_bins)})
    df["_color_hex"] = df["_bin_idx"].map({i: colors_list[i] for i in range(n_bins)})
    
    # --- 关键重构：新的面积计算 (按分级平均分配总面积) ---
    
    # 1. 计算每个分级（bin）中元素的数量
    bin_counts = df.groupby("_bin_idx").size().to_dict()
    
    # 2. 计算实际存在的、需要分配面积的分级数
    actual_bins_present = len(bin_counts)
    if actual_bins_present == 0:
        # 理论上应该不会发生，但以防万一
        print(f"警告：Treemap '{title}' 在分箱后没有有效数据。跳过保存到 {savepath}。")
        return
        
    # 3. 计算每个分级应占的总面积（基于 100000 的基数）
    TOTAL_AREA_BASE = 100000.0
    area_per_bin_group = TOTAL_AREA_BASE / actual_bins_present
    
    # 4. 计算每个数据点（行政单元）应占的面积
    def calculate_unit_area(row):
        bin_idx = row['_bin_idx']
        count = bin_counts[bin_idx]
        return area_per_bin_group / count

    df["_unit_area"] = df.apply(calculate_unit_area, axis=1)

    # 5. 排序策略：先按分级标签，后按原始数值降序排序
    df.sort_values(
        by=["_bin_idx", value_col],  # 先按分级索引，后按原始值
        ascending=[True, False],     # 分级按升序，原始值按降序
        inplace=True
    )
    df.reset_index(drop=True, inplace=True)


    # 准备 squarify 需要的参数
    if isinstance(path, (list, tuple)):
        label_col = path[0] 
    else:
        label_col = path
        
    # **使用计算出的平均面积作为 Treemap 的 sizes**
    sizes = df["_unit_area"].tolist() 
    
    # 创建标签 (分级标签 + 名称 + 原始数值)
    squarify_labels = [
        f"{row[label_col]}\n({row[value_col]})" 
        for _, row in df.iterrows()
    ]
    squarify_colors = df["_color_hex"].tolist()


    # ===== 4) 画 Treemap (使用 squarify/Matplotlib) =====
    plt.figure(figsize=(10, 8))
    
    squarify.plot(
        sizes=sizes,
        label=squarify_labels,
        color=squarify_colors,
        pad=True,
        ax=plt.gca(),
        text_kwargs={'fontsize': 8, 'color': 'black'}
    )

    plt.title(title, fontsize=16)
    plt.axis('off') # 隐藏坐标轴
    
    # 保存图像
    plt.savefig(
        savepath, 
        format=FIGFMT, 
        dpi=DPI, 
        bbox_inches='tight' 
    )
    plt.close()



def plot_treemap_liang(
    df: pd.DataFrame,
    label_col: str,
    percent_col: str,
    color_map: dict = None,
    savepath: str = None,
    title: str = None,
    colors: list = None,
    normalize_if_not_100: bool = True,
    figsize: tuple = (5, 5)
):
    """
    绘制 Treemap，修复了数值为 0 导致的 ZeroDivisionError
    """
    if label_col not in df.columns or percent_col not in df.columns:
        raise ValueError(f"df 必须包含列: {label_col}, {percent_col}")

    # --- 1. 数据清洗 ---
    d = df[[label_col, percent_col]].copy().dropna()
    d[percent_col] = pd.to_numeric(d[percent_col], errors="coerce")
    d = d.dropna(subset=[percent_col])

    # --- 2. 核心修复：过滤掉数值为 0 的项 ---
    # squarify 不支持大小为 0 的块
    d = d[d[percent_col] > 1e-10] 
    
    if d.empty:
        print("Warning: 所有数值均为 0 或有效数据为空，跳过绘图。")
        return None, None

    labels = d[label_col].astype(str).tolist()
    values = d[percent_col].to_numpy(dtype=float)

    # --- 3. 归一化 (保持原逻辑) ---
    total = np.nansum(values)
    if normalize_if_not_100:
        values = (values / total) * 100

    # --- 4. 颜色处理 ---
    if color_map:
        plot_colors = [color_map.get(lab, "#C0C0C0") for lab in labels]
    elif colors:
        plot_colors = colors[:len(labels)]
    else:
        # 使用 matplotlib 默认颜色循环
        plot_colors = plt.cm.get_cmap('Pastel1')(np.linspace(0, 1, len(labels)))

    # --- 5. 绘图 ---
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制 Treemap
    import squarify
    squarify.plot(
        sizes=values, 
        label=[f"{l}\n{v:.1f}%" for l, v in zip(labels, values)], 
        color=plot_colors, 
        alpha=0.8, 
        ax=ax,
        edgecolor="black", # 黑色边框区分块
        linewidth=1.5,     # 边框粗细
        text_kwargs={'fontsize': 12, 'fontweight': 'bold'}
    )

    if title:
        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)

    # 移除坐标轴
    ax.axis('off')
    plt.tight_layout()

    if savepath:
        # 使用全局配置中的 FIGFMT 和 DPI
        fig.savefig(savepath, bbox_inches="tight", format=FIGFMT, dpi=DPI)
    
    return fig, ax