import os
import time
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import cartopy.crs as ccrs
from cartopy.io import shapereader
import cartopy.feature as cfeature
from cnmaps import get_adm_maps, draw_maps
from matplotlib.colorbar import ColorbarBase
from typing import Literal, Optional, Tuple, Dict, Any, Sequence, List, Union
from matplotlib import colors as mcolors
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from cartopy.mpl.geoaxes import GeoAxes
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# 自定义工具箱
import ToolBoxes.Utils as TU
import ToolBoxes.Tool_PlotConfig as TPC
FIGFMT = TPC.FIGFMT
DPI    = TPC.DPI_medium

mpl.use('Agg')  # 不显示图，只保存
mpl.rcParams['font.family'] = 'Noto Sans'
warnings.filterwarnings("ignore", category=RuntimeWarning)


# =========================== 函数区 ===========================
def _set_yAxes(ax: Axes, in_df: pd.DataFrame, tickfontsize: float = 11, labelfontsize: float = 11, 
               ylims: List[float] | None = None, yticks: List[float] = None, yticklabels: List[str] | None = None,
               ylabel: str = '') -> Axes:
    """
    自动设置 y 轴范围与刻度：
    - 若未传 ylims/yticks，则自动：
        * 0 必须包含在 y 轴范围内
        * 全负时上方至少预留一个正刻度；全正时下方至少预留一个负刻度
        * 使用“漂亮步长”并将边界对齐到刻度，以避免贴边
    - 若传入 ylims/yticks，则严格尊重传入值
    """
    import numpy as _np
    import pandas as _pd

    def _nice_step(span: float, nbins: int = 5) -> float:
        """根据跨度给出‘漂亮’步长（1/2/2.5/5/10 × 10^k）"""
        if not _np.isfinite(span) or span <= 0:
            span = 1.0
        raw = span / max(nbins, 1)
        exp = _np.floor(_np.log10(raw))
        frac = raw / (10 ** exp)
        if frac <= 1:
            nice = 1.0
        elif frac <= 2:
            nice = 2.0
        elif frac <= 2.5:
            nice = 2.5
        elif frac <= 5:
            nice = 5.0
        else:
            nice = 10.0
        return float(nice * (10 ** exp))

    # ---------- 若用户显式给了 ylims/yticks：尊重它们 ----------
    if ylims is not None:
        ylims = _np.asarray(ylims, dtype=float)
        ymin, ymax = _np.nanmin(ylims), _np.nanmax(ylims)
        ax.set_ylim([ymin, ymax])
    else:
        # ---------- 从数据中取 min/max ----------
        col_mins, col_maxs = [], []
        for key in in_df.columns:
            s = _pd.to_numeric(in_df[key], errors="coerce")
            col_mins.append(_np.nanmin(s.values))
            col_maxs.append(_np.nanmax(s.values))
        data_min = _np.nanmin(_np.asarray(col_mins, dtype=float))
        data_max = _np.nanmax(_np.asarray(col_maxs, dtype=float))
        if not _np.isfinite(data_min) or not _np.isfinite(data_max):
            data_min, data_max = -1.0, 1.0  # 兜底

        # ---------- 计算步长并构造 ylims/yticks（保证包含 0） ----------
        # 三种情形：全负 / 全正 / 跨 0
        if data_max <= 0:  # 全负：上方至少一个正刻度
            step = _nice_step(abs(data_min), nbins=4)
            step = max(step, 1e-6)
            start = -_np.ceil(abs(data_min) / step) * step
            end   = 0.0 + step
        elif data_min >= 0:  # 全正：下方至少一个负刻度
            step = _nice_step(abs(data_max), nbins=4)
            step = max(step, 1e-6)
            start = 0.0 - step
            end   = _np.ceil(abs(data_max) / step) * step
        else:  # 跨 0：按数据范围取整到刻度边界
            span = data_max - data_min
            step = _nice_step(span, nbins=5)
            step = max(step, 1e-6)
            start = _np.floor(data_min / step) * step
            end   = _np.ceil(data_max / step) * step

        # 让 y 轴边界与刻度对齐，避免拥挤
        ymin, ymax = float(start), float(end)
        # 构造刻度（含 0）
        yticks = _np.arange(ymin, ymax + 0.5 * step, step, dtype=float)
        ax.set_ylim([ymin, ymax])

    # ---------- 设置刻度 ----------
    if yticks is not None and ylims is None:
        # 自动模式：我们上面已经构造了 yticks
        ax.set_yticks(yticks)
    elif yticks is not None:
        # 用户自定 yticks
        ax.set_yticks(yticks)
        ymin, ymax = _np.nanmin(yticks), _np.nanmax(yticks)
        # 若用户没传 ylims，但传了 yticks，也把 ylim 对齐到刻度范围
        if ylims is None:
            ax.set_ylim([ymin, ymax])

    # ---------- yticklabels ----------
    if yticklabels is None:
        # 根据步长决定小数位（需要确保至少两个刻度以推断步长）
        curr_ticks = ax.get_yticks()
        if len(curr_ticks) >= 2:
            step = curr_ticks[1] - curr_ticks[0]
        else:
            step = _nice_step(ax.get_ylim()[1] - ax.get_ylim()[0], nbins=5)

        if step >= 1:
            fmt = "{:.0f}"
        elif step >= 0.1:
            fmt = "{:.1f}"
        elif step >= 0.01:
            fmt = "{:.2f}"
        else:
            fmt = "{:.3f}"
        ax.set_yticklabels([fmt.format(v) for v in ax.get_yticks()], fontsize=tickfontsize)
    else:
        ax.set_yticklabels(yticklabels, fontsize=tickfontsize)

    ax.set_ylabel(ylabel, fontsize=labelfontsize, fontweight='bold', labelpad=8)
    return ax


    
def _set_xAxes(ax: Axes, in_df: pd.DataFrame, tickfontsize: float = 11, labelfontsize: float = 11,
               xlims: List[float] | None = None, xticks: List[float] = None, xticklabels: List[str] | None = None,
               xlabel: str = '', max_xticks: int = 8, force_first_last_tick: bool = True) -> Axes:
    """
    自动设置 x 轴（通常为时间）：
      - 若未传 xticks/xticklabels，会自动稀疏刻度（最多 max_xticks 个），可选择强制包含首尾。
      - 既支持 int 年份，也支持 datetime 索引（会转成年份）。
    """
    # 取索引，尽量变成年份整数
    try:
        x = in_df.index.astype(int).to_numpy()
        x_years = x
    except Exception:
        # 可能是 datetime 或字符串
        idx = pd.to_datetime(in_df.index, errors='coerce')
        if idx.notna().any():
            x_years = idx.dt.year.to_numpy()
        else:
            # 尝试逐个转 int
            def _to_int_safe(v):
                try: return int(v)
                except Exception: return np.nan
            x_years = pd.Series(in_df.index).map(_to_int_safe).to_numpy()
        keep = np.isfinite(x_years)
        in_df = in_df.loc[keep]
        x_years = x_years[keep].astype(int)

    # xlim
    if xlims is not None:
        xlims = np.asarray(xlims)
        xmin = np.nanmin(xlims)
        xmax = np.nanmax(xlims)
    else:
        xmax = np.nanmax(np.asarray(x_years))
        xmin = np.nanmin(np.asarray(x_years))
    ax.set_xlim([xmin, xmax])

    # xticks/labels
    if xticks is None:
        years_sorted = np.array(sorted(set(x_years)), dtype=int)
        n = len(years_sorted)
        if n <= max_xticks:
            xticks = years_sorted
        else:
            step = int(np.ceil(n / max_xticks))
            idx = np.arange(0, n, step)
            if force_first_last_tick and idx[-1] != n - 1:
                idx = np.r_[idx, n - 1]
            xticks = years_sorted[idx]
    ax.set_xticks(xticks)

    if xticklabels is None:
        ax.set_xticklabels([str(int(v)) for v in xticks], fontsize=tickfontsize)
    else:
        ax.set_xticklabels(xticklabels, fontsize=tickfontsize)

    ax.set_xlabel(xlabel, fontsize=labelfontsize, fontweight='bold', labelpad=15)
    return ax



def Plot_Yangtze_Serial_Comparation_Bar(
    ax: Axes,
    case_dict: Dict[str, Dict[str, Any]] | Dict[str, Any],
    xlims: List[float] | None = None,
    ylims: List[float] | None = None,
    xticks: List[float] = None,
    yticks: List[float] = None,
    xticklabels: List[str] | None = None,
    yticklabels: List[str] | None = None,
    xtickfontsize: float = 11,
    ytickfontsize: float = 11,
    xlabel: str = '',
    ylabel: str = '',
    xlabelfontsize: float = 11,
    ylabelfontsize: float = 11,
    kind: Literal["auto", "single", "group"] = "auto",
    bar_width: float | None = None,
    xtick_rotation: float = 0,
    bar_alpha: float = 0.95,
    fillna_as_zero: bool = False,
    pos_color: str = "#7686C5",
    neg_color: str = "#71BEC6",
    season_order: Sequence[str] | None = None,
    color_map: Dict[str, str] | None = None,
    # x 轴稀疏显示控制
    max_xticks: int = 6,
    force_first_last_tick: bool = True,
) -> Axes:
    """
    年度柱状图（单系列差值 / 多系列分组）——兼容两种 case_dict 形态：
      A) 嵌套：{'系列名': {'x': [...], 'y': [...], 'color': ...}, ...}
      B) 扁平：{'x': [...], 'y': [...]}  -> 自动包装为 {'Series': {...}}

    参数说明（核心）：
      - kind="single"：每年 1 根柱（常用于年际差值，自动按正负分色）。
      - kind="group" ：每年多根并排柱（常用于 4 个 case / 4 季节）。
      - max_xticks：自动稀疏 x 轴刻度的最多显示数量。
      - force_first_last_tick：是否强制包含首尾年份刻度。
      - fillna_as_zero：缺失是否按 0 画柱（默认 False：缺失不画柱，保留空位）。
    """
    # ---------- 1) 规范化输入：若是扁平结构，自动包成单系列 ----------
    if not all(isinstance(v, dict) for v in case_dict.values()):
        if "x" in case_dict and "y" in case_dict:
            case_dict = {
                "Series": {
                    "x": np.asarray(case_dict.get("x")),
                    "y": np.asarray(case_dict.get("y")),
                    "color": case_dict.get("color", None),
                }
            }
        else:
            raise ValueError("case_dict 既不是嵌套结构，也不包含 'x' 和 'y' 键。")

    # ---------- 2) 收集年份 ----------
    all_years = []
    for _, info in case_dict.items():
        x_years = pd.Series(info["x"]).map(lambda v: int(v) if pd.notna(v) else np.nan).dropna().astype(int).to_list()
        all_years.extend(x_years)
    years = np.array(sorted(set(all_years)), dtype=int)
    n_years = len(years)
    if n_years == 0:
        raise ValueError("没有有效的年份可用于绘图。")

    series_names = list(case_dict.keys())
    n_series = len(series_names)

    # ---------- 3) 自动判别模式 ----------
    if kind == "auto":
        kind = "single" if n_series == 1 else "group"

    # ========== SINGLE：每年一根柱（常用于差值） ==========
    if kind == "single":
        name = series_names[0]
        info = case_dict[name]
        x = pd.Series(info["x"]).map(lambda v: int(v) if pd.notna(v) else np.nan)
        y = pd.Series(info["y"]).astype(float)

        s = pd.Series(y.values, index=x.values).dropna()
        heights = np.array([s.get(yr, np.nan) for yr in years], dtype=float)
        if fillna_as_zero:
            heights = np.nan_to_num(heights, nan=0.0)

        # 颜色：按正负
        colors = []
        for h in heights:
            if not np.isfinite(h):
                colors.append((0, 0, 0, 0))  # 透明，不画
            else:
                colors.append(pos_color if h >= 0 else neg_color)

        base_pos = np.arange(n_years, dtype=float)
        if bar_width is None:
            bar_width = 0.6

        mask = np.isfinite(heights) if not fillna_as_zero else np.ones_like(heights, dtype=bool)
        if np.any(mask):
            ax.bar(
                base_pos[mask], heights[mask],
                width=bar_width,
                color=np.array(colors, dtype=object)[mask],
                alpha=bar_alpha,
                label=name,
                linewidth=0
            )

        # y 轴（使用你的自动 y 轴函数，已保证包含 0 且留白）
        df_ylim = pd.DataFrame({name: heights}, index=years)
        _set_yAxes(
            ax=ax, in_df=df_ylim,
            tickfontsize=ytickfontsize, labelfontsize=ylabelfontsize,
            ylims=ylims, yticks=yticks, yticklabels=yticklabels, ylabel=ylabel
        )

        # x 轴：稀疏刻度（基于年份顺序）
        ax.set_xlim([base_pos.min() - 0.6, base_pos.max() + 0.6])
        if xticks is None:
            if n_years <= max_xticks:
                xticks = base_pos
                xticklabels = [str(y) for y in years]
            else:
                stride = int(np.ceil(n_years / max_xticks))
                idx = np.arange(0, n_years, stride)
                if force_first_last_tick and (len(idx) == 0 or idx[-1] != n_years - 1):
                    idx = np.r_[idx, n_years - 1]
                xticks = base_pos[idx]
                xticklabels = [str(int(years[i])) for i in idx]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontsize=xtickfontsize, rotation=xtick_rotation)
        ax.set_xlabel(xlabel, fontsize=xlabelfontsize, fontweight='bold', labelpad=15)

        # 样式
        ax.axhline(0.0, color='black', linewidth=1, linestyle='--', alpha=0.8)
        ax.grid(True, axis='y', linestyle=":", alpha=0.9)
        ax.tick_params(axis='x', pad=1, labelsize=xtickfontsize)
        ax.tick_params(axis='y', labelsize=ytickfontsize)
        ax.legend(frameon=False, loc='upper left')
        return ax

    # ========== GROUP：每年多根并排柱（常用于 4 个 case / 四季） ==========
    if season_order is None:
        try:
            season_order = list(TU.get_seasons())  # 例如 ["DJF","MAM","JJA","SON"]
        except Exception:
            season_order = series_names

    if set(series_names).issubset(set(season_order)):
        ordered_series = [s for s in season_order if s in series_names]
    else:
        ordered_series = series_names

    default_palette = ["#4575b4", "#91bfdb", "#fdae61", "#abdda4", "#984ea3", "#ff7f00"]
    if color_map is None:
        color_map = {}
        for i, sname in enumerate(ordered_series):
            color_map[sname] = case_dict.get(sname, {}).get('color', default_palette[i % len(default_palette)])

    base_pos = np.arange(n_years, dtype=float)
    if bar_width is None:
        bar_width = min(0.8 / max(len(ordered_series), 1), 0.28)  # 组宽 0.8 内均分

    df_for_ylim = {}
    for idx, name in enumerate(ordered_series):
        info = case_dict[name]
        color = info.get('color', color_map.get(name, default_palette[idx % len(default_palette)]))
        x = pd.Series(info['x']).map(lambda v: int(v) if pd.notna(v) else np.nan)
        y = pd.Series(info['y']).astype(float)

        s = pd.Series(y.values, index=x.values).dropna()
        heights = np.array([s.get(yr, np.nan) for yr in years], dtype=float)
        if fillna_as_zero:
            heights = np.nan_to_num(heights, nan=0.0)
            mask = np.ones_like(heights, dtype=bool)
        else:
            mask = np.isfinite(heights)

        df_for_ylim[name] = heights

        offset = (idx - (len(ordered_series) - 1) / 2) * bar_width
        pos = base_pos + offset

        if np.any(mask):
            ax.bar(
                pos[mask], heights[mask],
                width=bar_width,
                color=color,
                alpha=bar_alpha,
                label=name,
                linewidth=0
            )

    # y 轴
    df_ylim = pd.DataFrame(df_for_ylim, index=years)
    _set_yAxes(
        ax=ax, in_df=df_ylim,
        tickfontsize=ytickfontsize, labelfontsize=ylabelfontsize,
        ylims=ylims, yticks=yticks, yticklabels=yticklabels, ylabel=ylabel
    )

    # x 轴：稀疏刻度
    ax.set_xlim([base_pos.min() - 0.6, base_pos.max() + 0.6])
    if xticks is None:
        if n_years <= max_xticks:
            xticks = base_pos
            xticklabels = [str(y) for y in years]
        else:
            stride = int(np.ceil(n_years / max_xticks))
            idx = np.arange(0, n_years, stride)
            if force_first_last_tick and (len(idx) == 0 or idx[-1] != n_years - 1):
                idx = np.r_[idx, n_years - 1]
            xticks = base_pos[idx]
            xticklabels = [str(int(years[i])) for i in idx]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=xtickfontsize, rotation=xtick_rotation)
    ax.set_xlabel(xlabel, fontsize=xlabelfontsize, fontweight='bold', labelpad=15)

    # 样式
    ax.axhline(0.0, color='black', linewidth=1, linestyle='--', alpha=0.8)
    ax.grid(True, axis='y', linestyle=":", alpha=0.9)
    ax.tick_params(axis='x', pad=1, labelsize=xtickfontsize)
    ax.tick_params(axis='y', labelsize=ytickfontsize)
    ax.legend(frameon=False, ncols=min(len(ordered_series), 4), loc='upper left')
    return ax



def plot_stacked_bar_discrete_cmap(
    df: pd.DataFrame,
    levels: List[float],
    cmap: Union[str] = "RdBu_r",
    savepath: Optional[str] = None,
    x_col: Optional[str] = None,
    hueorder: Optional[List[str]] = None,
    xorder: Optional[List[str]] = None,
    *,
    ax=None,
    figsize=(4, 4),
    bar_width=0.72,
    edgecolor="white",
    linewidth=0.4,
    nan_color="lightgray",
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    xtick_rotation=45,
    legend=False,
    legend_title="",
):
    """
    绘制堆叠柱状图（stacked bar），并使用 levels + cmap 对“每段柱子”按数值分档着色。
    美化点：移除上右边框，X/Y轴添加箭头。
    
    参数
    ----
    ... (参数说明同上)
    """

    if df is None or len(df) == 0:
        raise ValueError("df 为空。")

    # 1) 自动识别 x_col
    if x_col is None:
        non_num_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        if not non_num_cols:
            raise ValueError("未找到非数值列作为省份名称列，请显式传入 x_col。")
        x_col = non_num_cols[0]

    if x_col not in df.columns:
        raise ValueError(f"x_col='{x_col}' 不在 df.columns 中。")

    # 2) 数值列（用于堆叠）
    value_cols = [c for c in df.columns if c != x_col and pd.api.types.is_numeric_dtype(df[c])]
    if not value_cols:
        raise ValueError("df 中未找到可用于堆叠的数值列（除 x_col 外的 numeric 列）。")

    if hueorder is not None:
        miss = [c for c in hueorder if c not in value_cols]
        if miss:
            raise ValueError(f"hueorder 中存在不在数值列里的列：{miss}")
        value_cols = [c for c in hueorder if c in value_cols]

    # 3) x 轴顺序
    plot_df = df.copy()
    if xorder is not None:
        xorder_exist = [x for x in xorder if x in set(plot_df[x_col].tolist())]
        plot_df = plot_df.set_index(x_col).reindex(xorder_exist).reset_index()
    else:
        pass

    x_labels = plot_df[x_col].astype(str).tolist()
    n = len(x_labels)
    x = np.arange(n)

    # 4) 颜色分档：levels -> N 个 bin
    levels = np.asarray(levels, dtype=float)
    if levels.ndim != 1 or levels.size < 2:
        raise ValueError("levels 至少需要两个边界值（长度>=2）。")
    if np.any(np.diff(levels) <= 0):
        raise ValueError("levels 必须严格递增（例如 [-5,-2,0,2,5]）。")

    n_bins = levels.size - 1
    cmap_obj = plt.get_cmap(cmap, n_bins) # 离散化

    def value_to_color(v):
        """把单个数值映射到离散颜色（NaN 用 nan_color）。"""
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return nan_color
        idx = np.digitize([v], levels, right=False)[0] - 1
        idx = int(np.clip(idx, 0, n_bins - 1))
        return cmap_obj((idx + 0.5) / n_bins)

    # 5) 准备画布
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    bottom_pos = np.zeros(n, dtype=float)
    bottom_neg = np.zeros(n, dtype=float)

    # 6) 逐列堆叠
    for col in value_cols:
        vals = plot_df[col].to_numpy(dtype=float)

        colors = [value_to_color(v) for v in vals]

        vals_pos = np.where(np.isnan(vals), 0.0, np.clip(vals, 0, np.inf))
        vals_neg = np.where(np.isnan(vals), 0.0, np.clip(vals, -np.inf, 0))

        if np.any(vals_pos != 0):
            ax.bar(
                x, vals_pos, bottom=bottom_pos, width=bar_width,
                color=colors, edgecolor=edgecolor, linewidth=linewidth
            )
            bottom_pos += vals_pos

        if np.any(vals_neg != 0):
            ax.bar(
                x, vals_neg, bottom=bottom_neg, width=bar_width,
                color=colors, edgecolor=edgecolor, linewidth=linewidth
            )
            bottom_neg += vals_neg

   # 7) 轴样式 - 美化修改区域
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=xtick_rotation, ha="right", va="top", fontsize=10)
    ax.tick_params(axis='y', labelsize=10)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    ax.axhline(0, linewidth=0.8)

    # --- 开始美化 ---
    
    # 移除右侧和上侧边框
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # X 轴箭头 (保持不变，因为 X 轴需要指向右侧)
    # X 轴箭头 (0.99, 0) 是一个常用的相对位置
    ax.plot(
        1.0, 0,
        marker='>', 
        transform=ax.get_yaxis_transform(), 
        color='black', 
        clip_on=False, 
        markeredgecolor='black', 
        markersize=5
    )
    
    # Y 轴箭头 (修正 X 坐标问题)
    # 使用 ax.transAxes 将箭头固定在绘图区域的左上角 (0, 1.0)
    ax.plot(
        0, 1.0, 
        marker='^', 
        transform=ax.transAxes, # ***关键修正：使用 ax.transAxes***
        color='black', 
        clip_on=False, 
        markeredgecolor='black', 
        markersize=5
    )
    # --- 结束美化 ---

    # 8) legend：显示 levels 分档
    if legend:
        handles = []
        for i in range(n_bins):
            left, right = levels[i], levels[i + 1]
            lab = f"[{left:g}, {right:g})" if i < n_bins - 1 else f"[{left:g}, {right:g}]"
            handles.append(Patch(facecolor=cmap_obj((i + 0.5) / n_bins), edgecolor="none", label=lab))
        handles.append(Patch(facecolor=nan_color, edgecolor="none", label="NaN"))
        ax.legend(handles=handles, title=legend_title, frameon=False, ncol=min(6, n_bins + 1))

    fig.tight_layout()

    # 9) 保存
    if savepath is not None:
        fig.savefig(savepath, dpi=DPI, bbox_inches="tight") # 使用定义的 DPI 变量

    return fig, ax


