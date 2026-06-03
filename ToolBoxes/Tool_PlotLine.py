import os
import time
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import cartopy.crs as ccrs
from matplotlib.axes import Axes
from cartopy.io import shapereader
import cartopy.feature as cfeature
from matplotlib.lines import Line2D
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib import colors as mcolors
from cnmaps import get_adm_maps, draw_maps
from matplotlib.colorbar import ColorbarBase
from typing import Literal, Optional, Tuple, Dict, Any, Sequence, List, Union

# 自定义工具箱
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
    自动设置y轴范围：
    如果传入了ylims, 则按传入值进行配置，如果没有传入, 则根据in_df计算。
    """
    
    # 设置y轴范围
    if ylims is not None:
        ylims = np.asarray(ylims)
        ymin = np.nanmin(ylims)
        ymax = np.nanmax(ylims)
    else:
        max_val = []
        min_val = []
        for key in in_df.columns:
            max_val.append(in_df[key].max())
            min_val.append(in_df[key].min())
        ymax = np.nanmax(np.asarray(max_val))
        ymin = np.nanmin(np.asarray(min_val))
    ax.set_ylim([ymin, ymax])

    # 设置y轴标签
    if yticks is None:
        yticks = np.linspace(ymin, ymax, 5)
    ax.set_yticks(yticks)

    if yticklabels is None:
        # 根据步长决定小数位，避免 0.05 显示成 0.1 导致重复
        step = yticks[1] - yticks[0]
        if step >= 1:
            fmt = "{:.0f}"
        elif step >= 0.1:
            fmt = "{:.1f}"
        elif step >= 0.01:
            fmt = "{:.2f}"
        else:
            fmt = "{:.3f}"
        ax.set_yticklabels([fmt.format(v) for v in yticks], fontsize=tickfontsize)
    else:
        ax.set_yticklabels(yticklabels, fontsize=tickfontsize)
    ax.set_ylabel(ylabel, fontsize=labelfontsize, fontweight='bold', labelpad=15)



    

def _set_xAxes(ax: Axes, in_df: pd.DataFrame, tickfontsize: float = 11, labelfontsize: float = 11,
               xlims: List[float] | None = None, xticks: List[float] = None, xticklabels: List[str] | None = None,
               xlabel: str = '') -> Axes:
    """
    自动设置y轴范围：
    如果传入了ylims, 则按传入值进行配置，如果没有传入, 则根据in_df计算。
    """
    try:
        x = in_df.index.astype(int).to_numpy()
    except Exception:
        def _to_int_safe(v):
            try: return int(v)
            except Exception: return np.nan
        x = pd.Series(in_df.index).map(_to_int_safe).to_numpy()
        keep = np.isfinite(x)
        in_df = in_df.loc[keep]
        x = x[keep].astype(int)

    # 设置y轴范围
    if xlims is not None:
        xlims = np.asarray(xlims)
        xmin = np.nanmin(xlims)
        xmax = np.nanmax(xlims)
    else:
        xmax = np.nanmax(np.asarray(x))
        xmin = np.nanmin(np.asarray(x))
    ax.set_xlim([xmin, xmax])

    # 设置y轴标签
    if xticks is None:
        xticks = np.linspace(xmin, xmax, 5)
    ax.set_xticks(xticks)

    if xticklabels is None:
        # 根据步长决定小数位，避免 0.05 显示成 0.1 导致重复
        step = xticks[1] - xticks[0]
        if step >= 1:
            fmt = "{:.0f}"
        elif step >= 0.1:
            fmt = "{:.1f}"
        elif step >= 0.01:
            fmt = "{:.2f}"
        else:
            fmt = "{:.3f}"
        ax.set_xticklabels([fmt.format(v) for v in xticks], fontsize=tickfontsize)
    else:
        ax.set_xticklabels(xticklabels, fontsize=tickfontsize)
    ax.set_xlabel(xlabel, fontsize=labelfontsize, fontweight='bold', labelpad=15)

    


def Plot_Yangtze_Diurnal_Line(ax: Axes, in_df: pd.DataFrame, case_dict: Dict[str,str], xlims: List[float] | None = None,
                     ylims: List[float] | None = None, xticks: List[float] = None, yticks: List[float] = None,
                     xticklabels: List[str] | None = None, yticklabels: List[str] | None = None, xtickfontsize: float = 11, ytickfontsize: float = 11,
                     xlabel: str = '', ylabel: str = '', xlabelfontsize: float = 11, ylabelfontsize: float = 11,) -> Axes:
    """
    24 小时折线图（笛卡尔坐标）：
    - keys_dict：{列名: [颜色, marker]}
    - fill=True：对非 NaN 段相对 0 填充
    """

    # 只绘制 case_dict 中存在的列
    cols = [k for k in case_dict.keys() if k in in_df.columns]
    if not cols:
        raise ValueError("case_dict 中没有找到可用的列名。")
    df = in_df[cols].copy()

    # ---------- 逐列画线（NaN 断线） ----------
    # 获取x轴坐标
    try:
        x = in_df.index.astype(int).to_numpy()
    except Exception:
        def _to_int_safe(v):
            try: return int(v)
            except Exception: return np.nan
        x = pd.Series(in_df.index).map(_to_int_safe).to_numpy()
        keep = np.isfinite(x)
        in_df = in_df.loc[keep]
        x = x[keep].astype(int)

    for key, style in case_dict.items():
        if key not in df.columns:
            continue
        color  = style[0] if len(style) > 0 else None
        marker = style[1] if len(style) > 1 else None

        y = df[key].to_numpy(dtype=float)
        y_masked = np.ma.masked_invalid(y)  # NaN 断线

        ax.plot(
            x, y_masked,
            color=color, marker=marker, linewidth=1.5, markersize=4,
            label=key
        )

    # 配置x和y轴
    _set_xAxes(ax=ax, in_df=df, tickfontsize=xtickfontsize, labelfontsize=xlabelfontsize,xlims=xlims,xticks=xticks, xticklabels=xticklabels, xlabel=xlabel)
    _set_yAxes(ax=ax, in_df=df, tickfontsize=ytickfontsize, labelfontsize=ylabelfontsize,ylims=ylims,yticks=yticks, yticklabels=yticklabels, ylabel=ylabel)

    # 网格与样式
    ax.grid(True, linestyle=":", alpha=0.9)
    ax.tick_params(axis='x', pad=1, labelsize=10)
    ax.tick_params(axis='y', labelsize=11)

    # 参考基线：y=0
    ax.axhline(0.0, color='black', linewidth=1, linestyle='--', alpha=0.8)

    return ax
    


def Plot_Yangtze_Serial_Comparation_Line(ax: Axes, case_dict: Dict[str,str], xlims: List[float] | None = None,
                     ylims: List[float] | None = None, xticks: List[float] = None, yticks: List[float] = None,
                     xticklabels: List[str] | None = None, yticklabels: List[str] | None = None, xtickfontsize: float = 11, ytickfontsize: float = 11,
                     xlabel: str = '', ylabel: str = '', xlabelfontsize: float = 11, ylabelfontsize: float = 11,) -> Axes:
    """
    序列对比折线图（笛卡尔坐标）：
    - keys_dict：{列名: [颜色, marker]}
    - fill=True：对非 NaN 段相对 0 填充
    """

    # ---------- 逐列画线（NaN 断线） ----------
    # 获取x轴坐标
    df_ = {}
    for key, info in case_dict.items():
        color = info['color']
        marker = info['marker']
        y = info['y']
        x = info['x']
        y_masked = np.ma.masked_invalid(y)  # NaN 断线
        ax.plot(
            x, y_masked,
            color=color, marker=marker, linewidth=1.5, markersize=4,
            label=key
        )
        df_[key] = pd.Series(y, index=x)
    df =  pd.DataFrame(df_)

    # 配置x和y轴
    _set_xAxes(ax=ax, in_df=df, tickfontsize=xtickfontsize, labelfontsize=xlabelfontsize,xlims=xlims,xticks=xticks, xticklabels=xticklabels, xlabel=xlabel)
    _set_yAxes(ax=ax, in_df=df, tickfontsize=ytickfontsize, labelfontsize=ylabelfontsize,ylims=ylims,yticks=yticks, yticklabels=yticklabels, ylabel=ylabel)

    # 网格与样式
    ax.grid(True, linestyle=":", alpha=0.9)
    ax.tick_params(axis='x', pad=1, labelsize=10)
    ax.tick_params(axis='y', labelsize=11)

    # 参考基线：y=0
    ax.axhline(0.0, color='black', linewidth=1, linestyle='--', alpha=0.8)

    return ax
