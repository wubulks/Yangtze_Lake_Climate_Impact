import os
import time
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Literal, Dict, List
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
import ToolBoxes.Tool_PlotConfig as TPC
from matplotlib import font_manager as fm
FIGFMT = TPC.FIGFMT
DPI    = TPC.DPI_medium

mpl.use('Agg')  # 不显示图，只保存
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
ThetaZeroLoc = Literal['N','NE','E','SE','S','SW','W','NW']   # 0点位置
ThetaDirect = Literal[1, -1]  # -1=Theta 顺时针方向增大, 1=Theta 沿逆时针方向增大
dejavu_bold_prop = fm.FontProperties(
    family="DejaVu Sans",
    size=15,
    weight="bold"
)


def _to_radius(series: float | np.ndarray | List[float], rmin: float) -> float | np.ndarray:
    """
    将半径数据平移到以 rmin 为原点的显示半径：
    返回类型与输入“对应”：
      - 输入 float -> 返回 float
      - 输入 np.ndarray -> 返回 np.ndarray
      - 输入 list[float] -> 返回 np.ndarray
    """
    if np.isscalar(series):  # float 或 0-d ndarray
        return float(series) - float(rmin)
    if isinstance(series, np.ndarray):
        # 就地不拷贝 dtype 转换（若已是 float）
        return series.astype(float, copy=False) - float(rmin)
    if isinstance(series, list):
        return np.asarray(series, dtype=float) - float(rmin)
    # 兜底：支持 pandas/xarray 的 .to_numpy() 情况
    try:
        return (series.astype(float) - float(rmin)).to_numpy()
    except Exception:
        return np.asarray(series, dtype=float) - float(rmin)



def _deg2rad(angle_deg: float | np.ndarray | List[float]) -> float | np.ndarray:
    """
    将角度（degree）转换为弧度（radian）。
    - 输入 float -> 返回 float
    - 输入 np.ndarray -> 返回 np.ndarray（同形）
    - 输入 list[float] -> 返回 np.ndarray
    等价于 numpy.deg2rad，但对类型作了更明确的处理。
    """
    if np.isscalar(angle_deg):  # float 或 0-d ndarray
        return float(angle_deg) * (np.pi / 180.0)
    if isinstance(angle_deg, np.ndarray):
        return np.deg2rad(angle_deg.astype(float, copy=False))
    if isinstance(angle_deg, list):
        return np.deg2rad(np.asarray(angle_deg, dtype=float))
    # 兜底：兼容 pandas/xarray 等带 astype/to_numpy 的对象
    try:
        return np.deg2rad(angle_deg.astype(float).to_numpy())
    except Exception:
        return np.deg2rad(np.asarray(angle_deg, dtype=float))



def _rad2deg(angle_rad: float | np.ndarray | List[float]) -> float | np.ndarray:
    """
    将弧度（radian）转换为角度（degree）。
    - 输入 float -> 返回 float
    - 输入 np.ndarray -> 返回 np.ndarray
    - 输入 list[float] -> 返回 np.ndarray
    """
    if np.isscalar(angle_rad):  # float 或 0-d ndarray
        return float(angle_rad) * (180.0 / np.pi)
    if isinstance(angle_rad, np.ndarray):
        return np.rad2deg(angle_rad.astype(float, copy=False))
    if isinstance(angle_rad, list):
        return np.rad2deg(np.asarray(angle_rad, dtype=float))
    # 兜底：兼容 pandas/xarray 等带 astype/to_numpy 的对象
    try:
        return np.rad2deg(angle_rad.astype(float).to_numpy())
    except Exception:
        return np.rad2deg(np.asarray(angle_rad, dtype=float))



def check_proj(ax: Axes) -> None:
    """保证是极坐标"""
    if getattr(ax, 'name', '') != 'polar':
        raise TypeError("ax 需为极坐标轴：fig.add_subplot(..., projection='polar')")


def set_direction(ax:Axes, zero_loc: ThetaZeroLoc, direct: ThetaDirect) -> Axes:
    """极坐标方向设置"""
    ax.set_theta_zero_location(zero_loc)   
    ax.set_theta_direction(direct)   
    return ax      



def divide_rose_to_equal_parts_rad(num: int, total_arc_deg: float = 2*np.pi) -> List[np.ndarray]:
    """平分圆盘"""
    total_arc_rad = _deg2rad(total_arc_deg)
    part_arr = np.arange(num)
    theta = (part_arr / num) * total_arc_rad
    theta_closed = np.r_[theta, theta[0]]
    return theta, theta_closed



def divide_rose_to_equal_parts_deg(num: int, total_arc_deg: float = 2*np.pi) -> List[np.ndarray]:
    """平分圆盘"""
    part_arr = np.arange(num)
    theta = (part_arr / num) * total_arc_deg
    theta_closed = np.r_[theta, theta[0]]
    return theta, theta_closed



def set_rose_theta_extent(ax: Axes, min_deg: float, max_deg: float) -> Axes:
    """设置圆盘角度范围，例如半圆/整圆, 度数制"""
    if not np.isfinite(min_deg) or not np.isfinite(max_deg):
        raise ValueError("min_deg/max_deg 必须为有限数")
    if max_deg <= min_deg:
        min_deg, max_deg = max_deg, min_deg
    thetadiff = np.abs(max_deg - min_deg)
    if thetadiff > 360:
        raise ValueError(f"max_deg - min_deg is {thetadiff}, is grreater than 360.")
    ax.set_thetalim(thetamin=min_deg, thetamax=max_deg)
    return ax



def set_rose_radial_extent(ax: Axes, rmin: float, rmax: float) -> Axes:
    """设置纵轴范围"""
    if not np.isfinite(rmin) or not np.isfinite(rmax):
        raise ValueError("rmin/rmax 必须为有限数")
    if rmax <= rmin:
        rmin, rmax = rmax, rmin
    ax.set_rlim(bottom=rmin, top=rmax)
    return ax



def get_ring_height(in_df: pd .DataFrame, rmax: float, case_dict: Dict[str, str], ring_keys: List[str] = None,
                     height_ratio: float = 0.04, gap_ratio: float = 0.015, avail_ratio: float = 0.15) -> List[float]:
    """计算圆环厚度和间隔厚度"""
    if (ring_keys) and (ring_keys is not None):
        for key in ring_keys:
            if key not in in_df.columns:
                raise ValueError("the key name for plot ring is not in the in_df. plz check data")
    n = len(ring_keys)
    total_needed = n*height_ratio + (n-1)*gap_ratio
    if total_needed > avail_ratio:
        scale = avail_ratio / total_needed
        ring_h_ratio = height_ratio * scale
        ring_gap_ratio = gap_ratio * scale
    else:
        ring_h_ratio = height_ratio
        ring_gap_ratio = gap_ratio
    ring_height = rmax * ring_h_ratio
    ring_gap    = rmax * ring_gap_ratio
    return ring_height, ring_gap




def Plot_Yangtze_Diurnal_Rose(ax: Axes, in_df: pd.DataFrame, rlims: List[float], rlevs: np.ndarray,
                              case_dict: Dict[str, str], tick_interval: int = 1, ring_keys: List[str] = None,
                              nan_gap_frac: float = 0.28):
    """绘制昼夜分析玫瑰图（外环在 NaN 处真的断开）"""

    # 检查坐标系，必须是极坐标
    check_proj(ax=ax)

    # 设置极坐标方向
    ax = set_direction(ax=ax, zero_loc='S', direct=-1)  # 0 弧度在南(底部), 顺时针增长度数

    # 设置圆盘角度范围
    ax = set_rose_theta_extent(ax=ax, min_deg=0, max_deg=360)

    # 设置圆盘纵轴范围
    rmin, rmax = np.asarray(rlims).min(), np.asarray(rlims).max()
    ax = set_rose_radial_extent(ax=ax, rmin=rmin, rmax=rmax)

    # 计算角度数组
    theta, theta_closed = divide_rose_to_equal_parts_rad(num=24, total_arc_deg=360)

    # 添加日/夜半圆底色（置底） 
    R = rmax - rmin
    th_night = np.linspace(-np.pi/2,  np.pi/2, 361)
    th_day   = np.linspace( np.pi/2, 3*np.pi/2, 361)
    ax.fill_between(th_day,   0, R, color="#fff59d", alpha=0.14, zorder=0)  # 日
    ax.fill_between(th_night, 0, R, color="#939391", alpha=0.14, zorder=0)  # 夜
    ax.plot([ np.pi/2,  np.pi/2], [0, R], linewidth=0.8, alpha=0.35, zorder=1)
    ax.plot([3*np.pi/2, 3*np.pi/2], [0, R], linewidth=0.8, alpha=0.35, zorder=1)
    ax.text(-np.pi/2+(-np.pi/90), R*3.5/5, "Day",   ha='center', va='bottom', fontsize=20, fontweight='bold')
    ax.text(-np.pi/2-(-np.pi/90), R*3.5/5, "Night", ha='center', va='top',    fontsize=20, fontweight='bold')

    # 原点
    r0 = -rmin

    # 绘制曲线 - 修改为分段绘制，在 NaN 处断开
    legend_handles = []
    for key, (color, marker) in case_dict.items():
        if key not in in_df.columns:
            continue
        # 半径与闭合
        r = _to_radius(in_df[key], rmin)   # r = y - rmin
        r_closed = np.r_[r, r[0]]
        
        # 检测 NaN 位置并分段绘制
        valid = np.isfinite(r_closed)
        
        # 如果全为有效数据，直接绘制
        if np.all(valid):
            ax.plot(theta_closed, r_closed, color=color, marker=marker, 
                   linewidth=1.5, markersize=8, zorder=6, label=key)
        else:
            # 分段绘制：找到连续的有效数据段
            segments = []
            current_segment = []
            
            for i in range(len(theta_closed)):
                if valid[i]:
                    current_segment.append((theta_closed[i], r_closed[i]))
                else:
                    if current_segment:  # 结束当前段
                        segments.append(np.array(current_segment).T)  # 转置为 (x, y) 格式
                        current_segment = []
            
            # 处理最后一个段
            if current_segment:
                segments.append(np.array(current_segment).T)
            
            # 绘制所有段
            for i, segment in enumerate(segments):
                if i == 0:
                    # 只在第一段添加标签
                    ax.plot(segment[0], segment[1], color=color, marker=marker,
                           linewidth=3.5, markersize=10, zorder=6, label=key)
                else:
                    # 后续段不添加标签避免重复
                    ax.plot(segment[0], segment[1], color=color, marker=marker,
                           linewidth=3.5, markersize=10, zorder=6)
        # # 给每条线留一个图例句柄
        # legend_handles.append(Line2D([0],[0], color=color, lw=1.5, marker=marker, label=key))


    # 绘制圆环指示"正/负指示环"（在 NaN 处完全跳过绘制）
    # ----------------------------------------------------------------------
    if ring_keys is None:
        ring_keys = []
    ring_height, ring_gap = get_ring_height(in_df=in_df, rmax=rmax, case_dict=case_dict, ring_keys=ring_keys,
                                            height_ratio=0.08, gap_ratio=0.015, avail_ratio=0.18)
    width = 2*np.pi/24
    neg_color_default = "#42a5f5"  # 负值统一用蓝色（可改）
    pos_color_default = "#f38181"  # 正值统一用红色（可改）
    bottom = R  # 从外向内 
    
    for idx, key in enumerate(ring_keys):
        if key not in in_df.columns:
            continue

        y = in_df[key].to_numpy(dtype=float)
        n = len(y)
        if n == 0:
            continue

        pos_color = pos_color_default
        neg_color = neg_color_default

        if idx == 0: 
            bottom = R - ring_height
        else:
            bottom -= (ring_height + ring_gap)

        # 有效性（Finite 即非 NaN、非 inf/-inf）
        valid = np.isfinite(y)

        # 若全无效，跳过该环
        if not np.any(valid):
            continue

        # 方法1：直接跳过所有 NaN 时段
        for h in range(n):
            if not valid[h]:
                # 完全跳过 NaN 时段，不绘制任何条带
                continue

            # 只在有效数据处绘制完整宽度的条带
            start_rad = theta[h] - width/2
            col = pos_color if (y[h] >= 0) else neg_color
            ax.bar(start_rad, height=ring_height, width=width, bottom=bottom,
                   color=col, edgecolor="none", align="edge", alpha=0.9,
                   zorder=1, clip_on=False)

        # 图例：该环的"≥ 0"说明
        legend_handles.append(
            Line2D([0],[0],
                color=pos_color, lw=6,
                label="≥ 0",)
        )
                

    # 额外补充"负值<0"的图例项
    legend_handles.append(
        Line2D([0],[0], color=neg_color_default, lw=6, label=r"< 0"),
    )
    # ----------------------------------------------------------------------

    # 若收集了句柄，统一设置图例
    if legend_handles:
        legend = ax.legend(
            handles=legend_handles,
            bbox_to_anchor=(1.1, 1.1),
            loc="upper right",
            framealpha=0.6,
            prop=dejavu_bold_prop,
        )
    
    # 角向刻度（小时）
    tick_hours  = list(range(0, 24, tick_interval))
    tick_thetas = [((h) / 24.0) * 2 * np.pi for h in tick_hours]
    ax.set_xticks(tick_thetas)
    ax.set_xticklabels([str(h) for h in tick_hours], fontsize=21)

    # 径向范围与“圈”（原值刻度） 
    ax.set_rlim(0, R)
    rlevs = np.asarray(list(rlevs), float)
    rlevs = rlevs[np.isfinite(rlevs)]
    rlevs = rlevs[(rlevs >= rmin) & (rlevs <= rmax)]
    if rlevs.size:
        rticks = rlevs - rmin
        ax.set_yticks(rticks)
        ax.set_yticklabels([f"\n{c:.3g}" for c in rlevs], fontsize=21)
    ax.set_rlabel_position(180)
    for t in ax.get_yticklabels():
        t.set_ha('center'); t.set_va('center')
    ax.tick_params(axis='y', labelsize=21, zorder=20)
    ax.tick_params(axis='x', pad=5, labelsize=21, zorder=20)

    # r=0 的参考圆
    ax.axhline(r0, color='black', linewidth=2.5, linestyle='--')

    # 恢复整圈 + 网格
    ax.set_thetamin(0); ax.set_thetamax(360)
    ax.grid(True, linestyle=":", alpha=0.9, zorder=5, linewidth=1.5)

    return ax



def plot_regclimimpact_diurnal_rose(in_df, varname, season, target, checkmethod, FigOutDir, rosecfg, varinfo, suffix):
    unit      = varinfo.unit
    longname  = varinfo.longname
    abbr      = varinfo.abbr
    in_dict = rosecfg.colors_dict
    levs = rosecfg.roselevs[0]
    nlevs = rosecfg.roselevs[1]
    circles = np.linspace(levs[0], levs[-1], nlevs)
    fig_rose, rose_ax = plt.subplots(figsize=(6, 6), layout='constrained', subplot_kw={'projection': 'polar'})
    # 只取显著格点（可选）
    rose_ax = Plot_Yangtze_Diurnal_Rose(ax=rose_ax, in_df=in_df, rlims=[levs[0], levs[-1]], rlevs=circles, case_dict=in_dict, ring_keys=["Strong"])
    # if season == 'DJF':
    #     rose_ax.set_ylabel(f"{longname} ({unit})", fontsize=12, fontweight='bold', labelpad=15)
    rose_ax.set_title(f"{season}", fontsize=18, fontweight='bold', pad=12)
    rose_ax.set_position([0.1, 0.1, 0.8, 0.8])
    savepath = f'{FigOutDir}/{target}_Diff_Rose_{varname}_{season}_{suffix}_{checkmethod}.{FIGFMT}'
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    fig_rose.savefig(savepath, dpi=DPI, bbox_inches='tight')
    plt.close(fig_rose)
