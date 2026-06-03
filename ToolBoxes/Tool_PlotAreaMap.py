import os
import time
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.patheffects as pe
import matplotlib as mpl
import cartopy.crs as ccrs
import geopandas as gpd             # 确保在文件开头有这一行
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from cartopy.io import shapereader
import cartopy.feature as cfeature
from typing import Any, List, Union, Dict
from cnmaps import get_adm_maps, draw_maps
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import colors as mcolors
import matplotlib.patches as patches
from cartopy.mpl.geoaxes import GeoAxes
AxesLike = Union[Axes, GeoAxes]
import mapclassify as mc
import matplotlib.cm as cm
import gc

# 自定义工具箱
import ToolBoxes.Tool_DataPrepare as TDP
import ToolBoxes.Tool_PlotConfig as TPC
import ToolBoxes.Tool_YangtzeColorMap as TYCM
FIGFMT = TPC.FIGFMT
DPI    = TPC.DPI_medium
BASEDATA = TPC.BASEDATA

mpl.use('Agg')  # 不显示图，只保存
# 设置全局默认参数
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Noto Sans', 'Arial', 'DejaVu Sans']
# mpl.rcParams['mathtext.fontset'] = 'custom'
# mpl.rcParams['mathtext.default'] = 'regular'
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.it'] = 'Noto Sans:bold:italic'
mpl.rcParams['mathtext.bf'] = 'Noto Sans:bold'
mpl.rcParams['mathtext.default'] = 'rm'
warnings.filterwarnings("ignore", category=RuntimeWarning)
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# =========================== 函数区 ===========================
def is_dark(color, threshold=0.2) -> bool:
    """
    计算颜色的相对亮度（0~1），越大越亮。
    支持 '#RRGGBB' / 'tab:blue' / (r,g,b) 等 matplotlib 可识别颜色。
    依据 WCAG 的 sRGB -> linear 转换与 luminance 公式。
    亮度 < threshold 认为是深色
    """
    r, g, b = mcolors.to_rgb(color)  # -> 0~1

    def to_linear(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r_lin, g_lin, b_lin = map(to_linear, (r, g, b))
    # WCAG relative luminance

    light = 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin

    return light < threshold


def _angle_from_long_axis_deg(geom) -> float:
    """
    用最小外接旋转矩形的最长边方向作为角度（度）。
    返回角度被限制在 [-45, 45]。
    """
    if geom is None or geom.is_empty:
        return 0.0

    # MultiPolygon 取面积最大的那块
    if geom.geom_type == "MultiPolygon":
        geom = max(list(geom.geoms), key=lambda g: g.area)

    mrr = geom.minimum_rotated_rectangle
    coords = np.asarray(mrr.exterior.coords)  # 5x2，首尾重复

    vecs = coords[1:] - coords[:-1]           # 4 条边向量
    lens = np.hypot(vecs[:, 0], vecs[:, 1])
    i = int(np.argmax(lens))                  # 最长边
    dx, dy = vecs[i]

    ang = np.degrees(np.arctan2(dy, dx))      # [-180, 180]
    # 归一到 [-90, 90]（方向等价）
    ang = (ang + 90) % 180 - 90

    # 你希望 -45~0：强制为负 & 截断
    if ang < 0:
        ang = -ang
    ang = max(25.0, min(0.0, float(ang)))

    return float(ang)



def set_map_extent(ax: AxesLike, lon_range: List[float], lat_range: List[float]) -> AxesLike:
    """Set the map extent (bounding box) based on longitude and latitude ranges."""
    # Create ranges for longitude and latitude
    lon_min, lon_max = np.min(lon_range), np.max(lon_range)
    lat_min, lat_max = np.min(lat_range), np.max(lat_range)
    # Define the extent to set
    ranges = [lon_min, lon_max, lat_min, lat_max]
    # Set the extent, using PlateCarree projection for lon/lat
    ax.set_extent(ranges, crs=ccrs.PlateCarree())
    return ax



def add_cnmaps(ax: AxesLike, level: str, color: str = 'grey', 
               zorder: int = 99, linewidth: float = 0.4) -> AxesLike:
    """在地图上添加中国行政区划边界。"""

    if level == '国':
        maps = get_adm_maps(country='中华人民共和国')
    elif level == '省':
        maps = get_adm_maps(level='省')
    elif level == '市':
        maps = get_adm_maps(level='市')
    elif level == '区县':
        maps = get_adm_maps(level='区县')
    draw_maps(maps, ax=ax, color=color, zorder=zorder, linewidth=linewidth)
    return ax



def mask_ocean(ax: AxesLike, facecolor: str = 'auto', 
               edgecolor: str = 'auto', zorder: int = 105) -> None:
    """掩模海洋区域"""

    facecolor = ax.get_facecolor() if facecolor == 'auto' else facecolor
    edgecolor = 'none' if edgecolor == 'auto' else edgecolor
    ax.add_feature(
        cfeature.OCEAN,
        facecolor=facecolor,
        edgecolor=edgecolor,
        zorder=zorder
    )
    return ax



def add_shapfile(ax: AxesLike, shp_path: str, edgecolor: str = "black", facecolor: str = "none",
                 linewidth: float = 0.5, zorder: int = 130) -> AxesLike:
    """添加特定的矢量区域"""
    gdf = gpd.read_file(shp_path)
    # 强制覆盖 CRS 为 WGS84（不会重投影，只是“贴标签/覆盖标签”）
    gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    source_crs = ccrs.PlateCarree()

    ax.add_geometries(
        gdf.geometry,
        crs=source_crs,
        edgecolor=edgecolor,
        facecolor=facecolor,
        linewidth=linewidth,
        zorder=zorder
    )
    return ax



def plot_contourf_map_lambert(ax: AxesLike, data_in: np.ndarray, lon2d: np.ndarray, lat2d: np.ndarray,
                              levels: List[float], cmap: Any, zorder: int) -> AxesLike:
    """绘制单个等值线填色图，使用 Lambert 投影。"""
    pcm = ax.contourf(lon2d, lat2d, data_in, levels=levels, cmap=cmap,
                  transform=ccrs.PlateCarree(), extend='both', zorder=zorder)
    return ax, pcm


def plot_contour_map_lambert(ax: AxesLike, data_in: np.ndarray, lon2d: np.ndarray, lat2d: np.ndarray,
                              levels: List[float], colors: Any, linewidths: float, linestyles: str, zorder: int) -> AxesLike:
    """绘制单个等值线，使用 Lambert 投影。"""
    pcm = ax.contour(lon2d, lat2d, data_in, levels=levels, colors=colors, linestyles=linestyles,
                  transform=ccrs.PlateCarree(), linewidths=linewidths, zorder=zorder)
    return ax, pcm



def plot_pcolormesh_map_lambert(ax: AxesLike, data_in: np.ndarray, lon2d: np.ndarray, lat2d: np.ndarray,
                                levels: List[float], cmap: Any, zorder: int) -> AxesLike:
    """绘制单个等值线填色图，使用 Lambert 投影。"""
    pcm = ax.pcolormesh(lon2d, lat2d, data_in, cmap=cmap, vmin=levels[0], vmax=levels[-1],
                        transform=ccrs.PlateCarree(), shading='auto', zorder=zorder)
    return ax, pcm



def plot_significant_mark(ax: AxesLike, lon2d: np.ndarray, lat2d: np.ndarray, sign_mask: np.ndarray, s: int, marker: str='.',
                          color: str='grey', edgecolors: str='none', alpha: float = 1, zorder: int=99) -> AxesLike:
    ax.scatter(
                lon2d * sign_mask, lat2d * sign_mask, s=2, color=color, marker=marker,
                edgecolors=edgecolors, alpha=alpha, transform=ccrs.PlateCarree(), zorder=zorder
            )
    return ax



def Plot_Yangtze_BaseMap_Lambert(ax: AxesLike, data_in: np.ndarray, lon2d: np.ndarray, lat2d: np.ndarray, levels:np.ndarray = None,
                                 cmap: str = None, sign_mask: np.ndarray = None, text_leftbottom: str = None, text_righttop: str = None,
                                 text_leftbottom_fz: int = 14, text_righttop_fz: int = 14,
                                 marker: str = '.', mkcolor: str = 'grey', mkedgecolor: str = 'none', contourf: bool = False) -> AxesLike:
    """绘制长江流域的底图，使用 Lambert 投影。"""
    
    # 限制显示区域
    # ax = set_map_extent(ax, lon_range=[105.82, 123.2], lat_range=[24.2, 34.34])
    ax = set_map_extent(ax, lon_range=[106.2, 123.2], lat_range=[24.2, 34.15])
    
    # 绘制填色图
    if data_in is not None:
        if contourf:
            ax, pcm = plot_contourf_map_lambert(ax=ax, data_in=data_in, lon2d=lon2d, lat2d=lat2d, levels=levels, cmap=cmap, zorder=10)
        else:
            ax, pcm = plot_pcolormesh_map_lambert(ax=ax, data_in=data_in, lon2d=lon2d, lat2d=lat2d, levels=levels, cmap=cmap, zorder=10)
        
    # 对海洋进行掩模
    ax = mask_ocean(ax=ax, facecolor='auto', edgecolor='auto', zorder=15)
    
    # 添加对台湾省的掩模
    shp_path = f'{BASEDATA}/Taiwan.gpkg'
    ax = add_shapfile(ax=ax, shp_path=shp_path, edgecolor='none', facecolor='.9', zorder=30)

    # 添加湖泊网格标识
    shp_path = f'{BASEDATA}/SC_WATER.gpkg'
    ax = add_shapfile(ax=ax, shp_path=shp_path, edgecolor='black', facecolor='none', linewidth=0.6, zorder=30)

    # 添加中国行政区
    # ax = add_cnmaps(ax=ax, level='国', color='grey', zorder=99, linewidth=0.4)
    ax = add_cnmaps(ax=ax, level='国', color='k', zorder=49, linewidth=0.4)

    # 添加文本
    if text_leftbottom is not None:
        ax.text(0.02, 0.02, f"{text_leftbottom}", ha='left', va='bottom', fontsize=text_leftbottom_fz, 
                transform=ax.transAxes, fontweight='bold',zorder=30)
    if text_righttop is not None:
        ax.text(0.98, 0.98, f"{text_righttop}", ha='right', va='top', fontsize=text_righttop_fz, 
                transform=ax.transAxes, fontweight='bold',zorder=30)

    # 自动调整图幅比例
    ax.set_aspect('auto')
      
    if sign_mask is not None:
        ax = plot_significant_mark(ax=ax, lon2d=lon2d, lat2d=lat2d, sign_mask=sign_mask,
                            s=2.8, marker=marker, color=mkcolor, edgecolors=mkedgecolor, alpha=0.8, zorder=12)

    return ax


def Plot_Wind_Vector_Map_Lambert(
    ax,
    U: np.ndarray,
    V: np.ndarray,
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    text_leftbottom: str = None,
    text_righttop: str = None,
    # quiver 视觉参数
    text_leftbottom_fz: int = 10,
    text_righttop_fz: int = 10,
    scale: float = 35.0,            # ⚠ 数值越大，箭头越短（matplotlib quiver 规则）
    width: float = 0.0011,          # 箭头更细
    color: str = 'black',           # 保留但不生效，箭头统一黑色
    alpha: float = 0.9,
    skip: int = 8,
    # 风速过滤 & 长度映射参数
    min_speed: float = 0.1,         # 小于此风速不画箭头
    len_vmin: float = 1.0,          # 参与长度映射的最小风速 (m/s)
    len_vmax: float = 10.0,         # 参与长度映射的最大风速 (m/s)
    max_arrow_length: float = 1.1,  # 归一后“最大箭头长度”（无量纲，配合 scale 使用）
    min_arrow_frac: float = 0.3,    # 最短箭头 = max_arrow_length * min_arrow_frac
    arrow_scale: float = 1.5,       # 整体缩放一圈，<1 更短，>1 更长
    constlen: bool = False,
    # 背景场（热平流）
    bg_field: np.ndarray = None,
    mapcfg: Any = None,
    bg_cmap: str = 'RdBu_r',
    bg_levels: List[float] = None,
    bg_alpha: float = 0.8,
    add_colorbar: bool = True,
    add_legend: bool = False,
    enhanced_style: bool = True,
):
    """
    底色：热平流（bg_field），箭头：风向 + 风速（长度），箭头统一黑色。

    箭头长度控制逻辑：
    - constlen = False：
        - 给定风速区间 [len_vmin, len_vmax] (m/s)
        - 将风速裁剪到这个区间
        - 线性映射到 [min_arrow_length, max_arrow_length]
          其中 min_arrow_length = max_arrow_length * min_arrow_frac
        - 再整体乘上 arrow_scale
        - quiver 使用 scale 控制整体长度（数值越大，越短）
    - constlen = True：
        - 所有有效格点使用相同的箭头长度（max_arrow_length * arrow_scale）
        - 只显示风向，不显示风速大小差异
    """

    # 1. 地图范围（按你原来设置）
    ax = set_map_extent(ax, lon_range=[106.2, 123.2], lat_range=[24.2, 34.15])

    # 2. 风速大小 (m/s)
    wind_speed = np.sqrt(U**2 + V**2)

    contourf = None  # 防止 bg_field 为 None 时变量未定义

    # ========= 背景场：热平流 =========
    if bg_field is not None:
        ax, contourf = plot_pcolormesh_map_lambert(
            ax=ax,
            data_in=bg_field,
            lon2d=lon2d,
            lat2d=lat2d,
            levels=bg_levels,
            cmap=bg_cmap,
            zorder=10,
        )

    # ========= 稀疏化风场 =========
    if skip > 1:
        U_plot = U[::skip, ::skip]
        V_plot = V[::skip, ::skip]
        lon_plot = lon2d[::skip, ::skip]
        lat_plot = lat2d[::skip, ::skip]
        wind_speed_plot = wind_speed[::skip, ::skip]
    else:
        U_plot = U
        V_plot = V
        lon_plot = lon2d
        lat_plot = lat2d
        wind_speed_plot = wind_speed

    # ========= 有效格点掩膜：过滤小风 =========
    if min_speed is not None and min_speed > 0:
        valid_mask = wind_speed_plot >= min_speed
    else:
        valid_mask = np.isfinite(wind_speed_plot)

    # 初始化绘制数组
    U_draw = np.zeros_like(U_plot, dtype=float)
    V_draw = np.zeros_like(V_plot, dtype=float)

    # 如果一个格点都没有，就直接跳过
    if not np.any(valid_mask):
        # 只画背景场，返回
        ax.set_aspect('auto')
        return ax

    # ========= 在有效格点上做长度映射 =========
    # 取出有效格点的风速
    speed_valid = wind_speed_plot[valid_mask]

    # 1) 将 len_vmax 合法化
    if len_vmax <= len_vmin:
        len_vmax = len_vmin + 1e-6

    if constlen:
        # ✅ 所有有效格点使用同一箭头长度（最长箭头）
        arrow_length = np.full_like(speed_valid, max_arrow_length * arrow_scale, dtype=float)
    else:
        # 2) 将风速裁剪到 [len_vmin, len_vmax]
        speed_clip = np.clip(speed_valid, len_vmin, len_vmax)

        # 3) 线性归一到 [0, 1]
        norm = (speed_clip - len_vmin) / (len_vmax - len_vmin)  # [0,1]

        # 4) 映射到 [min_arrow_length, max_arrow_length]
        min_arrow_length = max_arrow_length * float(min_arrow_frac)
        arrow_length = min_arrow_length + norm * (max_arrow_length - min_arrow_length)

        # 5) 再整体缩放
        arrow_length = arrow_length * arrow_scale

    # ========= 单位方向向量 =========
    U_valid = U_plot[valid_mask]
    V_valid = V_plot[valid_mask]
    vec_len = np.sqrt(U_valid**2 + V_valid**2)

    # 避免 0 风速 / NaN 带来的问题
    nonzero = vec_len > 0
    # 对应非零格点的方向
    U_dir = np.zeros_like(U_valid, dtype=float)
    V_dir = np.zeros_like(V_valid, dtype=float)
    U_dir[nonzero] = U_valid[nonzero] / vec_len[nonzero]
    V_dir[nonzero] = V_valid[nonzero] / vec_len[nonzero]

    # 最终要画的矢量
    U_draw_valid = U_dir * arrow_length
    V_draw_valid = V_dir * arrow_length

    # 填回大数组
    U_draw[valid_mask] = U_draw_valid
    V_draw[valid_mask] = V_draw_valid

    # ========= 绘制黑色箭头 =========
    quiver = ax.quiver(
        lon_plot, lat_plot, U_draw, V_draw,
        color='k',                         # 强制黑色
        # 使用默认 scale 机制：scale 越大，箭头越短
        scale=scale,
        width=width,
        alpha=alpha * (1.05 if enhanced_style else 1.0),
        headlength=7,
        headwidth=5,
        headaxislength=6,
        transform=ccrs.PlateCarree(),
        zorder=25,
        edgecolors='k',
        linewidth=0.3,
    )

    # ========= 颜色条：表示热平流 =========
    if add_colorbar and (contourf is not None):
        cbar = plt.colorbar(
            contourf, ax=ax, orientation='vertical',
            shrink=0.8, pad=0.05
        )
        cbar.set_label('Warm advection (K/h)', fontsize=10)
        cbar.ax.tick_params(labelsize=9)

    # 添加文本
    if text_leftbottom is not None:
        ax.text(0.02, 0.02, f"{text_leftbottom}", ha='left', va='bottom', fontsize=text_leftbottom_fz, 
                transform=ax.transAxes, fontweight='bold',zorder=30)

    if text_righttop is not None:
        ax.text(0.98, 0.98, f"{text_righttop}", ha='right', va='top', fontsize=text_righttop_fz, 
                transform=ax.transAxes, fontweight='bold',zorder=30)

    ax.set_aspect('auto')
    return ax



def create_mapplot():
    """
    创建一个地图绘图框架
    """
    fig = plt.figure(figsize=(6, 4), layout='constrained')
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal(central_longitude=114.3, central_latitude=29.5))
    # fig, ax = plt.subplots(1, 1, figsize=(6, 4),
    #                         subplot_kw={'projection': ccrs.LambertConformal(central_longitude=105, central_latitude=35)})
    return fig, ax



def plot_spatial_diffmap_withsign(xarr_in: xr.DataArray, varname: str, timeperiod: str,
                                  target: str, lon2d: np.ndarray, lat2d: np.ndarray,
                                  checkmethod: str, lkinfos: Any, mapcfg: Any, 
                                  onlysig: bool = True, savepath: str = None):
    """
    处理区域气候影响的差异图(带有显著性标记)
    """
    time0 = time.time()
    diffcm  = mapcfg.cmap
    maplevs = mapcfg.levs[0]

    # 数据准备
    diffdata, sig_mask, dash_mask, strong_mask, weak_mask, suffix, area_all, area_strong = TDP.prepara_for_mapplot(xarr_in, lkinfos, 'mean_diff', checkmethod, onlysig)

    # ---------- 区域差异图 主图
    fig, ax_diff = create_mapplot()
    ax_diff = Plot_Yangtze_BaseMap_Lambert(ax=ax_diff, data_in=diffdata, lon2d=lon2d, lat2d=lat2d, levels=maplevs, cmap=diffcm, sign_mask=sig_mask, contourf=False)
    # ax_diff,_ = plot_contour_map_lambert(ax=ax_diff, data_in=dash_mask, lon2d=lon2d, lat2d=lat2d, levels=[0.5], colors='#A51E49', linewidths=1.2, linestyles='-', zorder=12)
    if timeperiod == "Annual":
        ax_diff.text(0.98, 0.98, varname, transform=ax_diff.transAxes, fontsize=18, fontweight='bold', ha='right', va='top',zorder=25)
    else:
        ax_diff.text(0.98, 0.98, timeperiod, transform=ax_diff.transAxes, fontsize=18, fontweight='bold', ha='right', va='top',zorder=25)
   # ---------- 区域差异图 左上角内嵌柱图 
    # 或者使用坐标轴相对坐标（0-1范围）
    # rect_ax = patches.Rectangle(
    #     (0.01, 0.01),   # 左下角坐标（相对于坐标轴）
    #     0.30,           # 宽度（相对于坐标轴）
    #     0.15,           # 高度（相对于坐标轴）
    #     transform=ax_diff.transAxes,  # 关键：使用坐标轴变换
    #     facecolor='white',
    #     edgecolor='none',
    #     zorder=50
    # )
    # ax_diff.add_patch(rect_ax)
    
    ax_in = inset_axes(
        ax_diff, width="30%", height="12%", loc="lower left",
        bbox_to_anchor=(0.07, 0.02, 1, 1), bbox_transform=ax_diff.transAxes, borderpad=0
    )
    ax_in.set_zorder(51)
    # 设置内嵌图的整体背景为纯白色
    ax_in.set_facecolor("white")
    # ax_in.patch.set_alpha(1.0)     
    ax_in.patch.set_alpha(0.0)

    # 将坐标轴背景也设置为纯白色
    ax_in.tick_params(axis='both', colors='black')  # 设置刻度颜色为黑色
    ax_in.xaxis.label.set_color('black')
    ax_in.yaxis.label.set_color('black')

    # 设置坐标轴的轴线颜色
    for spine in ax_in.spines.values():
        spine.set_edgecolor('black')  # 保持轴线可见，但可以设置为黑色
        spine.set_linewidth(1.5)
        # 如果您希望轴线也是白色，可以改为：
        spine.set_edgecolor('white')

    # 现在绘制柱状图
    # ax_in.bar(0, area_strong, width=0.3, edgecolor='none',
    #         facecolor='#C60069', alpha=0.5, linewidth=1.2)
    # ax_in.barh(0, area_all, height=0.3,
    #         edgecolor='blue', color='lightblue', alpha=0.9, linewidth=1.2)
    ax_in.barh(0, area_all, height=0.3,
            edgecolor='blue', color='lightblue', alpha=0.5, linewidth=1.2)
    ax_in.set_xlim(0, 165)
    ax_in.set_ylim(-0.25, 0.25)

    # # 隐藏顶部和右侧的轴线
    for spine in ["top", "right", "bottom"]:
        ax_in.spines[spine].set_visible(False)
    ax_in.spines["left"].set_edgecolor("black")

    # 设置刻度
    ax_in.set_xticks([])
    ax_in.yaxis.set_ticks_position("left")
    ax_in.xaxis.set_visible(False)
    ax_in.tick_params(labelsize=16, colors='black')  # 确保刻度标签为黑色
    ax_in.set_yticks([])
    ax_in.set_xticks([])
    ax_in.tick_params(colors='black')

    # 添加文本
    ## ax_in.text(0.2, area_all + 1, f"{area_all:.2f}", ha='center', va='bottom', fontsize=18, color='black', fontweight='bold')
    # ax_in.text(area_all * 1.02, 0, rf"{area_all:.2f}", ha='left', va='center', fontsize=16, color='black', fontweight='bold')
    
    # ax_in.text(area_all * 1.02, 0, rf"{area_all:.2f} $\times 10^4$ km$^2$", ha='left', va='center', fontsize=18, color='black', fontweight='bold')
    ax_in.text(
        area_all * 1.02, 0,
        rf"{area_all:.2f} $\mathbf{{\times\ 10^4\ km^2}}$",
        ha='left',
        va='center',
        fontsize=19,
        color='black',
        fontweight='bold'
    )
    # ax_in.text(0.2, area_strong, f"{area_strong:.2f}", ha='left', va='center', fontsize=7, color='black', fontweight='bold')
    # ax_in.set_ylabel("Area of significant\nregion ($10^4$ km$^2$)", fontsize=9, labelpad=2)
    ax_diff.text(0.10, 0.12, fr"Affected area", transform=ax_diff.transAxes, fontsize=19, fontweight='bold', ha='left', va='bottom',zorder=20)

    # ax_diff.text(0.07, 0.025, fr"Affected area: {area_all:.2f} $\times 10^4$ km$^2$", transform=ax_diff.transAxes, fontsize=17, fontweight='bold', ha='left', va='bottom',zorder=20)
    for spine in ax_diff.spines.values():
        spine.set_linewidth(1)  # 可以换成你想要的粗细

    if savepath is not None:
        plt.savefig(savepath, dpi=DPI, bbox_inches='tight') #
    plt.close(fig)
    fig.clf() # 清理内存
    gc.collect()



def plot_spatial_rcmap_withsign(xarr_in: xr.DataArray, varname: str, timeperiod: str,
                                  target: str, lon2d: np.ndarray, lat2d: np.ndarray,
                                  checkmethod: str, lkinfos: Any, mapcfg: Any, 
                                  onlysig: bool = True, savepath: str = None):
    """
    处理区域气候影响的差异图(带有显著性标记)
    """
    time0 = time.time()
    rccm  = mapcfg.cmap
    rclevs = mapcfg.levs[0]

    # 数据准备
    rcdata, sig_mask, dash_mask, strong_mask, weak_mask, suffix, area_all, area_strong = TDP.prepara_for_mapplot(xarr_in, lkinfos, 'RC_overall', checkmethod, onlysig)

    # ---------- 区域RC图 主图
    fig, ax_rc = create_mapplot()
    ax_rc=Plot_Yangtze_BaseMap_Lambert(ax=ax_rc, data_in=rcdata, lon2d=lon2d, lat2d=lat2d, levels=rclevs, cmap=rccm, sign_mask=sig_mask, contourf=False)
    # ax_rc,_ = plot_contour_map_lambert(ax=ax_rc, data_in=dash_mask, lon2d=lon2d, lat2d=lat2d, levels=[0.5], colors='#A51E49', linewidths=1.2, linestyles='-', zorder=12)
    # print(f"  Base map plot time: {time.time()-time0:.2f} seconds"); time0 = time.time()
    if timeperiod == "Annual":
        ax_rc.text(0.02, 0.96, varname, transform=ax_rc.transAxes, fontsize=16, fontweight='bold', ha='left', va='top',zorder=20)
    else:
        ax_rc.text(0.02, 0.96, timeperiod, transform=ax_rc.transAxes, fontsize=16, fontweight='bold', ha='left', va='top',zorder=20)
    
    for spine in ax_rc.spines.values():
        spine.set_linewidth(1)  # 可以换成你想要的粗细
    
    if savepath is not None:
        plt.savefig(savepath, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    fig.clf() # 清理内存
    gc.collect()


def plot_spatial_map(data_in: xr.DataArray, text_leftbottom:str, text_righttop: str,
                     lon2d: np.ndarray, lat2d: np.ndarray, mask2d: np.ndarray,
                     lkinfos: Any, mapcfg: Any, savepath: str, metrics_df: pd.DataFrame = None, addmetrics: str=None, contourf=True):

    """
    处理区域气候影响的均值图
    """
    time0 = time.time()
    oceanmask    = lkinfos['ocean']
    cmap  = mapcfg.cmap
    levs = mapcfg.levs[0]

    # 数据准备
    # fig, ax = create_mapplot()
    fig = plt.figure(figsize=(9, 6), layout='constrained')
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal(central_longitude=114.3, central_latitude=29.5))
    data = np.where(mask2d, data_in, np.nan)
    data[oceanmask == 1] = np.nan
    Plot_Yangtze_BaseMap_Lambert(ax=ax, data_in=data, lon2d=lon2d, lat2d=lat2d, levels=levs, cmap=cmap, text_leftbottom=text_leftbottom, text_righttop=text_righttop, text_righttop_fz=28, text_leftbottom_fz=28, contourf=contourf)

    for spine in ax.spines.values():    
        spine.set_linewidth(1.5)  # 可以换成你想要的粗细
    
    if metrics_df is not None:
        # 或者使用坐标轴相对坐标（0-1范围）
        rect_ax = patches.Rectangle(
            (0.70, 0.015),   # 左下角坐标（相对于坐标轴）
            0.25,           # 宽度（相对于坐标轴）
            0.25,           # 高度（相对于坐标轴）
            transform=ax.transAxes,  # 关键：使用坐标轴变换
            facecolor='white',
            edgecolor='none',
            zorder=50
        )
        ax.add_patch(rect_ax)
        meanrmse = metrics_df["mean_rmse"].values[0]
        meanbias = metrics_df["mean_bias"].values[0]
        mean_rbias = metrics_df["mean_rbias"].values[0]
        mean_rrmse = metrics_df["mean_rrmse"].values[0]
        mean_mae = metrics_df["mean_mae"].values[0]
        mean_mape = metrics_df["mean_mape"].values[0]
        mean_nmae_sigma = metrics_df["mean_nmae_sigma"].values[0]
        mean_nmae_iqr = metrics_df["mean_nmae_iqr"].values[0]
        mean_nmae_mean = metrics_df["mean_nmae_mean"].values[0]
        mean_nmae_mean_wet = metrics_df["mean_nmae_mean_wet"].values[0]
        mean_tcc = metrics_df["mean_tcc"].values[0]
        acc = metrics_df["acc"].values[0]
        ax.text(0.71, 0.24, f"RMSE: {meanrmse:.2f}", ha='left', va='top', fontsize=26,
                transform=ax.transAxes, fontweight='bold',zorder=100)
        ax.text(0.71, 0.17, f"Bias: {meanbias:.2f}", ha='left', va='top', fontsize=26,
                transform=ax.transAxes, fontweight='bold',zorder=100)
        if addmetrics is not None:
            if "RRMSE" == addmetrics:
                ax.text(0.71, 0.10, f"RRMSE: {mean_rrmse:.2f}", ha='left', va='top', fontsize=26,
                    transform=ax.transAxes, fontweight='bold',zorder=100)
            elif "MAE" == addmetrics:
                ax.text(0.71, 0.10, f"MAE: {mean_mae:.2f}", ha='left', va='top', fontsize=26,
                    transform=ax.transAxes, fontweight='bold',zorder=100)
            elif "MAPE" == addmetrics:
                ax.text(0.71, 0.10, f"MAPE: {mean_mape:.2f}", ha='left', va='top', fontsize=26,
                    transform=ax.transAxes, fontweight='bold',zorder=100)
            elif "NMAE" == addmetrics:
                ax.text(0.71, 0.10, f"NMAE: {mean_nmae_mean_wet:.2f}", ha='left', va='top', fontsize=26,
                    transform=ax.transAxes, fontweight='bold',zorder=100)
            elif "RB" == addmetrics:
                ax.text(0.71, 0.10, f"RBias: {mean_rbias:.2f}", ha='left', va='top', fontsize=26,
                    transform=ax.transAxes, fontweight='bold',zorder=100)
            # ax.text(0.78, 0.10, f"MAE: {mean_mae:.2f}", ha='left', va='top', fontsize=26,
            #         transform=ax.transAxes, fontweight='bold',zorder=100)
            # ax.text(0.78, 0.10, f"MAPE: {mean_mape:.2f}", ha='left', va='top', fontsize=26,
            #         transform=ax.transAxes, fontweight='bold',zorder=100)
            # ax.text(0.78, 0.10, f"NMAE$_{{\\sigma}}$: {mean_nmae_sigma:.2f}", ha='left', va='top', fontsize=26,
            #         transform=ax.transAxes, fontweight='bold',zorder=100)
            # ax.text(0.02, 0.76, f"TCC: {mean_tcc:.2f}", ha='left', va='top', fontsize=14,
            #         transform=ax.transAxes, fontweight='bold',zorder=100)
            # ax.text(0.02, 0.70, f"ACC: {acc:.2f}", ha='left', va='top', fontsize=14,
            #         transform=ax.transAxes, fontweight='bold',zorder=100)
    os.system(f"rm -f {savepath}")
    plt.savefig(savepath, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    fig.clf()
    gc.collect()
    


def plot_hot_wet_coupling( hot: Dict, wet: Dict, hotwet: Dict, coupls: Dict, 
                           lon2d: np.ndarray, lat2d: np.ndarray, cmap: Any, FigOutDir: str):
    """
    Plot the spatial coupling of Hot and Wet events with 9 different regions and Hot-Wet states (up, down, none) 
    with distinct colors and symbols.

    Parameters:
    - hot: EventState object for Hot (contains up, down, zero, none)
    - wet: EventState object for Wet (contains up, down, zero, none)
    - hotwet: EventState object for HotWet (contains up, down, none)
    - lon2d, lat2d: 2D arrays of longitude and latitude for plotting
    - FigOutDir: Output directory for saving the figure

    Returns:
    - ax_diff: Axis object with the plot
    """
    # Create map
    fig, ax = create_mapplot()

    # Create combined mask for valid data points (both Hot and Wet)
    coupl_mask = hot[coupls["Hot"]] & wet[coupls["Wet"]]  # Combined mask for valid data points

    # Get individual masks for Hot-Wet Up, Down, and None
    hotwet_up = hotwet["up"] & coupl_mask
    hotwet_down = hotwet["down"] & coupl_mask
    hotwet_none = hotwet["none"] & coupl_mask

    # Initialize a 2D grid for the plot, setting all values to np.nan (for Hot-Wet None initially)
    coupl_grid = np.full_like(lon2d, np.nan)

    # Now assign values to the coupl grid based on the conditions
    coupl_grid[hotwet_up] = 1      # Hot-Wet Up -> 1
    coupl_grid[hotwet_down] = -1    # Hot-Wet Down -> -1
    coupl_grid[hotwet_none] = 0     # Hot-Wet None -> 0 (only where not Up or Down)

    # Plot the regions using pcolormesh
    levels = [-1.5, -0.5, 0.5, 1.5]  # Define levels for pcolormesh
    ax = Plot_Yangtze_BaseMap_Lambert(ax=ax, data_in=coupl_grid, lon2d=lon2d, lat2d=lat2d, cmap=cmap, contourf=False, levels=levels)
    ax.text(0.98, 0.03, f"Hot-{coupls['Hot']} & Wet-{coupls['Wet']}", transform=ax.transAxes,
            fontsize=16, fontweight='bold', ha='right', va='bottom',zorder=150)

    # Save the figure
    savepath = f'{FigOutDir}/HotWet_Coupling_Hot-{coupls["Hot"]}_Wet-{coupls["Wet"]}.{FIGFMT}'
    plt.savefig(savepath, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    fig.clf() # 清理内存
    gc.collect()


def plot_categorical_map(data_in: np.ndarray, target: str, 
                         lon2d: np.ndarray, lat2d: np.ndarray,
                         savepath: str=None,
                         text_leftbottom: str=None, text_righttop: str=None,
                         lkinfos: Any=None, mapcfg: Any=None,
                         mask2d: np.ndarray=None, ):
    """
    绘制一个“分类数据”的空间分布图。
    """
    cmap  = mapcfg.cmap
    levs = mapcfg.levs[0]

    # 创建底图
    # fig, ax = create_mapplot()
    fig = plt.figure(figsize=(9, 6), layout='constrained')
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal(central_longitude=114.3, central_latitude=29.5))

    if lkinfos is not None:
        oceanmask = lkinfos['ocean']
        data_in[oceanmask == 1] = np.nan
    ax = Plot_Yangtze_BaseMap_Lambert(ax=ax, data_in=data_in, lon2d=lon2d, lat2d=lat2d, levels=levs,
                                 cmap=cmap, text_leftbottom=text_leftbottom, text_righttop=text_righttop, text_righttop_fz=18,contourf=True)

    # 或者使用坐标轴相对坐标（0-1范围）
    rect_ax = patches.Rectangle(
        (0.785, 0.015),   # 左下角坐标（相对于坐标轴）
        0.20,           # 宽度（相对于坐标轴）
        0.35,           # 高度（相对于坐标轴）
        transform=ax.transAxes,  # 关键：使用坐标轴变换
        facecolor='white',
        edgecolor='none',
        zorder=50
    )
    ax.add_patch(rect_ax)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)  # 可以换成你想要的粗细

    # 数据准备
    if mask2d is not None:
        data_1d = np.where(mask2d, data_in, np.nan)
    else:
        data_1d = data_in

    # up and down percentage inset bar plot
    data_up_pct = np.nansum(data_1d >= 0) / np.nansum(np.isfinite(data_1d)) * 100
    data_down_pct = np.nansum(data_1d < 0) / np.nansum(np.isfinite(data_1d)) * 100

    ax_in = inset_axes(
        ax, width="15%", height="30%", loc="lower left",
        bbox_to_anchor=(0.825, 0.025, 1, 1), bbox_transform=ax.transAxes, borderpad=0
    )
    ax_in.set_zorder(51)
    # 设置内嵌图的整体背景为纯白色
    ax_in.set_facecolor("white")
    ax_in.patch.set_alpha(1.0)     

    # 将坐标轴背景也设置为纯白色
    ax_in.tick_params(axis='both', colors='black')  # 设置刻度颜色为黑色
    ax_in.xaxis.label.set_color('black')
    ax_in.yaxis.label.set_color('black')

    # 设置坐标轴的轴线颜色
    for spine in ax_in.spines.values():
        spine.set_edgecolor('black')  # 保持轴线可见，但可以设置为黑色
        # 如果您希望轴线也是白色，可以改为：
        # spine.set_edgecolor('white')

    # 现在绘制柱状图
    ax_in.bar(0, -data_down_pct, width=0.3, edgecolor='#53589A',
            facecolor='#53589A', alpha=1, linewidth=1.2)
    ax_in.bar(0, data_up_pct, width=0.3, edgecolor='#FFA96B',
            facecolor='#FFA96B', alpha=1, linewidth=1.2)
    ax_in.axhline(0, color='black', linestyle='--')

    ax_in.set_xlim(-0.3, 1.0)
    ax_in.set_ylim(-50, 90)

    # 添加文本
    ax_in.text(0, data_up_pct + 5, f"{data_up_pct:.2f}%", ha='center', va='bottom', fontsize=12, color='black', fontweight='bold')
    ax_in.text(0, -data_down_pct - 5 , f"{-data_down_pct:.2f}%", ha='center', va='top', fontsize=12, color='black', fontweight='bold')

    ax_in.text(0.2, 15, f"Increase", ha='left', va='bottom', fontsize=13, color='black', fontweight='bold')
    ax_in.text(0.2, -15, f"Decrease", ha='left', va='top', fontsize=13, color='black', fontweight='bold')

    # 隐藏顶部和右侧的轴线
    for spine in ["top", "right"]:
        ax_in.spines[spine].set_visible(False)

    # 设置刻度
    ax_in.set_xticks([])
    ax_in.yaxis.set_ticks_position("left")
    ax_in.xaxis.set_visible(False)
    ax_in.tick_params(labelsize=11, colors='black')  # 确保刻度标签为黑色
    
    if savepath is not None:
        plt.savefig(savepath, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    fig.clf() # 清理内存
    gc.collect()




def plot_warmadvection_spatial_map(data_in: xr.DataArray, U_data: xr.DataArray, V_data: xr.DataArray, timeperiod: str,
                                   target: str, lon2d: np.ndarray, lat2d: np.ndarray, savepath: str, mapcfg: Any):
    """
    处理区域气候影响的暖色散图
    """
    cmap  = mapcfg.cmap
    levs = mapcfg.levs[0]

    # ---------- 区域差异图 主图
    fig, ax = create_mapplot()

    Plot_Wind_Vector_Map_Lambert(ax=ax, U=U_data, V=V_data, lon2d=lon2d, lat2d=lat2d,
                                 text_righttop=timeperiod,  text_righttop_fz=14, color='black', alpha=0.8, 
                                 bg_field=data_in, bg_cmap=cmap, bg_levels=levs, bg_alpha=0.8)

    plt.savefig(savepath, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    fig.clf() # 清理内存
    gc.collect()



def plot_spatial_map_UV(U_data: xr.DataArray, V_data: xr.DataArray, varname: str, dataname:str, timeperiod: str,
                        target: str, lon2d: np.ndarray, lat2d: np.ndarray, lkinfos: str, savepath: str, mapcfg: Any): 
    """
    处理区域气候影响的均值图
    """
    time0 = time.time()
    oceanmask = lkinfos['ocean']
    cmap  = mapcfg.cmap
    levs = mapcfg.levs[0]

    # 数据准备
    fig = plt.figure(figsize=(9, 6), layout='constrained')
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal(central_longitude=114.3, central_latitude=29.5))
    U_data[oceanmask == 1] = np.nan
    V_data[oceanmask == 1] = np.nan
    UV_magnitude = np.sqrt(U_data**2 + V_data**2)
    data = UV_magnitude
    data[oceanmask == 1] = np.nan

    print(timeperiod)
    ax = Plot_Wind_Vector_Map_Lambert(ax=ax, U=U_data, V=V_data, lon2d=lon2d, lat2d=lat2d, skip=8, max_arrow_length=0.5, 
                                 text_righttop=timeperiod,  text_righttop_fz=28, color='black', alpha=0.8, constlen=True, add_colorbar=False,
                                 bg_field=data, bg_cmap=cmap, bg_levels=levs, bg_alpha=0.8)

    # 对海洋进行掩模
    ax = mask_ocean(ax=ax, facecolor='auto', edgecolor='auto', zorder=26)

    # 添加对台湾省的掩模
    shp_path = f'{BASEDATA}/Taiwan.gpkg'
    ax = add_shapfile(ax=ax, shp_path=shp_path, edgecolor='none', facecolor='.9', zorder=30)

    # 添加湖泊网格标识
    shp_path = f'{BASEDATA}/SC_WATER.gpkg'
    ax = add_shapfile(ax=ax, shp_path=shp_path, edgecolor='black', facecolor='none', linewidth=0.6, zorder=30)

    # 添加中国行政区
    # ax = add_cnmaps(ax=ax, level='国', color='grey', zorder=99, linewidth=0.4)
    ax = add_cnmaps(ax=ax, level='国', color='k', zorder=99, linewidth=0.4)

    for spine in ax.spines.values():    
        spine.set_linewidth(1.5)  # 可以换成你想要的粗细

    plt.savefig(savepath, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    fig.clf() # 清理内存
    gc.collect()



def plot_city_stat_map_lambert(
        gdf: "gpd.GeoDataFrame",
        value_col: str,
        mapcfg: Any,
        savepath: str = None,
        kwargs: Dict = None
    ) -> AxesLike:

    """
    使用 Lambert 投影绘制基于市级 GeoDataFrame 的空间分布图        
        text_leftbottom: str = None,
        text_righttop: str = None,
        city_edgecolor: str = "k",
        city_linewidth: float = 0.5,
        cbar_label: str = None,
        city_name_col: str = None,        # 城市名称列名
        city_name_fontsize: float = 8.0,  # 名称字号
        city_name_color: str = "k",       # 名称颜色
        use_input_levels: bool = False    # 删除此参数，因为我们不再分类
    ) -> AxesLike:
    使用 Lambert 投影绘制基于市级 GeoDataFrame 的空间分布图（分市填色）。
    此版本将数据根据输入的 cmap 进行连续染色。
    """

    # ========= kwargs 默认值 =========
    if kwargs is None:
        kwargs = {}
    lon_range: Tuple[float, float] = kwargs.get("lon_range", (106.2, 123.2))
    lat_range: Tuple[float, float] = kwargs.get("lat_range", (24.2, 34.15))

    city_edgecolor: str = kwargs.get("city_edgecolor", "k")
    city_linewidth: float = kwargs.get("city_linewidth", 0.5)

    text_leftbottom: Optional[str] = kwargs.get("text_leftbottom", None)
    text_righttop: Optional[str] = kwargs.get("text_righttop", None)

    city_name_col: Optional[str] = kwargs.get("city_name_col", None)
    city_name_fontsize: float = kwargs.get("city_name_fontsize", 8)
    city_name_color_default: str = kwargs.get("city_name_color", "black")  # fallback
    name_margin_lon: float = kwargs.get("name_margin_lon", 0.3)
    name_margin_lat: float = kwargs.get("name_margin_lat", 0.3)

    # 你已有的 is_dark(color) 可以直接用；这里给个阈值参数
    dark_threshold: float = kwargs.get("dark_threshold", 0.25)

    # ===== 1) 基本设置：cmap & levels(norm) =====
    cmap = mapcfg.cmap
    if isinstance(cmap, str):
        cmap = cm.get_cmap(cmap)

    levels = np.asarray(mapcfg.levs[0], dtype=float)
    if levels.ndim != 1 or levels.size < 2:
        raise ValueError("levels 必须是一维边界数组，且长度 >= 2")
    if not np.all(np.diff(levels) > 0):
        levels = np.unique(levels)
        if levels.size < 2 or not np.all(np.diff(levels) > 0):
            raise ValueError("levels 必须严格递增")

    norm = mcolors.BoundaryNorm(levels, ncolors=cmap.N, clip=False)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)

    # ========= 3) 建图 & extent =========
    fig = plt.figure(figsize=(9, 6), layout='constrained')
    ax = fig.add_subplot(
        1, 1, 1,
        projection=ccrs.LambertConformal(central_longitude=114.3, central_latitude=29.5)
    )
    ax = set_map_extent(ax, lon_range=list(lon_range), lat_range=list(lat_range))

    # ========= 4) CRS 统一到 EPSG:4326 =========
    gdf_plot = gdf.copy()
    if gdf_plot.crs is None:
        gdf_plot = gdf_plot.set_crs("EPSG:4326")
    else:
        gdf_plot = gdf_plot.to_crs("EPSG:4326")

    # 关键：先固定顺序，再计算 colors（避免 geoms 与 facecolors 错位）
    gdf_plot = gdf_plot.reset_index(drop=True)

    # === 计算每个城市的旋转角（建议用投影坐标算更稳定）===
    gdf_proj = gdf_plot.to_crs(epsg=3857)  # 用米单位投影算方向更靠谱
    gdf_plot["_angle"] = gdf_proj.geometry.apply(_angle_from_long_axis_deg).to_numpy()

    # ========= 5) 取值 -> RGBA（一次完成） =========
    # 注意：to_numeric 防止字符串等导致崩溃
    values = pd.to_numeric(gdf_plot[value_col], errors="coerce").to_numpy(dtype=float)

    rgba_arr = sm.to_rgba(values)               # (N,4)
    nan_mask = ~np.isfinite(values)
    rgba_arr[nan_mask, 3] = 0.0                 # NaN 透明

    # 给 colorbar 用（可选）
    sm.set_array(values)

    # 保存每个城市的 RGBA（用于标注市名判断深浅）
    gdf_plot["_rgba"] = [tuple(rgba) for rgba in rgba_arr]

    # 给 cartopy 的 facecolor（与 geometry 一一对应）
    facecolors = rgba_arr.tolist()

    # ========= 6) 掩模海洋 + 市级面填色 =========
    ax = mask_ocean(ax=ax, facecolor="auto", edgecolor="auto", zorder=5)
    ax = add_cnmaps(ax=ax, level='省', color='k', zorder=99, linewidth=1)

    # ✅ 这里不会不对：因为 facecolors 与 geoms 都来自 reset_index 后的 gdf_plot
    geoms = list(gdf_plot.geometry)
    ax.add_geometries(
        geoms,
        crs=ccrs.PlateCarree(),
        facecolor=facecolors,
        edgecolor=city_edgecolor,
        linewidth=city_linewidth,
        zorder=20,
    )

    # 7) 台湾掩模
    shp_path_tw = f"{BASEDATA}/Taiwan.gpkg"
    ax = add_shapfile(
        ax=ax,
        shp_path=shp_path_tw,
        edgecolor="none",
        facecolor=".9",
        zorder=30,
    )

    # ========= 8) 左下/右上文字 =========
    if text_leftbottom:
        ax.text(0.02, 0.02, text_leftbottom, ha="left", va="bottom",
                fontsize=14, fontweight="bold", transform=ax.transAxes, zorder=40)
    if text_righttop:
        ax.text(0.98, 0.98, text_righttop, ha="right", va="top",
                fontsize=14, fontweight="bold", transform=ax.transAxes, zorder=40)

    # ========= 9) 城市名称标注：根据底色自动选黑/白 =========
    if city_name_col is not None:
        lon_min, lon_max = lon_range
        lat_min, lat_max = lat_range

        sub = gdf_plot[[city_name_col, "_rgba", "_angle", "geometry"]]

        for name, rgba, ang, geom in sub.itertuples(index=False, name=None):
            if pd.isna(name) or geom is None or geom.is_empty:
                continue

            pt = geom.representative_point()
            x, y = pt.x, pt.y

            if not (lon_min + name_margin_lon <= x <= lon_max - name_margin_lon and
                    lat_min + name_margin_lat <= y <= lat_max - name_margin_lat):
                continue

            # rgba: (r,g,b,a)
            if rgba is None or len(rgba) != 4 or rgba[3] == 0:
                txt_color = city_name_color_default
            else:
                # 你已有 is_dark(color)；它能吃 (r,g,b) tuple
                txt_color = "white" if is_dark(rgba[:3], threshold=dark_threshold) else "black"
            outline = "black" if txt_color == "white" else "white"
            name = name.strip()
            if " " in name:
                name = name.replace(" ", "\n")  # 多词名换行显示
            ax.text(
                x, y, str(name),
                fontsize=city_name_fontsize,
                fontweight="bold",
                color=txt_color,
                ha="center", va="center",
                transform=ccrs.PlateCarree(),
                zorder=60,
                # path_effects=[pe.withStroke(linewidth=2.0, foreground=outline)]
            )

    for spine in ax.spines.values():    
        spine.set_linewidth(1.5)  # 可以换成你想要的粗细

    # ========= 10) 保存 =========
    if savepath:
        fig.savefig(savepath, dpi=DPI, format=FIGFMT, bbox_inches="tight")
        plt.close(fig)
        fig.clf() # 清理内存
        gc.collect()
    else:
        return ax