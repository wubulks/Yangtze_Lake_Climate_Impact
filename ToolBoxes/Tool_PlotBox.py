import os
import time
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib import font_manager as fm

# 自定义工具箱
import ToolBoxes.Utils as TU
import ToolBoxes.Tool_DataPrepare as TDP
import ToolBoxes.Tool_PlotConfig as TPC
FIGFMT = TPC.FIGFMT
DPI    = TPC.DPI_medium


mpl.use('Agg')  # 不显示图，只保存
mpl.rcParams['font.family'] = 'Noto Sans'
warnings.filterwarnings("ignore", category=RuntimeWarning)



def plot_diff_boxplot(df_in: pd.DataFrame, orders: list, varname: str, target: str,
                      checkmethod: str, FigOutDir: str, boxcfg: TPC.boxConfig,
                      varinfo: TPC.varInfo, suffix: str, tickfmt = '.0f'):
    """
    绘制区域气候影响
    修正版：支持各季节数据长度不一致
    """
    time0 = time.time()
    diff_levs = boxcfg.diff_boxlevs[0]
    ndiff_levs= boxcfg.diff_boxlevs[1]
    rc_levs   = boxcfg.rc_boxlevs[0]
    nrc_levs  = boxcfg.rc_boxlevs[1]
    unit      = varinfo.unit
    longname  = varinfo.longname
    abbr      = varinfo.abbr

    whis = 1.5
    # ===================================
    #         差异图绘制
    # ===================================
    fig, ax = plt.subplots(figsize=(7, 4), layout='constrained') 
    plot_df = df_in[np.isfinite(df_in['Value'].to_numpy())].copy()
    
    if plot_df.empty:
        # —— 无有效数据：不调用 boxplot，画占位坐标轴，避免报错 ——
        ax.set_xlabel('')
        ax.set_ylabel(f'Frequency difference ({unit})', fontsize=14)
        # 你的预设范围/刻度（如果有 difflevs/ndifflevs 就用它们）
        ax.set_xlim(-0.5, len(orders) - 0.5)
        ax.set_xticks(range(len(orders)))
        ax.set_xticklabels(orders)
        # 如果有固定 y 轴范围：
        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.text(0.5, 0.5, 'No valid data', transform=ax.transAxes,
                ha='center', va='center', color='0.5', fontsize=12)
    else:
        # —— 有有效数据：正常画箱线图 ——
        sns.boxplot(
            data=plot_df, ax=ax, x='Season', y='Value', order=orders,
            whis=whis, showfliers=False, width=0.5,
            boxprops=dict(facecolor='lightblue', edgecolor='blue', linewidth=1.2, alpha=0.5),
            medianprops=dict(color='red', linewidth=1.5),
            whiskerprops=dict(color='blue', linewidth=1.2),
            capprops=dict(color='blue', linewidth=1.2)
        )
        ax.set_xticklabels(orders, fontsize=16, rotation=30)
        # 只用参与箱线统计的“内点”算均值（你已有 sel_valid_data）
        diff_df_in = TDP.sel_valid_data(plot_df, whis)
        means = (diff_df_in.groupby('Season')['Value'].mean().reindex(orders))
        # 安全地叠加均值点（可能全是 NaN）
        if not means.dropna().empty:
            ax.scatter(range(len(orders)), means.values,
                    marker='.', s=40, color='black', zorder=3, label='mean')

        ax.set_xlim(-0.5, len(orders) - 0.5)
        # 如需固定 y 轴范围：
        # ax.set_ylim(difflevs[0], difflevs[-1])
    if varname == "Annual":
        #  y轴在右边
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
    rc_df_in=TDP.sel_valid_data(df_in, whis)
    means = rc_df_in.groupby('Season')['Value'].mean().reindex(orders)
    ax.scatter(range(len(orders)), means.values,
            marker='.', s=40, color='black', zorder=3, label='mean')
    ax.set_xlim(-0.5, len(orders) - 0.5)
    ax.set_ylim(diff_levs[0], diff_levs[-1])
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('')
    ax.set_ylabel(f'Frequency difference\n({unit})', fontsize=18, fontweight="bold")
    ticks = np.linspace(diff_levs[0], diff_levs[-1], ndiff_levs)
    ax.set_yticks(ticks)
    # use tickfmt for ytick labels
    ax.set_yticklabels([format(tick, tickfmt) for tick in ticks], fontsize=12)
    ax.set_aspect('auto')
    savepath = f'{FigOutDir}/{target}_Diff_Box_{varname}_{suffix}_{checkmethod}.{FIGFMT}'
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    plt.savefig(savepath, dpi=DPI, bbox_inches='tight')
    plt.close()
    # print(f"  Save fig time: {time.time()-time0:.2f} seconds"); time0 = time.time()
    return ax




def plot_rc_boxplot(df_in: pd.DataFrame, orders: list, varname: str, target: str,
                    checkmethod: str, FigOutDir: str, boxcfg: TPC.boxConfig,
                    varinfo: TPC.varInfo, suffix: str):
    """
    绘制区域气候影响
    修正版：支持各季节数据长度不一致
    """
    time0 = time.time()
    rc_levs   = boxcfg.rc_boxlevs[0]
    nrc_levs  = boxcfg.rc_boxlevs[1]
    unit      = varinfo.unit
    longname  = varinfo.longname
    abbr      = varinfo.abbr

    whis = 1.5
    # ===================================
    #         差异图绘制
    # ===================================
    fig, ax = plt.subplots(figsize=(7, 4), layout='constrained') 
    plot_df = df_in[np.isfinite(df_in['Value'].to_numpy())].copy()
    
    if plot_df.empty:
        # —— 无有效数据：不调用 boxplot，画占位坐标轴，避免报错 ——
        ax.set_xlabel('')
        ax.set_ylabel(f'Relative contribution ({unit})', fontsize=14)
        # 你的预设范围/刻度（如果有 difflevs/ndifflevs 就用它们）
        ax.set_xlim(-0.5, len(orders) - 0.5)
        ax.set_xticks(range(len(orders)))
        ax.set_xticklabels(orders)
        # 如果有固定 y 轴范围：
        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.text(0.5, 0.5, 'No valid data', transform=ax.transAxes,
                ha='center', va='center', color='0.5', fontsize=12)
    else:
        # —— 有有效数据：正常画箱线图 ——
        sns.boxplot(
            data=plot_df, ax=ax, x='Season', y='Value', order=orders,
            whis=whis, showfliers=False, width=0.5,
            boxprops=dict(facecolor='lightblue', edgecolor='blue', linewidth=1.2, alpha=0.5),
            medianprops=dict(color='red', linewidth=1.5),
            whiskerprops=dict(color='blue', linewidth=1.2),
            capprops=dict(color='blue', linewidth=1.2)
        )

        # 只用参与箱线统计的“内点”算均值（你已有 sel_valid_data）
        diff_df_in = TDP.sel_valid_data(plot_df, whis)
        means = (diff_df_in.groupby('Season')['Value'].mean().reindex(orders))
        # 安全地叠加均值点（可能全是 NaN）
        if not means.dropna().empty:
            ax.scatter(range(len(orders)), means.values,
                    marker='.', s=40, color='black', zorder=3, label='mean')

        ax.set_xlim(-0.5, len(orders) - 0.5)
        # 如需固定 y 轴范围：
        # ax.set_ylim(difflevs[0], difflevs[-1])
    if varname == "Annual":
        #  y轴在右边
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
    rc_df_in=TDP.sel_valid_data(df_in, whis)
    means = rc_df_in.groupby('Season')['Value'].mean().reindex(orders)
    ax.scatter(range(len(orders)), means.values,
            marker='.', s=40, color='black', zorder=3, label='mean')
    ax.set_xlim(-0.5, len(orders) - 0.5)
    ax.set_ylim(rc_levs[0], rc_levs[-1])
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('')
    ax.set_ylabel('Relative contribution (%)', fontsize=14, fontweight="bold")
    ticks = np.linspace(rc_levs[0], rc_levs[-1], nrc_levs)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f'{tick:.0f}' for tick in ticks], fontsize=12)
    ax.set_aspect('auto')
    savepath = f'{FigOutDir}/{target}_RC_Box_{varname}_{suffix}_{checkmethod}.{FIGFMT}'
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    plt.savefig(savepath, dpi=DPI, bbox_inches='tight')
    plt.close()
    # print(f"  Save fig time: {time.time()-time0:.2f} seconds"); time0 = time.time()
    return ax



