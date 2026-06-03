import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 自定义工具箱
import ToolBoxes.Utils as TU
import ToolBoxes.Tool_DataPrepare as TDP
import ToolBoxes.Tool_PlotConfig as TPC
FIGFMT = TPC.FIGFMT
DPI    = TPC.DPI_medium

def plot_kde_1d(x: np.ndarray, savepath: str):
    """
    输入：
      x: 1D np.array
      savepath: 输出路径（含文件名）
    输出：
      保存一张 KDE 曲线图
    """
    x = np.asarray(x).ravel()
    x = x[np.isfinite(x)]  # 去掉 NaN/inf

    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")

    if x.size < 2:
        ax.text(0.5, 0.5, "No valid data", transform=ax.transAxes,
                ha="center", va="center", color="0.5", fontsize=12)
        ax.set_xlabel("Value", fontsize=14)
        ax.set_ylabel("Density", fontsize=14)
    else:
        sns.kdeplot(
            x=x,
            ax=ax,
            bw_adjust=1.2,     # 平滑度（越大越平滑）
            gridsize=512,
            cut=0,             # 不超出数据范围外延
            fill=False,
            linewidth=2.0,
        )
        ax.set_xlabel("Value", fontsize=14)
        ax.set_ylabel("Density", fontsize=14)
        ax.tick_params(axis="both", labelsize=12)

    os.makedirs(os.path.dirname(savepath) or ".", exist_ok=True)
    fig.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close(fig)


