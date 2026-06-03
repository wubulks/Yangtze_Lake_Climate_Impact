import os
import seaborn as sns
import matplotlib.pyplot as plt

# 自定义模块
import ToolBoxes.Tool_PlotConfig as TPC
FIGFMT = TPC.FIGFMT
DPI    = TPC.DPI_medium
HighThresPercentile = TPC.HighThresPercentile
LowThresPercentile  = TPC.LowThresPercentile
ThresWindows        = TPC.ThresWindows

def plot_Joint(df_in, x, y, hue, palette, savepath, lat=None, lon=None):

        # Density plot (KDE) for Hot anomalies
        plt.figure(figsize=(6, 6))

        # Create a jointplot for 'With Lake' data
        g = sns.jointplot(
            data=df_in,
            x="RH", y="T", hue="Type", 
            kind="kde", fill=False,
            bw_method=0.5,  # 控制带宽，调整平滑程度
            levels= 7,  # 设置等高线的数量为10
            palette=palette,
            linewidths=1.5,
        )
        g.figure.set_size_inches(6, 6)
        ax = g.ax_joint
        # 先拿一下当前坐标范围
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

        # 只在图中可见的 x>0, y>0 范围里填色
        x0 = max(0, x_min)
        y0 = max(0, y_min)

        if x_max > 0 and y_max > 0:
            ax.fill_between(
                [x0, x_max],          # x 从 x0 到 x_max
                [y0, y0],             # 下边界 y1 = y0
                [y_max, y_max],       # 上边界 y2 = y_max
                color='#fbe8d3',
                alpha=0.4,
                zorder=0              # 放在所有线下面
            )

        # Optional: Additional styling for the plot
        ax.axhline(0, color='black', linestyle='--')
        ax.axvline(0, color='black', linestyle='--')

        # Set axis labels and title
        # $T_{d} > T^{85\mathrm{th}}_{d}$
        # ax.set_xlabel(r'RH$_{d}$ $-$ RH$^{HighThresPercentile\mathrm{th}}_{d}$ (%)', fontsize=14)
        # ax.set_ylabel(r'T$_{d}$ $-$ T$^{HighThresPercentile\mathrm{th}}_{d}$ (°C)', fontsize=14)
        ax.set_xlabel(
            rf'$\mathbf{{RH}}_{{\mathbf{{d}}}} - \mathbf{{RH}}^{{\mathbf{{{HighThresPercentile}}}\,\mathbf{{th}}}}_{{\mathbf{{d}}}}$ $\mathbf{{(\%)}}$',
            fontsize=14,
            fontweight='bold'
        )
        ax.set_ylabel(
            rf'$\mathbf{{T}}_{{\mathbf{{d}}}} - \mathbf{{T}}_{{\mathbf{{d}}}}^{{\mathbf{{{HighThresPercentile}}}\,\mathbf{{th}}}}$ (°C)',
            fontsize=14,
            fontweight='bold'
        )
        # ax.set_xlabel(r'RH $-$ RH$_{HighThresPercentile\mathrm{th}}$ (%)', fontsize=14)
        # ax.set_ylabel(r'T $-$ T$_{HighThresPercentile\mathrm{th}}$ (°C)', fontsize=14)
        ax.tick_params(axis="both", which="major", labelsize=12)

        if lat is not None and lon is not None:
            ax.text(0.05, 0.05, f'Lat: ${lat:.2f}°$', fontsize=14, transform=ax.transAxes, fontweight='normal')
            ax.text(0.05, 0.10, f'Lon: ${lon:.2f}°$', fontsize=14, transform=ax.transAxes, fontweight='normal')

        # --- 调整 legend 字体大小 ---
        leg = g.ax_joint.get_legend()
        if leg is None:  # 有些版本 legend 可能挂在 figure 上
            leg = g.figure.legends[0] if g.figure.legends else None

        if leg is not None:
            leg.set_loc("upper left")
            leg.set_title(leg.get_title().get_text(), prop={"size": 12})  # 标题字体（可选）
            for t in leg.get_texts():   # 条目字体
                t.set_fontsize(12)

        # Save the figure
        g.figure.savefig(savepath, dpi=DPI, format=FIGFMT, bbox_inches='tight')
        plt.close()



def plot_Advance_Joint(df_in, x, y, hue, palette, savepath, HighThresPercentile, LowThresPercentile, refname):
    # Density plot (KDE) for Hot anomalies
    plt.figure(figsize=(10, 6))

    # Create the jointplot (only once)
    g = sns.jointplot(
        data=df_in,
        x=x, y=y, hue=hue, 
        kind="kde", fill=False,  # Default fill=False, we'll handle filling manually for ref
        bw_method=1,  # 控制带宽，调整平滑程度
        levels=7,  # 设置等高线的数量为7
        palette=palette,
        linewidths=1.5,
    )

    ax = g.ax_joint

    # Set axis limits
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.3, 1.3)

    # # Loop over the hue categories
    # for hue_value in df_in[hue].unique():
    #     hue_data = df_in[df_in[hue] == hue_value]
        
    #     if hue_value == refname:
    #         # For refname, fill the area with kde plot (using density shading)
    #         sns.kdeplot(
    #             data=hue_data,
    #             x=x, y=y,
    #             levels=3, fill=True,  # This will enable filling with density
    #             bw_method=0.5,
    #             color=palette[hue_value],  # Directly use the color here instead of palette
    #             ax=ax
    #         )
    #     else:
    #         # For other hues, draw only the contour lines (no fill)
    #         sns.kdeplot(
    #             data=hue_data,
    #             x=x, y=y,
    #             levels=3, color=palette[hue_value],
    #             ax=ax
    #         )

    # Optional: Additional styling for the plot
    plt.axhline(HighThresPercentile, color='black', linestyle='--')
    plt.axvline(HighThresPercentile, color='black', linestyle='--')

    # 先拿一下当前坐标范围
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    ax.fill_between(
            [HighThresPercentile, x_max],          # x 从 x0 到 x_max
            [HighThresPercentile, HighThresPercentile],             # 下边界 y1 = y0
            [y_max, y_max],       # 上边界 y2 = y_max
            color='#fbe8d3',
            alpha=0.4,
            zorder=0              # 放在所有线下面
        )

    ax.set_aspect('equal', adjustable='box')

    # Set axis labels and title
    plt.xlabel(f'{x}', fontsize=14)
    plt.ylabel(f'{y}', fontsize=14)

    # Save the figure
    plt.savefig(savepath, dpi=DPI, format=FIGFMT, bbox_inches='tight')
    plt.close()



