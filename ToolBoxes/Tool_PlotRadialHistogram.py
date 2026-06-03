import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import warnings
import matplotlib.cm as cm
import math
from matplotlib.patches import Arc
import matplotlib.colors as mcolors
from typing import List, Tuple, Union, Any
warnings.filterwarnings('ignore')

# 自定义工具箱
import ToolBoxes.Tool_PlotConfig as TPC
FIGFMT = TPC.FIGFMT
DPI    = TPC.DPI_medium
BASEDATA = TPC.BASEDATA


def is_dark(color, threshold=0.1) -> float:
    """
    计算颜色的相对亮度（0~1），越大越亮。
    支持 '#RRGGBB' / 'tab:blue' / (r,g,b) 等 matplotlib 可识别颜色。
    依据 WCAG 的 sRGB -> linear 转换与 luminance 公式。
    """
    r, g, b = mcolors.to_rgb(color)  # -> 0~1

    def to_linear(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r_lin, g_lin, b_lin = map(to_linear, (r, g, b))
    # WCAG relative luminance

    light = 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin

    return light < threshold



def radial_histogram(data_in, pry_cat_colname, sec_cat_colname, data_levels, savepath=None,optional_kwargs=None,**kwargs):
    '''
    
                --- 此函数用于绘制极坐标堆叠条形图 --- 
    数据分三级，一级为主要分类，二级为次要分类，三级为数据级别(堆叠分类)。
    
    必选参数：
        data_in             (DataFrame) : 包含数据的DataFrame                            
        pry_cat_colname           (str) : 主要分类的列名                             
        sec_cat_colname           (str) : 次要分类的列名                             
        data_levels              (list) : 数据级别的列名列表
        
    可选参数：                            
        primary_cats             (list) : 主要分类的列表。默认为数据中的所有唯一主要分类              
        secondary_cats           (list) : 次要分类的列表。默认为数据中的所有唯一次要分类              
        inner_circle_radius     (float) : 内圆的半径。默认为0                            
        blank_length              (int) : 每个主要分类之间的空白条形数。默认为2                 
        levels_color             (list) : 每个数据级别的颜色。默认为蓝色调色板的颜色                
        radii                    (list) : 每个数据级别的半径。默认为数据级别的最大值和总和的最大值之间的5个等距值 
        ylims                    (list) : [ymin, ymax], y轴的最小值和最大值。默认为数据级别的最大值和总和的最大值
        sort_by_Total            (bool) : 是否按总和对次要分类进行排序。默认为True               
        sort_ascending           (bool) : 是否按升序对次要分类进行排序。默认为False              
        bar_linestyle             (str) : 条形的线条样式。默认为虚线                       
        bar_linewidth           (float) : 条形的线条宽度。默认为1                          
        bar_edgecolor             (str) : 条形的边缘颜色。默认为白色                       
        bar_alpha               (float) : 条形的透明度。默认为1
        circle_on                (bool) : 是否绘制圆圈。默认为True                           
        circle_label_fontsize     (int) : 圆圈标签的字体大小。默认为10                     
        circle_label_fontcolor    (str) : 圆圈标签的字体颜色。默认为黑色   
        circle_label_fontweight   (str) : 圆圈标签的字体粗细。默认为normal                  
        circle_linestyle          (str) : 圆圈的线条样式。默认为虚线                       
        circle_linewidth        (float) : 圆圈的线条宽度。默认为1                          
        circle_edgecolor          (str) : 圆圈的边缘颜色。默认为灰色                       
        circle_alpha            (float) : 圆圈的透明度。默认为1                           
        circle_fill              (bool) : 是否填充圆圈。默认为False                      
        bottom_circle_linestyle   (str) : 底部圆圈的线条样式。默认为实线                     
        bottom_circle_linewidth (float) : 底部圆圈的线条宽度。默认为2                        
        bottom_circle_linecolor   (str) : 底部圆圈的线条颜色。默认为黑色                     
        pry_fontsize              (int) : 主要分类标签的字体大小。默认为13                   
        pry_fontcolor             (str) : 主要分类标签的字体颜色。默认为黑色       
        pry_fontweight            (str) : 主要分类标签的字体粗细。默认为bold            
        sec_fontsize              (int) : 次要分类标签的字体大小。默认为10                   
        sec_fontcolor             (str) : 次要分类标签的字体颜色。默认为黑色
        pry_fontweight            (str) : 次要分类标签的字体粗细。默认为normal                   
        title                     (str) : 图表的标题                               
        title_fontsize            (int) : 图表标题的字体大小。默认为15                     
        title_fontcolor           (str) : 图表标题的字体颜色。默认为黑色 
        title_fontweight          (str) : 图表标题的字体粗细。默认为normal
        legend_on                (bool) : 是否显示图例。默认为True     
        legend_label_fontsize     (int) : 图例标签的字体大小。默认为10               
        legend_bbox              (list) : 图例的位置[横坐标，纵坐标]。默认为[0.5, 0.5]。[0,0]为左下角，[1,1]为右上角
        offset_pry_text         (float) : 主要分类标签的偏移量。默认为-5
        offset_inner            (float) : 内圆圈的偏移量。默认为-2
        stack_on                 (bool) : 【新增】是否启用堆叠功能。默认为True。
                                          若为False，则 bar 长度为 data_levels 中所有列的总和。
        bar_color_single          (str) : 【新增】当 stack_on=False 时，单个 Bar 的颜色。
                                          默认为 levels_color 中的第一个颜色。
        color_by_colname          (str) : 【新增】当 stack_on=False 时，用于决定 Bar 颜色的数值列名。
                                           如果指定，则 Bar 颜色将根据该列值进行渐变映射。
        color_cmap                (str) : 【新增】颜色映射的名称 (如 'viridis', 'RdPu')。
    
    '''
    if optional_kwargs and isinstance(optional_kwargs, dict):
        # 将位置参数中的字典内容合并到 **kwargs 中，
        # 如果 **kwargs 中有重复键，以 **kwargs 中的值为准 (保留函数调用时的命名参数优先级)
        temp_kwargs = optional_kwargs.copy()
        temp_kwargs.update(kwargs)
        kwargs = temp_kwargs

    primary_cats = kwargs.get('primary_cats', data_in[pry_cat_colname].unique())
    filtered_data = data_in[data_in[pry_cat_colname].isin(primary_cats)]
    secondary_cats = kwargs.get('secondary_cats', filtered_data[sec_cat_colname].unique())
    radii = kwargs.get('radii', None)
    ylims = kwargs.get('ylims', None)
    title = kwargs.get('title', None)
    levels_color = kwargs.get('levels_color', None)
    inner_circle_radius = kwargs.get('inner_circle_radius', 5)
    blank_length = kwargs.get('blank_length', 2)
    sort_ascending = kwargs.get('sort_ascending', False)
    sort_by_Total = kwargs.get('sort_by_Total', True)
    bar_linestyle = kwargs.get('bar_linestyle', '-')
    bar_linewidth = kwargs.get('bar_linewidth', 1)
    bar_edgecolor = kwargs.get('bar_edgecolor', 'white')
    bar_alpha = kwargs.get('bar_alpha', 1)
    circle_on = kwargs.get('circle_on', True)
    circle_linestyle = kwargs.get('circle_linestyle', '--')
    circle_linewidth = kwargs.get('circle_linewidth', 0.5)
    circle_edgecolor = kwargs.get('circle_edgecolor', 'grey')
    circle_alpha = kwargs.get('circle_alpha', 1)
    circle_fill = kwargs.get('circle_fill', False)
    bottom_circle_linestyle = kwargs.get('bottom_circle_linestyle', '-')
    bottom_circle_linewidth = kwargs.get('bottom_circle_linewidth', 1.5)
    bottom_circle_linecolor = kwargs.get('bottom_circle_linecolor', 'black')
    pry_fontsize = kwargs.get('pry_fontsize', 7)
    sec_fontsize = kwargs.get('sec_fontsize', 6)
    title_fontsize = kwargs.get('title_fontsize', 15)
    circle_label_fontsize = kwargs.get('circle_label_fontsize', 8)
    pry_fontcolor = kwargs.get('pry_fontcolor', 'black')
    sec_fontcolor = kwargs.get('sec_fontcolor', 'black')
    title_fontcolor = kwargs.get('title_fontcolor', 'black')
    bar_label_fontcolor = kwargs.get('bar_label_fontcolor', 'black')
    circle_label_fontcolor = kwargs.get('circle_label_fontcolor', 'black')
    pry_fontweight = kwargs.get('pry_fontweight', 'bold')
    sec_fontweight = kwargs.get('sec_fontweight', 'normal')
    title_fontweight = kwargs.get('title_fontweight', 'normal')
    circle_label_fontweight = kwargs.get('circle_label_fontweight', 'normal')
    legend_on = kwargs.get('legend_on', True)
    legend_label_fontsize = kwargs.get('legend_label_fontsize', 10)
    legend_bbox = kwargs.get('legend_bbox', [0.5, 0.5])
    offset_pry_text = kwargs.get('offset_pry_text', -3)
    offset_sec_text = kwargs.get('offset_sec_text', 50)
    offset_inner = kwargs.get('offset_inner', -2)
    stack_on = kwargs.get('stack_on', True)
    bar_color_single = kwargs.get('bar_color_single', None)
    color_by_colname = kwargs.get('color_by_colname', None) # 新增
    color_cmap = kwargs.get('color_cmap', 'viridis')  # 新增，默认 'viridis'
    value_levels = kwargs.get("levels", kwargs.get("level", None))  # 允许 level 或 levels
    value_cmap   = kwargs.get("cmap", kwargs.get("color_cmap", "viridis"))  # 允许 cmap 或复用 color_cmap
    extend       = kwargs.get("extend", "neither")  # "neither"|"min"|"max"|"both"
    unit_sec_text = kwargs.get("unit_sec_text", "")  # 次要分类标签的单位后缀，默认为空字符串

    if len(data_levels) == 0:
        raise ValueError("data_levels 列表不能为空。请至少提供一个数据级别的列名。")
    elif len(data_levels) == 1 and stack_on:
        print("警告：仅提供了一个数据级别列，堆叠模式下无法显示堆叠效果。建议提供多个数据级别列或关闭堆叠模式 (stack_on=False)。")
        stack_on = False

    # 计算主要和次要分类的数量
    n_pry = len(primary_cats)
    n_sec = len(secondary_cats)

    # 计算数据级别的最大值和总和的最大值
    max_level = data_in[data_levels].max().max()
    data_in['total'] = data_in[data_levels].sum(axis=1)
    max_sum = data_in['total'].max()
    min_sum = data_in['total'].min()
    
    # 向上取整到最接近的 10 的倍数
    if ylims is None:
        ymax = math.ceil(max_sum / 10) * 10
        ymin = math.floor(min_sum / 10) * 10
    else:
        ymin, ymax = ylims
        
    if radii is None:
        radii = np.linspace(ymin, ymax, 5).tolist()
    
    # 计算每个条形的宽度
    width_per_bar = (2 * np.pi) / (n_sec + ((n_pry+2) * blank_length))
    
    # 创建极坐标图

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1) 

    # 声明颜色相关的局部变量
    __bar_color_single = None
    __color_cmap = None
    __color_norm = None

    # 创建颜色级别
    if stack_on:
        # 堆叠模式：使用 levels_color 列表
        levels_color = kwargs.get('levels_color', sns.color_palette("Blues_r", len(data_levels)))
        if len(levels_color) < len(data_levels):
             levels_color.extend(sns.color_palette("Blues_r", len(data_levels) - len(levels_color)))
             
    elif (not stack_on) and color_by_colname:
        if color_by_colname not in data_in.columns:
            raise ValueError(f"用于着色的列名 '{color_by_colname}' 在数据中不存在。")

        # cmap
        __color_cmap = cm.get_cmap(value_cmap) if isinstance(value_cmap, str) else value_cmap

        color_data = data_in[color_by_colname].to_numpy(dtype=float)

        if value_levels is not None:
            levs = np.asarray(value_levels, dtype=float)
            if levs.ndim != 1 or levs.size < 2:
                raise ValueError("levels 必须是一维边界数组，且长度 >= 2")
            if not np.all(np.diff(levs) > 0):
                levs = np.unique(levs)
                if levs.size < 2 or not np.all(np.diff(levs) > 0):
                    raise ValueError("levels 必须严格递增")

            __color_norm = mcolors.BoundaryNorm(levs, ncolors=__color_cmap.N, clip=False)
            __color_levels = levs  # 供 colorbar 用
        else:
            __color_norm = mcolors.Normalize(vmin=np.nanmin(color_data), vmax=np.nanmax(color_data))
            __color_levels = None
        
    else:
        # 非堆叠 + 单一颜色模式：使用 bar_color_single 或 levels_color[0]
        levels_color_single = kwargs.get('levels_color', sns.color_palette("Blues_r", 1))
        if bar_color_single is None:
            __bar_color_single = levels_color_single[0]
        else:
            __bar_color_single = bar_color_single # 使用外部传入的单一颜色

    # 绘制每个数据级别的圆圈
    if circle_on:
        for i, radius in enumerate(radii):
            circle = plt.Circle((0, 0), radius + inner_circle_radius, transform=ax.transData._b, color=circle_edgecolor, 
                                fill=circle_fill, linestyle=circle_linestyle, linewidth=circle_linewidth, alpha=circle_alpha)
            ax.add_artist(circle)

    # 在每个圆圈旁添加文本标签
    if circle_on:
        for radius in radii:
            ax.text(0, radius + inner_circle_radius, str(radius), ha='center', va='center', color=circle_label_fontcolor,
                    fontsize=circle_label_fontsize, rotation_mode='anchor', fontweight=circle_label_fontweight)

    # 初始化起始角度
    if blank_length % 2 == 0:
        angle = width_per_bar * (blank_length + 0.5)
    else:
        angle = width_per_bar * (blank_length+1)

    # 绘制每个主要和次要分类的条形图
    for primary_cat in primary_cats:
        if sort_by_Total:
            primary_cat_data = data_in[data_in[pry_cat_colname] == primary_cat].sort_values(by=['total'], ascending=sort_ascending)
        else:
            primary_cat_data = data_in[data_in[pry_cat_colname] == primary_cat]
        sec_agl = []
        for secondary_cat in primary_cat_data[sec_cat_colname].unique():
            secondary_cat_data = primary_cat_data[primary_cat_data[sec_cat_colname] == secondary_cat]
            if stack_on:
                # 模式一：堆叠 (Stacking) 模式
                bottom = inner_circle_radius
                for j, data_level in enumerate(data_levels):
                    value = float(secondary_cat_data[data_level].iloc[0])
                    # value = secondary_cat_data[data_level] # 注意：这里应取单行值，而不是 sum()，因为上面已经按 unique 过滤
                    ax.bar(angle, value, width=width_per_bar, color=levels_color[j], bottom=bottom, 
                            edgecolor=bar_edgecolor, linewidth=bar_linewidth, alpha=bar_alpha, 
                            linestyle=bar_linestyle, label=data_level)
                    bottom += value
                total_value = bottom - inner_circle_radius
            else:
                # 模式二：非堆叠 (Non-Stacking) 模式
                # total_value = secondary_cat_data['total'] # Bar 的长度
                total_value = float(secondary_cat_data["total"].iloc[0])
                
                # 确定 Bar 的颜色
                bar_color = __bar_color_single
                if color_by_colname:
                    color_value = float(secondary_cat_data[color_by_colname].iloc[0])
                    # 像 contourf 一样：ScalarMappable + BoundaryNorm/Normalize
                    _sm = cm.ScalarMappable(norm=__color_norm, cmap=__color_cmap)
                    bar_color = _sm.to_rgba(color_value)
                
                # 绘制单个 Bar
                ax.bar(angle, total_value, width=width_per_bar, color=bar_color, 
                        bottom=inner_circle_radius, 
                        edgecolor=bar_edgecolor, linewidth=bar_linewidth, alpha=bar_alpha, 
                        linestyle=bar_linestyle, label=color_by_colname if color_by_colname else 'Total')
           
            # 添加与bar平行的text
            text_angle_deg = -np.degrees(angle)+90
            alignment = {'va': 'center', 'ha': 'left'}
            # 检查文本是否位于圆的下半部分
            if text_angle_deg < -90 and text_angle_deg >= -270:
                text_angle_deg += 180
                alignment['ha'] = 'right'
            
            if alignment['ha'] == 'right':
                if unit_sec_text:
                    text = f"({total_value:.2f}{unit_sec_text}) {secondary_cat} "
                else:
                    text = f"{secondary_cat}"
            else:
                if unit_sec_text:
                    text = f"{secondary_cat} ({total_value:.2f}{unit_sec_text})"
                else:
                    text = f"{secondary_cat}"
            
            # 添加次要分类标签
            if is_dark(bar_color):
                sec_fontcolor = 'white'
            else:
                sec_fontcolor = 'black'
            ax.text(angle, inner_circle_radius+offset_sec_text, text, rotation=text_angle_deg, rotation_mode='anchor', **alignment, 
                    fontsize=sec_fontsize, fontweight=sec_fontweight, color=sec_fontcolor)            
            
            sec_agl.append(angle)
            angle += width_per_bar
        angle += width_per_bar * blank_length
        # 在每个主要分类旁添加文本标签
        if len(sec_agl):
            angles = np.linspace(sec_agl[0]-width_per_bar/2, sec_agl[-1]+width_per_bar/2, 100)
            ax.plot(angles, [inner_circle_radius+offset_inner] * len(angles), color=bottom_circle_linecolor, linewidth=bottom_circle_linewidth)
            center_angle = np.mean(sec_agl)
            text_angle_deg = -np.degrees(center_angle)
            alignment = {'va': 'center', 'ha': 'center'}
            # 检查文本是否位于圆的下半部分
            if text_angle_deg < -90 and text_angle_deg >= -270:
                text_angle_deg += 180
            # 添加主要分类标签
            ax.text(center_angle, inner_circle_radius+offset_pry_text, primary_cat, 
                        rotation=text_angle_deg, **alignment,
                        rotation_mode='anchor', fontsize=pry_fontsize,fontweight=pry_fontweight, color=pry_fontcolor)

    ax.set_ylim(ymin, ymax + inner_circle_radius)
    if title is not None:
        plt.title(title, fontsize=title_fontsize, color=title_fontcolor, fontweight=title_fontweight)
    # 图例处理：单条模式下只显示一个图例
    if stack_on:
        # 堆叠模式：显示矩形图例
        handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in levels_color[:len(data_levels)]]
        labels = data_levels
        if legend_on:
             plt.legend(handles, labels, loc='center', bbox_to_anchor=kwargs.get('legend_bbox', [0.5, 0.5]), fontsize=kwargs.get('legend_label_fontsize', 10))
    
    elif color_by_colname and legend_on:
        # 非堆叠 + 渐变着色模式：显示颜色条 (Colorbar)
        _sm = cm.ScalarMappable(norm=__color_norm, cmap=__color_cmap)
        _sm.set_array([])

        cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])

        if __color_levels is not None:
            cbar = fig.colorbar(
                _sm, cax=cbar_ax, orientation="vertical",
                ticks=__color_levels, boundaries=__color_levels,
                spacing="uniform", extend=extend
            )
        else:
            cbar = fig.colorbar(_sm, cax=cbar_ax, orientation="vertical")

        cbar.set_label(f"Color by: {color_by_colname}")
    elif legend_on:
        # 非堆叠 + 单一颜色模式：显示单个 Bar 的图例
        handles = [plt.Rectangle((0, 0), 1, 1, color=__bar_color_single)]
        labels = ['Total'] 
        plt.legend(handles, labels, loc='center', bbox_to_anchor=kwargs.get('legend_bbox', [0.5, 0.5]), fontsize=kwargs.get('legend_label_fontsize', 10))
    
    plt.grid(False)
    plt.axis('off')
    # 在函数的末尾，plt.show() 之前添加保存逻辑：
    if savepath is not None:
        try:
            fig.savefig(savepath, dpi=DPI, format=FIGFMT, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
        except Exception as e:
            print(f"保存图片失败: {e}")
