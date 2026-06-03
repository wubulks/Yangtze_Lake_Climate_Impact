import os
import time
import calendar
import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm
import statsmodels.api as sm
from scipy.stats import pearsonr
from multiprocessing import Pool
from dataclasses import dataclass
from joblib import Parallel, delayed
from typing import Literal, Optional, Tuple, Dict, Any, Sequence, List
from statsmodels.stats.multitest import multipletests

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from typing import List, Any

# 自定义模块
import ToolBoxes.Utils as TU
import ToolBoxes.Tool_ImageToolkit as TIT
import ToolBoxes.Tool_InputOutput as TIO
import ToolBoxes.Tool_PlotBox as TPB
import ToolBoxes.Tool_PlotColorBar as TPCB
import ToolBoxes.Tool_PlotConfig as TPC
import ToolBoxes.Tool_DataPrepare as TDP
import ToolBoxes.Tool_InputOutput as TIO
import ToolBoxes.Tool_PlotAreaMap as TPAM
import ToolBoxes.Tool_YangtzeColorMap as TYCM
FIGFMT = TPC.FIGFMT
DPI    = TPC.DPI_medium


def Plot_SpatialMap_2DVar(
            casename: List[str], varname: str, xarr_in: xr.DataArray,
            lon2d: np.ndarray, lat2d: np.ndarray,
            FigOutDir: str, lkinfos: Any) -> None:
                      
    """绘制 PBLH 空间分布图"""
    target = "SpatialMap"
    FigOutDir_var = f'{FigOutDir}/Single'
    os.makedirs(FigOutDir_var, exist_ok=True)
    seasons = TU.get_seasons()
    var_info = TYCM.Variable_Infos(varname)
    mean_levels = TYCM.Seasonal_Mean_Cmap(varname)
    varInfo = TPC.varInfo(longname=var_info['longname'], unit=var_info['bunit'], abbr=var_info['abbr'])
    label = f"{varInfo.longname} ({varInfo.unit})"
    oceanmask = lkinfos["ocean"]
    mask2d = oceanmask==0

    tasks = []
    for season in seasons:
        meancfg = TPC.mapConfig(levs=mean_levels['levels'][season], cmap=mean_levels['cmap'])
        xarr_sel = xarr_in.sel(time=xarr_in.time.dt.season == season).squeeze(drop=True)
        xarr_clim = xarr_sel.mean("time").squeeze(drop=True)
        savepath = f"{FigOutDir_var}/{target}_{varname}_{casename}_seasonal_{season}.{FIGFMT}"
        # tasks.append((TPAM.plot_spatial_map, (xarr_clim, "", season, lon2d, lat2d, mask2d, lkinfos, meancfg, savepath, None)))
        savepath = f'{FigOutDir_var}/{target}_{varname}_{casename}_seasonal_{season}_HColorbar.{FIGFMT}'
        tasks.append((TPCB.plot_spatial_cbar_core_H, (meancfg, 9.0, label, savepath, 18, 24, 0.03, 'max')))            
        savepath = f'{FigOutDir_var}/{target}_{varname}_{casename}_seasonal_{season}_VColorbar.{FIGFMT}'
        tasks.append((TPCB.plot_spatial_cbar_core_V, (meancfg, 6.0, label, savepath, 16, 18, 0.05, 'max', '{:.0f}')))            
    savepath = f'{FigOutDir_var}/{target}_{varname}_{casename}_VColorbar.{FIGFMT}'
    tasks.append((TPCB.plot_spatial_cbar_core_V, (meancfg, 11.0, label, savepath, 20, 26, 0.035, 'max')))            
    savepath = f'{FigOutDir_var}/{target}_{varname}_{casename}_HColorbar.{FIGFMT}'
    tasks.append((TPCB.plot_spatial_cbar_core_H, (meancfg, 12.0, label, savepath, 26, 32, 0.03, 'max')))            

    # 并行执行
    ntasks = len(tasks)
    with Parallel(n_jobs=16, backend="loky",  pre_dispatch="2*n_jobs", return_as="generator") as parallel:
        gen = parallel(
            delayed(func)(*args) for func, args in tasks
        )
        res_list = [p for p in tqdm(gen, total= ntasks,
                                        desc=f"    ➠ Parallel", unit="task",
                                        dynamic_ncols=True)]
    print("    All parallel plots done.")



def Plot_SpatialMap_UV(
            casename: List[str], varname: str, 
            U_data: xr.DataArray, V_data: xr.DataArray,
            lon2d: np.ndarray, lat2d: np.ndarray,
            FigOutDir: str, lkinfos: Any, level: float | None = None) -> None:
    """绘制 PBLH 空间分布图"""
    target = "SpatialMap"
    FigOutDir_var = f'{FigOutDir}/Single'
    os.makedirs(FigOutDir_var, exist_ok=True)
    seasons = TU.get_seasons()
    var_info = TYCM.Variable_Infos(varname)
    mean_levels = TYCM.Seasonal_Mean_Cmap(varname)
    varInfo = TPC.varInfo(longname=var_info['longname'], unit=var_info['bunit'], abbr=var_info['abbr'])
    label = f"{varInfo.longname} ({varInfo.unit})"

    tasks = []
    for season in seasons:
        meancfg = TPC.mapConfig(levs=mean_levels['levels'][season], cmap=mean_levels['cmap'])
        U_sel = U_data.sel(time=U_data.time.dt.season == season).squeeze(drop=True)
        U_clim = U_sel.mean("time").squeeze(drop=True).values
        V_sel = V_data.sel(time=V_data.time.dt.season == season).squeeze(drop=True)
        V_clim = V_sel.mean("time").squeeze(drop=True).values
        if level is None:
            mapsavepath = f'{FigOutDir_var}/{target}_{varname}_{casename}_seasonal_{season}.{FIGFMT}'
            Hcbarsavepath = f'{FigOutDir_var}/{target}_{varname}_{casename}_seasonal_{season}_HColorbar.{FIGFMT}'
            Vcbarsavepath = f'{FigOutDir_var}/{target}_{varname}_{casename}_seasonal_{season}_VColorbar.{FIGFMT}'
        else:
            mapsavepath = f'{FigOutDir_var}/{target}_{varname}_{casename}_{level}hPa_seasonal_{season}.{FIGFMT}'
            Hcbarsavepath = f'{FigOutDir_var}/{target}_{varname}_{casename}_{level}hPa_seasonal_{season}_HColorbar.{FIGFMT}'
            Vcbarsavepath = f'{FigOutDir_var}/{target}_{varname}_{casename}_{level}hPa_seasonal_{season}_VColorbar.{FIGFMT}'
        # tasks.append((TPAM.plot_spatial_map_UV, (U_clim, V_clim, varname, casename, season, target, lon2d, lat2d, lkinfos, mapsavepath, meancfg)))
        tasks.append((TPCB.plot_spatial_cbar_core_H, (meancfg, 9.0, label, Hcbarsavepath, 18, 24, 0.03, 'max')))      
        tasks.append((TPCB.plot_spatial_cbar_core_V, (meancfg, 6.0, label, Vcbarsavepath, 16, 18, 0.05, 'max', '{:.0f}')))          

    if level is None:
        Vcbarsavepath = f'{FigOutDir_var}/{target}_{varname}_{casename}_VColorbar.{FIGFMT}'
        Hcbarsavepath = f'{FigOutDir_var}/{target}_{varname}_{casename}_HColorbar.{FIGFMT}'

    else:
        Vcbarsavepath = f'{FigOutDir_var}/{target}_{varname}_{casename}_{level}hPa_VColorbar.{FIGFMT}'
        Hcbarsavepath = f'{FigOutDir_var}/{target}_{varname}_{casename}_{level}hPa_HColorbar.{FIGFMT}'
    tasks.append((TPCB.plot_spatial_cbar_core_V, (meancfg, 11.0, label, Vcbarsavepath, 20, 26, 0.035, 'max', '{:.0f}')))            
    tasks.append((TPCB.plot_spatial_cbar_core_H, (meancfg, 12.0, label, Hcbarsavepath, 26, 32, 0.03, 'max')))            
    
    print(f"    Total {len(tasks)} tasks for plotting spatial map of {varname}.")

    # 并行执行
    ntasks = len(tasks)
    with Parallel(n_jobs=16, backend="loky",  pre_dispatch="2*n_jobs", return_as="generator") as parallel:
        gen = parallel(
            delayed(func)(*args) for func, args in tasks
        )
        res_list = [p for p in tqdm(gen, total= ntasks,
                                        desc=f"    ➠ Parallel", unit="task",
                                        dynamic_ncols=True)]
    print("    All parallel plots done.")



def Merge_SpatialMap_2DVar(
            casename: str, varname: str, FigOutDir: str, level: float | None = None) -> None:
    """合并 PBLH 空间分布图"""
    seasons = TU.get_seasons()
    mapcrop_params = TYCM.MapPlot_CropParams_noColorbar()
    figbox_space = TYCM.Merge_Fig_Space_Params()
    cbar_space = TYCM.MapCbar_CropParams_H()
    FigOutDir_var = f'{FigOutDir}/Single'
    os.makedirs(FigOutDir_var, exist_ok=True)

    #############################
    # Mean maps
    #############################
    target = 'SpatialMap'
    cols_space = [[0, 0, 0], [0, 0, 0]]
    rows_space = [0.01,]
    rows = [[], []]
    for i, season in enumerate(seasons):
        if level is None:
            mapsavepath = f'{FigOutDir_var}/{target}_{varname}_{casename}_seasonal_{season}.{FIGFMT}'
            cbarsavepath = f'{FigOutDir_var}/{target}_{varname}_{casename}_HColorbar.{FIGFMT}'
        else:
            mapsavepath = f'{FigOutDir_var}/{target}_{varname}_{casename}_{level}hPa_seasonal_{season}.{FIGFMT}'
            cbarsavepath = f'{FigOutDir_var}/{target}_{varname}_{casename}_{level}hPa_HColorbar.{FIGFMT}'
        croped_fig = TIT.crop_image_from_path(mapsavepath, crop_params=mapcrop_params, mode="ratio")
        rows[0].append(croped_fig)
    cropped_cbar = TIT.crop_image_from_path(cbarsavepath, crop_params=cbar_space, mode="ratio")
    rows[1].append(cropped_cbar)

    seas_img = TIT.merge_images_Row(
        rows_images=rows,
        cols_space=cols_space,
        rows_space=rows_space,
        box_space=figbox_space,
        background_color='#FFFFFFFF',
        space_mode="ratio",
        alignment=["justify","center", ],
        draw_ticks=False, tick_step=0.01,
        font_path="/share/home/dq048/.local/share/fonts/NotoSans-Bold.ttf",
    )
    if level is None:
        savepath = f'{FigOutDir}/{target}_Map_Seasonal_Comparison_{casename}_{varname}.{FIGFMT}'
    else:
        savepath = f'{FigOutDir}/{target}_Map_Seasonal_Comparison_{casename}_{varname}_{level}hPa.{FIGFMT}'
    TIT.save(seas_img, savepath, dpi=DPI)



def Plot_Diurnal_Cycle(
            caselist: List[str], varname: str, 
            checkmethod: str, OutDir: str,
            FigOutDir: str, lkinfos: Any) -> None:
    """绘制昼夜循环差异 (线图版)"""
    
    # --- 1. 初始化设置 ---
    FigOutDir_var = f'{FigOutDir}/Single'
    os.makedirs(FigOutDir_var, exist_ok=True)
    
    seasons = TU.get_seasons()
    keep_hours = TU.get_all_hours()
    var_info = TYCM.Variable_Infos(varname)
    varInfo = TPC.varInfo(longname=var_info['longname'], unit=var_info['bunit'], abbr=var_info['abbr'])
    
    # 获取Mask
    oceanmask = lkinfos["ocean"]
    lakearea = lkinfos["all"]
    
    # 定义绘图属性
    plot_styles = {
        'Lake':   {'color': '#f38181', 'marker': 'o', 'label': 'Lake'}, # 红色系
        'NoLake': {'color': '#625772', 'marker': 'D', 'label': 'NoLake'}  # 紫色系
    }
    
    # 建立 Case 名称到 Label 的映射 (假设 caselist[0] 是 Lake, [1] 是 NoLake)
    case_map = {
        'Lake': caselist[0],
        'NoLake': caselist[1]
    }

    # --- 2. 循环季节处理数据 ---
    for season in seasons:
        ds_dict = TIO.read_hourly_significance(caselist, varname, season, keep_hours, checkmethod, OutDir)
        
        data_records = [] # 用于构建 DataFrame

        for hour in keep_hours:
            h_bjt_str = TU.UTC_to_BJT_str(hour)
            bjt_hour = TU.UTC_to_BJT(hour) # 假设返回的是数字 0-23
            
            # 单个时间点的记录
            record = {'hour': bjt_hour}
            
            for label, case_name in case_map.items():
                # 获取原始数据
                raw_data = ds_dict[h_bjt_str][case_name].values
                
                # 时间维平均 (axis=0)
                diff = np.nanmean(raw_data, axis=0)
                
                # 应用 Mask: 屏蔽湖泊区域 (根据原代码逻辑: diff[lakearea == 1] = np.nan)
                # 同时也应用 oceanmask (如果需要)
                if diff.shape == lakearea.shape:
                    diff[lakearea == 0] = np.nan
                
                # 计算区域平均值 (Spatial Mean) -> 这对于线图是必须的
                # 如果你想画总和，请改用 np.nansum
                spatial_mean = np.nanmean(diff)
                
                record[label] = spatial_mean
            
            data_records.append(record)

        # 构建 DataFrame (Index: Hour, Columns: Lake, NoLake)
        df_plot = pd.DataFrame(data_records)
        df_plot = df_plot.set_index('hour').sort_index()

        # --- 3. 开始绘图 ---
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        
        # 绘制两条线
        for label, style in plot_styles.items():
            ax.plot(df_plot.index, df_plot[label], 
                    color=style['color'], 
                    marker=style['marker'], 
                    label=style['label'],
                    linewidth=2,
                    markersize=6)
        
        # 设置坐标轴
        ax.set_xlabel("Hour (BJT)", fontsize=12)
        ax.set_ylabel(f"{varInfo.longname} ({varInfo.unit})", fontsize=12)
        ax.set_title(f"Diurnal Cycle of {varInfo.longname} ({season})", fontsize=14)
        
        # 设置横轴刻度 (0-23)
        ax.set_xticks(np.arange(0, 24, 3)) # 每3小时一个刻度，避免拥挤
        ax.set_xlim(-0.5, 23.5)
        
        # 网格和图例
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='best', frameon=False, fontsize=12)
        
        # 保存图片
        out_name = f"{FigOutDir_var}/DiurnalCycle_{varname}_{season}_Line.png"
        plt.tight_layout()
        plt.savefig(out_name)
        plt.close()
        
        print(f"Figure saved: {out_name}")

        