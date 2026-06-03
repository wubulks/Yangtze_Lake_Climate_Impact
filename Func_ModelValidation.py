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



def Plot_ModelValidation_Seasonal(casename: str,
                                  reflist: List[str],
                                  varname: str,
                                  lon2d: xr.DataArray,
                                  lat2d: xr.DataArray,
                                  OutDir: str,
                                  FigOutDir: str,
                                  lkinfos: Dict[str, Any])-> None:
    """绘制模型验证的季节性箱线图和空间分布图
    参数:
    caselist -- 案例名称列表 (list)，如 ["Lake", "NoLake"]
    varname -- 变量名称 (str)，如 "T2"
    varinfo -- 变量信息 (TPC.varInfo)
    seasonlist -- 季节列表 (list)，如 ["DJF", "MAM", "JJA", "SON"]
    lon2d -- 经度二维场 (DataArray)
    lat2d -- 纬度二维场 (DataArray)
    OutDir -- 输出目录 (str)
    FigOutDir -- 图形输出目录 (str)
    """
    seasons = TU.get_seasons()
    var_info = TYCM.Variable_Infos(varname)
    mean_levels = TYCM.Seasonal_Mean_Cmap(varname)
    diff_levels = TYCM.Seasonal_Diff_Cmap(varname)
    ds_dict = TIO.read_seasonal_metrics(casename, varname, reflist, OutDir)
    mask = TIO.read_mask(f"{OutDir}/mask_all.nc")
    print(reflist)
    datalist = [casename, * reflist]
    FigOutDir_var = f"{FigOutDir}/{varname}/Single"
    os.makedirs(FigOutDir_var, exist_ok=True)
    varInfo = TPC.varInfo(longname=var_info['longname'], unit=var_info['bunit'], abbr=var_info['abbr'])
    label = f"{varInfo.longname} ({varInfo.unit})"
    tasks = []
    if varname == "T2m":
        addmark = "MAE"
    elif varname == "Prec":
        addmark = "NMAE"
    elif varname == "Q2m":
        addmark = "RB"
    for season in seasons:
        meancfg = TPC.mapConfig(levs=mean_levels['levels'][season], cmap=mean_levels['cmap'])
        for dataname in datalist:
            data_in = ds_dict[season][dataname] 
            savepath = f"{FigOutDir_var}/Mean_{casename}_{varname}_{dataname}_seasonal_{season}.{FIGFMT}"
            if dataname != casename:
                metrics_df = pd.read_excel(f"{OutDir}/{casename}/Perform_Mean_{casename}_{dataname}_{varname}_{season}.xlsx")
                tasks.append((TPAM.plot_spatial_map, (data_in, dataname, season, lon2d, lat2d, mask, lkinfos, meancfg, savepath, metrics_df, addmark, True)))
                tasks.append((TPAM.plot_spatial_map, (data_in, "", season, lon2d, lat2d, mask, lkinfos, meancfg, savepath, metrics_df, addmark, True)))
            else:
                tasks.append((TPAM.plot_spatial_map, (data_in, dataname, season, lon2d, lat2d, mask, lkinfos, meancfg, savepath, None, addmark, True)))
                tasks.append((TPAM.plot_spatial_map, (data_in, "", season, lon2d, lat2d, mask, lkinfos, meancfg, savepath, None, addmark, True)))
        savepath = f'{FigOutDir_var}/Mean_Map_{casename}_{varname}_Seasonal_{season}_HColorbar.{FIGFMT}'
        tasks.append((TPCB.plot_spatial_cbar_core_H, (meancfg, 14.0, label, savepath, 30, 36, 0.03, 'max', '{:.0f}')))            
        diffcfg = TPC.mapConfig(levs=diff_levels['levels'][season], cmap=diff_levels['cmap'])
        for refname in reflist:
            refdata = ds_dict[season][refname]
            modeldata = ds_dict[season][casename]
            diffdata = ds_dict[season][f"{casename}-{refname}"]
            diffdata = modeldata - refdata
            metrics_df = pd.read_excel(f"{OutDir}/{casename}/Perform_Mean_{casename}_{refname}_{varname}_{season}.xlsx")
            savepath = f"{FigOutDir_var}/Diff_{casename}_{varname}_{refname}_seasonal_{season}.{FIGFMT}"
            tasks.append((TPAM.plot_spatial_map, (diffdata, refname, season, lon2d, lat2d, mask, lkinfos, diffcfg, savepath, metrics_df, addmark, True)))
        savepath = f'{FigOutDir_var}/Diff_Map_{casename}_{varname}_Seasonal_{season}_HColorbar.{FIGFMT}'
        tasks.append((TPCB.plot_spatial_cbar_core_H, (diffcfg, 14.0, label, savepath, 30, 36, 0.03, 'both')))            

    # 并行执行
    ntasks = len(tasks)
    with Parallel(n_jobs=8, backend="loky",  pre_dispatch="2*n_jobs", return_as="generator") as parallel:
        gen = parallel(
            delayed(func)(*args) for func, args in tasks
        )
        res_list = [p for p in tqdm(gen, total= ntasks,
                                        desc=f"    ➠ Parallel", unit="task",
                                        dynamic_ncols=True)]
    print("    All parallel plots done.")




def Merge_ModelValidation_Seasonal(casename: str,
                                   reflist: List[str],
                                   varname: str,
                                   FigOutDir: str)-> None:
    """合并模型验证的季节性统计结果到一个Excel文件中
    参数:
    caselist -- 案例名称列表 (list)，如 ["Lake", "NoLake"]
    varname -- 变量名称 (str)，如 "T2"
    FigOutDir -- 图形输出目录 (str)
    """
    seasons = TU.get_seasons()
    mapcrop_params = TYCM.MapPlot_CropParams_noColorbar()
    figbox_space = TYCM.Merge_Fig_Space_Params()
    cbar_space = TYCM.MapCbar_CropParams_H()


    #############################
    # Mean maps
    #############################
    target = 'Mean'
    cols_space = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    rows_space = [0.01, 0.01, 0.01]
    rows = [ [] for _ in range(len([casename, *reflist, "Colorbar"]))]
    for i, dataname in enumerate([casename, *reflist]):
        rowpos = i
        for j, season in enumerate(seasons):
            figpath = f"{FigOutDir}/{varname}/Single/{target}_{casename}_{varname}_{dataname}_seasonal_{season}.{FIGFMT}"
            print(figpath)
            croped_fig = TIT.crop_image_from_path(figpath, crop_params=mapcrop_params, mode="ratio")
            rows[rowpos].append(croped_fig)
    # colorbar
    rowpos = -1
    sigsuffix=""
    checkmethod=""
    for j, season in enumerate(seasons):
        figpath = f"{FigOutDir}/{varname}/Single/Mean_Map_{casename}_{varname}_Seasonal_{season}_HColorbar.{FIGFMT}"
        cropped_cbar = TIT.crop_image_from_path(figpath, crop_params=cbar_space, mode="ratio")
        # cropped_cbar = TIT.adjust_image_to_ref_canvas(target_img=cropped_cbar, ref_img=croped_fig, axis='width')
    rows[rowpos].append(cropped_cbar)
    seas_img = TIT.merge_images_Row(
        rows_images=rows,
        cols_space=cols_space,
        rows_space=rows_space,
        box_space=figbox_space,
        background_color='#FFFFFFFF',
        space_mode="ratio",
        alignment=["justify"]*(len(rows)-1) + ["center"],
        draw_ticks=False, tick_step=0.01,
        font_path="/share/home/dq048/.local/share/fonts/NotoSans-Bold.ttf",
    )
    savepath = f'{FigOutDir}/{varname}/{target}_Map_Seasonal_Comparison_{casename}_{varname}.{FIGFMT}'
    seas_img.save(savepath, dpi=[DPI]*16)

    #############################
    # Diff maps
    #############################
    target = 'Diff'
    cols_space = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    rows_space = [0.01, 0.01]
    rows = [ [] for _ in range(len([*reflist, "Colorbar"]))]
    for i, dataname in enumerate([*reflist]):
        rowpos = i
        for j, season in enumerate(seasons):
            figpath = f"{FigOutDir}/{varname}/Single/{target}_{casename}_{varname}_{dataname}_seasonal_{season}.{FIGFMT}"
            print(figpath)
            croped_fig = TIT.crop_image_from_path(figpath, crop_params=mapcrop_params, mode="ratio")
            rows[rowpos].append(croped_fig)
    # colorbar
    rowpos = -1
    sigsuffix=""
    checkmethod=""
    for j, season in enumerate(seasons):
        figpath = f"{FigOutDir}/{varname}/Single/Diff_Map_{casename}_{varname}_Seasonal_{season}_HColorbar.{FIGFMT}"
        cropped_cbar = TIT.crop_image_from_path(figpath, crop_params=cbar_space, mode="ratio")
        # cropped_cbar = TIT.adjust_image_to_ref_canvas(target_img=cropped_cbar, ref_img=croped_fig, axis='width')
    rows[rowpos].append(cropped_cbar)

    seas_img = TIT.merge_images_Row(
        rows_images=rows,
        cols_space=cols_space,
        rows_space=rows_space,
        box_space=figbox_space,
        background_color='#FFFFFFFF',
        space_mode="ratio",
        alignment=["justify"]*(len(rows)-1) + ["center"],
        draw_ticks=False, tick_step=0.01,
        font_path="/share/home/dq048/.local/share/fonts/NotoSans-Bold.ttf",
    )
    savepath = f'{FigOutDir}/{varname}/{target}_Map_Seasonal_Comparison_{casename}_{varname}.{FIGFMT}'
    TIT.save(seas_img, savepath, dpi=DPI)


