import os
import gc 
import time
import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm
from scipy import stats
from dataclasses import dataclass
from joblib import Parallel, delayed
from contextlib import contextmanager
from typing import Literal, Optional, Tuple, Dict, Any, Sequence, List
from statsmodels.stats.multitest import multipletests

# 自定义工具箱
import ToolBoxes.Utils as TU
import ToolBoxes.Tool_ImageToolkit as TIT
import ToolBoxes.Tool_YangtzeColorMap as TYCM
import ToolBoxes.Tool_PlotConfig as TPC
FIGFMT = TPC.FIGFMT
DPI    = TPC.DPI_medium



def Merge_Fig_1_RegClimImpact(FigOutDir: str) -> None:
    """
    合并区域气候效应图(气温diff, 气温diurnal, 相对湿度diff, 相对湿度diurnal)
    """
    seasons = TU.get_seasons()
    map_cropparams = TYCM.MapPlot_CropParams_noColorbar()
    figbox_space = {'left': 0.15, 'top': 0.01, 'right': 0.01, 'bottom': 0.01}
    cbar_cropparams = {"left": 0.005, "top": 0.005, "right": 0.3, "bottom": 0.005}
    rose_cropparams = TYCM.RosePlot_CropParams_noColorbar()
    onlysig = True
    suffix = "onlysig" if onlysig else "all"
    target = "RegClimImpact"
    FigOutDir_in = f'{FigOutDir}/RegClimImpact'
    FigOutDir_out = f'{FigOutDir}/Manuscript'
    os.makedirs(FigOutDir_out, exist_ok=True)
    
    cols_space = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    rows_space = [0.01, 0.1, 0.01]
    rows = [[], [], [], []]
    # 第一行：气温diff
    rowpos = 0
    varname = "T2m"
    checkmethod = "Paired_t-test"
    for season in seasons:
        figpath = f'{FigOutDir_in}/{varname}/Single/{target}_Diff_Map_{varname}_{season}_{suffix}_{checkmethod}.{FIGFMT}'
        crop_diff = TIT.crop_image_from_path(figpath, crop_params=map_cropparams, mode="ratio")
        rows[rowpos].append(crop_diff)
    cbarpath = f'{FigOutDir_in}/{varname}/Single/{target}_Diff_Map_{varname}_Seasonal_Colorbar_{suffix}_{checkmethod}.{FIGFMT}'
    crop_cbar = TIT.crop_image_from_path(cbarpath, crop_params=cbar_cropparams, mode="ratio")
    crop_cbar = TIT.adjust_image_to_ref_canvas(target_img=crop_cbar, ref_img=crop_diff, axis="height")
    rows[rowpos].append(crop_cbar)

    # 第二行：气温rose
    rowpos = 1
    varname = "T2m"
    checkmethod = "Paired_t-test"
    for season in seasons:
        figpath = f'{FigOutDir_in}/{varname}/Single/{target}_Diff_Rose_{varname}_{season}_{suffix}_{checkmethod}.{FIGFMT}'
        crop_rose = TIT.crop_image_from_path(figpath, crop_params=rose_cropparams, mode="ratio")
        crop_rose = TIT.adjust_image_to_ref_canvas(target_img=crop_rose, ref_img=crop_diff)
        rows[rowpos].append(crop_rose)

    # 第三行：相对湿度diff
    rowpos = 2
    varname = "RH"
    checkmethod = "Paired_t-test"
    for season in seasons:
        figpath = f'{FigOutDir_in}/{varname}/Single/{target}_Diff_Map_{varname}_{season}_{suffix}_{checkmethod}.{FIGFMT}'
        crop_diff = TIT.crop_image_from_path(figpath, crop_params=map_cropparams, mode="ratio")
        rows[rowpos].append(crop_diff)
    cbarpath = f'{FigOutDir_in}/{varname}/Single/{target}_Diff_Map_{varname}_Seasonal_Colorbar_{suffix}_{checkmethod}.{FIGFMT}'
    crop_cbar = TIT.crop_image_from_path(cbarpath, crop_params=cbar_cropparams, mode="ratio")
    crop_cbar = TIT.adjust_image_to_ref_canvas(target_img=crop_cbar, ref_img=crop_diff, axis="height")
    rows[rowpos].append(crop_cbar)

    # 第四行： 相对湿度rose
    rowpos = 3
    varname = "RH"
    checkmethod = "Paired_t-test"
    for season in seasons:
        figpath = f'{FigOutDir_in}/{varname}/Single/{target}_Diff_Rose_{varname}_{season}_{suffix}_{checkmethod}.{FIGFMT}'
        crop_rose = TIT.crop_image_from_path(figpath, crop_params=rose_cropparams, mode="ratio")
        crop_rose = TIT.adjust_image_to_ref_canvas(target_img=crop_rose, ref_img=crop_diff)
        rows[rowpos].append(crop_rose)

    series_texts = {
        # 气温diff
        'T2m': {'x': 0.03,  'y': 0.20,  "fontsize": 0.023, "text": "2-m air temperature (°C)", "ha": "right", "va": "center", "color": 'black', "fontweight": "bold", "rotation":-90},
        '(a)': {'x': 0.045, 'y': 0.01,  "fontsize": 0.015, "text": "(a)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        '(b)': {'x': 0.275, 'y': 0.01,  "fontsize": 0.015, "text": "(b)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        '(c)': {'x': 0.505, 'y': 0.01,  "fontsize": 0.015, "text": "(c)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        '(d)': {'x': 0.737, 'y': 0.01,  "fontsize": 0.015, "text": "(d)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        # 气温rose
        '(e)': {'x': 0.045, 'y': 0.21,  "fontsize": 0.015, "text": "(e)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        '(f)': {'x': 0.275, 'y': 0.21,  "fontsize": 0.015, "text": "(f)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        '(g)': {'x': 0.505, 'y': 0.21,  "fontsize": 0.015, "text": "(g)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        '(h)': {'x': 0.737, 'y': 0.21,  "fontsize": 0.015, "text": "(h)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        # 相对湿度diff
        'RH' : {'x': 0.03,  'y': 0.71,  "fontsize": 0.023, "text": "Relative humidity (%)", "ha": "right", "va": "center", "color": "black", "fontweight": "bold", "rotation":-90},
        '(i)': {'x': 0.045, 'y': 0.52,  "fontsize": 0.015, "text": "(i)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        '(j)': {'x': 0.275, 'y': 0.52,  "fontsize": 0.015, "text": "(j)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        '(k)': {'x': 0.505, 'y': 0.52,  "fontsize": 0.015, "text": "(k)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        '(l)': {'x': 0.737, 'y': 0.52,  "fontsize": 0.015, "text": "(l)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        # 相对湿度rose
        '(m)': {'x': 0.045, 'y': 0.718, "fontsize": 0.015, "text": "(m)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        '(n)': {'x': 0.275, 'y': 0.718, "fontsize": 0.015, "text": "(n)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        '(o)': {'x': 0.505, 'y': 0.718, "fontsize": 0.015, "text": "(o)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        '(p)': {'x': 0.737, 'y': 0.718, "fontsize": 0.015, "text": "(p)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
    }

    seas_img = TIT.merge_images_Row(
        rows_images=rows,
        cols_space=cols_space,
        rows_space=rows_space,
        box_space=figbox_space,
        background_color='#FFFFFF',
        space_mode="ratio",
        alignment=["left","left", "left", "left"],
        draw_ticks=False, tick_step=0.01,
        texts=series_texts,
        font_path="/home/wumej22/.local/share/fonts/NotoSans-Bold.ttf",
    )

    # # 2. 保存图像（自动根据扩展名选择保存方式）
    # savepath = f'{FigOutDir_out}/Fig_1_{target}_Seasonal_Comparison.pdf'
    # TIT.save(seas_img, savepath, dpi=DPI)  # 会自动使用reportlab保存PDF

    # 或者保存为PNG
    savepath_png = f'{FigOutDir_out}/Fig_1_{target}_Seasonal_Comparison.{FIGFMT}'
    TIT.save(seas_img, savepath_png, dpi=DPI)
    savepath_pdf = f'{FigOutDir_out}/Fig_1_{target}_Seasonal_Comparison.pdf'
    TIT.save(seas_img, savepath_pdf, dpi=DPI, page_size="original")



def Merge_Fig_2_Influence_Mechanism(FigOutDir: str) -> None:
    """
    合并区域气候效应图(气温diff, 气温diurnal, 相对湿度diff, 相对湿度diurnal)
    """
    seasons = TU.get_seasons()
    map_cropparams = TYCM.MapPlot_CropParams_noColorbar()
    figbox_space = {'left': 0.01, 'top': 0.01, 'right': 0.01, 'bottom': 0.01}
    target = "Influence_Mechanism"
    FigOutDir_out = f'{FigOutDir}/Manuscript'
    os.makedirs(FigOutDir_out, exist_ok=True)
    
    cols_space = [[0, 0, 0], [0, 0, 0]]
    rows_space = [0.01, 0.01]

    # === Summer ===
    # (1) Influence_Mechanism
    infmec_path = f"{FigOutDir_out}/Single/JJA_cap.png"
    crop_params =  {"left": 0.01, "top": 0.01, "right": 0.01, "bottom": 0.01}
    Summer_infmec_img = TIT.crop_image_from_path(infmec_path, crop_params=crop_params, mode="ratio")
    Summer_infmec_img = TIT.resize_image_scale(Summer_infmec_img, 0.777)
    print(Summer_infmec_img.size)

    # (2) PBLH
    pblh_path = f"{FigOutDir}/SpatialMap/Single/SpatialMap_PBLH_Lake_seasonal_JJA.png"
    crop_params =  {"left": 0.0, "top": 0.0, "right": 0.0, "bottom": 0.0}
    Summer_pblh_img = TIT.crop_image_from_path(pblh_path, crop_params=crop_params, mode="ratio")
    print(Summer_pblh_img.size)
    print(Summer_pblh_img.size[0]/Summer_infmec_img.size[0])
    
    # (3) 10UV
    uv_path = f"{FigOutDir}/SpatialMap/Single/SpatialMap_UV_Lake_1000hPa_seasonal_JJA.png"
    crop_params =  {"left": 0.0, "top": 0.0, "right": 0.0, "bottom": 0.0}
    Summer_uv_img = TIT.crop_image_from_path(uv_path, crop_params=crop_params, mode="ratio")

    fig_rows = [[Summer_infmec_img], [Summer_pblh_img], [Summer_uv_img]]
    Summer_img = TIT.merge_images_Row(
        rows_images=fig_rows,
        cols_space=cols_space,
        rows_space=rows_space,
        box_space=figbox_space,
        background_color='#FFFFFF',
        space_mode="ratio",
        alignment=["left", "left", "left"], # , "justify"
        draw_ticks=False, tick_step=0.01,
        # texts=series_texts,
        font_path="/home/wumej22/.local/share/fonts/NotoSans-Bold.ttf",
    )

    # === Winter ===
    # (1) Influence_Mechanism
    infmec_path = f"{FigOutDir_out}/Single/DJF_cap.png"
    crop_params =  {"left": 0.01, "top": 0.01, "right": 0.01, "bottom": 0.01}
    Winter_infmec_img = TIT.crop_image_from_path(infmec_path, crop_params=crop_params, mode="ratio")
    Winter_infmec_img = TIT.resize_image_scale(Winter_infmec_img, 0.776)
    print(Winter_infmec_img.size)

    # (2) PBLH
    pblh_path = f"{FigOutDir}/SpatialMap/Single/SpatialMap_PBLH_Lake_seasonal_DJF.png"
    crop_params =  {"left": 0.0, "top": 0.0, "right": 0.0, "bottom": 0.0}
    Winter_pblh_img = TIT.crop_image_from_path(pblh_path, crop_params=crop_params, mode="ratio")
    
    # (3) 10UV
    uv_path = f"{FigOutDir}/SpatialMap/Single/SpatialMap_UV_Lake_1000hPa_seasonal_DJF.png"
    crop_params =  {"left": 0.0, "top": 0.0, "right": 0.0, "bottom": 0.0}
    Winter_uv_img = TIT.crop_image_from_path(uv_path, crop_params=crop_params, mode="ratio")

    fig_rows = [[Winter_infmec_img], [Winter_pblh_img], [Winter_uv_img]]
    Winter_img = TIT.merge_images_Row(
        rows_images=fig_rows,
        cols_space=cols_space,
        rows_space=rows_space,
        box_space=figbox_space,
        background_color='#FFFFFF',
        space_mode="ratio",
        alignment=["left", "left", "left"], # , "justify"
        draw_ticks=False, tick_step=0.01,
        # texts=series_texts,
        font_path="/home/wumej22/.local/share/fonts/NotoSans-Bold.ttf",
    )

    pblhcbar_path = f"{FigOutDir}/SpatialMap/Single/SpatialMap_PBLH_Lake_seasonal_JJA_VColorbar.png"
    crop_params =  {"left": 0.05, "top": 0.0, "right": 0.0, "bottom": 0.0}
    pblhcbar_img = TIT.crop_image_from_path(pblhcbar_path, crop_params=crop_params, mode="ratio")
    pblhcbar_img = TIT.adjust_image_to_ref_canvas(target_img=pblhcbar_img, ref_img=Winter_pblh_img, axis="height")
    uvcbar_path = f"{FigOutDir}/SpatialMap/Single/SpatialMap_UV_Lake_1000hPa_seasonal_DJF_VColorbar.png"
    uvcbar_img = TIT.crop_image_from_path(uvcbar_path, crop_params=crop_params, mode="ratio")
    uvcbar_img = TIT.adjust_image_to_ref_canvas(target_img=uvcbar_img, ref_img=Winter_uv_img, axis="height")
    blank_img = TIT.create_blank_image(uvcbar_img.size[0], Summer_infmec_img.size[1])

    fig_rows = [[blank_img], [pblhcbar_img], [uvcbar_img]]
    cbar_img = TIT.merge_images_Row(
        rows_images=fig_rows,
        cols_space=cols_space,
        rows_space=rows_space,
        box_space=figbox_space,
        background_color='#FFFFFF',
        space_mode="ratio",
        alignment=["left", "left", "left"], # , "justify"
        draw_ticks=False, tick_step=0.01,
        # texts=series_texts,
        font_path="/home/wumej22/.local/share/fonts/NotoSans-Bold.ttf",
    )

    series_texts = {
        '(a)': {'x': 0.02, 'y': 0.01,  "fontsize": 0.02, "text": "(a)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        '(b)': {'x': 0.48, 'y': 0.01,  "fontsize": 0.02, "text": "(b)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        '(c)': {'x': 0.02, 'y': 0.40,  "fontsize": 0.02, "text": "(c)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        '(d)': {'x': 0.48, 'y': 0.40,  "fontsize": 0.02, "text": "(d)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        '(e)': {'x': 0.02, 'y': 0.70,  "fontsize": 0.02, "text": "(e)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        '(f)': {'x': 0.48, 'y': 0.70,  "fontsize": 0.02, "text": "(f)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
    }

    cols_space = [[0, 0,]]
    rows_space = [0.01]
    fig_rows = [[Summer_img, Winter_img, cbar_img]]
    big_img = TIT.merge_images_Row(
        rows_images=fig_rows,
        cols_space=cols_space,
        rows_space=rows_space,
        box_space=figbox_space,
        background_color='#FFFFFF',
        space_mode="ratio",
        alignment=["justify"], # , "justify"
        draw_ticks=False, tick_step=0.01,
        texts=series_texts,
        font_path="/home/wumej22/.local/share/fonts/NotoSans-Bold.ttf",
    )

    savepath_png = f'{FigOutDir_out}/Fig_2_{target}.{FIGFMT}'
    TIT.save(big_img, savepath_png, dpi=DPI)
    savepath_pdf = f'{FigOutDir_out}/Fig_2_{target}.pdf'
    TIT.save(big_img, savepath_pdf, dpi=DPI, page_size="original")





def Merge_Fig_3_ExtremeEventFreq(FigOutDir: str) -> None:
    figbox_space = TYCM.Merge_Fig_Space_Params()
    cbar_space = TYCM.MapCbar_CropParams_V()
    eventslist = ["ColdWet", "Wet", "HotWet", "Cold", "Hot", "ColdDry", "Dry", "HotDry"]
    crop_params =  {"left": 0.0, "top": 0.0, "right": 0.0, "bottom": 0.0}
    FigOutDir_in = f'{FigOutDir}/ExtremeEventFreq'
    FigOutDir_out = f'{FigOutDir}/Manuscript'
    target = "ExtremeEventFreq"
    checkmethod="Wilcoxon_signed-rank_test"
    cols_space = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    rows_space = [0.01, 0.1, 0.01]
    fig_rows = [["", "", ""], ["", "", ""], ["", "", ""]]
    eventorder = {"ColdWet": [0, 0], "Wet":    [0, 1], "HotWet": [0, 2],
                  "Cold":    [1, 0], "Define": [1, 1], "Hot":    [1, 2],
                  "ColdDry": [2, 0], "Dry":    [2, 1], "HotDry": [2, 2],}

    definepath = f"{FigOutDir_in}/Single/Extreme_Event_Definition_Scheme.{FIGFMT}"
    define_img = TIT.crop_image_from_path(definepath, crop_params=crop_params, mode="ratio")
    fig_rows[eventorder["Define"][0]][eventorder["Define"][1]] = define_img
    
    for event in eventslist:
        figpath = f'{FigOutDir_in}/Single/ExtremeEvent_{event}_Freq_Change_CircularRing_{checkmethod}.{FIGFMT}'
        event_img = TIT.crop_image_from_path(figpath, crop_params=crop_params, mode="ratio")
        event_img = TIT.adjust_image_to_ref_canvas(event_img, define_img)
        fig_rows[eventorder[event][0]][eventorder[event][1]] = event_img

    series_texts = {
        '(a)': {'x': 0.02,  'y': 0.05,  "fontsize": 0.015, "text": "(a)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        '(b)': {'x': 0.02, 'y': 0.382,  "fontsize": 0.015, "text": "(b)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        '(c)': {'x': 0.02, 'y': 0.715,  "fontsize": 0.015, "text": "(c)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        '(d)': {'x': 0.34, 'y': 0.05,  "fontsize": 0.015, "text": "(d)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        '(e)': {'x': 0.34, 'y': 0.382,  "fontsize": 0.015, "text": "(e)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        '(f)': {'x': 0.34, 'y': 0.715,  "fontsize": 0.015, "text": "(f)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        '(g)': {'x': 0.69, 'y': 0.05,  "fontsize": 0.015, "text": "(g)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        '(h)': {'x': 0.69, 'y': 0.382,  "fontsize": 0.015, "text": "(h)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        '(i)': {'x': 0.69, 'y': 0.715,  "fontsize": 0.015, "text": "(i)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
    }

    freq_img = TIT.merge_images_Row(
        rows_images=fig_rows,
        cols_space=cols_space,
        rows_space=rows_space,
        box_space=figbox_space,
        background_color='#FFFFFF',
        space_mode="ratio",
        alignment=["justify", "justify", "justify"],
        draw_ticks=False, tick_step=0.01,
        # texts=series_texts,
        font_path="/home/wumej22/.local/share/fonts/NotoSans-Bold.ttf",
    )

    savepath_png = f'{FigOutDir_out}/Fig_3_{target}_Comparison.{FIGFMT}'
    TIT.save(freq_img, savepath_png, dpi=DPI)
    savepath_pdf = f'{FigOutDir_out}/Fig_3_{target}_Comparison.pdf'
    TIT.save(freq_img, savepath_pdf, dpi=DPI, page_size="original")



def Merge_Fig_4_HotWetCoupling(FigOutDir: str) -> None:
    figbox_space = TYCM.Merge_Fig_Space_Params()
    cbar_space = TYCM.MapCbar_CropParams_V() 
    crop_params =  {"left": 0.0, "top": 0.0, "right": 0.0, "bottom": 0.0}
    FigOutDir_out = f'{FigOutDir}/Manuscript'
    target = "HotWetCoupling"
    cols_space = [[0]]
    rows_space = [0]
    lambda_u_path = f"{FigOutDir}/CouplingTest/Single/CouplingMetrics_Lake-NoLake_T2m-RH_lambda_u.{FIGFMT}"
    lambda_u_img = TIT.crop_image_from_path(lambda_u_path, crop_params=crop_params, mode="ratio")
    cbar_path = f"{FigOutDir}/CouplingTest/Single/CouplingMetrics_Lake-NoLake_T2m-RH_lambda_u_VColorbar.{FIGFMT}"
    cbar_img = TIT.crop_image_from_path(cbar_path, crop_params=crop_params, mode="ratio")
    fig_rows = [[lambda_u_img, cbar_img]]
    # lambda_u
    lambda_u_img = TIT.merge_images_Row(
        rows_images=fig_rows,
        cols_space=cols_space,
        rows_space=rows_space,
        box_space=figbox_space,
        background_color='#FFFFFF',
        space_mode="ratio",
        alignment=["justify"],
        draw_ticks=False, tick_step=0.01,
        # texts=series_texts,
        font_path="/home/wumej22/.local/share/fonts/NotoSans-Bold.ttf",
    )

    joint_path = f"{FigOutDir}/CouplingTest/Joint/Extreme_Temperature_Humidity_Coupling_State_Site_x104_y83_with_KDE.{FIGFMT}"
    joint_img = TIT.crop_image_from_path(joint_path, crop_params=crop_params, mode="ratio")
    # joint_img = TIT.adjust_image_to_ref_canvas(joint_img, lambda_u_img)
    fig_rows = [[lambda_u_img, joint_img]]

    series_texts = {
        # 气温diff
        '(a)': {'x': 0.03,  'y': 0.05,  "fontsize": 0.05, "text": "(a)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        '(b)': {'x': 0.68,  'y': 0.05,  "fontsize": 0.05, "text": "(b)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
    }

    big_img = TIT.merge_images_Row(
        rows_images=fig_rows,
        cols_space=cols_space,
        rows_space=rows_space,
        box_space=figbox_space,
        background_color='#FFFFFF',
        space_mode="ratio",
        alignment=["justify"],
        draw_ticks=False, tick_step=0.01,
        texts=series_texts,
        font_path="/home/wumej22/.local/share/fonts/NotoSans-Bold.ttf",
    )

    savepath_png = f'{FigOutDir_out}/Fig_4_{target}_Comparison.{FIGFMT}'
    TIT.save(big_img, savepath_png, dpi=DPI)
    savepath_pdf = f'{FigOutDir_out}/Fig_4_{target}_Comparison.pdf'
    TIT.save(big_img, savepath_pdf, dpi=DPI, page_size="original")



def Merge_Fig_5_AffectedPopulation(FigOutDir: str) -> None:
    figbox_space = TYCM.Merge_Fig_Space_Params()
    cbar_space = TYCM.MapCbar_CropParams_V() 
    FigOutDir_out = f'{FigOutDir}/Manuscript'
    target = "Affected_Population"
    checkmethod = "Wilcoxon_signed-rank_test"
    event = "HotWet"
    cols_space = [[0],[0]]
    rows_space = [0, 0]

    crop_params =  {"left": 0.05, "top": 0.0, "right": 0.05, "bottom": 0.07}
    joint_path = f'{FigOutDir}/ExtremeEventAddtional/Single/{target}_RadialHistogram_{event}-up_{checkmethod}.{FIGFMT}'
    up_img = TIT.crop_image_from_path(joint_path, crop_params=crop_params, mode="ratio")
    up_img = TIT.resize_image_scale(up_img, 1.4)


    crop_params =  {"left": 0.0098, "top": 0.0098, "right": 0.0098, "bottom": 0.0098}
    citypath = f'{FigOutDir}/ExtremeEventAddtional/Single/{target}_Map_{event}-up_{checkmethod}-shpfile.{FIGFMT}'
    city_img = TIT.crop_image_from_path(citypath, crop_params=crop_params, mode="ratio")
    cbar_path = f'{FigOutDir}/ExtremeEventAddtional/Single/{target}_Map_{event}_HColorbar-shpfile_{checkmethod}.{FIGFMT}'
    crop_params =  {"left": 0.0, "top": 0.0, "right": 0.0, "bottom": 0.0}
    cbar_img = TIT.crop_image_from_path(cbar_path, crop_params=crop_params, mode="ratio")
    cbar_img = TIT.resize_image_scale(cbar_img, 0.95)
    # cbar_img = TIT.adjust_image_to_ref_canvas(cbar_img, city_img, axis="width")
    fig_rows = [[city_img], [cbar_img]]
    # lambda_u
    down_img = TIT.merge_images_Row(
        rows_images=fig_rows,
        cols_space=cols_space,
        rows_space=rows_space,
        box_space=figbox_space,
        background_color='#FFFFFF',
        space_mode="ratio",
        alignment=["center", "center"],
        draw_ticks=False, tick_step=0.01,
        # texts=series_texts,
        font_path="/home/wumej22/.local/share/fonts/NotoSans-Bold.ttf",
    )

    big_img = TIT.overlay_images(refimg=up_img, addimg=down_img, x=0.51, y=0.54, ha="center", va="center", scale=0.455)

    savepath_png = f'{FigOutDir_out}/Fig_5_{target}.{FIGFMT}'
    savepath_pdf = f'{FigOutDir_out}/Fig_5_{target}.pdf'
    TIT.save(big_img, savepath_png, dpi=DPI)
    TIT.save(big_img, savepath_pdf, dpi=DPI)


    # # pct
    # crop_params =  {"left": 0.05, "top": 0.0, "right": 0.05, "bottom": 0.05}
    # joint_path = f'{FigOutDir}/ExtremeEventAddtional/Single/{target}_RadialHistogram_{event}-up_pct_{checkmethod}.{FIGFMT}'
    # up_img = TIT.crop_image_from_path(joint_path, crop_params=crop_params, mode="ratio")
    # up_img = TIT.resize_image_scale(up_img, 1.3)

    # crop_params =  {"left": 0.0098, "top": 0.0098, "right": 0.0098, "bottom": 0.0098}
    # citypath = f'{FigOutDir}/ExtremeEventAddtional/Single/{target}_Map_{event}-up_{checkmethod}-shpfile.{FIGFMT}'
    # city_img = TIT.crop_image_from_path(citypath, crop_params=crop_params, mode="ratio")
    # cbar_path = f'{FigOutDir}/ExtremeEventAddtional/Single/{target}_Map_{event}_HColorbar-shpfile_{checkmethod}.{FIGFMT}'
    # crop_params =  {"left": 0.0, "top": 0.0, "right": 0.0, "bottom": 0.0}
    # cbar_img = TIT.crop_image_from_path(cbar_path, crop_params=crop_params, mode="ratio")
    # cbar_img = TIT.resize_image_scale(cbar_img, 0.95)
    # # cbar_img = TIT.adjust_image_to_ref_canvas(cbar_img, city_img, axis="width")
    # fig_rows = [[city_img], [cbar_img]]
    # # lambda_u
    # down_img = TIT.merge_images_Row(
    #     rows_images=fig_rows,
    #     cols_space=cols_space,
    #     rows_space=rows_space,
    #     box_space=figbox_space,
    #     background_color='#FFFFFF',
    #     space_mode="ratio",
    #     alignment=["center", "center"],
    #     draw_ticks=False, tick_step=0.01,
    #     # texts=series_texts,
    #     font_path="/home/wumej22/.local/share/fonts/NotoSans-Bold.ttf",
    # )

    # big_img = TIT.overlay_images(refimg=up_img, addimg=down_img, x=0.51, y=0.54, ha="center", va="center", scale=0.455)

    # savepath_png = f'{FigOutDir_out}/Fig_5_{target}_PCT.{FIGFMT}'
    # savepath_pdf = f'{FigOutDir_out}/Fig_5_{target}_PCT.pdf'
    # TIT.save(big_img, savepath_png, dpi=DPI)
    # TIT.save(big_img, savepath_pdf, dpi=DPI, page_size="original")




def Merge_Fig_S1_ModelValidation(FigOutDir: str) -> None:
    seasons = TU.get_seasons()
    mapcrop_params = TYCM.MapPlot_CropParams_noColorbar()
    figbox_space = TYCM.Merge_Fig_Space_Params()
    cbar_space = TYCM.MapCbar_CropParams_H()
    FigOutDir_out = f'{FigOutDir}/Manuscript'
    target = "ModelValidation"
    cols_space = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    rows_space = [0.01, 0.01]
    
    varlist  = ["T2m", "Prec", "Q2m"]
    reflist = ["CMFDv2", "ERA5-Land"]
    for varname in varlist:
        fig_rows = [[], [], []]
        for i, dataname in enumerate([*reflist]):
            rowpos = i
            for j, season in enumerate(seasons):
                figpath = f"{FigOutDir}/ModelValidation/{varname}/Single/Diff_Lake_{varname}_{dataname}_seasonal_{season}.{FIGFMT}"
                croped_fig = TIT.crop_image_from_path(figpath, crop_params=mapcrop_params, mode="ratio")
                fig_rows[rowpos].append(croped_fig)
        rowpos = -1
        cbarpath = f"{FigOutDir}/ModelValidation/{varname}/Single/Diff_Map_Lake_{varname}_Seasonal_{season}_HColorbar.{FIGFMT}"
        cropped_cbar = TIT.crop_image_from_path(cbarpath, crop_params=cbar_space, mode="ratio")
        fig_rows[rowpos].append(cropped_cbar)

        series_texts = {
            # CMFDv2
            '(a)': {'x': 0.01,  'y': 0.015,  "fontsize": 0.03, "text": "(a)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
            '(b)': {'x': 0.258, 'y': 0.015,  "fontsize": 0.03, "text": "(b)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
            '(c)': {'x': 0.508, 'y': 0.015,  "fontsize": 0.03, "text": "(c)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
            '(d)': {'x': 0.755, 'y': 0.015,  "fontsize": 0.03, "text": "(d)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
            # ERA5-Land
            '(e)': {'x': 0.01,  'y': 0.45,  "fontsize": 0.03, "text": "(e)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
            '(f)': {'x': 0.258, 'y': 0.45,  "fontsize": 0.03, "text": "(f)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
            '(g)': {'x': 0.508, 'y': 0.45,  "fontsize": 0.03, "text": "(g)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
            '(h)': {'x': 0.755, 'y': 0.45,  "fontsize": 0.03, "text": "(h)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        }

        big_img = TIT.merge_images_Row(
            rows_images=fig_rows,
            cols_space=cols_space,
            rows_space=rows_space,
            box_space=figbox_space,
            background_color='#FFFFFF',
            space_mode="ratio",
            alignment=["justify", "justify", "center"],
            draw_ticks=False, tick_step=0.01,
            texts=series_texts,
            font_path="/home/wumej22/.local/share/fonts/NotoSans-Bold.ttf",
        )

        savepath_png = f'{FigOutDir_out}/Fig_S1_ModelValidation_{varname}.{FIGFMT}'
        TIT.save(big_img, savepath_png, dpi=DPI)

    # Mean Prec
    cols_space = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    rows_space = [0.01, 0.01, 0.01]
    
    varlist  = ["Prec"]
    reflist = ["CMFDv2", "ERA5-Land"]
    for varname in varlist:
        fig_rows = [[], [], [], []]
        for i, dataname in enumerate(["Lake", *reflist]):
            rowpos = i
            for j, season in enumerate(seasons):
                figpath = f"{FigOutDir}/ModelValidation/{varname}/Single/Mean_Lake_{varname}_{dataname}_seasonal_{season}.{FIGFMT}"
                print(figpath)
                croped_fig = TIT.crop_image_from_path(figpath, crop_params=mapcrop_params, mode="ratio")
                fig_rows[rowpos].append(croped_fig)
        rowpos = -1
        cbarpath = f"{FigOutDir}/ModelValidation/{varname}/Single/Mean_Map_Lake_{varname}_Seasonal_{season}_HColorbar.{FIGFMT}"
        cropped_cbar = TIT.crop_image_from_path(cbarpath, crop_params=cbar_space, mode="ratio")
        fig_rows[rowpos].append(cropped_cbar)

        series_texts = {
            # Lake
            '(a)': {'x': 0.01,  'y': 0.0135,  "fontsize": 0.025, "text": "(a)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
            '(b)': {'x': 0.258, 'y': 0.0135,  "fontsize": 0.025, "text": "(b)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
            '(c)': {'x': 0.508, 'y': 0.0135,  "fontsize": 0.025, "text": "(c)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
            '(d)': {'x': 0.755, 'y': 0.0135,  "fontsize": 0.025, "text": "(d)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
            # CMFDv2
            '(e)': {'x': 0.01,  'y': 0.313,  "fontsize": 0.025, "text": "(e)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
            '(f)': {'x': 0.258, 'y': 0.313,  "fontsize": 0.025, "text": "(f)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
            '(g)': {'x': 0.508, 'y': 0.313,  "fontsize": 0.025, "text": "(g)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
            '(h)': {'x': 0.755, 'y': 0.313,  "fontsize": 0.025, "text": "(h)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
            # ERA5-Land
            '(i)': {'x': 0.01,  'y': 0.612,  "fontsize": 0.025, "text": "(i)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
            '(j)': {'x': 0.258, 'y': 0.612,  "fontsize": 0.025, "text": "(j)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
            '(k)': {'x': 0.508, 'y': 0.612,  "fontsize": 0.025, "text": "(k)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
            '(l)': {'x': 0.755, 'y': 0.612,  "fontsize": 0.025, "text": "(l)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        }

        big_img = TIT.merge_images_Row(
            rows_images=fig_rows,
            cols_space=cols_space,
            rows_space=rows_space,
            box_space=figbox_space,
            background_color='#FFFFFF',
            space_mode="ratio",
            alignment=["justify", "justify", "justify", "center"],
            draw_ticks=False, tick_step=0.01,
            texts=series_texts,
            font_path="/home/wumej22/.local/share/fonts/NotoSans-Bold.ttf",
        )

        savepath_png = f'{FigOutDir_out}/Fig_S1_ModelValidation_{varname}_Mean.{FIGFMT}'
        TIT.save(big_img, savepath_png, dpi=DPI)



def Merge_Fig_S5_RegClimImpact(FigOutDir: str) -> None:
    seasons = TU.get_seasons()
    map_cropparams = TYCM.MapPlot_CropParams_noColorbar()
    figbox_space = {'left': 0.15, 'top': 0.01, 'right': 0.01, 'bottom': 0.01}
    cbar_cropparams = {"left": 0.005, "top": 0.005, "right": 0.3, "bottom": 0.005}
    rose_cropparams = TYCM.RosePlot_CropParams_noColorbar()
    onlysig = True
    suffix = "onlysig" if onlysig else "all"
    target = "RegClimImpact"
    FigOutDir_in = f'{FigOutDir}/RegClimImpact'
    FigOutDir_out = f'{FigOutDir}/Manuscript'
    os.makedirs(FigOutDir_out, exist_ok=True)
    
    cols_space = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    rows_space = [0.01, 0.1, 0.01]
    rows = [[], [], [], []]

    #############################
    # 季节diff map & rose map
    #############################
    cols_space = [[0, 0, 0, 0], [0, 0, 0, 0]]
    rows_space = [0.01, 0.01]
    checkmethod = "Paired_t-test"

    for varname in ["T2m-Max", "T2m-Min", "LHF", "SHF", "Q2m", "UV"]: # "UV10",
        varinfo = TYCM.Variable_Infos(varname)
        longname = varinfo["longname"]
        unit = varinfo["unit"]
        rows = [[], []]
        rowpos = 0
        for season in seasons:
            if varname in ["UV"]:
                figpath = f'{FigOutDir}/PressureLevel/1000hPa/Single/PressureLevel_Diff_Map_{varname}_{season}_{suffix}_{checkmethod}.{FIGFMT}'
            else:
                figpath = f'{FigOutDir_in}/{varname}/Single/{target}_Diff_Map_{varname}_{season}_{suffix}_{checkmethod}.{FIGFMT}'
            crop_diff = TIT.crop_image_from_path(figpath, crop_params=map_cropparams, mode="ratio")
            rows[rowpos].append(crop_diff)

        if varname in ["UV"]:
            cbarpath = f'{FigOutDir}/PressureLevel/1000hPa/Single/PressureLevel_Diff_Map_{varname}_Seasonal_Colorbar_{suffix}_{checkmethod}.{FIGFMT}'
        else:
            cbarpath = f'{FigOutDir_in}/{varname}/Single/{target}_Diff_Map_{varname}_Seasonal_Colorbar_{suffix}_{checkmethod}.{FIGFMT}'
        crop_cbar = TIT.crop_image_from_path(cbarpath, crop_params=cbar_cropparams, mode="ratio")
        crop_cbar = TIT.adjust_image_to_ref_canvas(target_img=crop_cbar, ref_img=crop_diff, axis="height")
        rows[rowpos].append(crop_cbar)

        rowpos = 1
        for season in seasons:
            if varname in ["UV"]:
                figpath = f'{FigOutDir}/PressureLevel/1000hPa/Single/PressureLevel_Diff_Rose_{varname}_{season}_{suffix}_{checkmethod}.{FIGFMT}'
            else:
                figpath = f'{FigOutDir_in}/{varname}/Single/{target}_Diff_Rose_{varname}_{season}_{suffix}_{checkmethod}.{FIGFMT}'
            crop_rose = TIT.crop_image_from_path(figpath, crop_params=rose_cropparams, mode="ratio")
            crop_rose = TIT.adjust_image_to_ref_canvas(target_img=crop_rose, ref_img=crop_diff)
            rows[rowpos].append(crop_rose)

        series_texts = {
            'T2m': {'x': 0.03,  'y': 0.46,  "fontsize": 0.05, "text": rf"{longname} ({unit})", "ha": "right", "va": "center", "color": 'black', "fontweight": "bold", "rotation":-90},
            # diff
            '(a)': {'x': 0.0445, 'y': 0.02,  "fontsize": 0.03, "text": "(a)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
            '(b)': {'x': 0.2745, 'y': 0.02,  "fontsize": 0.03, "text": "(b)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
            '(c)': {'x': 0.5045, 'y': 0.02,  "fontsize": 0.03, "text": "(c)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
            '(d)': {'x': 0.7365, 'y': 0.02,  "fontsize": 0.03, "text": "(d)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
            # rose
            '(e)': {'x': 0.0445, 'y': 0.42,  "fontsize": 0.03, "text": "(e)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
            '(f)': {'x': 0.2745, 'y': 0.42,  "fontsize": 0.03, "text": "(f)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
            '(g)': {'x': 0.5045, 'y': 0.42,  "fontsize": 0.03, "text": "(g)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
            '(h)': {'x': 0.7365, 'y': 0.42,  "fontsize": 0.03, "text": "(h)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        }
    
        seas_img = TIT.merge_images_Row(
            rows_images=rows,
            cols_space=cols_space,
            rows_space=rows_space,
            box_space=figbox_space,
            background_color='#FFFFFF',
            space_mode="ratio",
            alignment=["left","left"],  #, "left"
            draw_ticks=False, tick_step=0.01,
            texts=series_texts,
            font_path="/home/wumej22/.local/share/fonts/NotoSans-Bold.ttf",
        )
        savepath_png = f'{FigOutDir_out}/Fig_S5_{target}_{varname}_Seasonal_Comparison.{FIGFMT}'
        TIT.save(seas_img, savepath_png, dpi=DPI)





def Merge_Fig_S6_ExtremeEvents(FigOutDir: str) -> None:
    figbox_space = TYCM.Merge_Fig_Space_Params()
    mapcrop_params = TYCM.MapPlot_CropParams_noColorbar()
    boxcrop_params = TYCM.Extreme_Events_BoxPlot_CropParams()
    cbar_space = TYCM.MapCbar_CropParams_V()
    eventslist = ["ColdWet", "Wet", "HotWet", "Cold", "Hot", "ColdDry", "Dry", "HotDry"]
    crop_params =  {"left": 0.0, "top": 0.0, "right": 0.0, "bottom": 0.0}
    FigOutDir_in = f'{FigOutDir}/ExtremeEventFreq'
    FigOutDir_out = f'{FigOutDir}/Manuscript'
    target = "ExtremeEventFreq"
    suffix = 'onlysig'
    checkmethod="Wilcoxon_signed-rank_test"
    cols_space = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    rows_space = [0.01, 0.01, 0.01]
    fig_rows = [["", "", ""], ["", "", ""], ["", "", ""]]
    eventorder = {"ColdWet": [0, 0], "Wet":    [0, 1], "HotWet": [0, 2],
                  "Cold":    [1, 0], "Box":    [1, 1], "Hot":    [1, 2],
                  "ColdDry": [2, 0], "Dry":    [2, 1], "HotDry": [2, 2],}

    for event in eventslist:
        figpath = f'{FigOutDir_in}/Single/Freq_Diff_Map_{event}_Annual_{suffix}_{checkmethod}.{FIGFMT}'
        cropped_map = TIT.crop_image_from_path(figpath, crop_params=mapcrop_params, mode="ratio")
        diffcbarpath = f'{FigOutDir_in}/Single/Freq_Diff_Map_{event}_Annual_Colorbar_{suffix}_{checkmethod}.{FIGFMT}'
        cropped_cbar = TIT.crop_image_from_path(diffcbarpath, crop_params=cbar_space, mode="ratio")
        cropped_cbar = TIT.adjust_image_to_ref_canvas(cropped_cbar, cropped_map, axis="height")
        fig_rows[eventorder[event][0]][eventorder[event][1]] = cropped_map
        event_img = TIT.merge_images_Row(
            rows_images=[[cropped_map, cropped_cbar]],
            cols_space=[[0]],
            rows_space=[0.0],
            box_space=figbox_space,
            background_color='#FFFFFF',
            space_mode="ratio",
            alignment=["left"],
            draw_ticks=False, tick_step=0.01,
            font_path="/share/home/dq048/.local/share/fonts/NotoSans-Bold.ttf",
        )
        fig_rows[eventorder[event][0]][eventorder[event][1]] = event_img

    anual_boxpath_diff = f'{FigOutDir_in}/Single/Freq_Diff_Box_Annual_{suffix}_{checkmethod}.{FIGFMT}'
    cropped_box_diff = TIT.crop_image_from_path(anual_boxpath_diff, crop_params=boxcrop_params, mode="ratio")
    cropped_box_diff = TIT.adjust_image_to_ref_canvas(cropped_box_diff, event_img, axis="width", align="left")
    fig_rows[1][1] = cropped_box_diff

    series_texts = {
        '(a)': {'x': 0.015, 'y': 0.02, "fontsize": 0.025, "text": "(a)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        '(b)': {'x': 0.344, 'y': 0.02, "fontsize": 0.025, "text": "(b)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        '(c)': {'x': 0.675, 'y': 0.02, "fontsize": 0.025, "text": "(c)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        '(d)': {'x': 0.015, 'y': 0.352,  "fontsize": 0.025, "text": "(d)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        '(e)': {'x': 0.344, 'y': 0.352,  "fontsize": 0.025, "text": "(e)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        '(f)': {'x': 0.675, 'y': 0.352,  "fontsize": 0.025, "text": "(f)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        '(g)': {'x': 0.015, 'y': 0.683, "fontsize": 0.025, "text": "(g)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        '(h)': {'x': 0.344, 'y': 0.683, "fontsize": 0.025, "text": "(h)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        '(i)': {'x': 0.675, 'y': 0.683, "fontsize": 0.025, "text": "(i)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
    }

    freq_img = TIT.merge_images_Row(
        rows_images=fig_rows,
        cols_space=cols_space,
        rows_space=rows_space,
        box_space=figbox_space,
        background_color='#FFFFFF',
        space_mode="ratio",
        alignment=["left", "left", "left"],
        draw_ticks=False, tick_step=0.01,
        texts=series_texts,
        font_path="/home/wumej22/.local/share/fonts/NotoSans-Bold.ttf",
    )

    savepath_png = f'{FigOutDir_out}/Fig_S6_{target}_Comparison.{FIGFMT}'
    TIT.save(freq_img, savepath_png, dpi=DPI)



def Merge_Fig_S7_CUOFF_NCP(FigOutDir: str) -> None:
    seasons = TU.get_seasons()
    mapcrop_params = TYCM.MapPlot_CropParams_noColorbar()
    figbox_space = {'left': 0.05, 'top': 0.2, 'right': 0.05, 'bottom': 0.01}
    cbar_space = TYCM.MapCbar_CropParams_H()
    FigOutDir_out = f'{FigOutDir}/Manuscript'
    target = "ModelValidation"
    cols_space = [[0.1], [0.1], [0,]]
    rows_space = [0.01, 0.01]
    varlist  = ["T2m", "Prec"]
    reflist = ["CMFDv2", "ERA5-Land"]
    caselist = ["CUOFF", "Lake"]
    for varname in varlist:
        fig_rows = [[], [], []]
        for i, dataname in enumerate([*reflist]):
            rowpos = i
            season = 'JJA'
            for casename in caselist:
                figpath = f"{FigOutDir}/ModelValidation_NCP/{varname}/Single/Diff_{casename}_{varname}_{dataname}_seasonal_{season}.{FIGFMT}"
                croped_fig = TIT.crop_image_from_path(figpath, crop_params=mapcrop_params, mode="ratio")
                fig_rows[rowpos].append(croped_fig)
        rowpos = -1
        cbarpath = f"{FigOutDir}/ModelValidation_NCP/{varname}/Single/Diff_Map_Lake_{varname}_Seasonal_{season}_HColorbar.{FIGFMT}"
        cropped_cbar = TIT.crop_image_from_path(cbarpath, crop_params=cbar_space, mode="ratio")
        fig_rows[rowpos].append(cropped_cbar)

        series_texts = {
            # CMFDv2
            'CUOFF': {'x': 0.25, 'y': 0.078,  "fontsize": 0.03, "text": "Traditional configuration\n(close cumulus parameterization)", "ha": "center", "va": "bottom", "color": "black", "fontweight": "bold"},
            'Lake': {'x': 0.75, 'y': 0.078,  "fontsize": 0.03, "text": "This study\n(new cumulus parameterization, NCP)", "ha": "center", "va": "bottom", "color": "black", "fontweight": "bold"},
            '(a)': {'x': 0.035, 'y': 0.094,  "fontsize": 0.03, "text": "(a)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
            '(b)': {'x': 0.534, 'y': 0.094,  "fontsize": 0.03, "text": "(b)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
            '(c)': {'x': 0.035, 'y': 0.49,  "fontsize": 0.03, "text": "(c)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
            '(d)': {'x': 0.534, 'y': 0.49,  "fontsize": 0.03, "text": "(d)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        }

        big_img = TIT.merge_images_Row(
            rows_images=fig_rows,
            cols_space=cols_space,
            rows_space=rows_space,
            box_space=figbox_space,
            background_color='#FFFFFF',
            space_mode="ratio",
            alignment=["justify", "justify", "center"],
            draw_ticks=False, tick_step=0.01,
            texts=series_texts,
            font_path="/home/wumej22/.local/share/fonts/NotoSans-Bold.ttf",
        )

        savepath_png = f'{FigOutDir_out}/Fig_S7_CUOFF_NCP_{varname}.{FIGFMT}'
        TIT.save(big_img, savepath_png, dpi=DPI)




def Merge_Fig_S8_define_extreme_event(FigOutDir: str) -> None:
    figbox_space = TYCM.Merge_Fig_Space_Params()
    cbar_space = TYCM.MapCbar_CropParams_V() 
    FigOutDir_out = f'{FigOutDir}/Manuscript'
    target = "StaticStability"
    seasons = TU.get_seasons()
    cols_space = [[0, 0], [0, 0]]
    rows_space = [0, 0]
    fig_rows = [[]]

    crop_params =  {"left": 0.005, "top": 0.005, "right": 0.005, "bottom": 0.005}
    figpath = f"{FigOutDir}/Step1_Schematic_N_Years.{FIGFMT}"
    S1_img = TIT.crop_image_from_path(figpath, crop_params=crop_params, mode="ratio")
    figpath = f"{FigOutDir}/Step2_Threshold_Distribution.{FIGFMT}"
    S2_img = TIT.crop_image_from_path(figpath, crop_params=crop_params, mode="ratio")
    fig_rows = [[S1_img, S2_img]]
    series_texts = {
        '(a)': {'x': 0.051, 'y': 0.08, "fontsize": 0.03, "text": "(a)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        '(b)': {'x': 0.545, 'y': 0.08,  "fontsize": 0.03, "text": "(b)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
    }

    big_img = TIT.merge_images_Row(
        rows_images=fig_rows,
        cols_space=cols_space,
        rows_space=rows_space,
        box_space=figbox_space,
        background_color='#FFFFFF',
        space_mode="ratio",
        alignment=["justify"],
        draw_ticks=False, tick_step=0.01,
        texts=series_texts,
        font_path="/home/wumej22/.local/share/fonts/NotoSans-Bold.ttf",
    )

    savepath_png = f'{FigOutDir_out}/Fig_S8_define_extreme_event.{FIGFMT}'
    TIT.save(big_img, savepath_png, dpi=DPI)



