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
INFIGFMT = TPC.FIGFMT
OUTFIGFMT = "pdf"
DPI    = TPC.DPI_medium


def make_figdir(FigOutDir: str) -> None:
    """创建目录，如果目录已存在则不进行任何操作"""
    FigOutDir_IN = f"{FigOutDir}/Review_1/{INFIGFMT}"
    os.makedirs(FigOutDir_IN, exist_ok=True)
    FigOutDir_OUT = f"{FigOutDir}/Review_1/{OUTFIGFMT}"
    os.makedirs(FigOutDir_OUT, exist_ok=True)



def Fig_1(FigOutDir: str, FigLabel: str) -> None:
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
    FigOutDir_out = f'{FigOutDir}/Review_1'
    os.makedirs(FigOutDir_out, exist_ok=True)
    
    cols_space = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    rows_space = [0.01, 0.1, 0.01]
    rows = [[], [], [], []]
    # 第一行：气温diff
    rowpos = 0
    varname = "T2m"
    checkmethod = "Paired_t-test"
    for season in seasons:
        figpath = f'{FigOutDir_in}/{varname}/Single/{target}_Diff_Map_{varname}_{season}_{suffix}_{checkmethod}.{INFIGFMT}'
        crop_diff = TIT.crop_image_from_path(figpath, crop_params=map_cropparams, mode="ratio")
        rows[rowpos].append(crop_diff)
    cbarpath = f'{FigOutDir_in}/{varname}/Single/{target}_Diff_Map_{varname}_Seasonal_Colorbar_{suffix}_{checkmethod}.{INFIGFMT}'
    crop_cbar = TIT.crop_image_from_path(cbarpath, crop_params=cbar_cropparams, mode="ratio")
    crop_cbar = TIT.adjust_image_to_ref_canvas(target_img=crop_cbar, ref_img=crop_diff, axis="height")
    rows[rowpos].append(crop_cbar)

    # 第二行：气温rose
    rowpos = 1
    varname = "T2m"
    checkmethod = "Paired_t-test"
    for season in seasons:
        figpath = f'{FigOutDir_in}/{varname}/Single/{target}_Diff_Rose_{varname}_{season}_{suffix}_{checkmethod}.{INFIGFMT}'
        crop_rose = TIT.crop_image_from_path(figpath, crop_params=rose_cropparams, mode="ratio")
        crop_rose = TIT.adjust_image_to_ref_canvas(target_img=crop_rose, ref_img=crop_diff)
        rows[rowpos].append(crop_rose)

    # 第三行：相对湿度diff
    rowpos = 2
    varname = "RH"
    checkmethod = "Paired_t-test"
    for season in seasons:
        figpath = f'{FigOutDir_in}/{varname}/Single/{target}_Diff_Map_{varname}_{season}_{suffix}_{checkmethod}.{INFIGFMT}'
        crop_diff = TIT.crop_image_from_path(figpath, crop_params=map_cropparams, mode="ratio")
        rows[rowpos].append(crop_diff)
    cbarpath = f'{FigOutDir_in}/{varname}/Single/{target}_Diff_Map_{varname}_Seasonal_Colorbar_{suffix}_{checkmethod}.{INFIGFMT}'
    crop_cbar = TIT.crop_image_from_path(cbarpath, crop_params=cbar_cropparams, mode="ratio")
    crop_cbar = TIT.adjust_image_to_ref_canvas(target_img=crop_cbar, ref_img=crop_diff, axis="height")
    rows[rowpos].append(crop_cbar)

    # 第四行： 相对湿度rose
    rowpos = 3
    varname = "RH"
    checkmethod = "Paired_t-test"
    for season in seasons:
        figpath = f'{FigOutDir_in}/{varname}/Single/{target}_Diff_Rose_{varname}_{season}_{suffix}_{checkmethod}.{INFIGFMT}'
        crop_rose = TIT.crop_image_from_path(figpath, crop_params=rose_cropparams, mode="ratio")
        crop_rose = TIT.adjust_image_to_ref_canvas(target_img=crop_rose, ref_img=crop_diff)
        rows[rowpos].append(crop_rose)

    series_texts = {
        # 气温diff
        'T2m': {'x': 0.03,  'y': 0.20,  "fontsize": 0.023, "text": "2-m air temperature (°C)", "ha": "right", "va": "center", "color": 'black', "fontweight": "bold", "rotation":-90},
        '(a)': {'x': 0.045, 'y': 0.01,  "fontsize": 0.018, "text": "(a)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        '(b)': {'x': 0.275, 'y': 0.01,  "fontsize": 0.018, "text": "(b)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        '(c)': {'x': 0.505, 'y': 0.01,  "fontsize": 0.018, "text": "(c)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        '(d)': {'x': 0.737, 'y': 0.01,  "fontsize": 0.018, "text": "(d)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        # 气温rose
        '(e)': {'x': 0.045, 'y': 0.21,  "fontsize": 0.018, "text": "(e)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        '(f)': {'x': 0.275, 'y': 0.21,  "fontsize": 0.018, "text": "(f)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        '(g)': {'x': 0.505, 'y': 0.21,  "fontsize": 0.018, "text": "(g)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        '(h)': {'x': 0.737, 'y': 0.21,  "fontsize": 0.018, "text": "(h)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        # 相对湿度diff
        'RH' : {'x': 0.03,  'y': 0.71,  "fontsize": 0.023, "text": "Relative humidity (%)", "ha": "right", "va": "center", "color": "black", "fontweight": "bold", "rotation":-90},
        '(i)': {'x': 0.045, 'y': 0.52,  "fontsize": 0.018, "text": "(i)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        '(j)': {'x': 0.275, 'y': 0.52,  "fontsize": 0.018, "text": "(j)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        '(k)': {'x': 0.505, 'y': 0.52,  "fontsize": 0.018, "text": "(k)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        '(l)': {'x': 0.737, 'y': 0.52,  "fontsize": 0.018, "text": "(l)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        # 相对湿度rose
        '(m)': {'x': 0.045, 'y': 0.718, "fontsize": 0.018, "text": "(m)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        '(n)': {'x': 0.275, 'y': 0.718, "fontsize": 0.018, "text": "(n)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        '(o)': {'x': 0.505, 'y': 0.718, "fontsize": 0.018, "text": "(o)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        '(p)': {'x': 0.737, 'y': 0.718, "fontsize": 0.018, "text": "(p)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
    }

    big_img = TIT.merge_images_Row(
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
    # savepath = f'{FigOutDir_out}/Fig_1_{target}_Seasonal_Comparison.{OUTFIGFMT}'
    # TIT.save(big_img, savepath, dpi=DPI)  # 会自动使用reportlab保存PDF
    # resize_img = TIT.set_image_width_inch(big_img, width_inch=21, dpi=DPI)

    # 或者保存为PNG
    savepath_png = f'{FigOutDir_out}/{INFIGFMT}/{FigLabel}.{INFIGFMT}'
    savepath_pdf = f'{FigOutDir_out}/{OUTFIGFMT}/{FigLabel}.{OUTFIGFMT}'
    TIT.save(big_img, savepath_png, dpi=DPI)
    TIT.save(big_img, savepath_pdf, dpi=DPI, page_size="original")
    
    # savepath_png = f'{FigOutDir_out}/Fig_1_resize.{INFIGFMT}'
    # TIT.save(resize_img, savepath_png, dpi=DPI)
    # savepath_pdf = f'{FigOutDir_out}/Fig_1_resize.{OUTFIGFMT}'
    # TIT.save(resize_img, savepath_pdf, dpi=DPI, page_size="original")




def Fig_2(FigOutDir: str, FigLabel: str) -> None:
    """
    合并区域气候效应图(气温diff, 气温diurnal, 相对湿度diff, 相对湿度diurnal)
    """
    seasons = TU.get_seasons()
    map_cropparams = TYCM.MapPlot_CropParams_noColorbar()
    figbox_space = {'left': 0.01, 'top': 0.01, 'right': 0.01, 'bottom': 0.01}
    target = "Influence_Mechanism"
    FigOutDir_out = f'{FigOutDir}/Review_1'
    os.makedirs(FigOutDir_out, exist_ok=True)
    
    cols_space = [[0, 0, 0], [0, 0, 0]]
    rows_space = [0.01, 0.01]

    # === Summer ===
    # (1) Influence_Mechanism
    infmec_path = f"{FigOutDir_out}/Single/JJA_cap.png"
    crop_params =  {"left": 0.01, "top": 0.01, "right": 0.01, "bottom": 0.01}
    Summer_infmec_img = TIT.crop_image_from_path(infmec_path, crop_params=crop_params, mode="ratio")
    Summer_infmec_img = TIT.resize_image_scale(Summer_infmec_img, 0.777)

    # (2) PBLH
    pblh_path = f"{FigOutDir}/SpatialMap/Single/SpatialMap_PBLH_Lake_seasonal_JJA.png"
    crop_params =  {"left": 0.0, "top": 0.0, "right": 0.0, "bottom": 0.0}
    Summer_pblh_img = TIT.crop_image_from_path(pblh_path, crop_params=crop_params, mode="ratio")
    
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

    savepath_png = f'{FigOutDir_out}/{INFIGFMT}/{FigLabel}.{INFIGFMT}'
    savepath_pdf = f'{FigOutDir_out}/{OUTFIGFMT}/{FigLabel}.{OUTFIGFMT}'
    TIT.save(big_img, savepath_png, dpi=DPI)
    TIT.save(big_img, savepath_pdf, dpi=DPI, page_size="original")



def Fig_3(FigOutDir: str, FigLabel: str) -> None:
    figbox_space = TYCM.Merge_Fig_Space_Params()
    cbar_space = TYCM.MapCbar_CropParams_V()
    eventslist = ["ColdWet", "Wet", "HotWet", "Cold", "Hot", "ColdDry", "Dry", "HotDry"]
    crop_params =  {"left": 0.0, "top": 0.0, "right": 0.0, "bottom": 0.0}
    FigOutDir_in = f'{FigOutDir}/ExtremeEventFreq'
    FigOutDir_out = f'{FigOutDir}/Review_1'
    target = "ExtremeEventFreq"
    checkmethod="Wilcoxon_signed-rank_test"
    cols_space = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    rows_space = [0.01, 0.1, 0.01]
    fig_rows = [["", "", ""], ["", "", ""], ["", "", ""]]
    eventorder = {"ColdWet": [0, 0], "Wet":    [0, 1], "HotWet": [0, 2],
                  "Cold":    [1, 0], "Define": [1, 1], "Hot":    [1, 2],
                  "ColdDry": [2, 0], "Dry":    [2, 1], "HotDry": [2, 2],}

    definepath = f"{FigOutDir_in}/Single/Extreme_Event_Definition_Scheme.{INFIGFMT}"
    define_img = TIT.crop_image_from_path(definepath, crop_params=crop_params, mode="ratio")
    fig_rows[eventorder["Define"][0]][eventorder["Define"][1]] = define_img
    
    for event in eventslist:
        figpath = f'{FigOutDir_in}/Single/ExtremeEvent_{event}_Freq_Change_CircularRing_{checkmethod}.{INFIGFMT}'
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

    savepath_png = f'{FigOutDir_out}/{INFIGFMT}/{FigLabel}.{INFIGFMT}'
    savepath_pdf = f'{FigOutDir_out}/{OUTFIGFMT}/{FigLabel}.{OUTFIGFMT}'
    TIT.save(freq_img, savepath_png, dpi=DPI)
    TIT.save(freq_img, savepath_pdf, dpi=DPI, page_size="original")



def Fig_4(FigOutDir: str, FigLabel: str) -> None:
    figbox_space = TYCM.Merge_Fig_Space_Params()
    cbar_space = TYCM.MapCbar_CropParams_V() 
    crop_params =  {"left": 0.0, "top": 0.0, "right": 0.0, "bottom": 0.0}
    FigOutDir_out = f'{FigOutDir}/Review_1'
    target = "HotWetCoupling"
    cols_space = [[0]]
    rows_space = [0]
    lambda_u_path = f"{FigOutDir}/CouplingTest/Single/CouplingMetrics_Lake-NoLake_T2m-RH_lambda_u.{INFIGFMT}"
    lambda_u_img = TIT.crop_image_from_path(lambda_u_path, crop_params=crop_params, mode="ratio")
    cbar_path = f"{FigOutDir}/CouplingTest/Single/CouplingMetrics_Lake-NoLake_T2m-RH_lambda_u_VColorbar.{INFIGFMT}"
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

    joint_path = f"{FigOutDir}/CouplingTest/Joint/Extreme_Temperature_Humidity_Coupling_State_Site_x104_y83_with_KDE.{INFIGFMT}"
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

    savepath_png = f'{FigOutDir_out}/{INFIGFMT}/{FigLabel}.{INFIGFMT}'
    savepath_pdf = f'{FigOutDir_out}/{OUTFIGFMT}/{FigLabel}.{OUTFIGFMT}'
    TIT.save(big_img, savepath_png, dpi=DPI)
    TIT.save(big_img, savepath_pdf, dpi=DPI, page_size="original")



def Fig_5(FigOutDir: str, FigLabel: str) -> None:
    figbox_space = TYCM.Merge_Fig_Space_Params()
    cbar_space = TYCM.MapCbar_CropParams_V() 
    FigOutDir_out = f'{FigOutDir}/Review_1'
    target = "Affected_Population"
    checkmethod = "Wilcoxon_signed-rank_test"
    event = "HotWet"
    cols_space = [[0],[0]]
    rows_space = [0, 0]

    crop_params =  {"left": 0.05, "top": 0.0, "right": 0.05, "bottom": 0.07}
    joint_path = f'{FigOutDir}/ExtremeEventAddtional/Single/{target}_RadialHistogram_{event}-up_{checkmethod}.{INFIGFMT}'
    up_img = TIT.crop_image_from_path(joint_path, crop_params=crop_params, mode="ratio")
    up_img = TIT.resize_image_scale(up_img, 1.4)


    crop_params =  {"left": 0.0098, "top": 0.0098, "right": 0.0098, "bottom": 0.0098}
    citypath = f'{FigOutDir}/ExtremeEventAddtional/Single/{target}_Map_{event}-up_{checkmethod}-shpfile.{INFIGFMT}'
    city_img = TIT.crop_image_from_path(citypath, crop_params=crop_params, mode="ratio")
    cbar_path = f'{FigOutDir}/ExtremeEventAddtional/Single/{target}_Map_{event}_HColorbar-shpfile_{checkmethod}.{INFIGFMT}'
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

    savepath_png = f'{FigOutDir_out}/{INFIGFMT}/{FigLabel}.{INFIGFMT}'
    savepath_pdf = f'{FigOutDir_out}/{OUTFIGFMT}/{FigLabel}.{OUTFIGFMT}'
    TIT.save(big_img, savepath_png, dpi=DPI)
    TIT.save(big_img, savepath_pdf, dpi=DPI, page_size="original")



#################################
# Supporting_information
#################################

def Fig_S1(FigOutDir: str, FigLabel: str) -> None:
    FigOutDir_out = f'{FigOutDir}/Review_1'
    figpath = f"./Figures/StudyArea.{INFIGFMT}"
    crop_params =  {"left": 0.0, "top": 0.0, "right": 0.0, "bottom": 0.0}
    big_img = TIT.crop_image_from_path(figpath, crop_params=crop_params, mode="ratio")
    savepath_png = f'{FigOutDir_out}/{INFIGFMT}/{FigLabel}.{INFIGFMT}'
    savepath_pdf = f'{FigOutDir_out}/{OUTFIGFMT}/{FigLabel}.{OUTFIGFMT}'
    TIT.save(big_img, savepath_png, dpi=DPI)
    TIT.save(big_img, savepath_pdf, dpi=DPI, page_size="original")




def Fig_S2(FigOutDir: str, FigLabel: str) -> None:
    figbox_space = TYCM.Merge_Fig_Space_Params()
    cbar_space = TYCM.MapCbar_CropParams_V() 
    FigOutDir_out = f'{FigOutDir}/Review_1'
    target = "StaticStability"
    seasons = TU.get_seasons()
    cols_space = [[0, 0], [0, 0]]
    rows_space = [0, 0]
    fig_rows = [[]]

    crop_params =  {"left": 0.005, "top": 0.005, "right": 0.005, "bottom": 0.005}
    figpath = f"{FigOutDir}/Step1_Schematic_N_Years.{INFIGFMT}"
    S1_img = TIT.crop_image_from_path(figpath, crop_params=crop_params, mode="ratio")
    figpath = f"{FigOutDir}/Step2_Threshold_Distribution.{INFIGFMT}"
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

    savepath_png = f'{FigOutDir_out}/{INFIGFMT}/{FigLabel}.{INFIGFMT}'
    savepath_pdf = f'{FigOutDir_out}/{OUTFIGFMT}/{FigLabel}.{OUTFIGFMT}'
    TIT.save(big_img, savepath_png, dpi=DPI)
    TIT.save(big_img, savepath_pdf, dpi=DPI, page_size="original")



def Fig_S3(FigOutDir: str, FigLabel: str) -> None:
    seasons = TU.get_seasons()
    mapcrop_params = TYCM.MapPlot_CropParams_noColorbar()
    figbox_space = TYCM.Merge_Fig_Space_Params()
    cbar_space = TYCM.MapCbar_CropParams_H()
    FigOutDir_out = f'{FigOutDir}/Review_1'
    target = "ModelValidation"
    cols_space = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    rows_space = [0.01, 0.01]
    
    varlist  = ["T2m"]
    reflist = ["CMFDv2", "ERA5-Land"]
    for varname in varlist:
        fig_rows = [[], [], []]
        for i, dataname in enumerate([*reflist]):
            rowpos = i
            for j, season in enumerate(seasons):
                figpath = f"{FigOutDir}/ModelValidation/{varname}/Single/Diff_Lake_{varname}_{dataname}_seasonal_{season}.{INFIGFMT}"
                croped_fig = TIT.crop_image_from_path(figpath, crop_params=mapcrop_params, mode="ratio")
                fig_rows[rowpos].append(croped_fig)
        rowpos = -1
        cbarpath = f"{FigOutDir}/ModelValidation/{varname}/Single/Diff_Map_Lake_{varname}_Seasonal_{season}_HColorbar.{INFIGFMT}"
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

        savepath_png = f'{FigOutDir_out}/{INFIGFMT}/{FigLabel}.{INFIGFMT}'
        savepath_pdf = f'{FigOutDir_out}/{OUTFIGFMT}/{FigLabel}.{OUTFIGFMT}'
        TIT.save(big_img, savepath_png, dpi=DPI)
        TIT.save(big_img, savepath_pdf, dpi=DPI, page_size="original")



def Fig_S4(FigOutDir: str, FigLabel: str) -> None:
    seasons = TU.get_seasons()
    mapcrop_params = TYCM.MapPlot_CropParams_noColorbar()
    figbox_space = TYCM.Merge_Fig_Space_Params()
    cbar_space = TYCM.MapCbar_CropParams_H()
    FigOutDir_out = f'{FigOutDir}/Review_1'
    target = "ModelValidation"
    cols_space = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    rows_space = [0.01, 0.01]
    
    varlist  = ["Q2m"]
    reflist = ["CMFDv2", "ERA5-Land"]
    for varname in varlist:
        fig_rows = [[], [], []]
        for i, dataname in enumerate([*reflist]):
            rowpos = i
            for j, season in enumerate(seasons):
                figpath = f"{FigOutDir}/ModelValidation/{varname}/Single/Diff_Lake_{varname}_{dataname}_seasonal_{season}.{INFIGFMT}"
                croped_fig = TIT.crop_image_from_path(figpath, crop_params=mapcrop_params, mode="ratio")
                fig_rows[rowpos].append(croped_fig)
        rowpos = -1
        cbarpath = f"{FigOutDir}/ModelValidation/{varname}/Single/Diff_Map_Lake_{varname}_Seasonal_{season}_HColorbar.{INFIGFMT}"
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

        savepath_png = f'{FigOutDir_out}/{INFIGFMT}/{FigLabel}.{INFIGFMT}'
        savepath_pdf = f'{FigOutDir_out}/{OUTFIGFMT}/{FigLabel}.{OUTFIGFMT}'
        TIT.save(big_img, savepath_png, dpi=DPI)
        TIT.save(big_img, savepath_pdf, dpi=DPI, page_size="original")




def Fig_S5(FigOutDir: str, FigLabel: str) -> None:
    seasons = TU.get_seasons()
    mapcrop_params = TYCM.MapPlot_CropParams_noColorbar()
    figbox_space = TYCM.Merge_Fig_Space_Params()
    cbar_space = TYCM.MapCbar_CropParams_H()
    FigOutDir_out = f'{FigOutDir}/Review_1'
    target = "ModelValidation"
    cols_space = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    rows_space = [0.01, 0.01]
    
    varlist  = ["Prec"]
    reflist = ["CMFDv2", "ERA5-Land"]
    for varname in varlist:
        fig_rows = [[], [], []]
        for i, dataname in enumerate([*reflist]):
            rowpos = i
            for j, season in enumerate(seasons):
                figpath = f"{FigOutDir}/ModelValidation/{varname}/Single/Diff_Lake_{varname}_{dataname}_seasonal_{season}.{INFIGFMT}"
                croped_fig = TIT.crop_image_from_path(figpath, crop_params=mapcrop_params, mode="ratio")
                fig_rows[rowpos].append(croped_fig)
        rowpos = -1
        cbarpath = f"{FigOutDir}/ModelValidation/{varname}/Single/Diff_Map_Lake_{varname}_Seasonal_{season}_HColorbar.{INFIGFMT}"
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

        savepath_png = f'{FigOutDir_out}/{INFIGFMT}/{FigLabel}.{INFIGFMT}'
        savepath_pdf = f'{FigOutDir_out}/{OUTFIGFMT}/{FigLabel}.{OUTFIGFMT}'
        TIT.save(big_img, savepath_png, dpi=DPI)
        TIT.save(big_img, savepath_pdf, dpi=DPI, page_size="original")




def Fig_S6(FigOutDir: str, FigLabel: str) -> None:
    seasons = TU.get_seasons()
    mapcrop_params = TYCM.MapPlot_CropParams_noColorbar()
    figbox_space = TYCM.Merge_Fig_Space_Params()
    cbar_space = TYCM.MapCbar_CropParams_H()
    FigOutDir_out = f'{FigOutDir}/Review_1'
    target = "ModelValidation"
    cols_space = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    rows_space = [0.01, 0.01]
    
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
                figpath = f"{FigOutDir}/ModelValidation/{varname}/Single/Mean_Lake_{varname}_{dataname}_seasonal_{season}.{INFIGFMT}"
                croped_fig = TIT.crop_image_from_path(figpath, crop_params=mapcrop_params, mode="ratio")
                fig_rows[rowpos].append(croped_fig)
        rowpos = -1
        cbarpath = f"{FigOutDir}/ModelValidation/{varname}/Single/Mean_Map_Lake_{varname}_Seasonal_{season}_HColorbar.{INFIGFMT}"
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

        savepath_png = f'{FigOutDir_out}/{INFIGFMT}/{FigLabel}.{INFIGFMT}'
        savepath_pdf = f'{FigOutDir_out}/{OUTFIGFMT}/{FigLabel}.{OUTFIGFMT}'
        TIT.save(big_img, savepath_png, dpi=DPI)
        TIT.save(big_img, savepath_pdf, dpi=DPI, page_size="original")



def Fig_S7(FigOutDir: str, FigLabel: str) -> None:
    FigOutDir_out = f'{FigOutDir}/Review_1'

    case = "Lake"
    map_mode = "Daily"
    box_mode_label = "Daily_vs_SeasonalMean"
    target_map = "StationMetricsMap"
    target_box = "StationMetricsBox"

    map_crop = {"left": 0.01, "top": 0.01, "right": 0.01, "bottom": 0.01}
    cbar_crop = {"left": 0.01, "top": 0.018, "right": 0.02, "bottom": 0.018}
    box_crop = {"left": 0.01, "top": 0.01, "right": 0.01, "bottom": 0.01}


    rows_config = [
            {"var": "T2m", "map_metric": "MAE",  "box_metrics": ["RMSE", "MAE"]},
            {"var": "Prec", "map_metric": "MAE", "box_metrics": ["RMSE", "MAE"]},
        ]
    if len(rows_config) != 2:
        raise ValueError("rows_config must contain exactly two row configs for a 2-row figure.")

    normalized_rows = []
    for irow, row_cfg in enumerate(rows_config, start=1):
        if not isinstance(row_cfg, dict):
            raise TypeError(f"rows_config[{irow - 1}] must be a dict.")
        var = row_cfg.get("var")
        map_metric = row_cfg.get("map_metric")
        box_metrics = row_cfg.get("box_metrics")
        if var is None or map_metric is None or box_metrics is None:
            raise ValueError(
                f"rows_config[{irow - 1}] must define 'var', 'map_metric', and 'box_metrics'."
            )
        if len(box_metrics) != 2:
            raise ValueError(f"rows_config[{irow - 1}]['box_metrics'] must contain exactly two metrics.")
        normalized_rows.append({
            "var": str(var),
            "map_metric": str(map_metric),
            "box_metrics": [str(box_metrics[0]), str(box_metrics[1])],
        })


    def _check_image(path_in):
        if not os.path.exists(path_in):
            raise FileNotFoundError(f"Required station validation panel image not found: {path_in}")
        return path_in

    rows = []
    for row_cfg in normalized_rows:
        var = row_cfg["var"]
        map_metric = row_cfg["map_metric"]
        box_metric_1, box_metric_2 = row_cfg["box_metrics"]

        map_path = _check_image(
            f"{FigOutDir}/ModelValidation_Station/{target_map}/{case}/{var}/{map_metric}/"
            f"{case}_{var}_{map_mode}_Station_Map_{map_metric}.{INFIGFMT}"
        )
        cbar_path = _check_image(
            f"{FigOutDir}/ModelValidation_Station/{target_map}/{case}/{var}/{map_metric}/"
            f"{case}_{var}_{map_mode}_Station_Map_Colorbar_{map_metric}.{INFIGFMT}"
        )
        box_path_1 = _check_image(
            f"{FigOutDir}/ModelValidation_Station/{target_box}/{case}/{var}/{box_metric_1}/"
            f"{case}_{var}_{box_mode_label}_Station_Box_{box_metric_1}.{INFIGFMT}"
        )
        box_path_2 = _check_image(
            f"{FigOutDir}/ModelValidation_Station/{target_box}/{case}/{var}/{box_metric_2}/"
            f"{case}_{var}_{box_mode_label}_Station_Box_{box_metric_2}.{INFIGFMT}"
        )

        map_img = TIT.crop_image_from_path(map_path, crop_params=map_crop, mode="ratio")
        cbar_img = TIT.crop_image_from_path(cbar_path, crop_params=cbar_crop, mode="ratio")
        cbar_img = TIT.adjust_image_to_ref_canvas(
            target_img=cbar_img,
            ref_img=map_img,
            axis="height",
        )
        map_panel = TIT.merge_images_Row(
            rows_images=[[map_img, cbar_img]],
            cols_space=[[0.005]],
            rows_space=[],
            box_space={"left": 0, "top": 0, "right": 0, "bottom": 0},
            background_color="#FFFFFF",
            space_mode="ratio",
            alignment=["left"],
            draw_ticks=False,
            font_path="/home/wumej22/.local/share/fonts/NotoSans-Bold.ttf",
        )

        box_img_1 = TIT.crop_image_from_path(box_path_1, crop_params=box_crop, mode="ratio")
        box_img_2 = TIT.crop_image_from_path(box_path_2, crop_params=box_crop, mode="ratio")
        box_img_1 = TIT.adjust_image_to_ref_canvas(
            target_img=box_img_1,
            ref_img=map_panel,
            axis="height",
        )
        box_img_2 = TIT.adjust_image_to_ref_canvas(
            target_img=box_img_2,
            ref_img=map_panel,
            axis="height",
        )
        rows.append([map_panel, box_img_1, box_img_2])

    panel_texts = {
        "label1": {"x": 0.020, "y": 0.25, "fontsize": 0.04, "text": "2-m air temperature", "ha": "center", "va": "center", "color": "black", "fontweight": "bold", "rotation": -90},
        "label2": {"x": 0.020, "y": 0.75, "fontsize": 0.04, "text": "Precipitation", "ha": "center", "va": "center", "color": "black", "fontweight": "bold", "rotation": -90},
        "(a)": {"x": 0.045, "y": 0.015, "fontsize": 0.032, "text": "(a)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        "(b)": {"x": 0.585, "y": 0.030, "fontsize": 0.032, "text": "(b)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        "(c)": {"x": 0.822, "y": 0.030, "fontsize": 0.032, "text": "(c)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        "(d)": {"x": 0.045, "y": 0.515, "fontsize": 0.032, "text": "(d)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        "(e)": {"x": 0.585, "y": 0.530, "fontsize": 0.032, "text": "(e)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        "(f)": {"x": 0.822, "y": 0.530, "fontsize": 0.032, "text": "(f)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
    }

    big_img = TIT.merge_images_Row(
        rows_images=rows,
        cols_space=[[0.010, 0.010], [0.010, 0.010]],
        rows_space=[0.02],
        box_space={"left": 0.08, "top": 0.01, "right": 0.01, "bottom": 0.01},
        background_color="#FFFFFF",
        space_mode="ratio",
        alignment=["left", "left"],
        draw_ticks=False,
        tick_step=0.01,
        texts=panel_texts,
        font_path="/home/wumej22/.local/share/fonts/NotoSans-Bold.ttf",
    )

    savepath_png = f"{FigOutDir_out}/{INFIGFMT}/{FigLabel}.{INFIGFMT}"
    savepath_pdf = f"{FigOutDir_out}/{OUTFIGFMT}/{FigLabel}.{OUTFIGFMT}"
    TIT.save(big_img, savepath_png, dpi=DPI)
    TIT.save(big_img, savepath_pdf, dpi=DPI, page_size="original")



def Fig_S8(FigOutDir: str, FigLabel: str) -> None:
    FigOutDir_out = f'{FigOutDir}/Review_1'
    figpath = f"./Figures/RegClimImpact/T2m/Single/RegClimImpact_DepthDependence_T2m_Paired_t-test.{INFIGFMT}"
    crop_params =  {"left": 0.0, "top": 0.0, "right": 0.0, "bottom": 0.0}
    big_img = TIT.crop_image_from_path(figpath, crop_params=crop_params, mode="ratio")
    savepath_png = f'{FigOutDir_out}/{INFIGFMT}/{FigLabel}.{INFIGFMT}'
    savepath_pdf = f'{FigOutDir_out}/{OUTFIGFMT}/{FigLabel}.{OUTFIGFMT}'
    TIT.save(big_img, savepath_png, dpi=DPI)
    TIT.save(big_img, savepath_pdf, dpi=DPI, page_size="original")




def Fig_S9(FigOutDir: str, FigLabel: str) -> None:
    seasons = TU.get_seasons()
    map_cropparams = TYCM.MapPlot_CropParams_noColorbar()
    figbox_space = {'left': 0.15, 'top': 0.01, 'right': 0.01, 'bottom': 0.01}
    cbar_cropparams = {"left": 0.005, "top": 0.005, "right": 0.3, "bottom": 0.005}
    rose_cropparams = TYCM.RosePlot_CropParams_noColorbar()
    onlysig = True
    suffix = "onlysig" if onlysig else "all"
    target = "RegClimImpact"
    FigOutDir_in = f'{FigOutDir}/RegClimImpact'
    FigOutDir_out = f'{FigOutDir}/Review_1'
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

    for varname in ["Q2m",]: # "UV10",
        varinfo = TYCM.Variable_Infos(varname)
        longname = varinfo["longname"]
        unit = varinfo["unit"]
        rows = [[], []]
        rowpos = 0
        for season in seasons:
            figpath = f'{FigOutDir_in}/{varname}/Single/{target}_Diff_Map_{varname}_{season}_{suffix}_{checkmethod}.{INFIGFMT}'
            crop_diff = TIT.crop_image_from_path(figpath, crop_params=map_cropparams, mode="ratio")
            rows[rowpos].append(crop_diff)

        cbarpath = f'{FigOutDir_in}/{varname}/Single/{target}_Diff_Map_{varname}_Seasonal_Colorbar_{suffix}_{checkmethod}.{INFIGFMT}'
        crop_cbar = TIT.crop_image_from_path(cbarpath, crop_params=cbar_cropparams, mode="ratio")
        crop_cbar = TIT.adjust_image_to_ref_canvas(target_img=crop_cbar, ref_img=crop_diff, axis="height")
        rows[rowpos].append(crop_cbar)

        rowpos = 1
        for season in seasons:
            figpath = f'{FigOutDir_in}/{varname}/Single/{target}_Diff_Rose_{varname}_{season}_{suffix}_{checkmethod}.{INFIGFMT}'
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
        savepath_png = f'{FigOutDir_out}/{INFIGFMT}/{FigLabel}.{INFIGFMT}'
        savepath_pdf = f'{FigOutDir_out}/{OUTFIGFMT}/{FigLabel}.{OUTFIGFMT}'
        TIT.save(seas_img, savepath_png, dpi=DPI)
        TIT.save(seas_img, savepath_pdf, dpi=DPI, page_size="original")




def Fig_S10(FigOutDir: str, FigLabel: str) -> None:
    seasons = TU.get_seasons()
    map_cropparams = TYCM.MapPlot_CropParams_noColorbar()
    figbox_space = {'left': 0.15, 'top': 0.01, 'right': 0.01, 'bottom': 0.01}
    cbar_cropparams = {"left": 0.005, "top": 0.005, "right": 0.3, "bottom": 0.005}
    rose_cropparams = TYCM.RosePlot_CropParams_noColorbar()
    onlysig = True
    suffix = "onlysig" if onlysig else "all"
    target = "RegClimImpact"
    FigOutDir_in = f'{FigOutDir}/RegClimImpact'
    FigOutDir_out = f'{FigOutDir}/Review_1'
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

    for varname in ["LHF",]: # "UV10",
        varinfo = TYCM.Variable_Infos(varname)
        longname = varinfo["longname"]
        unit = varinfo["unit"]
        rows = [[], []]
        rowpos = 0
        for season in seasons:
            figpath = f'{FigOutDir_in}/{varname}/Single/{target}_Diff_Map_{varname}_{season}_{suffix}_{checkmethod}.{INFIGFMT}'
            crop_diff = TIT.crop_image_from_path(figpath, crop_params=map_cropparams, mode="ratio")
            rows[rowpos].append(crop_diff)

        cbarpath = f'{FigOutDir_in}/{varname}/Single/{target}_Diff_Map_{varname}_Seasonal_Colorbar_{suffix}_{checkmethod}.{INFIGFMT}'
        crop_cbar = TIT.crop_image_from_path(cbarpath, crop_params=cbar_cropparams, mode="ratio")
        crop_cbar = TIT.adjust_image_to_ref_canvas(target_img=crop_cbar, ref_img=crop_diff, axis="height")
        rows[rowpos].append(crop_cbar)

        rowpos = 1
        for season in seasons:
            figpath = f'{FigOutDir_in}/{varname}/Single/{target}_Diff_Rose_{varname}_{season}_{suffix}_{checkmethod}.{INFIGFMT}'
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
        savepath_png = f'{FigOutDir_out}/{INFIGFMT}/{FigLabel}.{INFIGFMT}'
        savepath_pdf = f'{FigOutDir_out}/{OUTFIGFMT}/{FigLabel}.{OUTFIGFMT}'
        TIT.save(seas_img, savepath_png, dpi=DPI)
        TIT.save(seas_img, savepath_pdf, dpi=DPI, page_size="original")




def Fig_S11(FigOutDir: str, FigLabel: str) -> None:
    seasons = TU.get_seasons()
    map_cropparams = TYCM.MapPlot_CropParams_noColorbar()
    figbox_space = {'left': 0.15, 'top': 0.01, 'right': 0.01, 'bottom': 0.01}
    cbar_cropparams = {"left": 0.005, "top": 0.005, "right": 0.3, "bottom": 0.005}
    rose_cropparams = TYCM.RosePlot_CropParams_noColorbar()
    onlysig = True
    suffix = "onlysig" if onlysig else "all"
    target = "RegClimImpact"
    FigOutDir_in = f'{FigOutDir}/RegClimImpact'
    FigOutDir_out = f'{FigOutDir}/Review_1'
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

    for varname in ["UV",]: # "UV10",
        varinfo = TYCM.Variable_Infos(varname)
        longname = varinfo["longname"]
        unit = varinfo["unit"]
        rows = [[], []]
        rowpos = 0
        for season in seasons:
            figpath = f'{FigOutDir}/PressureLevel/1000hPa/Single/PressureLevel_Diff_Map_{varname}_{season}_{suffix}_{checkmethod}.{INFIGFMT}'
            crop_diff = TIT.crop_image_from_path(figpath, crop_params=map_cropparams, mode="ratio")
            rows[rowpos].append(crop_diff)
        cbarpath = f'{FigOutDir}/PressureLevel/1000hPa/Single/PressureLevel_Diff_Map_{varname}_Seasonal_Colorbar_{suffix}_{checkmethod}.{INFIGFMT}'
        crop_cbar = TIT.crop_image_from_path(cbarpath, crop_params=cbar_cropparams, mode="ratio")
        crop_cbar = TIT.adjust_image_to_ref_canvas(target_img=crop_cbar, ref_img=crop_diff, axis="height")
        rows[rowpos].append(crop_cbar)

        rowpos = 1
        for season in seasons:
            figpath = f'{FigOutDir}/PressureLevel/1000hPa/Single/PressureLevel_Diff_Rose_{varname}_{season}_{suffix}_{checkmethod}.{INFIGFMT}'
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
        savepath_png = f'{FigOutDir_out}/{INFIGFMT}/{FigLabel}.{INFIGFMT}'
        savepath_pdf = f'{FigOutDir_out}/{OUTFIGFMT}/{FigLabel}.{OUTFIGFMT}'
        TIT.save(seas_img, savepath_png, dpi=DPI)
        TIT.save(seas_img, savepath_pdf, dpi=DPI, page_size="original")



def Fig_S12(FigOutDir: str, FigLabel: str) -> None:
    FigOutDir_out = f'{FigOutDir}/Review_1'
    figpath = f"./Figures/ExtremeEventFreq/ExtremeEvent_Freq_HeatMap.{INFIGFMT}"
    crop_params =  {"left": 0.0, "top": 0.0, "right": 0.0, "bottom": 0.0}
    big_img = TIT.crop_image_from_path(figpath, crop_params=crop_params, mode="ratio")
    savepath_png = f'{FigOutDir_out}/{INFIGFMT}/{FigLabel}.{INFIGFMT}'
    savepath_pdf = f'{FigOutDir_out}/{OUTFIGFMT}/{FigLabel}.{OUTFIGFMT}'
    TIT.save(big_img, savepath_png, dpi=DPI)
    TIT.save(big_img, savepath_pdf, dpi=DPI, page_size="original")




def Fig_S13(FigOutDir: str, FigLabel: str) -> None:
    figbox_space = TYCM.Merge_Fig_Space_Params()
    mapcrop_params = TYCM.MapPlot_CropParams_noColorbar()
    boxcrop_params = TYCM.Extreme_Events_BoxPlot_CropParams()
    cbar_space = TYCM.MapCbar_CropParams_V()
    eventslist = ["ColdWet", "Wet", "HotWet", "Cold", "Hot", "ColdDry", "Dry", "HotDry"]
    crop_params =  {"left": 0.0, "top": 0.0, "right": 0.0, "bottom": 0.0}
    FigOutDir_in = f'{FigOutDir}/ExtremeEventFreq'
    FigOutDir_out = f'{FigOutDir}/Review_1'
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
        figpath = f'{FigOutDir_in}/Single/Freq_Diff_Map_{event}_Annual_{suffix}_{checkmethod}.{INFIGFMT}'
        cropped_map = TIT.crop_image_from_path(figpath, crop_params=mapcrop_params, mode="ratio")
        diffcbarpath = f'{FigOutDir_in}/Single/Freq_Diff_Map_{event}_Annual_Colorbar_{suffix}_{checkmethod}.{INFIGFMT}'
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

    anual_boxpath_diff = f'{FigOutDir_in}/Single/Freq_Diff_Box_Annual_{suffix}_{checkmethod}.{INFIGFMT}'
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

    savepath_png = f'{FigOutDir_out}/{INFIGFMT}/{FigLabel}.{INFIGFMT}'
    savepath_pdf = f'{FigOutDir_out}/{OUTFIGFMT}/{FigLabel}.{OUTFIGFMT}'
    TIT.save(freq_img, savepath_png, dpi=DPI)
    TIT.save(freq_img, savepath_pdf, dpi=DPI, page_size="original")



def Fig_S14(FigOutDir: str, FigLabel: str) -> None:
    FigOutDir_out = f'{FigOutDir}/Review_1'
    figpath = f"./Figures/Review_1/Single/ExtremeEvent_Freq.{INFIGFMT}"
    crop_params =  {"left": 0.0, "top": 0.0, "right": 0.0, "bottom": 0.0}
    big_img = TIT.crop_image_from_path(figpath, crop_params=crop_params, mode="ratio")
    savepath_png = f'{FigOutDir_out}/{INFIGFMT}/{FigLabel}.{INFIGFMT}'
    savepath_pdf = f'{FigOutDir_out}/{OUTFIGFMT}/{FigLabel}.{OUTFIGFMT}'
    TIT.save(big_img, savepath_png, dpi=DPI)
    TIT.save(big_img, savepath_pdf, dpi=DPI, page_size="original")



def Fig_S15(FigOutDir: str, FigLabel: str) -> None:
    FigOutDir_out = f'{FigOutDir}/Review_1'
    figpath = f"./Figures/Review_1/Single/Freq_Diff_Map_Annual_Comparison_onlysig_Wilcoxon_signed-rank_test.{INFIGFMT}"
    crop_params =  {"left": 0.0, "top": 0.0, "right": 0.0, "bottom": 0.0}
    big_img = TIT.crop_image_from_path(figpath, crop_params=crop_params, mode="ratio")
    savepath_png = f'{FigOutDir_out}/{INFIGFMT}/{FigLabel}.{INFIGFMT}'
    savepath_pdf = f'{FigOutDir_out}/{OUTFIGFMT}/{FigLabel}.{OUTFIGFMT}'
    TIT.save(big_img, savepath_png, dpi=DPI)
    TIT.save(big_img, savepath_pdf, dpi=DPI, page_size="original")



def Fig_S16(FigOutDir: str, FigLabel: str) -> None:
    FigOutDir_out = f'{FigOutDir}/Review_1'
    figpath = f"./Figures/CouplingTest/Joint/Extreme_Temperature_Humidity_Coupling_State_All_Sites_with_KDE.{INFIGFMT}"
    crop_params =  {"left": 0.0, "top": 0.0, "right": 0.0, "bottom": 0.0}
    big_img = TIT.crop_image_from_path(figpath, crop_params=crop_params, mode="ratio")
    savepath_png = f'{FigOutDir_out}/{INFIGFMT}/{FigLabel}.{INFIGFMT}'
    savepath_pdf = f'{FigOutDir_out}/{OUTFIGFMT}/{FigLabel}.{OUTFIGFMT}'
    TIT.save(big_img, savepath_png, dpi=DPI)
    TIT.save(big_img, savepath_pdf, dpi=DPI, page_size="original")




def Fig_S17(FigOutDir: str, FigLabel: str) -> None:
    seasons = TU.get_seasons()
    mapcrop_params = TYCM.MapPlot_CropParams_noColorbar()
    figbox_space = {'left': 0.05, 'top': 0.2, 'right': 0.05, 'bottom': 0.01}
    cbar_space = TYCM.MapCbar_CropParams_H()
    FigOutDir_out = f'{FigOutDir}/Review_1'
    target = "ModelValidation"
    cols_space = [[0.1], [0.1], [0,]]
    rows_space = [0.01, 0.01]
    varlist  = ["T2m", ]
    reflist = ["CMFDv2", "ERA5-Land"]
    caselist = ["CUOFF", "Lake"]
    for varname in varlist:
        fig_rows = [[], [], []]
        for i, dataname in enumerate([*reflist]):
            rowpos = i
            season = 'JJA'
            for casename in caselist:
                figpath = f"{FigOutDir}/ModelValidation_NCP/{varname}/Single/Diff_{casename}_{varname}_{dataname}_seasonal_{season}.{INFIGFMT}"
                croped_fig = TIT.crop_image_from_path(figpath, crop_params=mapcrop_params, mode="ratio")
                fig_rows[rowpos].append(croped_fig)
        rowpos = -1
        cbarpath = f"{FigOutDir}/ModelValidation_NCP/{varname}/Single/Diff_Map_Lake_{varname}_Seasonal_{season}_HColorbar.{INFIGFMT}"
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

        savepath_png = f'{FigOutDir_out}/{INFIGFMT}/{FigLabel}.{INFIGFMT}'
        savepath_pdf = f'{FigOutDir_out}/{OUTFIGFMT}/{FigLabel}.{OUTFIGFMT}'
        TIT.save(big_img, savepath_png, dpi=DPI)
        TIT.save(big_img, savepath_pdf, dpi=DPI, page_size="original")




def Fig_S18(FigOutDir: str, FigLabel: str) -> None:
    seasons = TU.get_seasons()
    mapcrop_params = TYCM.MapPlot_CropParams_noColorbar()
    figbox_space = {'left': 0.05, 'top': 0.2, 'right': 0.05, 'bottom': 0.01}
    cbar_space = TYCM.MapCbar_CropParams_H()
    FigOutDir_out = f'{FigOutDir}/Review_1'
    target = "ModelValidation"
    cols_space = [[0.1], [0.1], [0,]]
    rows_space = [0.01, 0.01]
    varlist  = [ "Prec"]
    reflist = ["CMFDv2", "ERA5-Land"]
    caselist = ["CUOFF", "Lake"]
    for varname in varlist:
        fig_rows = [[], [], []]
        for i, dataname in enumerate([*reflist]):
            rowpos = i
            season = 'JJA'
            for casename in caselist:
                figpath = f"{FigOutDir}/ModelValidation_NCP/{varname}/Single/Diff_{casename}_{varname}_{dataname}_seasonal_{season}.{INFIGFMT}"
                croped_fig = TIT.crop_image_from_path(figpath, crop_params=mapcrop_params, mode="ratio")
                fig_rows[rowpos].append(croped_fig)
        rowpos = -1
        cbarpath = f"{FigOutDir}/ModelValidation_NCP/{varname}/Single/Diff_Map_Lake_{varname}_Seasonal_{season}_HColorbar.{INFIGFMT}"
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

        savepath_png = f'{FigOutDir_out}/{INFIGFMT}/{FigLabel}.{INFIGFMT}'
        savepath_pdf = f'{FigOutDir_out}/{OUTFIGFMT}/{FigLabel}.{OUTFIGFMT}'
        TIT.save(big_img, savepath_png, dpi=DPI)
        TIT.save(big_img, savepath_pdf, dpi=DPI, page_size="original")


