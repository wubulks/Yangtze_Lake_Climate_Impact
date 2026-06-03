import os
import gc 
import time
import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm
from scipy import stats
import seaborn as sns # 建议引入 seaborn 简化分组绘图逻辑
import matplotlib.pyplot as plt
from dataclasses import dataclass
from joblib import Parallel, delayed
from contextlib import contextmanager
from typing import Literal, Optional, Tuple, Dict, Any, Sequence, List
from statsmodels.stats.multitest import multipletests

# 自定义工具箱
import ToolBoxes.Utils as TU
import ToolBoxes.Tool_ImageToolkit as TIT
import ToolBoxes.Tool_PlotRose as TPR
import ToolBoxes.Tool_PlotBox as TPB
import ToolBoxes.Tool_PlotColorBar as TPCB
import ToolBoxes.Tool_PlotConfig as TPC
import ToolBoxes.Tool_DataPrepare as TDP
import ToolBoxes.Tool_InputOutput as TIO
import ToolBoxes.Tool_PlotAreaMap as TPAM
import ToolBoxes.Tool_YangtzeColorMap as TYCM
import ToolBoxes.Tool_SignificanceTest as TST
FIGFMT = TPC.FIGFMT
DPI    = TPC.DPI_medium




def RegClimImpact_Additional(
        varname: str, caselist: List[str],
        checkmethod: str, OutDir:str, FigOutDir: str,
        lkinfos: Any, onlysig: bool = True) -> None:
    """
    区域气候影响绘图
    """
    target = "RegClimImpact"
    seasons = TU.get_seasons()
    keep_hours = TU.get_all_hours()
    lakemask  = lkinfos['all']
    area_km2  = lkinfos['area'] / (1000*1000)
    oceanmask = lkinfos['ocean']
    is_ocean = (oceanmask != 0)
    is_inland = ~is_ocean
    is_lake = (lakemask > 0) & is_inland
    is_land = is_inland & (~is_lake)
    FigOutDir_var = f"{FigOutDir}/{varname}/Single/"


    # lake_df = pd.DataFrame(columns=['Season', 'Min', 'Max', 'Median', 'Mean', 'Q25', 'Q75'])
    # land_df = pd.DataFrame(columns=['Season', 'Min', 'Max', 'Median', 'Mean', 'Q25', 'Q75'])
    lake_In_df = {}
    lake_De_df = {}
    land_In_df = {}
    land_De_df = {}
    all_In_df = {}
    all_De_df = {}

    lake_samples = {}  # season -> 1D array
    land_samples = {}  # season -> 1D array

    for season in seasons:
        path_seasonal = f'{OutDir}/RegClimImpact/Significance_RegClimImpact_{caselist[0]}-{caselist[1]}_{varname}_seasonal_{season}_{checkmethod}.nc'
        xarr_seasonal= TIO.read_newnc(path_seasonal)
        diffdata, sig_mask, dash_mask, strong_mask, weak_mask, suffix, area_all, area_strong = TDP.prepara_for_mapplot(xarr_seasonal, lkinfos, 'mean_diff', checkmethod, onlysig)

        all_In = np.abs(np.where((sig_mask == 1) & (diffdata > 0), diffdata, np.nan))
        all_De = np.abs(np.where((sig_mask == 1) & (diffdata < 0), diffdata, np.nan))
        lake_In = np.abs(np.where((sig_mask == 1) & (diffdata > 0) & is_lake, diffdata, np.nan))
        lake_De = np.abs(np.where((sig_mask == 1) & (diffdata < 0) & is_lake, diffdata, np.nan))
        land_In = np.abs(np.where((sig_mask == 1) & (diffdata > 0) & is_land, diffdata, np.nan))
        land_De = np.abs(np.where((sig_mask == 1) & (diffdata < 0) & is_land, diffdata, np.nan))

        # Increse
        all_In_min = np.nanmin(all_In)
        all_In_mean = np.nanmean(all_In)
        all_In_med = np.nanmedian(all_In)
        all_In_max = np.nanmax(all_In)
        all_In_q25 = np.nanpercentile(all_In, 25)
        all_In_q75 = np.nanpercentile(all_In, 75)
        all_In_df[season] = {'Min': round(all_In_min, 2), 'Mean': round(all_In_mean, 2),
                             'Max': round(all_In_max, 2), 'Median': round(all_In_med, 2),
                             'Q25': round(all_In_q25, 2), 'Q75': round(all_In_q75, 2)}

        lake_In_min = np.nanmin(lake_In)
        lake_In_mean = np.nanmean(lake_In)
        lake_In_med = np.nanmedian(lake_In)
        lake_In_max = np.nanmax(lake_In)
        lake_In_q25 = np.nanpercentile(lake_In, 25)
        lake_In_q75 = np.nanpercentile(lake_In, 75)
        lake_In_df[season] = {'Min': round(lake_In_min, 2), 'Mean': round(lake_In_mean, 2),
                             'Max': round(lake_In_max, 2), 'Median': round(lake_In_med, 2),
                             'Q25': round(lake_In_q25, 2), 'Q75': round(lake_In_q75, 2)}
        
        land_In_min = np.nanmin(land_In)
        land_In_mean = np.nanmean(land_In)
        land_In_med = np.nanmedian(land_In)
        land_In_max = np.nanmax(land_In)
        land_In_q25 = np.nanpercentile(land_In, 25)
        land_In_q75 = np.nanpercentile(land_In, 75)
        land_In_df[season] = {'Min': round(land_In_min, 2), 'Mean': round(land_In_mean, 2),
                             'Max': round(land_In_max, 2), 'Median': round(land_In_med, 2),
                             'Q25': round(land_In_q25, 2), 'Q75': round(land_In_q75, 2)}

        # Decrease
        all_De_min = np.nanmin(all_De)
        all_De_mean = np.nanmean(all_De)
        all_De_med = np.nanmedian(all_De)
        all_De_max = np.nanmax(all_De)
        all_De_q25 = np.nanpercentile(all_De, 25)
        all_De_q75 = np.nanpercentile(all_De, 75)
        all_De_df[season] = {'Min': round(all_De_min, 2), 'Mean': round(all_De_mean, 2),
                             'Max': round(all_De_max, 2), 'Median': round(all_De_med, 2),
                             'Q25': round(all_De_q25, 2), 'Q75': round(all_De_q75, 2)}

        lake_De_min = np.nanmin(lake_De)
        lake_De_mean = np.nanmean(lake_De)
        lake_De_med = np.nanmedian(lake_De)
        lake_De_max = np.nanmax(lake_De)
        lake_De_q25 = np.nanpercentile(lake_De, 25)
        lake_De_q75 = np.nanpercentile(lake_De, 75)
        lake_De_df[season] = {'Min': round(lake_De_min, 2), 'Mean': round(lake_De_mean, 2),
                             'Max': round(lake_De_max, 2), 'Median': round(lake_De_med, 2),
                             'Q25': round(lake_De_q25, 2), 'Q75': round(lake_De_q75, 2)}

        land_De_min = np.nanmin(land_De)
        land_De_mean = np.nanmean(land_De)
        land_De_med = np.nanmedian(land_De)
        land_De_max = np.nanmax(land_De)
        land_De_q25 = np.nanpercentile(land_De, 25)
        land_De_q75 = np.nanpercentile(land_De, 75)
        land_De_df[season] = {'Min': round(land_De_min, 2), 'Mean': round(land_De_mean, 2),
                             'Max': round(land_De_max, 2), 'Median': round(land_De_med, 2),
                             'Q25': round(land_De_q25, 2), 'Q75': round(land_De_q75, 2)}

    lake_In_df = pd.DataFrame.from_dict(lake_In_df, orient='index', columns=['Min', 'Mean', 'Max', 'Median', 'Q25', 'Q75'])
    lake_De_df = pd.DataFrame.from_dict(lake_De_df, orient='index', columns=['Min', 'Mean', 'Max', 'Median', 'Q25', 'Q75'])
    land_In_df = pd.DataFrame.from_dict(land_In_df, orient='index', columns=['Min', 'Mean', 'Max', 'Median', 'Q25', 'Q75'])
    land_De_df = pd.DataFrame.from_dict(land_De_df, orient='index', columns=['Min', 'Mean', 'Max', 'Median', 'Q25', 'Q75'])
    all_In_df = pd.DataFrame.from_dict(all_In_df, orient='index', columns=['Min', 'Mean', 'Max', 'Median', 'Q25', 'Q75'])   
    all_De_df = pd.DataFrame.from_dict(all_De_df, orient='index', columns=['Min', 'Mean', 'Max', 'Median', 'Q25', 'Q75'])

    print(f"\n\n\nAdditional Info for {varname}:")
    print("All Increase Summary:")
    print(all_In_df)
    print("\nAll Decrease Summary:")
    print(all_De_df)
    print("\nLake Increase Summary:")
    print(lake_In_df)
    print("\nLake Decrease Summary:")
    print(lake_De_df)
    print("\nLand Increase Summary:")
    print(land_In_df)
    print("\nLand Decrease Summary:")
    print(land_De_df)

