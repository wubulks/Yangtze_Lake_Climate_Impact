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


def CWRFSkill_Index(OutDir: str, lkinfos: Dict[str, Any])-> None:
    """
    计算CWRF模式的技能指数
    """
    reflist = ["ERA5-Land", "CMFDv2"]
    varlist = ["T2m", "Q2m", "Prec"]
    seasons = TU.get_seasons()

    # 读取数据
    print("\n\n模型技能得分")
    casename = 'Lake'
    for varname in varlist:
        print(f"  -> {varname} ")
        for season in seasons:
            print(f"    -> {season} ")
            for refname in reflist:
                print(f"      -> 参考数据: {refname} ")
                dataname = refname
                metrics_df = pd.read_excel(f"{OutDir}/ModelValidation/{casename}/Perform_Mean_{casename}_{dataname}_{varname}_{season}.xlsx")
                metrics_df = metrics_df.round(2)  # 保留一位小数
                print(metrics_df)



def LakeRegClimImpact(OutDir: str, lkinfos: Dict[str, Any])-> None:
    """
    计算湖泊对区域气候的调节影响
    """
    varlist = ["T2m", "RH", "Q2m"]
    seasons = TU.get_seasons()
    checkmethod = "Paired_t-test"
    landmask = lkinfos['ocean']==0

    # 读取数据
    print("\n\n湖泊对区域气候的调节影响")
    caselist = ["Lake", "NoLake"]
    for varname in varlist:
        print(f"  -> {varname} ")
        for season in seasons:
            print(f"    -> {season} ")
            path_seasonal = f'{OutDir}/RegClimImpact/Significance_RegClimImpact_{caselist[0]}-{caselist[1]}_{varname}_seasonal_{season}_{checkmethod}.nc'
            xarr_seasonal= TIO.read_newnc(path_seasonal)
            data = xarr_seasonal["mean_diff"].values
            p_value = xarr_seasonal["p_value"].values
            sigmask = p_value < 0.05
            data[~sigmask] = np.nan
            land = data[landmask]
            land_pos = land[land>0]
            land_neg = land[land<0]
            mean_land = np.nanmean(land)
            min_land = np.nanmin(land)
            max_land = np.nanmax(land)
            med_land = np.nanmedian(land)
            q25_land = np.nanpercentile(land, 25)
            q75_land = np.nanpercentile(land, 75)
            mean_land_pos = np.nanmean(land_pos)
            min_land_pos = np.nanmin(land_pos)
            max_land_pos = np.nanmax(land_pos)
            med_land_pos = np.nanmedian(land_pos)
            q25_land_pos = np.nanpercentile(land_pos, 25)
            q75_land_pos = np.nanpercentile(land_pos, 75)
            qmean_land_pos = np.nanmean(land_pos[(land_pos>q25_land_pos)&(land_pos<q75_land_pos)])
            mean_land_neg = np.nanmean(land_neg)
            min_land_neg = np.nanmin(land_neg)
            max_land_neg = np.nanmax(land_neg)
            med_land_neg = np.nanmedian(land_neg)
            q25_land_neg = np.nanpercentile(land_neg, 25)
            q75_land_neg = np.nanpercentile(land_neg, 75)     
            qmean_land_neg = np.nanmean(land_neg[(land_neg>q25_land_neg)&(land_neg<q75_land_neg)])    
            print(f"      -> 平均: mean [{mean_land:8.2f}], min [{min_land:8.2f}], max [{max_land:8.2f}], median [{med_land:8.2f}], Q25 [{q25_land:8.2f}], Q75 [{q75_land:8.2f}], Q-mean [{qmean_land_pos:8.2f}|{qmean_land_neg:8.2f}]")
            print(f"      -> 正向: mean [{mean_land_pos:8.2f}], min [{min_land_pos:8.2f}], max [{max_land_pos:8.2f}], median [{med_land_pos:8.2f}], Q25 [{q25_land_pos:8.2f}], Q75 [{q75_land_pos:8.2f}], Q-mean [{qmean_land_pos:8.2f}]")
            print(f"      -> 负向: mean [{mean_land_neg:8.2f}], min [{min_land_neg:8.2f}], max [{max_land_neg:8.2f}], median [{med_land_neg:8.2f}], Q25 [{q25_land_neg:8.2f}], Q75 [{q75_land_neg:8.2f}], Q-mean [{qmean_land_neg:8.2f}]")







