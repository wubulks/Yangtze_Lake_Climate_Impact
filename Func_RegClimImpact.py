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
fdr_alpha = TPC.fdr_alpha

# 区域气候影响显著性检验
def RegClimImpactSignificanceOfChange_seasonal(
        xarr1: xr.Dataset, xarr2: xr.Dataset,  checkmethod: str, *,
        var: str, outdir: str, caselist = ['Lake', 'NoLake'], rspmethod = 'mean') -> None:
    """
    对两个 xarray 数组进行配对检验，返回 None。
    xarr1: 第一个情景实验数据
    xarr2: 第二个情景实验数据
    checkmethod: 检验方法
    alternative: 备择假设
    alpha_ci: 显著性水平
    clt_n: 中心极限定理样本量
    n_sample: 重抽样次数
    ci: 置信区间
    center_null: 是否中心化零假设
    random_state: 随机种子
    """
    # checkmethod = "auto"          # ["auto", "Paired_t-test", "Wilcoxon_signed-rank_test", "Paired_permutation_test","Paired_bootstrap"]
    alternative = "two-sided"     # ["two-sided", "greater", "less"] 
    alpha_ci = 0.05               # 显著性水平
    clt_n = 30                    # 中心极限定理样本量
    n_sample = 10000              # 重抽样次数
    ci = 0.95                     # 置信区间
    center_null = True            # 是否中心化零假设
    random_state = 666            # 随机种子
    n_jobs = 92
    outcasedir = f'{outdir}/RegClimImpact/'

    # 按季节尺度分析
    rspfreq = 'season'   # 'month
    seasons = TU.get_seasons()
    season_months = TU.get_season_months()
    for season in seasons:
        print(f'    ---- Processing season: {season} ----')
        months_sel = season_months[season]
        xarr1_season = xarr1.sel(time=xarr1.time.dt.season == season).squeeze(drop=True)
        xarr2_season = xarr2.sel(time=xarr2.time.dt.season == season).squeeze(drop=True)
        if rspfreq == 'month':
            xarr1_season = xarr1_season.assign_coords(month=("time", pd.PeriodIndex(xarr1_season.time.to_index(), freq="M")))
            xarr2_season = xarr2_season.assign_coords(month=("time", pd.PeriodIndex(xarr2_season.time.to_index(), freq="M")))
            if rspmethod == 'mean':
                xarr1_rsp = xarr1_season.groupby("month").mean("time")
                xarr2_rsp = xarr2_season.groupby("month").mean("time")
            elif rspmethod == 'max':
                xarr1_rsp = xarr1_season.groupby("month").max("time")
                xarr2_rsp = xarr2_season.groupby("month").max("time")
            elif rspmethod == 'min':
                xarr1_rsp = xarr1_season.groupby("month").min("time")
                xarr2_rsp = xarr2_season.groupby("month").min("time")   
            elif rspmethod == 'sum':
                xarr1_rsp = xarr1_season.groupby("month").sum("time")
                xarr2_rsp = xarr2_season.groupby("month").sum("time")
            xarr1_rsp = xarr1_rsp.assign_coords(month=("month", xarr1_rsp.indexes["month"].to_timestamp()))
            xarr2_rsp = xarr2_rsp.assign_coords(month=("month", xarr2_rsp.indexes["month"].to_timestamp()))
            timedims = xarr1_rsp.month
        elif rspfreq == 'season':
            if rspmethod == 'mean':
                xarr1_rsp = xarr1_season.groupby("time.year").mean("time")
                xarr2_rsp = xarr2_season.groupby("time.year").mean("time")
            elif rspmethod == 'max':
                xarr1_rsp = xarr1_season.groupby("time.year").max("time")
                xarr2_rsp = xarr2_season.groupby("time.year").max("time")
            elif rspmethod == 'min':
                xarr1_rsp = xarr1_season.groupby("time.year").min("time")
                xarr2_rsp = xarr2_season.groupby("time.year").min("time")   
            elif rspmethod == 'sum':
                xarr1_rsp = xarr1_season.groupby("time.year").sum("time")
                xarr2_rsp = xarr2_season.groupby("time.year").sum("time")
            timedims = xarr1_rsp.year
        arr1_check = xarr1_rsp.values.squeeze()
        arr2_check = xarr2_rsp.values.squeeze()
        p_arr, mean_diff_arr, effect_size_arr, method_arr, ci_low_arr, ci_high_arr = TST.SignificanceTest(arr1_check, arr2_check,
                                                            checkmethod=checkmethod, alternative=alternative,
                                                            alpha_ci=alpha_ci, clt_n=clt_n, n_sample=n_sample,
                                                            ci=ci, info=f'{season} ', center_null=center_null,
                                                            random_state=random_state, n_jobs=n_jobs)
        p_ravel = p_arr.ravel(); mask = np.isfinite(p_ravel)
        rej_fdr = np.full_like(p_ravel, 0, dtype=np.int8)  # 初始化拒绝标记
        p_fdr = np.full_like(p_ravel, np.nan, dtype=float)  # 初始化 FDR 校正后的 p 值
        rej_sub, p_sub, _, _ = multipletests(p_ravel[mask], alpha=fdr_alpha, method="fdr_bh")
        rej_fdr[mask] = rej_sub.astype(np.int8); p_fdr[mask] = p_sub
        rej_fdr = rej_fdr.reshape(p_arr.shape); p_fdr = p_fdr.reshape(p_arr.shape)
        savepath = f'{outcasedir}/Significance_RegClimImpact_{caselist[0]}-{caselist[1]}_{var}_seasonal_{season}_{checkmethod}.nc'
        RC_overall, RC_anomaly = TST.RelativeContribution(arr1_check, arr2_check, time_axis=0)
        in_dict = {'p_value': [["y", "x"], p_arr],
                   'mean_diff': [["y", "x"], mean_diff_arr],
                   'effect_size': [["y", "x"], effect_size_arr],
                   'checkmethod': [["y", "x"], method_arr],
                   'p_fdr': [["y", "x"], p_fdr],
                   'rejected': [["y", "x"], rej_fdr],
                   'ci_low': [["y", "x"], ci_low_arr],
                   'ci_high': [["y", "x"], ci_high_arr],
                    'RC_overall': [["y", "x"], RC_overall],
                    'RC_anomaly': [["y", "x"], RC_anomaly],
                   caselist[0]: [["time", "y", "x"], arr1_check],
                   caselist[1]: [["time", "y", "x"], arr2_check],
                   }
        coords = {"y": xarr1.y, "x": xarr1.x, "time": timedims}
        TIO.save_newnc(savepath=savepath, in_dict=in_dict, coords=coords)

    # 按“年际平均”分析，形状 (nyear, y, x)
    print(f'    ---- Processing season: Annual mean ----')
    if rspmethod == 'mean':
        xarr1_year_ = xarr1.groupby("time.year").mean("time")
        xarr2_year_ = xarr2.groupby("time.year").mean("time")
    elif rspmethod == 'max':
        xarr1_year_ = xarr1.groupby("time.year").max("time")
        xarr2_year_ = xarr2.groupby("time.year").max("time")
    elif rspmethod == 'min':
        xarr1_year_ = xarr1.groupby("time.year").min("time")
        xarr2_year_ = xarr2.groupby("time.year").min("time")   
    elif rspmethod == 'sum':
        xarr1_year_ = xarr1.groupby("time.year").sum("time")
        xarr2_year_ = xarr2.groupby("time.year").sum("time")
    arr1_year = xarr1_year_.values.squeeze()
    arr2_year = xarr2_year_.values.squeeze()
    p_arr, mean_diff_arr, effect_size_arr, method_arr, ci_low_arr, ci_high_arr = TST.SignificanceTest(arr1_year, arr2_year,
                                                        checkmethod=checkmethod, alternative=alternative,
                                                        alpha_ci=alpha_ci, clt_n=clt_n, n_sample=n_sample,
                                                        ci=ci, info=f'{season} ', center_null=center_null,
                                                        random_state=random_state, n_jobs=n_jobs)
    p_ravel = p_arr.ravel(); mask = np.isfinite(p_ravel)
    rej_fdr = np.full_like(p_ravel, 0, dtype=np.int8)  # 初始化拒绝标记
    p_fdr = np.full_like(p_ravel, np.nan, dtype=float)  # 初始化 FDR 校正后的 p 值
    rej_sub, p_sub, _, _ = multipletests(p_ravel[mask], alpha=fdr_alpha, method="fdr_bh")
    rej_fdr[mask] = rej_sub.astype(np.int8); p_fdr[mask] = p_sub
    rej_fdr = rej_fdr.reshape(p_arr.shape); p_fdr = p_fdr.reshape(p_arr.shape)
    savepath = f'{outcasedir}/Significance_RegClimImpact_{caselist[0]}-{caselist[1]}_{var}_yearly_{checkmethod}.nc'
    RC_overall, RC_anomaly = TST.RelativeContribution(arr1_year, arr2_year, time_axis=0)
    in_dict = {'p_value': [["y", "x"], p_arr],
                'mean_diff': [["y", "x"], mean_diff_arr],
                'effect_size': [["y", "x"], effect_size_arr],
                'checkmethod': [["y", "x"], method_arr],
                'p_fdr': [["y", "x"], p_fdr],
                'rejected': [["y", "x"], rej_fdr],
                'ci_low': [["y", "x"], ci_low_arr],
                'ci_high': [["y", "x"], ci_high_arr],
                'RC_overall': [["y", "x"], RC_overall],
                'RC_anomaly': [["y", "x"], RC_anomaly],
                caselist[0]: [["time", "y", "x"], arr1_year],
                caselist[1]: [["time", "y", "x"], arr2_year],
                }
    coords = {"y": xarr1.y, "x": xarr1.x, "time": xarr1_year_.year}
    TIO.save_newnc(savepath=savepath, in_dict=in_dict, coords=coords)



# 区域气候影响显著性检验
def RegClimImpactSignificanceOfChange_diurnal(
        xarr1: xr.Dataset, xarr2: xr.Dataset,  checkmethod: str, *,
        var: str, outdir: str, caselist = ['Lake', 'NoLake'], rspmethod = 'mean') -> None:
    """
    对两个 xarray 数组进行配对检验，返回 TestResult。
    xarr1: 第一个情景实验数据
    xarr2: 第二个情景实验数据
    checkmethod: 检验方法
    alternative: 备择假设
    alpha_ci: 显著性水平
    clt_n: 中心极限定理样本量
    n_sample: 重抽样次数
    ci: 置信区间
    center_null: 是否中心化零假设
    random_state: 随机种子
    """
    # checkmethod = "auto"               # ["auto", "Paired_t-test", "Wilcoxon_signed-rank_test", "Paired_permutation_test","Paired_bootstrap"]
    alternative = "two-sided"     # ["two-sided", "greater", "less"] 
    alpha_ci = 0.05               # 显著性水平
    clt_n = 30                    # 中心极限定理样本量
    n_sample = 10000              # 重抽样次数
    ci = 0.95                     # 置信区间
    center_null = True            # 是否中心化零假设
    checkflag = False
    random_state = 666            # 随机种子
    n_jobs = 92
    outcasedir = f'{outdir}/RegClimImpact/'

    # 按季节尺度分析
    # 输入数据为：(time, hour, y, x)
    # rspmethod = 'mean'
    rspfreq = 'season'   # 'month
    seasons = TU.get_seasons()
    season_months = TU.get_season_months()
    for season in seasons:
        print(f'    ---- Processing season: {season} ----')
        months = season_months[season]
        xarr1_season = xarr1.sel(time=xarr1.time.dt.season == season).squeeze(drop=True)
        xarr2_season = xarr2.sel(time=xarr2.time.dt.season == season).squeeze(drop=True)
        xarr1_season = xarr1_season.squeeze()
        xarr2_season = xarr2_season.squeeze()

        if rspfreq == 'month':
            xarr1_season = xarr1_season.stack(time=("year", "month"))
            xarr2_season = xarr2_season.stack(time=("year", "month"))
            mi = xarr1_season.indexes["time"]            # MultiIndex
            years  = mi.get_level_values("year").values
            months = mi.get_level_values("month").values
            time_index = pd.to_datetime({
                "year": years,
                "month": months,
                "day": 1
            })
            xarr1_season = xarr1_season.drop_vars(['time', 'year', 'month'])
            xarr2_season = xarr2_season.drop_vars(['time', 'year', 'month'])
            xarr1_rsp = xarr1_season.assign_coords(time=("time", time_index))
            xarr2_rsp = xarr2_season.assign_coords(time=("time", time_index))
            xarr1_rsp = xarr1_rsp.transpose("time", "hour", "y", "x")
            xarr2_rsp = xarr2_rsp.transpose("time", "hour", "y", "x")
            timedims = xarr1_rsp.time
        elif rspfreq == 'season':
            # 形状 (year, hour, y, x)
            if rspmethod == 'mean':
                # 按年计算季节平均
                xarr1_rsp = xarr1_season.groupby("time.year").mean("time")
                xarr2_rsp = xarr2_season.groupby("time.year").mean("time")
            elif rspmethod == 'max':
                xarr1_rsp = xarr1_season.groupby("time.year").max("time")
                xarr2_rsp = xarr2_season.groupby("time.year").max("time")
            elif rspmethod == 'min':
                xarr1_rsp = xarr1_season.groupby("time.year").min("time")
                xarr2_rsp = xarr2_season.groupby("time.year").min("time")
            elif rspmethod == 'sum':
                xarr1_rsp = xarr1_season.groupby("time.year").sum("time")
                xarr2_rsp = xarr2_season.groupby("time.year").sum("time")
            timedims = xarr1_rsp.year
        arr1_season = xarr1_rsp.values.squeeze()
        arr2_season = xarr2_rsp.values.squeeze()
        if rspfreq == 'month':
            nyear, nhour, ny, nx = arr1_season.shape
        elif rspfreq == 'season':
            nyear, nhour, ny, nx = arr1_season.shape
        p_arr = np.full((nhour, ny, nx), np.nan)
        mean_diff_arr = np.full((nhour, ny, nx), np.nan)
        effect_size_arr = np.full((nhour, ny, nx), np.nan)
        method_arr = np.full((nhour, ny, nx), np.nan)
        p_fdr_ = np.full((nhour, ny, nx), np.nan)
        rej_fdr_ = np.full((nhour, ny, nx), 0, dtype=np.int8)  # 初始化拒绝标记
        ci_low_arr = np.full((nhour, ny, nx), np.nan)
        ci_high_arr = np.full((nhour, ny, nx), np.nan)
        RC_overall = np.full((nhour, ny, nx), np.nan)
        RC_anomaly = np.full((nhour, ny, nx), np.nan)
        for hour in range(nhour):
            p_out, mean_diff_out, effect_size_out, method_out, ci_low_out, ci_high_out = TST.SignificanceTest(arr1_season[:, hour, :, :], arr2_season[:, hour, :, :],
                                                                checkmethod=checkmethod, alternative=alternative,
                                                                alpha_ci=alpha_ci, clt_n=clt_n, n_sample=n_sample,
                                                                ci=ci, info=f'{season} {hour:02d}H', center_null=center_null,
                                                                random_state=random_state, n_jobs=n_jobs,checkflag=checkflag)
            p_arr[hour, :, :] = p_out
            mean_diff_arr[hour, :, :] = mean_diff_out
            effect_size_arr[hour, :, :] = effect_size_out
            method_arr[hour, :, :] = method_out
            ci_low_arr[hour, :, :] = ci_low_out
            ci_high_arr[hour, :, :] = ci_high_out
            p_ravel = p_arr[hour, :, :].ravel(); mask = np.isfinite(p_ravel)
            rej_fdr = np.full_like(p_ravel, 0, dtype=np.int8)  # 初始化拒绝标记
            p_fdr = np.full_like(p_ravel, np.nan, dtype=float)  # 初始化 FDR 校正后的 p 值
            rej_sub, p_sub, _, _ = multipletests(p_ravel[mask], alpha=fdr_alpha, method="fdr_bh")
            rej_fdr[mask] = rej_sub.astype(np.int8); p_fdr[mask] = p_sub
            rej_fdr = rej_fdr.reshape((ny, nx)); p_fdr = p_fdr.reshape((ny, nx))
            rej_fdr_[hour, :, :] = rej_fdr
            p_fdr_[hour, :, :] = p_fdr
            RC_overall[hour, :, :], RC_anomaly[hour, :, :] = TST.RelativeContribution(arr1_season[:, hour, :, :], arr2_season[:, hour, :, :], time_axis=0)
        savepath = f'{outcasedir}/Significance_RegClimImpact_{caselist[0]}-{caselist[1]}_{var}_seasonaldiurnalmean_{season}_{checkmethod}.nc'
        in_dict = {'p_value': [["hour", "y", "x"], p_arr],
                   'mean_diff': [["hour", "y", "x"], mean_diff_arr],
                   'effect_size': [["hour", "y", "x"], effect_size_arr],
                   'checkmethod': [["hour", "y", "x"], method_arr],
                   'p_fdr': [["hour", "y", "x"], p_fdr_],
                   'rejected': [["hour", "y", "x"], rej_fdr_],
                   'ci_low': [["hour", "y", "x"], ci_low_arr],
                   'ci_high': [["hour", "y", "x"], ci_high_arr],
                   'RC_overall': [["hour", "y", "x"], RC_overall],
                   'RC_anomaly': [["hour", "y", "x"], RC_anomaly],
                   caselist[0]: [["year", "hour", "y", "x"], arr1_season],
                   caselist[1]: [["year", "hour", "y", "x"], arr2_season],
                   }
        coords = {"y": xarr1.y, "x": xarr1.x, "year": timedims, "hour": xarr2_rsp.hour}
        TIO.save_newnc(savepath=savepath, in_dict=in_dict, coords=coords)

    # 按“年际平均”分析，形状 (nhour, ny, nx)
    print(f'    ---- Processing year-hourly mean ----')
    if rspmethod == 'mean':
        xarr1_year_ = xarr1.groupby("time.year").mean("time")
        xarr2_year_ = xarr2.groupby("time.year").mean("time")
    elif rspmethod == 'max':
        xarr1_year_ = xarr1.groupby("time.year").max("time")
        xarr2_year_ = xarr2.groupby("time.year").max("time")
    elif rspmethod == 'min':
        xarr1_year_ = xarr1.groupby("time.year").min("time")
        xarr2_year_ = xarr2.groupby("time.year").min("time")
    elif rspmethod == 'sum':
        xarr1_year_ = xarr1.groupby("time.year").sum("time")
        xarr2_year_ = xarr2.groupby("time.year").sum("time")
    arr1_year = xarr1_year_.values.squeeze()
    arr2_year = xarr2_year_.values.squeeze()
    nyear, nhour, ny, nx = arr1_year.shape
    p_arr = np.full((nhour, ny, nx), np.nan)
    mean_diff_arr = np.full((nhour, ny, nx), np.nan)
    effect_size_arr = np.full((nhour, ny, nx), np.nan)
    method_arr = np.full((nhour, ny, nx), np.nan)
    p_fdr_ = np.full((nhour, ny, nx), np.nan)
    rej_fdr_ = np.full((nhour, ny, nx), 0, dtype=np.int8)  # 初始化拒绝标记
    ci_low_arr = np.full((nhour, ny, nx), np.nan)
    ci_high_arr = np.full((nhour, ny, nx), np.nan)
    RC_overall = np.full((nhour, ny, nx), np.nan)
    RC_anomaly = np.full((nhour, ny, nx), np.nan)
    for hour in range(nhour):
        p_out, mean_diff_out, effect_size_out, method_out, ci_low_out, ci_high_out = TST.SignificanceTest(arr1_year[:, hour, :, :], arr2_year[:, hour, :, :],
                                                            checkmethod=checkmethod, alternative=alternative,
                                                            alpha_ci=alpha_ci, clt_n=clt_n, n_sample=n_sample,
                                                            ci=ci, info=f'{season} ', center_null=center_null,
                                                            random_state=random_state, n_jobs=n_jobs)
        p_arr[hour, :, :] = p_out
        mean_diff_arr[hour, :, :] = mean_diff_out
        effect_size_arr[hour, :, :] = effect_size_out
        method_arr[hour, :, :] = method_out
        ci_low_arr[hour, :, :] = ci_low_out
        ci_high_arr[hour, :, :] = ci_high_out
        p_ravel = p_arr[hour, :, :].ravel(); mask = np.isfinite(p_ravel)
        rej_fdr = np.full_like(p_ravel, 0, dtype=np.int8)  # 初始化拒绝标记
        p_fdr = np.full_like(p_ravel, np.nan, dtype=float)  # 初始化 FDR 校正后的 p 值
        rej_sub, p_sub, _, _ = multipletests(p_ravel[mask], alpha=fdr_alpha, method="fdr_bh")
        rej_fdr[mask] = rej_sub.astype(np.int8); p_fdr[mask] = p_sub
        rej_fdr = rej_fdr.reshape((ny, nx)); p_fdr = p_fdr.reshape((ny, nx))
        rej_fdr_[hour, :, :] = rej_fdr
        p_fdr_[hour, :, :] = p_fdr
        RC_overall[hour, :, :], RC_anomaly[hour, :, :] = TST.RelativeContribution(arr1_year[:, hour, :, :], arr2_year[:, hour, :, :], time_axis=0)

    in_dict = {'p_value': [["hour", "y", "x"], p_arr],
                'mean_diff': [["hour", "y", "x"], mean_diff_arr],
                'effect_size': [["hour", "y", "x"], effect_size_arr],
                'checkmethod': [["hour", "y", "x"], method_arr],
                'p_fdr': [["hour", "y", "x"], p_fdr_],
                'rejected': [["hour", "y", "x"], rej_fdr_],
                'ci_low': [["hour", "y", "x"], ci_low_arr],
                'ci_high': [["hour", "y", "x"], ci_high_arr],
                'RC_overall': [["hour", "y", "x"], RC_overall],
                'RC_anomaly': [["hour", "y", "x"], RC_anomaly],
                caselist[0]: [["year", "hour", "y", "x"], arr1_year],
                caselist[1]: [["year", "hour", "y", "x"], arr2_year],
                }
    coords = {"y": xarr1.y, "x": xarr1.x, "time": xarr2_year_.year, "hour": xarr2_year_.hour}
    savepath = f'{outcasedir}/Significance_RegClimImpact_{caselist[0]}-{caselist[1]}_{var}_yearlydiurnalmean_{checkmethod}.nc'
    TIO.save_newnc(savepath=savepath, in_dict=in_dict, coords=coords)




# 区域气候影响绘图
def Plot_RegClimImpact(
        varname: str, caselist: List[str],
        lon2d: np.ndarray, lat2d: np.ndarray,
        checkmethod: str, OutDir:str, FigOutDir: str,
        lkinfos: Any, onlysig: bool = True) -> None:
    """
    区域气候影响绘图
    """
    target = "RegClimImpact"
    DPI = TPC.DPI_medium
    FIGFMT = TPC.FIGFMT
    seasons = TU.get_seasons()
    keep_hours = TU.get_all_hours()
    var_info = TYCM.Variable_Infos(varname)
    seas_levels = TYCM.Seasonal_RegClimImpact_Cmap(varname)
    diurnal_levels = TYCM.Diurnal_RegClimImpact_Cmap(varname)
    FigOutDir_var = f'{FigOutDir}/{varname}/Single'
    os.makedirs(FigOutDir_var, exist_ok=True)
    # 绘制区域差异图（含显著性标记）
    tasks = []
    season_diffcfg = TPC.mapConfig(levs=seas_levels['diff_maplevs_seasonal'], cmap=seas_levels['diff_cmap'])
    season_rccfg = TPC.mapConfig(levs=seas_levels['rc_maplevs_seasonal'], cmap=seas_levels['rc_cmap'])  
    annual_diffcfg = TPC.mapConfig(levs=seas_levels['diff_maplevs_annual'], cmap=seas_levels['diff_cmap'])
    annual_rccfg = TPC.mapConfig(levs=seas_levels['rc_maplevs_annual'], cmap=seas_levels['rc_cmap'])  
    boxcfg = TPC.boxConfig(diff_boxlevs=seas_levels['diff_boxlevs'], rc_boxlevs=seas_levels['rc_boxlevs'])
    rosecfg = TPC.roseConfig(roselevs=diurnal_levels['roselevels'], colors_dict={'Strong':('grey', 'o')})
    varInfo = TPC.varInfo(longname=var_info['longname'], unit=var_info['unit'], abbr=var_info['abbr'])
    suffix = f"onlysig" if onlysig else "all"
    xarr_seasons = {}
    for season in seasons:
        path_seasonal = f'{OutDir}/RegClimImpact/Significance_RegClimImpact_{caselist[0]}-{caselist[1]}_{varname}_seasonal_{season}_{checkmethod}.nc'
        xarr_seasonal= TIO.read_newnc(path_seasonal)
        xarr_seasons[season] = xarr_seasonal
        # 绘制季节差异图（含显著性标记) 
        savepath = f'{FigOutDir_var}/{target}_Diff_Map_{varname}_{season}_{suffix}_{checkmethod}.{FIGFMT}'
        # tasks.append((TPAM.plot_spatial_diffmap_withsign, (xarr_seasonal, varname, season, target, lon2d, lat2d, 
        #                                                    checkmethod, lkinfos, season_diffcfg, onlysig, savepath)))
        # 绘制区域相对贡献图（含显著性标记）
        # savepath = f'{FigOutDir_var}/{target}_RC_Map_{varname}_{season}_{suffix}_{checkmethod}.{FIGFMT}'
        # tasks.append((TPAM.plot_spatial_rcmap_withsign, (xarr_seasonal, varname, season, target, lon2d, lat2d, 
        #                                                  checkmethod, lkinfos, season_rccfg, onlysig, savepath)))
        # 读取季节昼夜数据
        ds_dict = TIO.read_hourly_significance(caselist, varname, season, keep_hours, checkmethod, OutDir)
        in_df, sig_mask, strong_mask, weak_mask, suffix = TDP.prepara_for_diurnal_rose(ds_dict, lkinfos, varname, checkmethod, onlysig)
        tasks.append((TPR.plot_regclimimpact_diurnal_rose, (in_df, varname, season, target, checkmethod, FigOutDir_var, rosecfg, varInfo, suffix)))

    # 绘制季节差异箱线图和相对贡献箱线图
    diff_df, rc_df, suffix = TDP.prepara_for_boxplot_seasonal(xarr_seasons, lkinfos, checkmethod, onlysig=True)
    tasks.append((TPB.plot_diff_boxplot, (diff_df, seasons, varname, target, checkmethod, FigOutDir_var, boxcfg, varInfo, suffix, '.1f')))
    # tasks.append((TPB.plot_rc_boxplot, (rc_df, seasons, varname, target, checkmethod, FigOutDir_var, boxcfg, varInfo, suffix)))

    path_annual = f'{OutDir}/RegClimImpact/Significance_RegClimImpact_{caselist[0]}-{caselist[1]}_{varname}_yearly_{checkmethod}.nc'
    xarr_annual = TIO.read_newnc(path_annual)
    # 绘制年度差异图（含显著性标记）
    # savepath = f'{FigOutDir_var}/{target}_Diff_Map_{varname}_Annual_{suffix}_{checkmethod}.{FIGFMT}'
    # tasks.append((TPAM.plot_spatial_diffmap_withsign, (xarr_annual, varname, "Annual", target, lon2d, lat2d, 
    #                                                    checkmethod, lkinfos, annual_diffcfg, onlysig, savepath)))
    # 绘制区域相对贡献图（含显著性标记）
    # savepath = f'{FigOutDir_var}/{target}_RC_Map_{varname}_Annual_{suffix}_{checkmethod}.{FIGFMT}'
    # tasks.append((TPAM.plot_spatial_rcmap_withsign, (xarr_annual, varname, "Annual", target, lon2d, lat2d, 
    #                                                  checkmethod, lkinfos, annual_rccfg, onlysig, savepath)))
    # 绘制colorbar
    label = f"{varInfo.longname} ({varInfo.unit})"
    savepath = f'{FigOutDir_var}/{target}_Diff_Map_{varname}_Annual_Colorbar_{suffix}_{checkmethod}.{FIGFMT}'
    tasks.append((TPCB.plot_spatial_cbar_core_V, (annual_diffcfg, 4, label, savepath, 18, 18, 0.06, 'both','{:5.1f}'))) 
    savepath = f'{FigOutDir_var}/{target}_Diff_Map_{varname}_Seasonal_Colorbar_{suffix}_{checkmethod}.{FIGFMT}'
    tasks.append((TPCB.plot_spatial_cbar_core_V, (season_diffcfg, 4, label, savepath, 18, 18, 0.06, 'both','{:5.1f}'))) 
    savepath = f'{FigOutDir_var}/{target}_RC_Map_{varname}_Annual_Colorbar_{suffix}_{checkmethod}.{FIGFMT}'
    tasks.append((TPCB.plot_spatial_cbar_core_V, (annual_rccfg, 4, "Relative Contribution (%)", savepath, 18, 18, 0.06, 'both','{:5.1f}'))) 
    savepath = f'{FigOutDir_var}/{target}_RC_Map_{varname}_Seasonal_Colorbar_{suffix}_{checkmethod}.{FIGFMT}'
    tasks.append((TPCB.plot_spatial_cbar_core_V, (season_rccfg, 4, "Relative Contribution (%)", savepath, 18, 18, 0.06, 'both','{:5.1f}'))) 
    
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


    

def Merge_RegClimImpact(varname: str, caselist: List[str], checkmethod: str, FigOutDir: str, onlysig: bool = True):
    """
    合并区域气候影响绘图
    """
    seasons = TU.get_seasons()
    mapcrop_params = TYCM.MapPlot_CropParams_noColorbar()
    figbox_space = TYCM.Merge_Fig_Space_Params()
    cbar_space = TYCM.MapCbar_CropParams_V()
    rosecrop_params = TYCM.RosePlot_CropParams_noColorbar()
    suffix = "onlysig" if onlysig else "all"
    # {'left': 0.01, 'top': 0.01, 'right': 0.01, 'bottom': 0.01}

    #############################
    # Mean maps
    #############################
    target = "RegClimImpact"
    cols_space = [[0, 0, 0, 0], [0, 0, 0, 0]]  #[0, 0, 0, 0], 
    rows_space = [0.01, 0.01]  #, 0.01
    rows = [[], []]
    # 季节diff map & rc map
    rowpos = 0
    for season in seasons:
        diffmappath = f'{FigOutDir}/{varname}/Single/{target}_Diff_Map_{varname}_{season}_{suffix}_{checkmethod}.{FIGFMT}'
        cropped_diff = TIT.crop_image_from_path(diffmappath, crop_params=mapcrop_params, mode="ratio")
        rows[rowpos].append(cropped_diff)
    diffcbarpath = f'{FigOutDir}/{varname}/Single/{target}_Diff_Map_{varname}_Seasonal_Colorbar_{suffix}_{checkmethod}.{FIGFMT}'
    cropped_diff_cbar = TIT.crop_image_from_path(diffcbarpath, crop_params=cbar_space, mode="ratio")
    rows[rowpos].append(cropped_diff_cbar)
    # rowpos = 1
    # for season in seasons:
    #     diffmappath = f'{FigOutDir}/{varname}/Single/{target}_RC_Map_{varname}_{season}_{suffix}_{checkmethod}.{FIGFMT}'
    #     cropped_diff = TIT.crop_image_from_path(diffmappath, crop_params=mapcrop_params, mode="ratio")
    #     rows[rowpos].append(cropped_diff)
    # diffcbarpath = f'{FigOutDir}/{varname}/Single/{target}_RC_Map_{varname}_Seasonal_Colorbar_{suffix}_{checkmethod}.{FIGFMT}'
    # cropped_diff_cbar = TIT.crop_image_from_path(diffcbarpath, crop_params=cbar_space, mode="ratio")
    # rows[rowpos].append(cropped_diff_cbar)
    rowpos = 1
    for season in seasons:
        rosepath_diff = f'{FigOutDir}/{varname}/Single/{target}_Diff_Rose_{varname}_{season}_{suffix}_{checkmethod}.{FIGFMT}'
        cropped_rose = TIT.crop_image_from_path(rosepath_diff, crop_params=rosecrop_params, mode="ratio")
        cropped_rose = TIT.adjust_image_to_ref_canvas(target_img=cropped_rose, ref_img=cropped_diff)
        rows[rowpos].append(cropped_rose)

    series_texts = {
        # 气温diff
        '(a)': {'x': 0.01,  'y': 0.02,  "fontsize": 0.03, "text": "(a)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        '(b)': {'x': 0.245, 'y': 0.02,  "fontsize": 0.03, "text": "(b)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        '(c)': {'x': 0.48, 'y': 0.02,  "fontsize": 0.03, "text": "(c)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        '(d)': {'x': 0.716, 'y': 0.02,  "fontsize": 0.03, "text": "(d)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        # 气温rose
        '(e)': {'x': 0.01,  'y': 0.42,  "fontsize": 0.03, "text": "(e)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        '(f)': {'x': 0.245, 'y': 0.42,  "fontsize": 0.03, "text": "(f)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        '(g)': {'x': 0.48, 'y': 0.42,  "fontsize": 0.03, "text": "(g)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        '(h)': {'x': 0.716, 'y': 0.42,  "fontsize": 0.03, "text": "(h)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
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
    savepath = f'{FigOutDir}/{target}_{varname}_Seasonal_Comparison.{FIGFMT}'
    TIT.save(seas_img, savepath, dpi=DPI)




def Depth_Dependece_Impacts(
        varname: str, caselist: List[str],
        lon2d: np.ndarray, lat2d: np.ndarray,
        checkmethod: str, OutDir:str, FigOutDir: str,
        lkinfos: Any, onlysig: bool = True) -> None:
    """
    区域气候影响绘图
    """
    target = "RegClimImpact"
    DPI = TPC.DPI_medium
    FIGFMT = TPC.FIGFMT
    seasons = TU.get_seasons()
    keep_hours = TU.get_all_hours()
    var_info = TYCM.Variable_Infos(varname)
    seas_levels = TYCM.Seasonal_RegClimImpact_Cmap(varname)
    diurnal_levels = TYCM.Diurnal_RegClimImpact_Cmap(varname)
    FigOutDir_var = f'{FigOutDir}/{varname}/Single'
    os.makedirs(FigOutDir_var, exist_ok=True)

    shallowlake_mask = lkinfos['shallow'] == 1
    deeplake_mask = lkinfos['deep'] == 1
    gt10_mask = lkinfos['lakedp'] > 20
    le10_mask = lkinfos['lakedp'] <= 20
    # dist_steps_deep = (lkinfos['dist_steps_deep'] == 2) & (lkinfos['dist_steps_deep'] > 0)
    # dist_steps_shallow = (lkinfos['dist_steps_shallow'] == 2) & (lkinfos['dist_steps_shallow'] > 0)
    dist_steps_deep = (lkinfos['dist_steps_deep'] == 0) 
    dist_steps_shallow = (lkinfos['dist_steps_shallow'] == 0) 
    xarr_seasons = {}
    for season in seasons:
        xarr_seasons[season] = {}
        path_seasonal = f'{OutDir}/RegClimImpact/Significance_RegClimImpact_{caselist[0]}-{caselist[1]}_{varname}_seasonal_{season}_{checkmethod}.nc'
        xarr_seasonal= TIO.read_newnc(path_seasonal)
        mean_diff_arr = xarr_seasonal['mean_diff'].values
        rejected = xarr_seasonal['p_value'].values
        rejected = np.where(rejected <= 0.05, 1, 0)

        # 浅水湖
        mask = shallowlake_mask & (rejected == 1)
        # mask = dist_steps_shallow & (rejected == 1)
        shallow_lakes = mean_diff_arr[mask]
        xarr_seasons[season]['shallow'] = shallow_lakes

        # 深水湖
        mask = deeplake_mask & (rejected == 1)
        # mask = dist_steps_deep & (rejected == 1)
        deep_lakes = mean_diff_arr[mask]
        xarr_seasons[season]['deep'] = deep_lakes   

    # --- 数据整理：转换为 DataFrame 格式更方便绘图 ---
    plot_data = []
    for season in seasons:
        for depth_type in ['shallow', 'deep']:
            values = xarr_seasons[season][depth_type]
            # 过滤掉 NaN 
            values = values[~np.isnan(values)]
            for v in values:
                plot_data.append({
                    'Season': season,
                    'Depth': 'Shallow Lake' if depth_type == 'shallow' else 'Deep Lake',
                    'Diff': v
                })
    df = pd.DataFrame(plot_data)

    # --- 开始绘图 ---
    # --- 绘图部分 ---
    # 设置 Seaborn 风格
    sns.set_context("paper", font_scale=1.2)
    sns.set_style("ticks")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 使用 seaborn 的 boxplot，自动处理分组和颜色
    palette = {"Shallow Lake": "#A5BFE4", "Deep Lake": "#1B355A"}
        
    sns.boxplot(
            data=df, 
            x='Season', 
            y='Diff', 
            hue='Depth',
            palette=palette,
            width=0.6,
            linewidth=1.2,
            showfliers=False,  # 不显示异常值，使图表更整洁
            showmeans=True,    # 核心需求：显示均值
            meanprops={
                "marker": "o",
                "markerfacecolor": "black",
                "markeredgecolor": "black",
                "markersize": "5"
            },
            ax=ax
        )
    # 1. 添加 y=0 参考线
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5, zorder=1)

    # 2. 轴标签与标题 (支持 LaTeX 渲染)
    ax.set_xlabel('Season', fontweight='bold')
    ax.set_ylabel(f'Difference in {var_info["longname"]} ({var_info["unit"]})', fontweight='bold')
    # ax.set_title(f'Depth Dependence of {var_info["longname"]} Regional Climate Impact', fontsize=14, pad=15)

    # 3. 美化图例
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, title='Lake Type', frameon=False, loc='best')

    # 4. 移除顶部和右侧边框 (Despine)
    sns.despine(offset=10, trim=True)

    # 5. 网格线 (可选，仅保留 y 轴)
    ax.yaxis.grid(True, linestyle=':', alpha=0.6)

    # 保存图片
    plt.tight_layout()
    savepath = f'{FigOutDir_var}/{target}_DepthDependence_{varname}_{checkmethod}.{FIGFMT}'
    plt.savefig(savepath, dpi=DPI, bbox_inches='tight')
    print(f"Figure saved to: {savepath}")
    plt.close()



def probability_density_distribution(
        varname: str, caselist: List[str],
        xarr1: xr.DataArray, xarr2: xr.DataArray,
        checkmethod: str, OutDir:str, FigOutDir: str,
        lkinfos: Any, onlysig: bool = True) -> None:
    """
    区域气候影响概率密度分布图
    """
    target = "RegClimImpact_PDF"
    DPI = TPC.DPI_medium
    FIGFMT = TPC.FIGFMT

    seasons = TU.get_seasons()
    var_info = TYCM.Variable_Infos(varname)
    varInfo = TPC.varInfo(longname=var_info['longname'], unit=var_info['unit'], abbr=var_info['abbr'])
    seas_levels = TYCM.Seasonal_RegClimImpact_Cmap(varname)
    FigOutDir_var = f'{FigOutDir}/{varname}/Single'
    os.makedirs(FigOutDir_var, exist_ok=True)

    # 1) 先对齐时间（很重要，避免两组 time 不一致）
    xarr1, xarr2 = xr.align(xarr1, xarr2, join="inner")

    # 2) 找出“坏时刻”：任意一个格点出现 NaN/Inf 就认为该时刻坏
    bad_t1 = (~np.isfinite(xarr1)).any(dim=("y", "x"))
    bad_t2 = (~np.isfinite(xarr2)).any(dim=("y", "x"))
    bad_time = bad_t1 | bad_t2

    # 3) 同步剔除坏时刻
    xarr1 = xarr1.where(~bad_time, drop=True)
    xarr2 = xarr2.where(~bad_time, drop=True)

    # --- 计算显著性 ---
    Lake_all_list = []
    NoLake_all_list = []

    # for season in seasons:
    #     # 读取显著性 mask (Y, X)
    #     path_seasonal = f'{OutDir}/RegClimImpact/Significance_RegClimImpact_{caselist[0]}-{caselist[1]}_{varname}_seasonal_{season}_{checkmethod}.nc'
    #     xarr_seasonal = TIO.read_newnc(path_seasonal)
    #     # 0.05 显著性水平
    #     sig_mask = xarr_seasonal['p_value'].values <= 0.05  

    #     # 选取季节数据 (Time, Y, X)
    #     # 注意：使用 .values 转换为 numpy 数组进行操作
    #     arr1_season = xarr1.sel(time=xarr1.time.dt.season == season).values
    #     arr2_season = xarr2.sel(time=xarr2.time.dt.season == season).values

    #     # --- 核心修复：广播 Mask ---
    #     # sig_mask 是 (ny, nx)，通过 [None, :, :] 扩展为 (1, ny, nx)
    #     # 从而匹配 (nt, ny, nx) 的数据
    #     mask_3d = sig_mask[np.newaxis, :, :] 
        
    #     # 提取显著格点（结果会自动扁平化为一维数组）
    #     arr1_sig = arr1_season[np.broadcast_to(mask_3d, arr1_season.shape)]
    #     arr2_sig = arr2_season[np.broadcast_to(mask_3d, arr2_season.shape)]

    #     # 剔除可能的 NaN 值并存入列表
    #     Lake_all_list.append(arr1_sig[~np.isnan(arr1_sig)])
    #     NoLake_all_list.append(arr2_sig[~np.isnan(arr2_sig)])

    path_seasonal = f'{OutDir}/RegClimImpact/Significance_RegClimImpact_{caselist[0]}-{caselist[1]}_{varname}_yearly_{checkmethod}.nc'
    xarr_seasonal = TIO.read_newnc(path_seasonal)
    sig_mask = (xarr_seasonal['p_value'] <= 0.05)   # DataArray (y,x)
    # --- 核心修复：广播 Mask ---
    # sig_mask 是 (ny, nx)，通过 [None, :, :] 扩展为 (1, ny, nx)
    # 从而匹配 (nt, ny, nx) 的数据
    sig_mask = sig_mask.broadcast_like(xarr1.isel(time=0, drop=True))
    # 用 xarray 直接 mask，更干净（得到 (time,y,x)，显著处保留，否则 NaN）
    x1_sig = xarr1.where(sig_mask)
    x2_sig = xarr2.where(sig_mask)

    # 转 numpy 并扁平化
    arr1_sig = x1_sig.values.ravel()
    arr2_sig = x2_sig.values.ravel()

    # 这里可以继续做“同步有效值过滤”（这时长度一致了）
    valid = np.isfinite(arr1_sig) & np.isfinite(arr2_sig)
    Lake_all_list.append(arr1_sig[valid])
    NoLake_all_list.append(arr2_sig[valid])

    # 合并所有季节的显著点
    all_lake_data = np.concatenate(Lake_all_list)
    all_nolake_data = np.concatenate(NoLake_all_list)
    total_points = len(all_lake_data)

    max_samples = 50000

    # 0) 同步去掉 NaN/Inf（可选但强烈建议，避免 kdeplot 报错/偏差）
    valid = np.isfinite(all_lake_data) & np.isfinite(all_nolake_data)
    all_lake_data = all_lake_data[valid]
    all_nolake_data = all_nolake_data[valid]

    total_points = len(all_lake_data)

    # 1) 先按 all_lake_data 排序，并同步排列 NoLake
    sort_idx = np.argsort(all_lake_data)
    all_lake_data = all_lake_data[sort_idx]
    all_nolake_data = all_nolake_data[sort_idx]

    # 2) 排序后做均匀随机抽样（无放回，同步索引）
    if total_points > max_samples:
        print(f"数据点过多 ({total_points})，正在排序后均匀抽样至 {max_samples}...")

        np.random.seed(42)
        sample_indices = np.random.choice(total_points, size=max_samples, replace=False)
        sample_indices.sort()  # 保持抽样后仍按 all_lake_data 递增

        all_lake_data = all_lake_data[sample_indices]
        all_nolake_data = all_nolake_data[sample_indices]

    # --- 绘图配置 ---
    plt.figure(figsize=(10, 6), dpi=DPI)
    colors = ['#2c7bb6', '#d7191c']  # 蓝色代表有湖泊，红色代表无湖泊（或实验对比）
    labels = caselist

    # 检查数据是否为空（防止显著性格点为0时报错）
    if len(all_lake_data) > 0 and len(all_nolake_data) > 0:
        # 绘制 KDE 曲线
        sns.kdeplot(all_lake_data, color=colors[0], label=labels[0], bw_adjust=1.3,
                    fill=False, alpha=0.3, linewidth=2.5)
        sns.kdeplot(all_nolake_data, color=colors[1], label=labels[1], bw_adjust=1.3,
                    fill=False, alpha=0.3, linewidth=2.5)

        # 完善图表细节
        plt.title(f'Probability Density Function: {varname}\n(Significant Points Only)', 
                  fontsize=15, fontweight='bold', pad=15)
        plt.xlabel(f'{varInfo.longname} ({varInfo.unit})', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.legend(frameon=True, fontsize=11)
        plt.grid(axis='y', linestyle='--', alpha=0.4)

        # 保存图片
        fig_name = f'{FigOutDir_var}/{target}_{varname}_AllSeasonsMerged_{checkmethod}.{FIGFMT}'
        plt.savefig(fig_name, bbox_inches='tight')
        print(f'Successfully saved: {fig_name}')
    else:
        print("Warning: No significant grid points found. Plot skipped.")

    plt.close()
    








