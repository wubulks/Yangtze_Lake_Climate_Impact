import os
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import metpy.calc as mpcalc
from metpy.constants import Rd
from metpy.units import units
import matplotlib as mpl
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from statsmodels.stats.multitest import multipletests
from typing import Any, List, Union, Dict
from tqdm import tqdm

# 自定义工具箱
import ToolBoxes.Utils as TU
import ToolBoxes.Tool_SignificanceTest as TST
import ToolBoxes.Tool_YangtzeColorMap as TYCM
import ToolBoxes.Tool_PlotAreaMap as TPAM
import ToolBoxes.Tool_PlotColorBar as TPCB
import ToolBoxes.Tool_PlotRose as TPR
import ToolBoxes.Tool_PlotConfig as TPC
import ToolBoxes.Tool_InputOutput as TIO
import ToolBoxes.Tool_DataPrepare as TDP
import ToolBoxes.Tool_ImageToolkit as TIT

FIGFMT = TPC.FIGFMT
DPI    = TPC.DPI_medium

mpl.use('Agg')  # 不显示图，只保存
mpl.rcParams['font.family'] = 'Noto Sans'
warnings.filterwarnings("ignore", category=RuntimeWarning)


def PressureLevelSignificanceOfChange_seasonal(
        xarr1: xr.Dataset, xarr2: xr.Dataset, level: int, checkmethod: str, *,
        var: str, OutDir: str, caselist = ['Lake', 'NoLake'], rspmethod = 'mean') -> None:
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
    alternative = "two-sided"     # ["two-sided", "greater", "less"] 
    alpha_ci = 0.05               # 显著性水平
    clt_n = 30                    # 中心极限定理样本量
    n_sample = 10000              # 重抽样次数
    ci = 0.95                     # 置信区间
    center_null = True            # 是否中心化零假设
    random_state = 666            # 随机种子
    n_jobs = 96

    target = f"PressureLevel"
    rspfreq = 'season'
    outcase = f"{OutDir}/{target}/{level}hPa"
    os.makedirs(outcase, exist_ok=True)

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
        if np.any(mask):  # ✅ 只有在“至少有一个有效 p 值”时才做 FDR 校正
            rej_sub, p_sub, _, _ = multipletests(p_ravel[mask], alpha=0.05, method="fdr_bh")
            rej_fdr[mask] = rej_sub.astype(np.int8)
            p_fdr[mask] = p_sub
        else:
            # 没有任何有效 p 值：保持全 NaN / 0，或者打印一行提示
            print("    [Warn] No finite p-values, skip FDR correction for this case.")
        rej_fdr = rej_fdr.reshape(p_arr.shape); p_fdr = p_fdr.reshape(p_arr.shape)
        savepath = f'{outcase}/Significance_{target}_{caselist[0]}-{caselist[1]}_{var}_seasonal_{season}_{checkmethod}.nc'
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
    if np.any(mask):  # ✅ 只有在“至少有一个有效 p 值”时才做 FDR 校正
        rej_sub, p_sub, _, _ = multipletests(p_ravel[mask], alpha=0.05, method="fdr_bh")
        rej_fdr[mask] = rej_sub.astype(np.int8)
        p_fdr[mask] = p_sub
    else:
        # 没有任何有效 p 值：保持全 NaN / 0，或者打印一行提示
        print("    [Warn] No finite p-values, skip FDR correction for this case.")
    rej_fdr = rej_fdr.reshape(p_arr.shape); p_fdr = p_fdr.reshape(p_arr.shape)
    savepath = f'{outcase}/Significance_{target}_{caselist[0]}-{caselist[1]}_{var}_yearly_{checkmethod}.nc'
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
def PressureLevelSignificanceOfChange_diurnal(
        xarr1: xr.Dataset, xarr2: xr.Dataset, level: int, checkmethod: str, *,
        var: str, OutDir: str, caselist = ['Lake', 'NoLake'], rspmethod = 'mean') -> None:
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
    n_jobs = 96

    target = f"PressureLevel"
    rspfreq = 'season'
    outcase = f"{OutDir}/{target}/{level}hPa"
    os.makedirs(outcase, exist_ok=True)

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
            rej_sub, p_sub, _, _ = multipletests(p_ravel[mask], alpha=0.05, method="fdr_bh")
            rej_fdr[mask] = rej_sub.astype(np.int8); p_fdr[mask] = p_sub
            rej_fdr = rej_fdr.reshape((ny, nx)); p_fdr = p_fdr.reshape((ny, nx))
            rej_fdr_[hour, :, :] = rej_fdr
            p_fdr_[hour, :, :] = p_fdr
            RC_overall[hour, :, :], RC_anomaly[hour, :, :] = TST.RelativeContribution(arr1_season[:, hour, :, :], arr2_season[:, hour, :, :], time_axis=0)
        savepath = f'{outcase}/Significance_{target}_{caselist[0]}-{caselist[1]}_{var}_{level}hPa_seasonaldiurnalmean_{season}_{checkmethod}.nc'
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
        rej_sub, p_sub, _, _ = multipletests(p_ravel[mask], alpha=0.05, method="fdr_bh")
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
    savepath = f'{outcase}/Significance_{target}_{caselist[0]}-{caselist[1]}_{var}_{level}hPa_yearlydiurnalmean_{checkmethod}.nc'
    TIO.save_newnc(savepath=savepath, in_dict=in_dict, coords=coords)








def Plot_PressureLevel(
                        varname: str, caselist: list[str], level: int,
                        lon2d: np.ndarray, lat2d: np.ndarray,
                        checkmethod: str, OutDir:str, FigOutDir: str,
                        lkinfos: Any, onlysig: bool = True) -> None:
    """绘制热平流显著性变化图"""
    target = f"PressureLevel"
    outcase = f"{OutDir}/{target}/{level}hPa"
    seasons = TU.get_seasons()
    keep_hours = TU.get_all_hours()
    FigOutDir_var = f'{FigOutDir}/{level}hPa/Single'
    os.makedirs(FigOutDir_var, exist_ok=True)
    oceanmask = lkinfos['ocean']
    """绘制热平流显著性变化图"""
    tasks = []
    var_info = TYCM.Variable_Infos(varname)
    levels = TYCM.Seasonal_PressureLevel_Cmap(varname)
    season_diffcfg = TPC.mapConfig(levs=levels['diff_maplevs_seasonal'], cmap=levels['diff_cmap'])
    season_rccfg = TPC.mapConfig(levs=levels['rc_maplevs_seasonal'], cmap=levels['rc_cmap'])
    annual_diffcfg = TPC.mapConfig(levs=levels['diff_maplevs_annual'], cmap=levels['diff_cmap'])
    annual_rccfg = TPC.mapConfig(levs=levels['rc_maplevs_annual'], cmap=levels['rc_cmap'])
    suffix = f"onlysig" if onlysig else "all"
    
    meancfg = TPC.mapConfig(levs=levels['maplevs_seasonal'], cmap=levels['diff_cmap'])
    varInfo = TPC.varInfo(longname=var_info['longname'], unit=var_info['bunit'], abbr=var_info['abbr'])
    rosecfg = TPC.roseConfig(roselevs=levels['roselevels'], colors_dict={'Strong':('grey', 'o')})

    for season in seasons:
        path_seasonal = f'{outcase}/Significance_{target}_{caselist[0]}-{caselist[1]}_{varname}_seasonal_{season}_{checkmethod}.nc'
        xarr_seasonal = TIO.read_newnc(path_seasonal)
        savepath = f'{FigOutDir_var}/{target}_Diff_Map_{varname}_{season}_{suffix}_{checkmethod}.{FIGFMT}'
        tasks.append((TPAM.plot_spatial_diffmap_withsign, (xarr_seasonal, varname, season, target, lon2d, lat2d, 
                                                           checkmethod, lkinfos, season_diffcfg, onlysig, savepath)))
        # savepath = f'{FigOutDir_var}/{target}_RC_Map_{varname}_{season}_{suffix}_{checkmethod}.{FIGFMT}'
        # tasks.append((TPAM.plot_spatial_rcmap_withsign, (xarr_seasonal, varname, season, target, lon2d, lat2d, 
        #                                                  checkmethod, lkinfos, season_rccfg, onlysig, savepath)))
        # 读取季节昼夜数据
        ds_dict = TIO.read_hourly_significance_PressureLevel(caselist, varname, season, level, keep_hours, checkmethod, OutDir)
        in_df, sig_mask, strong_mask, weak_mask, suffix = TDP.prepara_for_diurnal_rose(ds_dict, lkinfos, varname, checkmethod, onlysig)
        tasks.append((TPR.plot_regclimimpact_diurnal_rose, (in_df, varname, season, target, checkmethod, FigOutDir_var, rosecfg, varInfo, suffix)))
    
    path_annual = f'{outcase}/Significance_{target}_{caselist[0]}-{caselist[1]}_{varname}_yearly_{checkmethod}.nc'
    xarr_annual = TIO.read_newnc(path_annual)
    savepath = f'{FigOutDir_var}/{target}_Diff_Map_{varname}_Annual_{suffix}_{checkmethod}.{FIGFMT}'
    tasks.append((TPAM.plot_spatial_diffmap_withsign, (xarr_annual, varname, "Annual", target, lon2d, lat2d, 
                                                       checkmethod, lkinfos, annual_diffcfg, onlysig, savepath)))
    # savepath = f'{FigOutDir_var}/{target}_RC_Map_{varname}_Annual_{suffix}_{checkmethod}.{FIGFMT}'
    # tasks.append((TPAM.plot_spatial_rcmap_withsign, (xarr_annual, varname, "Annual", target, lon2d, lat2d, 
    #                                                  checkmethod, lkinfos, annual_rccfg, onlysig, savepath)))
    label = f"{varInfo.longname} ({varInfo.unit})"
    savepath = f'{FigOutDir_var}/{target}_Diff_Map_{varname}_Annual_Colorbar_{suffix}_{checkmethod}.{FIGFMT}'
    tasks.append((TPCB.plot_spatial_cbar_core_V, (annual_diffcfg, 4, label, savepath, 18, 18, 0.06, 'both','{:5.1f}'))) 
    savepath = f'{FigOutDir_var}/{target}_Diff_Map_{varname}_Seasonal_Colorbar_{suffix}_{checkmethod}.{FIGFMT}'
    tasks.append((TPCB.plot_spatial_cbar_core_V, (season_diffcfg, 4, label, savepath, 18, 18, 0.06, 'both','{:5.1f}'))) 
    # savepath = f'{FigOutDir_var}/{target}_RC_Map_{varname}_Annual_Colorbar_{suffix}_{checkmethod}.{FIGFMT}'
    # tasks.append((TPCB.plot_spatial_cbar_core_V, (annual_rccfg, 4, "Relative Contribution (%)", savepath, 18, 18, 0.06, 'both','{:5.1f}'))) 
    # savepath = f'{FigOutDir_var}/{target}_RC_Map_{varname}_Seasonal_Colorbar_{suffix}_{checkmethod}.{FIGFMT}'
    # tasks.append((TPCB.plot_spatial_cbar_core_V, (season_rccfg, 4, "Relative Contribution (%)", savepath, 18, 18, 0.06, 'both','{:5.1f}'))) 

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




def Merge_PressureLevel(varname: str, caselist:list[str], level:int, checkmethod:str, FigOutDir:str, onlysig: bool = True) -> None:
    """合并热平流显著性测试结果文件（季节+年际）为单一文件"""
    target = f"PressureLevel"
    FigOutDir_var = f'{FigOutDir}/{level}hPa/Single'
    seasons = TU.get_seasons()
    mapcrop_params = TYCM.MapPlot_CropParams_noColorbar()
    figbox_space = TYCM.Merge_Fig_Space_Params()
    cbar_space = TYCM.MapCbar_CropParams_V()
    rosecrop_params = TYCM.RosePlot_CropParams_noColorbar()
    suffix = "onlysig" if onlysig else "all"
    var_info = TYCM.Variable_Infos(varname)
    #############################
    # Diff maps
    #############################
    cols_space = [[0, 0, 0, 0], [0, 0, 0, 0]]
    rows_space = [0.01, 0.01]
    rows = [[], []]
    # 季节diff map & rc map
    rowpos = 0
    for season in seasons:
        savepath = f'{FigOutDir_var}/{target}_Diff_Map_{varname}_{season}_{suffix}_{checkmethod}.{FIGFMT}'
        cropped_diff = TIT.crop_image_from_path(savepath, crop_params=mapcrop_params, mode="ratio")
        rows[rowpos].append(cropped_diff)
    savepath = f'{FigOutDir_var}/{target}_Diff_Map_{varname}_Seasonal_Colorbar_{suffix}_{checkmethod}.{FIGFMT}'
    cropped_diff_cbar = TIT.crop_image_from_path(savepath, crop_params=cbar_space, mode="ratio")
    rows[rowpos].append(cropped_diff_cbar)
    rowpos = 1
    for season in seasons:
        savepath = f'{FigOutDir_var}/{target}_RC_Map_{varname}_{season}_{suffix}_{checkmethod}.{FIGFMT}'
        cropped_diff = TIT.crop_image_from_path(savepath, crop_params=mapcrop_params, mode="ratio")
        rows[rowpos].append(cropped_diff)
    savepath = f'{FigOutDir_var}/{target}_RC_Map_{varname}_Seasonal_Colorbar_{suffix}_{checkmethod}.{FIGFMT}'
    cropped_diff_cbar = TIT.crop_image_from_path(savepath, crop_params=cbar_space, mode="ratio")
    rows[rowpos].append(cropped_diff_cbar)
    seas_img = TIT.merge_images_Row(
        rows_images=rows,
        cols_space=cols_space,
        rows_space=rows_space,
        box_space=figbox_space,
        background_color='#FFFFFFFF',
        space_mode="ratio",
        alignment=["left","left"],
        draw_ticks=False, tick_step=0.01,
        font_path="/share/home/dq048/.local/share/fonts/NotoSans-Bold.ttf",
    )
    savepath = f'{FigOutDir}/{target}_{varname}_{level}hPa_Seasonal_Comparison.{FIGFMT}'
    seas_img.save(savepath, dpi=[DPI]*16)




def DTheta_Dz(
        Theta_da: xr.DataArray,
        Height_da: xr.DataArray,
        vertical_dim=0
    ) -> xr.DataArray:
    """
    使用 MetPy 计算位温沿垂直方向的梯度 dθ/dz，返回单位为 K/km。

    参数
    ----
    Theta_da : xr.DataArray
        位温场，单位 K。
    Height_da : xr.DataArray
        高度场，单位 m，与 Theta_da 形状一致。
    vertical_dim : int 或 str, 默认 0
        垂直维的索引或名称（同 StaticStability）。

    返回
    ----
    dtheta_dz_da : xr.DataArray
        位温垂直梯度 dθ/dz，单位 K/km，形状与 Theta_da 相同。
    """
    if isinstance(vertical_dim, int):
        z_dim = Theta_da.dims[vertical_dim]
    else:
        z_dim = vertical_dim

    if Theta_da.shape != Height_da.shape:
        raise ValueError("Theta_da 和 Height_da 的形状必须一致。")

    # 把垂直维放到最后
    other_dims = [d for d in Theta_da.dims if d != z_dim]
    Theta_tr = Theta_da.transpose(*other_dims, z_dim)
    Height_tr = Height_da.transpose(*other_dims, z_dim)

    theta_q = (Theta_tr.values * units.kelvin)
    z_q = (Height_tr.values * units.meter)

    # 在最后一维上计算 dθ/dz
    dtheta_dz_q = mpcalc.first_derivative(theta_q, x=z_q, axis=-1)
    
    # 将单位从 K/m 转换为 K/km
    dtheta_dz_q = dtheta_dz_q.to('kelvin/kilometer')

    dtheta_dz_tr = xr.DataArray(
        dtheta_dz_q.magnitude,
        dims=Theta_tr.dims,
        coords=Theta_tr.coords,
        name="dtheta_dz"
    )
    dtheta_dz_tr.attrs["long_name"] = "vertical gradient of potential temperature"
    dtheta_dz_tr.attrs["units"] = "K km^-1"  # 更新单位为 K/km

    dtheta_dz_da = dtheta_dz_tr.transpose(*Theta_da.dims)
    return dtheta_dz_da



def cal_static_stability(
        temp_da: xr.DataArray,
        pressure_da: xr.DataArray,
        vertical_dim=0,
        ensure_monotonic=True
    ) -> xr.DataArray:
    """
    计算静力稳定度（Static Stability）∂θ/∂p。

    参数：
    temp_da : xr.DataArray
        温度数据，单位可以是 K 或 °C。
    pressure_da : xr.DataArray
        气压数据，单位可以是 hPa 或 Pa。
    vertical_dim : int 或 str, optional
        垂直维度索引或名称，默认为0。
    ensure_monotonic : bool, optional
        是否确保气压单调递减（从地面到高空），默认为True。

    返回：
    xr.DataArray
        静力稳定度 ∂θ/∂p，单位 K/Pa 或 K/hPa。
    """
    # 获取垂直维度名称
    if isinstance(vertical_dim, int):
        z_dim = temp_da.dims[vertical_dim]
    else:
        z_dim = vertical_dim
    
    # 确保气压数据单调递减（从地面到高空）
    if ensure_monotonic:
        # 检查气压是否单调递减
        pressure_values = pressure_da.values
        is_monotonic = np.all(np.diff(pressure_values, axis=vertical_dim) < 0)
        
        if not is_monotonic:
            print(f"警告：气压数据不是单调递减的，沿维度 {z_dim} 排序...")
            
            # 沿着垂直维度对气压和温度进行排序
            sort_idx = np.argsort(pressure_values, axis=vertical_dim)
            
            # 创建索引数组
            idx = np.ogrid[tuple(slice(s) for s in pressure_values.shape)]
            idx[vertical_dim] = sort_idx
            
            # 排序气压
            pressure_sorted = pressure_values[tuple(idx)]
            pressure_da = xr.DataArray(
                pressure_sorted,
                dims=pressure_da.dims,
                coords=pressure_da.coords,
                attrs=pressure_da.attrs
            )
            
            # 排序温度（使用相同的索引）
            temp_values = temp_da.values
            temp_sorted = temp_values[tuple(idx)]
            temp_da = xr.DataArray(
                temp_sorted,
                dims=temp_da.dims,
                coords=temp_da.coords,
                attrs=temp_da.attrs
            )
    
    # 转换为 MetPy 单位
    pressure_q = pressure_da.values * units(pressure_da.attrs.get('units', 'Pa'))
    temperature_q = temp_da.values * units(temp_da.attrs.get('units', 'K'))
    
    # 确保温度是开尔文
    if temperature_q.units != units.kelvin:
        temperature_q = temperature_q.to('kelvin')
    
    # 检查数据中是否有 NaN
    if np.any(np.isnan(pressure_q.magnitude)) or np.any(np.isnan(temperature_q.magnitude)):
        print("警告：输入数据中存在 NaN，结果中可能出现 NaN")
    
    # 计算 σ = (Rd/p) * (∂θ/∂p)
    try:
        sigma = mpcalc.static_stability(pressure_q, temperature_q, vertical_dim=vertical_dim)
    except Exception as e:
        print(f"计算 static_stability 时出错: {e}")
        # 尝试直接计算 ∂θ/∂p
        theta = mpcalc.potential_temperature(pressure_q, temperature_q)
        dtheta_dp = mpcalc.first_derivative(theta, x=pressure_q, axis=vertical_dim)
        dtheta_dp_q = dtheta_dp
    else:
        # 计算 ∂θ/∂p = σ * (p/Rd)
        dtheta_dp_q = sigma * (pressure_q / Rd)
    
    # 转换为合适的输出单位
    if str(pressure_q.units) in ['hectopascal', 'hPa']:
        dtheta_dp_q = dtheta_dp_q.to('K/hPa')
        output_unit = "K hPa^-1"
    else:  # 假定是 Pa
        dtheta_dp_q = dtheta_dp_q.to('K/Pa')
        output_unit = "K Pa^-1"
    
    # 创建 DataArray
    result = xr.DataArray(
        dtheta_dp_q.magnitude,
        dims=temp_da.dims,
        coords=temp_da.coords,
        name="static_stability"
    )
    result.attrs["long_name"] = "Static stability (∂θ/∂p)"
    result.attrs["units"] = output_unit
    
    return result



def cal_dtheta_dp_directly(
        temp_da: xr.DataArray,
        pressure_da: xr.DataArray,
        vertical_dim=0
    ) -> xr.DataArray:
    """
    直接计算位温对气压的导数 ∂θ/∂p。

    参数：
    temp_da : xr.DataArray
        温度数据，单位 K。
    pressure_da : xr.DataArray
        气压数据，单位 hPa。
    vertical_dim : int 或 str, optional
        垂直维度。

    返回：
    xr.DataArray
        静力稳定度 ∂θ/∂p，单位 K/hPa。
    """
    if isinstance(vertical_dim, int):
        z_dim = temp_da.dims[vertical_dim]
    else:
        z_dim = vertical_dim
    
    # 确保气压是递减的（地面到高空）
    # 将垂直维度放到最后以便计算
    other_dims = [d for d in temp_da.dims if d != z_dim]
    temp_tr = temp_da.transpose(*other_dims, z_dim)
    pressure_tr = pressure_da.transpose(*other_dims, z_dim)
    
    # 转换为 MetPy 单位
    pressure_q = (pressure_tr.values * units.hPa)
    temp_q = (temp_tr.values * units.kelvin)
    
    # 计算位温
    theta_q = mpcalc.potential_temperature(pressure_q, temp_q)
    
    # 检查气压是否单调递减，如果不是则反向
    pressure_1d = pressure_tr.isel({d: 0 for d in other_dims}).values
    if not np.all(np.diff(pressure_1d) < 0):
        print(f"警告：气压不是单调递减，正在反转垂直维度 {z_dim}")
        pressure_q = pressure_q[..., ::-1]
        theta_q = theta_q[..., ::-1]
    
    # 计算 ∂θ/∂p
    dtheta_dp_q = mpcalc.first_derivative(theta_q, x=pressure_q, axis=-1)
    
    # 如果反转了，再反转回来
    if not np.all(np.diff(pressure_1d) < 0):
        dtheta_dp_q = dtheta_dp_q[..., ::-1]
    
    # 转换为 K/hPa
    dtheta_dp_q = dtheta_dp_q.to('K/hPa')
    
    # 创建 DataArray
    dtheta_dp_tr = xr.DataArray(
        dtheta_dp_q.magnitude,
        dims=temp_tr.dims,
        coords=temp_tr.coords,
        name="dtheta_dp"
    )
    dtheta_dp_tr.attrs["long_name"] = "Static stability (∂θ/∂p)"
    dtheta_dp_tr.attrs["units"] = "K hPa^-1"
    
    # 转回原始维度顺序
    dtheta_dp_da = dtheta_dp_tr.transpose(*temp_da.dims)
    
    return dtheta_dp_da




def Cal_StaticStability_and_DThetaDz(
        caselist: list[str],
        case1: xr.Dataset,
        case2: xr.Dataset,
        pressures: list,
        outdir: str,
    ) -> None:
    """
    计算静力稳定度 N^2 和位温垂直梯度 dθ/dz。
    """

    target = "StaticStability"
    outcase = f"{outdir}/{target}"
    os.makedirs(outcase, exist_ok=True)
    seasons = TU.get_seasons()

    # 1. 读入位温和高度
    Lake_Theta = case1["Theta"]
    NoLake_Theta = case2["Theta"]
    Lake_Height = case1["Height"]
    NoLake_Height = case2["Height"]
    Lake_Temp = case1["T"]
    NoLake_Temp = case2["T"]

    # 3. 计算 N^2 和 dθ/dz（对整个 4D 数据一次性算）
    #    这里直接调用你前面写好的两个函数
    # 4. 用 NaN 初始化的 DataArray，把结果填进去（保留你原来的接口风格）
    ntime, nz, ny, nx = case1["Theta"].shape
    for season in seasons:
        Lake_Theta_Sel = Lake_Theta.sel(time=Lake_Theta.time.dt.season == season).squeeze(drop=True)
        NoLake_Theta_Sel = NoLake_Theta.sel(time=NoLake_Theta.time.dt.season == season).squeeze(drop=True)
        Lake_Height_Sel = Lake_Height.sel(time=Lake_Height.time.dt.season == season).squeeze(drop=True)
        NoLake_Height_Sel = NoLake_Height.sel(time=NoLake_Height.time.dt.season == season).squeeze(drop=True)
        Lake_Temp_Sel = Lake_Temp.sel(time=Lake_Temp.time.dt.season == season).squeeze(drop=True)
        NoLake_Temp_Sel = NoLake_Temp.sel(time=NoLake_Temp.time.dt.season == season).squeeze(drop=True)
        Lake_Theta_Cilm = Lake_Theta_Sel.mean("time").squeeze(drop=True)
        NoLake_Theta_Cilm = NoLake_Theta_Sel.mean("time").squeeze(drop=True)
        Lake_Height_Cilm = Lake_Height_Sel.mean("time").squeeze(drop=True)
        NoLake_Height_Cilm = NoLake_Height_Sel.mean("time").squeeze(drop=True)
        Lake_Temp_Cilm = Lake_Temp_Sel.mean("time").squeeze(drop=True)
        NoLake_Temp_Cilm = NoLake_Temp_Sel.mean("time").squeeze(drop=True)
        # 广播成和 Lake_Temp 一样的多维场 (比如 24×186×285)

        # 2. 自动识别垂直维度名称
        size_keys = list(NoLake_Temp_Cilm.sizes.keys())
        if "level" in NoLake_Temp_Cilm.sizes:
            z_dim = size_keys.index("level")
            z_dim_name = "level"
        elif "bottom_top" in NoLake_Temp_Cilm.sizes:
            z_dim = size_keys.index("bottom_top")
            z_dim_name = "bottom_top"
        elif "z" in NoLake_Temp_Cilm.sizes:
            z_dim = size_keys.index("z")
            z_dim_name = "z"
        else:
            raise ValueError("无法识别垂直维度名称，请确保数据中包含 'level'、'bottom_top' 或 'z' 之一。")
        
        pressures = np.asarray(pressures)
        pressure_da = np.broadcast_to(pressures.reshape((nz, 1, 1)),Lake_Temp_Cilm.shape)
        pressure_da = xr.DataArray(pressure_da,dims=Lake_Temp_Cilm.dims,coords=Lake_Temp_Cilm.coords)
        pressure_da.attrs["units"] = "hPa"

        Lake_dTheta_season = DTheta_Dz(Lake_Theta_Cilm, Lake_Height_Cilm, vertical_dim=z_dim_name)
        NoLake_dTheta_season = DTheta_Dz(NoLake_Theta_Cilm, NoLake_Height_Cilm, vertical_dim=z_dim_name)
        Lake_ss_season = cal_dtheta_dp_directly(Lake_Temp_Cilm, pressure_da, vertical_dim=z_dim)
        NoLake_ss_season = cal_dtheta_dp_directly(NoLake_Temp_Cilm, pressure_da, vertical_dim=z_dim)

        # 5. 保存完整时间序列的 dTheta_dz（3D） dTheta_dp（3D）
        in_dict = {
            f"{caselist[0]}_dTheta":  [['level', 'y', 'x'],     Lake_dTheta_season.values],
            f"{caselist[1]}_dTheta":  [['level', 'y', 'x'],   NoLake_dTheta_season.values],
            f"{caselist[0]}_StaticStability":  [['level', 'y', 'x'],     Lake_ss_season.values],
            f"{caselist[1]}_StaticStability":  [['level', 'y', 'x'],   NoLake_ss_season.values],
            f"{caselist[0]}_Theta":  [['level', 'y', 'x'],     Lake_Theta_Cilm.values],
            f"{caselist[1]}_Theta":  [['level', 'y', 'x'],   NoLake_Theta_Cilm.values],
            f"{caselist[0]}_Height":  [['level', 'y', 'x'],     Lake_Height_Cilm.values],
            f"{caselist[1]}_Height":  [['level', 'y', 'x'],   NoLake_Height_Cilm.values],
            f"{caselist[0]}_Temp":  [['level', 'y', 'x'],     Lake_Temp_Cilm.values],
            f"{caselist[1]}_Temp":  [['level', 'y', 'x'],   NoLake_Temp_Cilm.values],
        }

        coords = {"level": Lake_Theta_Cilm["level"], "y": Lake_Theta_Cilm["y"], "x": Lake_Theta_Cilm["x"]}
        TIO.save_newnc(
            savepath=f"{outcase}/StaticStability_{caselist[0]}_{caselist[1]}_seasonal_{season}.nc",
            in_dict=in_dict,
            coords=coords
        )




def Plot_StaticStability_and_DThetaDz(
        caselist: list[str],
        levels: List[int],
        OutDir: str,
        FigOutDir: str,
        lkinfos: Any,
    ) -> None:
    """绘制静力稳定度和位温垂直梯度图（Lake / NoLake 分开保存，四季在同一张图）"""
    target = "StaticStability"
    outcase = f"{OutDir}/{target}"
    FigOutDir_var = f"{FigOutDir}/{target}/Single"
    os.makedirs(FigOutDir_var, exist_ok=True)

    lakeonly = False

    oceanmask = lkinfos["ocean"]
    lkmask = lkinfos["all"]
    seasons = TU.get_seasons()  # 比如 ["DJF", "MAM", "JJA", "SON"]
    varnames = ["dTheta", "StaticStability"]

    # 不同变量的 x 轴范围
    xlim_dict: Dict[str, List[float]] = {
        "dTheta": [2.5, 8],
        "StaticStability": [0.0, 0.6],
    }

    # 2) 按从大到小排序 pressure levels，并转成数组
    #    这些 levels 的顺序要与你存文件时的 level 轴顺序一致（0 对应最大压强）
    levels_sorted = np.array(sorted(levels, reverse=True))

    # 给四季分配颜色
    season_colors = {
        "DJF": "#9C0C5E",
        "MAM": "#FF684A",
        "JJA": "#FFA96B",
        "SON": "#225dbd",
    }

    for varname in varnames:
        # 先准备两个容器，把每个季节的垂直廓线存起来
        Lake_profiles: Dict[str, np.ndarray] = {}
        NoLake_profiles: Dict[str, np.ndarray] = {}
        var_info = TYCM.Variable_Infos(varname)
        # 当前变量的横轴范围
        if varname not in xlim_dict:
            raise KeyError(f"xlim_dict 中没有为变量 {varname} 定义范围！")
        x_min, x_max = xlim_dict[varname]

        # ========= 先把四个季节的剖面全部读出来 =========
        for season in seasons:
            # 1) 读取数据（按季节）
            ncfile = (
                f"{outcase}/StaticStability_{caselist[0]}_{caselist[1]}_seasonal_{season}.nc"
            )
            xarr = TIO.read_newnc(ncfile)

            # 2) 一致性检查
            if "level" in xarr.sizes:
                nz = xarr.sizes["level"]
                if nz != len(levels_sorted):
                    raise ValueError(
                        f"输入 levels 长度 ({len(levels_sorted)}) 与文件中 level 维度长度 ({nz}) 不一致！"
                    )

            print(f"Processing variable: {varname} for season: {season}")
            Lake_varxarr = xarr[f"{caselist[0]}_{varname}"]  # (level, y, x)
            NoLake_varxarr = xarr[f"{caselist[1]}_{varname}"]

            Lake_var = Lake_varxarr.values
            NoLake_var = NoLake_varxarr.values

            # 只取湖区（lkmask 应为 2D (y, x)，广播到 (level, y, x)）
            Lake_var_lk = np.where(lkmask, Lake_var, np.nan)
            NoLake_var_lk = np.where(lkmask, NoLake_var, np.nan)

            # 在湖区上对 (y, x) 做空间平均 -> (nlevel,)
            if lakeonly:
                Lake_var_mean = np.nanmean(Lake_var_lk, axis=(1, 2))
                NoLake_var_mean = np.nanmean(NoLake_var_lk, axis=(1, 2))
            else: 
                Lake_var_mean = np.nanmean(Lake_var, axis=(1, 2))
                NoLake_var_mean = np.nanmean(NoLake_var, axis=(1, 2))


            print("  Lake max", np.nanmax(Lake_var_mean))
            print("  NoLake max", np.nanmax(NoLake_var_mean))
            print("  Lake min", np.nanmin(Lake_var_mean))
            print("  NoLake min", np.nanmin(NoLake_var_mean))

            # 存起来，等四季都算完再统一画
            Lake_profiles[season] = Lake_var_mean
            NoLake_profiles[season] = NoLake_var_mean

        # ========= 图 1：只画 Lake（四季一张图） =========
        fig_lake, ax_lake = plt.subplots(figsize=(6, 12 ), layout='constrained')

        for season in seasons:
            ax_lake.plot(
                Lake_profiles[season],
                levels_sorted,
                label=season,
                color=season_colors.get(season, None),
                marker="o",
                markersize=7,
                lw=3,
            )
        ax_lake.set_xlim(x_min, x_max)
        ax_lake.invert_yaxis()  # y 轴：大压强在下面
        ax_lake.set_xticks([2, 3, 4, 5, 6, 7, 8, 9])
        ax_lake.set_yticks([1000, 900, 800, 700, 600, 500, 400, 300, 200, 100])
        ax_lake.tick_params(axis='both', which='major', labelsize=20)
        ax_lake.set_xlabel(f"{var_info['abbr']} ({var_info['unit']})", fontsize=26)
        ax_lake.set_ylabel("Pressure Level (hPa)", fontsize=26)
        # ax_lake.set_title(
        #     f"{var_info['longname']} - {caselist[0]}",
        #     fontsize=14,
        # )
        ax_lake.grid(True, linestyle="--", alpha=0.5)
        ax_lake.legend(fontsize=16)

        if lakeonly:
            savepath_lake = (f"{FigOutDir_var}/{target}_{varname}_Climatology_AllSeasons_LakeArea_{caselist[0]}.{FIGFMT}")
        else:
            savepath_lake = (f"{FigOutDir_var}/{target}_{varname}_Climatology_AllSeasons_AllArea_{caselist[0]}.{FIGFMT}")
        fig_lake.savefig(savepath_lake, dpi=DPI)
        plt.close(fig_lake)

        # ========= 图 2：只画 NoLake（四季一张图） =========
        fig_nl, ax_nolake = plt.subplots(figsize=(6, 12 ), layout='constrained')

        for season in seasons:
            ax_nolake.plot(
                NoLake_profiles[season],
                levels_sorted,
                label=season,
                color=season_colors.get(season, None),
                marker="s",
                markersize=7,
                lw=3,
            )
        ax_nolake.set_xlim(x_min, x_max)
        ax_nolake.invert_yaxis()
        ax_nolake.set_xlabel(f"{var_info['abbr']} ({var_info['unit']})", fontsize=14)
        ax_nolake.set_ylabel("Pressure Level (hPa)", fontsize=14)
        # ax_nolake.set_title(
        #     f"{var_info['longname']} - {caselist[1]}",
        #     fontsize=14,
        # )
        ax_nolake.grid(True, linestyle="--", alpha=0.5)
        ax_nolake.legend(fontsize=16)

        if lakeonly:
            savepath_nolake = (f"{FigOutDir_var}/{target}_{varname}_Climatology_AllSeasons_LakeArea_{caselist[1]}.{FIGFMT}")
        else:
            savepath_nolake = (f"{FigOutDir_var}/{target}_{varname}_Climatology_AllSeasons_AllArea_{caselist[1]}.{FIGFMT}")
        fig_nl.savefig(savepath_nolake, dpi=DPI)
        plt.close(fig_nl)




