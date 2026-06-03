import os
import time
import calendar
import cmaps
import matplotlib as mpl
import numpy as np
from numba import njit
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
import geopandas as gpd
from shapely.geometry import Point
import seaborn as sns
from matplotlib.font_manager import FontProperties
# 自定义模块
import ToolBoxes.Utils as TU
import ToolBoxes.Tool_InputOutput as TIO
import ToolBoxes.Tool_ImageToolkit as TIT
import ToolBoxes.Tool_ExtremeEventMetrics as TEEM
import ToolBoxes.Tool_WetBulbTemperature as WBT
import ToolBoxes.Tool_SignificanceTest as TST
import ToolBoxes.Tool_PlotBox as TPB
import ToolBoxes.Tool_PlotBar as TPBH
import ToolBoxes.Tool_PlotColorBar as TPCB
import ToolBoxes.Tool_PlotConfig as TPC
import ToolBoxes.Tool_DataPrepare as TDP
import ToolBoxes.Tool_PlotAreaMap as TPAM
import ToolBoxes.Tool_PlotTreemap as TPT
import ToolBoxes.Tool_PlotCircularRing as TPCR
import ToolBoxes.Tool_PlotRadialHistogram as TPRH
import ToolBoxes.Tool_YangtzeColorMap as TYCM
import ToolBoxes.Tool_PlotJoint as TPJ
import ToolBoxes.Tool_PlotHeatMap as TPHM
FIGFMT = TPC.FIGFMT
DPI    = TPC.DPI_medium
HighThresPercentile = TPC.HighThresPercentile
LowThresPercentile  = TPC.LowThresPercentile
ThresWindows        = TPC.ThresWindows
fdr_alpha = TPC.fdr_alpha
mpl.use('Agg')  # 不显示图，只保存


# 计算极端事件阈值
def CalExtremeEventThres(xarr: xr.Dataset, refname: str, *, outdir: str, ) -> None:
    t0 = time.time()
    n_jobs = 96
    xarr = xarr.squeeze()
    xarr = TU.xarray_leap_to_noleap(xarr)
    RH = xarr["RH"].values
    T = xarr["T2m"].values
    timelist = pd.DatetimeIndex(xarr["time"].values)
    ntime, ny, nx = RH.shape
    print(f"    ➠ RH: ntime: {ntime}, ny: {ny}, nx: {nx}")
    ntime, ny, nx = T.shape
    print(f"    ➠ T2m: ntime: {ntime}, ny: {ny}, nx: {nx}")
    RH = RH.reshape(ntime, ny * nx)
    T = T.reshape(ntime, ny * nx)
    thres_RH_Wet = np.full((ntime, ny * nx), np.nan, dtype=np.float32)
    thres_RH_Dry = np.full((ntime, ny * nx), np.nan, dtype=np.float32)
    thres_T_Cold = np.full((ntime, ny * nx), np.nan, dtype=np.float32)
    thres_T_Hot = np.full((ntime, ny * nx), np.nan, dtype=np.float32)
    thres_Tw_ColdDry = np.full((ntime, ny * nx), np.nan, dtype=np.float32)
    thres_Tw_ColdWet = np.full((ntime, ny * nx), np.nan, dtype=np.float32)
    thres_Tw_HotDry = np.full((ntime, ny * nx), np.nan, dtype=np.float32)
    thres_Tw_HotWet = np.full((ntime, ny * nx), np.nan, dtype=np.float32)
    # 构造所有任务的输入参数列表
    tasks = [
        (icell, RH[:, icell], T[:, icell], timelist)
        for icell in range(ny * nx)
    ]
    
    t1 = time.time()
    with Parallel(n_jobs=n_jobs, backend="loky", return_as="generator") as parallel:
        gen = parallel(
            delayed(TEEM.calculate_thresholds)(*task)
            for task in tasks
        )
        for res in tqdm(gen, total=ny * nx, desc='calculate thresholds ', unit="cell",
                           dynamic_ncols=True, leave=False):
            icell = res['icell']
            thres_RH_Wet[:, icell] = res['thres_RH_Wet']
            thres_RH_Dry[:, icell] = res['thres_RH_Dry']
            thres_T_Hot[:, icell] = res['thres_T_Hot']
            thres_T_Cold[:, icell] = res['thres_T_Cold']
            thres_Tw_ColdDry[:, icell] = res['thres_Tw_ColdDry']
            thres_Tw_ColdWet[:, icell] = res['thres_Tw_ColdWet']
            thres_Tw_HotDry[:, icell] = res['thres_Tw_HotDry']
            thres_Tw_HotWet[:, icell] = res['thres_Tw_HotWet']
    # reshape back
    thres_RH_Wet = thres_RH_Wet.reshape(ntime, ny, nx)
    thres_RH_Dry = thres_RH_Dry.reshape(ntime, ny, nx)
    thres_T_Hot = thres_T_Hot.reshape(ntime, ny, nx)
    thres_T_Cold = thres_T_Cold.reshape(ntime, ny, nx)
    thres_Tw_ColdDry = thres_Tw_ColdDry.reshape(ntime, ny, nx)
    thres_Tw_ColdWet = thres_Tw_ColdWet.reshape(ntime, ny, nx)
    thres_Tw_HotDry = thres_Tw_HotDry.reshape(ntime, ny, nx)
    thres_Tw_HotWet = thres_Tw_HotWet.reshape(ntime, ny, nx)
    in_dict = {
        'thres_RH_Wet': (['time','y','x'], thres_RH_Wet),
        'thres_RH_Dry':  (['time','y','x'], thres_RH_Dry),
        'thres_T_Hot': (['time','y','x'], thres_T_Hot),
        'thres_T_Cold':  (['time','y','x'], thres_T_Cold),
        'thres_Tw_ColdDry': (['time','y','x'], thres_Tw_ColdDry),
        'thres_Tw_ColdWet': (['time','y','x'], thres_Tw_ColdWet),
        'thres_Tw_HotDry': (['time','y','x'], thres_Tw_HotDry),
        'thres_Tw_HotWet': (['time','y','x'], thres_Tw_HotWet),
    }
    coords = {'y': xarr.y, 'x': xarr.x, 'time': xarr.time}
    savepath = f"{outdir}/ExtremeAnalysis/Extreme_thresholds_ref_{refname}.nc"
    TIO.save_newnc(savepath=savepath, in_dict=in_dict, coords=coords)
    print(f"    ➠ Time Spent: {time.time() - t0:.2f} seconds")



# 识别极端事件
def IdentifyExtremeEvents(xarr_in: xr.Dataset, xarr_ref: xr.Dataset, caselist: List[str], *, outdir: str) -> None:
    """
    xarr_in: Input dataset for identifying extreme events
    xarr_ref: Reference dataset for calculating extreme event thresholds
    caselist: List of case names, with the first case being the input dataset
              and the second being the reference dataset
    outdir: Output directory for saving results
    """
    t0 = time.time()
    n_jobs = 96
    xarr_in = xarr_in.squeeze()
    xarr_ref = xarr_ref.squeeze()
    xarr_in = TU.xarray_leap_to_noleap(xarr_in)
    xarr_ref = TU.xarray_leap_to_noleap(xarr_ref)
    RH_in = xarr_in["RH"].values
    T_in = xarr_in["T2m"].values
    RH_ref = xarr_ref["RH"].values
    T_ref = xarr_ref["T2m"].values
    timelist = pd.DatetimeIndex(xarr_in["time"].values)
    ntime, ny, nx = RH_in.shape
    RH_in = RH_in.reshape(ntime, ny * nx)
    T_in = T_in.reshape(ntime, ny * nx)
    RH_ref = RH_ref.reshape(ntime, ny * nx)
    T_ref = T_ref.reshape(ntime, ny * nx)
    nfeatures = 4
    # 预分配内存
    Wet = np.full((nfeatures, ny * nx), np.nan, dtype=np.float32)
    Dry = np.full((nfeatures, ny * nx), np.nan, dtype=np.float32)
    Hot = np.full((nfeatures, ny * nx), np.nan, dtype=np.float32)
    Cold = np.full((nfeatures, ny * nx), np.nan, dtype=np.float32)
    ColdWet = np.full((nfeatures, ny * nx), np.nan, dtype=np.float32)
    ColdDry = np.full((nfeatures, ny * nx), np.nan, dtype=np.float32)
    HotWet = np.full((nfeatures, ny * nx), np.nan, dtype=np.float32)
    HotDry = np.full((nfeatures, ny * nx), np.nan, dtype=np.float32)
    Wet_Flag = np.full((ntime, ny * nx), np.nan, dtype=np.float32)
    Dry_Flag = np.full((ntime, ny * nx), np.nan, dtype=np.float32)
    Hot_Flag = np.full((ntime, ny * nx), np.nan, dtype=np.float32)
    Cold_Flag = np.full((ntime, ny * nx), np.nan, dtype=np.float32)
    ColdWet_Flag = np.full((ntime, ny * nx), np.nan, dtype=np.float32)
    ColdDry_Flag = np.full((ntime, ny * nx), np.nan, dtype=np.float32)
    HotWet_Flag = np.full((ntime, ny * nx), np.nan, dtype=np.float32)
    HotDry_Flag = np.full((ntime, ny * nx), np.nan, dtype=np.float32)
    RH_Dry_diff = np.full((ntime, ny * nx), np.nan, dtype=np.float32)
    RH_Wet_diff = np.full((ntime, ny * nx), np.nan, dtype=np.float32)
    T_Cold_diff = np.full((ntime, ny * nx), np.nan, dtype=np.float32)
    T_Hot_diff = np.full((ntime, ny * nx), np.nan, dtype=np.float32)
    Tw_HotDry_diff = np.full((ntime, ny * nx), np.nan, dtype=np.float32)
    Tw_HotWet_diff = np.full((ntime, ny * nx), np.nan, dtype=np.float32)
    Tw_ColdDry_diff = np.full((ntime, ny * nx), np.nan, dtype=np.float32)
    Tw_ColdWet_diff = np.full((ntime, ny * nx), np.nan, dtype=np.float32)
    thres_RH_Wet = np.full((ntime, ny * nx), np.nan, dtype=np.float32)
    thres_RH_Dry = np.full((ntime, ny * nx), np.nan, dtype=np.float32)
    thres_T_Hot = np.full((ntime, ny * nx), np.nan, dtype=np.float32)
    thres_T_Cold = np.full((ntime, ny * nx), np.nan, dtype=np.float32)
    # 构造所有任务
    tasks = [
        (icell, RH_in[:, icell], T_in[:, icell], RH_ref[:, icell], T_ref[:, icell], timelist)
        for icell in range(ny * nx)
    ]

    with Parallel(n_jobs=n_jobs, backend="loky", return_as="generator") as parallel:
        gen = parallel(
            delayed(TEEM._worker_classify_extreme_events)(*task)
            for task in tasks
        )
        for icell, res in tqdm(gen, total=ny * nx, desc='classify extreme events ', unit="cell",
                        dynamic_ncols=True, leave=False):
            Wet[:, icell] = res.ex_wet_feature
            Dry[:, icell] = res.ex_dry_feature
            Hot[:, icell] = res.ex_Hot_feature
            Cold[:, icell] = res.ex_cold_feature
            ColdWet[:, icell] = res.cex_ColdWet_feature
            ColdDry[:, icell] = res.cex_ColdDry_feature
            HotWet[:, icell] = res.cex_HotWet_feature
            HotDry[:, icell] = res.cex_HotDry_feature
            Wet_Flag[:, icell] = res.ex_wet_flag
            Dry_Flag[:, icell] = res.ex_dry_flag
            Hot_Flag[:, icell] = res.ex_Hot_flag
            Cold_Flag[:, icell] = res.ex_cold_flag
            ColdWet_Flag[:, icell] = res.cex_ColdWet_flag
            ColdDry_Flag[:, icell] = res.cex_ColdDry_flag
            HotWet_Flag[:, icell] = res.cex_HotWet_flag
            HotDry_Flag[:, icell] = res.cex_HotDry_flag
            RH_Dry_diff[:, icell] = res.RH_Dry_diff
            RH_Wet_diff[:, icell] = res.RH_Wet_diff
            T_Cold_diff[:, icell] = res.T_Cold_diff
            T_Hot_diff[:, icell] = res.T_Hot_diff
            Tw_HotDry_diff[:, icell] = res.Tw_HotDry_diff
            Tw_HotWet_diff[:, icell] = res.Tw_HotWet_diff
            Tw_ColdDry_diff[:, icell] = res.Tw_ColdDry_diff
            Tw_ColdWet_diff[:, icell] = res.Tw_ColdWet_diff
            thres_RH_Wet[:, icell] = res.thres_RH_Wet
            thres_RH_Dry[:, icell] = res.thres_RH_Dry
            thres_T_Hot[:, icell] = res.thres_T_Hot
            thres_T_Cold[:, icell] = res.thres_T_Cold

            
    Wet = Wet.reshape(nfeatures, ny, nx)
    Dry = Dry.reshape(nfeatures, ny, nx)
    Hot = Hot.reshape(nfeatures, ny, nx)
    Cold = Cold.reshape(nfeatures, ny, nx)
    ColdWet = ColdWet.reshape(nfeatures, ny, nx)
    ColdDry = ColdDry.reshape(nfeatures, ny, nx)
    HotWet = HotWet.reshape(nfeatures, ny, nx)
    HotDry = HotDry.reshape(nfeatures, ny, nx)
    Wet_Flag = Wet_Flag.reshape(ntime, ny, nx)
    Dry_Flag = Dry_Flag.reshape(ntime, ny, nx)
    Hot_Flag = Hot_Flag.reshape(ntime, ny, nx)
    Cold_Flag = Cold_Flag.reshape(ntime, ny, nx)
    ColdWet_Flag = ColdWet_Flag.reshape(ntime, ny, nx)
    ColdDry_Flag = ColdDry_Flag.reshape(ntime, ny, nx)
    HotWet_Flag = HotWet_Flag.reshape(ntime, ny, nx)
    HotDry_Flag = HotDry_Flag.reshape(ntime, ny, nx)
    RH_Dry_diff = RH_Dry_diff.reshape(ntime, ny, nx)
    RH_Wet_diff = RH_Wet_diff.reshape(ntime, ny, nx)
    T_Cold_diff = T_Cold_diff.reshape(ntime, ny, nx)
    T_Hot_diff = T_Hot_diff.reshape(ntime, ny, nx)
    Tw_HotDry_diff = Tw_HotDry_diff.reshape(ntime, ny, nx)
    Tw_HotWet_diff = Tw_HotWet_diff.reshape(ntime, ny, nx)
    Tw_ColdDry_diff = Tw_ColdDry_diff.reshape(ntime, ny, nx)
    Tw_ColdWet_diff = Tw_ColdWet_diff.reshape(ntime, ny, nx)
    thres_RH_Wet = thres_RH_Wet.reshape(ntime, ny, nx)
    thres_RH_Dry = thres_RH_Dry.reshape(ntime, ny, nx)
    thres_T_Hot = thres_T_Hot.reshape(ntime, ny, nx)
    thres_T_Cold = thres_T_Cold.reshape(ntime, ny, nx)
    # 你的 3 个“特征”标签
    feature_names = ["Freq", "exRH", "exT", "exTw"]
    ex_feature_ids = np.arange(len(feature_names), dtype="int8")
    ex_feature_coord = xr.DataArray(ex_feature_ids, dims="feature_id")
    # 用 0/1/2 当坐标，并通过属性告诉含义
    in_dict = {
            'Wet':            (['feature_id', 'y', 'x'], Wet),
            'Dry':            (['feature_id', 'y', 'x'], Dry),
            'Hot':            (['feature_id', 'y', 'x'], Hot),
            'Cold':           (['feature_id', 'y', 'x'], Cold),
            'ColdWet':        (['feature_id', 'y', 'x'], ColdWet),
            'ColdDry':        (['feature_id', 'y', 'x'], ColdDry),
            'HotWet':         (['feature_id', 'y', 'x'], HotWet),
            'HotDry':         (['feature_id', 'y', 'x'], HotDry),
            'Wet_Flag':       (['time',       'y', 'x'], Wet_Flag),
            'Dry_Flag':       (['time',       'y', 'x'], Dry_Flag),
            'Hot_Flag':       (['time',       'y', 'x'], Hot_Flag),
            'Cold_Flag':      (['time',       'y', 'x'], Cold_Flag),
            'ColdWet_Flag':   (['time',       'y', 'x'], ColdWet_Flag),
            'ColdDry_Flag':   (['time',       'y', 'x'], ColdDry_Flag),
            'HotWet_Flag':    (['time',       'y', 'x'], HotWet_Flag),
            'HotDry_Flag':    (['time',       'y', 'x'], HotDry_Flag),
            'RH_Dry_diff':    (['time',       'y', 'x'], RH_Dry_diff),
            'RH_Wet_diff':    (['time',       'y', 'x'], RH_Wet_diff),
            'T_Cold_diff':    (['time',       'y', 'x'], T_Cold_diff),
            'T_Hot_diff':     (['time',       'y', 'x'], T_Hot_diff),
            'Tw_HotDry_diff': (['time',       'y', 'x'], Tw_HotDry_diff),
            'Tw_HotWet_diff': (['time',       'y', 'x'], Tw_HotWet_diff),
            'Tw_ColdDry_diff':(['time',       'y', 'x'], Tw_ColdDry_diff),
            'Tw_ColdWet_diff':(['time',       'y', 'x'], Tw_ColdWet_diff),
            'thres_RH_Wet':   (['time',       'y', 'x'], thres_RH_Wet),
            'thres_RH_Dry':   (['time',       'y', 'x'], thres_RH_Dry),
            'thres_T_Hot':    (['time',       'y', 'x'], thres_T_Hot),
            'thres_T_Cold':   (['time',       'y', 'x'], thres_T_Cold),
            'ex_features':    (['feature_id'          ], feature_names),
        }
    coords = {'y': xarr_ref.y, 'x': xarr_ref.x, 'time': xarr_ref.time, 'feature_id': ex_feature_coord}
    savepath = f"{outdir}/ExtremeAnalysis/Extreme_Events_{caselist[0]}_ref_{caselist[1]}_identified.nc"
    TIO.save_newnc(savepath=savepath, in_dict=in_dict, coords=coords)
    print(f"    ➠ Finished Identifying Extreme Events. Time Spent: {time.time() - t0:.2f} seconds")




# 统计极端事件次数
def CountExtremeEvents(caselist: list[str], *, outdir: str) -> None:
    time0 = time.time()
    minday = 1
    n_jobs = 96
    filepath = f"{outdir}/ExtremeAnalysis/Extreme_Events_{caselist[0]}_ref_{caselist[1]}_identified.nc"
    print(f"    ➠ Loaded extreme events data for {caselist[0]} with reference to {caselist[1]}")
    xarr = TIO.read_newnc(filepath)
    events_infos = {'Wet_Flag': 1, 'Dry_Flag': 1, 'Hot_Flag': 1, 'Cold_Flag': 1,
                    'ColdWet_Flag': 2, 'ColdDry_Flag': 2, 'HotWet_Flag': 2, 'HotDry_Flag': 2}
    in_dict = {}
    for event_name, flagvalue in events_infos.items():
        print(f"    ➠ Analyzing extreme event: {event_name}")
        events_flag = xarr[event_name].values  # (time, y, x)
        ntime, ny, nx = events_flag.shape
        events_flag = events_flag.reshape(ntime, ny * nx)  # (time, cells)
        events = np.full((ntime, ny * nx), np.nan, dtype=np.float32)
        # 关键：生成器必须在 with 内部被消费
        with Parallel(n_jobs=n_jobs, backend="loky", return_as="generator") as parallel:
            gen = parallel(
                delayed(TEEM._worker_identify_extreme_events)(
                    events_flag[:, icell], flagvalue, minday, icell
                )
                for icell in range(ny * nx)
            )
            for icell, res in tqdm(gen, total=ny * nx,
                                   desc=f'Count {event_name} ', unit="cell",
                                   dynamic_ncols=True, leave=False):
                events[:, icell] = res
        events = events.reshape(ntime, ny, nx)
        var_name = event_name.replace('Flag', 'Identified')  # e.g. Hot_Flag -> Hot_Identified
        in_dict[var_name] = (['time', 'y', 'x'], events)
    coords = {'y': xarr.y, 'x': xarr.x, 'time': xarr.time}
    savepath = f"{outdir}/ExtremeAnalysis/Extreme_Events_{caselist[0]}_ref_{caselist[1]}_identified_normalize.nc"
    TIO.save_newnc(savepath=savepath, in_dict=in_dict, coords=coords)
    
    # 季节聚合
    xarr_events = xr.Dataset({name: (dims, data) for name, (dims, data) in in_dict.items()},
                             coords=coords)
    # 逐季求和（DJF/MAM/JJA/SON，DJF 跨年）
    seasonal = xarr_events.resample(time='QS-DEC').sum(dim='time', skipna=True)
    # 2) 给每个季节时间戳标注 season（字符串）与 years（整年；DJF 记到次年）
    season = seasonal['time'].dt.season
    years = seasonal['time'].dt.year + (seasonal['time'].dt.month == 12)
    seasonal = seasonal.assign_coords(
        season=season,                       # 直接传 DataArray，避免歧义
        year=years
    )
    # 3) 把 time 变成 (year, season) 两个维度
    seasonal = (seasonal
        .set_index(time=['year', 'season'])  # MultiIndex: (year, season)
        .unstack('time')                            # 展开成二维
    )
    # 某些 xarray 版本会把层级叫做 time_level_0/1，这里兼容性重命名一下
    rename_map = {}
    if 'time_level_0' in seasonal.dims: rename_map['time_level_0'] = 'year'
    if 'time_level_1' in seasonal.dims: rename_map['time_level_1'] = 'season'
    if rename_map:
        seasonal = seasonal.rename(rename_map)
    # 4) 调整维度顺序为 year, season, y, x
    seasonal = seasonal.transpose('year', 'season', 'y', 'x')
    # ——（可选）提供一个数值季节坐标，便于非 Python 端读取——
    season_map = {'DJF': 1, 'MAM': 2, 'JJA': 3, 'SON': 4}
    if 'season' in seasonal.dims:
        seasonal = seasonal.assign_coords(
            season_id=('season', [season_map[s] for s in seasonal['season'].values])
        )
        # 写上 CF 风格的含义说明
        seasonal['season_id'].attrs.update(
            long_name='season id',
            flag_values=np.array([1, 2, 3, 4], dtype='int8'),
            flag_meanings='DJF MAM JJA SON'
        )
    savepath = f"{outdir}/ExtremeAnalysis/Extreme_Events_{caselist[0]}_ref_{caselist[1]}_seasonal_event_count.nc"
    seasonal.to_netcdf(savepath)
    print(f"    ➠ Saved to {savepath}")
    print(f"    ➠ Finished Counting Extreme Events. Time Spent: {time.time() - time0:.2f} seconds")


def ComputeWetGivenHotProbability(caselist: list[str], *, outdir: str, refcase: str | None = None) -> None:
    """逐网格计算各实验中 Hot 事件发生时 Wet 事件发生的概率。

    P(Wet|Hot) = count(Wet_Flag == 1 and Hot_Flag == 1) / count(Hot_Flag == 1) * 100
    """
    time0 = time.time()
    if refcase is None:
        refcase = caselist[1]

    print("    ➠ Computing grid-wise P(Wet|Hot)...")
    case_results = {}

    for case in caselist:
        filepath = f"{outdir}/ExtremeAnalysis/Extreme_Events_{case}_ref_{refcase}_identified.nc"
        print(f"    ➠ Loaded extreme events data for {case} with reference to {refcase}")
        xarr = TIO.read_newnc(filepath)

        hot_flag = xarr["Hot_Flag"].values == 1
        wet_flag = xarr["Wet_Flag"].values == 1

        hot_count = np.sum(hot_flag, axis=0).astype(np.float32)
        wet_hot_count = np.sum(hot_flag & wet_flag, axis=0).astype(np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            wet_given_hot = wet_hot_count / hot_count * 100.0
        wet_given_hot = wet_given_hot.astype(np.float32)
        wet_given_hot[hot_count == 0] = np.nan

        coords = {"y": xarr.y, "x": xarr.x}
        in_dict = {
            "Wet_given_Hot_probability": (["y", "x"], wet_given_hot),
            "Hot_count": (["y", "x"], hot_count),
            "Wet_and_Hot_count": (["y", "x"], wet_hot_count),
        }
        savepath = f"{outdir}/ExtremeAnalysis/Extreme_Events_Wet_given_Hot_probability_{case}_ref_{refcase}.nc"
        TIO.save_newnc(savepath=savepath, in_dict=in_dict, coords=coords)

        case_results[case] = {
            "probability": wet_given_hot,
            "hot_count": hot_count,
            "wet_hot_count": wet_hot_count,
            "coords": coords,
        }

    if len(caselist) >= 2:
        case0, case1 = caselist[0], caselist[1]
        probability_diff = case_results[case0]["probability"] - case_results[case1]["probability"]
        hot_count_diff = case_results[case0]["hot_count"] - case_results[case1]["hot_count"]
        wet_hot_count_diff = case_results[case0]["wet_hot_count"] - case_results[case1]["wet_hot_count"]
        in_dict = {
            "Wet_given_Hot_probability_diff": (["y", "x"], probability_diff.astype(np.float32)),
            "Hot_count_diff": (["y", "x"], hot_count_diff.astype(np.float32)),
            "Wet_and_Hot_count_diff": (["y", "x"], wet_hot_count_diff.astype(np.float32)),
        }
        savepath = f"{outdir}/ExtremeAnalysis/Extreme_Events_Wet_given_Hot_probability_diff_{case0}_minus_{case1}_ref_{refcase}.nc"
        TIO.save_newnc(savepath=savepath, in_dict=in_dict, coords=case_results[case0]["coords"])

    print(f"    ➠ Finished Computing P(Wet|Hot). Time Spent: {time.time() - time0:.2f} seconds")


def PlotWetGivenHotProbabilityDiff(caselist: list[str],
                                   lon2d: np.ndarray, lat2d: np.ndarray,
                                   lkinfos: dict, outdir: str, figoutdir: str,
                                   refcase: str | None = None) -> None:
    """绘制逐网格 P(Wet|Hot) 的实验差值图。"""
    if refcase is None:
        refcase = caselist[1]

    print("    ➠ Plotting grid-wise P(Wet|Hot) difference...")
    case1, case2 = caselist[0], caselist[1]
    diffpath = f"{outdir}/ExtremeAnalysis/Extreme_Events_Wet_given_Hot_probability_diff_{case1}_minus_{case2}_ref_{refcase}.nc"
    xarr = TIO.read_newnc(diffpath)

    diffdata = xarr["Wet_given_Hot_probability_diff"].values
    oceanmask = lkinfos["ocean"]
    diffdata[oceanmask == 1] = np.nan

    figoutdir_var = f"{figoutdir}/Single"
    os.makedirs(figoutdir_var, exist_ok=True)

    tasks = []
    mapcfg = TPC.mapConfig(levs=[np.linspace(-5, 5, 11), 5], cmap=cmaps.MPL_PuOr.reversed())
    target = "WetGivenHotProbability"
    savepath = f"{figoutdir_var}/{target}_{case1}_minus_{case2}_ref_{refcase}.{FIGFMT}"
    tasks.append((TPAM.plot_categorical_map, (diffdata, target, lon2d, lat2d, savepath,
                                              None, None, lkinfos, mapcfg, None)))

    label = r"P(Wet|Hot) Difference (%)"
    savepath = f"{figoutdir_var}/{target}_{case1}_minus_{case2}_ref_{refcase}_VColorbar.{FIGFMT}"
    tasks.append((TPCB.plot_spatial_cbar_core_V, (mapcfg, 6.0, label, savepath, 14, 18, 0.04, "both", "{:5.1f}")))

    for func, args in tqdm(tasks, total=len(tasks),
                           desc="    ➠ Plot", unit="task",
                           dynamic_ncols=True):
        func(*args)
    print("    All plots done.")



# 计算极端事件强度
def Calculate_seasonal_extreme_Intensity(caselist: list[str], *, outdir: str) -> None:
    time0 = time.time()
    print(f"    ➠ Loaded extreme events data for {caselist[0]} with reference to {caselist[1]}")
    seasons = TU.get_seasons()
    eventslist = ['Wet', 'Dry', 'Hot', 'Cold', 'ColdWet', 'ColdDry', 'HotWet', 'HotDry'] 
    filepath = f"{outdir}/ExtremeAnalysis/Extreme_Events_{caselist[0]}_ref_{caselist[1]}_identified.nc"
    xarr = TIO.read_newnc(filepath)
    filepath = f"{outdir}/ExtremeAnalysis/Extreme_Events_{caselist[0]}_ref_{caselist[1]}_identified_normalize.nc"
    xarr_flag = TIO.read_newnc(filepath)
    exT_Cold = xarr['T_Cold_diff'].values  #(time, y, x)
    exT_Hot = xarr['T_Hot_diff'].values  #(time, y, x)
    exRH_Dry = xarr['RH_Dry_diff'].values  #(time, y, x)
    exRH_Wet = xarr['RH_Wet_diff'].values  #(time, y, x)
    exTw_HotDry = xarr['Tw_HotDry_diff'].values  #(time, y, x)
    exTw_HotWet = xarr['Tw_HotWet_diff'].values  #(time, y, x)
    exTw_ColdDry = xarr['Tw_ColdDry_diff'].values  #(time, y, x)
    exTw_ColdWet = xarr['Tw_ColdWet_diff'].values  #(time, y, x)
    ntime, ny, nx = exT_Cold.shape
    for event in eventslist:
        exRH_Wet_season = np.full((len(seasons), ny, nx), np.nan, dtype=np.float32)
        exT_Hot_season = np.full((len(seasons), ny, nx), np.nan, dtype=np.float32)
        exRH_Dry_season = np.full((len(seasons), ny, nx), np.nan, dtype=np.float32)
        exT_Cold_season = np.full((len(seasons), ny, nx), np.nan, dtype=np.float32)
        exTw_ColdDry_season = np.full((len(seasons), ny, nx), np.nan, dtype=np.float32)
        exTw_ColdWet_season = np.full((len(seasons), ny, nx), np.nan, dtype=np.float32)
        exTw_HotDry_season = np.full((len(seasons), ny, nx), np.nan, dtype=np.float32)
        exTw_HotWet_season = np.full((len(seasons), ny, nx), np.nan, dtype=np.float32)
        for season_idx, season in enumerate(seasons):
            xarr_season = xarr.sel(time=xarr.time.dt.season == season).squeeze(drop=True)
            xarr_flag_season = xarr_flag.sel(time=xarr_flag.time.dt.season == season).squeeze(drop=True)
            exT_Cold = xarr_season['T_Cold_diff'].values  #(time, y, x)
            exRH_Dry = xarr_season['RH_Dry_diff'].values  #(time, y, x)
            exRH_Wet = xarr_season['RH_Wet_diff'].values  #(time, y, x)
            exT_Hot = xarr_season['T_Hot_diff'].values  #(time, y, x)
            exTw_HotDry = xarr_season['Tw_HotDry_diff'].values  #(time, y, x)
            exTw_HotWet = xarr_season['Tw_HotWet_diff'].values  #(time, y, x)
            exTw_ColdDry = xarr_season['Tw_ColdDry_diff'].values  #(time, y, x)
            exTw_ColdWet = xarr_season['Tw_ColdWet_diff'].values  #(time, y, x)
            # 按事件掩膜
            event_loc = xarr_flag_season[f'{event}_Identified'].values
            exT_Hot[event_loc != 1] = np.nan
            exRH_Wet[event_loc != 1] = np.nan 
            exT_Cold[event_loc != 1] = np.nan
            exRH_Dry[event_loc != 1] = np.nan
            exTw_HotDry[event_loc != 1] = np.nan
            exTw_HotWet[event_loc != 1] = np.nan
            exTw_ColdDry[event_loc != 1] = np.nan
            exTw_ColdWet[event_loc != 1] = np.nan
            exT_Cold_season[season_idx, :, :] = np.nanmean(exT_Cold, axis=0).squeeze()
            exRH_Dry_season[season_idx, :, :] = np.nanmean(exRH_Dry, axis=0).squeeze()
            exT_Hot_season[season_idx, :, :] = np.nanmean(exT_Hot, axis=0).squeeze()
            exRH_Wet_season[season_idx, :, :] = np.nanmean(exRH_Wet, axis=0).squeeze()
            exTw_HotDry_season[season_idx, :, :] = np.nanmean(exTw_HotDry, axis=0).squeeze()
            exTw_HotWet_season[season_idx, :, :] = np.nanmean(exTw_HotWet, axis=0).squeeze()
            exTw_ColdDry_season[season_idx, :, :] = np.nanmean(exTw_ColdDry, axis=0).squeeze()
            exTw_ColdWet_season[season_idx, :, :] = np.nanmean(exTw_ColdWet, axis=0).squeeze()
        # 如果还需要“逐年年平均”（对四季再平均）
        exRH_Dry_yearly = np.nanmean(exRH_Dry_season, axis=0).squeeze()
        exT_Cold_yearly = np.nanmean(exT_Cold_season, axis=0).squeeze()
        exRH_Wet_yearly = np.nanmean(exRH_Wet_season, axis=0).squeeze()
        exT_Hot_yearly = np.nanmean(exT_Hot_season, axis=0).squeeze()
        exTw_HotDry_yearly = np.nanmean(exTw_HotDry_season, axis=0).squeeze()
        exTw_HotWet_yearly = np.nanmean(exTw_HotWet_season, axis=0).squeeze()
        exTw_ColdDry_yearly = np.nanmean(exTw_ColdDry_season, axis=0).squeeze()
        exTw_ColdWet_yearly = np.nanmean(exTw_ColdWet_season, axis=0).squeeze()
        # 保存结果  
        in_dict = {
            "exT_Cold_seasonal": (["season", "y", "x"], exT_Cold_season),  # exT_season shape = (4, ny, nx)
            "exRH_Dry_seasonal": (["season", "y", "x"], exRH_Dry_season),
            "exT_Hot_seasonal": (["season", "y", "x"], exT_Hot_season),
            "exRH_Wet_seasonal": (["season", "y", "x"], exRH_Wet_season),
            "exTw_HotDry_seasonal": (["season", "y", "x"], exTw_HotDry_season),
            "exTw_HotWet_seasonal": (["season", "y", "x"], exTw_HotWet_season),
            "exTw_ColdDry_seasonal": (["season", "y", "x"], exTw_ColdDry_season),
            "exTw_ColdWet_seasonal": (["season", "y", "x"], exTw_ColdWet_season),
            # 如果还需要“逐年年平均”（对四季再平均）
            "exT_Cold_yearly":  (["y", "x"], exT_Cold_yearly),            # 年平均就不用 season 轴
            "exRH_Dry_yearly":  (["y", "x"], exRH_Dry_yearly),
            "exT_Hot_yearly": (["y", "x"], exT_Hot_yearly),
            "exRH_Wet_yearly": (["y", "x"], exRH_Wet_yearly),
            "exTw_HotDry_yearly": (["y", "x"], exTw_HotDry_yearly),
            "exTw_HotWet_yearly": (["y", "x"], exTw_HotWet_yearly),
            "exTw_ColdDry_yearly": (["y", "x"], exTw_ColdDry_yearly),
            "exTw_ColdWet_yearly": (["y", "x"], exTw_ColdWet_yearly),
        }
        coords = {
            "y": xarr.y,
            "x": xarr.x,
            **TU.make_season_coords()
        }
        savepath = f"{outdir}/ExtremeAnalysis/Intensity_{event}_exT_exRH_exTw_{caselist[0]}_ref_{caselist[1]}_yseasonmean.nc"
        TIO.save_newnc(savepath=savepath, in_dict=in_dict, coords=coords)
    print(f"    ➠ Finished Calculating Seasonal Extreme Intensity. Time Spent: {time.time() - time0:.2f} seconds")

    # 计算每月平均
    # 提取年份（保持升序且去重）
    years = np.asarray(np.unique(xarr.time.dt.year.values), dtype=int)
    for event in eventslist:
        # 结果数组：(year, season, y, x)
        exT_Cold_yseas = np.full((len(years), len(seasons), ny, nx), np.nan, dtype=np.float32)
        exT_Hot_yseas = np.full((len(years), len(seasons), ny, nx), np.nan, dtype=np.float32)
        exRH_Dry_yseas = np.full((len(years), len(seasons), ny, nx), np.nan, dtype=np.float32)
        exRH_Wet_yseas = np.full((len(years), len(seasons), ny, nx), np.nan, dtype=np.float32)
        exTw_HotDry_yseas = np.full((len(years), len(seasons), ny, nx), np.nan, dtype=np.float32)
        exTw_HotWet_yseas = np.full((len(years), len(seasons), ny, nx), np.nan, dtype=np.float32)
        exTw_ColdDry_yseas = np.full((len(years), len(seasons), ny, nx), np.nan, dtype=np.float32)
        exTw_ColdWet_yseas = np.full((len(years), len(seasons), ny, nx), np.nan, dtype=np.float32)
        for yi, year in enumerate(years):
            for si, season in enumerate(seasons):
                # 子集：该年的该季
                sel_time = (xarr.time.dt.year == year) & (xarr.time.dt.season == season)
                if not sel_time.any():
                    # 该年没有这个季的数据，保持 NaN
                    continue
                xarr_season = xarr.sel(time=sel_time).squeeze(drop=True)
                xarr_flag_season = xarr_flag.sel(time=sel_time).squeeze(drop=True)
                # 取值并按事件掩膜
                exT_Hot = xarr_season['T_Hot_diff'].values  # (time, y, x)
                exT_Cold = xarr_season['T_Cold_diff'].values    # (time, y, x)
                exRH_Wet = xarr_season['RH_Wet_diff'].values  # (time, y, x)
                exRH_Dry = xarr_season['RH_Dry_diff'].values    # (time, y, x)
                exTw_HotDry = xarr_season['Tw_HotDry_diff'].values  # (time, y, x)
                exTw_HotWet = xarr_season['Tw_HotWet_diff'].values  # (time, y, x)
                exTw_ColdDry = xarr_season['Tw_ColdDry_diff'].values  # (time, y, x)
                exTw_ColdWet = xarr_season['Tw_ColdWet_diff'].values  # (time, y, x)
                event_loc = xarr_flag_season[f'{event}_Identified'].values  # (time, y, x)
                # 与你现有逻辑一致：非事件处置 NaN，再对 time 做 nanmean
                exT_Hot[event_loc != 1] = np.nan
                exT_Cold[event_loc != 1] = np.nan
                exRH_Wet[event_loc != 1] = np.nan
                exRH_Dry[event_loc != 1] = np.nan
                exTw_HotDry[event_loc != 1] = np.nan
                exTw_HotWet[event_loc != 1] = np.nan
                exTw_ColdDry[event_loc != 1] = np.nan
                exTw_ColdWet[event_loc != 1] = np.nan
                exT_Hot_yseas[yi, si, :, :] = np.nanmean(exT_Hot, axis=0).squeeze()
                exT_Cold_yseas[yi, si, :, :] = np.nanmean(exT_Cold, axis=0).squeeze()
                exRH_Wet_yseas[yi, si, :, :] = np.nanmean(exRH_Wet, axis=0).squeeze()
                exRH_Dry_yseas[yi, si, :, :] = np.nanmean(exRH_Dry, axis=0).squeeze()
                exTw_HotDry_yseas[yi, si, :, :] = np.nanmean(exTw_HotDry, axis=0).squeeze()
                exTw_HotWet_yseas[yi, si, :, :] = np.nanmean(exTw_HotWet, axis=0).squeeze()
                exTw_ColdDry_yseas[yi, si, :, :] = np.nanmean(exTw_ColdDry, axis=0).squeeze()
                exTw_ColdWet_yseas[yi, si, :, :] = np.nanmean(exTw_ColdWet, axis=0).squeeze()
        # 如果还需要“逐年年平均”（对四季再平均），与原逻辑一致在季节维做 nanmean
        exT_Hot_yearly = np.nanmean(exT_Hot_yseas, axis=1)  # -> (year, y, x)
        exRH_Wet_yearly = np.nanmean(exRH_Wet_yseas, axis=1)  # -> (year, y, x)
        exT_Cold_yearly = np.nanmean(exT_Cold_yseas, axis=1)    # -> (year, y, x)
        exRH_Dry_yearly = np.nanmean(exRH_Dry_yseas, axis=1)    # -> (year, y, x)
        exTw_HotDry_yearly = np.nanmean(exTw_HotDry_yseas, axis=1)  # -> (year, y, x)
        exTw_HotWet_yearly = np.nanmean(exTw_HotWet_yseas, axis=1)  # -> (year, y, x)
        exTw_ColdDry_yearly = np.nanmean(exTw_ColdDry_yseas, axis=1)  # -> (year, y, x)
        exTw_ColdWet_yearly = np.nanmean(exTw_ColdWet_yseas, axis=1)  # -> (year, y, x)
        # 保存结果
        in_dict = {
            # 主需求：按年×季节
            "exT_Cold_year_season": (["year", "season", "y", "x"], exT_Cold_yseas),
            "exT_Hot_year_season": (["year", "season", "y", "x"], exT_Hot_yseas),
            "exRH_Dry_year_season": (["year", "season", "y", "x"], exRH_Dry_yseas),
            "exRH_Wet_year_season": (["year", "season", "y", "x"], exRH_Wet_yseas),
            "exTw_HotDry_year_season": (["year", "season", "y", "x"], exTw_HotDry_yseas),
            "exTw_HotWet_year_season": (["year", "season", "y", "x"], exTw_HotWet_yseas),
            "exTw_ColdDry_year_season": (["year", "season", "y", "x"], exTw_ColdDry_yseas),
            "exTw_ColdWet_year_season": (["year", "season", "y", "x"], exTw_ColdWet_yseas),
            # 可选：逐年年平均（如果你需要保留）
            "exRH_Dry_yearly": (["year", "y", "x"], exRH_Dry_yearly),
            "exRH_Wet_yearly": (["year", "y", "x"], exRH_Wet_yearly),
            "exT_Cold_yearly": (["year", "y", "x"], exT_Cold_yearly),
            "exT_Hot_yearly": (["year", "y", "x"], exT_Cold_yearly),
            "exTw_HotDry_yearly": (["year", "y", "x"], exTw_HotDry_yearly),
            "exTw_HotWet_yearly": (["year", "y", "x"], exTw_HotWet_yearly),
            "exTw_ColdDry_yearly": (["year", "y", "x"], exTw_ColdDry_yearly),
            "exTw_ColdWet_yearly": (["year", "y", "x"], exTw_ColdWet_yearly),
        }
        coords = {
            "year": years,
            "season": seasons,
            "y": xarr.y,
            "x": xarr.x,
            **TU.make_season_coords()  # 如果该函数已提供 season 坐标/属性也可直接复用
        }
        savepath = f"{outdir}/ExtremeAnalysis/Intensity_{event}_exT_exRH_exTw_{caselist[0]}_ref_{caselist[1]}_seasonmean.nc"
        TIO.save_newnc(savepath=savepath, in_dict=in_dict, coords=coords)
    
    # 保存每日强度
    for event in eventslist:
        event_loc = xarr_flag[f'{event}_Identified'].values
        exT_Hot = xarr['T_Hot_diff'].values  #(time, y, x)
        exRH_Wet = xarr['RH_Wet_diff'].values  #(time, y, x)
        exT_Cold = xarr['T_Cold_diff'].values  #(time, y, x)
        exRH_Dry = xarr['RH_Dry_diff'].values  #(time, y, x)
        exTw_HotDry = xarr['Tw_HotDry_diff'].values  #(time, y, x)
        exTw_HotWet = xarr['Tw_HotWet_diff'].values  #(time, y, x)
        exTw_ColdDry = xarr['Tw_ColdDry_diff'].values  #(time, y, x)
        exTw_ColdWet = xarr['Tw_ColdWet_diff'].values  #(time, y, x)
        exT_Hot[event_loc != 1] = np.nan
        exRH_Wet[event_loc != 1] = np.nan 
        exT_Cold[event_loc != 1] = np.nan
        exRH_Dry[event_loc != 1] = np.nan
        exTw_HotDry[event_loc != 1] = np.nan
        exTw_HotWet[event_loc != 1] = np.nan
        exTw_ColdDry[event_loc != 1] = np.nan
        exTw_ColdWet[event_loc != 1] = np.nan
        exT_Cold_daily = exT_Cold
        exRH_Dry_daily = exRH_Dry
        exT_Hot_daily = exT_Hot
        exRH_Wet_daily = exRH_Wet
        exTw_HotDry_daily = exTw_HotDry
        exTw_HotWet_daily = exTw_HotWet
        exTw_ColdDry_daily = exTw_ColdDry
        exTw_ColdWet_daily = exTw_ColdWet
        # 保存结果  
        in_dict = {
            "exT_Cold_daily": (["time", "y", "x"], exT_Cold_daily),  # exT_season shape = (4, ny, nx)   
            "exRH_Dry_daily": (["time", "y", "x"], exRH_Dry_daily),
            "exT_Hot_daily": (["time", "y", "x"], exT_Hot_daily),
            "exRH_Wet_daily": (["time", "y", "x"], exRH_Wet_daily),
            "exTw_HotDry_daily": (["time", "y", "x"], exTw_HotDry_daily),
            "exTw_HotWet_daily": (["time", "y", "x"], exTw_HotWet_daily),
            "exTw_ColdDry_daily": (["time", "y", "x"], exTw_ColdDry_daily),
            "exTw_ColdWet_daily": (["time", "y", "x"], exTw_ColdWet_daily),
        }
        coords = {
            "y": xarr.y,
            "x": xarr.x,
            "time": xarr.time,
        }
        savepath = f"{outdir}/ExtremeAnalysis/Intensity_{event}_exT_exRH_exTw_{caselist[0]}_ref_{caselist[1]}_daily.nc"
        TIO.save_newnc(savepath=savepath, in_dict=in_dict, coords=coords)
    
    print(f"    ➠ Finished Calculating Seasonal Extreme Intensity. Time Spent: {time.time() - time0:.2f} seconds")



# 极端事件显著性检验核心函数
def ExtremeEvents_SignificanceCore(
        arr1: xr.DataArray, arr2: xr.DataArray,
        *, event: str, target: str, var:str, outdir: str, caselist: list, checkmethod: str, alternative: str,
        alpha_ci: float, clt_n: int, n_sample: int, ci: float,
        center_null: bool, random_state: int, n_jobs: int,
        is_seasonal: bool = True) -> None:
    """
    通用的极端事件显著性检验核心函数
    对极端事件进行分析
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
    # print(arr1.dims, arr2.dims)
    if is_seasonal:
        seasons = TU.get_seasons()
        season_days = TU.get_season_days()
        nyear, _, ny, nx = arr1.shape
        p_arr = np.full((ny, nx), np.nan)
        mean_diff_arr = np.full_like(p_arr, np.nan)
        effect_size_arr = np.full_like(p_arr, np.nan)
        method_arr = np.full_like(p_arr, np.nan)
        p_fdr_ = np.full_like(p_arr, np.nan)
        rej_fdr_ = np.full_like(p_arr, 0, dtype=np.int8)
        ci_low_arr = np.full_like(p_arr, np.nan)
        ci_high_arr = np.full_like(p_arr, np.nan)
        RC_overall = np.full_like(p_arr, np.nan)
        RC_anomaly = np.full_like(p_arr, np.nan)

        for season_idx, season in enumerate(seasons):
            xarr1_season = arr1.sel(season=season).squeeze(drop=True)
            xarr2_season = arr2.sel(season=season).squeeze(drop=True)
            arr1_check = xarr1_season.values
            arr2_check = xarr2_season.values

            # 显著性检验
            p_out, mean_diff_out, effect_size_out, method_out, ci_low_out, ci_high_out = TST.SignificanceTest(
                arr1_check, arr2_check, checkmethod=checkmethod,
                alternative=alternative, alpha_ci=alpha_ci, clt_n=clt_n,
                n_sample=n_sample, ci=ci, info=f'{season} ',
                center_null=center_null, random_state=random_state, n_jobs=n_jobs)

            p_arr[ :, :] = p_out
            mean_diff_arr[ :, :] = mean_diff_out
            effect_size_arr[ :, :] = effect_size_out
            method_arr[ :, :] = method_out
            ci_low_arr[ :, :] = ci_low_out
            ci_high_arr[ :, :] = ci_high_out

            # FDR 校正
            p_ravel = p_out.ravel()
            mask = np.isfinite(p_ravel)
            rej_fdr = np.full_like(p_ravel, 0, dtype=np.int8)
            p_fdr = np.full_like(p_ravel, np.nan, dtype=float)
            rej_sub, p_sub, _, _ = multipletests(p_ravel[mask], alpha=fdr_alpha, method="fdr_bh")
            rej_fdr[mask] = rej_sub.astype(np.int8)
            p_fdr[mask] = p_sub
            rej_fdr_[ :, :] = rej_fdr.reshape((ny, nx))
            p_fdr_[ :, :] = p_fdr.reshape((ny, nx))

            # 相对贡献
            RC_overall[ :, :], RC_anomaly[ :, :] = TST.RelativeContribution(
                arr1_check, arr2_check, time_axis=0)

            in_dict = {
                'p_value': [["y", "x"], p_arr],
                'mean_diff': [["y", "x"], mean_diff_arr],
                'effect_size': [["y", "x"], effect_size_arr],
                'checkmethod': [["y", "x"], method_arr],
                'p_fdr': [["y", "x"], p_fdr_],
                'rejected': [["y", "x"], rej_fdr_],
                'ci_low': [["y", "x"], ci_low_arr],
                'ci_high': [["y", "x"], ci_high_arr],
                'RC_overall': [["y", "x"], RC_overall],
                'RC_anomaly': [["y", "x"], RC_anomaly],
                caselist[0]: [["year", "y", "x"], arr1_check],
                caselist[1]: [["year", "y", "x"], arr2_check],
            }
            coords = {"y": arr1.y, "x": arr1.x, "season": arr1.season, "year": arr1.year}
            outpath=f'{outdir}/ExtremeAnalysis/Significance_ExtremeEvent_{target}_{caselist[0]}-{caselist[1]}_{event}_{var}_seasonal_{season}_{checkmethod}.nc'
            TIO.save_newnc(savepath=outpath, in_dict=in_dict, coords=coords)

    else:
        # 年尺度检验
        arr1_check, arr2_check = arr1.values, arr2.values
        p_arr, mean_diff_arr, effect_size_arr, method_arr, ci_low_arr, ci_high_arr = TST.SignificanceTest(
            arr1_check, arr2_check, checkmethod=checkmethod,
            alternative=alternative, alpha_ci=alpha_ci, clt_n=clt_n,
            n_sample=n_sample, ci=ci, center_null=center_null,
            random_state=random_state, n_jobs=n_jobs)

        p_ravel = p_arr.ravel()
        mask = np.isfinite(p_ravel)
        rej_fdr = np.full_like(p_ravel, 0, dtype=np.int8)
        p_fdr = np.full_like(p_ravel, np.nan, dtype=float)
        rej_sub, p_sub, _, _ = multipletests(p_ravel[mask], alpha=fdr_alpha, method="fdr_bh")
        rej_fdr[mask] = rej_sub.astype(np.int8)
        p_fdr[mask] = p_sub
        rej_fdr = rej_fdr.reshape(p_arr.shape)
        p_fdr = p_fdr.reshape(p_arr.shape)

        RC_overall, RC_anomaly = TST.RelativeContribution(arr1_check, arr2_check, time_axis=0)
        in_dict = {
            'p_value': [["y", "x"], p_arr],
            'mean_diff': [["y", "x"], mean_diff_arr],
            'effect_size': [["y", "x"], effect_size_arr],
            'checkmethod': [["y", "x"], method_arr],
            'p_fdr': [["y", "x"], p_fdr],
            'rejected': [["y", "x"], rej_fdr],
            'ci_low': [["y", "x"], ci_low_arr],
            'ci_high': [["y", "x"], ci_high_arr],
            'RC_overall': [["y", "x"], RC_overall],
            'RC_anomaly': [["y", "x"], RC_anomaly],
            caselist[0]: [["year", "y", "x"], arr1_check],
            caselist[1]: [["year", "y", "x"], arr2_check],
        }
        coords = {"y": arr1.y, "x": arr1.x, "year": arr1.year}
        outpath=f'{outdir}/ExtremeAnalysis/Significance_ExtremeEvent_{target}_{caselist[0]}-{caselist[1]}_{event}_{var}_yearly_{checkmethod}.nc'
        TIO.save_newnc(savepath=outpath, in_dict=in_dict, coords=coords)



# 极端事件频率显著性分析
def SignificanceOfExtremeEvents_Freq(xarr1, xarr2, events, *, caselist=['Lake','NoLake'], outdir: str):
    """极端事件频率显著性分析"""
    target = 'Freq'
    var = 'Freq'
    params = dict(
        checkmethod="Wilcoxon_signed-rank_test",
        alternative="two-sided",
        alpha_ci=0.1,
        clt_n=30,
        n_sample=10000,
        ci=0.9,
        center_null=True,
        random_state=666,
        n_jobs=96,
        caselist=caselist
    )
    suffix = '_Identified'
    for event in events:
        print(f'    ---- Processing event: {event} ----')
        arr1 = xarr1[event + suffix].copy()
        arr2 = xarr2[event + suffix].copy()
        # 季节性
        ExtremeEvents_SignificanceCore(arr1, arr2, **params, event=event,
                          target=target, var=var, outdir=outdir, is_seasonal=True)
        # 年尺度
        arr1_y = xarr1.sum("season")[event + suffix].copy()
        arr2_y = xarr2.sum("season")[event + suffix].copy()
        ExtremeEvents_SignificanceCore(arr1_y, arr2_y, **params, event=event,
                          target=target, var=var, outdir=outdir, is_seasonal=False)



# 极端事件强度显著性分析
def SignificanceOfExtremeEvents_Intensity(xarr1, xarr2, event, event_vars, *, caselist=['Lake','NoLake'], outdir: str):
    """极端事件强度显著性分析"""
    target = 'Intensity'
     # 显著性检验参数
    params = dict(
        checkmethod="Wilcoxon_signed-rank_test",   # "auto", "Paired_t-test", "Wilcoxon_signed-rank_test", "Paired_permutation_test","Paired_bootstrap", "Sign_test", "Cliffs_delta", "Mann-Whitney_U_test", "Cramer-von_Mises_test",
        alternative="two-sided",                   # ["two-sided", "greater", "less"]
        alpha_ci=0.05,                             # 显著性水平
        clt_n=30,                                  # 中心极限定理样本量
        n_sample=10000,                            # 重抽样次数
        ci=0.95,                                   # 置信区间
        center_null=True,                          # 是否中心化零假设
        random_state=666,                          # 随机种子
        n_jobs=96,                                 # 并行数
        caselist=caselist                         # 情景列表
    )
    print(f'    ---- Processing event: {event} ----')
    for var in event_vars:
        arr1 = xarr1[var + "_year_season"].copy()
        arr2 = xarr2[var + "_year_season"].copy()
        ExtremeEvents_SignificanceCore(arr1, arr2, **params, event=event,
                          target=target, var=var, outdir=outdir, is_seasonal=True)

        arr1_y = xarr1[var + "_yearly"].copy()
        arr2_y = xarr2[var + "_yearly"].copy()
        ExtremeEvents_SignificanceCore(arr1_y, arr2_y, **params, event=event,
                           target=target, var=var, outdir=outdir,is_seasonal=False)




def compute_hotwet_probability(caselist, checkmethod, OutDir, lkinfos):
    """计算极端事件联合发生概率变化"""
    print("    ➠ Computing Extreme Event Joint Occurrence Probability Changes...")
    target = "Freq"
    oceanmask = lkinfos['ocean']
    inlandtotal = np.sum(oceanmask == 0)

    # Helper to load Hot, Wet, HotWet
    def load_event(event):
        outpath=f'{OutDir}/ExtremeAnalysis/Significance_ExtremeEvent_{target}_{caselist[0]}-{caselist[1]}_{event}_{target}_yearly_{checkmethod}.nc'
        xarr = TIO.read_newnc(outpath)
        rejected, diffsign = TDP.preparation_for_probability(xarr, checkmethod, lkinfos)
        return {"sign": rejected, "diff": diffsign}

    Hot     = load_event("Hot")
    Wet     = load_event("Wet")
    HotWet  = load_event("HotWet")

    # ========= Define masks =========
    Hot_up     = (Hot["sign"] == 1) & (Hot["diff"] == 1)
    Hot_down   = (Hot["sign"] == 1) & (Hot["diff"] == -1)
    Hot_zero   = (Hot["sign"] == 1) & (Hot["diff"] == 0)
    Hot_none   = (Hot["sign"] == 0)

    Wet_up     = (Wet["sign"] == 1) & (Wet["diff"] == 1)
    Wet_down   = (Wet["sign"] == 1) & (Wet["diff"] == -1)
    Wet_zero   = (Wet["sign"] == 1) & (Wet["diff"] == 0)
    Wet_none   = (Wet["sign"] == 0)

    HotWet_total = (HotWet["sign"] == 1)
    HotWet_up   = (HotWet["sign"] == 1) & (HotWet["diff"] == 1)
    HotWet_down = (HotWet["sign"] == 1) & (HotWet["diff"] == -1)
    HotWet_zero = (HotWet["sign"] == 1) & (HotWet["diff"] == 0)
    HotWet_none = (HotWet["sign"] == 0)
    print("    Debug Info:")
    print("    Hot_up count:", np.sum(Hot_up))
    print("    Hot_down count:", np.sum(Hot_down))
    print("    Hot_zero count:", np.sum(Hot_zero))
    print("    Hot_none count:", np.sum(Hot_none))
    print("    Wet_up count:", np.sum(Wet_up))
    print("    Wet_down count:", np.sum(Wet_down))
    print("    Wet_zero count:", np.sum(Wet_zero))
    print("    Wet_none count:", np.sum(Wet_none))
    print("    HotWet_total count:", np.sum(HotWet_total))
    print("    HotWet_up count:", np.sum(HotWet_up))
    print("    HotWet_down count:", np.sum(HotWet_down))
    print("    HotWet_zero count:", np.sum(HotWet_zero))
    print("    HotWet_none count:", np.sum(HotWet_none))

    # group into dicts for looping
    Hot_states = {
        "Hot_up": Hot_up,
        "Hot_down": Hot_down,
        "Hot_None": Hot_none,
        # "Hot_zero": Hot_zero
    }
    Wet_states = {
        "Wet_up": Wet_up,
        "Wet_down": Wet_down,
        "Wet_None": Wet_none,
        # "Wet_zero": Wet_zero
    }
    HotWet_states = {
        "Total": HotWet_total,
        "Up": HotWet_up,
        "Down": HotWet_down,
        "None": HotWet_none,
        # "Zero": HotWet_zero
    }

    # create dataframes
    index = ["Hot_up",  "Hot_down", "Hot_None",]  #"Hot_zero",
    columns = ["Wet_up",  "Wet_down", "Wet_None",] #"Wet_zero",

    df_total = pd.DataFrame(index=index, columns=columns)
    df_up    = pd.DataFrame(index=index, columns=columns)
    df_down  = pd.DataFrame(index=index, columns=columns)
    df_none  = pd.DataFrame(index=index, columns=columns)
    # df_zero  = pd.DataFrame(index=index, columns=columns)

    # total
    # total_count = 0
    # for hot_key, hot_mask in Hot_states.items():
    #     for wet_key, wet_mask in Wet_states.items():
    #         combined =  hot_mask & wet_mask
    #         count = np.sum(combined)
    #         total_count += count

    total_count = HotWet_states["Total"].sum()
    print("Total land grid cells:", total_count)
    for hot_key, hot_mask in Hot_states.items():
        for wet_key, wet_mask in Wet_states.items():
            hw_mask = HotWet_states["Total"]
            combined =  hot_mask & wet_mask & hw_mask
            count = np.sum(combined)

            p = count / total_count * 100 if total_count > 0 else np.nan
            df_total.loc[hot_key, wet_key] = p

    for hot_key, hot_mask in Hot_states.items():
        for wet_key, wet_mask in Wet_states.items():
            total_count = np.sum(hot_mask & wet_mask)
            
            # up
            hw_mask = HotWet_states["Up"]
            combined =  hot_mask & wet_mask & hw_mask
            count = np.sum(combined)
            p_up = count / total_count * 100 if total_count > 0 else np.nan
            df_up.loc[hot_key, wet_key] = p_up
            print(f"Debug: Hot-Wet Up | hot_key={hot_key}, wet_key={wet_key}, total_count={total_count}, up_count={count}, p_up={p_up}")
            
            # down
            hw_mask = HotWet_states["Down"]
            combined = hot_mask & wet_mask & hw_mask
            count = np.sum(combined)
            p_down = count / total_count * 100 if total_count > 0 else np.nan
            df_down.loc[hot_key, wet_key] = p_down
            print(f"Debug: Hot-Wet Down | hot_key={hot_key}, wet_key={wet_key}, total_count={total_count}, down_count={count}, p_down={p_down}")
            
            # none
            hw_mask = HotWet_states["None"]
            combined = hot_mask & wet_mask & hw_mask
            count = np.sum(combined)
            p_none = count / total_count * 100 if total_count > 0 else np.nan
            df_none.loc[hot_key, wet_key] = p_none
            print(f"Debug: Hot-Wet None | hot_key={hot_key}, wet_key={wet_key}, total_count={total_count}, none_count={count}, p_none={p_none}")
            
            # hw_mask = HotWet_states["Zero"]
            # combined = hot_mask & wet_mask & hw_mask
            # count = np.sum(combined)
            # p_zero = count / total_count * 100 if total_count > 0 else np.nan
            # # df_none.loc[hot_key, wet_key] = p_none + p_zero
            # df_zero.loc[hot_key, wet_key] = p_zero

    print("      [OK] Probability tables computed.")
    print("Total land grid cells used for probability calculation:", total_count)
    print("\n\nHotWet_total table:")
    print(df_total)
    print("\n\nHotWet_up table:")
    print(df_up)
    # print("\n\nHotWet_zero table:")
    # print(df_zero)
    print("\n\nHotWet_down table:")
    print(df_down)
    print("\n\nHotWet_none table:")
    print(df_none)


    # ======== Save results ========
    df_total.to_csv(f"{OutDir}/ExtremeAnalysis/Excel/New_Probability_HotWet_total_{caselist[0]}_minus_{caselist[1]}_{checkmethod}.csv")
    df_up.to_csv(f"{OutDir}/ExtremeAnalysis/Excel/New_Probability_HotWet_up_{caselist[0]}_minus_{caselist[1]}_{checkmethod}.csv")
    df_down.to_csv(f"{OutDir}/ExtremeAnalysis/Excel/New_Probability_HotWet_down_{caselist[0]}_minus_{caselist[1]}_{checkmethod}.csv")
    df_none.to_csv(f"{OutDir}/ExtremeAnalysis/Excel/New_Probability_HotWet_none_{caselist[0]}_minus_{caselist[1]}_{checkmethod}.csv")
    # df_zero.to_csv(f"{OutDir}/ExtremeAnalysis/Excel/New_Probability_HotWet_zero_{caselist[0]}_minus_{caselist[1]}_{checkmethod}.csv")
    print("All Probability tables computed OK.")

    df_big = pd.DataFrame(columns=["Hot_State", "Wet_State", "HotWet_Up(%)", "HotWet_Down(%)", "HotWet_None(%)"])
    # Initialize an empty list to store rows
    rows = []
    # Loop through the hot and wet states
    for hot_key in Hot_states.keys():
        for wet_key in Wet_states.keys():
            up_p = df_up.loc[hot_key, wet_key]
            down_p = df_down.loc[hot_key, wet_key]
            none_p = df_none.loc[hot_key, wet_key]
            
            # Append the data as a dictionary
            rows.append({
                "Hot_State": hot_key,
                "Wet_State": wet_key,
                "HotWet_Up(%)": up_p,
                "HotWet_Down(%)": down_p,
                "HotWet_None(%)": none_p
            })

    # Convert the list of rows into a DataFrame
    df_big = pd.DataFrame(rows)

    # Save the summary table
    df_big.to_csv(f"{OutDir}/ExtremeAnalysis/Excel/New_Probability_HotWet_Summary_{caselist[0]}_minus_{caselist[1]}_{checkmethod}.csv")
    print("Summary Probability table saved OK.")



# 区域极端事件绘图
def Plot_ExtremeEvents_Freq(
        event: str, event_vars: list, caselist: list[str], 
        lon2d: np.ndarray, lat2d: np.ndarray,
        checkmethod: str, OutDir: str, FigOutDir: str,
        lkinfos: dict, onlysig: bool = True) -> None:
    """区域极端事件绘图"""

    target = "Freq"
    DPI = TPC.DPI_medium
    FIGFMT = TPC.FIGFMT
    seasons = TU.get_seasons()
    var_info = TYCM.ExtremeEvent_Infos(event)
    seas_levels = TYCM.Seasonal_ExtremeEvents_Freq_Cmap(event)
    FigOutDir_var = f'{FigOutDir}/Single'
    os.makedirs(FigOutDir_var, exist_ok=True)
    # 绘制区域差异图
    tasks = []
    season_diffcfg = TPC.mapConfig(levs=seas_levels['diff_maplevs_seasonal'], cmap=seas_levels['diff_cmap'])
    season_rccfg = TPC.mapConfig(levs=seas_levels['rc_maplevs_seasonal'], cmap=seas_levels['rc_cmap'])  
    annual_diffcfg = TPC.mapConfig(levs=seas_levels['diff_maplevs_annual'], cmap=seas_levels['diff_cmap'])
    annual_rccfg = TPC.mapConfig(levs=seas_levels['rc_maplevs_annual'], cmap=seas_levels['rc_cmap'])  
    boxcfg = TPC.boxConfig(diff_boxlevs=seas_levels['diff_boxlevs'], rc_boxlevs=seas_levels['rc_boxlevs'])
    varInfo = TPC.varInfo(longname=var_info['longname'], unit=var_info['bunit'], abbr=var_info['abbr'])
    suffix = 'onlysig' if onlysig else 'all'
    xarr_seasons = {}
    for season in seasons:
        path_seasonal = f'{OutDir}/ExtremeAnalysis/Significance_ExtremeEvent_{target}_{caselist[0]}-{caselist[1]}_{event}_{target}_seasonal_{season}_{checkmethod}.nc'
        xarr_seasonal= TIO.read_newnc(path_seasonal)
        xarr_seasons[season] = xarr_seasonal
        # 绘制季节差异图（含显著性标记) 
        savepath = f'{FigOutDir_var}/{target}_Diff_Map_{event}_{season}_{suffix}_{checkmethod}.{FIGFMT}'
        tasks.append((TPAM.plot_spatial_diffmap_withsign, (xarr_seasonal, varInfo.abbr, season, target, lon2d, lat2d, 
                                                           checkmethod, lkinfos, season_diffcfg, onlysig, savepath)))
        # 绘制区域相对贡献图（含显著性标记）
        # savepath = f'{FigOutDir_var}/{target}_RC_Map_{event}_{season}_{suffix}_{checkmethod}.{FIGFMT}'
        # tasks.append((TPAM.plot_spatial_rcmap_withsign, (xarr_seasonal, varInfo.abbr, season, target, lon2d, lat2d,
        #                                                  checkmethod, lkinfos, season_rccfg, onlysig, savepath)))
    # 绘制季节差异箱线图和相对贡献箱线图
    diff_df, rc_df, suffix = TDP.prepara_for_boxplot_seasonal(xarr_seasons, lkinfos, checkmethod, onlysig=True)
    tasks.append((TPB.plot_diff_boxplot, (diff_df, seasons, varInfo.abbr, target, checkmethod, FigOutDir_var, boxcfg, varInfo, suffix)))
    # tasks.append((TPB.plot_rc_boxplot, (rc_df, seasons, varInfo.abbr, target, checkmethod, FigOutDir_var, boxcfg, varInfo, suffix)))

    path_annual = f'{OutDir}/ExtremeAnalysis/Significance_ExtremeEvent_{target}_{caselist[0]}-{caselist[1]}_{event}_{target}_yearly_{checkmethod}.nc'
    xarr_annual = TIO.read_newnc(path_annual)
    # 绘制年度差异图（含显著性标记）
    savepath = f'{FigOutDir_var}/{target}_Diff_Map_{event}_Annual_{suffix}_{checkmethod}.{FIGFMT}'
    tasks.append((TPAM.plot_spatial_diffmap_withsign, (xarr_annual, varInfo.abbr, "Annual", target, lon2d, lat2d, 
                                                       checkmethod, lkinfos, annual_diffcfg, onlysig, savepath)))
    # 绘制区域相对贡献图（含显著性标记）
    # savepath = f'{FigOutDir_var}/{target}_RC_Map_{event}_Annual_{suffix}_{checkmethod}.{FIGFMT}'
    # tasks.append((TPAM.plot_spatial_rcmap_withsign, (xarr_annual, varInfo.abbr, "Annual", target, lon2d, lat2d, 
    #                                                  checkmethod, lkinfos, annual_rccfg, onlysig, savepath)))
    # 绘制colorbar
    label = rf"{varInfo.abbr} ({varInfo.unit})"
    savepath = f'{FigOutDir_var}/{target}_Diff_Map_{event}_Annual_Colorbar_{suffix}_{checkmethod}.{FIGFMT}'
    tasks.append((TPCB.plot_spatial_cbar_core_V, (annual_diffcfg, 4, label, savepath, 18, 18, 0.06, 'both','{:5.1f}'))) 
    savepath = f'{FigOutDir_var}/{target}_Diff_Map_{event}_Seasonal_Colorbar_{suffix}_{checkmethod}.{FIGFMT}'
    tasks.append((TPCB.plot_spatial_cbar_core_V, (season_diffcfg, 4, label, savepath, 18, 18, 0.06, 'both','{:5.1f}'))) 
    # label = "Relative Contribution (%)"
    # savepath = f'{FigOutDir_var}/{target}_RC_Map_{event}_Annual_Colorbar_{suffix}_{checkmethod}.{FIGFMT}'
    # tasks.append((TPCB.plot_spatial_cbar_core_V, (annual_rccfg, 4, label, savepath, 18, 18, 0.06, 'both','{:5.1f}'))) 
    # savepath = f'{FigOutDir_var}/{target}_RC_Map_{event}_Seasonal_Colorbar_{suffix}_{checkmethod}.{FIGFMT}'
    # tasks.append((TPCB.plot_spatial_cbar_core_V, (season_rccfg, 4, label, savepath, 18, 18, 0.06, 'both','{:5.1f}'))) 
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




# 区域极端事件绘图
def Plot_ExtremeEvents_Intensity(
    event: str, event_vars: list, caselist : list[str], 
    lon2d: np.ndarray, lat2d: np.ndarray,
    checkmethod: str, OutDir: str, FigOutDir: str,
    lkinfos: dict, onlysig: bool = True) -> None:
    """区域极端事件绘图"""

    target = "Intensity"
    seasons = TU.get_seasons()
    for varname in event_vars:
        var_info = TYCM.ExtremeEvent_Infos(varname)
        seas_levels = TYCM.Seasonal_ExtremeEvents_Intensity_Cmap(varname)
        FigOutDir_var = f'{FigOutDir}/Single'
        os.makedirs(FigOutDir_var, exist_ok=True)
        # 绘制区域差异图
        tasks = []
        season_diffcfg = TPC.mapConfig(levs=seas_levels['diff_maplevs_seasonal'], cmap=seas_levels['diff_cmap'])
        season_rccfg = TPC.mapConfig(levs=seas_levels['rc_maplevs_seasonal'], cmap=seas_levels['rc_cmap'])  
        annual_diffcfg = TPC.mapConfig(levs=seas_levels['diff_maplevs_annual'], cmap=seas_levels['diff_cmap'])
        annual_rccfg = TPC.mapConfig(levs=seas_levels['rc_maplevs_annual'], cmap=seas_levels['rc_cmap'])  
        boxcfg = TPC.boxConfig(diff_boxlevs=seas_levels['diff_boxlevs'], rc_boxlevs=seas_levels['rc_boxlevs'])
        varInfo = TPC.varInfo(longname=var_info['longname'], unit=var_info['bunit'], abbr=var_info['abbr'])
        suffix = f"onlysig" if onlysig else "all"
        xarr_seasons = {}
        for season in seasons:
            path_seasonal = f'{OutDir}/ExtremeAnalysis/Significance_ExtremeEvent_{target}_{caselist[0]}-{caselist[1]}_{event}_{varname}_seasonal_{season}_{checkmethod}.nc'
            xarr_seasonal= TIO.read_newnc(path_seasonal)
            xarr_seasons[season] = xarr_seasonal
            # 绘制季节差异图（含显著性标记) 
            savepath = f'{FigOutDir_var}/{target}_Diff_Map_{event}_{season}_{suffix}_{checkmethod}.{FIGFMT}'
            tasks.append((TPAM.plot_spatial_diffmap_withsign, (xarr_seasonal, varInfo.abbr, season, target, lon2d, lat2d, 
                                                               checkmethod, lkinfos, season_diffcfg, onlysig, savepath)))
            # 绘制区域相对贡献图（含显著性标记）
            savepath = f'{FigOutDir_var}/{target}_RC_Map_{event}_{season}_{suffix}_{checkmethod}.{FIGFMT}'
            tasks.append((TPAM.plot_spatial_rcmap_withsign, (xarr_seasonal, varInfo.abbr, season, target, lon2d, lat2d, 
                                                             checkmethod, lkinfos, season_rccfg, onlysig, savepath)))

        # 绘制季节差异箱线图和相对贡献箱线图
        diff_df, rc_df, suffix = TDP.prepara_for_boxplot_seasonal(xarr_seasons, lkinfos, checkmethod, onlysig=True)
        tasks.append((TPB.plot_diff_boxplot, (diff_df, seasons, varInfo.abbr, target, checkmethod, FigOutDir_var, boxcfg, varInfo, suffix)))
        tasks.append((TPB.plot_rc_boxplot, (rc_df, seasons, varInfo.abbr, target, checkmethod, FigOutDir_var, boxcfg, varInfo, suffix)))

        path_annual = f'{OutDir}/ExtremeAnalysis/Significance_ExtremeEvent_{target}_{caselist[0]}-{caselist[1]}_{event}_{varname}_yearly_{checkmethod}.nc'
        xarr_annual = TIO.read_newnc(path_annual)
        # 绘制年度差异图（含显著性标记）
        savepath = f'{FigOutDir_var}/{target}_Diff_Map_{event}_Annual_{suffix}_{checkmethod}.{FIGFMT}'
        tasks.append((TPAM.plot_spatial_diffmap_withsign, (xarr_annual, varInfo.abbr, "Annual", target, lon2d, lat2d, 
                                                           checkmethod, lkinfos, annual_diffcfg, onlysig, savepath)))
        # 绘制区域相对贡献图（含显著性标记）
        savepath = f'{FigOutDir_var}/{target}_RC_Map_{event}_Annual_{suffix}_{checkmethod}.{FIGFMT}'
        tasks.append((TPAM.plot_spatial_rcmap_withsign, (xarr_annual, varInfo.abbr, "Annual", target, lon2d, lat2d,
                                                         checkmethod, lkinfos, annual_rccfg, onlysig, savepath)))
        # 绘制colorbar
        label = f"{varInfo.longname} ({varInfo.unit})"
        savepath = f'{FigOutDir_var}/{target}_Diff_Map_{event}_Annual_Colorbar_{suffix}_{checkmethod}.{FIGFMT}'
        tasks.append((TPCB.plot_spatial_cbar_core_V, (annual_diffcfg, 4, label, savepath, 18, 18, 0.06, 'both'))) 
        savepath = f'{FigOutDir_var}/{target}_RC_Map_{event}_Seasonal_Colorbar_{suffix}_{checkmethod}.{FIGFMT}'
        tasks.append((TPCB.plot_spatial_cbar_core_V, (season_diffcfg, 4, label, savepath, 18, 18, 0.06, 'both'))) 
        label = "Relative Contribution (%)"
        savepath = f'{FigOutDir_var}/{target}_RC_Map_{event}_Annual_Colorbar_{suffix}_{checkmethod}.{FIGFMT}'
        tasks.append((TPCB.plot_spatial_cbar_core_V, (annual_rccfg, 4, label, savepath, 18, 18, 0.06, 'both'))) 
        savepath = f'{FigOutDir_var}/{target}_RC_Map_{event}_Seasonal_Colorbar_{suffix}_{checkmethod}.{FIGFMT}'
        tasks.append((TPCB.plot_spatial_cbar_core_V, (season_rccfg, 4, label, savepath, 18, 18, 0.06, 'both'))) 

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

        

def Plot_ExtremeEvents_Additional_Freq(
    eventdict: dict, caselist : list[str], 
    lon2d: np.ndarray, lat2d: np.ndarray,
    checkmethod: str, OutDir: str, FigOutDir: str,
    lkinfos: dict, onlysig: bool = True) -> None:
    """极端事件附加绘图函数"""
    tasks = []
    target = "Freq"
    FigOutDir_var = f'{FigOutDir}/Single'
    eventslist = list(eventdict.keys())
    xarr_events = {}
    level_dict = TYCM.Seasonal_ExtremeEvents_Freq_Cmap('Annual')
    var_info = TYCM.ExtremeEvent_Infos(target)
    varInfo = TPC.varInfo(longname=var_info['longname'], unit=var_info['bunit'], abbr=var_info['abbr'])
    for event in eventslist:
        path_annual = f'{OutDir}/ExtremeAnalysis/Significance_ExtremeEvent_{target}_{caselist[0]}-{caselist[1]}_{event}_{target}_yearly_{checkmethod}.nc'
        xarr_annual = TIO.read_newnc(path_annual)
        xarr_events[event] = xarr_annual
    boxcfg = TPC.boxConfig(diff_boxlevs=level_dict['diff_boxlevs'], rc_boxlevs=level_dict['rc_boxlevs'])
    diff_df, rc_df, suffix = TDP.prepara_for_boxplot_annual(xarr_events, eventslist, lkinfos, checkmethod, onlysig=onlysig)
    orderlist = ["Cold", "Hot", "Dry", "Wet", "Cold-Dry","Cold-Wet", "Hot-Dry", "Hot-Wet",]
    tasks.append((TPB.plot_diff_boxplot, (diff_df, orderlist, 'Annual', target, checkmethod, FigOutDir_var, boxcfg, varInfo, suffix)))
    tasks.append((TPB.plot_rc_boxplot, (rc_df, orderlist, 'Annual', target, checkmethod, FigOutDir_var, boxcfg, varInfo, suffix)))

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




def Plot_ExtremeEvents_Additional_Intensity(
    eventdict: dict, caselist : list[str], 
    lon2d: np.ndarray, lat2d: np.ndarray,
    checkmethod: str, OutDir: str, FigOutDir: str,
    lkinfos: dict, onlysig: bool = True) -> None:
    """极端事件附加绘图函数"""
    tasks = []
    target = "Intensity"
    FigOutDir_var = f'{FigOutDir}/Single'
    eventslist = list(eventdict.keys())
    xarr_events = {}
    level_dict = TYCM.Seasonal_ExtremeEvents_Intensity_Cmap('Annual')
    for event in eventslist:
        event_vars = eventdict[event][1:]
        for varname in event_vars:
            var_info = TYCM.ExtremeEvent_Infos(varname)
            varInfo = TPC.varInfo(longname=var_info['longname'], unit=var_info['bunit'], abbr=var_info['abbr'])
            path_annual = f'{OutDir}/ExtremeAnalysis/Significance_ExtremeEvent_{target}_{caselist[0]}-{caselist[1]}_{event}_{varname}_yearly_{checkmethod}.nc'
            xarr_annual = TIO.read_newnc(path_annual)
            xarr_events[event] = xarr_annual
    boxcfg = TPC.boxConfig(diff_boxlevs=level_dict['diff_boxlevs'], rc_boxlevs=level_dict['rc_boxlevs'])
    diff_df, rc_df, suffix = TDP.prepara_for_boxplot_annual(xarr_events, eventslist, lkinfos, checkmethod, onlysig=onlysig)
    tasks.append((TPB.plot_rc_boxplot, (rc_df, eventslist, 'Annual', target, checkmethod, FigOutDir_var, boxcfg, varInfo, suffix)))

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




def Merge_ExtremeEvents_Freq(event_dicts, FigOutDir: list[str], checkmethod: str, onlysig: bool = True) -> None:
    """合并极端事件频率结果文件"""
    eventslist = list(event_dicts.keys())
    target = "Freq"
    eventslist = list(event_dicts.keys())
    seasons = TU.get_seasons()
    suffix = "onlysig" if onlysig else "all"
    #############################
    # merge plots
    #############################
    mapcrop_params = TYCM.MapPlot_CropParams_noColorbar()
    boxcrop_params = TYCM.Extreme_Events_BoxPlot_CropParams()
    figbox_space = TYCM.Merge_Fig_Space_Params()
    cbar_space = TYCM.MapCbar_CropParams_V()
    print(f"    - Merging annual difference plots...")
    cols_space = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    rows_space = [0.01, 0.01, 0.01]
    fig_rows = [["ColdWet",  "Wet", "HotWet"], 
                ["Cold",       "",    "Hot"],
                ["ColdDry", "Dry", "HotDry"]]
    eventorder = {"ColdWet": [0, 0],
                "Wet":     [0, 1],
                "HotWet":  [0, 2],
                "Cold":    [1, 0],
                "Hot":     [1, 2],
                "ColdDry": [2, 0],
                "Dry":     [2, 1],
                "HotDry":  [2, 2],
                }
    FigOutDir_event = f'{FigOutDir}/Single'
    # diff map & rc map
    for subtarget in ["Diff"]: # , "RC"
        for event in eventslist:
            diffmappath = f'{FigOutDir_event}/{target}_{subtarget}_Map_{event}_Annual_{suffix}_{checkmethod}.{FIGFMT}'
            cropped_map = TIT.crop_image_from_path(diffmappath, crop_params=mapcrop_params, mode="ratio")
            diffcbarpath = f'{FigOutDir_event}/{target}_{subtarget}_Map_{event}_Annual_Colorbar_{suffix}_{checkmethod}.{FIGFMT}'
            cropped_cbar = TIT.crop_image_from_path(diffcbarpath, crop_params=cbar_space, mode="ratio")
            cropped_cbar = TIT.adjust_image_to_ref_canvas(cropped_cbar, cropped_map, axis="height")
            event_img = TIT.merge_images_Row(
                rows_images=[[cropped_map, cropped_cbar]],
                cols_space=[[0]],
                rows_space=[0.0],
                box_space=figbox_space,
                background_color='#FFFFFFFF',
                space_mode="ratio",
                alignment=["left"],
                draw_ticks=False, tick_step=0.01,
                font_path="/share/home/dq048/.local/share/fonts/NotoSans-Bold.ttf",
            )
            fig_rows[eventorder[event][0]][eventorder[event][1]] = event_img
        anual_boxpath_diff = f'{FigOutDir_event}/{target}_{subtarget}_Box_Annual_{suffix}_{checkmethod}.{FIGFMT}'
        cropped_box_diff = TIT.crop_image_from_path(anual_boxpath_diff, crop_params=boxcrop_params, mode="ratio")
        cropped_box_diff = TIT.adjust_image_to_ref_canvas(cropped_box_diff, event_img, axis="width", align="left")
        fig_rows[1][1] = cropped_box_diff
        
        anual_img = TIT.merge_images_Row(
            rows_images=fig_rows,
            cols_space=cols_space,
            rows_space=rows_space,
            box_space=figbox_space,
            background_color='#FFFFFFFF',
            space_mode="ratio",
            alignment=["left", "left", "left"],
            draw_ticks=False, tick_step=0.01,
            font_path="/share/home/dq048/.local/share/fonts/NotoSans-Bold.ttf",
        )
        savepath = f'{FigOutDir}/{target}_{subtarget}_Map_Annual_Comparison_{suffix}_{checkmethod}.{FIGFMT}'
        TIT.save(anual_img, savepath, dpi=DPI)



def Merge_ExtremeEvents_Intensity(event_dicts: Dict, FigOutDir: list[str], checkmethod: str, onlysig: bool = True) -> None:
    """合并极端事件频率结果文件"""
    eventslist = list(event_dicts.keys())
    target = "Intensity"
    eventslist = list(event_dicts.keys())
    seasons = TU.get_seasons()
    suffix = "onlysig" if onlysig else "all"
    #############################
    # merge plots
    #############################
    mapcrop_params = TYCM.MapPlot_CropParams_noColorbar()
    boxcrop_params = TYCM.Extreme_Events_BoxPlot_CropParams()
    figbox_space = TYCM.Merge_Fig_Space_Params()
    cbar_space = TYCM.MapCbar_CropParams_V()
    print(f"    - Merging annual difference plots...")
    cols_space = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    rows_space = [0.01, 0.01, 0.01]
    fig_rows = [["ColdWet",  "Wet", "HotWet"], 
                ["Cold",       "",    "Hot"],
                ["ColdDry", "Dry", "HotDry"]]
    eventorder = {"ColdWet": [0, 0],
                "Wet":     [0, 1],
                "HotWet":  [0, 2],
                "Cold":    [1, 0],
                "Hot":     [1, 2],
                "ColdDry": [2, 0],
                "Dry":     [2, 1],
                "HotDry":  [2, 2],
                }
    # diff map & rc map
    FigOutDir_event = f'{FigOutDir}/Single'
    for subtarget in ["Diff", "RC"]:
        for event in eventslist:
            diffmappath = f'{FigOutDir_event}/{target}_{subtarget}_Map_{event}_Annual_{suffix}_{checkmethod}.{FIGFMT}'
            cropped_map = TIT.crop_image_from_path(diffmappath, crop_params=mapcrop_params, mode="ratio")
            diffcbarpath = f'{FigOutDir_event}/{target}_{subtarget}_Map_{event}_Annual_Colorbar_{suffix}_{checkmethod}.{FIGFMT}'
            cropped_cbar = TIT.crop_image_from_path(diffcbarpath, crop_params=cbar_space, mode="ratio")
            event_img = TIT.merge_images_Row(
                rows_images=[[cropped_map, cropped_cbar]],
                cols_space=[[0]],
                rows_space=[0.0],
                box_space=figbox_space,
                background_color='#FFFFFFFF',
                space_mode="ratio",
                alignment=["left"],
                draw_ticks=False, tick_step=0.01,
                font_path="/share/home/dq048/.local/share/fonts/NotoSans-Bold.ttf",
            )
            fig_rows[eventorder[event][0]][eventorder[event][1]] = event_img
        if subtarget == "Diff":
            # create a blank image for center
            blank_img = TIT.create_blank_image(event_img.width, event_img.height, color='#FFFFFFFF')
            fig_rows[1][1] = blank_img
        else:
            anual_boxpath_diff = f'{FigOutDir_event}/{target}_RC_Box_Annual_{suffix}_{checkmethod}.{FIGFMT}'
            cropped_box_diff = TIT.crop_image_from_path(anual_boxpath_diff, crop_params=boxcrop_params, mode="ratio")
            cropped_box_diff = TIT.adjust_image_to_ref_canvas(cropped_box_diff, event_img, axis="width")
            fig_rows[1][1] = cropped_box_diff
        
        anual_img = TIT.merge_images_Row(
            rows_images=fig_rows,
            cols_space=cols_space,
            rows_space=rows_space,
            box_space=figbox_space,
            background_color='#FFFFFFFF',
            space_mode="ratio",
            alignment=["left", "left", "left"],
            draw_ticks=False, tick_step=0.01,
            font_path="/share/home/dq048/.local/share/fonts/NotoSans-Bold.ttf",
        )
        savepath = f'{FigOutDir}/{target}_{subtarget}_Map_Annual_Comparison_{suffix}_{checkmethod}.{FIGFMT}'
        TIT.save(anual_img, savepath, dpi=DPI)



def Plot_HotWet_Probability(
        caselist: list[str], 
        lon2d: np.ndarray, lat2d: np.ndarray,
        checkmethod: str, OutDir: str, FigOutDir: str,
        lkinfos: dict, onlysig: bool = True) -> None:
    """绘制HotWet的耦合比例"""
    print("    ➠ Computing Extreme Event Joint Occurrence Probability Changes...")
    target = "Freq"
    oceanmask = lkinfos['ocean']
    inlandtotal = np.sum(oceanmask == 0)

    # Helper to load Hot, Wet, HotWet
    def load_event(event):
        outpath=f'{OutDir}/ExtremeAnalysis/Significance_ExtremeEvent_{target}_{caselist[0]}-{caselist[1]}_{event}_{target}_yearly_{checkmethod}.nc'
        xarr = TIO.read_newnc(outpath)
        rejected, diffsign = TDP.preparation_for_probability(xarr, checkmethod, lkinfos)
        return {"sign": rejected, "diff": diffsign}

    Hot_State     = load_event("Hot")
    Wet_State     = load_event("Wet")
    HotWet_State  = load_event("HotWet")

    # ========= Define masks =========
    Hot_up     = (Hot_State["sign"] == 1) & (Hot_State["diff"] == 1)
    Hot_down   = (Hot_State["sign"] == 1) & (Hot_State["diff"] == -1)
    Hot_zero   = (Hot_State["sign"] == 1) & (Hot_State["diff"] == 0)
    Hot_none   = (Hot_State["sign"] == 0)
    Hot = {"up": Hot_up, "down": Hot_down, "zero": Hot_zero, "none": Hot_none}

    Wet_up     = (Wet_State["sign"] == 1) & (Wet_State["diff"] == 1)
    Wet_down   = (Wet_State["sign"] == 1) & (Wet_State["diff"] == -1)
    Wet_zero   = (Wet_State["sign"] == 1) & (Wet_State["diff"] == 0)
    Wet_none   = (Wet_State["sign"] == 0)
    Wet = {"up": Wet_up, "down": Wet_down, "zero": Wet_zero, "none": Wet_none}

    HotWet_total = (HotWet_State["sign"] == 1)
    HotWet_up   = (HotWet_State["sign"] == 1) & (HotWet_State["diff"] == 1)
    HotWet_down = (HotWet_State["sign"] == 1) & (HotWet_State["diff"] == -1)
    HotWet_zero = (HotWet_State["sign"] == 1) & (HotWet_State["diff"] == 0)
    HotWet_none = (HotWet_State["sign"] == 0)
    HotWet = {"up": HotWet_up, "down": HotWet_down, "zero": HotWet_zero, "none": HotWet_none}
    
    # ======== Plotting ========
    tasks = []
    cmaps = TYCM.HotWet_Coupling_Cmap()
    FigOutDir_event = f'{FigOutDir}/Single'
    for state1 in ["up", "down", "none"]:
        for state2 in ["up", "down", "none"]:
            tasks.append((TPAM.plot_hot_wet_coupling, (Hot, Wet, HotWet, {"Hot":state1, "Wet":state2}, lon2d, lat2d, cmaps, FigOutDir_event)))
    label = "Hot-Wet Coupling State"
    ticklabels = ["down", "none", "up"]
    savepath = f'{FigOutDir_event}/HotWet_Coupling_Colorbar.{FIGFMT}'
    tasks.append((TPCB.plot_hot_wet_coupling_colorbar, (cmaps, ticklabels, label, savepath)))

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




def Merge_HotWet_Probability(caselist: list[str], FigOutDir: str):
    """合并HotWet耦合比例结果文件"""
    map_cropparams = TYCM.MapPlot_CropParams_noColorbar()
    cbar_cropparams = TYCM.Zero_space_CropParams()
    figbox_space = TYCM.Merge_Fig_Space_Params()
    cols_space = [[0, 0], [0, 0], [0, 0],[0]]
    rows_space = [0.01, 0.01, 0.01]
    # diff map & rc map
    FigOutDir_event = f'{FigOutDir}/Single'
    fig_rows = [["hotup_wetup",   "hotup_wetdown",   "hotup_wetnone"  ], 
                ["hotdown_wetup", "hotdown_wetdown", "hotdown_wetnone"],
                ["hotnone_wetup", "hotnone_wetdown", "hotnone_wetnone"],
                ["colorbar"]      
               ]
    for row, state1 in enumerate(["up", "down", "none"]):
        for col, state2 in enumerate(["up", "down", "none"]):
            diffmappath = f'{FigOutDir_event}/HotWet_Coupling_Hot-{state1}_Wet-{state2}.{FIGFMT}'
            cropped_map = TIT.crop_image_from_path(diffmappath, crop_params=map_cropparams, mode="ratio")
            fig_rows[row][col] = cropped_map
    colorbar_path = f'{FigOutDir_event}/HotWet_Coupling_Colorbar.{FIGFMT}'
    colorbar = TIT.crop_image_from_path(colorbar_path, crop_params=cbar_cropparams, mode="ratio")
    fig_rows[3][0] = colorbar

    img = TIT.merge_images_Row(
        rows_images=fig_rows,
        cols_space=cols_space,
        rows_space=rows_space,
        box_space=figbox_space,
        background_color='#FFFFFFFF',
        space_mode="ratio",
        alignment=["justify", "justify", "justify", "center"],
        draw_ticks=False, tick_step=0.01,
        font_path="/share/home/dq048/.local/share/fonts/NotoSans-Bold.ttf",
    )
    savepath = f'{FigOutDir}/HotWet_Coupling.{FIGFMT}'
    TIT.save(img, savepath, dpi=DPI)



def Calculate_extreme_Intensity(casename, xarr_in: xr.Dataset, refname: str, outdir: str)->None:
    """计算极端时间强度"""
    thres_path = f"{outdir}/ExtremeAnalysis/Extreme_thresholds_ref_{refname}.nc"
    thres_xarr = TIO.read_newnc(thres_path)
    thres_Hot = thres_xarr["thres_T_Hot"].values
    thres_Wet = thres_xarr["thres_RH_Wet"].values
    xarr_in = xarr_in.squeeze()
    xarr_in = TU.xarray_leap_to_noleap(xarr_in)
    RH = xarr_in["RH"].values
    T = xarr_in["T2m"].values
    Tw = WBT.wet_bulb_temperature_stull(T, RH, T_unit='C', clip_domain=True)

    Hot_diff = T - thres_Hot
    Wet_diff = RH - thres_Wet

    in_dict = {
        'T_Hot_diff': (('time', 'y', 'x'), Hot_diff),
        'RH_Wet_diff': (('time', 'y', 'x'), Wet_diff),
        'Tw': (('time', 'y', 'x'), Tw)
    }
    coords = {'y': xarr_in.y, 'x': xarr_in.x, 'time': xarr_in.time}
    savepath = f"{outdir}/ExtremeAnalysis/Extreme_Intensity_{casename}_ref_{refname}_daily.nc"
    TIO.save_newnc(savepath=savepath, in_dict=in_dict, coords=coords)




def Plot_exT_exRH_Coupling_state(caselist:List, refname: str, lon2d: np.ndarray, lat2d: np.ndarray,OutDir: str,
                                 FigOutDir:str, checkmethod: str, lkinfos: Dict, onlysig: bool = True)->None:
                                
    """计算极端时间强度"""
    figoutdir_var = f"{FigOutDir}/Joint"
    os.makedirs(figoutdir_var, exist_ok=True)

    Lake_Intensity = f"{OutDir}/ExtremeAnalysis/Extreme_Intensity_Lake_ref_{refname}_daily.nc"
    NoLake_Intensity = f"{OutDir}/ExtremeAnalysis/Extreme_Intensity_NoLake_ref_{refname}_daily.nc"
    xarr_Lake = TIO.read_newnc(Lake_Intensity)
    xarr_NoLake = TIO.read_newnc(NoLake_Intensity)
    Hot_diff_Lake = xarr_Lake["T_Hot_diff"].values
    Wet_diff_Lake = xarr_Lake["RH_Wet_diff"].values
    Hot_diff_NoLake = xarr_NoLake["T_Hot_diff"].values
    Wet_diff_NoLake = xarr_NoLake["RH_Wet_diff"].values

    # Helper to load Hot, Wet, HotWet
    def load_event(event):
        target = "Freq"
        outpath=f'{OutDir}/ExtremeAnalysis/Significance_ExtremeEvent_{target}_{caselist[0]}-{caselist[1]}_{event}_{target}_yearly_{checkmethod}.nc'
        xarr = TIO.read_newnc(outpath)
        rejected, diffsign = TDP.preparation_for_probability(xarr, checkmethod, lkinfos)
        return {"sign": rejected, "diff": diffsign}
    target = "Freq"
    outpath=f'{OutDir}/ExtremeAnalysis/Significance_ExtremeEvent_{target}_{caselist[0]}-{caselist[1]}_HotWet_{target}_yearly_{checkmethod}.nc'
    xarr = TIO.read_newnc(outpath)
    rcdata, sig_mask, dash_mask, strong_mask, weak_mask, suffix, area_all, area_strong = TDP.prepara_for_mapplot(xarr, lkinfos, "RC_overall", checkmethod, onlysig)

    Hot_State     = load_event("Hot")
    Wet_State     = load_event("Wet")
    HotWet_State  = load_event("HotWet")

    HotWet_up   = (HotWet_State["sign"] == 1) & (HotWet_State["diff"] == 1)
    Hot_down   = (Hot_State["sign"] == 1) & (Hot_State["diff"] == -1)
    Wet_up     = (Wet_State["sign"] == 1) & (Wet_State["diff"] == 1)

    # Find sites with HotWet up, Hot down, Wet up
    coupling = (Hot_down & Wet_up & HotWet_up & strong_mask)
    # coupling = (Hot_down & Wet_up & HotWet_up)
    sites = np.argwhere(coupling)  # coupling 为布尔数组
    # 打印站点信息
    print(len(sites))
    print(HotWet_up.sum(), Hot_down.sum(), Wet_up.sum())

    # 使用 for 循环拆解 (y, x) 坐标，并根据这些坐标选择数据
    Hot_site_Lake = []
    Wet_site_Lake = []
    Hot_site_NoLake = []
    Wet_site_NoLake = []

    for y, x  in sites:
        Hot_site_Lake.append(Hot_diff_Lake[:, y, x])  # 选择对应站点的温度数据
        Wet_site_Lake.append(Wet_diff_Lake[:, y, x])  # 选择对应站点的湿度数据
        Hot_site_NoLake.append(Hot_diff_NoLake[:, y, x])  # 选择对应站点的温度数据
        Wet_site_NoLake.append(Wet_diff_NoLake[:, y, x])  # 选择对应站点的湿度数据

    # 将结果转为 numpy 数组
    Hot_site_Lake = np.array(Hot_site_Lake)
    Wet_site_Lake = np.array(Wet_site_Lake)
    Hot_site_NoLake = np.array(Hot_site_NoLake)
    Wet_site_NoLake = np.array(Wet_site_NoLake)

    # 创建 DataFrame
    df_big = pd.DataFrame({
        'T': np.concatenate([Hot_site_Lake.flatten(), Hot_site_NoLake.flatten()]),
        'RH': np.concatenate([Wet_site_Lake.flatten(), Wet_site_NoLake.flatten()]),
        'Type': ['Lake'] * len(Hot_site_Lake.flatten()) + ['NoLake'] * len(Hot_site_NoLake.flatten())
    })

    print("    ➠ Plotting Extreme Temperature and Humidity Coupling State with KDE...")

    palette = {
        'Lake': '#9F7D7C',   
        'NoLake': '#FA1E1C'   
    }

    # # 绘制 KDE 密度图
    # savepath = f"{figoutdir_var}/Extreme_Temperature_Humidity_Coupling_State_All_Sites_with_KDE.{FIGFMT}"
    # TPJ.plot_Joint(df_big, "RH", "T", "Type", palette, savepath)
    print("    ➠ Plotting Extreme Temperature and Humidity Coupling State for Each Site with KDE...")
    tasks = []
    # sites = sites[0:10]
    for y, x  in sites:
        Hot_site_Lake = Hot_diff_Lake[:, y, x]
        Wet_site_Lake = Wet_diff_Lake[:, y, x]
        Hot_site_NoLake = Hot_diff_NoLake[:, y, x]
        Wet_site_NoLake = Wet_diff_NoLake[:, y, x]
        lat = lat2d[y, x]
        lon = lon2d[y, x]

        df_in = pd.DataFrame({
            'T': np.concatenate([Hot_site_Lake, Hot_site_NoLake]),
            'RH': np.concatenate([Wet_site_Lake, Wet_site_NoLake]),
            'Type': ['Lake'] * len(Hot_site_Lake) + ['NoLake'] * len(Hot_site_NoLake)
        })

        savepath = f"{figoutdir_var}/Extreme_Temperature_Humidity_Coupling_State_Site_x{x}_y{y}_with_KDE.{FIGFMT}"
        tasks.append((TPJ.plot_Joint, (df_in, "RH", "T", "Type", palette, savepath, lat, lon)))

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



@njit
def _account_affected_population_numba(meshnum_arr, elmindex, pop, mesh_int):
    """
    numba 内核：在 LandScan 栅格上只扫描一遍，累加到 mesh 级别人口，
    再映射回 CWRF 网格。
    """

    # 0) 如果 meshnum 为空，直接返回 0
    if meshnum_arr.size == 0:
        pop_cwrf = np.zeros(mesh_int.shape, np.float64)
        pop_global = np.zeros(pop.shape, np.float64)
        return pop_cwrf, pop_global

    # 1) 找到 id 的最大值，用来分配数组（保证 elmindex / mesh_int / meshnum 都不越界）
    max_id = 0

    # elmindex 中的最大 id
    ny, nx = elmindex.shape
    for j in range(ny):
        for i in range(nx):
            mid = elmindex[j, i]
            if mid > max_id:
                max_id = mid

    # mesh_int 中的最大 id
    my, mx = mesh_int.shape
    for j in range(my):
        for i in range(mx):
            mid = mesh_int[j, i]
            if mid > max_id:
                max_id = mid

    # meshnum 数组本身的最大 id
    for k in range(meshnum_arr.size):
        mid = meshnum_arr[k]
        if mid > max_id:
            max_id = mid

    # 2) 标记哪些 mesh 是我们关心的
    selected = np.zeros(max_id + 1, np.bool_)
    for k in range(meshnum_arr.size):
        m = meshnum_arr[k]
        if m > 0:
            selected[m] = True

    # 3) 准备输出数组
    pop_cwrf = np.zeros(mesh_int.shape, np.float64)
    pop_global = np.zeros(pop.shape, np.float64)
    mesh_pop = np.zeros(max_id + 1, np.float64)   # 每个 mesh 号的总人口

    # 4) 扫描 LandScan：累加 mesh 级人口，并生成 pop_global
    for j in range(ny):
        for i in range(nx):
            mid = elmindex[j, i]
            if mid > 0 and mid <= max_id and selected[mid]:
                val = pop[j, i]
                pop_global[j, i] = val
                mesh_pop[mid] += val

    # 5) 建立 mesh_id -> 在 CWRF 网格中“第一个出现位置”的映射
    mesh_y = -np.ones(max_id + 1, np.int64)
    mesh_x = -np.ones(max_id + 1, np.int64)
    for j in range(my):
        for i in range(mx):
            mid = mesh_int[j, i]
            if mid > 0 and mesh_y[mid] == -1:
                mesh_y[mid] = j
                mesh_x[mid] = i

    # 6) 把 mesh 级人口填回 CWRF 网格（只填第一个格点，保持原来 argwhere(...)[0] 的行为）
    for k in range(meshnum_arr.size):
        m = meshnum_arr[k]
        if m > 0 and m <= max_id and mesh_y[m] != -1:
            pop_cwrf[mesh_y[m], mesh_x[m]] = mesh_pop[m]

    return pop_cwrf, pop_global



def account_affected_population(meshnum, elmindex, pop, mesh_int):
    """
    Python 包装：保证 dtype，调用 numba 内核。
    逻辑与原始版本等价：
        - pop_global：记录所有被选中 mesh 的 LandScan 人口分布
        - pop_cwrf：每个 mesh 的总人口，写在 mesh_int == mnum 的第一个格点上
    """
    meshnum_arr = np.asarray(meshnum, dtype=np.int64)

    # elmindex / mesh_int 一定要是整数，pop 用 float64 计算更稳
    elmindex_i = elmindex.astype(np.int64, copy=False)
    mesh_int_i = mesh_int.astype(np.int64, copy=False)
    pop_f = pop.astype(np.float64, copy=False)

    pop_cwrf, pop_global = _account_affected_population_numba(
        meshnum_arr, elmindex_i, pop_f, mesh_int_i
    )

    #返回前先四舍五入为整数（因为人口数应该是整数）
    pop_cwrf = np.rint(pop_cwrf).astype(np.int64)
    pop_global = np.rint(pop_global).astype(np.int64)

    # 如果你还想要总人口，可以在外面再 sum 一下
    # total_pop_cwrf = float(pop_cwrf.sum())
    # total_pop_global = float(pop_global.sum())

    return pop_cwrf, pop_global



def compute_affected_population(
        caselist: List[str],
        OutDir: str,
        checkmethod: str,
        lkinfos: Dict,
    ) -> None:
    """计算极端事件影响人口（up / down / all 三种情况）"""
    print("    ➠ Calculating Affected Population...")
    target = "Freq"
    eventslist = ["ColdWet", "Wet", "HotWet", "Cold", "Hot", "ColdDry", "Dry", "HotDry"]
    # eventslist = ["HotWet"]
    
    # 1) 读取人口数据 & mesh 映射
    oceanmask = lkinfos['ocean']  # 与 WRF 网格大小一致

    # poppath = "/hydata01/wumej22/Experiment/CWRF/Yangtze_C/RefData/Landscan/landscan-global-2024-population_sel.nc"
    poppath = "/hydata01/wumej22/Experiment/CWRF/Yangtze_C/RefData/Landscan/landscan-global-2000-2024-population_sel_mean.nc"  #使用多年平均 (浮点值，在聚合后需要四舍五入为整数)
    pop_xarr = TIO.read_newnc(poppath)  # 43200 x 21600
    if pop_xarr['lat'][0] > pop_xarr['lat'][-1]:
        pop_xarr = pop_xarr.sortby('lat')
    pop_da = pop_xarr["population"]    # 43200 x 21600
    pop_da = pop_da.fillna(0.0)
    pop_val = pop_da.values.astype(float)

    bufferzone = 15
    meshpath = "/hydata01/wumej22/Experiment/CWRF/Yangtze_C/RefData/Landscan/Yangtze_C_mesh_43200_21600_sel.nc"
    mesh_xarr = TIO.read_newnc(meshpath)
    if mesh_xarr['lat'][0] > mesh_xarr['lat'][-1]:
        mesh_xarr = mesh_xarr.sortby('lat')
    # 说明：
    #   - mesh: 与 WRF 网格一致的 2D 数组（same as sig_mask），每个格点有一个 mesh 编号
    #   - elmindex: 43200 x 21600 的数组，每个 LandScan 像元存储其对应的 mesh 编号
    mesh = mesh_xarr["meshnum"].values          # 形状 == sig_mask（WRF 网格）
    mesh = mesh[bufferzone:-bufferzone, bufferzone:-bufferzone]
    mesh_int = mesh.astype(int)
    elmindex_da = mesh_xarr["elmindex"]         # DataArray, 子区域的 2D, int
    elmindex_val = elmindex_da.values.astype(int)  # numpy 数组，用于 lookup / 循环

    for event in eventslist:
        print(f"      ➠ Event: {event}")
        # 2) 对每个极端事件，读取显著性结果，得到 up / down / all 三个布尔掩模
        ncpath = (
            f"{OutDir}/ExtremeAnalysis/"
            f"Significance_ExtremeEvent_{target}_{caselist[0]}-{caselist[1]}_"
            f"{event}_{target}_yearly_{checkmethod}.nc"
        )
        xarr = TIO.read_newnc(ncpath)
        rejected, diffsign = TDP.preparation_for_probability(xarr, checkmethod, lkinfos)
    
        up_sigmask = (rejected == 1) & (diffsign  == 1)   # 频率显著上升
        down_sigmask = (rejected == 1) & (diffsign  == -1) # 频率显著下降
        sig_bool = (rejected == 1)                      # 所有显著变化（不分正负）

        up_bool = up_sigmask
        down_bool = down_sigmask
        all_bool = sig_bool

        # 3) 利用 up / down / all 掩模，在 WRF 网格上筛选出「对应的 mesh 编号」
        # up
        mesh_up_selected = mesh_int.copy()
        mesh_up_selected[~up_bool] = -999
        meshnum_up = np.unique(mesh_up_selected[mesh_up_selected > 0])

        # down
        mesh_down_selected = mesh_int.copy()
        mesh_down_selected[~down_bool] = -999
        meshnum_down = np.unique(mesh_down_selected[mesh_down_selected > 0])

        # all
        mesh_selected = mesh_int.copy()
        mesh_selected[~all_bool] = -999
        meshnum_all = np.unique(mesh_selected[mesh_selected > 0])

        # 如果整体都没有显著网格，可以直接跳过这个事件
        if meshnum_all.size == 0:
            print("        ⚠ 没有显著网格，跳过该事件")
            continue

        # 4) 循环meshnum，累加人口
        pop_up_cwrf, pop_up_global = account_affected_population(meshnum_up, elmindex_val, pop_val, mesh_int)
        pop_down_cwrf, pop_down_global = account_affected_population(meshnum_down, elmindex_val, pop_val, mesh_int)
        pop_all_cwrf, pop_all_global = account_affected_population(meshnum_all, elmindex_val, pop_val, mesh_int)
        
        # 7) 保存结果 —— 如果觉得太慢，可以把 LandScan 分辨率的几个变量先注释掉
        dict_in = {
            "pop_up_cwrf":   (("y", "x"), pop_up_cwrf),
            "pop_down_cwrf": (("y", "x"), pop_down_cwrf),
            "pop_all_cwrf":  (("y", "x"), pop_all_cwrf),
            # 下面这些是 43200×21600 的大变量，不一定必须留
            "pop_up_global":   (pop_da.dims, pop_up_global),
            "pop_down_global": (pop_da.dims, pop_down_global),
            "pop_all_global":  (pop_da.dims, pop_all_global),
        }

        coords = {
            "y": xarr.y,
            "x": xarr.x,
            elmindex_da.dims[0]: elmindex_da.coords[elmindex_da.dims[0]],
            elmindex_da.dims[1]: elmindex_da.coords[elmindex_da.dims[1]],
        }

        savepath = f"{OutDir}/ExtremeAnalysis/Affected_Population_{target}_{caselist[0]}-{caselist[1]}_{event}_{checkmethod}.nc"
        TIO.save_newnc(savepath, dict_in, coords)

    print("    ➠ Done.")



def summarize_affected_population_by_city(
        caselist: List[str],
        OutDir: str,
        checkmethod: str,
        lkinfos: Dict,
    ) -> None:
    """
    按市级行政区汇总极端事件影响人口（每个事件单独保存一个 shp）。

    说明：
    - 使用 LandScan 子域上的城市 mesh（CityMesh_on_Landscan.nc），
      将每个城市对应的人口栅格加总，得到 up / down / all 三种人口。
    - 每个事件单独输出一个 shp：
        Affected_Population_by_City_{Lake-NoLake}_{Event}_{checkmethod}.shp
    """
    print("    ➠ Summarizing Affected Population by City...")
    target = "Affected_Population"
    eventslist = ["ColdWet", "Wet", "HotWet", "Cold", "Hot", "ColdDry", "Dry", "HotDry"]
    # eventslist = ["HotWet"]

    # --------------------------------------------------
    # 1) 读 city mesh（和 LandScan 子域对齐的城市编号栅格）
    # --------------------------------------------------
    citymeshpath = (
        "/hydata01/wumej22/Experiment/CWRF/Yangtze_C/RefData/Landscan/"
        "CityMesh_on_Landscan.nc"
    )
    citymesh_xarr = TIO.read_newnc(citymeshpath)
    citymesh_da = citymesh_xarr["mesh"]
    citymesh_val = citymesh_da.values

    # --------------------------------------------------
    # 2) 读市级 shp，取城市代码（注意编码）
    # --------------------------------------------------
    shp_path = (
        "/home/wumej22/hydata/Experiment/Process/Yangtze_C_6km_r1/BaseData/"
        "china_yangtze_en.gpkg"
    )
    gdf_base = gpd.read_file(shp_path)  # ★ 关键：指定编码，避免中文变成 ???
    citycodes = gdf_base["code"].values.tolist()

    # 2.1 形状检查用的基准
    mesh_shape = citymesh_val.shape

    # poppath = "/hydata01/wumej22/Experiment/CWRF/Yangtze_C/RefData/Landscan/landscan-global-2024-population_sel.nc"
    poppath = "/hydata01/wumej22/Experiment/CWRF/Yangtze_C/RefData/Landscan/landscan-global-2000-2024-population_sel_mean.nc"  #使用多年平均 (浮点值，在聚合后需要四舍五入为整数)
    pop_xarr = TIO.read_newnc(poppath)  # 43200 x 21600
    if pop_xarr['lat'][0] > pop_xarr['lat'][-1]:
        pop_xarr = pop_xarr.sortby('lat')
    pop_da = pop_xarr["population"]    # 43200 x 21600
    pop_da = pop_da.fillna(0.0)
    pop_val = pop_da.values.astype(float)

    # --------------------------------------------------
    # 3) 预先为每个城市构建掩模，提高效率
    # --------------------------------------------------
    print("    ➠ Pre-computing city masks and total population...")
    city_mask_dict: Dict[int, np.ndarray] = {}
    city_total_pop_dict: Dict[int, float] = {} # 存储每个城市的总人口

    for citycode in citycodes:
        mask = (citymesh_val == citycode)
        city_mask_dict[citycode] = mask
        
        # 计算该城市的总人口基数 (从原始 LandScan 数据统计)
        # 使用 max(1, ...) 避免分母为 0 导致除法报错
        total_pop = float(pop_val[mask].sum())
        city_total_pop_dict[citycode] = total_pop

    # --------------------------------------------------
    # 4) 遍历每个事件，单独计算 & 单独输出 shp
    # --------------------------------------------------
    for event in eventslist:
        print(f"      ➠ Event: {event}")

        # 4.1 读取该事件的全球人口栅格（LandScan 分辨率）
        ncpath = (
            f"{OutDir}/ExtremeAnalysis/"
            f"Affected_Population_Freq_{caselist[0]}-{caselist[1]}_"
            f"{event}_{checkmethod}.nc"
        )
        xarr = TIO.read_newnc(ncpath)
        pop_up_global = xarr["pop_up_global"].values
        pop_down_global = xarr["pop_down_global"].values
        pop_all_global = xarr["pop_all_global"].values

        # 4.2 形状检查
        if pop_up_global.shape != mesh_shape:
            raise ValueError(
                f"[ERROR] citymesh_da.shape={mesh_shape}, "
                f"pop_up_global.shape={pop_up_global.shape}，网格不一致，请检查！"
            )

        # 4.3 准备 GeoDataFrame 列
        gdf_event = gdf_base.copy()
        gdf_event["total_pop"] = 0.0
        gdf_event["up_pop"]    = 0.0
        gdf_event["down_pop"]  = 0.0
        gdf_event["all_pop"]   = 0.0
        
        # 比例列
        gdf_event["up_pct"]    = 0.0
        gdf_event["down_pct"]  = 0.0
        gdf_event["all_pct"]   = 0.0

        # 4.4 按城市汇总人口
        for idx, citycode in enumerate(citycodes):
            mask = city_mask_dict[citycode]
            total_base = city_total_pop_dict[citycode]

            # 汇总受影响人口数
            city_up_pop   = float(pop_up_global[mask].sum())
            city_down_pop = float(pop_down_global[mask].sum())
            city_all_pop  = float(pop_all_global[mask].sum())

            # 写入数值
            gdf_event.at[idx, "total_pop"] = total_base
            gdf_event.at[idx, "up_pop"]    = city_up_pop
            gdf_event.at[idx, "down_pop"]  = city_down_pop
            gdf_event.at[idx, "all_pop"]   = city_all_pop

            # 计算比例 (如果总人口 > 0)
            if total_base > 0:
                gdf_event.at[idx, "up_pct"]   = (city_up_pop / total_base) * 100.0
                gdf_event.at[idx, "down_pct"] = (city_down_pop / total_base) * 100.0
                gdf_event.at[idx, "all_pct"]  = (city_all_pop / total_base) * 100.0

        # 4.5 保存当前事件的 shp（注意写出时也指定相同编码）
        outshp_path = (
            f"{OutDir}/ExtremeAnalysis/"
            f"Affected_Population_by_City_{caselist[0]}-{caselist[1]}_"
            f"{event}_{checkmethod}.gpkg"
        )
        print(f"        ➠ Saving city summary shapefile for {event} to: {outshp_path}")
        gdf_event.to_file(outshp_path, encoding="utf-8", driver="GPKG")  # ★ 关键：写出编码

    print("    ➠ Done.")




def plot_affected_population(
    caselist: List[str],
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    OutDir: str,
    checkmethod: str,
    lkinfos: Dict,
    FigOutDir: str,
) -> None:
    """绘制极端事件影响人口（up / down / all 三种情况）"""
    print("    ➠ Calculating Affected Population...")
    target = "Affected_Population"
    # eventslist = ["ColdWet", "Wet", "HotWet", "Cold", "Hot", "ColdDry", "Dry", "HotDry"]
    eventslist = ["HotWet", ]

    # eventslist = ["HotWet"]
    FigOutDir_var = f'{FigOutDir}/Single'
    os.makedirs(FigOutDir_var, exist_ok=True)
    oceanmask = lkinfos['ocean']  # 与 WRF 网格大小一致
    mask = oceanmask == 0

    # userlevels = [0, 200, 500, 1_000, 2_000, 4_000, 6_000, 8_000, 10_000, 1_000_000]
    # labels_en = [
    #     r"$0$–$200k$",  r"$200k$–$500k$", r"$500k$–$1M$",
    #     r"$1M$–$2M$", r"$2M$–$4M$", r"$4M$–$6M$",
    #     r"$6M$–$8M$", r"$8M$–$10M$", r"$\geq 10M$",
    # ]

    userlevels = [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000]
    pctlevels = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60] 
    labels_en = [
        r"$0$–$250k$",  r"$250k$–$500k$", r"$500k$–$750k$",
        r"$750k$–$1M$", r"$1M$–$1.25M$", r"$1.25M$–$1.5M$",
        r"$1.5M$–$1.75M$", r"$1.75M$–$2M$", r"$2M$–$2.25M$",
        r"$2.25M$–$2.5M$", r"$2.5M$–$2.75M$", r"$2.75M$–$3M$",
        r"$\geq 3M$",
    ]

    tasks = []
    for event in eventslist:
        print(f"      ➠ Event: {event}")
         # 配置变量信息 & 色标信息
        event_info = TYCM.ExtremeEvent_Infos(event)
        varInfo = TPC.varInfo(longname=event_info['longname'], unit=event_info['bunit'], abbr=event_info['abbr'])
        levels_info = TYCM.Affected_Population_Cmap(event)
        mapcfg = TPC.mapConfig(levs=levels_info['grid'], cmap=levels_info['cmap'])
        citycfg = TPC.mapConfig(levs=levels_info['city'], cmap=levels_info['cmap'])
        # citycfg = TPC.mapConfig(levs=[userlevels, len(userlevels)], cmap=levels_info['cmap'])

        # subtarget = "Freq"
        # savepath = f"{OutDir}/ExtremeAnalysis/Affected_Population_{subtarget}_{caselist[0]}-{caselist[1]}_{event}_{checkmethod}.nc"
        # xarr = TIO.read_newnc(savepath)
        # up_cwrf = xarr["pop_up_cwrf"].values
        # savepath = f'{FigOutDir_var}/{target}_Map_{event}-up_{checkmethod}.{FIGFMT}'
        # tasks.append((TPAM.plot_spatial_map, (up_cwrf, None, event, lon2d, lat2d, mask, lkinfos, mapcfg, savepath, None, False)))
        # label = f"{varInfo.longname}"
        # savepath = f'{FigOutDir_var}/{target}_Map_{event}_Colorbar_{checkmethod}.{FIGFMT}'
        # tasks.append((TPCB.plot_spatial_cbar_core_H, (mapcfg, 6.0, label, savepath, 14, 18, 0.03, 'both')))            

        # shapfile
        shp_path = f"{OutDir}/ExtremeAnalysis/Affected_Population_by_City_{caselist[0]}-{caselist[1]}_{event}_{checkmethod}.gpkg"
        gdf = gpd.read_file(shp_path)
        gdf, gdf_pry = TDP.preparation_for_radial_histogram(gdf, "up", "pr_name_en")
        gdf_df = gdf[["pr_name_en", "ct_name_en", "up_pop", "down_pop", "all_pop", "up_pop_K", "up_pct", "down_pct", "all_pct"]].copy()
        print(gdf_df.sort_values(by="up_pct", ascending=False))
        print(gdf_pry.sort_values(by="up_pop_K", ascending=False))

        # # # # city map
        savepath = f'{FigOutDir_var}/{target}_Map_{event}-up_{checkmethod}-shpfile.{FIGFMT}'
        kargs = { 'city_name_col': 'ct_name_en', 'city_linewidth': 0.4, 'city_edgecolor': 'grey'}
        tasks.append((TPAM.plot_city_stat_map_lambert, (gdf, "up_pop_K", citycfg, savepath, kargs)))  
        label = fr"Affected population (Thousands, K)"
        # savepath = f'{FigOutDir_var}/{target}_Map_{event}_VColorbar-shpfile_{checkmethod}.{FIGFMT}'
        # tasks.append((TPCB.plot_spatial_cbar_core_V, (citycfg, 6.0, label, savepath, 14, 18, 0.03, 'max')))       
        savepath = f'{FigOutDir_var}/{target}_Map_{event}_HColorbar-shpfile_{checkmethod}.{FIGFMT}'
        tasks.append((TPCB.plot_spatial_cbar_core_H, (citycfg, 9.0, label, savepath, 24, 30, 0.03, 'max', '{:.0f}'))) 
        savepath = f'{FigOutDir_var}/{target}_Bar_{event}-up_{checkmethod}-shpfile.{FIGFMT}'
        tasks.append((TPBH.plot_stacked_bar_discrete_cmap, (gdf_pry, userlevels, citycfg.cmap, savepath, "pr_name_en", ["up_pop_K"],
                     ['Hubei', 'Hunan', 'Jiangsu', 'Anhui', 'Henan', 'Jiangxi','Shaanxi', 'Fujian',  'Guizhou', 'Zhejiang', 'Sichuan', 'Chongqing']))) 

        # radial histogram
        savepath = f'{FigOutDir_var}/{target}_RadialHistogram_{event}-up_{checkmethod}.{FIGFMT}'
        kargs = {'stack_on': False, 'color_by_colname': 'up_pop_K', 'color_cmap': citycfg.cmap, 'cmap': citycfg.cmap, 'levels': userlevels,
                 'sort_by_Total': True, 'sort_ascending': False, 'inner_circle_radius': 5000,'legend_on': False,
                 'ylims': [0, 3100],'offset_pry_text': -500, 'offset_inner': -100, 'unit_sec_text': 'K',
                #  'primary_cats':['Hubei', 'Hunan', 'Jiangsu', 'Anhui', 'Jiangxi', 'Henan','Fujian', 'Shaanxi', 'Guizhou', 'Zhejiang', 'Chongqing', 'Sichuan'],
                 'primary_cats':['Hubei', 'Hunan', 'Jiangsu', 'Anhui', 'Henan', 'Jiangxi','Shaanxi', 'Fujian',  'Guizhou', 'Zhejiang', 'Sichuan', 'Chongqing'],
                 'blank_length': 2, 'circle_on': False, 'circle_linewidth': 1, 'radii': [0, 1000, 2000, 3000], }  #
        tasks.append((TPRH.radial_histogram, (gdf_df, "pr_name_en", "ct_name_en", ["up_pop_K"], savepath, kargs)))

        # radial_histogram (color by pct)
        savepath = f'{FigOutDir_var}/{target}_RadialHistogram_{event}-up_pct_{checkmethod}.{FIGFMT}'
        kargs = {'stack_on': False, 'color_by_colname': 'up_pop_K', 'color_cmap': citycfg.cmap, 'cmap': citycfg.cmap, 'levels': userlevels,
                 'sort_by_Total': True, 'sort_ascending': False, 'inner_circle_radius': 130,'legend_on': False,
                 'ylims': [0, 65],'offset_pry_text': -9, 'offset_inner': -3, 'offset_sec_text': 2, 'unit_sec_text': '%',
                 'primary_cats':['Hubei', 'Hunan', 'Jiangsu', 'Anhui', 'Jiangxi', 'Henan','Fujian', 'Shaanxi', 'Guizhou', 'Zhejiang', 'Chongqing', 'Sichuan'],
                 'blank_length': 2, 'circle_on': False, 'circle_linewidth': 0.7, 'radii': [0, 20, 40, 60], }  #
        # tasks.append((TPRH.radial_histogram, (gdf_df, "pr_name_en", "ct_name_en", ["up_pct"], savepath, kargs)))


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




def Plot_ExtremeEvents_Freq_New(
        caselist: list[str], 
        checkmethod: str, OutDir: str, FigOutDir: str,
        lkinfos: dict, onlysig: bool = True) -> None:
    """绘制HotWet的耦合比例"""
    print("    ➠ Computing Extreme Event Joint Occurrence Probability Changes...")
    target = "Freq"
    oceanmask = lkinfos['ocean']
    area_km2  = lkinfos['area'] / (1000*1000)
    lakemask  = lkinfos['all']
    landmask = (oceanmask == 0) & (lakemask == 0)

    inlandtotal = np.sum(oceanmask == 0)
    landtotal = np.sum(landmask)
    print(f"      ➠ Total inland grid points: {inlandtotal}")
    print(f"      ➠ Total land grid points: {landtotal}")
    

    # Helper to load Hot, Wet, HotWet
    def load_event(event):
        outpath=f'{OutDir}/ExtremeAnalysis/Significance_ExtremeEvent_{target}_{caselist[0]}-{caselist[1]}_{event}_{target}_yearly_{checkmethod}.nc'
        xarr = TIO.read_newnc(outpath)
        rejected, diffsign = TDP.preparation_for_probability(xarr, checkmethod, lkinfos)
        return {"sign": rejected, "diff": diffsign}

    eventslist = ["ColdWet", "Wet", "HotWet", "Cold", "Hot", "ColdDry", "Dry", "HotDry"]
    
    tasks = []
    for event in eventslist:
        event_state = load_event(event)
        eventinfo = TYCM.ExtremeEvent_Infos(event)
        print(f"      ➠ Event: {eventinfo['abbr']}")

        evente_up  = (event_state["sign"] == 1) & (event_state["diff"] == 1)
        evente_down = (event_state["sign"] == 1) & (event_state["diff"] == -1)
        total = np.nansum(evente_up | evente_down)
        up_per = np.nansum(evente_up) / total * 100
        down_per = np.nansum(evente_down) / total * 100
        up_area = np.nansum(area_km2[evente_up]) / 10000
        down_area = np.nansum(area_km2[evente_down])/ 10000
        print(f"        ➠ Total significant grid points [Inland]: {total} / {inlandtotal} ")
        print(f"        ➠ Total Percentage: {total / inlandtotal * 100:.2f} % ")
        print(f"        ➠ Up Percentage: {np.nansum(evente_up) / total * 100:.2f} % ")
        print(f"        ➠ Down Percentage: {np.nansum(evente_down) / total * 100:.2f} % ")
        df = pd.DataFrame({
            'State': ['Down', 'Up'],
            'Count': [down_per, up_per],
            'Area': [down_area, up_area],
        })
        colors = {"Up": "#FFA96B", "Down": "#53589A"}
        tasks.append((TPCR.plot_circular_ring, (df, 'State', 'Count', colors, eventinfo['abbr'],
                        f'{FigOutDir}/Single/ExtremeEvent_{event}_Freq_Change_CircularRing_{checkmethod}.{FIGFMT}')))

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



def Plot_ExtremeEvents_Freq_Liang(
        caselist: list[str], 
        checkmethod: str, OutDir: str, FigOutDir: str,
        lkinfos: dict, onlysig: bool = True) -> None:
    """绘制HotWet的耦合比例"""
    print("    ➠ Computing Extreme Event Joint Occurrence Probability Changes...")
    target = "Freq"
    oceanmask = lkinfos['ocean']
    area_km2  = lkinfos['area'] / (1000*1000)
    lakemask  = lkinfos['all']
    landmask = (oceanmask == 0) & (lakemask == 0)

    inlandtotal = np.sum(oceanmask == 0)
    landtotal = np.sum(landmask)
    print(f"      ➠ Total inland grid points: {inlandtotal}")
    print(f"      ➠ Total land grid points: {landtotal}")

    # Helper to load Hot, Wet, HotWet
    def load_event(event):
        outpath=f'{OutDir}/ExtremeAnalysis/Significance_ExtremeEvent_{target}_{caselist[0]}-{caselist[1]}_{event}_{target}_yearly_{checkmethod}.nc'
        xarr = TIO.read_newnc(outpath)
        rejected, diffsign = TDP.preparation_for_probability(xarr, checkmethod, lkinfos)
        return {"sign": rejected, "diff": diffsign}

    eventslist = ["ColdWet", "Wet", "HotWet", "Cold", "Hot", "ColdDry", "Dry", "HotDry"]
    
    tasks = []
    for event in eventslist:
        event_state = load_event(event)
        eventinfo = TYCM.ExtremeEvent_Infos(event)
        print(f"      ➠ Event: {eventinfo['abbr']}")
        evente_up  = (event_state["sign"] == 1) & (event_state["diff"] == 1)
        evente_down = (event_state["sign"] == 1) & (event_state["diff"] == -1)

        # lake up
        lake_up = evente_up & lakemask
        land_up = evente_up & landmask
        lake_down = evente_down & lakemask
        land_down = evente_down & landmask

        # percentage
        total = np.nansum(evente_up | evente_down)
        lake_up_per = np.nansum(lake_up) / total * 100
        lake_down_per = np.nansum(lake_down) / total * 100
        land_up_per = np.nansum(land_up) / total * 100
        land_down_per = np.nansum(land_down) / total * 100
        up_per = np.nansum(evente_up) / total * 100
        down_per = np.nansum(evente_down) / total * 100

        # area
        lake_up_area = np.nansum(area_km2[lake_up]) / 10000
        land_up_area = np.nansum(area_km2[land_up]) / 10000
        lake_down_area = np.nansum(area_km2[lake_down]) / 10000
        land_down_area = np.nansum(area_km2[land_down]) / 10000
        up_area = np.nansum(area_km2[evente_up]) / 10000
        down_area = np.nansum(area_km2[evente_down])/ 10000
        print(f"        ➠ Total significant grid points [Inland]: {total} / {inlandtotal} ")
        print(f"        ➠ Total Percentage: {total / inlandtotal * 100:.2f} % ")
        print(f"        ➠ Up Percentage: {np.nansum(evente_up) / total * 100:.2f} % ")
        print(f"        ➠ Down Percentage: {np.nansum(evente_down) / total * 100:.2f} % ")
        print(f"        ➠ Lake Up Percentage: {lake_up_per:.2f} % ")
        print(f"        ➠ Lake Down Percentage: {lake_down_per:.2f} % ")
        print(f"        ➠ Land Up Percentage: {land_up_per:.2f} % ")
        print(f"        ➠ Land Down Percentage: {land_down_per:.2f} % ")
        df = pd.DataFrame({
            'State': ['Lake-Down', 'Land-Down', 'Lake-Up',  'Land-Up'],
            'Count': [lake_down_per, land_down_per, lake_up_per, land_up_per],
            'Area': [lake_down_area, land_down_area, lake_up_area, land_up_area],
        })
        colors = {"Land-Up": "#EC843A", "Lake-Up": "#FCAE76", "Land-Down": "#53589A", "Lake-Down": "#6EA5C6"}
        tasks.append((TPCR.plot_circular_ring_liang, (df, 'State', 'Count', colors, eventinfo['abbr'],
                        f'{FigOutDir}/Single/ExtremeEvent_{event}_Freq_Change_CircularRing_{checkmethod}.{FIGFMT}')))

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



def Plot_ExtremeEvents_Freq_HeatMap(
        caselist: list[str], 
        checkmethod: str, OutDir: str, FigOutDir: str,
        lkinfos: dict, onlysig: bool = True) -> None:
    """绘制HotWet的耦合比例"""
    print("    ➠ Computing Extreme Event Joint Occurrence Probability Changes...")
    target = "Freq"
    lakemask  = lkinfos['all']
    area_km2  = lkinfos['area'] / (1000*1000)
    oceanmask = lkinfos['ocean']
    is_ocean = (oceanmask != 0)
    is_inland = ~is_ocean
    is_lake = (lakemask > 0) & is_inland
    is_land = is_inland & (~is_lake)

    inlandtotal = np.sum(is_inland)
    landtotal = np.sum(is_land)
    laketotal = np.sum(is_lake) # 打印一下这个值看看湖泊格点数对不对
    print(f"      ➠ Total inland grid points: {inlandtotal}")
    print(f"      ➠ Total lake grid points: {laketotal}")
    print(f"      ➠ Total land grid points: {landtotal}")

    # Helper to load Hot, Wet, HotWet
    def load_event(event):
        outpath=f'{OutDir}/ExtremeAnalysis/Significance_ExtremeEvent_{target}_{caselist[0]}-{caselist[1]}_{event}_{target}_yearly_{checkmethod}.nc'
        xarr = TIO.read_newnc(outpath)
        rejected, diffsign = TDP.preparation_for_probability(xarr, checkmethod, lkinfos)
        return {"sign": rejected, "diff": diffsign}

    eventslist = ["ColdWet", "Wet", "HotWet", "Cold", "Hot", "ColdDry", "Dry", "HotDry"]
    
    tasks = []
    for event in eventslist:
        event_state = load_event(event)
        eventinfo = TYCM.ExtremeEvent_Infos(event)
        print(f"      ➠ Event: {eventinfo['abbr']}")
        evente_up  = (event_state["sign"] == 1) & (event_state["diff"] == 1)
        evente_down = (event_state["sign"] == 1) & (event_state["diff"] == -1)

        # 使用布尔索引提取面积
        # 建议先定义一个 helper 函数防止重复除以 10000 出错
        def get_area(mask_condition):
            # mask_condition 是 event 掩码与 地表类型掩码 的交集
            return np.nansum(area_km2[mask_condition]) / 10000

        # lake up
        lake_up_mask = evente_up & is_lake
        land_up_mask = evente_up & is_land
        lake_down_mask = evente_down & is_lake
        land_down_mask = evente_down & is_land

        lake_up_area = get_area(lake_up_mask)
        land_up_area = get_area(land_up_mask)
        lake_down_area = get_area(lake_down_mask)
        land_down_area = get_area(land_down_mask)

        # area percentage
        norm_area = 164.97
        lake_up_area_pct = (lake_up_area / norm_area) * 100
        land_up_area_pct = (land_up_area / norm_area) * 100
        lake_down_area_pct = (lake_down_area / norm_area) * 100
        land_down_area_pct = (land_down_area / norm_area) * 100

        # percentage
        total = np.nansum(evente_up | evente_down)
        lake_up_per = np.nansum(lake_up_mask) / total * 100
        land_up_per = np.nansum(land_up_mask) / total * 100
        lake_down_per = -np.nansum(lake_down_mask) / total * 100
        land_down_per = -np.nansum(land_down_mask) / total * 100

        print(f"        ➠ Total significant grid points [Inland]: {total} / {inlandtotal} ")
        print(f"        ➠ Total Percentage: {total / inlandtotal * 100:.2f} % ")
        print(f"        ➠ Up Percentage: {np.nansum(evente_up) / total * 100:.2f} % ")
        print(f"        ➠ Down Percentage: {np.nansum(evente_down) / total * 100:.2f} % ")
        print(f"        ➠ Lake Up Percentage: {lake_up_per:.2f} % ")
        print(f"        ➠ Lake Down Percentage: {lake_down_per:.2f} % ")
        print(f"        ➠ Land Up Percentage: {land_up_per:.2f} % ")
        print(f"        ➠ Land Down Percentage: {land_down_per:.2f} % ")
        df = pd.DataFrame({
            'State': ['Lake-Down', 'Land-Down', 'Lake-Up',  'Land-Up'],
            'Percent': [lake_down_per, land_down_per, lake_up_per, land_up_per],
            'Area': [lake_down_area, land_down_area, lake_up_area, land_up_area],
            'Area_pct': [lake_down_area_pct, land_down_area_pct, lake_up_area_pct, land_up_area_pct],
        })
        print(df)

        if event in ["Cold", "ColdDry", ]:
            # tick_params = 'y'
            tick_params = 'xy'
        elif event in ["ColdWet", ]:
            tick_params = 'xy'
        elif event in ["Wet", "HotWet", ]:
            # tick_params = 'x'
            tick_params = 'xy'
        else:
            tick_params = 'xy'
            # tick_params = ''

        cmap = cmaps.BlueWhiteOrangeRed
        # tasks.append((TPHM.plot_heatmap, (df, 'State', 'Percent', 'Area', 'Area_pct', eventinfo['abbr'], cmap,
        #                 f'{FigOutDir}/Single/ExtremeEvent_{event}_Freq_Change_HeatMap_{checkmethod}.{FIGFMT}')))
        
        tasks.append((TPHM.plot_heatmap_ticks, (df, 'State', 'Percent', 'Area', 'Area_pct', eventinfo['abbr'], cmap, tick_params,
                        f'{FigOutDir}/Single/ExtremeEvent_{event}_Freq_Change_HeatMap_{checkmethod}.{FIGFMT}')))


    mapcfg = TPC.mapConfig(levs=([-100,-75, -50, -25, 0, 25, 50, 75, 100], 9), cmap=cmaps.BlueWhiteOrangeRed)
    label = "Percentage of Significant Area"
    # tasks.append((TPCB.plot_spatial_cbar_core_heatmap, (mapcfg, 12, label, 
    #                     f'{FigOutDir}/Single/ExtremeEvent_Freq_Change_HeatMap_ColorBar.{FIGFMT}',
    #                     None, None, 0.04, 'neither', '{:.0f}')))

    tasks.append((TPCB.plot_spatial_cbar_core_heatmap_new, (mapcfg, 12, label, 
                        f'{FigOutDir}/Single/ExtremeEvent_Freq_Change_HeatMap_ColorBar.{FIGFMT}',
                        None, None, 0.04, 'neither', '{:.0f}')))

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




def plot_extreme_event_define(FigOutDir: str) -> None:
    """绘制极端事件定义示意图"""
    FigOutDir_event = f'{FigOutDir}/Single'
    os.makedirs(FigOutDir_event, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=DPI)
    fig.patch.set_facecolor("white")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # -----------------------
    # 1) 中心十字箭头
    # -----------------------
    cx, cy = 0.5, 0.5
    arrow_kw = dict(
        arrowstyle="->",
        lw=2,
        color="black",
        shrinkA=0,
        shrinkB=0,
        mutation_scale=14,
    )
    ax.annotate("", xy=(cx, 0.70), xytext=(cx, cy), arrowprops=arrow_kw)  # up
    ax.annotate("", xy=(cx, 0.30), xytext=(cx, cy), arrowprops=arrow_kw)  # down
    ax.annotate("", xy=(0.30, cy), xytext=(cx, cy), arrowprops=arrow_kw)  # left
    ax.annotate("", xy=(0.70, cy), xytext=(cx, cy), arrowprops=arrow_kw)  # right

    # -----------------------
    # 2) 文本块（标题+定义）
    # -----------------------
    items = [
        ("Cold-Wet", (0.15, 0.85), [
            rf"$T_{{d}} < T^{{{LowThresPercentile}\mathrm{{th}}}}_{{d}}$",
            "$&$",
            rf"$RH_{{d}} \geq RH^{{{HighThresPercentile}\mathrm{{th}}}}_{{d}}$"
        ]),
        
        ("Wet", (0.50, 0.85), [
            rf"$RH_{{d}} \geq RH^{{{HighThresPercentile}\mathrm{{th}}}}_{{d}}$"
        ]),
        
        ("Hot-Wet", (0.85, 0.85), [
            rf"$T_{{d}} \geq T^{{{HighThresPercentile}\mathrm{{th}}}}_{{d}}$",
            "$&$",
            rf"$RH_{{d}} \geq RH^{{{HighThresPercentile}\mathrm{{th}}}}_{{d}}$"
        ]),
        
        ("Cold", (0.15, 0.50), [
            rf"$T_{{d}} < T^{{{LowThresPercentile}\mathrm{{th}}}}_{{d}}$"
        ]),
        
        ("Hot", (0.85, 0.50), [
            rf"$T_{{d}} \geq T^{{{HighThresPercentile}\mathrm{{th}}}}_{{d}}$"
        ]),
        
        ("Cold-Dry", (0.15, 0.15), [
            rf"$T_{{d}} < T^{{{LowThresPercentile}\mathrm{{th}}}}_{{d}}$",
            "$&$",
            rf"$RH_{{d}} < RH^{{{LowThresPercentile}\mathrm{{th}}}}_{{d}}$"
        ]),
        
        ("Dry", (0.50, 0.15), [
            rf"$RH_{{d}} < RH^{{{LowThresPercentile}\mathrm{{th}}}}_{{d}}$"
        ]),
        
        ("Hot-Dry", (0.85, 0.15), [
            rf"$T_{{d}} \geq T^{{{HighThresPercentile}\mathrm{{th}}}}_{{d}}$",
            "$&$",
            rf"$RH_{{d}} < RH^{{{LowThresPercentile}\mathrm{{th}}}}_{{d}}$"
        ]),
    ]

    for title, (x, y), lines in items:
        ax.text(x, y, title, ha="center", va="bottom",
                fontsize=22, fontweight="bold", color="black")
        for k, line in enumerate(lines):
            if line == "$&$":
                ax.text(x, y - 0.02 - 0.07 * k, line,
                    ha="center", va="top", 
                    fontsize=16, fontstyle="italic", color="black")
            else:
                ax.text(x, y - 0.02 - 0.06 * k, line,
                        ha="center", va="top", 
                        fontsize=16, fontstyle="italic", color="black")

    # -----------------------
    # 3) 保存/显示
    # -----------------------
    savepath = f"{FigOutDir}/Single/Extreme_Event_Definition_Scheme.{FIGFMT}"
    fig.savefig(savepath, dpi=DPI, facecolor='white',
                    bbox_inches="tight", pad_inches=0.2)
    return fig, ax




def Merge_Plot_ExtremeEvents_Freq_New(
        caselist: list[str], 
        checkmethod: str, OutDir: str, FigOutDir: str,
        lkinfos: dict, onlysig: bool = True) -> None:
    """拼接"""
    figbox_space = TYCM.Merge_Fig_Space_Params()
    cbar_space = TYCM.MapCbar_CropParams_V()
    eventslist = ["ColdWet", "Wet", "HotWet", "Cold", "Hot", "ColdDry", "Dry", "HotDry"]
    larger_params =  {"left": 0.0, "top": 0.0, "right": 0.0, "bottom": 0.0}
    # small_params =  {"left": 0.05, "top": 0.05, "right": 0.05, "bottom": 0.05}
    small_params =  {"left": 0.0, "top": 0.0, "right": 0.0, "bottom": 0.0}

    print(f"    - Merging annual difference plots...")
    cols_space = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    rows_space = [0.01, 0.01, 0.01]
    fig_rows = [["ColdWet",   "Wet", "HotWet"], 
                ["Cold",   "Define",    "Hot"],
                ["ColdDry",   "Dry", "HotDry"]]
    eventorder = {"ColdWet": [0, 0],
                "Wet":     [0, 1],
                "HotWet":  [0, 2],
                "Cold":    [1, 0],
                "Define":  [1, 1],
                "Hot":     [1, 2],
                "ColdDry": [2, 0],
                "Dry":     [2, 1],
                "HotDry":  [2, 2],
                }

    FigOutDir_event = f'{FigOutDir}/Single'

    definepath = f"{FigOutDir}/Single/Extreme_Event_Definition_Scheme.{FIGFMT}"
    define_img = TIT.crop_image_from_path(definepath, crop_params=small_params, mode="ratio")
    fig_rows[eventorder["Define"][0]][eventorder["Define"][1]] = define_img

    for event in eventslist:
        mappath = f'{FigOutDir}/Single/ExtremeEvent_{event}_Freq_Change_CircularRing_{checkmethod}.{FIGFMT}'
        event_img = TIT.crop_image_from_path(mappath, crop_params=larger_params, mode="ratio")
        event_img = TIT.adjust_image_to_ref_canvas(event_img, define_img)
        fig_rows[eventorder[event][0]][eventorder[event][1]] = event_img

    anual_img = TIT.merge_images_Row(
        rows_images=fig_rows,
        cols_space=cols_space,
        rows_space=rows_space,
        box_space=figbox_space,
        background_color='#FFFFFF',
        space_mode="ratio",
        alignment=["justify", "justify", "justify"],
        draw_ticks=False, tick_step=0.01,
        font_path="/share/home/dq048/.local/share/fonts/NotoSans-Bold.ttf",
    )

    savepath = f'{FigOutDir}/ExtremeEvent_Freq.{FIGFMT}'
    TIT.save(anual_img, savepath, dpi=DPI)



def Merge_Plot_ExtremeEvents_Freq_HeatMap(
        caselist: list[str], 
        checkmethod: str, OutDir: str, FigOutDir: str,
        lkinfos: dict, onlysig: bool = True) -> None:
    """拼接"""
    figbox_space = TYCM.Merge_Fig_Space_Params()
    cbar_space = TYCM.MapCbar_CropParams_V()
    eventslist = ["ColdWet", "Wet", "HotWet", "Cold", "Hot", "ColdDry", "Dry", "HotDry"]
    larger_params =  {"left": 0.0, "top": 0.0, "right": 0.0, "bottom": 0.0}
    # small_params =  {"left": 0.05, "top": 0.05, "right": 0.05, "bottom": 0.05}
    small_params =  {"left": 0.0, "top": 0.0, "right": 0.0, "bottom": 0.0}

    print(f"    - Merging annual difference plots...")
    cols_space = [[0.03, 0.03, 0.03], [0.03, 0.03, 0.03], [0.03, 0.03, 0.03]]
    rows_space = [0.03, 0.03, 0.03]
    fig_rows = [["ColdWet",   "Wet", "HotWet"], 
                ["Cold",   "Define",    "Hot"],
                ["ColdDry",   "Dry", "HotDry"]]
    eventorder = {"ColdWet": [0, 0],
                "Wet":     [0, 1],
                "HotWet":  [0, 2],
                "Cold":    [1, 0],
                "Define":  [1, 1],
                "Hot":     [1, 2],
                "ColdDry": [2, 0],
                "Dry":     [2, 1],
                "HotDry":  [2, 2],
                }

    FigOutDir_event = f'{FigOutDir}/Single'

    for event in eventslist:
        mappath = f'{FigOutDir}/Single/ExtremeEvent_{event}_Freq_Change_HeatMap_{checkmethod}.{FIGFMT}'
        event_img = TIT.crop_image_from_path(mappath, crop_params=larger_params, mode="ratio")
        fig_rows[eventorder[event][0]][eventorder[event][1]] = event_img

    definepath = f"{FigOutDir}/Single/Extreme_Event_Definition_Scheme.{FIGFMT}"
    define_img = TIT.crop_image_from_path(definepath, crop_params=small_params, mode="ratio")
    define_img = TIT.resize_image_scale(define_img, 1.1)
    define_img = TIT.adjust_image_to_ref_canvas(define_img, event_img)
    fig_rows[eventorder["Define"][0]][eventorder["Define"][1]] = define_img

    main_img = TIT.merge_images_Row(
        rows_images=fig_rows,
        cols_space=cols_space,
        rows_space=rows_space,
        box_space=figbox_space,
        background_color='#FFFFFF',
        space_mode="ratio",
        alignment=["justify", "justify", "justify"],
        draw_ticks=False, tick_step=0.01,
        font_path="/share/home/dq048/.local/share/fonts/NotoSans-Bold.ttf",
    )

    colorbarpath = f'{FigOutDir}/Single/ExtremeEvent_Freq_Change_HeatMap_ColorBar.{FIGFMT}'
    colorbar_img = TIT.crop_image_from_path(colorbarpath, crop_params=small_params, mode="ratio")
    colorbar_img = TIT.adjust_image_to_ref_canvas(colorbar_img, main_img, axis="height", align="center")
    big_img = TIT.merge_images_Row(
        rows_images=[[main_img, colorbar_img]],
        cols_space=[[0.01]],
        rows_space=[0.01],
        box_space=figbox_space,
        background_color='#FFFFFF',
        space_mode="ratio",
        alignment=["justify", "justify", "justify"],
        draw_ticks=False, tick_step=0.01,
        font_path="/share/home/dq048/.local/share/fonts/NotoSans-Bold.ttf",
    )

    savepath = f'{FigOutDir}/ExtremeEvent_Freq_HeatMap.{FIGFMT}'
    TIT.save(big_img, savepath, dpi=DPI)






def Plot_method_of_define_extreme_event(OutDir, FigOutDir):
    """
    修正 Y 轴刻度对齐问题的版本
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # --- 1. 参数设置 ---
    np.random.seed(42)
    n_years = 5       
    days_in_year = 100
    target_day = 46  
    window_half = 10  
    y_offset = 20     
    doy_offset = 45
    
    # 假设气温基准为 20度
    temp_mean = 20 

    fig, ax = plt.subplots(figsize=(9, 7), dpi=DPI)

    t = np.arange(days_in_year)
    # base_temp 现在在 temp_mean (20) 上下波动
    base_temp = temp_mean + 6 * np.sin(2 * np.pi * (t - target_day) / days_in_year)
    
    colors = [ '#900F49', '#F35A4A', '#FCBB86', '#EBB2BB', '#4F838E',]
    window_fill_color = '#3498db' 

    top_year_offset = (n_years - 1) * y_offset
    y_min_limit = -10 + temp_mean # 根据基准调整底部高度
    y_max_limit = top_year_offset + temp_mean + 15

    # --- 2. 绘制采样窗口 ---
    ax.axvspan(target_day - window_half, target_day + window_half, 
               color=window_fill_color, alpha=0.12, zorder=0)

    # --- 3. 核心绘图循环 ---
    for i in range(n_years):
        offset = (n_years - 1 - i) * y_offset  
        year_num = i + 1
        
        # 数据 = 基准气温 + 噪声 + 垂直偏移
        year_data = base_temp + np.random.normal(0, 1.2, days_in_year) + offset
        
        ax.plot(t, year_data, color='gray', alpha=0.5, linewidth=1.5, zorder=1)

        win_idx = (t >= target_day - window_half) & (t <= target_day + window_half)
        ax.plot(t[win_idx], year_data[win_idx], color=colors[i], linewidth=2.5, zorder=5)
        
        # --- 关键修正：刻度位置 ---
        tick_x = target_day - doy_offset - 2 
        # 刻度中心应该是 temp_mean + offset
        current_center = temp_mean + offset
        ax.vlines(tick_x, current_center - 8, current_center + 8, color='black', linewidth=0.8, zorder=6)
        
        # 标注 15, 20, 25 度（绝对气温）
        for val in [15, 20, 25]:
            pos_y = offset + val # 逻辑：偏移量 + 实际气温值
            ax.hlines(pos_y, tick_x - 0.5, tick_x, color='black', linewidth=0.6, zorder=6)
            ax.text(tick_x - 1, pos_y, f'{val}', ha='right', va='center', fontsize=9, zorder=6)

        ax.text(tick_x + 1, current_center + 2, f'Year $y_{year_num}$', 
                color=colors[i], fontweight='bold', ha='left', fontsize=12, zorder=6)

    # --- 4. 辅助标注 ---
    v_line_min = -5 + temp_mean 
    v_line_max = (n_years - 1) * y_offset + temp_mean + 5

    ax.vlines(x=target_day, ymin=v_line_min, ymax=v_line_max, 
            color='#E15759', linestyle='--', linewidth=1.2, alpha=0.8, zorder=10)
    ax.text(target_day, y_max_limit - 6, 'sample pool for\ncalendar day $d$', color='black', fontweight='bold', fontsize=12, va='center', ha='center')
    texts = r"$\bigcup_{y=1}^{N} \;\; \bigcup_{t=d-7}^{d+7} T_{y,t}$"
    ax.text(target_day-window_half-1, y_max_limit - 6, texts, color='black',  fontweight='bold', fontsize=15, va='center', ha='right')

    ax.annotate('', xy=(target_day - window_half, y_min_limit + 4), xytext=(target_day + window_half, y_min_limit + 4),
                arrowprops=dict(arrowstyle='<->', color='black', linewidth=0.8))
    ax.text(target_day, y_min_limit + 1, '15-day window', ha='center', fontweight='bold', fontsize=9)

    # --- 5. 界面优化 ---
    ax.set_xlim(target_day - doy_offset - 5, target_day + doy_offset)
    ax.set_ylim(y_min_limit, y_max_limit)
    
    ax.set_yticks([]) 
    ax.set_xticks([target_day - window_half, target_day, target_day + window_half])
    ax.set_xticklabels(['$d-7$', '$d$', '$d+7$'], fontsize=12)
    # ax.set_xlabel('day of year', fontsize=11, labelpad=8)
    fig.text(0.537, 0.05, 'Day of Year', 
             ha='center', va='center', fontsize=13)

    fig.text(0.105, 0.5, 'Daily Temperature (°C)', 
             ha='center', va='center', rotation='vertical', fontsize=13)

    plt.title('Step 1: Constructing Sample Pool', 
              fontsize=13, pad=20, fontweight='bold')

    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)

    plt.savefig(f"{FigOutDir}/Step1_Fixed_Ticks.{FIGFMT}", dpi=DPI, bbox_inches="tight")

    # --- 1. 模拟数据 ---
    np.random.seed(42)
    sample_pool = np.random.normal(loc=20, scale=7, size=5000)
    
    # 计算阈值
    t_cold = np.percentile(sample_pool, 5)
    t_hot = np.percentile(sample_pool, 95)

    # --- 2. 绘图设置 ---
    fig, ax = plt.subplots(figsize=(9, 7), dpi=DPI)
    
    # 绘制 PDF 曲线 (不直接 fill，后面手动 fill)
    sns.kdeplot(sample_pool, color='#222222', linewidth=2.5, ax=ax)
    
    # 获取 KDE 曲线的数据点
    line = ax.lines[0]
    x_pdf, y_pdf = line.get_data()
    
    # --- 3. 分段阴影填充 (关键修改) ---
    # 填充中间部分 (Normal range)
    ax.fill_between(x_pdf, 0, y_pdf, 
                    where=(x_pdf > t_cold) & (x_pdf < t_hot), 
                    color='#FFFFFF', alpha=1)
    
    # 填充极冷部分 (Cold)
    ax.fill_between(x_pdf, 0, y_pdf, 
                    where=(x_pdf <= t_cold), 
                    color='#2980b9', alpha=0.5, label='Extreme Cold')
    
    # 填充极热部分 (Hot)
    ax.fill_between(x_pdf, 0, y_pdf, 
                    where=(x_pdf >= t_hot), 
                    color='#c0392b', alpha=0.5, label='Extreme Hot')
    
    # --- 4. 标注阈值垂直线 ---
    ax.axvline(t_cold-0.15, color='#2980b9', linestyle='--', linewidth=2)
    ax.axvline(t_hot+0.15, color='#c0392b', linestyle='--', linewidth=2)

    # 固定 Y 轴高度以防标签出界
    ax.set_ylim(0, 0.07)

    # --- 5. 文本标注 (使用具体分值更清晰) ---
    ax.text(t_cold - 1, 0.05, f'Cold Events\n$T < T_d^{{15}}$\n({t_cold:.1f}°C)', 
            color='#2980b9', ha='right', fontweight='bold', fontsize=16)
    ax.text(t_hot + 1, 0.05, f'Hot Events\n$T > T_d^{{85}}$\n({t_hot:.1f}°C)', 
            color='#c0392b', ha='left', fontweight='bold', fontsize=16)

    # --- 6. 界面优化 ---
    ax.set_title(r'Step 2: Threshold Determination ($T_d^p$)', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Temperature (°C)', fontsize=13)
    ax.set_ylabel('Probability Density', fontsize=13)

    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_xlim(-5, 45)
    ax.set_xticklabels(['0', '10', '20', '30', '40'], fontsize=12)
    ax.set_yticks([0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07])
    ax.set_yticklabels(['0', '0.01', '0.02', '0.03', '0.04', '0.05', '0.06', '0.07'], fontsize=12)

    # 精简边框
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
        
    # 7. 保存
    savepath = f"{FigOutDir}/Step2_Threshold_Distribution.{FIGFMT}"
    plt.savefig(savepath, bbox_inches="tight", dpi=DPI)



def Plot_method_of_define_extreme_event_New(OutDir, FigOutDir):
    """
    修改版：策略2 - 使用省略号和 y_N 表示 N 年数据
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # --- 1. 参数设置 ---
    np.random.seed(42)
    n_years = 5       
    days_in_year = 100
    target_day = 46   
    window_half = 10  
    y_offset = 20     
    doy_offset = 45
    
    # 假设气温基准为 20度
    temp_mean = 20 

    fig, ax = plt.subplots(figsize=(9, 7), dpi=DPI) # 假设 DPI=300

    t = np.arange(days_in_year)
    base_temp = temp_mean + 6 * np.sin(2 * np.pi * (t - target_day) / days_in_year)
    
    colors = [ '#900F49', '#F35A4A', '#FCBB86', '#EBB2BB', '#4F838E',]
    window_fill_color = '#3498db' 

    top_year_offset = (n_years - 1) * y_offset
    y_min_limit = -10 + temp_mean 
    y_max_limit = top_year_offset + temp_mean + 15

    # --- 2. 绘制采样窗口 ---
    # 注意：这里的阴影区域需要覆盖所有年份，包括被省略的区域，视觉上更连贯
    ax.axvspan(target_day - window_half, target_day + window_half, 
               ymin=0.05, ymax=0.95, # 使用相对坐标让它不要顶格
               transform=ax.get_xaxis_transform(),
               color=window_fill_color, alpha=0.12, zorder=0)

    # --- 3. 核心绘图循环 (修改重点) ---
    for i in range(n_years):
        offset = (n_years - 1 - i) * y_offset  
        
        # === 策略 2 修改开始 ===
        
        # 索引 2: 绘制省略号，不画线
        if i == 3:
            # 计算省略号的位置 (大约在日历日 d 的位置)
            dot_x = target_day 
            dot_y = temp_mean + offset
            # 绘制垂直省略号
            tick_x_pos = target_day - 40
            ax.text(tick_x_pos, dot_y, r'$\vdots$', fontsize=30, color='gray', 
                    ha='center', va='center', zorder=10)
            # 也可以在左侧 y轴标签位置加一个小省略号
            tick_x_pos = target_day - 20
            ax.text(tick_x_pos, dot_y, r'$\vdots$', fontsize=30, color='gray', 
                    ha='center', va='center', zorder=10)
            # 也可以在左侧 y轴标签位置加一个小省略号
            tick_x_pos = target_day + 40
            ax.text(tick_x_pos, dot_y, r'$\vdots$', fontsize=30, color='gray', 
                    ha='center', va='center', zorder=10)
            # 也可以在左侧 y轴标签位置加一个小省略号
            tick_x_pos = target_day + 20
            ax.text(tick_x_pos, dot_y, r'$\vdots$', fontsize=30, color='gray', 
                    ha='center', va='center', zorder=10)
            continue # 跳过这一行的实际数据绘制
        
        # 确定标签文本
        if i == n_years - 1:
            # 最后一行强制改为 y_N
            year_label = r'Year $y_N$'
            line_color = colors[-1] # 使用最后一个颜色
        else:
            # 前面的行正常显示 y_1, y_2
            year_label = f'Year $y_{i+1}$'
            line_color = colors[i]
            
        # === 策略 2 修改结束 ===
        
        # 数据 = 基准气温 + 噪声 + 垂直偏移
        year_data = base_temp + np.random.normal(0, 1.2, days_in_year) + offset
        
        # 绘制整年灰线
        ax.plot(t, year_data, color='gray', alpha=0.5, linewidth=1.5, zorder=1)

        # 绘制窗口内高亮线
        win_idx = (t >= target_day - window_half) & (t <= target_day + window_half)
        ax.plot(t[win_idx], year_data[win_idx], color=line_color, linewidth=2.5, zorder=5)
        
        # --- 刻度位置 ---
        tick_x = target_day - doy_offset - 2 
        current_center = temp_mean + offset
        ax.vlines(tick_x, current_center - 8, current_center + 8, color='black', linewidth=0.8, zorder=6)
        
        # 标注 15, 20, 25 度
        for val in [15, 20, 25]:
            pos_y = offset + val 
            ax.hlines(pos_y, tick_x - 0.5, tick_x, color='black', linewidth=0.6, zorder=6)
            ax.text(tick_x - 1, pos_y, f'{val}', ha='right', va='center', fontsize=9, zorder=6)

        # 绘制年份标签 (使用上面确定好的 year_label)
        ax.text(tick_x + 1, current_center + 2, year_label, 
                color=line_color, fontweight='bold', ha='left', fontsize=12, zorder=6)

    # --- 4. 辅助标注 ---
    v_line_min = -5 + temp_mean 
    v_line_max = (n_years - 1) * y_offset + temp_mean + 5

    ax.vlines(x=target_day, ymin=v_line_min, ymax=v_line_max, 
            color='#E15759', linestyle='--', linewidth=1.2, alpha=0.8, zorder=10)
    
    # 修改这里的文字，使其适应省略后的布局
    ax.text(target_day, y_max_limit - 6, 'Sample pool for\ncalendar day $d$', color='black', fontweight='bold', fontsize=12, va='center', ha='center')
    
    # 公式标注 (顺便加上 N=25 的说明，让图更严谨)
    texts = r"$\bigcup_{y=1}^{N} \;\; \bigcup_{t=d-7}^{d+7} T_{y,t}$"
    ax.text(target_day-window_half-1, y_max_limit - 6, texts, color='black',  fontweight='bold', fontsize=15, va='center', ha='right')

    # 15-day window 箭头
    ax.annotate('', xy=(target_day - window_half, y_min_limit + 4), xytext=(target_day + window_half, y_min_limit + 4),
                arrowprops=dict(arrowstyle='<->', color='black', linewidth=0.8))
    ax.text(target_day, y_min_limit + 1, '15-day window', ha='center', fontweight='bold', fontsize=9)

    # --- 5. 界面优化 ---
    ax.set_xlim(target_day - doy_offset - 5, target_day + doy_offset)
    ax.set_ylim(y_min_limit, y_max_limit)
    
    ax.set_yticks([]) 
    ax.set_xticks([target_day - window_half, target_day, target_day + window_half])
    ax.set_xticklabels(['$d-7$', '$d$', '$d+7$'], fontsize=12)
    
    fig.text(0.537, 0.05, 'Day of year', ha='center', va='center', fontsize=13)
    fig.text(0.105, 0.5, 'Daily temperature (°C)', ha='center', va='center', rotation='vertical', fontsize=13)

    plt.title('Step 1: constructing sample pool', fontsize=13, pad=20, fontweight='bold')

    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)

    # 保存图片 (Step 1 结束)
    plt.savefig(f"{FigOutDir}/Step1_Schematic_N_Years.{FIGFMT}", dpi=DPI, bbox_inches="tight")
    
    # Step 2 的代码保持不变，此处省略以节省空间...
    print("Plot generated successfully.")


    # --- 1. 模拟数据 ---
    np.random.seed(42)
    sample_pool = np.random.normal(loc=20, scale=7, size=5000)
    
    # 计算阈值
    t_cold = np.percentile(sample_pool, 5)
    t_hot = np.percentile(sample_pool, 95)

    # --- 2. 绘图设置 ---
    fig, ax = plt.subplots(figsize=(9, 7), dpi=DPI)
    
    # 绘制 PDF 曲线 (不直接 fill，后面手动 fill)
    sns.kdeplot(sample_pool, color='#222222', linewidth=2.5, ax=ax)
    
    # 获取 KDE 曲线的数据点
    line = ax.lines[0]
    x_pdf, y_pdf = line.get_data()
    
    # --- 3. 分段阴影填充 (关键修改) ---
    # 填充中间部分 (Normal range)
    ax.fill_between(x_pdf, 0, y_pdf, 
                    where=(x_pdf > t_cold) & (x_pdf < t_hot), 
                    color='#FFFFFF', alpha=1)
    
    # 填充极冷部分 (Cold)
    ax.fill_between(x_pdf, 0, y_pdf, 
                    where=(x_pdf <= t_cold), 
                    color='#2980b9', alpha=0.5, label='Extreme Cold')
    
    # 填充极热部分 (Hot)
    ax.fill_between(x_pdf, 0, y_pdf, 
                    where=(x_pdf >= t_hot), 
                    color='#c0392b', alpha=0.5, label='Extreme Hot')
    
    # --- 4. 标注阈值垂直线 ---
    ax.axvline(t_cold-0.15, color='#2980b9', linestyle='--', linewidth=2)
    ax.axvline(t_hot+0.15, color='#c0392b', linestyle='--', linewidth=2)

    # 固定 Y 轴高度以防标签出界
    ax.set_ylim(0, 0.07)

    # --- 5. 文本标注 (使用具体分值更清晰) ---
    # ax.text(t_cold - 1, 0.05, f'Cold events\n$T < T_d^{{15}}$\n({t_cold:.2f}°C)', 
    #         color='#2980b9', ha='right', fontweight='bold', fontsize=16)
    # ax.text(t_hot + 1, 0.05, f'Hot events\n$T > T_d^{{85}}$\n({t_hot:.2f}°C)', 
    #         color='#c0392b', ha='left', fontweight='bold', fontsize=16)
    ax.text(
        t_hot + 1, 0.05,
        f'Hot events\n$T > T_d^{{{HighThresPercentile}}}$\n({t_hot:.2f}°C)',
        color='#c0392b', ha='left', fontweight='bold', fontsize=16
    )
    ax.text(
        t_cold - 1, 0.05,
        f'Cold events\n$T < T_d^{{{LowThresPercentile}}}$\n({t_cold:.2f}°C)',
        color='#2980b9', ha='right', fontweight='bold', fontsize=16
    )


    # --- 6. 界面优化 ---
    ax.set_title(r'Step 2: threshold determination ($T_d^p$)', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Temperature (°C)', fontsize=13)
    ax.set_ylabel('Probability density', fontsize=13)

    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_xlim(-5, 45)
    ax.set_xticklabels(['0', '10', '20', '30', '40'], fontsize=12)
    ax.set_yticks([0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07])
    ax.set_yticklabels(['0', '0.01', '0.02', '0.03', '0.04', '0.05', '0.06', '0.07'], fontsize=12)

    # 精简边框
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
        
    # 7. 保存
    savepath = f"{FigOutDir}/Step2_Threshold_Distribution.{FIGFMT}"
    plt.savefig(savepath, bbox_inches="tight", dpi=DPI)
