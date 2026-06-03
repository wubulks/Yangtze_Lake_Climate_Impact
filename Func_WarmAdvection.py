import os
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import metpy.calc as mpcalc
from metpy.units import units
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
import ToolBoxes.Tool_PlotConfig as TPC
import ToolBoxes.Tool_InputOutput as TIO
import ToolBoxes.Tool_DataPrepare as TDP
import ToolBoxes.Tool_ImageToolkit as TIT

FIGFMT = TPC.FIGFMT
DPI    = TPC.DPI_medium


warnings.filterwarnings(
    "ignore",
    message="Vertical dimension number not found. Defaulting to (..., Z, Y, X) order.",
    category=UserWarning,
    module="metpy.xarray"
)


def WarmAdvection(T_da, U_da, V_da, dx, dy):
    """
    计算指定气压层的热平流（K/h）

    参数:
    T -- 温度场 (DataArray)，单位 degC
    U -- 东西风 (DataArray)，单位 m/s
    V -- 南北风 (DataArray)，单位 m/s
    dx_da -- x方向网格间距 (DataArray)，单位 m
    dy_da -- y方向网格间距 (DataArray)，单位 m
    """
    # 1. 量化单位
    Tq = T_da.metpy.quantify().metpy.convert_units("K")
    Uq = U_da.metpy.quantify().metpy.convert_units("m/s")
    Vq = V_da.metpy.quantify().metpy.convert_units("m/s")

    # 如果我们知道dx和dy是常数，可以直接使用标量值
    # 假设dx和dy在整个网格上是常数
    dx_scalar = dx * units('m')
    dy_scalar = dy * units('m')
    
    # 3. 使用metpy的advection函数计算温度平流
    # 注意：我们需要指定x_dim和y_dim参数
    warm_adv = mpcalc.advection(
        Tq, 
        u=Uq, 
        v=Vq, 
        dx=dx_scalar, 
        dy=dy_scalar,
        x_dim=1,  # x维度是第二个维度
        y_dim=0   # y维度是第一个维度
    )
    
    # 4. 转换为K/hour
    warm_adv = warm_adv.metpy.convert_units("degC/day")
    warm_adv.name = "warm_advection"

    return warm_adv



def compute_single_time_step_warm_advection(itime, case1, case2, dx, dy):
    """
    为单个时间步计算热平流
    """
    # Lake case
    T = case1["T"].isel(time=itime).squeeze(drop=True)
    U = case1["U"].isel(time=itime).squeeze(drop=True)
    V = case1["V"].isel(time=itime).squeeze(drop=True)
    T.attrs['units'] = 'degC'
    U.attrs['units'] = 'm/s'
    V.attrs['units'] = 'm/s'
    warm_adv = WarmAdvection(T, U, V, dx, dy)
    Lake_warm_adv = warm_adv.values

    # NoLake case
    T = case2["T"].isel(time=itime).squeeze(drop=True)
    U = case2["U"].isel(time=itime).squeeze(drop=True)
    V = case2["V"].isel(time=itime).squeeze(drop=True)
    T.attrs['units'] = 'degC'
    U.attrs['units'] = 'm/s'
    V.attrs['units'] = 'm/s'
    warm_adv = WarmAdvection(T, U, V, dx, dy)
    NoLake_warm_adv = warm_adv.values

    return Lake_warm_adv, NoLake_warm_adv



def WarmAdvectionSignificanceOfChange_seasonal(caselist, case1, case2, dx, dy, checkmethod, outdir, level):
    """计算热平流并保存为 NetCDF 文件
    参数:
    caselist -- 案例名称列表 (list)，如 ["Lake", "NoLake"]
    case1 -- 案例1数据字典 (dict)，包含 "T", "U", "V" 三个 DataArray
    case2 -- 案例2数据字典 (dict)，包含 "T", "U", "V" 三个 DataArray
    dx -- x方向网格间距 (float)，单位 m
    dy -- y方向网格间距 (float)，单位 m
    level -- 气压层 (int)，默认 1000 hPa
    """
    alternative = "two-sided"     # ["two-sided", "greater", "less"] 
    alpha_ci = 0.05               # 显著性水平
    clt_n = 30                    # 中心极限定理样本量
    n_sample = 10000              # 重抽样次数
    ci = 0.95                     # 置信区间
    center_null = True            # 是否中心化零假设
    random_state = 666            # 随机种子
    n_jobs = 96

    target = f"WarmAdvection"
    outcase = f"{outdir}/{target}/{level}hPa"
    os.makedirs(outcase, exist_ok=True)
    seasons = TU.get_seasons()

    Lake_T = case1["T"]
    Lake_U = case1["U"]
    Lake_V = case1["V"]
    NoLake_T = case2["T"]
    NoLake_U = case2["U"]
    NoLake_V = case2["V"]

    ntime, ny, nx = case1["T"].shape
    Lake_warm_adv = xr.DataArray(
        np.zeros((ntime, ny, nx)),
        dims=["time", "y", "x"],
        coords={ "time": case1["T"].time, "y": case1["T"].y, "x": case1["T"].x },
        name="warm_advection"
    )
    Lake_warm_adv.attrs['units'] = 'degC/hour'
    NoLake_warm_adv = xr.DataArray(
        np.zeros((ntime, ny, nx)),
        dims=["time", "y", "x"],
        coords={ "time": case1["T"].time, "y": case1["T"].y, "x": case1["T"].x },
        name="warm_advection"
    )
    NoLake_warm_adv.attrs['units'] = 'degC/hour'

    # 并行执行
    ntasks = ntime
    with Parallel(n_jobs=n_jobs, backend="loky", pre_dispatch="2*n_jobs", return_as="generator") as parallel:
        gen = parallel(delayed(compute_single_time_step_warm_advection)(itime, case1, case2, dx, dy) for itime in range(ntime))
        results = [p for p in tqdm(gen, total=ntasks, desc="    ➠ Parallel", unit="task", dynamic_ncols=True)]
        
    # 确保初始化数据结构
    for itime, (Lake_warm_adv_step, NoLake_warm_adv_step) in enumerate(results):
        Lake_warm_adv[itime, :, :] = Lake_warm_adv_step
        NoLake_warm_adv[itime, :, :] = NoLake_warm_adv_step

    print("    All parallel plots done.")

    # 保存数据
    in_dict = {
        caselist[0]: [["time", "y", "x"], Lake_warm_adv.values],
        caselist[1]: [["time", "y", "x"], NoLake_warm_adv.values],
    }
    coords = {"y": Lake_warm_adv.y, "x": Lake_warm_adv.x, "time": Lake_warm_adv.time}
    TIO.save_newnc(savepath=f'{outcase}/WarmAdvection_{caselist[0]}_{caselist[1]}_{level}hPa.nc', in_dict=in_dict, coords=coords)

    # 按季节计算平均值
    for season in seasons:
        print(f'    ---- Processing season: {season} ----')

        # 选择特定季节的数据
        Lake_warm_adv_season = Lake_warm_adv.sel(time=Lake_warm_adv.time.dt.season == season)
        NoLake_warm_adv_season = NoLake_warm_adv.sel(time=NoLake_warm_adv.time.dt.season == season)

        # 计算季节平均
        Lake_warm = Lake_warm_adv_season.groupby("time.year").mean("time")
        NoLake_warm = NoLake_warm_adv_season.groupby("time.year").mean("time")
        arr1_check = Lake_warm.values.squeeze()
        arr2_check = NoLake_warm.values.squeeze()
        p_arr, mean_diff_arr, effect_size_arr, method_arr, ci_low_arr, ci_high_arr = TST.SignificanceTest(arr1_check, arr2_check,
                                                            checkmethod=checkmethod, alternative=alternative,
                                                            alpha_ci=alpha_ci, clt_n=clt_n, n_sample=n_sample,
                                                            ci=ci, info=f'{season} ', center_null=center_null,
                                                            random_state=random_state, n_jobs=n_jobs)
        p_ravel = p_arr.ravel(); mask = np.isfinite(p_ravel)
        rej_fdr = np.full_like(p_ravel, 0, dtype=np.int8)  # 初始化拒绝标记
        p_fdr = np.full_like(p_ravel, np.nan, dtype=float)  # 初始化 FDR 校正后的 p 值
        rej_sub, p_sub, _, _ = multipletests(p_ravel[mask], alpha=0.05, method="fdr_bh")
        rej_fdr[mask] = rej_sub.astype(np.int8); p_fdr[mask] = p_sub
        rej_fdr = rej_fdr.reshape(p_arr.shape); p_fdr = p_fdr.reshape(p_arr.shape)
        savepath = f'{outcase}/Significance_{target}_{caselist[0]}-{caselist[1]}_seasonal_{season}_{checkmethod}.nc'
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
        years = Lake_warm["year"].values  # 比如 [1996, 1997, ..., 2020]
        # 构造一个代表这一年这个季节的日期，这里随便举例：该年的 7 月 1 日
        timelist = np.array([np.datetime64(f"{y}-01-01") for y in years])
        coords = {"y": Lake_warm_adv.y, "x": Lake_warm_adv.x, "time": timelist}
        TIO.save_newnc(savepath=savepath, in_dict=in_dict, coords=coords)

        Lake_T_season = Lake_T.sel(time=Lake_T.time.dt.season == season)
        Lake_U_season = Lake_U.sel(time=Lake_U.time.dt.season == season)
        Lake_V_season = Lake_V.sel(time=Lake_V.time.dt.season == season)
        Lake_warm_adv_season = Lake_warm_adv.sel(time=Lake_warm_adv.time.dt.season == season)
        NoLake_T_season = NoLake_T.sel(time=NoLake_T.time.dt.season == season)
        NoLake_U_season = NoLake_U.sel(time=NoLake_U.time.dt.season == season)
        NoLake_V_season = NoLake_V.sel(time=NoLake_V.time.dt.season == season)
        NoLake_warm_adv_season = NoLake_warm_adv.sel(time=NoLake_warm_adv.time.dt.season == season)

        Lake_T_season_mean = Lake_T_season.mean("time").squeeze(drop=True)
        Lake_U_season_mean = Lake_U_season.mean("time").squeeze(drop=True)
        Lake_V_season_mean = Lake_V_season.mean("time").squeeze(drop=True)
        Lake_warm_adv_season_mean = Lake_warm_adv_season.mean("time").squeeze(drop=True)
        NoLake_T_season_mean = NoLake_T_season.mean("time").squeeze(drop=True)
        NoLake_U_season_mean = NoLake_U_season.mean("time").squeeze(drop=True)
        NoLake_V_season_mean = NoLake_V_season.mean("time").squeeze(drop=True)
        NoLake_warm_adv_season_mean = NoLake_warm_adv_season.mean("time").squeeze(drop=True)

        in_dict = {
            'T': [["y", "x"], Lake_T_season_mean.values],
            'U': [["y", "x"], Lake_U_season_mean.values],
            'V': [["y", "x"], Lake_V_season_mean.values],
            'WA': [["y", "x"], Lake_warm_adv_season_mean.values]
        }
        coords = {"y": Lake_T.y, "x": Lake_T.x}
        TIO.save_newnc(savepath=f"{outcase}/MeanStates_Lake_seasonal_{season}_{level}hPa.nc", in_dict=in_dict, coords=coords)

        in_dict = {
            'T': [["y", "x"], NoLake_T_season_mean.values],
            'U': [["y", "x"], NoLake_U_season_mean.values],
            'V': [["y", "x"], NoLake_V_season_mean.values],
            'WA': [["y", "x"], NoLake_warm_adv_season_mean.values]
        }
        coords = {"y": Lake_T.y, "x": Lake_T.x}
        TIO.save_newnc(savepath=f"{outcase}/MeanStates_NoLake_seasonal_{season}_{level}hPa.nc", in_dict=in_dict, coords=coords)



    # 按“年际平均”分析，形状 (nyear, y, x)
    print(f'    ---- Processing season: Annual mean ----')
    Lake_warm_adv_ = Lake_warm_adv.groupby("time.year").mean("time")
    NoLake_warm_adv_ = NoLake_warm_adv.groupby("time.year").mean("time")
    arr1_year = Lake_warm_adv_.values.squeeze()
    arr2_year = NoLake_warm_adv_.values.squeeze()
    p_arr, mean_diff_arr, effect_size_arr, method_arr, ci_low_arr, ci_high_arr = TST.SignificanceTest(arr1_year, arr2_year,
                                                        checkmethod=checkmethod, alternative=alternative,
                                                        alpha_ci=alpha_ci, clt_n=clt_n, n_sample=n_sample,
                                                        ci=ci, info=f'Annual ', center_null=center_null,
                                                        random_state=random_state, n_jobs=n_jobs)
    p_ravel = p_arr.ravel(); mask = np.isfinite(p_ravel)
    rej_fdr = np.full_like(p_ravel, 0, dtype=np.int8)  # 初始化拒绝标记
    p_fdr = np.full_like(p_ravel, np.nan, dtype=float)  # 初始化 FDR 校正后的 p 值
    rej_sub, p_sub, _, _ = multipletests(p_ravel[mask], alpha=0.05, method="fdr_bh")
    rej_fdr[mask] = rej_sub.astype(np.int8); p_fdr[mask] = p_sub
    rej_fdr = rej_fdr.reshape(p_arr.shape); p_fdr = p_fdr.reshape(p_arr.shape)
    savepath = f'{outcase}/Significance_{target}_{caselist[0]}-{caselist[1]}_yearly_{checkmethod}.nc'
    RC_overall, RC_anomaly = TST.RelativeContribution(arr1_year, arr2_year, time_axis=0)
    in_dict = {'p_value': [["y", "x"], p_arr],
               'mean_diff': [["y", "x"], mean_diff_arr],
               'effect_size': [["y", "x"], effect_size_arr],
               'checkmethod': [["y", "x"], method_arr],
               'p_fdr': [["y", "x"], p_fdr],
               'rej_fdr': [["y", "x"], rej_fdr],
               'RC_overall': [["y", "x"], RC_overall],
               'RC_anomaly': [["y", "x"], RC_anomaly],
               }
    years = Lake_warm_adv_["year"].values  # 比如 [1996, 1997, ..., 2020]
    # 构造一个代表这一年这个季节的日期，这里随便举例：该年的 7 月 1 日
    timelist = np.array([np.datetime64(f"{y}-01-01") for y in years])
    coords = {"y": Lake_T_season_mean.y, "x": Lake_T_season_mean.x, "time": timelist}
    TIO.save_newnc(savepath=savepath, in_dict=in_dict, coords=coords)

    # 先按年平均 -> (year, y, x)
    Lake_T_year = Lake_T.groupby("time.year").mean("time")
    Lake_T_clim = Lake_T_year.mean("year").squeeze(drop=True)
    Lake_U_year = Lake_U.groupby("time.year").mean("time")
    Lake_U_clim = Lake_U_year.mean("year").squeeze(drop=True)
    Lake_V_year = Lake_V.groupby("time.year").mean("time")
    Lake_V_clim = Lake_V_year.mean("year").squeeze(drop=True)
    Lake_warm_adv_year = Lake_warm_adv.groupby("time.year").mean("time")
    Lake_warm_adv_clim = Lake_warm_adv_year.mean("year").squeeze(drop=True)

    NoLake_T_year = NoLake_T.groupby("time.year").mean("time")
    NoLake_T_clim = NoLake_T_year.mean("year").squeeze(drop=True)
    NoLake_U_year = NoLake_U.groupby("time.year").mean("time")
    NoLake_U_clim = NoLake_U_year.mean("year").squeeze(drop=True)
    NoLake_V_year = NoLake_V.groupby("time.year").mean("time")
    NoLake_V_clim = NoLake_V_year.mean("year").squeeze(drop=True)
    NoLake_warm_adv_year = NoLake_warm_adv.groupby("time.year").mean("time")
    NoLake_warm_adv_clim = NoLake_warm_adv_year.mean("year").squeeze(drop=True)

    in_dict = {
        'T': [["y", "x"], Lake_T_clim.values],
        'U': [["y", "x"], Lake_U_clim.values],
        'V': [["y", "x"], Lake_V_clim.values],
        'WA': [["y", "x"], Lake_warm_adv_clim.values]
    }
    coords = {"y": Lake_T.y, "x": Lake_T.x}
    TIO.save_newnc(
        savepath=f"{outcase}/MeanStates_Lake_Annual_{level}hPa.nc",
        in_dict=in_dict, coords=coords
    )

    in_dict = {
        'T': [["y", "x"], NoLake_T_clim.values],
        'U': [["y", "x"], NoLake_U_clim.values],
        'V': [["y", "x"], NoLake_V_clim.values],
        'WA': [["y", "x"], NoLake_warm_adv_clim.values]
    }
    coords = {"y": Lake_T.y, "x": Lake_T.x}
    TIO.save_newnc(
        savepath=f"{outcase}/MeanStates_NoLake_Annual_{level}hPa.nc",
        in_dict=in_dict, coords=coords
    )



def Plot_WarmAdvection(
                    caselist: list[str], level: int,
                    lon2d: np.ndarray, lat2d: np.ndarray,
                    checkmethod: str, OutDir:str, FigOutDir: str,
                    lkinfos: Any, onlysig: bool = True) -> None:
    """绘制热平流显著性变化图"""
    target = f"WarmAdvection"
    outcase = f"{OutDir}/{target}/{level}hPa"
    seasons = TU.get_seasons()
    FigOutDir_var = f'{FigOutDir}/{level}hPa/Single'
    os.makedirs(FigOutDir_var, exist_ok=True)
    oceanmask = lkinfos['ocean']
    """绘制热平流显著性变化图"""
    tasks = []
    varname = 'WA'
    var_info = TYCM.Variable_Infos(varname)
    levels = TYCM.Seasonal_PressureLevel_Cmap('WA')
    season_diffcfg = TPC.mapConfig(levs=levels['diff_maplevs_seasonal'], cmap=levels['diff_cmap'])
    season_rccfg = TPC.mapConfig(levs=levels['rc_maplevs_seasonal'], cmap=levels['rc_cmap'])
    annual_diffcfg = TPC.mapConfig(levs=levels['diff_maplevs_annual'], cmap=levels['diff_cmap'])
    annual_rccfg = TPC.mapConfig(levs=levels['rc_maplevs_annual'], cmap=levels['rc_cmap'])
    suffix = f"onlysig" if onlysig else "all"
    
    meancfg = TPC.mapConfig(levs=levels['maplevs_seasonal'], cmap=levels['cmap'])
    varInfo = TPC.varInfo(longname=var_info['longname'], unit=var_info['unit'], abbr=var_info['abbr'])
    for season in seasons:
        path_seasonal = f'{outcase}/Significance_{target}_{caselist[0]}-{caselist[1]}_seasonal_{season}_{checkmethod}.nc'
        xarr_seasonal = TIO.read_newnc(path_seasonal)
        savepath = f'{FigOutDir_var}/{target}_Diff_Map_{varname}_{season}_{suffix}_{checkmethod}.{FIGFMT}'
        tasks.append((TPAM.plot_spatial_diffmap_withsign, (xarr_seasonal, varname, season, target, lon2d, lat2d, 
                                                           checkmethod, lkinfos, season_diffcfg, onlysig, savepath)))
        savepath = f'{FigOutDir_var}/{target}_RC_Map_{varname}_{season}_{suffix}_{checkmethod}.{FIGFMT}'
        tasks.append((TPAM.plot_spatial_rcmap_withsign, (xarr_seasonal, varname, season, target, lon2d, lat2d, 
                                                         checkmethod, lkinfos, season_rccfg, onlysig, savepath)))
    path_annual = f'{outcase}/Significance_{target}_{caselist[0]}-{caselist[1]}_yearly_{checkmethod}.nc'
    xarr_annual = TIO.read_newnc(path_annual)
    savepath = f'{FigOutDir_var}/{target}_Diff_Map_{varname}_Annual_{suffix}_{checkmethod}.{FIGFMT}'
    tasks.append((TPAM.plot_spatial_diffmap_withsign, (xarr_annual, varname, "Annual", target, lon2d, lat2d, 
                                                       checkmethod, lkinfos, annual_diffcfg, onlysig, savepath)))
    savepath = f'{FigOutDir_var}/{target}_RC_Map_{varname}_Annual_{suffix}_{checkmethod}.{FIGFMT}'
    tasks.append((TPAM.plot_spatial_rcmap_withsign, (xarr_annual, varname, "Annual", target, lon2d, lat2d, 
                                                     checkmethod, lkinfos, annual_rccfg, onlysig, savepath)))
    label = f"{varInfo.longname} ({varInfo.unit})"
    savepath = f'{FigOutDir_var}/{target}_Diff_Map_{varname}_Annual_Colorbar_{suffix}_{checkmethod}.{FIGFMT}'
    tasks.append((TPCB.plot_spatial_cbar_core_V, (annual_diffcfg, 4, label, savepath)))
    savepath = f'{FigOutDir_var}/{target}_Diff_Map_{varname}_Seasonal_Colorbar_{suffix}_{checkmethod}.{FIGFMT}'
    tasks.append((TPCB.plot_spatial_cbar_core_V, (season_diffcfg, 4, label, savepath)))
    savepath = f'{FigOutDir_var}/{target}_RC_Map_{varname}_Annual_Colorbar_{suffix}_{checkmethod}.{FIGFMT}'
    tasks.append((TPCB.plot_spatial_cbar_core_V, (annual_rccfg, 4, "Relative Contribution (%)", savepath)))
    savepath = f'{FigOutDir_var}/{target}_RC_Map_{varname}_Seasonal_Colorbar_{suffix}_{checkmethod}.{FIGFMT}'
    tasks.append((TPCB.plot_spatial_cbar_core_V, (season_rccfg, 4, "Relative Contribution (%)", savepath)))

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




def Merge_WarmAdvectionSignificance(FigOutDir:str, caselist:list[str], level:int, checkmethod:str, onlysig: bool = True) -> None:
    """合并热平流显著性测试结果文件（季节+年际）为单一文件"""
    target = f"WarmAdvection"
    FigOutDir_var = f'{FigOutDir}/{level}hPa/Single'
    seasons = TU.get_seasons()
    mapcrop_params = TYCM.MapPlot_CropParams_noColorbar()
    figbox_space = TYCM.Merge_Fig_Space_Params()
    cbar_space = TYCM.MapCbar_CropParams_V()
    rosecrop_params = TYCM.RosePlot_CropParams_noColorbar()
    suffix = "onlysig" if onlysig else "all"
    varname = 'WA'
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
    savepath = f'{FigOutDir}/{level}hPa/{target}_{varname}_{level}hPa_Seasonal_Comparison.{FIGFMT}'
    seas_img.save(savepath, dpi=[DPI]*16)
