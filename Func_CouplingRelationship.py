import os
import time
import calendar
import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.feature_selection import mutual_info_regression
from joblib import Parallel, delayed
from typing import Literal, Optional, Tuple, Dict, Any, Sequence, List
# 自定义模块
import ToolBoxes.Utils as TU
import ToolBoxes.Tool_InputOutput as TIO
import ToolBoxes.Tool_ImageToolkit as TIT
import ToolBoxes.Tool_SignificanceTest as TST
import ToolBoxes.Tool_YangtzeColorMap as TYCM
import ToolBoxes.Tool_DataPrepare as TDP
import ToolBoxes.Tool_PlotColorBar as TPCB
import ToolBoxes.Tool_PlotAreaMap as TPAM
import ToolBoxes.Tool_PlotConfig as TPC
import ToolBoxes.Tool_CouplingTest as TCT
import ToolBoxes.Tool_PlotJoint as TPJ
import ToolBoxes.Tool_PlotKDE as TPKDE
FIGFMT = TPC.FIGFMT
DPI    = TPC.DPI_medium

HighThresPercentile = TPC.HighThresPercentile
LowThresPercentile  = TPC.LowThresPercentile
ThresWindows        = TPC.ThresWindows


def CalCouplingMetrics(casename: str, xarr: xr.Dataset, var1: str, var2: str, outdir:str) -> None:
    """
    计算温度与湿度之间的相关性
    """
    target = "CouplingMetrics"
    n_jobs = 96  # 并行任务数
    t0 = time.time()
    print(f"\nCalculating Coupling Metrics between {var1} and {var2} for case {casename}...")
    xarr_in = TU.xarray_leap_to_noleap(xarr)
    V1 = xarr_in[var1].values
    V2 = xarr_in[var2].values
    ntime, ny, nx = V1.shape
    timelist = xarr_in['time'].values

    V1 = V1.reshape(ntime, ny * nx)
    V2 = V2.reshape(ntime, ny * nx)

    # 计算相关性指标
    pearson_r_arr = np.full((ny * nx), np.nan)
    pearson_p_arr = np.full((ny * nx), np.nan)
    spearman_r_arr = np.full((ny * nx), np.nan)
    spearman_p_arr = np.full((ny * nx), np.nan)
    kendall_tau_arr = np.full((ny * nx), np.nan)
    kendall_p_arr = np.full((ny * nx), np.nan)
    mutual_info_arr = np.full((ny * nx), np.nan)
    lambda_u_arr = np.full((ny * nx), np.nan)
    cov_arr = np.full((ny * nx), np.nan)
    r2_arr = np.full((ny * nx), np.nan)
    # 构造所有任务的输入参数列表
    tasks = [
        (icell, V1[:, icell], V2[:, icell], timelist)
        for icell in range(ny * nx)
    ]

    with Parallel(n_jobs=n_jobs, backend="loky", return_as="generator") as parallel:
        gen = parallel(
            delayed(TCT.calc_coupling_metrics)(*task)
            for task in tasks
        )
        for res in tqdm(gen, total=ny * nx, desc='Calculating correlations', unit="cell",
                           dynamic_ncols=True, leave=False):
            icell, corr_info = res
            pearson_r_arr[icell] = corr_info.pearson_r
            pearson_p_arr[icell] = corr_info.pearson_p
            spearman_r_arr[icell] = corr_info.spearman_r
            spearman_p_arr[icell] = corr_info.spearman_p
            kendall_tau_arr[icell] = corr_info.kendall_tau
            kendall_p_arr[icell] = corr_info.kendall_p
            mutual_info_arr[icell] = corr_info.mutual_info
            lambda_u_arr[icell] = corr_info.lambda_u
            cov_arr[icell] = corr_info.cov
            r2_arr[icell] = corr_info.r2
    # 重塑为二维空间格点
    pearson_r_arr = pearson_r_arr.reshape(ny, nx)
    pearson_p_arr = pearson_p_arr.reshape(ny, nx)
    spearman_r_arr = spearman_r_arr.reshape(ny, nx)
    spearman_p_arr = spearman_p_arr.reshape(ny, nx)
    kendall_tau_arr = kendall_tau_arr.reshape(ny, nx)
    kendall_p_arr = kendall_p_arr.reshape(ny, nx)
    mutual_info_arr = mutual_info_arr.reshape(ny, nx)
    lambda_u_arr = lambda_u_arr.reshape(ny, nx)
    cov_arr = cov_arr.reshape(ny, nx)
    r2_arr = r2_arr.reshape(ny, nx)
    # 保存为 NetCDF 文件
    in_dict = {
            "pearson_r": (["y", "x"], pearson_r_arr),
            "pearson_p": (["y", "x"], pearson_p_arr),
            "spearman_r": (["y", "x"], spearman_r_arr),
            "spearman_p": (["y", "x"], spearman_p_arr),
            "kendall_tau": (["y", "x"], kendall_tau_arr),
            "kendall_p": (["y", "x"], kendall_p_arr),
            "mutual_info": (["y", "x"], mutual_info_arr),
            "lambda_u": (["y", "x"], lambda_u_arr),
            "cov": (["y", "x"], cov_arr),
            "r2": (["y", "x"], r2_arr),
        }
    coords={"x": xarr.x, "y": xarr.y}
    savepath = f"{outdir}/{target}_{casename}_{var1}_{var2}.nc"
    TIO.save_newnc(savepath=savepath, in_dict=in_dict, coords=coords)
    t1 = time.time()
    print(f"Correlation calculation for case {casename} completed in {t1 - t0:.2f} seconds.")




def CalCouplingMetrics_Analyze(caselist: List[str], var1:str, var2:str, outdir: str, lkinfos:Dict) -> None:
    """
    计算温度与湿度之间的相关性
    """
    target = "CouplingMetrics"
    case1, case2 = caselist
    case1path = f"{outdir}/{target}_{case1}_{var1}_{var2}.nc"
    case2path = f"{outdir}/{target}_{case2}_{var1}_{var2}.nc"
    case1_data = TIO.read_newnc(case1path)
    case2_data = TIO.read_newnc(case2path)

    # 计算差异
    diff_pearson_r = case1_data['pearson_r'].values - case2_data['pearson_r'].values
    diff_spearman_r = case1_data['spearman_r'].values - case2_data['spearman_r'].values
    diff_mutual_info = case1_data['mutual_info'].values - case2_data['mutual_info'].values
    diff_kendall_tau = case1_data['kendall_tau'].values - case2_data['kendall_tau'].values
    diff_lambda_u = case1_data['lambda_u'].values - case2_data['lambda_u'].values
    diff_cov = case1_data['cov'].values - case2_data['cov'].values
    diff_r2 = case1_data['r2'].values - case2_data['r2'].values
    # eps_pearson = 0.01 * np.nanstd(diff_pearson_r)
    # eps_spearman = 0.01 * np.nanstd(diff_spearman_r)
    # eps_mutual_info = 0.01 * np.nanstd(diff_mutual_info)
    # eps_r2 = 0.01 * np.nanstd(diff_r2)
    eps_pearson = 0.0
    eps_spearman = 0.0
    eps_mutual_info = 0.0
    eps_kendall_tau = 0.0
    eps_lambda_u = 0.0
    eps_cov = 0.0
    eps_r2 = 0.0

    # 应用海洋掩膜
    oceanmask = lkinfos['ocean']
    diff_pearson_r[oceanmask == 1] = np.nan
    diff_spearman_r[oceanmask == 1] = np.nan
    diff_mutual_info[oceanmask == 1] = np.nan
    diff_kendall_tau[oceanmask == 1] = np.nan
    diff_lambda_u[oceanmask == 1] = np.nan
    diff_cov[oceanmask == 1] = np.nan
    diff_r2[oceanmask == 1] = np.nan

    # 计算有效格点掩膜
    valid_mask = (oceanmask == 0)
    total = np.nansum(oceanmask == 0)

    def calc_ratio(diff, eps):
        inc = np.nansum((diff >  eps) & valid_mask)
        dec = np.nansum((diff < -eps) & valid_mask)
        no  = np.nansum((np.abs(diff) <= eps) & valid_mask)
        return inc/total, dec/total, no/total

    # 计算每种指数增加，减少，不变的格点比例
    df = pd.DataFrame(columns=['Increase', 'Decrease', 'NoChange'],
                      index=['Pearson_r', 'Spearman_r', 'Mutual_Info', 'R2', 
                            'Kendall_tau', 'Lambda_u', 'Cov'])
    df.loc['Pearson_r'] = calc_ratio(diff_pearson_r, eps_pearson)
    df.loc['Spearman_r'] = calc_ratio(diff_spearman_r, eps_spearman)
    df.loc['Mutual_Info'] = calc_ratio(diff_mutual_info, eps_mutual_info)
    df.loc['Kendall_tau'] = calc_ratio(diff_kendall_tau, eps_kendall_tau)
    df.loc['Lambda_u'] = calc_ratio(diff_lambda_u, eps_lambda_u)
    df.loc['Cov'] = calc_ratio(diff_cov, eps_cov)
    df.loc['R2'] = calc_ratio(diff_r2, eps_r2)

    print(f"\nCoupling Metrics Analysis between {var1} and {var2}:")
    print(df)
    df.to_csv(f"{outdir}/{target}_Analyze_{case1}_vs_{case2}_{var1}_{var2}.csv")

    in_dict = {
            "diff_pearson_r": (["y", "x"], diff_pearson_r),
            "diff_spearman_r": (["y", "x"], diff_spearman_r),
            "diff_mutual_info": (["y", "x"], diff_mutual_info),
            "diff_kendall_tau": (["y", "x"], diff_kendall_tau),
            "diff_lambda_u": (["y", "x"], diff_lambda_u),
            "diff_cov": (["y", "x"], diff_cov),
            "diff_r2": (["y", "x"], diff_r2),
        }
    coords={"x": case1_data.x, "y": case1_data.y}
    savepath = f"{outdir}/{target}_Analyze_{case1}_vs_{case2}_{var1}_{var2}.nc"
    TIO.save_newnc(savepath=savepath, in_dict=in_dict, coords=coords)



def CalUpperTailDependence(casename: str, refname: str, event1: str, event2: str, DataOutDir:str, outdir:str) -> None:
    """
    计算两个事件之间的上尾依赖系数
    """
    target = "UpperTailDependence"
    n_jobs = 96  # 并行任务数
    t0 = time.time()
    print(f"\nCalculating Upper Tail Dependence between {event1} and {event2} for case {casename}...")
    xarr = TIO.read_newnc(f"{DataOutDir}/ExtremeAnalysis/Extreme_Events_{casename}_ref_{refname}_identified.nc")
    Event1 = xarr[f'{event1}_Flag'].values
    Event2 = xarr[f'{event2}_Flag'].values
    ntime, ny, nx = Event1.shape
    Event1 = Event1.reshape(ntime, ny * nx)
    Event2 = Event2.reshape(ntime, ny * nx)

    # 计算上尾依赖系数
    utdc_arr = np.full((ny * nx), np.nan)
    # 构造所有任务的输入参数列表
    tasks = [
        (icell, Event1[:, icell], Event2[:, icell])
        for icell in range(ny * nx)
    ]

    with Parallel(n_jobs=n_jobs, backend="loky", return_as="generator") as parallel:
        gen = parallel(
            delayed(TCT.calc_upper_tail_dependence_coefficient)(*task)
            for task in tasks
        )
        for res in tqdm(gen, total=ny * nx, desc='Calculating upper tail dependence', unit="cell",
                           dynamic_ncols=True, leave=False):
            icell, utdc = res
            utdc_arr[icell] = utdc
    # 重塑为二维空间格点
    utdc_arr = utdc_arr.reshape(ny, nx)
    # 保存为 NetCDF 文件
    in_dict = {
            "utdc": (["y", "x"], utdc_arr),
        }
    coords={"x": xarr.x, "y": xarr.y}
    savepath = f"{outdir}/{target}_{casename}_{event1}-{event2}.nc"
    TIO.save_newnc(savepath=savepath, in_dict=in_dict, coords=coords)
    t1 = time.time()
    print(f"Upper Tail Dependence calculation for case {casename} completed in {t1 - t0:.2f} seconds.")




def TailDependence_Analyze(caselist: List[str], event1:str, event2:str, outdir: str, lkinfos:Dict) -> None:
    """
    计算两个事件之间的上尾依赖系数差异
    """
    target = "UpperTailDependence"
    case1, case2 = caselist
    case1path = f"{outdir}/{target}_{case1}_{event1}-{event2}.nc"
    case2path = f"{outdir}/{target}_{case2}_{event1}-{event2}.nc"
    case1_data = TIO.read_newnc(case1path)
    case2_data = TIO.read_newnc(case2path)

    # 计算差异
    diff_utdc = case1_data['utdc'].values - case2_data['utdc'].values

    # 应用海洋掩膜
    oceanmask = lkinfos['ocean']
    diff_utdc[oceanmask == 1] = np.nan

    # 计算有效格点掩膜
    valid_mask = (oceanmask == 0)
    total = np.nansum(oceanmask == 0)

    inc = np.nansum((diff_utdc >  0) & valid_mask)
    dec = np.nansum((diff_utdc <  0) & valid_mask)
    no  = np.nansum((diff_utdc == 0) & valid_mask)

    df = pd.DataFrame(columns=['Increase', 'Decrease', 'NoChange'],
                      index=['UTDC'])
    df.loc['UTDC'] = [inc/total, dec/total, no/total]

    print(f"\nUpper Tail Dependence Analysis between {event1} and {event2}:")
    print(df)
    df.to_csv(f"{outdir}/{target}_Analyze_{case1}_vs_{case2}_{event1}_{event2}.csv")

    in_dict = {
            "diff_utdc": (["y", "x"], diff_utdc),
        }
    coords={"x": case1_data.x, "y": case1_data.y}
    savepath = f"{outdir}/{target}_Analyze_{case1}_vs_{case2}_{event1}_{event2}.nc"
    TIO.save_newnc(savepath=savepath, in_dict=in_dict, coords=coords)



def Plot_CouplingMetrics_Analyze(caselist: List[str], var1:str, var2:str,
                             lon2d: np.ndarray, lat2d: np.ndarray,
                             lkinfos:Dict,  outdir: str, figoutdir:str) -> None:
    """
    绘制温度与湿度之间的相关性差异图
    """
    target = "CouplingMetrics"
    case1, case2 = caselist
    casepath = f"{outdir}/{target}_Analyze_{case1}_vs_{case2}_{var1}_{var2}.nc"
    cdata = TIO.read_newnc(casepath)
    hotwetpath = f"{outdir}/../ExtremeAnalysis/Significance_ExtremeEvent_Freq_Lake-NoLake_HotWet_Freq_yearly_Wilcoxon_signed-rank_test.nc"
    hotwet = TIO.read_newnc(hotwetpath)
    sig_mask = hotwet['p_value'].values
    sig_mask = (sig_mask < 0.05)
    oceanmask = lkinfos['ocean']
    mask = (oceanmask == 0)
    figoutdir_var = f'{figoutdir}/Single'
    os.makedirs(figoutdir_var, exist_ok=True)
    tasks = []
    # matrics = ['pearson_r', 'spearman_r', 'mutual_info', 'r2', 'kendall_tau', 'lambda_u', 'cov']
    matrics = ['lambda_u']
    for matric in matrics:
        dataname = f'diff_{matric}'
        diffdata = cdata[dataname].values
        diffdata[oceanmask == 1] = np.nan

        # 绘图
        diff_levels = TYCM.Coupling_Change_Cmap(matric)
        var_info = TYCM.Coupling_Infos(matric)
        varInfo = TPC.varInfo(longname=var_info['longname'], unit=var_info['bunit'], abbr=var_info['abbr'])
        mapcfg = TPC.mapConfig(levs=diff_levels['levels'], cmap=diff_levels['cmap'])
        savepath = f"{figoutdir_var}/{target}_{case1}-{case2}_{var1}-{var2}_{matric}.{FIGFMT}"
        tasks.append((TPAM.plot_categorical_map, (diffdata, target, lon2d, lat2d, savepath,
                                                  None, None, lkinfos,
                                                  mapcfg, sig_mask)))

        # 绘制显著性掩膜图
        diffdata_sig = diffdata.copy()
        diffdata_sig[~sig_mask] = np.nan
        diffdata_sig = diffdata_sig[np.isfinite(diffdata_sig)].flatten()
        n_points = np.nansum(np.isfinite(diffdata_sig))
        print(f"  ➠ {matric}: Total valid points: {n_points}")
        data_up_pct = np.nansum(diffdata_sig >= 0) / n_points * 100
        data_down_pct = np.nansum(diffdata_sig < 0) / n_points * 100
        print(f"    ➠ {matric}: Significant Increase: {data_up_pct:.2f}%, Significant Decrease: {data_down_pct:.2f}%")
        hist_savepath = f'{figoutdir_var}/{target}_{case1}-{case2}_{var1}-{var2}_{matric}_Hist.{FIGFMT}'
        tasks.append((TPKDE.plot_kde_1d, (diffdata_sig, hist_savepath)))

        label = f"{varInfo.longname} ({varInfo.abbr})"
        savepath = f'{figoutdir_var}/{target}_{case1}-{case2}_{var1}-{var2}_{matric}_VColorbar.{FIGFMT}'
        tasks.append((TPCB.plot_spatial_cbar_core_V, (mapcfg, 6.0, label, savepath, 14, 18, 0.04, 'both')))               

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





def Merge_CouplingMetrics_Cases(caselist: List[str], var1:str, var2:str,
                             figoutdir: str) -> None:
    """
    合并不同实验的相关性结果为一个多案例文件
    """
    target = "CouplingMetrics"
    case1, case2 = caselist
    map_cropparams = TYCM.MapPlot_CropParams_noColorbar()
    cbar_cropparams = TYCM.Zero_space_CropParams()
    figbox_space = TYCM.Merge_Fig_Space_Params()
    cols_space = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
    rows_space = [0.01, 0.02, 0.01, 0.02, 0.01, 0.02]
    fig_rows = [["pearson_r",   "spearman_r",   "r2"  ],
                ["pearson_r_HColorbar", "spearman_r_HColorbar", "r2_HColorbar",],
                ["mutual_info", "kendall_tau", "lambda_u"],
                ["mutual_info_HColorbar","kendall_tau_HColorbar", "lambda_u_HColorbar"],
                ["cov", ],
                ["cov_HColorbar", ]
               ]
    for i, row in enumerate(fig_rows):
        for j, col in enumerate(row):
            figpath = f"{figoutdir}/Single/{target}_{case1}-{case2}_{var1}-{var2}_{col}.{FIGFMT}"
            if 'HColorbar' in col:
                crop_params = cbar_cropparams
            else:
                crop_params = map_cropparams
            cropped = TIT.crop_image_from_path(figpath, crop_params=crop_params, mode="ratio")
            fig_rows[i][j] = cropped

    img = TIT.merge_images_Row(
        rows_images=fig_rows,
        cols_space=cols_space,
        rows_space=rows_space,
        box_space=figbox_space,
        background_color='#FFFFFFFF',
        space_mode="ratio",
        alignment=["justify", "justify", "justify", "justify", "justify", "justify"],
        draw_ticks=False, tick_step=0.01,
        font_path="/share/home/dq048/.local/share/fonts/NotoSans-Bold.ttf",
    )
    savepath = f'{figoutdir}/{target}_{case1}-{case2}_{var1}-{var2}_Merged.{FIGFMT}'
    img.save(savepath, dpi=[DPI]*16)



def Normalization_Variable(casename: str, xarr_in: xr.Dataset, var: str, timefreq: str, outdir:str) -> None:
    """对变量进行归一化处理"""
    target = "Normalization"
    n_jobs = 96  # 并行任务数
    t0 = time.time()
    print(f"\nNormalization of variable {var} for case {casename}...")
    xarr = TU.xarray_leap_to_noleap(xarr_in)
    arr = xarr[var].values
    timelist = xarr['time'].values
    
    ntime, ny, nx = arr.shape
    arr = arr.reshape(ntime, ny * nx)
    norm_arr = np.full((ntime, ny * nx), np.nan)

    # 构造所有任务的输入参数列表
    tasks = [
        (icell, arr[:, icell], timelist)
        for icell in range(ny * nx)
    ]

    # # 绘制 RH 和 T2m 的直方图
    # plt.figure(figsize=(8, 6))
    # icell = 1000  # 示例格点索引
    # plt.hist(arr[:, icell], bins=20, alpha=0.7, color='blue', label='RH')
    # plt.legend()
    # plt.savefig(f"{target}_{icell}_Histogram_Site.{FIGFMT}", dpi=DPI)
    # plt.close()
    # raise 


    with Parallel(n_jobs=n_jobs, backend="loky", return_as="generator") as parallel:
        gen = parallel(
            delayed(TCT.get_norm_var)(*task)
            for task in tasks
        )
        for res in tqdm(gen, total=ny * nx, desc='Calculating correlations', unit="cell",
                           dynamic_ncols=True, leave=False):
            icell, arr_norm = res
            norm_arr[:, icell] = arr_norm

    # 重塑为二维空间格点
    norm_arr = norm_arr.reshape(ntime, ny, nx)
    # 保存为 NetCDF 文件
    in_dict = {
            var: (["time", "y", "x"], norm_arr),
        }
    coords={"time": xarr.time, "x": xarr.x, "y": xarr.y}
    savepath = f"{outdir}/{target}_{casename}_{var}_{timefreq}_normalized.nc"
    TIO.save_newnc(savepath=savepath, in_dict=in_dict, coords=coords)
    t1 = time.time()
    print(f"Normalization for case {casename} completed in {t1 - t0:.2f} seconds.")



def Plot_Coupling_state(caselist:List, var1: str, var2: str, timefreq: str, checkmethod: str,
                        lon2d: np.ndarray, lat2d: np.ndarray,
                        OutDir: str, FigOutDir:str, lkinfos: Dict, onlysig: bool = True)->None:
    """绘制耦合状态图"""
    case1, case2 = caselist
    case1path_var1 = f"{OutDir}/CouplingTest/Normalization_{case1}_{var1}_{timefreq}_normalized.nc"
    case1path_var2 = f"{OutDir}/CouplingTest/Normalization_{case1}_{var2}_{timefreq}_normalized.nc"
    case2path_var1 = f"{OutDir}/CouplingTest/Normalization_{case2}_{var1}_{timefreq}_normalized.nc"
    case2path_var2 = f"{OutDir}/CouplingTest/Normalization_{case2}_{var2}_{timefreq}_normalized.nc"
    case1_data_var1 = TIO.read_newnc(case1path_var1)
    case1_data_var2 = TIO.read_newnc(case1path_var2)
    case2_data_var1 = TIO.read_newnc(case2path_var1)
    case2_data_var2 = TIO.read_newnc(case2path_var2)

    # Helper to load Hot, Wet, HotWet
    def load_event(event):
        target = "Freq"
        outpath=f'{OutDir}/ExtremeAnalysis/Significance_ExtremeEvent_{target}_{caselist[0]}-{caselist[1]}_{event}_{target}_yearly_{checkmethod}.nc'
        xarr = TIO.read_newnc(outpath)
        rejected, diffsign = TDP.preparation_for_probability(xarr, checkmethod, lkinfos)
        return {"sign": rejected, "diff": diffsign}
    outpath=f'{OutDir}/ExtremeAnalysis/Significance_ExtremeEvent_Freq_{caselist[0]}-{caselist[1]}_HotWet_Freq_yearly_{checkmethod}.nc'
    xarr = TIO.read_newnc(outpath)
    rcdata, sig_mask, dash_mask, strong_mask, weak_mask, suffix, area_all, area_strong = TDP.prepara_for_mapplot(xarr, lkinfos, "RC_overall", checkmethod, onlysig)

    Hot_State     = load_event("Hot")
    Wet_State     = load_event("Wet")
    HotWet_State  = load_event("HotWet")
    HotWet_up   = (HotWet_State["sign"] == 1) & (HotWet_State["diff"] == 1)
    Hot_down   = (Hot_State["sign"] == 1) & (Hot_State["diff"] == -1)
    Wet_up     = (Wet_State["sign"] == 1) & (Wet_State["diff"] == 1)

    # Find sites with HotWet up, Hot down, Wet up
    # coupling = (Hot_down & Wet_up & HotWet_up & strong_mask)
    coupling = (Hot_down & Wet_up & HotWet_up)
    sites = np.argwhere(coupling)  # coupling 为布尔数组
    # 打印站点信息
    print(len(sites))
    print(HotWet_up.sum(), Hot_down.sum(), Wet_up.sum())

    case1_var1_sites = []
    case1_var2_sites = []
    case2_var1_sites = []
    case2_var2_sites = []

    for y, x in sites:
        case1_var1_sites.append(case1_data_var1[var1].values[:, y, x])
        case1_var2_sites.append(case1_data_var2[var2].values[:, y, x])
        case2_var1_sites.append(case2_data_var1[var1].values[:, y, x])
        case2_var2_sites.append(case2_data_var2[var2].values[:, y, x])

    case1_var1_sites = np.array(case1_var1_sites)
    case1_var2_sites = np.array(case1_var2_sites)
    case2_var1_sites = np.array(case2_var1_sites)
    case2_var2_sites = np.array(case2_var2_sites)

    df_big = pd.DataFrame({
        var1: np.concatenate([case1_var1_sites.flatten(), case2_var1_sites.flatten()]),
        var2: np.concatenate([case1_var2_sites.flatten(), case2_var2_sites.flatten()]),
        "Case": [case1]*case1_var1_sites.size + [case2]*case2_var1_sites.size
    })

    print(df_big.head())

    palette = {
        'Lake': '#9F7D7C',   
        'NoLake': '#FA1E1C'   
    }
    percentile_high = 0.85
    percentile_low  = 0.15

    target = "CouplingState"

    # # # 绘制 KDE 密度图
    # savepath = f"{FigOutDir}/{target}_{var1}_{var2}_Coupling_State_All_Sites_with_KDE.{FIGFMT}"
    # TPJ.plot_Advance_Joint(df_big, var2, var1, "Type", palette, savepath, percentile_high, percentile_low, case2)

    tasks = []
    sites = sites[0:10]
    for y, x  in sites:
        case1_var1_site = case1_data_var1[var1].values[:, y, x]
        case1_var2_site = case1_data_var2[var2].values[:, y, x]
        case2_var1_site = case2_data_var1[var1].values[:, y, x]
        case2_var2_site = case2_data_var2[var2].values[:, y, x]
        lat = lat2d[y, x]
        lon = lon2d[y, x]

        df_in = pd.DataFrame({
            var1: np.concatenate([case1_var1_site, case2_var1_site]),
            var2: np.concatenate([case1_var2_site, case2_var2_site]),
            'Type': ['Lake'] * len(case1_var1_site) + ['NoLake'] * len(case2_var1_site)
        })
        savepath = f"{FigOutDir}/{target}_{var1}_{var2}_Coupling_State_Site_x{x}_y{y}_with_KDE.{FIGFMT}"
        tasks.append((TPJ.plot_Advance_Joint, (df_in, var2, var1,  "Type", palette, savepath, percentile_high, percentile_low, case2, lon, lat))) 

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






