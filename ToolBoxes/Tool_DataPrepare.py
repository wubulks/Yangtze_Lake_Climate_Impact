import numpy as np
import xarray as xr
import pandas as pd
from typing import Any, List, Union
from dataclasses import dataclass
import ToolBoxes.Utils as Utils


#*********************画图配置类*******************************#
@dataclass
class mapConfig:
    diff_maplevs: List[float]
    rc_maplevs: List[float]
    diff_cmap: Any
    rc_cmap: Any

@dataclass
class boxConfig:
    diff_boxlevs: List[float]
    rc_boxlevs: List[float]

@dataclass 
class roseConfig:
    roselevs: List[float]
    colors_dict: Any
#*************************************************************#



def prepara_for_mapplot(xarr_in, lkinfos, varname, checkmethod, onlysig=True):
    """
    地图绘图前的数据准备
    """
    # 读取必要信息
    oceanmask    = lkinfos['ocean']
    dist_km_all  = (lkinfos['dist_m_all'] / 1000.0)  # m -> km
    area_km2     = lkinfos['area'] / (1000*1000)

    # 读取数据并做海洋掩膜 + 显著性筛选
    data_sel     = xarr_in[varname].values
    data_sel[oceanmask == 1] = np.nan

    # 显著性掩膜
    method   = xarr_in['checkmethod'].values
    # 只保留 t 检验显著性
    # rejected = xarr_in['rejected'].values
    # rejected = np.where(rejected == 1, 1, 0)
    rejected = xarr_in['p_value'].values
    rejected = np.where(rejected <= 0.05, 1, 0)
    rejected[oceanmask == 1] = 0

    if checkmethod == "auto":
        rejected[method != 1] = 0  # 只保留 t 检验显著性
    if checkmethod == "Paired_t-test":
        rejected[method != 1] = 0  # 只保留 t 检验显著性
    elif checkmethod == "Wilcoxon_signed-rank_test":
        rejected[method != 2] = 0  # 只保留Wilcoxon显著性
    elif checkmethod == "Paired_bootstrap":
        rejected[method != 4] = 0  # 只保留bootstrap显著性
    else:  # "t_test"
        raise ValueError("checkmethod must be 'auto' or 'Paired_bootstrap' or 'Paired_t-test' or 'Wilcoxon_signed-rank_test'")

    # 显著性筛选
    if onlysig:
        sig_mask = np.isfinite(data_sel) & np.isfinite(rejected) & (rejected > 0)
        data = np.where(sig_mask, data_sel, np.nan)
        suffix = "onlysig"
    else:
        data = data_sel
        suffix = "all"

    # 有效性筛选（与距离共同为有限）
    validmask = np.isfinite(data) & np.isfinite(dist_km_all)
    data_1d   = np.asarray(data[validmask]).ravel()
    data_1d_abs  = np.abs(data_1d)
    data_abs    = np.abs(data)
    data_q  = np.nanpercentile(data_1d_abs, 75)
    strong_mask = (data_abs >= data_q) & validmask
    weak_mask  = (data_abs  < data_q) & validmask

    valid_sig = np.isfinite(area_km2) & sig_mask
    area_all = np.nansum(area_km2[valid_sig].ravel())
    area_strong = np.nansum(area_km2[valid_sig & strong_mask].ravel())
    area_all = area_all / 10000
    area_strong = area_strong / 10000

    #------------ 基础设置 ------------
    dash_mask = np.isfinite(data) & np.isfinite(strong_mask) & strong_mask
    dash_mask = Utils.smooth_data(dash_mask.astype(float), sigma=1.1, min_area=100, cell_area=area_km2)

    return data, sig_mask, dash_mask, strong_mask, weak_mask, suffix, area_all, area_strong



def prepara_for_boxplot(xarr_in, lkinfos, varname, checkmethod, onlysig=True):
    """
    地图绘图前的数据准备
    """
    # 读取必要信息
    oceanmask    = lkinfos['ocean']
    dist_km_all  = (lkinfos['dist_m_all'] / 1000.0)  # m -> km
    area_km2     = lkinfos['area'] / (1000*1000)

    # 读取数据并做海洋掩膜 + 显著性筛选
    data_sel     = xarr_in[varname].values
    data_sel[oceanmask == 1] = np.nan

    # 显著性掩膜
    rejected = xarr_in['p_value'].values
    rejected = np.where(rejected <= 0.05, 1, 0)
    rejected[oceanmask == 1] = 0
    # 只保留 t 检验显著性
    method   = xarr_in['checkmethod'].values
    if checkmethod == "auto":
        rejected[method != 1] = 0  # 只保留 t 检验显著性
    if checkmethod == "Paired_t-test":
        rejected[method != 1] = 0  # 只保留 t 检验显著性
    elif checkmethod == "Wilcoxon_signed-rank_test":
        rejected[method != 2] = 0  # 只保留Wilcoxon显著性
    elif checkmethod == "Paired_bootstrap":
        rejected[method != 4] = 0  # 只保留bootstrap显著性
    else:  # "t_test"
        raise ValueError("checkmethod must be 'auto' or 'Paired_bootstrap' or 'Paired_t-test' or 'Wilcoxon_signed-rank_test'")

    # 显著性筛选
    if onlysig:
        sig_mask = np.isfinite(data_sel) & np.isfinite(rejected) & (rejected > 0)
        data = np.where(sig_mask, data_sel, np.nan)
        suffix = "onlysig"
    else:
        data = data_sel
        suffix = "all"

    # 有效性筛选（与距离共同为有限）
    validmask = np.isfinite(data) & np.isfinite(dist_km_all)
    data_1d   = np.asarray(data[validmask]).ravel()
    data_1d_abs  = np.abs(data_1d)
    data_q  = np.nanpercentile(data_1d_abs, 75)
    strong_mask_1d = (data_1d_abs >= data_q)
    weak_mask_1d  = (data_1d_abs  < data_q)

    data_strong = data_1d[strong_mask_1d]
    data_weak   = data_1d[weak_mask_1d]

    valid_sig = np.isfinite(area_km2) & sig_mask
    area_all = np.nansum(area_km2[valid_sig].ravel())
    area_all = area_all / 10000

    if area_all < 1:
        print(f"Warning: The significant area is too small ({area_all:.2f} x10^4 km^2). The boxplot may be unreliable.")
        data_1d = np.array([np.nan])
        data_strong = np.array([np.nan])
        data_weak = np.array([np.nan])
    
    return data_1d, data_strong, data_weak, suffix



def sel_valid_data(data, whis):
    g = data.groupby('Season')['Value']
    q1 = g.quantile(0.25)
    q3 = g.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - whis * iqr
    upper = q3 + whis * iqr

    # 把上下界并回原表，筛内点
    bounds = pd.DataFrame({'lower': lower, 'upper': upper})
    df_in = data.join(bounds, on='Season')
    df_in = df_in[(df_in['Value'] >= df_in['lower']) & (df_in['Value'] <= df_in['upper'])]
    return df_in



def prepara_for_boxplot_seasonal(xarr_seasonal, lkinfos, checkmethod, onlysig=True):
    seasons = Utils.get_seasons()
    diff_data_list = []
    rc_data_list = []
    for season in seasons:
        # 判断xarr_sel类型是否为字典
        if isinstance(xarr_seasonal, dict):
            xarr_sel = xarr_seasonal[season]
        else:
            xarr_sel = xarr_seasonal.sel(season=season)
        # 频率差异
        diff_1d, data_strong, data_weak, suffix = prepara_for_boxplot(xarr_sel, lkinfos, 'mean_diff', checkmethod, onlysig=onlysig)
        diff_data_list.append(pd.DataFrame({'Season': season, 'Value': diff_1d}))
        # 相对贡献
        rc_1d, data_strong, data_weak, suffix = prepara_for_boxplot(xarr_sel, lkinfos, 'RC_overall', checkmethod, onlysig=onlysig)
        rc_data_list.append(pd.DataFrame({'Season': season, 'Value': rc_1d}))
    # 合并为长格式
    diff_df = pd.concat(diff_data_list, ignore_index=True)
    rc_df   = pd.concat(rc_data_list, ignore_index=True)
    return diff_df, rc_df, suffix



def prepara_for_boxplot_annual(xarr_events, eventslist, lkinfos, checkmethod, onlysig=True):
    diff_data_list = []
    rc_data_list = []
    for event in eventslist:
        xarr_sel = xarr_events[event]
        event_name = Utils.standardize_event_name(event)
        # 频率差异
        diff_1d, data_strong, data_weak, suffix = prepara_for_boxplot(xarr_sel, lkinfos, 'mean_diff', checkmethod, onlysig=onlysig)
        diff_data_list.append(pd.DataFrame({'Season': event_name, 'Value': diff_1d}))
        # 相对贡献
        rc_1d, data_strong, data_weak, suffix = prepara_for_boxplot(xarr_sel, lkinfos, 'RC_overall', checkmethod, onlysig=onlysig)
        rc_data_list.append(pd.DataFrame({'Season': event_name, 'Value': rc_1d}))
    # 合并为长格式
    diff_df = pd.concat(diff_data_list, ignore_index=True)
    rc_df   = pd.concat(rc_data_list, ignore_index=True)
    return diff_df, rc_df, suffix



def prepara_for_diurnal_rose(ds_dict, lkinfos, varname, checkmethod, onlysig=True):
    keep_hours = Utils.get_all_hours()

    # 读取必要信息
    oceanmask    = lkinfos['ocean']
    dist_km_all  = (lkinfos['dist_m_all'] / 1000.0)  # m -> km
    area_km2     = lkinfos['area'] / (1000*1000)
    
    # 只取显著格点（可选）
    in_df = {}
    for hour in keep_hours:
        h_bjt_str = Utils.UTC_to_BJT_str(hour)
        bjt_hour = Utils.UTC_to_BJT(hour)
        in_df[bjt_hour] = {}
        diff = ds_dict[h_bjt_str]['mean_diff'].values
        diff[oceanmask == 1] = np.nan
        rejected = ds_dict[h_bjt_str]['p_value'].values
        rejected = np.where(rejected <= 0.05, 1, 0)
        rejected[oceanmask == 1] = 0
        method = ds_dict[h_bjt_str]['checkmethod'].values
        if checkmethod == "auto":
            rejected[method != 1] = 0  # 只保留 t 检验显著性
        if checkmethod == "Paired_t-test":
            rejected[method != 1] = 0  # 只保留 t 检验显著性
        elif checkmethod == "Wilcoxon_signed-rank_test":
            rejected[method != 2] = 0  # 只保留Wilcoxon显著性
        elif checkmethod == "Paired_bootstrap":
            rejected[method != 4] = 0  # 只保留bootstrap显著性
        else:  # "t_test"
            raise ValueError("checkmethod must be 'auto' or 'Paired_bootstrap' or 'Paired_t-test' or 'Wilcoxon_signed-rank_test'")

        # # 显著性筛选
        # if varname in ['Prec', 'Prec-Max', 'Prec-Min']:
        #     # 不采用rejected作为掩模
        #     onlysig = False
        #     rejected = np.where(np.isfinite(diff), 1, np.nan)
        if onlysig:
            sig_mask = np.isfinite(diff) & np.isfinite(rejected) & (rejected > 0)
            data = np.where(sig_mask, diff, np.nan)
            suffix = "onlysig"
        else:
            data = diff
            suffix = "all"

        # 共同有效性筛选（两者都必须是有限值）
        validmask = np.isfinite(data) & np.isfinite(dist_km_all)
        data_1d = np.asarray(data[validmask]).ravel()
        dist_km = np.asarray(dist_km_all[validmask]).ravel()
        data_1d_abs  = np.abs(data_1d)
        data_abs    = np.abs(data)
        data_q  = np.nanpercentile(data_1d_abs, 75)
        strong_mask = (data_abs >= data_q) & validmask
        weak_mask  = (data_abs  < data_q) & validmask

        strong_mask_1d = (data_1d_abs >= data_q)
        weak_mask_1d  = (data_1d_abs  < data_q)
        strongdis = dist_km_all[strong_mask]
        meanstrongdis = round(np.nanmean(strongdis), 1)
        if data_1d.size >= 10:
            # 距离分区（不重叠）
            strong_mean = np.nanmean(data_1d[strong_mask_1d])
            weak_mean = np.nanmean(data_1d[weak_mask_1d])
            in_df[bjt_hour]['Strong'] = strong_mean

    in_dict = {
        'Strong':   ('#f38181', 'o'),
        # 'Weak':    ('#625772', 'D'),
    }
    in_df = pd.DataFrame(in_df).T
    in_df = in_df.sort_index()

    return in_df, sig_mask, strong_mask, weak_mask, suffix




def preparation_for_probability(xarr_in, checkmethod, lkinfos):
    """
    返回：
        rejected: 1=显著，0=不显著
        diffsign: -1/0/1 表示差异方向（但不参与显著性判定）
    """
    oceanmask    = lkinfos['ocean']

    # 显著性（p<=0.05）
    rejected = xarr_in['p_value'].values
    rejected = np.where(rejected <= 0.05, 1, 0)

    # 筛选统计方法
    method = xarr_in['checkmethod'].values
    if checkmethod == "auto":
        rejected[method != 1] = 0
    elif checkmethod == "Wilcoxon_signed-rank_test":
        rejected[method != 2] = 0
    elif checkmethod == "Paired_bootstrap":
        rejected[method != 4] = 0
    else:
        raise ValueError("checkmethod must be 'auto' or 'Paired_bootstrap'")

    rejected[oceanmask == 1] = -1

    # 差异方向（不用于显著性，只用于趋势方向判断）
    meandiff = xarr_in['mean_diff'].values
    diffsign = np.zeros_like(meandiff, dtype=int)
    diffsign[meandiff > 0] = 1
    diffsign[meandiff < 0] = -1
    diffsign[oceanmask == 1] = -999   # 海洋
    # 注意：diff=0 不代表不显著，只代表没有均值变化

    return rejected, diffsign



def preparation_for_radial_histogram(        
        df: pd.DataFrame,
        trend: str,
        pry_name: str | None = None,
    ) -> pd.DataFrame:
    
    """
    为径向直方图绘图准备数据
    """
    popname = f"{trend}_pop"
    popname_K = f"{popname}_K"
    pctname = f"{trend}_pct"

    df[popname_K] = df[popname] / 1_000  # 转为百万
    # 去除那些column_name值为0的行
    df = df[df[popname_K] > 1]  # 只保留大于1k的
    df = df[df[pctname] > 1]  # 只保留大于1%的
    pry_df = df.copy()

    # 如果提供了主要分类名称，则生成一个主要分类的总和，新的dataframe
    if pry_name is not None:
        pry_df = pry_df.groupby('pr_name_en')[popname_K].sum().reset_index()
        pry_df = pry_df.sort_values(by=popname_K, ascending=False)

    if pry_name is not None:
        return df, pry_df
    else:
        return df

