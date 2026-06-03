import os
import time
import calendar
import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm
import statsmodels.api as sm
from multiprocessing import Pool
from scipy.stats import pearsonr
from dataclasses import dataclass
from joblib import Parallel, delayed
from typing import Literal, Optional, Tuple, Dict, Any, Sequence, List

# 自定义模块
import ToolBoxes.Tool_InputOutput as TIO
import ToolBoxes.Tool_WetBulbTemperature as WBT
import ToolBoxes.Tool_PlotConfig as TPC

HighThresPercentile = TPC.HighThresPercentile
LowThresPercentile  = TPC.LowThresPercentile
ThresWindows        = TPC.ThresWindows

Flag_Numpy_prctile = True


"""
极端事件指标计算库
================================================
"""


def reshape_data_yearly(data, timelist):
    """
    将一维数据按年重塑为 (day_of_year, year) 的表格：
    - 自动补全每年 1/1~12/31 的日期（缺测用 NaN 填充）
    - 删除所有闰年的 2 月 29
    - 输出一个 DataFrame，行索引是 1~365（day_of_year），列是每个年份（字符串）
    """
    # 1. 构造 DataFrame 并设 time 为索引
    df = pd.DataFrame({
        'data': data,
        'time': pd.to_datetime(timelist)
    }).set_index('time')
    # 2. 生成从最早年 1/1 到最晚年 12/31 的完整日期索引
    start = df.index.min().replace(month=1, day=1)
    end   = df.index.max().replace(month=12, day=31)
    full_idx = pd.date_range(start=start, end=end, freq='D')
    # 3. 重索引补全，删掉 2 月 29
    df_full = df.reindex(full_idx)
    mask = ~((df_full.index.month == 2) & (df_full.index.day == 29))
    df_full = df_full.loc[mask]
    out_data = df_full.copy()  # 备份原始数据
    # 4. 提取年和年内天序
    df_full = df_full.assign(
        year        = df_full.index.year,
        day_of_year = df_full.index.dayofyear
    )
    # 5. 构造结果表格：行 1~365，列为各年
    years = np.sort(df_full['year'].unique())
    out_df = pd.DataFrame(index=range(1, 366))
    for y in years:
        # 取出该年，按日期顺序保证长度为 365
        seq = df_full.loc[df_full['year'] == y, 'data'].values
        out_df[str(y)] = seq
    out_df.index.name = 'day_of_year'
    # 6. 打印信息并返回
    return out_df, out_data



def _prctile_exact_1d(x, q_arr):
    """
    对已经去除 NaN 且一维的 x 计算 Harrell–Davis exact 分位，
    返回长度 = len(q_arr) 的数组。
    """
    # x 已经是一维，但可能含 NaN → 再次过滤
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.full(q_arr.shape, np.nan, dtype=float)
    
    x_sorted = np.sort(x)
    n = x_sorted.size
    
    # 分界百分比
    p_min = 100 * (0.5 / n)
    p_max = 100 * ((n - 0.5) / n)
    
    out = np.empty(q_arr.shape, dtype=float)
    for idx, qi in enumerate(q_arr):
        if qi <= p_min:
            out[idx] = x_sorted[0]
        elif qi >= p_max:
            out[idx] = x_sorted[-1]
        else:
            k = qi / 100.0 * n + 0.5
            lo = int(np.floor(k))
            hi = int(np.ceil(k))
            if lo == hi:
                out[idx] = x_sorted[lo-1]
            else:
                frac = k - lo
                out[idx] = x_sorted[lo-1] + frac * (x_sorted[hi-1] - x_sorted[lo-1])
    return out



def prctile_exact(a, q, axis=None):
    """
    MATLAB prctile(...,q,'Method','exact') 等价实现，支持指定轴或全部元素。
    
    参数：
      a    : 输入数组，可含 NaN
      q    : 标量或一维百分位列表，例如 99 或 [50,90]
      axis : None/"all"（默认）→ 扁平化后计算，
             整数 k     → 沿第 k 轴逐切片计算
    
    返回：
      如果 axis=None 且 q 是标量 → 标量
      如果 axis=None 且 q 是列表 → 1D 数组，长度 = len(q)
      如果 axis=k    且 q 是标量 → 除第 k 轴外，其它维度与 a 相同
      如果 axis=k    且 q 是列表 → 结果最后一维长度 = len(q)，前面维度与去掉第 k 轴的 a 相同
    """
    # 规范化 q 数组
    q_arr = np.atleast_1d(q).astype(np.float64)
    
    # 处理 axis=None 或 'all'
    if axis is None or axis == 'all':
        # 把所有元素扁平化
        x = np.asarray(a).ravel()
        # 去 NaN
        x = x[~np.isnan(x)]
        # 无数据返回 NaN
        if x.size == 0:
            out = np.full(q_arr.shape, np.nan, dtype=float)
        else:
            out = _prctile_exact_1d(x, q_arr)
        # 如果 q 原本是标量
        return out[0] if np.isscalar(q) else out
    
    # 如果指定了某个轴
    a = np.asarray(a)
    if a.ndim == 1:
        # 退化成 axis=None
        return prctile_exact(a, q, axis=None)
    
    # 对每个切片沿 axis 求分位
    # apply_along_axis 会把每个一维子数组传给我们的一维计算器
    res = np.apply_along_axis(_prctile_exact_1d, axis, a, q_arr)
    
    # apply_along_axis 结果形状：
    #   a.shape[:axis] + a.shape[axis+1:] + out_q_shape
    # 如果 q 是标量，则最后一维长度 1，需要 squeeze
    if np.isscalar(q):
        return np.squeeze(res, axis=-1)
    return res



def find_turning_points(line):
    """
    Identify turning points (peaks and valleys) in a 1D array.
    Returns two arrays of indices: (peaks, valleys).
    """
    n = line.shape[0]
    if n < 3:
        return np.empty(0, np.int64), np.empty(0, np.int64)

    # 1st-order difference
    fod = np.empty(n-1, line.dtype)
    for i in range(n-1):
        fod[i] = line[i+1] - line[i]

    # sign of the difference
    sf = np.empty(n-1, np.int64)
    for i in range(n-1):
        if fod[i] > 0:
            sf[i] = 1
        elif fod[i] < 0:
            sf[i] = -1
        else:
            sf[i] = 0

    # difference of signs
    sd = np.empty(n-2, np.int64)
    for i in range(n-2):
        sd[i] = sf[i+1] - sf[i]

    # collect turning‐point indices and labels
    # label: 1 = peak, 0 = valley
    # MATLAB logic: sd>0 → valley (0); sd<0 → peak (1)
    tp_idx = []
    tp_lbl = []
    for i in range(sd.shape[0]):
        if sd[i] != 0:
            # sd[i] corresponds to a change at line index (i+1)+1 = i+2
            tp_idx.append(i + 2)
            tp_lbl.append(0 if sd[i] > 0 else 1)

    m = len(tp_idx)
    tp_idx = np.array(tp_idx, np.int64)
    tp_lbl = np.array(tp_lbl, np.int64)

    # split into peaks and valleys
    # count them to allocate
    pcount = 0
    vcount = 0
    for i in range(m):
        if tp_lbl[i] == 1:
            pcount += 1
        else:
            vcount += 1

    peaks = np.empty(pcount, np.int64)
    valleys = np.empty(vcount, np.int64)
    pi = 0
    vi = 0
    for i in range(m):
        if tp_lbl[i] == 1:
            peaks[pi] = tp_idx[i] - 1
            pi += 1
        else:
            valleys[vi] = tp_idx[i] - 1
            vi += 1

    return peaks, valleys



def cal_extreme_thres(data, pct=70, axis=None):
    """
    Calculate thresholds thres_RH and thres_T as the given percentile of combined data.
    """
    if Flag_Numpy_prctile:
        thres = np.nanpercentile(data, pct, axis=axis)
    else:
        # MATLAB 的 prctile 方法
        thres = prctile_exact(data, pct, axis=axis)
    return thres



def identify_event(data, thres=2, min_day=2):
    """
    Identify continuous segments where data == thres and length >= min_day.
    Returns a list of indices of all days belonging to such events.
    """
    loc_all = []
    in_event = False
    start = 0
    for idx, val in enumerate(data):
        if val == thres and not in_event:
            in_event = True
            start = idx
        elif val != thres and in_event:
            end = idx
            length = end - start
            if length >= min_day:
                loc_all.extend(range(start, end))
            in_event = False
    # check if event runs to end
    if in_event:
        end = len(data)
        if end - start >= min_day:
            loc_all.extend(range(start, end))
    return np.array(loc_all)



def _worker_thresholds(RH: np.ndarray, T: np.ndarray, timelist: pd.DataFrame, thres_windows: int=7, ndays: int = 365, percentile: float=85) -> pd.DataFrame:
    # 按时间顺序重新排列数据
    RH_yearly_df, RH_df = reshape_data_yearly(RH, timelist)
    T_yearly_df, T_df = reshape_data_yearly(T, timelist)
    days_of_year = RH_yearly_df.index.values
    years = RH_yearly_df.columns.values
    # 生成每个day对应的窗口索引（0-based）
    window_indices = np.array([
        ((np.arange(day - thres_windows, day + thres_windows + 1) - 1) % ndays)
        for day in range(1, ndays+1)  # 假设days_of_year是1~365
    ])  # 形状: (365, 15)
    # === 修改：将DataFrame转为NumPy数组加速索引 ===
    RH_yearly_np = RH_yearly_df.to_numpy()  # 形状: (365, num_years)
    T_yearly_np = T_yearly_df.to_numpy()

    # === 修改：向量化提取窗口数据 ===
    # 提取所有窗口的P和T数据，形状: (365, 15, num_years)
    RH_windows = RH_yearly_np[window_indices, :]
    T_windows = T_yearly_np[window_indices, :]
    # 展平为 (365, 15*num_years)
    RH_flattened = RH_windows.reshape(ndays, -1)
    T_flattened = T_windows.reshape(ndays, -1)
    # === 修改：批量计算阈值 ===
    # 预初始化阈值数组
    thres_RH = np.zeros(ndays)
    thres_T = np.zeros(ndays)
    thres_Tw = np.zeros(ndays)
    thres_RH = cal_extreme_thres(RH_flattened, pct=percentile, axis=1)
    thres_T = cal_extreme_thres(T_flattened, pct=percentile, axis=1)
    # 构建阈值DataFrame
    thres_df = pd.DataFrame({'RH': thres_RH, 'T': thres_T}, index=days_of_year)
    thres_df = pd.concat([thres_df]*len(years), axis=0)
    thres_df.index = RH_df.index
    return thres_df


@dataclass
class ExtremeEvents_RHT:
    """
    Abbreviations:
      HH = High Humidity, HL = Low Humidity,
      TH = Temp High, TL = Temp Low,
      WH = wet-Hot,  DH = dry-Hot,
      ColdWet = wet-cold,  DC = dry-cold
    * _feature arrays are [annualized_freq, humidity_intensity_or_bg, temp_intensity_or_bg]
      - For HH/HL: 2nd element is humidity intensity relative to threshold
      - For TH/TL: 3rd element is temperature intensity relative to threshold
    * _flag arrays: single extremes are 0/1; compound flags are sums before thresholding (0/1/2).
    """
    # ----- single-variable features -----
    ex_wet_feature: np.ndarray       # High humidity (wet)
    ex_dry_feature: np.ndarray       # Low humidity (dry)
    ex_Hot_feature: np.ndarray      # High temperature (Hot)
    ex_cold_feature: np.ndarray      # Low temperature (cold)
    # ----- compound features -----
    cex_ColdWet_feature: np.ndarray  # wet-cold  (ColdWet = wet ∩ cold)
    cex_ColdDry_feature: np.ndarray  # dry-cold  (DC = dry ∩ cold)
    cex_HotWet_feature: np.ndarray  # wet-Hot  (WH = wet ∩ Hot)
    cex_HotDry_feature: np.ndarray  # dry-Hot  (DH = dry ∩ Hot)
    # ----- flags -----
    ex_wet_flag: np.ndarray          # High humidity event flag (0/1)
    ex_dry_flag: np.ndarray          # Low humidity event flag (0/1)
    ex_Hot_flag: np.ndarray         # High temperature event flag (0/1)
    ex_cold_flag: np.ndarray         # Low temperature event flag (0/1)
    cex_ColdWet_flag: np.ndarray     # wet-cold event flag (0-1/2)
    cex_ColdDry_flag: np.ndarray     # dry-cold event flag (0-1/2)
    cex_HotWet_flag: np.ndarray     # wet-Hot event flag (0-1/2)
    cex_HotDry_flag: np.ndarray     # dry-Hot event flag (0-1/2)
    # ----- thresholds -----
    thres_RH_Wet: np.ndarray         # high humidity threshold (RH high)
    thres_RH_Dry:  np.ndarray         # low humidity threshold  (RH low)
    thres_T_Hot: np.ndarray          # high temperature threshold
    thres_T_Cold:  np.ndarray          # low  temperature threshold
    thres_Tw_ColdDry: np.ndarray      # wet-bulb temp threshold for dry-cold
    thres_Tw_ColdWet: np.ndarray      # wet-bulb temp threshold for wet-cold
    thres_Tw_HotDry: np.ndarray       # wet-bulb temp threshold for dry-Hot
    thres_Tw_HotWet: np.ndarray       # wet-bulb temp threshold for wet-Hot
    # ----- diffs (intensity) -----
    RH_Dry_diff: np.ndarray           # low-humidity (dry) diff (negative)
    RH_Wet_diff: np.ndarray          # high-humidity (wet) diff (positive)
    T_Cold_diff: np.ndarray            # low-temperature diff (negative)
    T_Hot_diff: np.ndarray           # high-temperature diff (positive)
    Tw_ColdDry_diff: np.ndarray       # wet-bulb temp diff for dry-cold
    Tw_ColdWet_diff: np.ndarray       # wet-bulb temp diff for wet-cold
    Tw_HotDry_diff: np.ndarray        # wet-bulb temp diff for dry-Hot
    Tw_HotWet_diff: np.ndarray        # wet-bulb temp diff for wet-Hot


# ---------- extreme event metrics ----------
def find_extreme_events_RHT(RH: np.ndarray, T: np.ndarray, RH_ref: np.ndarray, T_ref: np.ndarray, timelist: pd.DataFrame):
    """
    Event abbreviations (<=2 letters):
      Single: HH (Humidity High), HL (Humidity Low), Hot (Temp High), cold (Temp Low)
      Compound: WH (wet-Hot = HH ∩ Hot), DH (dry-Hot = HL ∩ Hot),
                ColdWet (wet-cold = HH ∩ cold), DC (dry-cold = HL ∩ cold)
    Returned object fields keep original names for compatibility.
    """
    DefineCompoundByWetBulb = True  # True: use wet-bulb temperature to define compound events; False: use sum of single-event flags
    cal_thres_window = 7
    percentile_high = HighThresPercentile
    percentile_low  = LowThresPercentile
    # ----- Prepare data -----
    # fraction = 365.25
    fraction = 1

    # Reorder by time
    Tw = WBT.wet_bulb_temperature_stull(T, RH, T_unit='C', clip_domain=True)
    RH_yearly_df, RH_df = reshape_data_yearly(RH, timelist)
    T_yearly_df, T_df = reshape_data_yearly(T, timelist)
    Tw_yearly_df, Tw_df = reshape_data_yearly(Tw, timelist)
    thres_df_high = _worker_thresholds(RH_ref, T_ref, timelist, cal_thres_window, 365, percentile_high)
    thres_df_low  = _worker_thresholds(RH_ref, T_ref, timelist, cal_thres_window, 365, percentile_low)
    thres_df_Tw_HotWet = WBT.wet_bulb_temperature_stull(thres_df_high['T'].values, thres_df_high['RH'].values, T_unit='C', clip_domain=True)
    thres_df_Tw_HotDry = WBT.wet_bulb_temperature_stull(thres_df_high['T'].values, thres_df_low['RH'].values, T_unit='C', clip_domain=True)
    thres_df_Tw_ColdWet = WBT.wet_bulb_temperature_stull(thres_df_low['T'].values, thres_df_high['RH'].values, T_unit='C', clip_domain=True)
    thres_df_Tw_ColdDry = WBT.wet_bulb_temperature_stull(thres_df_low['T'].values, thres_df_low['RH'].values, T_unit='C', clip_domain=True)
    
    # ----- Single-variable extremes -----
    wet = (RH_df['data'].values > thres_df_high['RH'].values).astype(int)   # High humidity
    dry = (RH_df['data'].values < thres_df_low['RH'].values).astype(int)    # Low humidity
    Hot = (T_df['data'].values > thres_df_high['T'].values).astype(int)   # High temperature
    cold = (T_df['data'].values < thres_df_low['T'].values).astype(int)    # Low temperature

    # Intensity (positive = exceed high threshold; negative = below low threshold)
    RH_Wet_diff = np.abs(RH_df['data'].to_numpy(dtype=np.float32) - thres_df_high['RH'].to_numpy(dtype=np.float32))
    RH_Dry_diff = np.abs(RH_df['data'].to_numpy(dtype=np.float32) - thres_df_low['RH'].to_numpy(dtype=np.float32))
    T_Hot_diff = np.abs(T_df['data'].to_numpy(dtype=np.float32) - thres_df_high['T'].to_numpy(dtype=np.float32))
    T_Cold_diff = np.abs(T_df['data'].to_numpy(dtype=np.float32) - thres_df_low['T'].to_numpy(dtype=np.float32))
    # Tw_HotWet_diff = np.abs(Tw_df['data'].to_numpy(dtype=np.float32) - thres_df_Tw_HotWet)
    # Tw_HotDry_diff = np.abs(Tw_df['data'].to_numpy(dtype=np.float32) - thres_df_Tw_HotDry)
    # Tw_ColdWet_diff = np.abs(Tw_df['data'].to_numpy(dtype=np.float32) - thres_df_Tw_ColdWet)
    # Tw_ColdDry_diff = np.abs(Tw_df['data'].to_numpy(dtype=np.float32) - thres_df_Tw_ColdDry)
    Tw_HotWet_diff = Tw_df['data'].to_numpy(dtype=np.float32)
    Tw_HotDry_diff = Tw_df['data'].to_numpy(dtype=np.float32)
    Tw_ColdWet_diff = Tw_df['data'].to_numpy(dtype=np.float32)
    Tw_ColdDry_diff = Tw_df['data'].to_numpy(dtype=np.float32)

    T_mean = np.nanmean(T_df['data'].to_numpy(dtype=np.float32))
    RH_mean = np.nanmean(RH_df['data'].to_numpy(dtype=np.float32))
    Tw_mean = np.nanmean(Tw_df['data'].to_numpy(dtype=np.float32))
    tot_day = wet.shape[0]

    # Features: [annualized frequency, humidity-intensity/background, temp-intensity/background]
    wet_feature = np.array([0, np.nan, np.nan, np.nan], dtype=float)
    if np.sum(wet) > 0:
        wet_feature[0] = (np.sum(wet) / tot_day) * fraction
        wet_feature[1] = np.nanmean(RH_Wet_diff[wet == 1])
        wet_feature[2] = T_mean
        wet_feature[3] = Tw_mean

    dry_feature = np.array([0, np.nan, np.nan, np.nan], dtype=float)
    if np.sum(dry) > 0:
        dry_feature[0] = (np.sum(dry) / tot_day) * fraction
        dry_feature[1] = np.nanmean(RH_Dry_diff[dry == 1])
        dry_feature[2] = T_mean
        dry_feature[3] = Tw_mean

    Hot_feature = np.array([0, np.nan, np.nan, np.nan], dtype=float)
    if np.sum(Hot) > 0:
        Hot_feature[0] = (np.sum(Hot) / tot_day) * fraction
        Hot_feature[1] = RH_mean
        Hot_feature[2] = np.nanmean(T_Hot_diff[Hot == 1])
        Hot_feature[3] = Tw_mean

    cold_feature = np.array([0, np.nan, np.nan, np.nan], dtype=float)
    if np.sum(cold) > 0:
        cold_feature[0] = (np.sum(cold) / tot_day) * fraction
        cold_feature[1] = RH_mean
        cold_feature[2] = np.nanmean(T_Cold_diff[cold == 1])
        cold_feature[3] = Tw_mean


    #----- Compound extremes (intensity defined by wet-bulb temperature) -----
    HotWet = wet + Hot   # wet-Hot
    HotDry = dry + Hot   # dry-Hot
    ColdWet = wet + cold   # wet-cold
    ColdDry = dry + cold   # dry-cold

    loc_HotWet = identify_event(HotWet, thres=2, min_day=1)
    loc_HotDry = identify_event(HotDry, thres=2, min_day=1)
    loc_ColdWet = identify_event(ColdWet, thres=2, min_day=1)
    loc_ColdDry = identify_event(ColdDry, thres=2, min_day=1)

    if DefineCompoundByWetBulb:
        HotWet_feature = np.array([0, np.nan, np.nan, np.nan], dtype=float)
        if loc_HotWet.size > 0:
            HotWet_feature[0] = (len(loc_HotWet) / tot_day) * fraction
            HotWet_feature[1] = np.nanmean(RH_Wet_diff[loc_HotWet])
            HotWet_feature[2] = np.nanmean(T_Hot_diff[loc_HotWet])
            HotWet_feature[3] = np.nanmean(Tw_HotWet_diff[loc_HotWet])

        HotDry_feature = np.array([0, np.nan, np.nan, np.nan], dtype=float)
        if loc_HotDry.size > 0:
            HotDry_feature[0] = (len(loc_HotDry) / tot_day) * fraction
            HotDry_feature[1] = np.nanmean(RH_Dry_diff[loc_HotDry])
            HotDry_feature[2] = np.nanmean(T_Hot_diff[loc_HotDry])
            HotDry_feature[3] = np.nanmean(Tw_HotDry_diff[loc_HotDry])

        ColdWet_feature = np.array([0, np.nan, np.nan, np.nan], dtype=float)
        if loc_ColdWet.size > 0:
            ColdWet_feature[0] = (len(loc_ColdWet) / tot_day) * fraction
            ColdWet_feature[1] = np.nanmean(RH_Wet_diff[loc_ColdWet])
            ColdWet_feature[2] = np.nanmean(T_Cold_diff[loc_ColdWet])
            ColdWet_feature[3] = np.nanmean(Tw_ColdWet_diff[loc_ColdWet])

        ColdDry_feature = np.array([0, np.nan, np.nan, np.nan], dtype=float)
        if loc_ColdDry.size > 0:
            ColdDry_feature[0] = (len(loc_ColdDry) / tot_day) * fraction
            ColdDry_feature[1] = np.nanmean(RH_Dry_diff[loc_ColdDry])
            ColdDry_feature[2] = np.nanmean(T_Cold_diff[loc_ColdDry])
            ColdDry_feature[3] = np.nanmean(Tw_ColdDry_diff[loc_ColdDry])


    # ----- Return (field names kept as original) -----
    extre_event = ExtremeEvents_RHT(
        # single-variable features
        ex_wet_feature = wet_feature,                        # high humidity
        ex_dry_feature = dry_feature,                        # low humidity (dry)
        ex_Hot_feature = Hot_feature,                        # high temp
        ex_cold_feature = cold_feature,                      # low temp
        # compound features
        cex_ColdWet_feature = ColdWet_feature,               # wet-cold
        cex_ColdDry_feature = ColdDry_feature,               # dry-cold
        cex_HotWet_feature = HotWet_feature,                 # wet-Hot
        cex_HotDry_feature = HotDry_feature,                 # dry-Hot
        # flags
        ex_wet_flag = wet,                                   # high humidity
        ex_dry_flag = dry,                                   # low humidity (dry)
        ex_Hot_flag = Hot,                                   # high temp
        ex_cold_flag = cold,                                 # low temp
        cex_ColdWet_flag = ColdWet,                          # wet-cold
        cex_ColdDry_flag = ColdDry,                          # dry-cold
        cex_HotWet_flag = HotWet,                            # wet-Hot
        cex_HotDry_flag = HotDry,                            # dry-Hot
        # thresholds 
        thres_RH_Wet = thres_df_high['RH'].values,          # high humidity threshold
        thres_RH_Dry  = thres_df_low['RH'].values,           # low  humidity threshold
        thres_T_Hot = thres_df_high['T'].values,            # high temperature threshold
        thres_T_Cold  = thres_df_low['T'].values,             # low  temperature threshold
        thres_Tw_ColdDry = thres_df_Tw_ColdDry,              # wet-bulb temp threshold for dry-cold
        thres_Tw_ColdWet = thres_df_Tw_ColdWet,              # wet-bulb temp threshold for wet-cold
        thres_Tw_HotDry = thres_df_Tw_HotDry,                # wet-bulb temp threshold for dry-Hot
        thres_Tw_HotWet = thres_df_Tw_HotWet,                # wet-bulb temp threshold for wet-Hot
        # diffs (intensity)
        RH_Dry_diff = RH_Dry_diff,                           # low-humidity (dry) diff (negative)
        RH_Wet_diff = RH_Wet_diff,                         # high-humidity (wet) diff (positive)
        T_Cold_diff = T_Cold_diff,                             # low-temperature diff (negative)
        T_Hot_diff = T_Hot_diff,                           # high-temperature diff (positive)
        Tw_ColdDry_diff = Tw_ColdDry_diff,                   # wet-bulb temp diff for dry-cold
        Tw_ColdWet_diff = Tw_ColdWet_diff,                   # wet-bulb temp diff for wet-cold
        Tw_HotDry_diff = Tw_HotDry_diff,                     # wet-bulb temp diff for dry-Hot
        Tw_HotWet_diff = Tw_HotWet_diff,                     # wet-bulb temp diff for wet-Hot
    )
    return extre_event



def _worker_extreme_events_RHT(args):
    """
    Unpack arguments, call find_extreme_events_RHT, return its five outputs.
    """
    P_slice, T_slice, timelist = args
    res = find_extreme_events_RHT(P_slice, T_slice, P_slice, T_slice, timelist)
    return  res.ex_wet_feature, res.ex_dry_feature, res.ex_Hot_feature, res.ex_cold_feature,  \
            res.cex_ColdWet_feature, res.cex_ColdDry_feature, res.cex_HotWet_feature, res.cex_HotDry_feature


def cal_cc(RH, T):
    """
    Compute precipitation-temperature scaling curve and extract statistics.
    Returns array: [PRCP_avg, TAVG_avg, P_peak, T_peak, Sfit, condition]
    """
    # Parameters
    ethres = 99.
    t_step = 1
    fs = 0.1
    p_check = 0.05
    min_length = 10

    # Annual avg precipitation and mean temperature
    rh_avg = np.nanmean(RH) * 365.25
    tavg_avg = np.nanmean(T)

    # Keep only days with T > 0
    mask = RH > 0
    RH_pos = RH[mask]
    T_pos = T[mask]
    LenF = RH_pos.size

    # initialization
    T_peak = np.nan
    RH_peak = np.nan
    Sfit = np.nan
    condition = np.nan 
    k = np.nan
    if LenF < min_length: # not enough data
        return np.array([rh_avg, tavg_avg, np.nan, np.nan, np.nan, np.nan])

    # Bin method
    T_min, T_max = np.nanmin(T_pos), np.nanmax(T_pos)
    T_range = np.arange(T_min, T_max+t_step, t_step)
    length = len(T_range)-2
    RH_bin = np.empty(length)*np.nan
    T_bin = np.empty(length)*np.nan
    for it in range(length):
        loc = np.where((T_pos > T_range[it]) & (T_pos <= T_range[it+2]))[0]
        if loc.size >= 50:
            RH_loc = RH_pos[loc]
            T_loc = T_pos[loc]
            if Flag_Numpy_prctile:
                RH_bin[it] = np.nanpercentile(RH_loc, ethres)
            else:
                RH_bin[it] = prctile_exact(RH_loc, ethres) # MATLAB 的 prctile 方法
            T_bin[it] = np.nanmean(T_loc)
    RH_bin[RH_bin < 0] = np.nan  # 负值置为 NaN
    mask = ~np.isnan(RH_bin) & ~np.isnan(T_bin)
    RH_bin = RH_bin[mask]
    T_bin = T_bin[mask]
    LenF = RH_bin.size

    if LenF >= min_length: 
        # Smooth with LOESS
        # span 对应 MATLAB 中的 fs（如果 fs 是比例）；如果 fs 是样本数，传 span=fs/len(RH_bin)
        # x = np.arange(len(RH_bin))       
        # model = loess(x, RH_bin, span=fs, degree=2) 
        # model.fit()                      
        # P_bin_s = model.predict(x)
        RH_bin_s = RH_bin   # 直接使用原始数据，与Matlab结果不同

        RH_bin_s[RH_bin_s < 0] = np.nan  # 负值置为 NaN
        RH_bin = RH_bin[~np.isnan(RH_bin_s)]
        T_bin = T_bin[~np.isnan(RH_bin_s)]
        RH_bin_s = RH_bin_s[~np.isnan(RH_bin_s)]
        LenF = RH_bin.size

        # Find peak
        peaks, valleys = find_turning_points(RH_bin_s)

        if peaks.size > 0:
            # take the highest peak
            loc_maxRH = peaks[np.argmax(RH_bin_s[peaks])]
            RH_peak = RH_bin_s[loc_maxRH]
            T_peak = T_bin[loc_maxRH]
            loc_min = np.argmin(RH_bin_s[:loc_maxRH])
            # period_1 = np.arange(loc_min, loc_maxRH+1)
            period_1 = np.arange(0, loc_maxRH+1)

            if not (period_1.size <= max(min_length*0.2, LenF*0.1)):
                xx = T_bin[period_1]
                yy = np.log(RH_bin_s[period_1])

                # 拟合
                r, pValue = pearsonr(xx, yy)

                if pValue < p_check:  # 相关性显著
                    # 线性拟合
                    k_temp = np.polyfit(xx, yy, 1)
                    k = k_temp[0]
                    yy_fit = np.polyval(k_temp, xx)
                    condition = 1
                else:
                    # ——1. 用 statsmodels 拟合线性模型，提取残差
                    X = sm.add_constant(xx)
                    model = sm.OLS(yy, X).fit()
                    residuals = model.resid
                    # ——2. 按残差绝对值的 95th 百分位剔除离群点
                    if Flag_Numpy_prctile:
                        threshold = np.nanpercentile(np.abs(residuals), 95)
                    else:
                        threshold = prctile_exact(np.abs(residuals), 95) # MATLAB 的 prctile 方法
                    mask = np.abs(residuals) <= threshold
                    xx_clean = xx[mask]
                    yy_clean = yy[mask]
                    if not (xx_clean.size <= max(min_length*0.2, LenF*0.1)):
                        # ——3. 用剔除离群后的数据计算 Pearson 相关系数及 p 值
                        r, pValue = pearsonr(xx_clean, yy_clean)
                        # ——4. 用剔除离群后的数据做一次 polyfit/polyval 拟合
                        k_temp = np.polyfit(xx_clean, yy_clean, 1)  # [slope, intercept]
                        k = k_temp[0]                               # 斜率
                        yy_fit = np.polyval(k_temp, xx_clean)       # 拟合值
                        condition = 2 if pValue < p_check else 3
                    else:
                        condition = 4
            else:
                condition = 4

    Sfit = (np.exp(k) - 1) * 100

    return np.array([rh_avg, tavg_avg, RH_peak, T_peak, Sfit, condition], dtype=np.float64)



def _worker_cc(args):
    """Wrapper to unpack arguments for starmap/map."""
    RH_slice, T_slice = args
    return cal_cc(RH_slice, T_slice)



def cal_RHT_extreme_warp(q, T2m, timelist):
    """
    Calculate extreme events and their features.
    """
    n_workers = 40
    pool = Pool(processes=n_workers)

    # === 降水–温度标度曲线 ===
    # 构造所有任务的输入参数列表
    ntime, nlat, nlon = q.shape
    tasks = [
        (q[:, i, j], T2m[:, i, j])
        for i in range(nlat)
        for j in range(nlon)
    ]
    results_flat = list(tqdm(pool.imap(_worker_cc, tasks), total=len(tasks), desc="Calculating scaling curve"))
    results_arr = np.array(results_flat, dtype=np.float64)
    results_arr = results_arr.reshape(nlat, nlon, -1)
    cc_results = results_arr.transpose(2, 0, 1)

    # === 判断极端事件分类 ===
    tasks = []
    for i in range(nlat):
        for j in range(nlon):
            tasks.append((q[:, i, j], T2m[:, i, j], timelist))
    # for task in tasks:
        # results = extreme_events_worker(task)
    results = list(tqdm(pool.imap(_worker_extreme_events_RHT, tasks), total=len(tasks), desc="Calculating extreme events"))
    pool.close()

    # 分离五个输出列表
    ex_hh_list, ex_hl_list, ex_th_list, ex_tl_list, cex_wc_list, cex_dc_list, cex_wh_list, cex_dh_list = zip(*results)
    # 转 numpy 并 reshape 回去
    HH_flat = np.stack(ex_hh_list, axis=1)    # shape ( nlat*nlon)
    HL_flat = np.stack(ex_hl_list, axis=1)    # shape ( nlat*nlon)
    TH_flat = np.stack(ex_th_list, axis=1)    # shape ( nlat*nlon)
    TL_flat = np.stack(ex_tl_list, axis=1)    # shape ( nlat*nlon)
    WC_flat = np.stack(cex_wc_list, axis=1)   # shape ( nlat*nlon)
    DC_flat = np.stack(cex_dc_list, axis=1)   # shape ( nlat*nlon)
    WH_flat = np.stack(cex_wh_list, axis=1)   # shape ( nlat*nlon)
    DH_flat = np.stack(cex_dh_list, axis=1)   # shape ( nlat*nlon)

    # thres_P = np.stack(thres_P, axis=1)  # shape (365, nlat*nlon)
    # thres_T = np.stack(thres_T, axis=1)  # shape (365, nlat*nlon)
    # reshape
    wet = HH_flat.reshape(4, nlat, nlon)
    dry = HL_flat.reshape(4, nlat, nlon)
    Hot = TH_flat.reshape(4, nlat, nlon)
    cold = TL_flat.reshape(4, nlat, nlon)
    ColdWet = WC_flat.reshape(4, nlat, nlon)
    ColdDry = DC_flat.reshape(4, nlat, nlon)
    HotWet = WH_flat.reshape(4, nlat, nlon)
    HotDry = DH_flat.reshape(4, nlat, nlon)

    return cc_results, wet, dry, Hot, cold, ColdWet, ColdDry, HotWet, HotDry



def cal_extreme_event(casename: str, q: xr.Dataset, t2m: xr.Dataset, mask: xr.DataArray, outdir: str) -> None:
    outdir_var = f"{outdir}/{casename}"
    os.makedirs(outdir_var, exist_ok=True)
    q_da = q['RH']
    t2m_da = t2m['T2m']
    timelist = q['time']
    if mask is None:
        q_da_m = q_da
        t2m_da_m = t2m_da
    else:
        mask = xr.DataArray(np.ones_like(q_da.isel(time=0)), coords={'y': q_da.y, 'x': q_da.x}, dims=['y', 'x'])
        q_da, t2m_da, mask_all = xr.align(q_da, t2m_da, mask, join="exact")
        q_da_m = q_da.where(mask_all)
        t2m_da_m = t2m_da.where(mask_all)
    q_np = q_da_m.values.squeeze()
    t2m_np = t2m_da_m.values.squeeze()
    goodT = (t2m_np > -273.15) & (t2m_np < 500)
    goodP = (q_np >= -1) & (q_np < 1e8)
    q_np[~goodP] = np.nan 
    t2m_np[~goodP] = np.nan
    q_np[~goodT] = np.nan
    t2m_np[~goodT] = np.nan
    cc_results, wet, dry, Hot, cold, ColdWet, ColdDry, HotWet, HotDry = cal_RHT_extreme_warp(q_np, t2m_np, timelist)
    in_dict = {
            'cc_results': (['cc_feature', 'y', 'x'], cc_results),
            'Wet':        (['ex_feature', 'y', 'x'], wet),
            'Dry':        (['ex_feature', 'y', 'x'], dry),
            'Hot':        (['ex_feature', 'y', 'x'], Hot),
            'Cold':       (['ex_feature', 'y', 'x'], cold),
            'ColdWet':    (['ex_feature', 'y', 'x'], ColdWet),
            'ColdDry':    (['ex_feature', 'y', 'x'], ColdDry),
            'HotWet':     (['ex_feature', 'y', 'x'], HotWet),
            'HotDry':     (['ex_feature', 'y', 'x'], HotDry),
        }
    coords = {
            'y': q_da.y,
            'x': q_da.x,
            'time': q_da.time,
            'cc_feature': ['q_avg', 'tavg_avg', 'RH_peak', 'T_peak', 'Sfit', 'condition'],
            'ex_feature': ['freq', 'exRH', 'exT', 'exTw'],
        }
    savepath = f"{outdir}/{casename}/Extreme_{casename}.nc"



def calculate_thresholds(icell, RH, T, timelist):
    percentile_high=HighThresPercentile
    percentile_low=LowThresPercentile
    thres_windows=7
    ndays = 365
    thres_df_high = _worker_thresholds(RH, T, timelist, thres_windows, ndays, percentile_high)  # 预热
    thres_df_low = _worker_thresholds(RH, T, timelist, thres_windows, ndays, percentile_low)  # 预热
    thres_df_Tw_HotWet = WBT.wet_bulb_temperature_stull(thres_df_high['T'].values, thres_df_high['RH'].values, T_unit='C', clip_domain=True)
    thres_df_Tw_HotDry = WBT.wet_bulb_temperature_stull(thres_df_high['T'].values, thres_df_low['RH'].values, T_unit='C', clip_domain=True)
    thres_df_Tw_ColdWet = WBT.wet_bulb_temperature_stull(thres_df_low['T'].values, thres_df_high['RH'].values, T_unit='C', clip_domain=True)
    thres_df_Tw_ColdDry = WBT.wet_bulb_temperature_stull(thres_df_low['T'].values, thres_df_low['RH'].values, T_unit='C', clip_domain=True)
    # thres_df_high 使用时间顺序排序
    thres_df_high = thres_df_high.sort_index()
    res = {'icell': icell, 
           'thres_RH_Wet': thres_df_high['RH'].values, 'thres_RH_Dry': thres_df_low['RH'].values,
           'thres_T_Hot': thres_df_high['T'].values, 'thres_T_Cold': thres_df_low['T'].values,
           'thres_Tw_HotWet': thres_df_Tw_HotWet, 'thres_Tw_HotDry': thres_df_Tw_HotDry,
           'thres_Tw_ColdWet': thres_df_Tw_ColdWet, 'thres_Tw_ColdDry': thres_df_Tw_ColdDry}
    
    return res



def _worker_classify_extreme_events(icell, RH, T, RH_ref, T_ref, timelist):
    res = find_extreme_events_RHT(RH, T, RH_ref, T_ref, timelist)  # 预热
    return icell, res



def _worker_identify_extreme_events(series: np.ndarray, flag_value: int, min_day: int, icell: int):
    """对单个网格列做事件识别，返回 (列索引, 标记向量)."""
    ntime = series.shape[0]
    out = np.full((ntime,), 0, dtype=np.float32)
    event_info = identify_event(series, thres=flag_value, min_day=min_day)  # 期望返回 (N,1)
    if event_info is not None:
        ev = np.asarray(event_info)
        if ev.size:  # 有事件
            # 出现事件的位置标 1
            out[ev.astype(int)] = 1
    return icell, out


