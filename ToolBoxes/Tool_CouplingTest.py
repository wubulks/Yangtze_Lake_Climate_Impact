
import os
import time
import calendar
import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass
from scipy.stats import pearsonr, spearmanr, kendalltau, rankdata
from sklearn.feature_selection import mutual_info_regression
from joblib import Parallel, delayed
from typing import Literal, Optional, Tuple, Dict, Any, Sequence, List
import ToolBoxes.Utils as TU
import ToolBoxes.Tool_PlotConfig as TPC

HighThresPercentile = TPC.HighThresPercentile
LowThresPercentile  = TPC.LowThresPercentile
ThresWindows        = TPC.ThresWindows


@dataclass
class corrInfo:
    pearson_r: float
    pearson_p: float
    spearman_r: float
    spearman_p: float
    kendall_tau: float
    kendall_p: float
    mutual_info: float
    lambda_u: float
    cov: float
    r2: float


def reshape_to_year_doy(data: np.ndarray,
                        timelist,
                        time_col: str = "time") -> np.ndarray:
    """
    将按时间展开的一维/多维数据，整理为 (nyear, ndoy, ...) 的形式。

    参数
    ----
    data : np.ndarray
        原始数据，时间维必须在第 0 维，长度为 N = nyear * ndoy。
        例如:
            - (N,)            -> (nyear, ndoy)
            - (N, nlat, nlon) -> (nyear, ndoy, nlat, nlon)
    timelist : pd.DataFrame / pd.Series / pd.Index / np.ndarray / list
        含有时间信息的对象，长度必须等于 data.shape[0]。
        - 如果是 DataFrame，使用 time_col 指定列名。
        - 如果是 ndarray / Series / Index / list，直接当作时间数组。
    time_col : str, default "time"
        如果 timelist 是 DataFrame，则使用该列作为时间列。

    返回
    ----
    data_yd : np.ndarray
        重新整理后的数组，形状为 (nyear, ndoy, ...)
    """
    data = np.asarray(data)
    ntime = data.shape[0]

    # ---- 1. 取出时间序列，统一成 pandas 的 datetime ----
    if isinstance(timelist, pd.DataFrame):
        if time_col not in timelist.columns:
            raise ValueError(
                f"time_col='{time_col}' 不在 timelist.columns 里：{timelist.columns}"
            )
        time_series = pd.to_datetime(timelist[time_col])
    elif isinstance(timelist, (pd.Series, pd.Index)):
        time_series = pd.to_datetime(timelist)
    else:
        # np.ndarray / list 等
        time_series = pd.to_datetime(np.asarray(timelist).ravel())

    if len(time_series) != ntime:
        raise ValueError(
            f"data 的时间长度 ({ntime}) 与 timelist 长度 ({len(time_series)}) 不一致。"
        )

    # ---- 2. 按时间排序 ----
    sort_idx = np.argsort(time_series.values)
    data_sorted = data[sort_idx]
    # 这里 time_sorted 是 DatetimeIndex 或 Series 都无所谓
    if hasattr(time_series, "iloc"):
        time_sorted = time_series.iloc[sort_idx]
    else:
        time_sorted = time_series[sort_idx]

    # ---- 3. 提取年份，并统计每年的样本数 ----
    # 关键：不要用 .dt
    years = pd.DatetimeIndex(time_sorted).year
    unique_years, counts_per_year = np.unique(years, return_counts=True)

    ndoy = counts_per_year[0]
    if not np.all(counts_per_year == ndoy):
        raise ValueError(
            "每年的天数不一致，可能是闰年日期未统一处理，"
            f"各年样本数为: {dict(zip(unique_years, counts_per_year))}"
        )

    nyear = len(unique_years)

    # ---- 4. 按年份分块，再 stack 回 (nyear, ndoy, ...) ----
    year_blocks = []
    for y in unique_years:
        mask_y = (years == y)
        year_data = data_sorted[mask_y, ...]
        if year_data.shape[0] != ndoy:
            raise ValueError(
                f"年份 {y} 的天数 ({year_data.shape[0]}) != ndoy ({ndoy})，"
                "请检查时间序列是否完整。"
            )
        year_blocks.append(year_data)

    data_yd = np.stack(year_blocks, axis=0)  # (nyear, ndoy, ...)

    return data_yd



def reshape_from_year_doy(data_yd: np.ndarray) -> np.ndarray:
    """
    将 (nyear, ndoy, ...) 形式的数据还原成按时间顺序展开的一维时间轴：
    (nyear * ndoy, ...)

    这个函数是 reshape_to_year_doy 的“反向操作”，前提是：
    - data_yd 的第 0 维是年份，第 1 维是年内日序（doy）
    - 年份内部的 day 顺序已经是时间顺序（通常是 1..365）

    参数
    ----
    data_yd : np.ndarray
        形状至少为 2 维，(nyear, ndoy, ...)

    返回
    ----
    data_time : np.ndarray
        形状为 (nyear * ndoy, ...)，按时间顺序展开
    """
    arr = np.asarray(data_yd)
    if arr.ndim < 2:
        raise ValueError(
            f"data_yd 至少需要 2 维，当前 ndim = {arr.ndim}，shape = {arr.shape}"
        )

    nyear, ndoy = arr.shape[0], arr.shape[1]
    rest_shape = arr.shape[2:]  # 后面的维度原样保留

    # 直接 reshape：年份 × 日序 按 row-major 顺序铺平成一条时间轴
    data_time = arr.reshape(nyear * ndoy, *rest_shape)

    return data_time



def calc_mutual_information(x, y, n_bins=30, method='equal_width'):
    """
    使用二维直方图估计两个 1D 序列的互信息 I(X;Y)。

    Parameters
    ----------
    x : array-like, shape (n,)
        变量 X（例如 T）
    y : array-like, shape (n,)
        变量 Y（例如 q 或 RH）
    n_bins : int, optional
        分箱数量，具体含义取决于分箱方法：
        - 对于 'equal_width': 每个维度上的等宽分箱数量
        - 对于 'equal_freq': 每个维度上的等频分箱数量
        默认为 30。数据越多可以适当加大；如果样本较少，可以减小。
    method : str, optional
        分箱方法：
        - 'equal_width': 等宽分箱，每个分箱的宽度相同
        - 'equal_freq': 等频分箱，每个分箱包含大致相同数量的样本
        默认为 'equal_width'

    Returns
    -------
    mi : float
        互信息，单位为 nats（如果想要 bits，可以除以 np.log(2)）
        如果有效样本点 < 2 或直方图为空，返回 np.nan
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    # 去除 NaN / inf
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size < 2 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return np.nan

    # 根据分箱方法计算二维直方图
    if method == 'equal_freq':
        # 等频分箱：n_bins 表示每个维度上要创建的分箱数量
        # 每个分箱包含大致相同数量的数据点
        x_bins = np.percentile(x, np.linspace(0, 100, n_bins + 1))
        y_bins = np.percentile(y, np.linspace(0, 100, n_bins + 1))
        pxy, x_edges, y_edges = np.histogram2d(x, y, bins=[x_bins, y_bins])
    else:  # 'equal_width'
        # 等宽分箱：n_bins 表示每个维度上等间距的分箱数量
        # 分箱边界从数据的最小值到最大值均匀分布
        pxy, x_edges, y_edges = np.histogram2d(x, y, bins=n_bins)

    if pxy.sum() == 0:
        return np.nan

    # 归一化得到联合概率分布
    pxy = pxy / pxy.sum()

    # 添加小常数避免数值问题，然后重新归一化
    pxy = pxy + 1e-12
    pxy = pxy / pxy.sum()

    # 边缘分布
    px = pxy.sum(axis=1, keepdims=True)  # 形状: (n_bins_x, 1)
    py = pxy.sum(axis=0, keepdims=True)  # 形状: (1, n_bins_y)

    # 只在 pxy > 0 的地方计算
    mask_nonzero = pxy > 0

    # 利用广播 px * py 得到独立情况下的联合分布
    p_prod = px * py

    # 避免 log(0)
    mi = np.sum(pxy[mask_nonzero] * np.log(pxy[mask_nonzero] / p_prod[mask_nonzero]))

    return float(mi)



def calc_u_rank_and_cdf(x, x_set, method: str="cdf"):
    """
    使用两种方法计算经验分位 U：
    1) rankdata 版： U_rank = rank / (N + 1)
    2) CDF 版    ： U_cdf  = (# <= x) / (N + 1)

    参数
    ----
    x : float 或 np.ndarray
        需要计算分位的值，可以是标量或数组。
        对于 T/RH 的场景，可以是某个网格点的时间序列，或者单个值。
    x_set : np.ndarray
        “参考样本集合”，用来定义经验分布。
        在你的季节窗口里，就是窗口内所有年 × 日的样本集合。
    method : str
        传给 scipy.stats.rankdata 的 method 参数，常用:
        - "average" (默认)
        - "min", "max", "dense", "ordinal"

    返回
    ----
    U_rank : np.ndarray
        使用 rankdata 方式得到的 U，形状与 x 相同。
    U_cdf : np.ndarray
        使用经验 CDF 方式得到的 U，形状与 x 相同。

    说明
    ----
    - 两种方法在数学上是等价的，只是实现方式不同。
    - 除以 (N+1) 而不是 N，是为了让 U ∈ (0, 1) 开区间，
      避免 U 精确等于 0 或 1，在做 Copula / log 等操作时更稳定。
    """
    # 转成数组，方便统一处理
    x = np.asarray(x)
    x_set = np.asarray(x_set)

    # 只用 x_set 里的有限值
    mask_valid = np.isfinite(x_set)
    x_set_valid = x_set[mask_valid]

    if x_set_valid.size == 0:
        raise ValueError("x_set 中没有有效数值（全是 NaN 或空）。")

    N = x_set_valid.size

    # 准备输出，形状与 x 相同
    U_rank = np.full_like(x, np.nan, dtype=float)
    U_cdf = np.full_like(x, np.nan, dtype=float)

    # 用 ndenumerate 支持 x 是标量 / 一维 / 多维都可以
    for idx, val in np.ndenumerate(x):
        if not np.isfinite(val):
            U_rank[idx] = np.nan
            U_cdf[idx] = np.nan
            continue

        # ============================
        # 1) rankdata 版
        #    把 x_set_valid 和 当前 val 拼在一起做 rank
        #    ranks[-1] 就是当前 val 的秩
        # ============================
        all_vals = np.concatenate([x_set_valid, [val]])
        ranks = rankdata(all_vals, method="average")
        rank_val = ranks[-1]  # 当前 val 的秩
        U_rank[idx] = rank_val / (N + 1.0)

        # ============================
        # 2) 经验 CDF 版
        #    (# <= val) / (N + 1)
        # ============================
        count_le = np.sum(x_set_valid <= val)
        U_cdf[idx] = count_le / (N + 1.0)

    return U_rank, U_cdf



def calc_u_cdf(x, x_set):
    """
    纯 NumPy 版经验 CDF 分位数计算：
        U_cdf = (# {x_set <= x}) / (N + 1)

    支持:
    - x 是标量或任意形状的数组
    - x_set 是一维数组（经验分布样本）

    返回:
    - U_cdf: 与 x 形状相同的 np.ndarray
    """
    x = np.asarray(x)
    x_set = np.asarray(x_set)

    # 只用 x_set 里的有限值
    mask_valid = np.isfinite(x_set)
    x_set_valid = x_set[mask_valid]

    if x_set_valid.size == 0:
        raise ValueError("x_set 中没有有效数值（全是 NaN 或空）。")

    N = x_set_valid.size

    # 广播比较：
    # x[..., None] : shape -> x.shape + (1,)
    # x_set_valid[None, ...] : shape -> (1, N)
    # 得到一个形状为 x.shape + (N,) 的布尔数组，最后对 axis=-1 求和
    # 注意：如果 x 是标量，这里也能正常工作
    diff = x[..., None] >= x_set_valid[None, ...]   # 或 <=，一致即可
    count_le = diff.sum(axis=-1)

    U_cdf = count_le / (N + 1.0)
    return U_cdf




def xy_to_copula_uv(x, y, win=7):
    """
    对有季节性（逐日、逐年）的 x, y 做季节标准化，
    输出 U, V (同样 shape)，表示在局部季节分布中的经验分位数。

    参数
    ----
    x : array, shape (nyear, ndoy)
    y : array, shape (nyear, ndoy)
    win : int, 窗口大小 (前后多少天)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if x.shape != y.shape:
        raise ValueError(f"x 和 y 形状不一致: {x.shape} vs {y.shape}")

    nyear, ndoy = x.shape
    U = np.full_like(x, np.nan, dtype=float)
    V = np.full_like(y, np.nan, dtype=float)

    for d0 in range(ndoy):
        # 构造日序窗口 (这里用简单的裁剪方式，不做环状，可按需要修改)
        d_start = max(0, d0 - win)
        d_end = min(ndoy, d0 + win + 1)  # Python 切片右开

        # 局部样本集合: 所有年份 × 窗口内的日序
        x_window = x[:, d_start:d_end].ravel()
        y_window = y[:, d_start:d_end].ravel()

        # 排除 NaN
        x_win_valid = x_window[np.isfinite(x_window)]
        y_win_valid = y_window[np.isfinite(y_window)]

        if (x_win_valid.size == 0) or (y_win_valid.size == 0):
            continue

        # 对每个年份在该日序上计算 U, V
        for year in range(nyear):
            x_val = x[year, d0]
            y_val = y[year, d0]

            # 该日该年的值本身如果是 NaN，就跳过
            if not (np.isfinite(x_val) and np.isfinite(y_val)):
                continue

            # 使用经验分布计算分位 (rank 版 + CDF 版)
            U_cdf = calc_u_cdf(x_val, x_win_valid)
            V_cdf = calc_u_cdf(y_val, y_win_valid)
            # 这里我选择用 CDF 版（两者理论上等价）
            U[year, d0] = float(U_cdf)
            V[year, d0] = float(V_cdf)

    return U, V



def calc_copula_based_dependence(x, y, timelist):
    """
    计算两个 1D 序列的 Copula-based 上尾相关系数 λ_U。

    参数
    ----
    x, y     : 1D 序列，长度 = ntime
    timelist : pd.DataFrame，与 x,y 对应的时间信息（长度 = ntime）

    返回
    ----
    lambda_u : float
        上尾相关系数估计值；若样本不足则返回 np.nan
    """
    cal_thres_window = ThresWindows
    percentile_high = HighThresPercentile / 100.0
    percentile_low  = LowThresPercentile / 100.0

    if x.shape != y.shape:
        raise ValueError(f"x 和 y 长度不一致: {x.shape} vs {y.shape}")

    if len(timelist) != x.size:
        raise ValueError(
            f"timelist 行数 ({len(timelist)}) 与 x,y 长度 ({x.size}) 不一致。"
        )

    # 1) 按年 × 日序 reshape
    x_yearly = reshape_to_year_doy(x, timelist)
    y_yearly = reshape_to_year_doy(y, timelist)

    # 2) 标准化 → U, V (经验 Copula 空间)
    U, V = xy_to_copula_uv(x_yearly, y_yearly, win=cal_thres_window)
   
    # 3) 展平成时间序列
    U_1d = reshape_from_year_doy(U)  
    V_1d = reshape_from_year_doy(V)

    # 4) 上尾相关系数 λ_U ≈ P(V > q | U > q)
    mask_valid = np.isfinite(U_1d) & np.isfinite(V_1d)
    u_flat = U_1d[mask_valid].ravel()
    v_flat = V_1d[mask_valid].ravel()

    if u_flat.size == 0:
        return np.nan

    mask_U_extreme = u_flat > percentile_high
    count_U_extreme = np.sum(mask_U_extreme)
    if count_U_extreme == 0:
        return np.nan

    mask_both_extreme = mask_U_extreme & (v_flat > percentile_high)
    count_both_extreme = np.sum(mask_both_extreme)

    lambda_u = count_both_extreme / count_U_extreme

    return lambda_u



def calc_cov_TRH(x, y):
    """
    计算 Cov(x, y) 的简单函数。

    参数
    ----
    x, y : 1D 数组，长度相同

    返回
    ----
    cov_TRH : float
        样本协方差 Cov(x, y)
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    # 去掉 NaN / inf
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size < 2:
        return np.nan

    x_mean = x.mean()
    y_mean = y.mean()
    cov_TRH = np.sum((x - x_mean) * (y - y_mean)) / (x.size - 1)
    return cov_TRH




def calc_coupling_metrics(icell, x, y, timelist):
    """
    计算 1D 序列的相关性指标：
    - Pearson 相关系数
    - Spearman 相关系数
    - Kendall's τ
    - 互信息 Mutual Information
    - Copula-based 上尾相关系数 λ_U
    - 线性回归 R²（y ~ a + b x）

    返回： (icell, corrInfo(...))
    """
    
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    time_arr = np.asarray(timelist).ravel()   # timelist 可能是 ndarray / Series / Index 等

    # 去 NaN / inf
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return np.nan
    x_valid = x[mask]
    y_valid = y[mask]
    timelist = pd.to_datetime(time_arr)

    if x.size < 2:
        return icell, corrInfo(pearson_r=np.nan, pearson_p=np.nan,
                        spearman_r=np.nan, spearman_p=np.nan,
                        r2=np.nan)

    # Pearson
    pearson_r, pearson_p = pearsonr(x_valid, y_valid)

    # Spearman
    spearman_r, spearman_p = spearmanr(x_valid, y_valid)

    # 一元线性回归下，R² = r²
    r2 = pearson_r ** 2

    # 互信息 Mutual Information
    X = x_valid.reshape(-1, 1)  # 自变量
    mutual_info = calc_mutual_information(x_valid, y_valid, n_bins=50, method='equal_freq')

    # Kendall's τ
    kendall_tau, kendall_p = kendalltau(x_valid, y_valid)

    # Copula-based dependence
    lambda_u = calc_copula_based_dependence(x, y, timelist)

    # 协方差 Cov(x, y)
    cov_xy = calc_cov_TRH(x_valid, y_valid)

    return icell, corrInfo(pearson_r=pearson_r, pearson_p=pearson_p,
                    spearman_r=spearman_r, spearman_p=spearman_p,
                    kendall_tau=kendall_tau, kendall_p=kendall_p,
                    mutual_info=mutual_info, lambda_u=lambda_u,
                    r2=r2, cov=cov_xy)



def calc_upper_tail_dependence_coefficient(icell: int, x: np.ndarray, y: np.ndarray) -> float:
    """
    计算两个变量的上尾依赖系数（Upper Tail Dependence Coefficient, UTDC）。
    UTDC 衡量在一个变量极端高值时，另一个变量也出现极端高值的概率。
    判断当事件X发生时，事件Y也发生的概率。

    参数:
    x -- 事件 X 发生与否的一维数组 (np.ndarray) 1=表示 X 发生，0 表示 X 不发生
    y -- 事件 Y 发生与否的一维数组 (np.ndarray) 1=表示 Y 发生，0 表示 Y 不发生

    返回:
    utcd -- 上尾依赖系数 (float)
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    # 去除 NaN / inf
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size < 2:
        return np.nan

    N_x = np.sum(x == 1)
    N_y = np.sum(y == 1)
    N_xy = np.sum((x == 1) & (y == 1))

    if N_y == 0:
        return 0.0

    utdc = N_xy / N_y
    return icell, float(utdc)



def get_norm_var(icell, x, timelist):
    """根据历史数据，计算某变量的标准化值"""
    x = np.asarray(x).ravel()
    time_arr = np.asarray(timelist).ravel()   # timelist 可能是 ndarray / Series / Index 等
    timelist = pd.to_datetime(time_arr)

    if len(timelist) != x.size:
        raise ValueError(
            f"timelist 行数 ({len(timelist)}) 与 x,y 长度 ({x.size}) 不一致。"
        )

    # 按年 × 日序 reshape
    x_yearly = reshape_to_year_doy(x, timelist)
    
    # 标准化
    nyear, ndoy = x_yearly.shape
    x_norm = np.full_like(x_yearly, np.nan, dtype=float)

    for d0 in range(ndoy):
        # 构造日序窗口 (这里用简单的裁剪方式，不做环状，可按需要修改)
        d_start = max(0, d0 - ThresWindows)
        d_end = min(ndoy, d0 + ThresWindows + 1)  # Python 切片右开

        # 局部样本集合: 所有年份 × 窗口内的日序
        x_window = x_yearly[:, d_start:d_end].ravel()

        # 排除 NaN
        x_win_valid = x_window[np.isfinite(x_window)]

        if x_win_valid.size == 0:
            continue

        for year in range(nyear):
            x_val = x_yearly[year, d0]

            # 该日该年的值本身如果是 NaN，就跳过
            if not np.isfinite(x_val):
                continue

            # 计算分位数
            x_cdf = calc_u_cdf(x_val, x_win_valid)
            x_norm[year, d0] = float(x_cdf)

    # 展平成时间序列
    x_norm_1d = reshape_from_year_doy(x_norm)  

    return icell, x_norm_1d




