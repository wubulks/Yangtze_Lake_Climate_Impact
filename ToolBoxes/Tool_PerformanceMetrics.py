import os
import calendar
import numpy as np
import xarray as xr
import pandas as pd
import ToolBoxes.Utils as TU
import ToolBoxes.Tool_InputOutput as TIO
"""
CWRF 指标库（与 ILAMB 指标体系对齐的实现）
================================================

本模块实现并补充了与 ILAMB（International Land Model Benchmarking）体系相一致的核心
统计指标与评分，并对每个函数添加了清晰的功能注释与公式说明。重点包括：

- 绝对误差类：RMSE、MAE、Bias
- 相关性类：TCC（时间相关系数，按时间去均值的距平相关）、ACC（空间距平相关，时间平均）
- 归一化/无量纲化：相对 RMSE（RRMSE）、Epsilon-Bias/Epsilon-RMSE/Epsilon-IAV
- 变率与相位：CRMSE（中心化 RMSE）、IAV（年际变率幅度）、Phase Score（相位评分）
- 空间分布评分：S_dist（基于 Taylor 2001 形式，见函数注释）

注：ILAMB 中常用做法是对“时间维度”上的要素先去掉时间均值（即构造距平），再
计算时间相关系数（TCC）或与参考场的中心化误差（CRMSE）。本实现遵循该思路。
"""


# ---------- helpers ----------
def climatology_time_mean(arr: np.ndarray, time_axis: int = 0) -> np.ndarray:
    """逐格点时间均值（气候态均值），用于构造距平：x' = x - \bar{x}。"""
    arr = np.asarray(arr, dtype=float)
    return np.nanmean(arr, axis=time_axis)



def _pcc_pair(p: np.ndarray, o: np.ndarray) -> float:
    """一对向量的 NaN 安全皮尔逊相关系数。

    至少 2 个有效样本才计算；若任一方标准差为 0 或非有限，返回 NaN。
    """
    p = TU._to_array(p).ravel()
    o = TU._to_array(o).ravel()
    m = TU._finite_mask(p, o)
    if m.sum() < 2:
        return np.nan
    x = p[m]; y = o[m]
    sx = x.std(ddof=1); sy = y.std(ddof=1)
    if not np.isfinite(sx) or not np.isfinite(sy) or sx == 0 or sy == 0:
        return np.nan
    xm = x.mean(); ym = y.mean()
    cov = ((x - xm) * (y - ym)).sum() / (x.size - 1)
    r = cov / (sx * sy)
    return float(r)



def _pcc_along_axis_3d(p: np.ndarray, o: np.ndarray, axis: int = 0) -> np.ndarray:
    """沿给定轴计算 TCC：先减去各自时间均值得到距平，再做相关。

    该实现与 ILAMB 中“时间相关系数”一致，即针对时间维度上的距平进行相关。
    返回：除去时间轴后的其余维度形状的相关系数数组。
    """
    p = TU._to_array(p); o = TU._to_array(o)
    N = p.shape[0]
    rest_shape = p.shape[1:]
    M = int(np.prod(rest_shape)) if rest_shape else 1
    p2 = p.reshape(N, M)
    o2 = o.reshape(N, M)
    m = TU._finite_mask(p2, o2)
    n = m.sum(axis=axis).astype(float)
    p_masked = np.where(m, p2, 0.0)
    o_masked = np.where(m, o2, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_a = p_masked.sum(axis=axis) / n
        mean_b = o_masked.sum(axis=axis) / n
        p_c = np.where(m, p2 - mean_a, 0.0)
        o_c = np.where(m, o2 - mean_b, 0.0)
        denom = n - 1.0
        cov = (p_c * o_c).sum(axis=axis) / denom
        var_p = (p_c * p_c).sum(axis=axis) / denom
        var_o = (o_c * o_c).sum(axis=axis) / denom
        std_p = np.sqrt(var_p)
        std_o = np.sqrt(var_o)
        r = cov / (std_p * std_o)
        bad = (n < 2) | ~np.isfinite(std_p) | ~np.isfinite(std_o) | (std_p == 0) | (std_o == 0)
        r[bad] = np.nan
    return r.reshape(rest_shape)



def _rmse(p: np.ndarray, o: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    """均方根误差 RMSE = sqrt(mean((p-o)^2))。"""
    p = TU._to_array(p); o = TU._to_array(o)
    return np.sqrt(np.nanmean((p - o) ** 2, axis=axis))



def _bias(p: np.ndarray, o: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    """平均偏差 Bias = mean(p - o)。"""
    p = TU._to_array(p); o = TU._to_array(o)
    return np.nanmean(p - o, axis=axis)


def _r2(p: np.ndarray, o: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    """决定系数 R2 = 1 - SS_res / SS_tot。

    这里将 p 视为模型预测值、o 视为观测值；SS_tot 基于成对有效样本中的
    观测均值计算。有效样本少于 2 或观测无方差时返回 NaN。
    """
    p = TU._to_array(p)
    o = TU._to_array(o)
    p, o = np.broadcast_arrays(p, o)
    m = TU._finite_mask(p, o)

    if axis is None:
        p_valid = p[m].ravel()
        o_valid = o[m].ravel()
        if o_valid.size < 2:
            return np.nan
        ss_res = np.sum((o_valid - p_valid) ** 2)
        ss_tot = np.sum((o_valid - np.mean(o_valid)) ** 2)
        if not np.isfinite(ss_tot) or ss_tot == 0:
            return np.nan
        return float(1.0 - ss_res / ss_tot)

    p = np.moveaxis(p, axis, 0)
    o = np.moveaxis(o, axis, 0)
    m = np.moveaxis(m, axis, 0)
    n = m.sum(axis=0).astype(float)
    p_masked = np.where(m, p, 0.0)
    o_masked = np.where(m, o, 0.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        o_mean = o_masked.sum(axis=0) / n
        ss_res = np.sum(np.where(m, (o - p) ** 2, 0.0), axis=0)
        ss_tot = np.sum(np.where(m, (o - o_mean) ** 2, 0.0), axis=0)
        r2 = 1.0 - ss_res / ss_tot
        bad = (n < 2) | ~np.isfinite(ss_tot) | (ss_tot == 0)
        r2 = np.where(bad, np.nan, r2)
    return r2


def _rbias(p: np.ndarray, o: np.ndarray, axis: int | None = None, how: str = "absmean", percent: bool = True) -> np.ndarray | float:
    """相对 Bias（Relative Bias, RB）
    默认（最常见，适合 q2m 等非负量）
    RB = (mean(p) - mean(o)) / mean(o) * 100%
    当分母 D=0 或非有限时返回 NaN。
    """
    p = TU._to_array(p); o = TU._to_array(o)
    mp = np.nanmean(p, axis=axis)
    mo = np.nanmean(o, axis=axis)
    if how == "absmean":
        d = np.nanmean(np.abs(o), axis=axis)
    else:
        d = mo
    rb = (mp - mo) / d
    if percent:
        rb = rb * 100.0

    return np.where((d != 0) & np.isfinite(d), rb, np.nan)



def _crms(p: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    """参考数据的中心化均方根（ILAMB 中用于归一化因子）：
    crms(o) = sqrt(mean((o - mean(o))^2))。
    """
    p = TU._to_array(p)
    return np.sqrt(np.nanmean((p - climatology_time_mean(p, time_axis=axis)) ** 2, axis=axis))



def _mae(p: np.ndarray, o: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    """平均绝对误差 MAE = mean(|p - o|)。"""
    p = TU._to_array(p); o = TU._to_array(o)
    return np.nanmean(np.abs(p - o), axis=axis)


def _mape(p: np.ndarray, o: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    """计算平均绝对百分比误差 MAPE = mean(|(p - o) / o|) * 100。"""
    p = TU._to_array(p); o = TU._to_array(o)
    return np.nanmean(np.abs((p - o) / o) * 100, axis=axis)


def _nmae_sigma_anom(p: np.ndarray, o: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    """去均值后的相对误差（百分比）:
    NMAEσ = mean(|(p' - o')|) / std(o') * 100，其中 p'=p-mean(p), o'=o-mean(o)。
    """
    p = TU._to_array(p); o = TU._to_array(o)
    if axis is None:
        p0 = p - np.nanmean(p)
        o0 = o - np.nanmean(o)
        mae = np.nanmean(np.abs(p0 - o0))
        sig = np.nanstd(o0)
        return float(mae / sig * 100) if sig > 0 else np.nan

    p0 = p - np.nanmean(p, axis=axis, keepdims=True)
    o0 = o - np.nanmean(o, axis=axis, keepdims=True)
    mae = np.nanmean(np.abs(p0 - o0), axis=axis)
    sig = np.nanstd(o0, axis=axis)
    return np.where(sig > 0, mae / sig * 100, np.nan)



def _nmae_iqr_anom(p: np.ndarray, o: np.ndarray, axis: int = 0) -> np.ndarray:
    """T2m推荐：去均值后 NMAE_IQR = MAE(p',o') / IQR(o') * 100."""
    p = TU._to_array(p); o = TU._to_array(o)
    p0 = p - np.nanmean(p, axis=axis, keepdims=True)
    o0 = o - np.nanmean(o, axis=axis, keepdims=True)

    mae = np.nanmean(np.abs(p0 - o0), axis=axis)

    q75 = np.nanpercentile(o0, 75, axis=axis)
    q25 = np.nanpercentile(o0, 25, axis=axis)
    iqr = q75 - q25

    return np.where(iqr > 0, mae / iqr * 100, np.nan)




def _nmae_mean(p: np.ndarray, o: np.ndarray, axis: int = 0, min_mean: float = 0.05) -> np.ndarray:
    """q2m推荐：NMAE_mean = MAE / mean(o) * 100；mean(o) 很小的格点置 NaN."""
    p = TU._to_array(p); o = TU._to_array(o)
    mae = np.nanmean(np.abs(p - o), axis=axis)
    mu  = np.nanmean(o, axis=axis)
    return np.where(mu >= min_mean, mae / mu * 100, np.nan)



def _nmae_mean_wet(p: np.ndarray, o: np.ndarray, axis: int = 0, wet_thr: float = 0.1) -> np.ndarray:
    """prec推荐：只在 wet (o>=wet_thr) 上算 NMAE_mean_wet = MAE_wet / mean(o_wet) * 100."""
    p = TU._to_array(p); o = TU._to_array(o)
    wet = o >= wet_thr

    # 在 wet 条件下的误差与均值
    p_w = np.where(wet, p, np.nan)
    o_w = np.where(wet, o, np.nan)

    mae = np.nanmean(np.abs(p_w - o_w), axis=axis)
    mu  = np.nanmean(o_w, axis=axis)

    return np.where(mu > 0, mae / mu * 100, np.nan)




def _rrmse(p: np.ndarray, o: np.ndarray, axis: int | None = None, how: str = "absmean") -> np.ndarray | float:
    """相对 RMSE：RRMSE = RMSE / D

    其中 D 为观测的尺度：
      - how="std"  : D = std(o)
      - how="mean" : D = mean(o)
      - how="absmean"(默认): D = mean(|o|)
    当 D=0 或非有限时返回 NaN。
    """
    rmse = _rmse(p, o, axis=axis)
    obs = TU._to_array(o)
    if how == "std":
        d = np.nanstd(obs, axis=axis)
    elif how == "mean":
        d = np.nanmean(obs, axis=axis)
    else:
        d = np.nanmean(np.abs(obs), axis=axis)
    return np.where((d != 0) & np.isfinite(d), rmse / d, np.nan)



def _crmse(p: np.ndarray, o: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    """中心化均方根误差（去掉各自时间均值后的 RMS 差）：
    CRMSE = sqrt(mean(( (p-\bar{p}) - (o-\bar{o}) )^2))。
    """
    p = TU._to_array(p); o = TU._to_array(o)
    p_anom = p - climatology_time_mean(p, time_axis=axis)
    o_anom = o - climatology_time_mean(o, time_axis=axis)
    diff = p_anom - o_anom
    return np.sqrt(np.nanmean(diff ** 2, axis=axis))



def _phase_score(p: np.ndarray, o: np.ndarray, axis: int = 0, smooth: int = 7) -> np.ndarray | float:
    """相位偏差评分（0~1）：基于平滑后峰值相位差。

    具体做法：
      1) 对 p、o 沿时间轴做简单滑动平均（长度 `smooth`）。
      2) 取各自峰值位置 tau_p, tau_o（argmax）。
      3) 令周期 period = 序列长度，计算相位差 theta（取最短角）。
      4) 打分 = 0.5 * [1 + cos(2π * theta / period)]。
    值越接近 1，表明相位越一致。
    """
    p_s = TU.movavg(p, smooth, axis=axis)
    o_s = TU.movavg(o, smooth, axis=axis)
    tau_p = np.argmax(p_s, axis=axis)
    tau_o = np.argmax(o_s, axis=axis)
    period = p.shape[axis]
    theta = (tau_p - tau_o) % period
    theta = np.where(theta > period / 2, theta - period, theta)
    return 0.5 * (1.0 + np.cos(2.0 * np.pi * theta / period))



def _iav(p: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    """年际变率幅度 IAV = sqrt(mean((x' )^2))，其中 x' 为沿 `axis` 的距平。"""
    p = TU._to_array(p)
    p_anom = p - climatology_time_mean(p, time_axis=axis)
    return np.sqrt(np.nanmean((p_anom) ** 2, axis=axis))



def _s_dist(p: np.ndarray, o: np.ndarray, axis: int = 0) -> float:
    """空间分布评分 S_dist（Taylor 形式，ILAMB 采用的常见写法之一）：

    记 a = mean_t(p)，b = mean_t(o) 为时间气候态（逐格点时间均值）。
    在可比较网格上：
      σ  = std(a) / std(b)，R = corr(a, b)
      S  = 2 * (1 + R) / (σ + 1/max(σ, eps))^2
    注：该形式与 Taylor(2001) 的 skill 评分同源，ILAMB 文献实现存在细微变体，
    此处采用与多数开源实现一致的无量纲 0~1 分值（值越大越好）。
    """
    eps = 1e-12
    p = TU._to_array(p); o = TU._to_array(o)
    p_c = climatology_time_mean(p, time_axis=axis)
    o_c = climatology_time_mean(o, time_axis=axis)
    m = TU._finite_mask(p_c, o_c)
    if m.sum() < 2:
        return np.nan
    a_flat = p_c[m].ravel()
    b_flat = o_c[m].ravel()
    std_b = np.std(b_flat, ddof=0)
    if not np.isfinite(std_b) or std_b <= eps:
        return np.nan
    sigma = np.std(a_flat, ddof=0) / std_b
    R = _pcc_pair(a_flat, b_flat)
    if not np.isfinite(R):
        return np.nan
    s_dist = 2.0 * (1.0 + R) / (sigma + 1.0 / max(sigma, eps)) ** 2
    return float(s_dist)



def _epsilon_bias(p: np.ndarray, o: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Epsilon Bias（均值归一化）：|Bias| / crms(o)，返回二维 (y,x) 数组。"""
    p = TU._to_array(p); o = TU._to_array(o)
    crms = _crms(o, axis=axis)
    bias = _bias(p, o, axis=axis)
    out = np.full_like(crms, np.nan, dtype=float)
    mask = ~np.isclose(crms, 0)
    out[mask] = np.abs(bias[mask]) / crms[mask]
    return out



def _epsilon_rmse(p: np.ndarray, o: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Epsilon RMSE（中心化）：CRMSE(p,o) / crms(o)。"""
    p = TU._to_array(p); o = TU._to_array(o)
    crms = _crms(o, axis=axis)
    crmse = _crmse(p, o, axis=axis)
    out = np.full_like(crms, np.nan, dtype=float)
    mask = ~np.isclose(crms, 0)
    out[mask] = crmse[mask] / crms[mask]
    return out



def _epsilon_iav(p: np.ndarray, o: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Epsilon IAV：|IAV(p) - IAV(o)| / max(IAV(o), 1e-12)。"""
    iav_p = _iav(p, axis=axis)
    iav_o = _iav(o, axis=axis)
    out = np.full_like(iav_o, np.nan, dtype=float)
    mask = ~np.isclose(iav_o, 0)
    out[mask] = np.abs(iav_p[mask] - iav_o[mask]) / np.maximum(iav_o[mask], 1e-12)
    return out



# ---------- 1D/2D 简易包装 ----------
def RMSE_1D(pred: np.ndarray, obs: np.ndarray) -> float:
    """1D RMSE。"""
    return float(_rmse(pred, obs))



def RRMSE_1D(pred: np.ndarray, obs: np.ndarray, denom: str = "absmean") -> float:
    """1D 相对 RMSE（见 _rrmse 的 `how`）。"""
    return float(_rrmse(pred, obs, axis=None, how=denom))



def PCC_1D(pred: np.ndarray, obs: np.ndarray) -> float:
    """1D 皮尔逊相关。"""
    return _pcc_pair(pred, obs)


def R2_1D(pred: np.ndarray, obs: np.ndarray) -> float:
    """1D 决定系数 R2 = 1 - SS_res / SS_tot。"""
    return float(_r2(pred, obs))


def Bias_1D(pred: np.ndarray, obs: np.ndarray) -> float:
    """1D 平均偏差。"""
    return float(_bias(pred, obs))


def RBias_1D(pred: np.ndarray, obs: np.ndarray, how: str = "absmean", percent: bool = True) -> float:
    """1D 相对偏差。"""
    return float(_rbias(pred, obs, axis=None, how=how, percent=percent))



def MAE_1D(pred: np.ndarray, obs: np.ndarray) -> float:
    """1D 平均绝对误差。"""
    return float(_mae(pred, obs))



def MAPE_1D(pred: np.ndarray, obs: np.ndarray) -> float:
    """1D 平均绝对百分比误差。"""
    return float(_mape(pred, obs))



def NMAE_sigma_anom_1D(pred: np.ndarray, obs: np.ndarray) -> float:
    """1D 去均值后的相对误差（百分比）。"""
    return float(_nmae_sigma_anom(pred, obs))



def NMAE_iqr_anom_1D(pred: np.ndarray, obs: np.ndarray) -> float:
    """1D 去均值后的 IQR 归一化相对误差（百分比）。"""
    return float(_nmae_iqr_anom(pred, obs))



def NMAE_mean_1D(pred: np.ndarray, obs: np.ndarray) -> float:
    """1D 基于均值归一化的相对误差（百分比）。"""
    return float(_nmae_mean(pred, obs))



def NMAE_mean_wet_1D(pred: np.ndarray, obs: np.ndarray) -> float:
    """1D 基于均值归一化的相对误差（仅 wet）。"""
    return float(_nmae_mean_wet(pred, obs))



def CRMSE_1D(pred: np.ndarray, obs: np.ndarray) -> float:
    """1D 中心化 RMSE（去除各自均值后的误差）。"""
    return float(_crmse(pred, obs))



def PS_1D(pred: np.ndarray, obs: np.ndarray, smooth: int = 7) -> float:
    """1D 相位评分（0~1）。"""
    return float(_phase_score(pred, obs, smooth=smooth))



# ---------- 2D ----------
def RMSE_2D(pred: np.ndarray, obs: np.ndarray, axis: int | None = None) -> float:
    """2D 全域 RMSE（展平后计算）。"""
    return float(_rmse(pred, obs))



def RRMSE_2D(pred: np.ndarray, obs: np.ndarray, denom: str = "absmean", axis: int | None = None) -> float:
    """2D 全域相对 RMSE（展平后计算）。"""
    return float(_rrmse(pred, obs, axis=None, how=denom))



def PCC_2D(pred: np.ndarray, obs: np.ndarray, axis: int = None) -> float:
    """2D 皮尔逊相关（展平）。"""
    return _pcc_pair(pred, obs)


def R2_2D(pred: np.ndarray, obs: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    """2D 决定系数；axis=None 时展平全域计算，否则沿指定轴计算。"""
    return _r2(pred, obs, axis=axis)



def Bias_2D(pred: np.ndarray, obs: np.ndarray, axis: int | None = None) -> float:
    """2D 全域 Bias（展平后 mean）。"""
    return float(_bias(pred, obs))


def RBias_2D(pred: np.ndarray, obs: np.ndarray, how: str = "absmean", percent: bool = True, axis: int | None = None) -> float:
    """2D 全域相对偏差（展平后 mean）。"""
    return float(_rbias(pred, obs, axis=None, how=how, percent=percent))


def MAE_2D(pred: np.ndarray, obs: np.ndarray, axis: int | None = None) -> float:
    """2D 全域 MAE（展平后 mean）。"""
    return float(_mae(pred, obs, axis=axis))


def MAPE_2D(pred: np.ndarray, obs: np.ndarray, axis: int | None = None) -> float:
    """2D 全域平均绝对百分比误差（展平后 mean）。"""
    return float(_mape(pred, obs, axis=axis))


def NMAE_sigma_anom_2D(pred: np.ndarray, obs: np.ndarray, axis: int | None = None) -> float:
    """2D 全域去均值后的相对误差（展平后计算）。"""
    return float(_nmae_sigma_anom(pred, obs, axis=axis))


def NMAE_iqr_anom_2D(pred: np.ndarray, obs: np.ndarray, axis: int | None = None) -> float:
    """2D 全域去均值后的 IQR 归一化相对误差（展平后计算）。"""
    return float(_nmae_iqr_anom(pred, obs, axis=axis))


def NMAE_mean_2D(pred: np.ndarray, obs: np.ndarray, axis: int | None = None) -> float:
    """2D 全域基于均值归一化的相对误差（展平后计算）。"""
    return float(_nmae_mean(pred, obs, axis=axis))


def NMAE_mean_wet_2D(pred: np.ndarray, obs: np.ndarray, axis: int | None = None) -> float:
    """2D 全域基于均值归一化的相对误差（仅 wet，展平后计算）。"""
    return float(_nmae_mean_wet(pred, obs, axis=axis))


def CRMSE_2D(pred: np.ndarray, obs: np.ndarray, axis: int | None = None) -> float:
    """2D 全域 CRMSE（展平后基于距平）。"""
    return float(_crmse(pred, obs, axis=axis))


def PS_2D(pred: np.ndarray, obs: np.ndarray, axis: int | None = None, smooth: int = 7) -> float:
    """2D 相位偏移评分（展平）。"""
    return float(_phase_score(pred, obs, axis=axis, smooth=smooth))


# ---------- 3D（沿时间轴） ----------
def RMSE_3D(pred: np.ndarray, obs: np.ndarray, axis: int = 0) -> np.ndarray:
    """3D RMSE 沿给定轴（常为时间）逐点计算，返回空间分布。"""
    return _rmse(pred, obs, axis=axis)



def RRMSE_3D(pred: np.ndarray, obs: np.ndarray, axis: int = 0, denom: str = "absmean") -> np.ndarray:
    """3D 相对 RMSE 沿给定轴逐点计算，返回空间分布。"""
    return _rrmse(pred, obs, axis=axis, how=denom)



def TCC_3D(pred: np.ndarray, obs: np.ndarray, axis: int = 0) -> np.ndarray:
    """3D 时间相关系数 TCC（沿给定轴做距平后相关），返回空间分布。"""
    pred = np.asarray(pred, dtype=float)
    obs = np.asarray(obs, dtype=float)
    clim_pred = climatology_time_mean(pred, time_axis=axis)
    clim_obs = climatology_time_mean(obs, time_axis=axis)
    cp = np.expand_dims(clim_pred, axis=axis)
    co = np.expand_dims(clim_obs, axis=axis)
    pred, obs = np.broadcast_arrays(pred, obs)
    pred = np.moveaxis(pred, axis, 0)
    obs = np.moveaxis(obs, axis, 0)
    P_anom = pred - clim_pred
    O_anom = obs - clim_obs
    return _pcc_along_axis_3d(P_anom, O_anom, axis=axis)


def R2_3D(pred: np.ndarray, obs: np.ndarray, axis: int = 0) -> np.ndarray:
    """3D 决定系数 R2 沿给定轴逐点计算，返回空间分布。"""
    return _r2(pred, obs, axis=axis)



def Bias_3D(pred: np.ndarray, obs: np.ndarray, axis: int = 0) -> np.ndarray:
    """3D Bias 沿给定轴逐点计算，返回空间分布。"""
    return _bias(pred, obs, axis=axis)


def RBias_3D(pred: np.ndarray, obs: np.ndarray, how: str = "absmean", percent: bool = True, axis: int = 0) -> np.ndarray:
    """3D 相对偏差沿给定轴逐点计算，返回空间分布。"""
    return _rbias(pred, obs, axis=axis, how=how, percent=percent)



def MAE_3D(pred: np.ndarray, obs: np.ndarray, axis: int = 0) -> np.ndarray:
    """3D MAE 沿给定轴逐点计算，返回空间分布。"""
    return _mae(pred, obs, axis=axis)


def MAPE_3D(pred: np.ndarray, obs: np.ndarray, axis: int = 0) -> np.ndarray:
    """3D MAPE 沿给定轴逐点计算，返回空间分布。"""
    return _mape(pred, obs, axis=axis)


def NMAE_sigma_anom_3D(pred: np.ndarray, obs: np.ndarray, axis: int = 0) -> np.ndarray:
    """3D 去均值后的相对误差（百分比）沿给定轴逐点计算，返回空间分布。"""
    return _nmae_sigma_anom(pred, obs, axis=axis)


def NMAE_iqr_anom_3D(pred: np.ndarray, obs: np.ndarray, axis: int = 0) -> np.ndarray:
    """3D 去均值后的 IQR 归一化相对误差（百分比）沿给定轴逐点计算，返回空间分布。"""
    return _nmae_iqr_anom(pred, obs, axis=axis)


def NMAE_mean_3D(pred: np.ndarray, obs: np.ndarray, axis: int = 0) -> np.ndarray:
    """3D 基于均值归一化的相对误差（百分比）沿给定轴逐点计算，返回空间分布。"""
    return _nmae_mean(pred, obs, axis=axis)


def NMAE_mean_wet_3D(pred: np.ndarray, obs: np.ndarray, axis: int = 0) -> np.ndarray:
    """3D 基于均值归一化的相对误差（仅 wet）沿给定轴逐点计算，返回空间分布。"""
    return _nmae_mean_wet(pred, obs, axis=axis)


def CRMSE_3D(pred: np.ndarray, obs: np.ndarray, axis: int = 0) -> np.ndarray:
    """3D CRMSE 沿给定轴逐点计算（基于距平），返回空间分布。"""
    return _crmse(pred, obs, axis=axis)


def ACC(pred: np.ndarray, obs: np.ndarray, area: np.ndarray, time_axis: int = 0, lat_axis: int = -2, lon_axis: int = -1) -> float:
    """距平相关系数 ACC（Anomaly Correlation Coefficient）。

    过程：对每个时间步，先构造距平（减去沿 `time_axis` 的均值），然后对该时间步的
    空间场做面积加权的空间相关（这里用协方差/方差形式实现）。最终对时间平均得到单个分值。
    与 ILAMB 的“空间距平相关”一致，用于衡量空间分布一致性随时间的平均表现。
    """
    clim_pred = climatology_time_mean(pred, time_axis=time_axis)
    clim_obs = climatology_time_mean(obs, time_axis=time_axis)
    anom_pred = pred - np.expand_dims(clim_pred, axis=time_axis)
    anom_obs = obs - np.expand_dims(clim_obs, axis=time_axis)
    num = TU.cal_area_weighted_sum(anom_pred * anom_obs, area=area, lat_axis=lat_axis, lon_axis=lon_axis)
    den = np.sqrt(TU.cal_area_weighted_sum(anom_pred ** 2, area=area, lat_axis=lat_axis, lon_axis=lon_axis) *
                  TU.cal_area_weighted_sum(anom_obs ** 2, area=area, lat_axis=lat_axis, lon_axis=lon_axis))
    acc_t = np.divide(num, den, out=np.full_like(num, np.nan, dtype=float), where=np.isfinite(den) & (den != 0))
    return float(np.nanmean(acc_t))


def Score_Bias_3D(pred: np.ndarray, obs: np.ndarray, area: np.ndarray, lat_axis: int = -2, lon_axis: int = -1, axis: int = 0) -> np.ndarray | float:
    """偏差得分（0~1）：exp(- Epsilon_Bias)。"""
    epsilon_bias = _epsilon_bias(pred, obs, axis=axis)
    epsilon_bias = np.exp(-epsilon_bias)
    score_bias = TU.cal_area_weighted_mean(epsilon_bias, area, lat_axis=lat_axis, lon_axis=lon_axis)
    return score_bias


def Score_RMSE_3D(pred: np.ndarray, obs: np.ndarray, area: np.ndarray, lat_axis: int = -2, lon_axis: int = -1, axis: int = 0) -> np.ndarray | float:
    """整体 RMSE 得分（0~1）：exp(- Epsilon_RMSE)。"""
    epsilon_rmse = _epsilon_rmse(pred, obs, axis=axis)
    epsilon_rmse = np.exp(-epsilon_rmse)
    score_rmse = TU.cal_area_weighted_mean(epsilon_rmse, area, lat_axis=lat_axis, lon_axis=lon_axis)
    return score_rmse


def Score_PS_3D(pred: np.ndarray, obs: np.ndarray, area: np.ndarray, lat_axis: int = -2, lon_axis: int = -1, axis: int = 0, smooth: int = 7) -> np.ndarray:
    """整体相位得分（0~1）：对逐格点 Phase Score 做面积加权平均。"""
    phase_score = _phase_score(pred, obs, axis=axis, smooth=smooth)
    score_phase = TU.cal_area_weighted_mean(phase_score, area, lat_axis=lat_axis, lon_axis=lon_axis)
    return score_phase


def Score_IAV_3D(pred: np.ndarray, obs: np.ndarray, area: np.ndarray, lat_axis: int = -2, lon_axis: int = -1, axis: int = 0) -> np.ndarray | float:
    """整体年际变率得分（0~1）：exp(- Epsilon_IAV)。"""
    epsilon_iav = _epsilon_iav(pred, obs, axis=axis)
    epsilon_iav = np.exp(-epsilon_iav)
    score_iav = TU.cal_area_weighted_mean(epsilon_iav, area, lat_axis=lat_axis, lon_axis=lon_axis)
    return score_iav


def Score_SD_3D(pred: np.ndarray, obs: np.ndarray, axis: int = 0) -> float:
    """空间分布得分（标量，0~1）：基于时间气候态的 S_dist。

    注：此分值本质是“空间结构匹配”评分，非逐格点量，因此返回单个标量，
    不写入格点 NetCDF，而与 ACC 一起写入 Excel 区域汇总。
    """
    return _s_dist(pred, obs, axis=axis)


# ---------- 计算与落盘 ----------
def _daily_climatology_by_season(da: xr.DataArray, season: str, keep_leap: bool = False) -> xr.DataArray:
    """按季节提取多年*日*平均（气候态），返回形状 (dayofyear, y, x)。

    - 先筛选季节（time.dt.season == season），再 groupby dayofyear 求多年平均；
    - 对 DJF 做跨年重排，输出 dayofyear 顺序为 [Dec(335-365/366), Jan-Feb(1-59)]，
      以保证序列在季节内时间上*连续*，便于相位检测；
    - keep_leap=True 时保留 dayofyear=366（仅闰年存在），否则删除。
    """
    sub = da.sel(time=da.time.dt.season == season)
    clim = sub.groupby('time.dayofyear').mean('time')  # (dayofyear, y, x)

    # 处理闰日
    if not keep_leap and 366 in clim['dayofyear'].values:
        clim = clim.drop_sel(dayofyear=366)
    # DJF 重排：将 [335..365/366] 放在前，再接 [1..59]
    if season == 'DJF':
        doy = clim['dayofyear'].values
        dec = doy[doy >= 335]
        jf_max = 60 if (keep_leap and (60 in doy)) else 59
        jf  = doy[(doy >= 1) & (doy <= jf_max)]
        order = np.concatenate([dec, jf])
        clim = clim.sel(dayofyear=order)
    return clim



def model_evaluation(casename: str, var: str, caseds: xr.Dataset, refds: dict[str, xr.Dataset], mask: xr.DataArray, cwrfarea: np.ndarray | xr.DataArray, outdir: str) -> None:
    """按季节计算并保存各项指标（与 ILAMB 口径），并区分两种输入：

    1) **季度平均（逐年季节平均，形状 (year,y,x)）**：用于 RMSE/Bias/TCC/RRMSE/MAE/CRMSE、ACC、
       Score_Bias_3D/Score_RMSE_3D/Score_IAV_3D/Score_SD_3D。
    2) **多年日平均（dayofyear,y,x）**：用于 Score_PS_3D（相位评分）。
    """
    refnames = refds.keys()
    seasons = TU.get_seasons()
    outdir_var = f"{outdir}/{casename}"
    os.makedirs(outdir_var, exist_ok=True)
    for refname in refnames:
        # 为“多年日平均”准备全时段的对齐（不按季节切）
        ref_da_all = refds[refname][var]
        case_da_all = caseds
        ref_da_all, case_da_all, mask_all = xr.align(ref_da_all, case_da_all, mask, join="exact")
        ref_all_m = ref_da_all.where(mask_all)
        case_all_m = case_da_all.where(mask_all)
        # 面积场对齐（按空间维）
        if not isinstance(cwrfarea, xr.DataArray):
            area = xr.DataArray(cwrfarea, coords={"y": mask_all["y"], "x": mask_all["x"]}, dims=("y", "x"))
        else:
            area = cwrfarea
        _, _, area = xr.align(ref_all_m.isel(time=0), case_all_m.isel(time=0), area, join="exact")
        area_np = area.values
        for season in seasons:
            # —— 季节子集（多年 *日* 数据） ——
            ref_da_daily = ref_all_m.sel(time=ref_all_m.time.dt.season == season)
            case_da_daily = case_all_m.sel(time=case_all_m.time.dt.season == season).squeeze(drop=True)
            # 1) 多年“季度平均”（逐年季节平均），形状 (nyear, y, x)
            ref_m_season_mean = ref_da_daily.groupby("time.year").mean("time")
            case_m_season_mean = case_da_daily.groupby("time.year").mean("time")
            refdata_qmean = ref_m_season_mean.values.squeeze()
            casedata_qmean = case_m_season_mean.values.squeeze()
            # 2) 多年“日气候态”（季节内 dayofyear），形状 (dayofyear, y, x)
            ref_clim_doy = _daily_climatology_by_season(ref_all_m, season=season, keep_leap=True)
            case_clim_doy = _daily_climatology_by_season(case_all_m, season=season, keep_leap=True)
            refdata_daily = ref_clim_doy.values.squeeze()
            casedata_daily = case_clim_doy.values.squeeze()
            # —— 季度平均 → 逐格点（沿年轴）指标 ——
            rmse  = RMSE_3D(casedata_qmean, refdata_qmean, axis=0)
            bias  = Bias_3D(casedata_qmean, refdata_qmean, axis=0)
            rbias = RBias_3D(casedata_qmean, refdata_qmean, axis=0)
            tcc   = TCC_3D(casedata_qmean, refdata_qmean, axis=0)
            rrmse = RRMSE_3D(casedata_qmean, refdata_qmean, axis=0, denom="absmean")
            mae   = MAE_3D(casedata_qmean, refdata_qmean, axis=0)
            mape  = MAPE_3D(casedata_qmean, refdata_qmean, axis=0)
            nmae_sigma  = NMAE_sigma_anom_3D(casedata_qmean, refdata_qmean, axis=0)
            nmae_iqr   = NMAE_iqr_anom_3D(casedata_qmean, refdata_qmean, axis=0)
            nmae_mean = NMAE_mean_3D(casedata_qmean, refdata_qmean, axis=0)
            nmae_mean_wet = NMAE_mean_wet_3D(casedata_qmean, refdata_qmean, axis=0)
            crmse = CRMSE_3D(casedata_qmean, refdata_qmean, axis=0)
            # —— 标量评分（基于季度平均） ——
            acc    = ACC(pred=casedata_qmean, obs=refdata_qmean, area=area_np, time_axis=0, lat_axis=-2, lon_axis=-1)
            s_bias = Score_Bias_3D(casedata_qmean, refdata_qmean, area=area_np, lat_axis=-2, lon_axis=-1, axis=0)
            s_rmse = Score_RMSE_3D(casedata_qmean, refdata_qmean, area=area_np, lat_axis=-2, lon_axis=-1, axis=0)
            s_iav  = Score_IAV_3D(casedata_qmean, refdata_qmean, area=area_np, lat_axis=-2, lon_axis=-1, axis=0)
            s_dist = Score_SD_3D(casedata_qmean, refdata_qmean, axis=0)
            # —— 相位评分（基于多年日平均 dayofyear） ——
            s_phase = Score_PS_3D(casedata_daily, refdata_daily, area=area_np, lat_axis=-2, lon_axis=-1, axis=0, smooth=7)
            # 气候态季节平均
            refdata_qclim = climatology_time_mean(refdata_qmean, time_axis=0)
            casedata_qclim = climatology_time_mean(casedata_qmean, time_axis=0)
            # 保存格点 NetCDF（仅 2D 场）
            grid_metrics = {"rmse": [["y", "x"], rmse], "bias": [["y", "x"], bias], "tcc": [["y", "x"], tcc],
                            "rrmse": [["y", "x"], rrmse], "mae": [["y", "x"], mae], "mape": [["y", "x"], mape],
                            "nmae_sigma": [["y", "x"], nmae_sigma], "nmae_iqr": [["y", "x"], nmae_iqr], 
                            "nmae_mean": [["y", "x"], nmae_mean], "nmae_mean_wet": [["y", "x"], nmae_mean_wet],
                            "crmse": [["y", "x"], crmse], "rbias": [["y", "x"], rbias],
                            "clim_ref": [["y", "x"], refdata_qclim], "clim_case": [["y", "x"], casedata_qclim]}
            coords = {"y": caseds.y, "x": caseds.x}
            savepath = f"{outdir}/{casename}/Perform_Area_{casename}_{refname}_{var}_{season}.nc"
            TIO.save_newnc(savepath=savepath, in_dict=grid_metrics, coords=coords)
            # 区域均值 + 标量，写 Excel
            mean_rmse  = TU.cal_area_weighted_mean(rmse, area_np, lat_axis=-2, lon_axis=-1)
            mean_bias  = TU.cal_area_weighted_mean(bias, area_np, lat_axis=-2, lon_axis=-1)
            mean_rbias = TU.cal_area_weighted_mean(rbias, area_np, lat_axis=-2, lon_axis=-1)
            mean_tcc   = TU.cal_area_weighted_mean(tcc, area_np, lat_axis=-2, lon_axis=-1)
            mean_mae   = TU.cal_area_weighted_mean(mae, area_np, lat_axis=-2, lon_axis=-1)
            mean_mape  = TU.cal_area_weighted_mean(mape, area_np, lat_axis=-2, lon_axis=-1)
            mean_crmse = TU.cal_area_weighted_mean(crmse, area_np, lat_axis=-2, lon_axis=-1)
            mean_rrmse = TU.cal_area_weighted_mean(rrmse, area_np, lat_axis=-2, lon_axis=-1)
            mean_nmae_sigma  = TU.cal_area_weighted_mean(nmae_sigma, area_np, lat_axis=-2, lon_axis=-1)
            mean_nmae_iqr    = TU.cal_area_weighted_mean(nmae_iqr, area_np, lat_axis=-2, lon_axis=-1)
            mean_nmae_mean   = TU.cal_area_weighted_mean(nmae_mean, area_np, lat_axis=-2, lon_axis=-1)
            mean_nmae_mean_wet = TU.cal_area_weighted_mean(nmae_mean_wet, area_np, lat_axis=-2, lon_axis=-1)
            mean_metrics = pd.DataFrame({
                "mean_rmse":   [mean_rmse],
                "mean_rbias":  [mean_rbias],
                "mean_bias":   [mean_bias],
                "mean_mae":    [mean_mae],
                "mean_mape":   [mean_mape],
                "mean_nmae_sigma":   [mean_nmae_sigma],
                "mean_nmae_iqr":   [mean_nmae_iqr],
                "mean_nmae_mean":   [mean_nmae_mean],
                "mean_nmae_mean_wet":   [mean_nmae_mean_wet],
                "mean_crmse":  [mean_crmse],
                "mean_rrmse":  [mean_rrmse],
                "mean_tcc":    [mean_tcc],
                "acc":    [acc],
                "s_dist": [s_dist],
                "s_bias": [s_bias],
                "s_rmse":[s_rmse],
                "s_iav":[s_iav],
                "s_phase":[s_phase],
            })
            savepath = f"{outdir}/{casename}/Perform_Mean_{casename}_{refname}_{var}_{season}.xlsx"
            TIO.save_excel(savepath=savepath, in_df=mean_metrics)


