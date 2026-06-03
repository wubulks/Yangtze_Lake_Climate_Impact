import math
import logging
import xarray as xr
import numpy as np
from numba import njit
from numba import prange
logger = logging.getLogger(__name__)

@njit
def pres_hybrid_ccm(ps, p0, hya, hyb):
    """
    定义计算混合层压力的函数
    
    参数:
    ps : numpy.ndarray
        地面气压，单位为 Pa 或 hPa，维度为 (time, lat, lon) 或 (time, lev, lat, lon)
    p0 : float
        参考压力，单位与 ps 相同。
    hya : numpy.ndarray
        混合系数 a(k)，无单位的一维数组，长度为 lev
    hyb : numpy.ndarray
        混合系数 b(k)，无单位的一维数组，长度为 lev
    
    返回:
    numeric
        计算的压力，维度为 (time, lev, lat, lon)
    """
    # 确保 hya 和 hyb 是二维的，维度 (lev, 1, 1)，以便广播
    if ps.ndim == 3:
        ntime, nlat, nlon = ps.shape
        pressure = np.zeros((ntime, len(hya), nlat, nlon))
    elif ps.ndim == 2:
        nlat, nlon = ps.shape
        pressure = np.zeros((len(hya), nlat, nlon))
    else:
        raise ValueError("ps 的维度必须是 (time, lat, lon) 或 (lat, lon)")
    klev = len(hya)
    hya_reshaped = hya.reshape(klev, 1, 1) # 将 hya 重塑为 (lev, 1, 1)
    hyb_reshaped = hyb.reshape(klev, 1, 1) # 将 hyb 重塑为 (lev, 1, 1)
    
    if ps.ndim == 3:
        for t in prange(ntime):
            for k in range(len(hya)):
                pressure[t, k, :, :] = hya_reshaped[k] * p0 + hyb_reshaped[k] * ps[t, :, :]
    else:
        for k in prange(len(hya)):
            pressure[k, :, :] = hya_reshaped[k] * p0 + hyb_reshaped[k] * ps[:, :]
    
    return pressure



def cz2ccm(ps, phis, tv, p0, hyam, hybm, hyai, hybi, debug=False):
    """
    Calculate geopotential height using the hybrid coordinate system.

    Parameters:
    -----------
    ps : ndarray
        2D array of surface pressures (Pa) with dimensions (lat, lon).
    phis : ndarray
        2D array of surface geopotential with dimensions (lat, lon).
    tv : ndarray
        3D array of virtual temperature (K) with dimensions (lev, lat, lon).
    p0 : float
        Base pressure (Pa).
    hyam : ndarray
        1D array of hybrid A coefficients for mid-levels.
    hybm : ndarray
        1D array of hybrid B coefficients for mid-levels.
    hyai : ndarray
        1D array of hybrid A coefficients for interfaces.
    hybi : ndarray
        1D array of hybrid B coefficients for interfaces.

    Returns:
    --------
    z2 : ndarray
        3D array of geopotential height (m) with dimensions (lev, lat, lon).
    """

    if debug:
        logging.info("cz2ccm function called with the following inputs:")
        logging.info(f"ps.shape: {ps.shape}")
        logging.info(f"phis.shape: {phis.shape}")
        logging.info(f"tv.shape: {tv.shape}")
        logging.info(f"p0: {p0}")
        logging.info(f"hyam.shape: {hyam.shape}")
        logging.info(f"hybm.shape: {hybm.shape}")
        logging.info(f"hyai.shape: {hyai.shape}")
        logging.info(f"hybi.shape: {hybi.shape}")
        logging.info(hyam)
        logging.info(hybm)
        logging.info(hyai)
        logging.info(hybi)

    nlat, mlon = ps.shape
    klev = len(hyam)
    klev1 = len(hyai)

    z2 = np.zeros((klev, nlat, mlon))
    pmln = np.zeros((klev + 1, nlat, mlon))
    pterm = np.zeros((klev, nlat, mlon))

    if debug:
        logging.info(f"nlat: {nlat}, mlon: {mlon}, klev: {klev}, klev1: {klev1}")
        logging.info(f"z2.shape: {z2.shape}")
        logging.info(f"pmln.shape: {pmln.shape}")
        logging.info(f"pterm.shape: {pterm.shape}")

    hyba = np.zeros((2, klev + 1))
    hybb = np.zeros((2, klev + 1))

    # Copy to temporary arrays
    hyba[0, :] = hyai
    hybb[0, :] = hybi

    # Copy midpoint coefficients to the second row (index 1) of HYBA and HYBB
    hyba[1, 1:klev1] = hyam
    hybb[1, 1:klev1] = hybm

    if debug:
        logging.info("Intermediate arrays initialized.")
        logging.info(f"hyba.shape: {hyba.shape}")
        logging.info(f"hybb.shape: {hybb.shape}")

    pmln[0, :, :] = np.log(p0 * hyba[1, klev] + ps * hybb[0, klev])
    pmln[-1, :, :] = np.log(p0 * hyba[1, 0] + ps * hybb[0, 0])
    for k in range(klev-1, 0, -1):
        idx = klev - k
        arg = p0 * hyba[1, idx] + ps * hybb[1, idx]
        pmln[k, :, :] = np.where(arg > 0.0, np.log(arg), 0.0)

    # Calculate geopotential height Z2
    R = 287.04  # Gas constant for dry air (J/(kg*K))
    G0 = 9.80616  # Gravity (m/s^2)
    RBYG = R / G0

    # Eq 3.a.109.2
    for k in range(1, klev-1):
        pterm[k, :, :] = RBYG * tv[k, :, :] * 0.5 * (pmln[k+1, :, :] - pmln[k-1, :, :])

    # Eq 3.a.109.5 and 3.a.109.2
    for k in range(0, klev-1):
        z2[k, :, :] = phis / G0 + RBYG * tv[k, :, :] * 0.5 * (pmln[k+1, :, :] - pmln[k, :, :])

    # Step 5: Special Case for Last Layer (3.a.109.5)
    k = klev-1
    z2[k, :, :] = phis / G0 + RBYG * tv[k, :, :] * (np.log(ps * hybb[0, 0]) - pmln[k, :, :])

    # Eq 3.a.109.4
    for k in range(0, klev-1):
        l = klev-1
        z2[k, :, :] = z2[k, :, :] + RBYG * tv[l, :, :] * (np.log(ps * hybb[0, 0]) - 0.5 * (pmln[l-1, :, :] + pmln[l, :, :]))

    # Add thickness of the remaining full layers (Eq 3.a.109.3)
    for k in range(0,klev-2):
        for l in range(k+1, klev):
            z2[k, :, :] = z2[k, :, :] + pterm[l, :, :]

    return z2



def relhum(w, t, p, clip=True):
    """
    参考： Murphy & Koop (2005)
    计算相对湿度(%)，与 NCL `relhum` 一致的相变处理：
      T >= 0°C: 相对于水
     -20°C < T < 0°C: 水/冰混相线性过渡
      T <= -20°C: 相对于冰
      计算结果与NCL实际算法有所不同，但差异极小。

    参数
    ----
    t : array_like
        温度 (K)。标量或任意形状数组。
    w : array_like
        混合比 (kg/kg)。需与 t 可广播到相同形状。
    p : array_like
        压力 (Pa)。需与 t 可广播到相同形状。

    返回
    ----
    rh : np.ndarray
        输出为 float64
    """

    # -------- dtype 策略：计算用 float64，最后按 NCL 规则决定输出精度 --------
    t64 = np.asarray(t, dtype=np.float64)
    w64 = np.asarray(w, dtype=np.float64)
    p64 = np.asarray(p, dtype=np.float64)

    # -------- 广播到共同形状 --------
    T, W, P = np.broadcast_arrays(t64, w64, p64)

    # 常数：干湿气体常数比 epsilon = Mw/Md ≈ 0.62197
    EPS = 0.62197

    # -------- 实际水汽分压 e (Pa) --------
    # w = epsilon * e / (p - e)  =>  e = p * w / (epsilon + w)
    e = P * W / (EPS + W)

    # -------- 饱和水汽压 (Murphy & Koop 2005, 单位 Pa) --------
    # 参考：
    #   ln(es_ice)   =  9.550426 - 5723.265/T + 3.53068*ln(T) - 0.00728332*T
    #   ln(es_water) = 54.842763 - 6763.22/T - 4.210*ln(T) + 0.000367*T
    #                  + tanh(0.0415*(T-218.8)) * (53.878 - 1331.22/T - 9.44523*ln(T) + 0.014025*T)
    logT = np.log(T)

    ln_es_ice = (
        9.550426
        - 5723.265 / T
        + 3.53068 * logT
        - 0.00728332 * T
    )
    es_ice = np.exp(ln_es_ice)

    ln_es_wat = (
        54.842763
        - 6763.22 / T
        - 4.210 * logT
        + 0.000367 * T
        + np.tanh(0.0415 * (T - 218.8))
        * (53.878 - 1331.22 / T - 9.44523 * logT + 0.014025 * T)
    )
    es_wat = np.exp(ln_es_wat)

    # -------- 相变规则：混相（-20°C~0°C）、水相（>=0°C）、冰相（<=-20°C） --------
    Tc = T - 273.15  # 转为摄氏度
    # 权重 alpha: 0°C -> 0（全水）, -20°C -> 1（全冰），线性过渡
    alpha = np.clip(-Tc / 20.0, 0.0, 1.0)
    es_mix = alpha * es_ice + (1.0 - alpha) * es_wat

    # -------- 相对湿度 (%)，允许 >100% 以表示过饱和 --------
    rh = 100.0 * e / es_mix
    if clip:
        rh = np.clip(rh, 0.0, 100.0)  # 下限裁剪为 0%，上限裁剪为 100%

    # 返回指定精度
    return rh.astype(np.float64, copy=False)


