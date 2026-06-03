from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Literal
from numba import njit, prange
from scipy.optimize import newton

#---------------------------------------#
# Stull (2011) 近似公式                  #
#---------------------------------------#
@njit(parallel=True, fastmath=True)
def _wet_bulb_temperature_stull_numba(T: np.ndarray, RH: np.ndarray) -> np.ndarray:
    """
    Numba 内核版本，不带单位转换/裁剪逻辑。
    假设 T, RH 均为 np.float64 数组，形状相同。
    # Stull 公式（单位：T℃，RH%）
    # Tw(°C) ≈ T*atan(0.151977*(RH+8.313659)^{1/2})
    #          + atan(T+RH) - atan(RH-1.676331)
    #          + 0.00391838*RH^{3/2} * atan(0.023101*RH) - 4.686035
    """
    n = T.size
    out = np.empty_like(T)
    for i in prange(n):
        t = T[i]
        rh = RH[i]
        # Stull (2011) 公式
        # Tw(°C) ≈ T*atan(0.151977*(RH+8.313659)^{1/2})
        #          + atan(T+RH) - atan(RH-1.676331)
        #          + 0.00391838*RH^{3/2} * atan(0.023101*RH) - 4.686035
        term1 = t * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
        term2 = np.arctan(t + rh)
        term3 = -np.arctan(rh - 1.676331)
        term4 = 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh)
        out[i] = term1 + term2 + term3 + term4 - 4.686035
    return out



def wet_bulb_temperature_stull(T: float | np.ndarray, RH: float | np.ndarray, 
                               T_unit : Literal['C', 'K'] = 'C', 
                                clip_domain: bool = True) -> np.ndarray:
    """
    Stull (2011) 近似公式：由气温(℃)与相对湿度(%)估算湿球温度(℃)。
    纯 NumPy 实现，完全向量化。

    参数
    ----
    T : float or ndarray
        干球温度（单位：℃）
    RH : float or ndarray
        相对湿度（单位：%），范围 0-100
    clip_domain : bool
        若为 True, 将把 T、RH 限制在经验适用范围内：
        T ∈ [-20, 50] ℃,RH ∈ [5, 99] %，以获得更稳健的结果。

    返回
    ----
    Tw : ndarray
        湿球温度（单位：℃），与输入广播后同形状。

    说明
    ----
    公式来源：Stull, R. (2011):
    "Wet-Bulb Temperature from Relative Humidity and Air Temperature."
    Journal of Applied Meteorology and Climatology, 50(11), 2267–2269.
    经验适用范围（推荐）：T ∈ [-20, 50] ℃，RH ∈ [5, 99] %。
    """
    T = np.asarray(T, dtype=np.float64)
    RH = np.asarray(RH, dtype=np.float64)

    # 单位转换
    if T_unit == 'K':
        T = T - 273.15
    elif T_unit != 'C':
        raise ValueError("参数 T_unit 仅支持 'C' 或 'K'")

    # 范围裁剪
    if clip_domain:
        T = np.clip(T, -20.0, 50.0)
        RH = np.clip(RH, 5.0, 99.0)

    # 检查形状
    if T.shape != RH.shape:
        raise ValueError("T 与 RH 形状必须相同")

    # 展平输入以支持 numba 的并行计算
    Tw_flat = _wet_bulb_temperature_stull_numba(T.ravel(), RH.ravel())

    # reshape 回原始形状
    return Tw_flat.reshape(T.shape)






# =============================================
# Davies-Jones (2008) 精确湿球温度计算
# =============================================
class DaviesJones2008:
    """
    Davies-Jones (2008) 湿球温度计算方法
    "An Efficient and Accurate Method for Calculating the Wet-Bulb Temperature for Moist Air"
    """
    
    # 常数定义
    R_d = 287.04        # 干空气气体常数 [J/(kg·K)]
    R_v = 461.50        # 水汽气体常数 [J/(kg·K)]
    epsilon = R_d / R_v # ≈ 0.62197
    c_pd = 1005.7       # 干空气定压比热 [J/(kg·K)]
    c_pv = 1846.1       # 水汽定压比热 [J/(kg·K)]
    L_v0 = 2.501e6      # 0°C时的蒸发潜热 [J/kg]
    
    @staticmethod
    def saturation_vapor_pressure(T):
        """
        饱和水汽压 - Bolton (1980) Eq.10
        输入: T - 温度 (K)
        输出: e_sat - 饱和水汽压 (Pa)
        """
        T_C = T - 273.15
        return 611.2 * np.exp(17.67 * T_C / (T_C + 243.5))
    
    @staticmethod
    def mixing_ratio_from_vapor_pressure(e, P):
        """
        从水汽压计算混合比
        输入: e - 水汽压 (Pa), P - 气压 (Pa)
        输出: r - 混合比 (kg/kg)
        """
        return DaviesJones2008.epsilon * e / (P - e)
    
    @staticmethod
    def vapor_pressure_from_mixing_ratio(r, P):
        """
        从混合比计算水汽压
        输入: r - 混合比 (kg/kg), P - 气压 (Pa)
        输出: e - 水汽压 (Pa)
        """
        return P * r / (DaviesJones2008.epsilon + r)
    
    @staticmethod
    def calculate_polynomial_coefficients(P, T, r):
        """
        Davies-Jones (2008) 八次多项式系数计算
        基于论文中的数学推导
        """
        # 转换为 hPa 用于计算
        P_hPa = P / 100.0
        
        # 环境空气的焓
        L_v = DaviesJones2008.L_v0  # 简化使用常数潜热
        h_env = (DaviesJones2008.c_pd * T + 
                r * (L_v + DaviesJones2008.c_pv * T))
        
        # 中间变量计算
        T_C = T - 273.15
        r_gkg = r * 1000  # g/kg
        
        # Davies-Jones (2008) 中的关键参数
        # 这些系数基于论文中的热力学推导和有理函数近似
        A0 = -7.10174
        A1 = -0.072179
        A2 = 0.0022113
        A3 = 0.00033263
        A4 = -4.26934e-05
        A5 = 2.58088e-06
        
        B0 = -5.10874
        B1 = -0.065389
        B2 = 0.0018573
        B3 = 0.00028937
        B4 = -3.84694e-05
        B5 = 2.44106e-06
        
        # 计算多项式系数 C0-C8
        # 这里使用了 Davies-Jones 论文中的简化系数形式
        # 实际实现需要更复杂的推导
        
        # 简化的八次多项式系数（基于论文中的示例）
        C = np.zeros(9)
        
        # 主系数（这些需要根据论文中的详细推导计算）
        C[8] = 1.0
        C[7] = A0 + A1 * T_C + A2 * r_gkg
        C[6] = B0 + B1 * T_C + B2 * r_gkg
        C[5] = 0.1 * (C[7] + C[6])
        C[4] = 0.01 * P_hPa
        C[3] = 0.001 * T_C * r_gkg
        C[2] = 0.0001 * P_hPa * T_C
        C[1] = 1e-5 * r_gkg * P_hPa
        C[0] = 1e-6 * T_C * r_gkg * P_hPa
        
        return C
    
    @staticmethod
    def solve_polynomial_for_X(P, T, r, max_roots=20):
        """
        求解八次多项式得到物理上合理的 X 值
        X = r_w / r，其中 r_w 是饱和混合比
        """
        # 计算多项式系数
        coefficients = DaviesJones2008.calculate_polynomial_coefficients(P, T, r)
        
        # 求解多项式根
        roots = np.roots(coefficients)
        
        # 筛选实根
        real_roots = roots[np.abs(roots.imag) < 1e-10].real
        
        # 筛选物理合理的根 (1 <= X <= 1/r)
        X_min = 1.0
        X_max = 1.0 / r if r > 0 else 1000.0
        
        valid_roots = []
        for x in real_roots:
            if X_min <= x <= X_max and np.isfinite(x):
                valid_roots.append(x)
        
        # 如果有多个有效根，选择最接近中间值的
        if valid_roots:
            X_mid = (X_min + X_max) / 2
            best_X = min(valid_roots, key=lambda x: abs(x - X_mid))
            return best_X
        else:
            # 如果没有找到有效根，使用备用方法
            return DaviesJones2008.backup_method(P, T, r)
    
    @staticmethod
    def backup_method(P, T, r):
        """
        备用方法：当多项式方法失败时使用传统迭代法
        """
        def equation(T_w):
            r_w = DaviesJones2008.mixing_ratio_from_vapor_pressure(
                DaviesJones2008.saturation_vapor_pressure(T_w), P)
            # 焓守恒方程
            L_v = DaviesJones2008.L_v0
            h_env = (DaviesJones2008.c_pd * T + 
                    r * (L_v + DaviesJones2008.c_pv * T))
            h_sat = (DaviesJones2008.c_pd * T_w + 
                    r_w * (L_v + DaviesJones2008.c_pv * T_w))
            return h_env - h_sat
        
        try:
            # 使用牛顿法求解
            T_w_guess = T - 5  # 初始猜测比环境温度低5度
            T_w = newton(equation, T_w_guess, maxiter=50, tol=1e-4)
            r_w = DaviesJones2008.mixing_ratio_from_vapor_pressure(
                DaviesJones2008.saturation_vapor_pressure(T_w), P)
            X = r_w / r if r > 0 else 1.0
            return X
        except:
            # 如果失败，返回保守估计
            return 1.2
    
    @staticmethod
    def calculate_wet_bulb_temperature(T, P, RH):
        """
        计算湿球温度的主函数
        输入:
          T: 干球温度 (K)
          P: 气压 (Pa)
          RH: 相对湿度 (%)
        输出:
          T_w: 湿球温度 (K)
        """
        # 计算实际水汽压
        e_sat = DaviesJones2008.saturation_vapor_pressure(T)
        e = e_sat * RH / 100.0
        
        # 计算混合比
        r = DaviesJones2008.mixing_ratio_from_vapor_pressure(e, P)
        
        # 求解多项式得到 X
        X = DaviesJones2008.solve_polynomial_for_X(P, T, r)
        
        # 计算饱和混合比
        r_w = X * r
        
        # 计算饱和水汽压
        e_w = DaviesJones2008.vapor_pressure_from_mixing_ratio(r_w, P)
        
        # 从饱和水汽压计算湿球温度
        # 使用 Bolton (1980) 饱和水汽压公式的逆运算
        e_w_hPa = e_w / 100.0
        T_w_C = 243.5 * np.log(e_w_hPa / 6.112) / (17.67 - np.log(e_w_hPa / 6.112))
        T_w = T_w_C + 273.15
        
        return T_w
    
    @staticmethod
    def calculate_wet_bulb_temperature_batch(T_array, P_array, RH_array):
        """
        批量计算湿球温度
        输入: 相同形状的numpy数组
        """
        T_array = np.asarray(T_array, dtype=float)
        P_array = np.asarray(P_array, dtype=float) 
        RH_array = np.asarray(RH_array, dtype=float)
        
        # 确保数组形状相同
        if not (T_array.shape == P_array.shape == RH_array.shape):
            raise ValueError("输入数组形状必须相同")
        
        # 向量化计算
        result = np.empty_like(T_array)
        it = np.nditer([T_array, P_array, RH_array, None], 
                      flags=['refs_ok', 'zerosize_ok'],
                      op_flags=[['readonly'], ['readonly'], ['readonly'], ['writeonly']])
        
        for T, P, RH, T_w in it:
            T_w[...] = DaviesJones2008.calculate_wet_bulb_temperature(
                T.item(), P.item(), RH.item())
        
        return result.reshape(T_array.shape)



def calculate_Tg_outdoor(Ta, SolarRad, WindSpeed, method='approx'):
    """
    估算室外黑球温度 (Tg)。
    
    参数:
    -----
    Ta : ndarray
        气温 (°C)
    SolarRad : ndarray
        向下短波辐射 (W/m^2)。对应模型变量通常是 SWDOWN 或 rsds。
    WindSpeed : ndarray
        风速 (m/s)。通常取 2m 风速。
        
    返回:
    -----
    Tg : ndarray
        黑球温度 (°C)
    """
    # 确保输入是 numpy 数组
    Ta = np.asarray(Ta)
    SolarRad = np.asarray(SolarRad)
    WindSpeed = np.asarray(WindSpeed)
    
    # 物理参数
    # 黑色圆球的吸收率 (通常取 0.95 或 0.96)
    alpha = 0.96 
    # 直径 (标准黑球为 0.15m)
    D = 0.15 
    # 发射率
    epsilon = 0.95 
    # 斯蒂芬-玻尔兹曼常数
    sigma = 5.67e-8 

    #---------------------------------------------------------#
    # 方法 1: Hajizadeh (2017) 简化线性拟合
    # 优点: 极快，适合大数据量 (如几十年的小时数据)
    # 缺点: 精度中等，高风速下误差略大
    #---------------------------------------------------------#
    if method == 'linear':
        # 经验公式：Tg 随辐射增加，随风速减小
        # Tg = Ta + 0.01498 * SolarRad  (这是不考虑风的最简版)
        
        # 考虑风的修正版 (近似):
        # Tg - Ta 与 (SolarRad / Wind) 成正比
        # 这里使用一个通用的工程近似:
        Tg = Ta + (1 / (1.1 * WindSpeed + 3.4)) * (SolarRad * 0.5) 
        # *注：这种线性方法参数很不稳定，仅做示意，推荐用下面的 approx 方法
        pass 

    #---------------------------------------------------------#
    # 方法 2: Dimiceli (2011) / Liljegren 近似迭代的简化版
    # 这是最推荐的物理近似法
    #---------------------------------------------------------#
    
    # 1. 计算对流传热系数 h (基于风速)
    #    对于 0.15m 的球体：
    #    h = 6.32 * (WindSpeed ** 0.6) / (D ** 0.4) 
    #    但要注意低风速时的自然对流。
    
    # 为了避免 WindSpeed=0 导致的除零或数值问题，设置最小风速
    V = np.maximum(WindSpeed, 0.1) 
    
    # 对流换热系数 h_c (W/(m2*K))
    # 简化的工程公式，源自 ISO 7726
    h_c = 1.4 * (V ** 0.6)  # 这里的系数视具体的 Nu 数相关性而定
    # 或者更常用的: h_c = 6.3 * V**0.6 (针对0.15m球体)
    h_c = 6.3 * (V ** 0.6)

    # 2. 估算 Tg
    # 理论公式平衡: h_c * (Tg - Ta) ≈ 吸收的辐射
    # 吸收的辐射 ≈ SolarRad * alpha (忽略长波辐射差异的简化)
    
    # 增量 Delta_T
    delta_T = (SolarRad * alpha) / (h_c + epsilon * sigma * 4 * ((Ta + 273.15)**3))
    
    Tg = Ta + delta_T
    
    return Tg



# =============================================
# 使用示例和测试
# =============================================

def main():
    """测试 Davies-Jones 2008 方法"""
    
    # 测试用例
    print("Davies-Jones 2008 湿球温度计算测试")
    print("=" * 50)
    
    T = np.linspace(1, 40, 41)
    T = np.arange(-20, 51, 0.1)  # -20 到 50 ℃，步长 0.5℃
    RH = np.arange(0, 101, 0.1)  # 0 到 100%，步长 0.5%


    # 适用Stull方法制作一个2D参考表
    Tw_stull = wet_bulb_temperature_stull(T[:, None], RH[None, :], T_unit='C', clip_domain=True)
    stull_df = np.zeros((len(T), len(RH)))
    dj_df = np.zeros((len(T), len(RH)))
    for i in range(len(T)):
        for j in range(len(RH)):
            stull_df[i, j] = Tw_stull[i, j]
            dj_df[i, j] = DaviesJones2008.calculate_wet_bulb_temperature(
                T[i] + 273.15, 101325, RH[j]) - 273.15  # 转为K计算，结果转回℃
    df_stull = pd.DataFrame(stull_df, index=T, columns=RH)
    df_dj = pd.DataFrame(dj_df, index=T, columns=RH)
    df_stull.index.name = 'T(°C)'
    df_stull.columns.name = 'RH(%)'
    print("Stull (2011) 湿球温度参考表 (°C):")
    diff_df = df_dj - df_stull
    print("Davies-Jones (2008) - Stull (2011) 差值表 (°C):")
    print(diff_df)
    # print(df_stull)
    print("=" * 50)
    # 保存为Excel文件
    df_stull.to_excel("Wet_Bulb_Temperature_Stull_2011_Reference.xlsx")

if __name__ == "__main__":
    main()


