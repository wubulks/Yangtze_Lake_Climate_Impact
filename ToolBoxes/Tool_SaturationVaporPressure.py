from __future__ import annotations
import numpy as np

def qsadv(T: float | np.ndarray, p: float | np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    计算饱和水汽压 es、对温度导数 esdT、饱和比湿 qs 及其对温度导数 qsdT。
    与 CoLM/Fortran 代码保持一致（Flatau 1992 多项式；水/冰两相；td 裁剪到 [-75, 75] ℃）。
    """
    # 向量化输入
    T = np.asarray(T, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)

    # ---- Fortran DATA 常量（科学计数法；值与源代码逐项一致）----
    a = np.array([
         6.11213476,       0.444007856,     0.143064234e-01, 
         0.264461437e-03,  0.305903558e-05, 0.196237241e-07,
         0.892344772e-10, -0.373208410e-12, 0.209339997e-15,
    ], dtype=np.float64)
    b = np.array([
         0.444017302,      0.286064092e-01, 0.794683137e-03, 
         0.121211669e-04,  0.103354611e-06, 0.404125005e-09,
        -0.788037859e-12, -0.114596802e-13, 0.381294516e-16,
    ], dtype=np.float64)
    c = np.array([
         6.11123516,       0.503109514,     0.188369801e-01, 
         0.420547422e-03,  0.614396778e-05, 0.602780717e-07, 
         0.387940929e-09,  0.149436277e-11, 0.262655803e-14,
    ], dtype=np.float64)
    d = np.array([
         0.503277922,      0.377289173e-01, 0.126801703e-02, 
         0.249468427e-04,  0.313703411e-06, 0.257180651e-08, 
         0.133268878e-10,  0.394116744e-13, 0.498070196e-16,
    ], dtype=np.float64)

    def horner(x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        # a0 + x*(a1 + x*(a2 + ...))；按 Fortran 展开顺序
        y = 0.0
        for c_ in coeffs[::-1]:
            y = y * x + c_
        return y

    # CoLM/Fortran input is usually K, but some project-level diagnostics are
    # stored in degC after preprocessing. Support both units defensively.
    td = np.where(T > 150.0, T - 273.16, T)
    td = np.clip(td, -75.0, 75.0)

    mask = td >= 0.0

    # 多项式评估（水/冰）
    es_w = a[0] + td*(a[1] + td*(a[2] + td*(a[3] + td*(a[4] + td*(a[5] + td*(a[6] + td*(a[7] + td*a[8])))))))
    esdT_w = b[0] + td*(b[1] + td*(b[2] + td*(b[3] + td*(b[4] + td*(b[5] + td*(b[6] + td*(b[7] + td*b[8])))))))

    es_i = c[0] + td*(c[1] + td*(c[2] + td*(c[3] + td*(c[4] + td*(c[5] + td*(c[6] + td*(c[7] + td*c[8])))))))
    esdT_i = d[0] + td*(d[1] + td*(d[2] + td*(d[3] + td*(d[4] + td*(d[5] + td*(d[6] + td*(d[7] + td*d[8])))))))

    # 选择水/冰相
    es   = np.where(mask, es_w,   es_i)
    esdT = np.where(mask, esdT_w, esdT_i)

    # 单位换算
    es   = es   * 100   # Pa
    esdT = esdT * 100   # Pa/K

    # 中间量与结果
    denom = p - 0.378 * es
    vp = np.divide(1.0, denom, where=denom != 0.0)
    vp = np.where(denom == 0.0, np.inf, vp)

    vp1  = 0.622 * vp
    vp2  = vp1 * vp

    qs   = es * vp1                 # kg/kg
    qsdT = esdT * vp2 * p           # 1/K

    return es, esdT, qs, qsdT
