import os
import gc 
import time
import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm
from scipy import stats
from dataclasses import dataclass
from joblib import Parallel, delayed
from contextlib import contextmanager
from typing import Literal, Optional, Tuple, Dict, Any, Sequence, List
from statsmodels.stats.multitest import multipletests

# 自定义工具箱
import ToolBoxes.Utils as TU
import ToolBoxes.Tool_InputOutput as TIO


# 可选：限制 BLAS 线程，避免与多进程过度并发
try:
    from threadpoolctl import threadpool_limits as _threadpool_limits
except Exception:
    @contextmanager
    def _threadpool_limits(limits=None):
        yield

"""
分析库（Significance Analysis Library）

● Paired_t-test
    适用范围：同一对象/地点的成对观测，关心均值差；差值近似正态，或样本量较大（中心极限定理适用）。
    优点：实现简单、统计功效高；可提供均值差的置信区间与效应量；计算极快。
    局限：对异常值和非正态分布敏感，不适合极端事件类变量。

● Wilcoxon_signed-rank test
    适用范围：成对观测，关注中位数（位置）差；差值分布不要求正态，但最好近似对称。
    优点：非参数、对异常值更稳健；在非正态数据下比 t 检验更可靠。
    局限：若差值分布极度偏斜或零差较多时，统计功效下降。

● Paired_permutation_test (mean difference, sign-flip)
    适用范围：成对观测；在零假设下，差值的正负可视为随机；用于检验均值差是否为 0。
    优点：完全分布自由；给出精确或近似精确的 p 值（随置换次数收敛）；对偏态与厚尾分布非常稳健。
    局限：计算量随样本量和置换次数迅速增加。

● Paired_bootstrap (mean difference)
    适用范围：成对观测，需要获得均值差的**置信区间（CI）**与近似 p 值；适用于任意分布或样本偏小。
    优点：非参数；可直接给出百分位法/BCa CI；对复杂分布与异方差稳健；可灵活扩展到其他统计量（如中位数、方差等）。
    局限：p 值为近似估计；结果受重采样次数影响。

● Sign_test（符号检验）
    适用范围：成对观测，差值方向（正/负）是主要关注点；不考虑幅度。
    优点：最稳健的非参数检验；对极端异常值完全不敏感；仅基于符号信息即可给出精确二项分布 p 值。
    局限：功效较低，忽略差值大小信息；适合方向性一致性分析（如“有湖是否普遍强于无湖”）。

● Cliff’s Delta（效应量检验）
    适用范围：独立或配对样本；用于量化效应强度与方向（如湖泊效应强弱）。
    优点：非参数指标；提供直观的效应强度（δ ∈ [-1, 1]）；
    |δ| < 0.147 → 弱效应；0.147–0.33 → 中效应；>0.33 → 强效应。
    局限：不提供显著性 p 值，但适合与 Wilcoxon / Permutation 检验联合解释。

● Mann–Whitney U test（Wilcoxon rank-sum test）
    适用范围：两组独立样本（如湖区 vs 非湖区格点），比较中位数或分布位置差异；不要求正态。
    优点：非参数；适用于非正态与异方差数据；当样本量中等时功效接近 t 检验。
    局限：假设两组分布形状相似（仅位置不同）；若分布差异显著，解释需谨慎。

● Cramér–von Mises test（两样本分布检验）
    适用范围：两组独立样本；检验两组的整体分布形状是否不同（不仅均值/中位数）。
    优点：对尾部分布变化更敏感；比 Kolmogorov–Smirnov 检验更平滑、功效更高。
    局限：不适用于配对样本；无法直接反映差异方向。
"""

# ---------- helpers ----------
def _to_array(arr1: object) -> np.ndarray:
    """将输入转为 numpy.ndarray（零拷贝/视图优先）。"""
    return np.asarray(arr1)



def _finite_mask(arr1: object, arr2: object) -> np.ndarray:
    """两数组对应位置均为有限值（非 NaN/Inf）的布尔掩码。"""
    arr1 = _to_array(arr1); arr2 = _to_array(arr2)
    return np.isfinite(arr1) & np.isfinite(arr2)


# =========================
# 数据检查 & 预处理
# =========================
def to_clean_paired_arrays(arr1: np.ndarray, arr2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    将两组成对数据转为同长 numpy 数组，并去除任一侧为 NaN 的配对样本。
    """
    if arr1.shape != arr2.shape:
        raise ValueError("arr1 与 arr2 的形状必须一致（配对数据）。")
    mask = _finite_mask(arr1, arr2)
    arr1, arr2 = arr1[mask], arr2[mask]
    if arr1.size < 2:
        arr1, arr2 = np.array([]), np.array([])
        # raise ValueError("有效配对样本量 < 2，无法进行检验。")
    return arr1, arr2



def is_degenerate(diff: np.ndarray) -> Tuple[bool, Optional[float]]:
    """
    判断差值是否退化：全相等。如果是，返回 (True, 常数差)；否则 (False, None)。
    """
    if np.allclose(diff, diff[0], equal_nan=False):
        return True, float(diff[0])
    return False, None



# =========================
# 诊断：正态性 & SD
# =========================
def shapiro_p(diff: np.ndarray) -> np.float64:
    """
    对差值做 Shapiro-Wilk 正态性检验。样本过大/过小返回 np.nan。
    """
    n = diff.size
    if n < 3 or n > 5000:
        return np.nan  # Shapiro-Wilk 对样本量要求
    try:
        return np.float64(stats.shapiro(diff).pvalue)
    except Exception:
        return np.nan



def sd_diff(diff: np.ndarray) -> np.float64:
    """差值的样本标准差（ddof=1），样本量为 1 时返回 NaN。"""
    return np.float64(np.std(diff, ddof=1)) if diff.size > 1 else np.nan

# =========================
# 效应量 & 置信区间
# =========================
def cohens_dz(diff: np.ndarray) -> np.float64:
    """
    配对设计的 Cohen's dz = mean(diff) / sd(diff)。
    若 sd(diff)=0 则返回 0（或按需改为 np.inf）。
    """
    sd = sd_diff(diff)
    if sd == 0:
        dz = np.float64(0)  # 或 np.inf
    elif np.isnan(sd):
        dz = np.float64(np.nan)
    else:
        dz = np.float64(np.mean(diff) / sd)
    return dz



def mean_diff_ci_t(
    diff: np.ndarray, alpha: float = 0.05, alternative: Literal["two-sided","greater","less"] = "two-sided"
) -> Tuple[np.float64, Tuple[np.float64, np.float64]]:
    """
    使用 t 分布给出均值差及其置信区间（配对设计）。
    对称（双侧）给出 (low, high)；单侧给出半无限区间。
    """
    n = diff.size
    if n < 2:
        return np.nan, (np.nan, np.nan)
    mean_d = np.float64(np.mean(diff))
    s = np.std(diff, ddof=1)
    se = s / np.sqrt(n)
    df = n - 1
    if alternative == "two-sided":
        tcrit = stats.t.ppf(1 - alpha/2, df)
        low = mean_d - tcrit*se
        high = mean_d + tcrit*se
    elif alternative == "greater":
        tcrit = stats.t.ppf(1 - alpha, df)
        low = mean_d - tcrit*se
        high = np.nan
    else:  # 'less'
        tcrit = stats.t.ppf(1 - alpha, df)
        low = np.nan
        high = mean_d + tcrit*se

    return mean_d, (low, high)  # 返回均值差及置信区间



def hodges_lehmann(diff: np.ndarray) -> np.float64:
    """
    Hodges–Lehmann 位移估计：配对差值的中位数（快速近似）。
    更严格可用所有成对中位数的中位数；这里用 median(diff) 实用且足够。
    """
    return np.float64(np.median(diff))


# =========================
# 单个检验包装
# =========================
def paired_t_test(
    arr1: np.ndarray, arr2: np.ndarray, *, alternative: Literal["two-sided","greater","less"]="two-sided"
) -> Tuple[np.float64, np.float64]:
    """
    配对 t 检验包装（与 scipy.stats.ttest_rel 一致）。
    --------------------------------------------------
    适用范围：
        - 数据服从近似正态分布；
        - 两组样本为一一对应（配对样本）；
        - 差值方差齐性；
    推荐用途：
        - 作为 Wilcoxon 或非参数检验的对照；
        - 样本量较大时（>30）近似正态也可使用。
    """
    arr1, arr2 = to_clean_paired_arrays(arr1, arr2)
    if arr1.size < 2:
        return np.nan, np.nan
    t_stat, p = stats.ttest_rel(arr1, arr2, alternative=alternative)
    return np.float64(t_stat), np.float64(p)



def wilcoxon_signed_rank_test(
    arr1: np.ndarray, arr2: np.ndarray, *, alternative: Literal["two-sided","greater","less"]="two-sided"
) -> Tuple[np.float64, np.float64, str]:
    """
    Wilcoxon 符号秩检验包装。优先用 zero_method='wilcox'；失败则回退到 'pratt'。
    --------------------------------------------------
    适用范围：
        - 配对样本；
        - 差值分布近似对称但不要求正态；
        - 存在异常值或小样本；
    推荐用途：
        - 气候模式、有湖 vs 无湖对比；
        - 样本非正态时的主力检验方法。
    返回 (W_stat, p_value, zero_method_used)。
    """
    arr1, arr2 = to_clean_paired_arrays(arr1, arr2)
    if arr1.size < 2:
        return np.nan, np.nan, ""
    try:
        w_stat, p = stats.wilcoxon(arr1, arr2, alternative=alternative, zero_method="wilcox")
        return np.float64(w_stat), np.float64(p), "wilcox"
    except ValueError:
        w_stat, p = stats.wilcoxon(arr1, arr2, alternative=alternative, zero_method="pratt")
        return np.float64(w_stat), np.float64(p), "pratt"



def paired_permutation_test(
    arr1: np.ndarray,
    arr2: np.ndarray,
    *,
    n_sample: int = 10000,
    alternative: Literal["two-sided","greater","less"] = "two-sided",
    random_state: Optional[int] = None,
    chunk_size: int = 20000,       # 分块大小（按内存情况可调）
) -> Tuple[np.float64, np.float64]:
    """
    配对置换检验（符号翻转）— 向量化 + 分块。
    --------------------------------------------------
    适用范围：
        - 配对样本；
        - 无任何分布假设；
        - 样本量适中（n ≤ 数千）；
    推荐用途：
        - 最严格的非参数显著性检验；
        - 对极端事件差异的精确 p 值估计；
    """
    rng = np.random.default_rng(random_state)
    arr1, arr2 = to_clean_paired_arrays(arr1, arr2)
    if arr1.size < 2:
        return np.nan, np.nan
    d = arr1 - arr2
    n = d.size
    obs = float(np.mean(d))

    # 统计“更极端”的次数（分块累加）
    count = 0
    remaining = n_sample
    while remaining > 0:
        b = min(chunk_size, remaining)
        # 生成 [-1, 1] 的随机符号矩阵 shape=(n, b)
        signs = rng.choice((-1.0, 1.0), size=(n, b))
        # 所有置换统计量：每列是一次置换后的均值差
        stats_perm = (d[:, None] * signs).mean(axis=0)

        if alternative == "two-sided":
            count += int(np.sum(np.abs(stats_perm) >= abs(obs)))
        elif alternative == "greater":
            count += int(np.sum(stats_perm >= obs))
        else:  # "less"
            count += int(np.sum(stats_perm <= obs))

        remaining -= b

    p = (count + 1) / (n_sample + 1)  # 常用无偏估计
    return np.float64(obs), np.float64(p)



def paired_bootstrap(
    arr1: np.ndarray,
    arr2: np.ndarray,
    *,
    n_sample: int = 10000,
    ci: float = 0.95,
    alternative: Literal["two-sided","greater","less"] = "two-sided",
    center_null: bool = True,
    random_state: Optional[int] = None,
    chunk_size: int = 1e8,       # 分块大小（按内存情况可调）
) -> Tuple[np.float64, np.float64, np.float64, np.float64]:
    """
    配对 Bootstrap（索引自助）— 向量化 + 分块。
    - 若 center_null=True：
      * 用中心化差值 d0 估 p（近似）
      * 用原始差值 d 估 CI（同一批索引，避免重复跑一遍）
    --------------------------------------------------
    适用范围：
        - 配对样本；
        - 任意分布；
        - 需要置信区间；
    推荐用途：
        - 获取平均差值的 [CI_low, CI_high]；
        - 检验结果稳健性。
    """
    rng = np.random.default_rng(random_state)
    arr1, arr2 = to_clean_paired_arrays(arr1, arr2)
    if arr1.size < 2:
        return np.nan, np.nan, np.nan, np.nan
    d = arr1 - arr2
    n = d.size
    obs = float(np.mean(d))

    d0 = d - obs if center_null else d

    # 分块收集所有 bootstrap 统计量（用于 p 与 CI）
    stats_for_p = []   # 基于 d0（若 center_null）；否则基于 d
    stats_for_ci = []  # 基于 d（CI 总是基于原始差值）

    remaining = n_sample
    while remaining > 0:
        b = min(chunk_size, remaining)
        idx = rng.integers(0, n, size=(b, n))  # 每行一组索引
        # 计算该块的统计量
        block_ci = d[idx].mean(axis=1)         # 原始 d 的均值差（CI 用）
        if center_null:
            block_p  = d0[idx].mean(axis=1)    # 中心化 d0 的均值差（p 值用）
        else:
            block_p  = block_ci                 # 不中心化：p 和 CI 同分布

        stats_for_ci.append(block_ci)
        stats_for_p.append(block_p)
        remaining -= b

    stats_for_ci = np.concatenate(stats_for_ci, axis=0)
    stats_for_p  = np.concatenate(stats_for_p, axis=0)

    # CI（百分位法）
    alpha = 1 - ci
    low, high = np.quantile(stats_for_ci, [alpha/2, 1 - alpha/2])

    # 近似 p 值
    if alternative == "two-sided":
        p_boot = (np.sum(np.abs(stats_for_p) >= abs(obs)) + 1) / (n_sample + 1)
    elif alternative == "greater":
        p_boot = (np.sum(stats_for_p >= obs) + 1) / (n_sample + 1)
    else:
        p_boot = (np.sum(stats_for_p <= obs) + 1) / (n_sample + 1)

    return np.float64(obs), np.float64(p_boot), np.float64(low), np.float64(high)



def sign_test(
    arr1: np.ndarray,
    arr2: np.ndarray,
    *,
    alternative: Literal["two-sided","greater","less"]="two-sided"
) -> Tuple[np.float64, np.float64]:
    """
    符号检验（Sign Test）
    --------------------------------------------------
    适用范围：
        - 配对样本；
        - 极度偏态或存在极端异常值；
        - 仅关注方向性（不关心幅度）；
    推荐用途：
        - 检查“有湖 > 无湖”的一致性；
        - 小样本或异常数据的最稳健检验；
    """
    arr1, arr2 = to_clean_paired_arrays(arr1, arr2)
    d = arr1 - arr2
    n_pos = np.sum(d > 0)
    n_neg = np.sum(d < 0)
    n = n_pos + n_neg
    if n == 0:
        return np.nan, np.nan
    p = stats.binom_test(min(n_pos, n_neg), n, 0.5, alternative=alternative)
    z = (n_pos - n_neg) / np.sqrt(n)
    return np.float64(z), np.float64(p)



def cliffs_delta(
    arr1: np.ndarray,
    arr2: np.ndarray
) -> np.float64:
    """
    Cliff's Delta 效应量检验（Effect size, 非参数）
    --------------------------------------------------
    适用范围：
        - 任意分布；
        - 样本独立或配对均可；
    推荐用途：
        - 描述“湖泊效应”的强度与方向；
        - 可配合显著性检验共同报告。
    阈值解释：
        |δ| < 0.147 → 弱效应；
        0.147 ≤ |δ| < 0.33 → 中等；
        |δ| ≥ 0.33 → 强效应；
    """
    arr1, arr2 = to_clean_paired_arrays(arr1, arr2)
    n1, n2 = len(arr1), len(arr2)
    count = 0
    for x in arr1:
        count += np.sum(x > arr2) - np.sum(x < arr2)
    delta = count / (n1 * n2)
    return np.float64(delta)



def cramer_von_mises_test(
    arr1: np.ndarray,
    arr2: np.ndarray
) -> Tuple[np.float64, np.float64]:
    """
    Cramér–von Mises 两样本检验（分布形状差异检验）
    --------------------------------------------------
    适用范围：
        - 两个独立样本；
        - 需要比较完整分布（不仅均值）；
    推荐用途：
        - 检查“有湖 vs 无湖”在极端事件分布形状上的差异；
        - 比 KS 检验对尾部分布差异更敏感。
    """
    arr1, arr2 = to_clean_paired_arrays(arr1, arr2)
    res = stats.cramervonmises_2samp(arr1, arr2)
    return np.float64(res.statistic), np.float64(res.pvalue)



def mann_whitney_u_test(
    arr1: np.ndarray,
    arr2: np.ndarray,
    *,
    alternative: Literal["two-sided","greater","less"]="two-sided"
) -> Tuple[np.float64, np.float64]:
    """
    Mann–Whitney U 检验（独立样本 Wilcoxon 秩和检验）
    --------------------------------------------------
    适用范围：
        - 两个独立样本；
        - 数据非正态、方差不齐；
        - 样本量中等（n1,n2≥10）；
    推荐用途：
        - 比较“湖区 vs 非湖区”的极端事件强度或频次；
        - 不要求配对关系。
    """
    arr1, arr2 = to_clean_paired_arrays(arr1, arr2)
    if arr1.size < 1 or arr2.size < 1:
        return np.nan, np.nan
    u_stat, p = stats.mannwhitneyu(arr1, arr2, alternative=alternative)
    return np.float64(u_stat), np.float64(p)


# =========================
# 选择逻辑（是否用配对 t）
# =========================
def should_use_paired_t(
    diff: np.ndarray, *, normality_p: float, clt_n: int = 25, alpha_normality: float = 0.05
) -> Tuple[bool, Dict[str, Any]]:  # type: ignore[name-defined]
    """
    根据差值正态性与样本量决定是否使用配对 t。
    规则：
      - 若 n >= clt_n → True（中心极限定理）
      - 否则若 Shapiro p >= alpha_normality → True
      - 其余 False（转 Wilcoxon）
    返回 (use_t, info)
    """
    n = diff.size
    info = {
        "n": int(n),
        "normality_p": float(normality_p),
        "used_clt": bool(n >= clt_n),
        "alpha_normality": float(alpha_normality),
        "clt_n": int(clt_n),
        "skew": float(stats.skew(diff, bias=False)) if n >= 3 else float("nan"),
        "iqr": float(np.subtract(*np.percentile(diff, [75,25]))) if n >= 4 else float("nan"),
    }
    # 规则 1：CLT
    if n >= clt_n:
        return True, info
    # 规则 2：正态性宽松通过
    if not np.isnan(normality_p) and normality_p >= alpha_normality:
        return True, info
    # 否则不用 t（回退到你代码里的 permutation/bootstrap/Wilcoxon）
    return False, info

# =========================
# 统一的结果数据结构
# =========================
@dataclass
class TestResult:
    n_eff: int      # 有效样本量
    checkmethod: Literal["Paired_t-test","Wilcoxon_signed-rank_test","degenerate",
                         "Paired_permutation_test", "Paired_bootstrap", "Sign_test",
                         "Cliffs_delta", "Mann-Whitney_U_test", "Cramer-von_Mises_test", "N/A"]
    statistic: float  # 检验统计量
    p: float          # p 值
    effect_size_name: Literal["cohens_dz","hodges_lehmann"] # 效应量名称
    effect_size: float # 效应量
    mean_diff: Optional[float] = None # 均值差
    ci: Optional[Tuple[float, float]] = None # 置信区间
    diagnostics: Optional[Dict[str, Any]] = None # 诊断信息（正态性 p 值、样本量等）
    notes: Optional[Sequence[str]] = None #备注（如“用 CLT 放宽正态性要求”）

# =========================
# 总控函数：自动选择并检验
# =========================
def paired_test_auto(
    arr1: np.ndarray,
    arr2: np.ndarray,
    *,
    checkmethod: Literal[
        "auto",
        "Paired_t-test",
        "Wilcoxon_signed-rank_test",
        "Paired_permutation_test",
        "Paired_bootstrap",
        "Sign_test",
        "Cliffs_delta",
        "Mann-Whitney_U_test",
        "Cramer-von_Mises_test",
    ] = "auto",
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    alpha_ci: float = 0.05,
    clt_n: int = 30,
    # 仅在 permutation/bootstrap 时使用：
    n_sample: int = 10000,
    ci: float = 0.95,
    center_null: bool = True,
    random_state: Optional[int] = None,
    checkflag: bool = False,
) -> TestResult:
    """
    配对检验总控（带 checkmethod 选择）：
      - auto: 先验自动选择（CLT 或差值正态 → 配对 t；否则 Wilcoxon）
      - Paired_t-test: 强制配对 t
      - Wilcoxon_signed-rank test: 强制 Wilcoxon
      - Paired_permutation_test: 强制置换检验（符号翻转）
      - Paired_bootstrap: 强制 Bootstrap（返回 CI 与近似 p）

    统一返回 TestResult：
      - statistic: t/W/Wilcoxon 的统计量或（permutation/bootstrap 的）均值差
      - p: 各方法对应的 p 值（bootstrap 为近似 p）
      - effect_size: 一律返回 Cohen's dz（基于配对差值）
      - mean_diff: 均值差（方便直观解读）
      - ci: 仅 t 检验与 bootstrap 填写（bootstrap 使用百分位 CI）
    """
    # 清洗与基本量
    arr1, arr2 = to_clean_paired_arrays(arr1, arr2)
    if arr1.size < 2:
        return TestResult(
            n_eff=0, checkmethod="N/A", statistic=np.nan, p=np.nan,
            effect_size_name="cohens_dz", effect_size=np.nan,
            mean_diff=np.nan, ci=(np.nan, np.nan),
            diagnostics={"sd_diff": np.nan, "normality_p": np.nan, "used_clt": False},
            notes=["有效配对样本量 < 2，无法进行检验。"]
        )
    diff = arr1 - arr2
    n = diff.size

    # 退化情形处理
    degenerate, const = is_degenerate(diff)
    if degenerate:
        if np.isclose(const, 0.0):
            return TestResult(
                n_eff=n, checkmethod="degenerate", statistic=0.0, p=1.0,
                effect_size_name="cohens_dz", effect_size=0.0,
                mean_diff=0.0, ci=(0.0, 0.0),
                diagnostics={"sd_diff": 0.0, "normality_p": float("nan"), "used_clt": False},
                notes=["所有配对差值均为 0，显著性检验不适用。"]
            )
        else:
            return TestResult(
                n_eff=n, checkmethod="degenerate", statistic=float("inf"), p=0.0,
                effect_size_name="cohens_dz", effect_size=float("inf"),
                mean_diff=float(const), ci=(float(const), float(const)),
                diagnostics={"sd_diff": 0.0, "normality_p": float("nan"), "used_clt": False},
                notes=["所有配对差值都为同一常数且非 0，差异显著（无需检验）。"]
            )

    # 诊断信息（供 auto / 统一返回）
    p_norm = shapiro_p(diff)
    sdd = sd_diff(diff)
    dz = cohens_dz(diff)

    # --- 分派逻辑 ---
    chosen = checkmethod

    if checkmethod == "auto":
        use_t, sel_info = should_use_paired_t(diff, normality_p=p_norm, clt_n=clt_n)
        chosen = "Paired_t-test" if use_t else "Paired_bootstrap"
        diag = {"sd_diff": sdd, **sel_info}
        note = f"选择 t 检验的依据：{'CLT' if sel_info['used_clt'] else 'Shapiro p >= α'}" if use_t \
               else "差值正态性不足且样本量未达 CLT 阈值，使用 Wilcoxon。"
    else:
        # 非 auto 模式下，也给基本诊断
        diag = {
            "sd_diff": sdd,
            "n": int(n),
            "normality_p": float(p_norm),
            "used_clt": False,
            "alpha_normality": 0.05,
            "clt_n": int(clt_n),
        }
        note = f"checkmethod='{checkmethod}'（强制选择）。"
    if checkflag:
        print(arr1, arr2, diff, p_norm, note)
        
    # --- 执行相应方法 ---
    if chosen == "Paired_t-test":
        t_stat, pval = paired_t_test(arr1, arr2, alternative=alternative)
        md, ci_t = mean_diff_ci_t(diff, alpha=alpha_ci, alternative=alternative)
        return TestResult(
            n_eff=n, checkmethod="Paired_t-test", statistic=float(t_stat), p=float(pval),
            effect_size_name="cohens_dz", effect_size=float(dz),
            mean_diff=float(md), ci=(float(ci_t[0]), float(ci_t[1])),
            diagnostics=diag, notes=[note]
        )

    elif chosen == "Wilcoxon_signed-rank_test":
        w_stat, pval, zero_method = wilcoxon_signed_rank_test(arr1, arr2, alternative=alternative)
        return TestResult(
            n_eff=n, checkmethod="Wilcoxon_signed-rank_test", statistic=float(w_stat), p=float(pval),
            effect_size_name="cohens_dz", effect_size=float(dz),  # 统一用 dz，直观对比
            mean_diff=float(np.mean(diff)), ci=(np.nan, np.nan),  # Wilcoxon 无 CI
            diagnostics={**diag, "wilcoxon_zero_method": zero_method},
            notes=[note]
        )

    elif chosen == "Paired_permutation_test":
        obs, p_perm = paired_permutation_test(
            arr1, arr2, n_sample=n_sample, alternative=alternative, random_state=random_state
        )
        return TestResult(
            n_eff=n, checkmethod="Paired_permutation_test",  
            statistic=float(obs), p=float(p_perm),
            effect_size_name="cohens_dz", effect_size=float(dz),
            mean_diff=float(np.mean(diff)), ci=(np.nan, np.nan),  # Permutation 无 CI
            diagnostics={**diag, "n_sample": int(n_sample)},
            notes=[f"permutation_test_mean_diff（符号翻转），alternative='{alternative}'。"]
        )

    elif chosen == "Paired_bootstrap":
        obs, p_boot, low, high = paired_bootstrap(
            arr1, arr2, n_sample=n_sample, ci=ci, alternative=alternative,
            center_null=center_null, random_state=random_state
        )
        return TestResult(
            n_eff=n, checkmethod="Paired_bootstrap",  
            statistic=float(obs), p=float(p_boot),
            effect_size_name="cohens_dz", effect_size=float(dz),
            mean_diff=float(np.mean(diff)), ci=(float(low), float(high)),
            diagnostics={**diag, "n_sample": int(n_sample), "ci": float(ci), "center_null": bool(center_null)},
            notes=[f"bootstrap test（百分位CI，近似 p），alternative='{alternative}'。"]
        )
    
    elif chosen == "Sign_test":
        z_stat, p_sign = sign_test(arr1, arr2, alternative=alternative)
        hl = hodges_lehmann(diff)
        return TestResult(
            n_eff=n, checkmethod="Sign_test", statistic=float(z_stat), p=float(p_sign),
            effect_size_name="hodges_lehmann", effect_size=float(hl),
            mean_diff=float(np.mean(diff)), ci=(np.nan, np.nan),  # Sign test 无 CI
            diagnostics=diag,
            notes=[note + " 使用 Hodges–Lehmann 位移估计作为效应量。"]
        )
    
    elif chosen == "Cliffs_delta":
        delta = cliffs_delta(arr1, arr2)
        return TestResult(
            n_eff=n, checkmethod="Cliffs_delta", statistic=float(delta), p=np.nan,
            effect_size_name="cliffs_delta", effect_size=float(delta),
            mean_diff=float(np.mean(diff)), ci=(np.nan, np.nan),  # Cliff's Delta 无 CI
            diagnostics=diag,
            notes=[note + " Cliff's Delta 无对应显著性检验，p 值设为 NaN。"]
        )
    
    elif chosen == "Mann-Whitney_U_test":
        u_stat, p_mw = mann_whitney_u_test(arr1, arr2, alternative=alternative)
        hl = hodges_lehmann(diff)
        return TestResult(
            n_eff=n, checkmethod="Mann-Whitney_U_test", statistic=float(u_stat), p=float(p_mw),
            effect_size_name="hodges_lehmann", effect_size=float(hl),
            mean_diff=float(np.mean(diff)), ci=(np.nan, np.nan),  # Mann-Whitney U 无 CI
            diagnostics=diag,
            notes=[note + " 使用 Hodges–Lehmann 位移估计作为效应量。"]
        )
    
    elif chosen == "Cramer-von_Mises_test":
        cvm_stat, p_cvm = cramer_von_mises_test(arr1, arr2)
        hl = hodges_lehmann(diff)
        return TestResult(
            n_eff=n, checkmethod="Cramer-von_Mises_test", statistic=float(cvm_stat), p=float(p_cvm),
            effect_size_name="hodges_lehmann", effect_size=float(hl),
            mean_diff=float(np.mean(diff)), ci=(np.nan, np.nan),  # Cramer-von Mises 无 CI
            diagnostics=diag,
            notes=[note + " 使用 Hodges–Lehmann 位移估计作为效应量。"]
        )

    else:
        raise ValueError(f"未知 checkmethod: {chosen}")



def _get_method_num(checkmethod: str) -> int:
    """将 checkmethod 字符串转换为数字编码（便于排序）"""
    if checkmethod == "Paired_t-test":
        return 1
    elif checkmethod == "Wilcoxon_signed-rank_test":
        return 2
    elif checkmethod == "Paired_permutation_test":
        return 3
    elif checkmethod == "Paired_bootstrap":
        return 4
    elif checkmethod == "Sign_test":
        return 5
    elif checkmethod == "Cliffs_delta":
        return 6
    elif checkmethod == "Mann-Whitney_U_test":
        return 7
    elif checkmethod == "Cramer-von_Mises_test":
        return 8
    elif checkmethod == "degenerate":
        return 0
    elif checkmethod == "N/A":
        return -1
    else:
        raise ValueError(f"wrong checkmethod: {checkmethod}")



def _grid_test_worker(
        i: int,
        arr1_2d: np.ndarray,
        arr2_2d: np.ndarray,
        *,
        checkmethod: Literal[
            "auto",
            "Paired_t-test",
            "Wilcoxon_signed-rank_test",
            "Paired_permutation_test",
            "Paired_bootstrap",
            "Sign_test",
            "Cliffs_delta",
            "Mann-Whitney_U_test",
            "Cramer-von_Mises_test",
        ],
        alternative: Literal["two-sided", "greater", "less"],
        alpha_ci: float,
        clt_n: int,
        n_sample: int,
        ci: float,
        center_null: bool,
        seed_i: Optional[int],
        checkflag:bool=False
    ):
    """对子网格 i 执行一次检验，返回该网格的 p 值（必要时你也可以返回更多字段）"""
    x = arr1_2d[:, i]
    y = arr2_2d[:, i]
    # 处理全 NaN 或有效样本过少的情况（paired_test_auto 内部也会校验，这里再兜一层）
    if not np.isfinite(x).any() or not np.isfinite(y).any():
        rtn_p = np.nan; rtn_mean_diff = np.nan; rtn_effect_size = np.nan; rtn_method = -1
    res = paired_test_auto(
        x, y,
        checkmethod=checkmethod,
        alternative=alternative,
        alpha_ci=alpha_ci,
        clt_n=clt_n,
        n_sample=n_sample,
        ci=ci,
        center_null=center_null,
        random_state=seed_i,
        checkflag=checkflag,
    )
    rtn_p = float(res.p)
    rtn_mean_diff = float(res.mean_diff) 
    rtn_effect_size = float(res.effect_size)
    rtn_method = _get_method_num(res.checkmethod)  # 将 checkmethod 转为数字编码
    rtn_info = res.notes
    rtn_ci_low = res.ci[0]
    rtn_ci_high = res.ci[1]
    # 单个网格失败不影响总体；该点置 NaN
    # rtn_p = np.nan; rtn_mean_diff = np.nan; rtn_effect_size = np.nan; rtn_method = -1

    return rtn_p, rtn_mean_diff, rtn_effect_size, rtn_method, rtn_info, rtn_ci_low, rtn_ci_high



def SignificanceTest(
    arr1: np.ndarray,
    arr2: np.ndarray,
    *,
    info: str = "",
    checkmethod: Literal[
        "auto",
        "Paired_t-test",
        "Wilcoxon_signed-rank_test",
        "Paired_permutation_test",
        "Paired_bootstrap",
        "Sign_test",
        "Cliffs_delta",
        "Mann-Whitney_U_test",
        "Cramer-von_Mises_test",
    ] = "auto",
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    alpha_ci: float = 0.05,
    clt_n: int = 30,
    n_sample: int = 10000,
    ci: float = 0.95,
    center_null: bool = True,
    random_state: Optional[int] = None,
    n_jobs: Optional[int] = None,
    checkflag: bool = False
) -> np.ndarray:
    """
    对两个 3D 数组 (time, lat, lon) 逐格做配对检验，返回 p 值矩阵 (lat, lon)。
    - checkmethod 可选：auto / Paired_t-test / Wilcoxon_signed-rank test / Paired_permutation_test / Paired_bootstrap
    - n_sample 同时作为 permutation 的 n_perm 与 bootstrap 的 n_boot
    - 并行：joblib loky 后端（多进程），自动 memmap 共享大数组
    """
    batch_size = 16
    if arr1.shape != arr2.shape or arr1.ndim != 3:
        raise ValueError("arr1/arr2 必须同形，且形状为 (time, lat, lon)。")

    ntime, nlat, nlon = arr1.shape

    if checkflag:
        print(f"SignificanceTest: arr1.shape={arr1.shape}, arr2.shape={arr2.shape}, ntime={ntime}, nlat={nlat}, nlon={nlon}")

    # 展平到 (time, ngrid)，便于按列分配任务
    arr1_2d = arr1.reshape(ntime, nlat * nlon)
    arr2_2d = arr2.reshape(ntime, nlat * nlon)

    # 为每个网格准备不同的随机种子（可复现）
    if random_state is None:
        seeds = [None] * (nlat * nlon)
    else:
        # 简单可复现的派生：seed_i = base + i
        base = int(random_state)
        seeds = [base + i for i in range(nlat * nlon)]
        # seeds = [base for i in range(nlat * nlon)]

    # 并发数
    if n_jobs is None:
        # I/O+CPU 混合场景下，4~8 往往更稳；这里给个保守默认
        n_jobs = max(1, os.cpu_count() - 2)

    # 并行执行；max_nbytes 维持默认启用 memmap（数组足够大时会自动共享）
    ntasks = nlat * nlon
    with Parallel(n_jobs=n_jobs, backend="loky", batch_size=batch_size, pre_dispatch="2*n_jobs", return_as="generator") as parallel:
        gen = parallel(
            delayed(_grid_test_worker)(
                i, arr1_2d, arr2_2d,
                checkmethod=checkmethod,
                alternative=alternative,
                alpha_ci=alpha_ci,
                clt_n=clt_n,
                n_sample=n_sample,
                ci=ci,
                center_null=center_null,
                seed_i=seeds[i],
                checkflag=checkflag
            )
            for i in range(ntasks)
        )
        res_list = [p for p in tqdm(gen, total=ntasks,
                                  desc=f"    ➠ {info}", unit="grid",
                                  dynamic_ncols=True, mininterval=0.2, leave=False)]
    # p 值
    p_list = [res[0] for res in res_list]
    p_arr = np.asarray(p_list, dtype=float).reshape(nlat, nlon)
    # 均值差
    mean_diff_list = [res[1] for res in res_list]
    mean_diff_arr = np.asarray(mean_diff_list, dtype=float).reshape(nlat, nlon)
    # 效应量
    effect_size_list = [res[2] for res in res_list]
    effect_size_arr = np.asarray(effect_size_list, dtype=float).reshape(nlat, nlon)
    # 方法统计
    method_list = [res[3] for res in res_list]
    method_arr = np.asarray(method_list, dtype=int).reshape(nlat, nlon)
    # 置信区间
    ci_low_list = [res[5] for res in res_list]
    ci_low_arr = np.asarray(ci_low_list, dtype=float).reshape(nlat, nlon)
    ci_high_list = [res[6] for res in res_list]
    ci_high_arr = np.asarray(ci_high_list, dtype=float).reshape(nlat, nlon)
    time.sleep(1)
    return p_arr, mean_diff_arr, effect_size_arr, method_arr, ci_low_arr, ci_high_arr



def RelativeContribution(arr1: np.array, arr2: np.array, time_axis: int = 0) -> Tuple[np.array, np.array]:
    """
    计算两种贡献率:
    1) 总体贡献率: (X_lake - X_nolake) / X_nolake * 100
    2) 距平贡献率: (X_lake - X_nolake) / (X_nolake - climatology) * 100
    参数:
        arr1 : np.array  有湖实验结果 (X_lake)
        arr2 : np.array  无湖实验结果 (X_nolake)
        time_axis : int  时间维度 (默认 0)
    返回:
        dict 包含:
            - 'overall': 总体贡献率
            - 'anomaly': 距平贡献率
    """
    arr1 = arr1.squeeze()
    arr2 = arr2.squeeze()
    # 差值 (湖泊效应)
    diff = arr1 - arr2
    meandiff = np.nanmean(diff, axis=time_axis)
    # ---- (1) 总体贡献率 ----
    mean_arr2 = np.nanmean(arr2, axis=time_axis)
    overall = (meandiff / mean_arr2) * 100
    # ---- (2) 距平贡献率 ----
    arr2_clim = np.nanmean(arr2, axis=time_axis, keepdims=True)
    anomaly_den = arr2 - arr2_clim
    # 避免除以零
    with np.errstate(divide='ignore', invalid='ignore'):
        anomaly = np.where(np.abs(anomaly_den) > 1e-6, diff / anomaly_den * 100, np.nan)
        anomaly = np.nanmean(anomaly, axis=time_axis)
    return overall, anomaly


