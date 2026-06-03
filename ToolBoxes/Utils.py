import heapq
import calendar
import numpy as np
import pandas as pd
import xarray as xr
from typing import Tuple
from collections import deque
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter, binary_opening, binary_closing, binary_fill_holes, label, generate_binary_structure


# ---------- base helpers ----------
def _to_array(arr1: object) -> np.ndarray:
    """将输入转为 numpy.ndarray（零拷贝/视图优先）。"""
    return np.asarray(arr1)



def _finite_mask(arr1: object, arr2: object) -> np.ndarray:
    """两数组对应位置均为有限值（非 NaN/Inf）的布尔掩码。"""
    arr1 = _to_array(arr1); arr2 = _to_array(arr2)
    return np.isfinite(arr1) & np.isfinite(arr2)



def movavg(arr: np.ndarray, k: int, axis: int = 0) -> np.ndarray:
    """多维简单滑动平均（端点采用 edge-padding）。
    参数
    ----
    arr : ndarray
        输入数组。
    k : int
        窗口长度（若为偶数，自动 +1 以保持对称）。k<=1 时直接返回原数组。
    axis : int
        平滑所沿的轴。
    """
    if k is None or k <= 1:
        return arr
    if k % 2 == 0:
        k += 1
    pad = k // 2
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (pad, pad)
    p_pad = np.pad(arr, pad_width, mode="edge")
    kernel = np.ones((k,)) / k
    return np.apply_along_axis(lambda x: np.convolve(x, kernel, mode="valid"), axis, p_pad)



def to_year_month_hour(ds: xr.Dataset, how: str = "mean") -> xr.Dataset:
    """
    how: "mean" | "sum" | "max" | "min"
    """
    t = ds["time"]
    ymh = pd.MultiIndex.from_arrays(
        [t.dt.year.values, t.dt.month.values, t.dt.hour.values],
        names=["year","month","hour"]
    )
    gb = ds.assign_coords(ymh=("time", ymh)).groupby("ymh")
    if   how == "mean": out = gb.mean("time", skipna=True)
    elif how == "sum":  out = gb.sum("time",  skipna=True)
    elif how == "max":  out = gb.max("time",  skipna=True)
    elif how == "min":  out = gb.min("time",  skipna=True)
    else: raise ValueError("how must be one of: mean/sum/max/min")
    out = out.unstack("ymh")  # → (year, month, hour, lat, lon)
    order = ["year","month","hour"] + [d for d in out.dims if d not in ("year","month","hour")]
    return out.transpose(*order)



def _model_func(name):
    if name == "exp":
        return lambda x, a, b, c: a * np.exp(-b * x) + c
    if name == "power":
        return lambda x, a, b, c: a * np.power(x, -b) + c
    if name == "linear":
        return lambda x, a, b: a * x + b
    raise ValueError(f"Unknown model name: {name}")



def _to_radius(series, rmin):
    return (series.astype(float) - rmin).to_numpy()



def _reshape_timeseries_to_doyyears(in_df: pd.DataFrame, value: str) -> pd.DataFrame:
    """将时间序列数据重塑为 DOY-Years 格式（年-日）"""
    in_df["time"] = pd.to_datetime(in_df["time"])
    in_df["year"] = in_df["time"].dt.year
    in_df["doy"] = in_df["time"].dt.dayofyear
    result = in_df.pivot(index="doy", columns="year", values=value)
    result.columns = [f"year{idx+1}" for idx in range(len(result.columns))]
    return result


def xarray_leap_to_noleap(xarr: xr.Dataset) -> xr.Dataset:
    """去掉闰年的2月29日，转为无闰年数据"""
    mask = ~((xarr.time.dt.month == 2) & (xarr.time.dt.day == 29))
    xarr_no_leap = xarr.sel(time=mask)
    return xarr_no_leap


def scaled_kde(v,
               bw_adjust: float = 1.5,
               gridsize: int = 1024,
               cut: float = 3.0,
               mode: str = 'peak') -> Tuple[np.ndarray, np.ndarray]:
    """
    鲁棒的一维 KDE 计算与缩放。

    Parameters
    ----------
    v : array-like
        一维数据（将被 ravel+过滤非有限值）。
    bw_adjust : float, default 1.5
        在 Scott 规则带宽上的倍率（与 seaborn 行为一致）。
    gridsize : int, default 1024
        评估网格点个数（>=32）。
    cut : float, default 3.0
        在数据端点外按带宽扩展的倍数（>=0）。
    mode : {'peak','sigma','none'}, default 'peak'
        'peak'  : y 归一到峰值为 1；
        'sigma' : y 乘以样本尺度（优先 std，退化为 IQR/1.349）；
        'none'  : 不额外缩放（KDE 本身为概率密度，积分为 1）。

    Returns
    -------
    xs : ndarray, shape (gridsize,)
    y  : ndarray, shape (gridsize,)
    """
    # ---------- 输入与参数校验 ----------
    v = np.asarray(v).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ValueError("scaled_kde: 输入没有任何有限数值。")

    try:
        bw_adjust = float(bw_adjust)
    except Exception:
        raise ValueError("scaled_kde: bw_adjust 必须是可转为浮点的数。")
    if not np.isfinite(bw_adjust) or bw_adjust <= 0:
        raise ValueError("scaled_kde: bw_adjust 必须是正数。")

    try:
        gridsize = int(gridsize)
    except Exception:
        raise ValueError("scaled_kde: gridsize 必须是整数。")
    if gridsize < 32:
        raise ValueError("scaled_kde: gridsize 应 >= 32。")

    try:
        cut = float(cut)
    except Exception:
        raise ValueError("scaled_kde: cut 必须是可转为浮点的数。")
    if cut < 0:
        raise ValueError("scaled_kde: cut 必须 >= 0。")

    mode = str(mode).lower()
    if mode not in {"peak", "sigma", "none"}:
        raise ValueError("scaled_kde: mode 只能为 {'peak','sigma','none'}。")

    n = v.size
    v_min, v_max = float(np.min(v)), float(np.max(v))
    data_range = v_max - v_min

    # ---------- 稳健尺度 ----------
    # 首选样本标准差；为 0/非有限时，退化为 IQR/1.349（≈std 的无偏近似），再退化为极小正数
    def _robust_sigma(x: np.ndarray) -> float:
        s = np.std(x, ddof=1) if x.size >= 2 else 0.0
        if not np.isfinite(s) or s <= 0:
            q1, q3 = np.quantile(x, [0.25, 0.75])
            iqr = float(q3 - q1)
            s = iqr / 1.349 if iqr > 0 else 0.0
        if not np.isfinite(s) or s <= 0:
            # 与数据尺度相关的极小正数，避免全常数时完全退化
            s = max(np.finfo(float).eps, 1e-12 * max(1.0, abs(v_min), abs(v_max)))
        return float(s)

    sigma = _robust_sigma(v)

    # Scott 规则（1D：n**(-1/5)）× 调整倍数，作为我们同时用于：
    # 1) KDE 带宽倍率（通过 set_bandwidth），2) 外扩距离尺度
    h = (n ** (-1.0 / 5.0)) * sigma * bw_adjust

    # ---------- x 轴范围 ----------
    if data_range <= 0:  # 全常数
        lo, hi = v_min - cut * 3.0 * h, v_max + cut * 3.0 * h
    else:
        lo, hi = v_min - cut * h, v_max + cut * h
    if not (np.isfinite(lo) and np.isfinite(hi)) or not (hi > lo):
        # 极端情况下兜底
        span = max(data_range, 6.0 * h, 1.0)
        mid = 0.5 * (v_min + v_max)
        lo, hi = mid - 0.5 * span, mid + 0.5 * span

    xs = np.linspace(lo, hi, gridsize)

    # ---------- KDE 计算（带稳健 fallback） ----------
    use_kde = (n >= 2) and (data_range > 0)
    y = None
    if use_kde:
        try:
            kde = gaussian_kde(v, bw_method='scott')
            # 与 seaborn 一致：在 Scott 规则上乘以 bw_adjust
            kde.set_bandwidth(kde.factor * bw_adjust)
            y = kde(xs)
        except Exception as e:
            warnings.warn(f"scaled_kde: KDE 失败（{e}），将退化为高斯核近似。")
            use_kde = False
    if not use_kde:
        # 用单峰高斯作为 delta 的光滑近似
        mu = float(np.mean(v))
        bw = max(h, np.finfo(float).eps)
        norm = 1.0 / (np.sqrt(2.0 * np.pi) * bw)
        y = norm * np.exp(-0.5 * ((xs - mu) / bw) ** 2)
    # 数值噪声下可能出现微小负值
    y = np.maximum(y, 0.0)
    # ---------- 缩放 ----------
    if mode == 'peak':
        ymax = float(np.max(y))
        if ymax > 0:
            y = y / ymax
    elif mode == 'sigma':
        y = y * sigma
    # mode == 'none'：不处理

    return xs, y



def smooth_data(data_in: np.ndarray,
                sigma: float,
                min_area: float,
                cell_area: np.ndarray,
                opening_iter: int = 1,
                closing_iter: int = 1,
                connectivity: int = 2) -> np.ndarray:
    """
    data_in      : 输入掩码 (bool/0-1)，True/1 表示目标区域
    sigma        : 高斯平滑的 sigma（单位=网格）
    min_area     : 最小连通域面积阈值（与 cell_area 的单位一致：如 m² 或 km²）
    cell_area    : 每个网格的面积（与 data_in 同形状）
    opening_iter : binary_opening 迭代次数
    closing_iter : binary_closing 迭代次数
    connectivity : 1=4邻域, 2=8邻域（2 更连贯）
    返回值       : 0/1 浮点掩码
    """
    data = np.asarray(data_in, dtype=float)
    area = np.asarray(cell_area)
    if area.shape != data.shape:
        raise ValueError(f"cell_area.shape {area.shape} 必须与 data_in.shape {data.shape} 一致")

    # 1) 平滑并阈值化
    prob = gaussian_filter(data, sigma=sigma, mode='nearest')
    mask = prob >= 0.5

    # 2) 形态学清理
    if opening_iter > 0:
        mask = binary_opening(mask, iterations=opening_iter)
    if closing_iter > 0:
        mask = binary_closing(mask, iterations=closing_iter)
    mask = binary_fill_holes(mask)

    # 3) 连通域标记（8 邻域更稳）
    structure = generate_binary_structure(2, connectivity)
    lbl, nlab = label(mask, structure=structure)
    if nlab == 0:
        return np.zeros_like(mask, dtype=float)

    # 4) 面积加权统计并按 min_area 过滤
    #    NaN 或 非正面积视为 0 权重
    area_w = np.where(np.isfinite(area) & (area > 0), area, 0.0)
    areas = np.bincount(lbl.ravel(), weights=area_w.ravel(), minlength=lbl.max()+1)

    keep = areas >= float(min_area)
    keep[0] = False  # 0 是背景
    mask_clean = keep[lbl]

    return mask_clean.astype(float)



# ---------- cwrf functions ----------
def get_cwrf_grid_area(DX: float, DY: float, MAPFAC_MX: np.ndarray, MAPFAC_MY: np.ndarray) -> np.ndarray:
    """计算 CWRF 网格面积： area = (DX / MAPFAC_MX) * (DY / MAPFAC_MY)。"""
    MAPFAC_MX = _to_array(MAPFAC_MX)
    MAPFAC_MY = _to_array(MAPFAC_MY)
    area = (DX / MAPFAC_MX) * (DY / MAPFAC_MY)
    return area



def cal_area_weighted_sum(arr: np.ndarray, area: np.ndarray, lat_axis: int = -2, lon_axis: int = -1) -> np.ndarray:
    """面积加权求和（NaN 安全）。
    area 需为 2D，形状 (ny, nx)。支持在高维数组上指定纬向与经向轴进行加权并沿
    这两轴求和。
    """
    arr = _to_array(arr)
    area = _to_array(area)
    if area.ndim != 2:
        raise ValueError("`area` must be 2D with shape (ny, nx).")
    ndim = arr.ndim
    lat_axis = lat_axis % ndim
    lon_axis = lon_axis % ndim
    if lat_axis == lon_axis:
        raise ValueError("`lat_axis` and `lon_axis` must be different axes.")
    shape = [1] * ndim
    shape[lat_axis] = area.shape[0]
    shape[lon_axis] = area.shape[1]
    W = area.reshape(shape)
    valid = _finite_mask(arr, W)
    num = np.sum(np.where(valid, arr * W, 0.0), axis=(lat_axis, lon_axis))
    cnt = np.sum(valid, axis=(lat_axis, lon_axis))
    return np.where(cnt > 0, num, np.nan)



def cal_area_weighted_mean(arr: np.ndarray, area: np.ndarray, lat_axis: int = -2, lon_axis: int = -1) -> np.ndarray:
    """面积加权平均：mean_w = sum(arr*area)/sum(area)。"""
    num = cal_area_weighted_sum(arr, area, lat_axis=lat_axis, lon_axis=lon_axis)
    den = cal_area_weighted_sum(np.ones_like(arr), area, lat_axis=lat_axis, lon_axis=lon_axis)
    return np.divide(num, den, out=np.full_like(num, np.nan, dtype=float), where=np.isfinite(den) & (den != 0))



def _neighbor_dirs(metric: str) -> Tuple[Tuple[int, int], ...]:
    if metric == "chebyshev":
        return ((-1,-1), (-1,0), (-1,1),
                ( 0,-1),         ( 0,1),
                ( 1,-1), ( 1,0), ( 1,1))
    elif metric == "manhattan":
        return ((-1,0), (0,-1), (0,1), (1,0))
    else:
        raise ValueError("neighbor_metric must be 'chebyshev' or 'manhattan'.")



def _bfs_distance(sources: np.ndarray, metric: str) -> np.ndarray:
    """多源 BFS: 步数 (源=0; -1=不可达/无源)"""
    h, w = sources.shape
    dist = np.full((h, w), -1, dtype=np.int32)
    ys, xs = np.where(sources)
    if not ys.size:
        return dist
    q = deque()
    for y, x in zip(ys, xs):
        dist[y, x] = 0
        q.append((y, x))
    dirs = _neighbor_dirs(metric)
    while q:
        y, x = q.popleft()
        d0 = dist[y, x]
        for dy, dx in dirs:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and dist[ny, nx] == -1:
                dist[ny, nx] = d0 + 1
                q.append((ny, nx))
    return dist



def _dijkstra_lcc_distance_meters(
    sources: np.ndarray,
    DX: float, DY: float,
    MAPFAC_MX: np.ndarray, MAPFAC_MY: np.ndarray,
    metric: str = "chebyshev",
) -> np.ndarray:
    """多源 Dijkstra: 物理距离 (米)"""
    h, w = sources.shape
    if sources.dtype != bool:
        sources = sources > 0
    if not np.any(sources):
        return np.full((h, w), np.nan, dtype=np.float64)
    dirs = _neighbor_dirs(metric)
    dist = np.full((h, w), np.inf, dtype=np.float64)
    pq: list[tuple[float,int,int]] = []
    ys, xs = np.where(sources)
    for y, x in zip(ys, xs):
        dist[y, x] = 0.0
        heapq.heappush(pq, (0.0, y, x))
    while pq:
        d0, y, x = heapq.heappop(pq)
        if d0 > dist[y, x]:
            continue
        dx_m_here = DX / float(MAPFAC_MX[y, x])
        dy_m_here = DY / float(MAPFAC_MY[y, x])
        for dy, dx in dirs:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < h and 0 <= nx < w):
                continue
            dx_m_next = DX / float(MAPFAC_MX[ny, nx])
            dy_m_next = DY / float(MAPFAC_MY[ny, nx])
            dx_m_eff = 0.5 * (dx_m_here + dx_m_next)
            dy_m_eff = 0.5 * (dy_m_here + dy_m_next)
            if dx == 0:
                w_edge = dy_m_eff
            elif dy == 0:
                w_edge = dx_m_eff
            else:
                w_edge = (dx_m_eff**2 + dy_m_eff**2) ** 0.5
            nd = d0 + w_edge
            if nd < dist[ny, nx]:
                dist[ny, nx] = nd
                heapq.heappush(pq, (nd, ny, nx))
    return dist



def get_lake_area_mask(
        arr: np.ndarray,
        lakedp: np.ndarray,
        *,
        neighbor_metric: str = "chebyshev",
        cap: int = 1999,
        write_netcdf: bool = True,
        nc_path: str = "lake_area_mask.nc",
        DX: float | None = None,
        DY: float | None = None,
        MAPFAC_MX: np.ndarray | None = None,
        MAPFAC_MY: np.ndarray | None = None,
    ) -> dict[str,np.ndarray]:
    """
    输出三类数据 (all, shallow, deep)，每类都有:
      - lake_code_*   (编码: core=1000, rings=1001..cap)
      - dist_steps_*  (步数, BFS; -1=无源)
      - dist_m_*      (米距, Dijkstra; NaN=无源)
    """
    arr = np.asarray(arr).squeeze()
    if arr.ndim != 2:
        raise ValueError("`arr` must be 2D.")
    h, w = arr.shape
    shallow = (arr == 5)
    deep    = (arr == 6)
    ocean   = (arr == 8)
    all_lake = shallow | deep

    results = {}

    for name, core in [("all", all_lake), ("shallow", shallow), ("deep", deep)]:
        steps = _bfs_distance(core, neighbor_metric)
        # lake_code
        code = np.zeros((h, w), dtype=np.int32)
        if np.any(core):
            code[core] = 1000
            d = steps.copy()
            d[d < 0] = (1 << 30)
            lab = 1000 + d
            if cap is not None:
                lab = np.minimum(lab, cap)
            code = lab.astype(np.int32)
        # meters
        if (DX is not None) and (DY is not None) and (MAPFAC_MX is not None) and (MAPFAC_MY is not None):
            dist_m = _dijkstra_lcc_distance_meters(core, DX, DY, MAPFAC_MX, MAPFAC_MY, metric=neighbor_metric)
        else:
            dist_m = np.full((h, w), np.nan, dtype=np.float64)
        results[f"lake_code_{name}"] = code
        results[f"dist_steps_{name}"] = steps
        results[f"dist_m_{name}"] = dist_m
    results["shallow"] = shallow.astype(np.int32)
    results["deep"] = deep.astype(np.int32)
    results["all"] = all_lake.astype(np.int32)
    results["ocean"] = ocean.astype(np.int32)
    results["lakedp"] = lakedp.astype(np.float32)

    if write_netcdf:
        coords = {"y": np.arange(h), "x": np.arange(w)}
        ds = xr.Dataset(
            {k:(("y","x"),v) for k,v in results.items()},
            coords=coords,
            attrs={
                "neighbor_metric": neighbor_metric,
                "cap": int(cap) if cap is not None else -1,
                "note": "Three sets (all/shallow/deep). lake_code: 1000 core, 1001..cap rings. "
                        "dist_steps: BFS steps. dist_m: Dijkstra with DX,DY,MAPFAC."
            }
        )
        ds.to_netcdf(nc_path, mode="w", format="NETCDF4")

    return results



# ---------- calendar helpers ----------
def get_months_day(year: int) -> list[int]:
    """给定年份各月的天数（考虑闰年）。"""
    return [calendar.monthrange(year, m)[1] for m in range(1, 13)]



def get_seasons() -> list[str]:
    """季节缩写列表：DJF, MAM, JJA, SON。"""
    return [ "MAM", "JJA", "SON", "DJF"]


def get_season_months() -> dict[str, int]:
    """季节到月份的映射。"""
    return {"DJF": [12, 1, 2], "MAM": [3, 4, 5], "JJA": [6, 7, 8], "SON": [9, 10, 11]}


def get_season_days() -> dict[str, int]:
    return {"DJF": 90, "MAM": 92, "JJA": 92, "SON": 91, "Year": 365, "LeapYear": 366, 'Annual': 365}


def get_months() -> list[str]:
    """月份英文缩写。"""
    return ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def get_main_hours() -> list[int]:
    """获取时间（UTC）列表。"""
    return [16, 22, 4, 10, ]


def get_all_hours() -> list[int]:
    """获取所有小时（UTC）列表。"""
    return list(range(24))


def UTC_to_BJT_str(utc_hour: int) -> str:
    """将 UTC 小时转换为北京时间 (BJT, UTC+8) 字符串。"""
    bjt_hour = (utc_hour + 8) % 24
    return f"{bjt_hour:02d} BJT"


def UTC_to_BJT(utc_hour: int) -> str:
    """将 UTC 小时转换为北京时间 (BJT, UTC+8) 字符串。"""
    bjt_hour = (utc_hour + 8) % 24
    return bjt_hour




def standardize_event_name(event_name: str) -> str:
    if event_name.lower()=="cold":
        return "Cold"
    elif event_name.lower()=="hot":
        return "Hot"
    elif event_name.lower()=="dry":
        return "Dry"
    elif event_name.lower()=="wet":
        return "Wet"
    elif event_name.lower()=="colddry":
        return "Cold-Dry"
    elif event_name.lower()=="coldwet":
        return "Cold-Wet"
    elif event_name.lower()=="hotdry":
        return "Hot-Dry"
    elif event_name.lower()=="hotwet":
        return "Hot-Wet"
    else:
        raise ValueError(f"Unknown event name: {event_name}")




def make_season_coords():
    """
    返回一个 dict，包含季节相关的坐标，用于 xarray Dataset。
    - 主维度: season (DJF/MAM/JJA/SON)，字符串形式
    - 辅助坐标: season_id (int8)，并带 CF-style flag_values/flag_meanings
    """
    seasons = ["DJF", "MAM", "JJA", "SON"]
    season_map = {"DJF": 1, "MAM": 2, "JJA": 3, "SON": 4}
    # 主坐标
    season_coord = xr.DataArray(
        np.array(seasons, dtype=object),
        dims="season",
        name="season",
    )
    season_coord.attrs.update({
        "long_name": "climatological season label"
    })
    # 数值型辅助坐标
    season_id = xr.DataArray(
        np.array([season_map[s] for s in seasons], dtype="int8"),
        dims="season",
        name="season_id",
    )
    season_id.attrs.update({
        "long_name": "season id",
        "flag_values": np.array([1, 2, 3, 4], dtype="int8"),
        "flag_meanings": "DJF MAM JJA SON"
    })
    return {"season": season_coord, "season_id": season_id}



def calculate_wind_speed(u_ds, v_ds, uvar, vvar, wspd_name='UV10', wdir_name='UV10_DIR'):
    """
    计算风速和风向，只保留这两个变量在输出数据集中
    
    参数:
    ----------
    u_ds : xarray.Dataset
        包含U10风分量的数据集
    v_ds : xarray.Dataset  
        包含V10风分量的数据集
    wspd_name : str, 默认 'WSPD'
        输出风速变量的名称
    wdir_name : str, 默认 'WDIR'
        输出风向变量的名称
        
    返回:
    -------
    xarray.Dataset
        只包含风速和风向的数据集
    """
    
    # 检查输入数据集是否具有相同的维度和坐标
    if not u_ds.sizes == v_ds.sizes:
        raise ValueError("U和V数据集维度不匹配")
    
    # 提取U和V分量数据
    if uvar in u_ds.variables:
        u_data = u_ds[uvar]
    else:
        # 尝试找到U分量变量
        u_vars = [var for var in u_ds.variables if var.upper() in ['U10', 'U', 'AU10', 'Plev_U']]
        if u_vars:
            u_data = u_ds[u_vars[0]]
        else:
            raise ValueError("在U数据集中找不到U10风分量")
    
    if vvar in v_ds.variables:
        v_data = v_ds[vvar]
    else:
        # 尝试找到V分量变量
        v_vars = [var for var in v_ds.variables if var.upper() in ['V10', 'V', 'AV10', 'Plev_V']]
        if v_vars:
            v_data = v_ds[v_vars[0]]
        else:
            raise ValueError("在V数据集中找不到V10风分量")
    
    # 计算风速: WSPD = sqrt(U² + V²)
    wind_speed = np.sqrt(u_data**2 + v_data**2)
        
    # 创建新的数据集，只包含风速和风向
    output_ds = xr.Dataset()
    
    # 添加风速
    output_ds[wspd_name] = wind_speed
    output_ds[wspd_name].attrs.update({
        'long_name': 'wind speed',
        'units': 'm/s',
        'description': 'Calculated from U and V components'
    })
        
    # 复制所有坐标和属性
    for coord in u_ds.coords:
        output_ds.coords[coord] = u_ds.coords[coord]
    
    # 复制全局属性
    output_ds.attrs.update(u_ds.attrs)
    
    # 数据范围检查和清理
    output_ds[wspd_name] = output_ds[wspd_name].where(
        (output_ds[wspd_name] >= 0) & (output_ds[wspd_name] <= 200), 
        np.nan
    )
    
    return output_ds