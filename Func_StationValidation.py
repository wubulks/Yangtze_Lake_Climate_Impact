#!/stu01/wumej22/Anaconda3/envs/cwrf_env/bin/python

import os
import math
import csv
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("FONTCONFIG_PATH", "/etc/fonts")
os.environ.setdefault("FONTCONFIG_FILE", "/etc/fonts/fonts.conf")
import cmaps
import matplotlib
import geopandas as gpd
import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from joblib import Parallel, delayed
from cnmaps import get_adm_maps, draw_maps
# 自定义工具箱
import ToolBoxes.Tool_PlotConfig as TPC
import ToolBoxes.Tool_PerformanceMetrics as TPM
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import ToolBoxes.Tool_PlotColorBar as TPCB
import ToolBoxes.Tool_ImageToolkit as TIT
import ToolBoxes.Tool_YangtzeColorMap as TYCM
import ToolBoxes.Utils as TU
FIGFMT = TPC.FIGFMT
DPI    = TPC.DPI_medium

matplotlib.use('Agg')  # 不显示图，只保存
matplotlib.rcParams['font.family'] = 'Noto Sans'
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans', 'Arial', 'DejaVu Sans']
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Noto Sans:bold'
matplotlib.rcParams['mathtext.it'] = 'Noto Sans:bold:italic'
matplotlib.rcParams['mathtext.bf'] = 'Noto Sans:bold'
matplotlib.rcParams['mathtext.default'] = 'rm'
# matplotlib.rcParams['axes.unicode_minus'] = False

Station_Metrics = [
    "RMSE", "RBAIS", "BIAS", "MAE", "MAPE",
    "NMAE_SIGMA", "NMAE_IQR", "NMAE_MEAN", "NMAE_WET",
    "CRESM", "RRMSE",
    "TCC", "R2", "STD_MODEL", "STD_OBS", "STD_RATIO",
]
Season_List = ["MAM", "JJA", "SON", "DJF"]
Season_Months = {
    "MAM": {3, 4, 5},
    "JJA": {6, 7, 8},
    "SON": {9, 10, 11},
    "DJF": {12, 1, 2},
}
Var_Label = {
    "T2m": "T2m (degC)",
    "Prec": "Precipitation (mm/day)",
}
MODEL_LABEL = "This study"
EXCLUDED_PLOT_STATIONS = {
    "57776",
    "58596",
    "58666",
    "58944",
    "58437",
    "58543",
    "58652",
    "58752",
}

Metric_Column_Map = {
    'NMAE': {
        'T2m': 'NMAE_MEAN',
        'Prec': 'NMAE_WET',
    },
}

Daily_Level_Dicts = {
    'T2m': {
        'RMSE': {
            'maplevs': [np.linspace(0, 5, 201), 5],
            'boxlevs': np.arange(0, 5.1, 1),
            'cmap': cmaps.WhiteBlueGreenYellowRed,
            'unit': '°C',
        },
        'Bias': {
            'maplevs': [np.linspace(-7, 7, 201), 5],
            'boxlevs': np.arange(-7, 7.1, 2),
            'cmap': cmaps.MPL_PuOr,
            'unit': '°C',
        },
        'NMAE': {
            'maplevs': [np.linspace(0, 20, 201), 5],
            'boxlevs': np.arange(0, 20.1, 5),
            'cmap': cmaps.WhiteBlueGreenYellowRed,
            'unit': '%',
        },
        'MAE': {
            'maplevs': [np.linspace(0, 3, 201), 4],
            'boxlevs': np.arange(0, 3.1, 1),
            'cmap': cmaps.WhiteBlueGreenYellowRed,
            'unit': '°C',
        },
        'TCC': {
            'maplevs': [np.linspace(0, 1, 201), 5],
            'boxlevs': np.arange(0, 1.1, 0.2),
            'cmap': cmaps.MPL_RdBu,
            'unit': '',
        },
        'R2': {
            'maplevs': [np.linspace(0, 1, 201), 5],
            'boxlevs': np.arange(0, 1.1, 0.2),
            'cmap': cmaps.MPL_RdBu,
            'unit': '',
        }
    },
    'Prec': {
        'RMSE': {
            'maplevs': [np.linspace(0, 5, 201), 5],
            'boxlevs': np.arange(0, 5.1, 1),
            'cmap': cmaps.WhiteBlueGreenYellowRed,
            'unit': 'mm/day',
        },
        'Bias': {
            'maplevs': [np.linspace(-7, 7, 201), 5],
            'boxlevs': np.arange(-7, 7.1, 2),
            'cmap': cmaps.MPL_PuOr,
            'unit': 'mm/day',
        },
        'NMAE': {
            'maplevs': [np.linspace(0, 70, 201), 5],
            'boxlevs': np.arange(0, 70.1, 10),
            'cmap': cmaps.WhiteBlueGreenYellowRed,
            'unit': '%',
        },
        'MAE': {
            'maplevs': [np.linspace(0, 7, 201), 5],
            'boxlevs': np.arange(0, 7.1, 0.2),
            'cmap': cmaps.WhiteBlueGreenYellowRed,
            'unit': 'mm/day',
        },
        'TCC': {
            'maplevs': [np.linspace(0, 1, 201), 5],
            'boxlevs': np.arange(0, 1.1, 0.2),
            'cmap': cmaps.MPL_RdBu,
            'unit': '',
        },
        'R2': {
            'maplevs': [np.linspace(0, 1, 201), 5],
            'boxlevs': np.arange(0, 1.1, 0.2),
            'cmap': cmaps.MPL_RdBu,
            'unit': '',
        },
    }
}

SeasonMean_Level_Dicts = {
    'T2m': {
        'RMSE': {
            'maplevs': [np.linspace(0, 5, 201), 5],
            'boxlevs': np.arange(0, 3.1, 0.5),
            'cmap': cmaps.WhiteBlueGreenYellowRed,
            'unit': '°C',
        },
        'Bias': {
            'maplevs': [np.linspace(-7, 7, 201), 5],
            'boxlevs': np.arange(-4, 4.1, 2),
            'cmap': cmaps.MPL_PuOr,
            'unit': '°C',
        },
        'NMAE': {
            'maplevs': [np.linspace(0, 20, 201), 5],
            'boxlevs': np.arange(0, 20.1, 5),
            'cmap': cmaps.WhiteBlueGreenYellowRed,
            'unit': '%',
        },
        'MAE': {
            'maplevs': [np.linspace(0, 5, 201), 5],
            'boxlevs': np.arange(0, 2.6, 0.5),
            'cmap': cmaps.WhiteBlueGreenYellowRed,
            'unit': '°C',
        },
        'TCC': {
            'maplevs': [np.linspace(0, 1, 201), 5],
            'boxlevs': np.arange(0, 1.1, 0.2),
            'cmap': cmaps.MPL_RdBu,
            'unit': '',
        },
        'R2': {
            'maplevs': [np.linspace(0, 1, 201), 5],
            'boxlevs': np.arange(0, 1.1, 0.2),
            'cmap': cmaps.MPL_RdBu,
            'unit': '',
        }
    },
    'Prec': {
        'RMSE': {
            'maplevs': [np.linspace(0, 5, 201), 5],
            'boxlevs': np.arange(0, 4.1, 1),
            'cmap': cmaps.WhiteBlueGreenYellowRed,
            'unit': 'mm/day',
        },
        'Bias': {
            'maplevs': [np.linspace(-7, 7, 201), 5],
            'boxlevs': np.arange(-4, 4.1, 2),
            'cmap': cmaps.MPL_PuOr,
            'unit': 'mm/day',
        },
        'NMAE': {
            'maplevs': [np.linspace(0, 70, 201), 5],
            'boxlevs': np.arange(0, 70.1, 10),
            'cmap': cmaps.WhiteBlueGreenYellowRed,
            'unit': '%',
        },
        'MAE': {
            'maplevs': [np.linspace(0, 5, 201), 5],
            'boxlevs': np.arange(0, 3.1, 0.5),
            'cmap': cmaps.WhiteBlueGreenYellowRed,
            'unit': 'mm/day',
        },
        'TCC': {
            'maplevs': [np.linspace(0, 1, 201), 5],
            'boxlevs': np.arange(0, 1.1, 0.2),
            'cmap': cmaps.MPL_RdBu,
            'unit': '',
        },
        'R2': {
            'maplevs': [np.linspace(0, 1, 201), 5],
            'boxlevs': np.arange(0, 1.1, 0.2),
            'cmap': cmaps.MPL_RdBu,
            'unit': '',
        }
    }
}

SeasonDaily_Level_Dicts = {
    'T2m': {
        'RMSE': {
            'maplevs': [np.linspace(0, 5, 201), 5],
            'boxlevs': np.arange(0, 4.1, 1),
            'cmap': cmaps.WhiteBlueGreenYellowRed,
            'unit': '°C',
        },
        'Bias': {
            'maplevs': [np.linspace(-7, 7, 201), 5],
            'boxlevs': np.arange(-5, 5.1, 2),
            'cmap': cmaps.MPL_PuOr,
            'unit': '°C',
        },
        'NMAE': {
            'maplevs': [np.linspace(0, 20, 201), 5],
            'boxlevs': np.arange(0, 20.1, 5),
            'cmap': cmaps.WhiteBlueGreenYellowRed,
            'unit': '%',
        },
        'MAE': {
            'maplevs': [np.linspace(0, 1, 201), 5],
            'boxlevs': np.arange(0, 1.1, 0.2),
            'cmap': cmaps.WhiteBlueGreenYellowRed,
            'unit': '°C',
        },
        'R2': {
            'maplevs': [np.linspace(0, 1, 201), 5],
            'boxlevs': np.arange(0, 1.1, 0.2),
            'cmap': cmaps.MPL_RdBu,
            'unit': '',
        },
    },
    'Prec': {
        'RMSE': {
            'maplevs': [np.linspace(0, 15, 201), 5],
            'boxlevs': np.arange(0, 21.1, 3),
            'cmap': cmaps.WhiteBlueGreenYellowRed,
            'unit': 'mm/day',
        },
        'Bias': {
            'maplevs': [np.linspace(-7, 7, 201), 5],
            'boxlevs': np.arange(-5, 5.1, 2),
            'cmap': cmaps.MPL_PuOr,
            'unit': 'mm/day',
        },
        'NMAE': {
            'maplevs': [np.linspace(0, 30, 201), 5],
            'boxlevs': np.arange(0, 70.1, 10),
            'cmap': cmaps.WhiteBlueGreenYellowRed,
            'unit': '%',
        },
        'MAE': {
            'maplevs': [np.linspace(0, 2, 201), 5],
            'boxlevs': np.arange(0, 2.1, 0.2),
            'cmap': cmaps.WhiteBlueGreenYellowRed,
            'unit': 'mm/day',
        },
        'R2': {
            'maplevs': [np.linspace(0, 1, 201), 5],
            'boxlevs': np.arange(0, 1.1, 0.2),
            'cmap': cmaps.MPL_RdBu,
            'unit': '',
        },
    }
}


def _format_metric_value(value, fmt=".3g"):
    """Format metric values for plot annotations."""
    if value is None or not np.isfinite(value):
        return "NaN"
    return format(float(value), fmt)


def _make_figure_type_dir(figoutdir, figure_type, case=None, var=None):
    """Return a clean output folder grouped by figure type."""
    parts = [figoutdir, figure_type]
    if case is not None:
        parts.append(str(case))
    if var is not None:
        parts.append(str(var))
    outdir = os.path.join(*parts)
    os.makedirs(outdir, exist_ok=True)
    return outdir


def _safe_path_token(value):
    return str(value).replace(" ", "_").replace("/", "-")


def _filter_excluded_stations(stations, excluded_stations=EXCLUDED_PLOT_STATIONS):
    excluded = {str(station) for station in excluded_stations}
    return [str(station) for station in stations if str(station) not in excluded]


def _make_radius_ticks(rmax):
    if rmax <= 1.6:
        step = 0.2
    elif rmax <= 3.2:
        step = 0.5
    else:
        step = 1.0
    upper = math.ceil(rmax / step) * step
    return np.arange(0.0, upper + 0.5 * step, step)


def _metric_label(metric, var=None, var_label=Var_Label):
    metric_name = str(metric)
    if var is not None and metric_name.upper() in {"RMSE", "BIAS", "MAE", "CRESM"}:
        return f"{metric_name} ({var_label.get(var, var)})"
    if metric_name.upper() in {"TCC", "R2", "STD_RATIO"}:
        return metric_name
    if metric_name.upper() in {"RBAIS", "MAPE"}:
        return f"{metric_name} (%)"
    return metric_name


def _metric_for_data(var, metric):
    """根据变量映射实际读取的指标列。"""
    metric_name = str(metric)
    metric_upper = metric_name.upper()
    for key, vardict in Metric_Column_Map.items():
        if metric_upper == str(key).upper():
            return vardict.get(var, metric_name)
    return metric_name


def _as_mpl_cmap(cmap):
    return plt.get_cmap(cmap) if isinstance(cmap, str) else cmap


def _station_metric_leveldict(scale_type):
    """根据验证尺度返回站点空间图 level/cmap 配置。"""
    scale_type = str(scale_type).strip().lower()
    if scale_type in {"daily", "day"}:
        return Daily_Level_Dicts
    if scale_type in {"seasonal_mean", "seasonalmean", "seasonmean"}:
        return SeasonMean_Level_Dicts
    if scale_type in {"seasonal_daily", "seasonaldaily", "seasondaily"}:
        return SeasonDaily_Level_Dicts
    raise ValueError(f"Unsupported station metric scale_type: {scale_type}")


def _get_station_metric_cfg(leveldict, var, metric):
    """大小写不敏感地读取站点空间图配色配置。"""
    if var not in leveldict:
        raise KeyError(f"Missing variable in station level dict: {var}")
    metric = str(metric)
    if metric in leveldict[var]:
        return leveldict[var][metric]
    metric_upper = metric.upper()
    for key, cfg in leveldict[var].items():
        if str(key).upper() == metric_upper:
            return cfg
    raise KeyError(f"Missing metric in station level dict: {var}, {metric}")


def _find_metric_col(df, metric, var=None):
    """大小写不敏感地查找指标列名。"""
    metric_upper = str(_metric_for_data(var, metric) if var is not None else metric).upper()
    for col in df.columns:
        if str(col).upper() == metric_upper:
            return col
    raise KeyError(f"metrics_df 缺少指标列: {metric}")


def _station_metric_mapcfg(var, metric, mode="Daily"):
    """
    站点空间图配色配置。

    优先使用本文件中定义的 Daily/SeasonMean/SeasonDaily_Level_Dicts；
    若某变量或指标未配置，则回退到基于数据范围的动态色阶。
    """
    if mode not in {"Daily", "SeasonalMean", "SeasonalDaily"}:
        raise ValueError(f"Unsupported mode: {mode}")
    if mode == "Daily":
        cfg = _get_station_metric_cfg(Daily_Level_Dicts, var, metric)
    elif mode == "SeasonalMean":
        cfg = _get_station_metric_cfg(SeasonMean_Level_Dicts, var, metric)
    elif mode == "SeasonalDaily":
        cfg = _get_station_metric_cfg(SeasonDaily_Level_Dicts, var, metric)
    mapcfg = TPC.mapConfig(levs=cfg['maplevs'], cmap=cfg['cmap'])
    return mapcfg


def _station_metric_boxlevs(var, metric, mode="Daily"):
    """读取站点箱型图 y 轴刻度配置。"""
    if mode not in {"Daily", "SeasonalMean", "SeasonalDaily"}:
        raise ValueError(f"Unsupported mode: {mode}")
    if mode == "Daily":
        cfg = _get_station_metric_cfg(Daily_Level_Dicts, var, metric)
    elif mode == "SeasonalMean":
        cfg = _get_station_metric_cfg(SeasonMean_Level_Dicts, var, metric)
    elif mode == "SeasonalDaily":
        cfg = _get_station_metric_cfg(SeasonDaily_Level_Dicts, var, metric)
    return np.asarray(cfg['boxlevs'], dtype=float)



def _format_numeric_tick(value):
    if abs(value - round(value)) < 1.0e-10:
        return f"{value:.1f}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _format_corr_tick(value):
    if value in (0.95, 0.99):
        return f"{value:.2f}"
    return f"{value:.1f}"


def extract_case_station_data(caseds, station_df, bufferzone=0, station_col="station_id"):
    """
    从 CWRFPOST 的 caseds 中提取站点所在格点时间序列。

    station_df 中的 i/j 为原始 wrfinput 的 0-based west_east/south_north 索引；
    caseds 已经裁掉 bufferzone，因此这里统一换算成裁切后的 x/y 索引。
    返回的 DataFrame：index 为日期，columns 为站点号。
    """
    required_cols = {station_col, "i", "j"}
    missing_cols = required_cols - set(station_df.columns)
    if missing_cols:
        raise ValueError(f"station_df 缺少必要列: {sorted(missing_cols)}")

    if isinstance(caseds, xr.Dataset):
        if len(caseds.data_vars) != 1:
            raise ValueError("caseds 是 Dataset 时只能包含一个变量，或请先传入目标 DataArray。")
        caseds = next(iter(caseds.data_vars.values()))

    for dim in ["time", "y", "x"]:
        if dim not in caseds.dims:
            raise ValueError(f"caseds 缺少维度: {dim}")

    station_info = station_df[[station_col, "i", "j"]].dropna().drop_duplicates(station_col).copy()
    station_info[station_col] = station_info[station_col].astype(str)
    station_info["x_idx"] = station_info["i"].astype(int) - int(bufferzone)
    station_info["y_idx"] = station_info["j"].astype(int) - int(bufferzone)

    valid_mask = (
        station_info["x_idx"].between(0, caseds.sizes["x"] - 1)
        & station_info["y_idx"].between(0, caseds.sizes["y"] - 1)
    )
    invalid_count = int((~valid_mask).sum())
    if invalid_count:
        print(f"    Warning: {invalid_count} stations are outside caseds grid after bufferzone correction.")
    station_info = station_info.loc[valid_mask].copy()
    if station_info.empty:
        raise ValueError("没有站点落在 caseds 网格范围内。")

    station_ids = station_info[station_col].to_numpy()
    x_idx = xr.DataArray(station_info["x_idx"].to_numpy(), dims="station", coords={"station": station_ids})
    y_idx = xr.DataArray(station_info["y_idx"].to_numpy(), dims="station", coords={"station": station_ids})
    station_da = caseds.isel(x=x_idx, y=y_idx).transpose("time", "station")

    out_df = station_da.to_pandas()
    out_df.index = pd.to_datetime(out_df.index).strftime("%Y-%m-%d")
    out_df.index.name = "Date"
    return out_df


def save_case_station_csv(case, var, caseds, station_df, outdir, bufferzone=0):
    """保存站点提取结果为 {case}_{var}.csv。"""
    os.makedirs(outdir, exist_ok=True)
    out_df = extract_case_station_data(caseds, station_df, bufferzone=bufferzone)
    savepath = f"{outdir}/{case}_{var}.csv"
    out_df.to_csv(savepath, na_rep="NaN")
    print(f"    Saved station data: {savepath}")
    return savepath


def read_station_table(filepath, starttime, endtime):
    """读取 Date + station columns 格式的站点日序列表。"""
    df = pd.read_csv(filepath, dtype={"Date": str})
    if "Date" not in df.columns:
        raise ValueError(f"{filepath} 缺少 Date 列。")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    df.columns = df.columns.astype(str)
    df = df.apply(pd.to_numeric, errors="coerce")
    return df.loc[pd.to_datetime(starttime):pd.to_datetime(endtime)]


def read_station_obs_data(var, starttime, endtime, station_var_dir, station_obs_file):
    """读取站点观测变量表；T2m 对应平均温度，Prec 对应降水。"""
    if var not in station_obs_file:
        raise ValueError(f"未配置站点观测变量: {var}")
    return read_station_table(f"{station_var_dir}/{station_obs_file[var]}", starttime, endtime)


def read_case_station_data(case, var, starttime, endtime, outdir):
    """读取已提取的模式站点序列 {case}_{var}.csv。"""
    return read_station_table(f"{outdir}/{case}_{var}.csv", starttime, endtime)


def read_station_metrics(case, var, outdir, suffix="", station_metrics=Station_Metrics):
    """读取逐站点验证指标 {case}_{var}{suffix}_StationMetrics.csv。"""
    filepath = f"{outdir}/{case}_{var}{suffix}_StationMetrics.csv"
    metrics_df = pd.read_csv(filepath, dtype={"station_id": str})
    for col in station_metrics:
        if col in metrics_df.columns:
            metrics_df[col] = pd.to_numeric(metrics_df[col], errors="coerce")
    return metrics_df


def calc_station_metrics(pred, obs, station_metrics=Station_Metrics):
    """调用 Tool_PerformanceMetrics.py 中的 1D 函数计算单站指标。"""
    valid = np.isfinite(pred) & np.isfinite(obs)
    n_valid = int(valid.sum())
    row = {"N": n_valid}
    if n_valid == 0:
        row.update({metric: np.nan for metric in station_metrics})
        return row

    pred_valid = pred[valid]
    obs_valid = obs[valid]
    pred_std = np.std(pred_valid, ddof=1) if n_valid >= 2 else np.nan
    obs_std = np.std(obs_valid, ddof=1) if n_valid >= 2 else np.nan
    if (
        n_valid >= 2
        and np.isfinite(pred_std)
        and np.isfinite(obs_std)
        and pred_std > 0
        and obs_std > 0
    ):
        tcc = np.corrcoef(pred_valid, obs_valid)[0, 1]
        std_ratio = pred_std / obs_std
    else:
        tcc = np.nan
        std_ratio = np.nan

    with np.errstate(invalid="ignore", divide="ignore"):
        row.update({
            "RMSE": TPM.RMSE_1D(pred_valid, obs_valid),
            "RBAIS": TPM.RBias_1D(pred_valid, obs_valid),
            "BIAS": TPM.Bias_1D(pred_valid, obs_valid),
            "MAE": TPM.MAE_1D(pred_valid, obs_valid),
            "MAPE": TPM.MAPE_1D(pred_valid, obs_valid),
            "NMAE_SIGMA": TPM.NMAE_sigma_anom_1D(pred_valid, obs_valid),
            "NMAE_IQR": TPM.NMAE_iqr_anom_1D(pred_valid, obs_valid),
            "NMAE_MEAN": TPM.NMAE_mean_1D(pred_valid, obs_valid),
            "NMAE_WET": TPM.NMAE_mean_wet_1D(pred_valid, obs_valid),
            "CRESM": TPM.CRMSE_1D(pred_valid, obs_valid),
            "RRMSE": TPM.RRMSE_1D(pred_valid, obs_valid),
            "TCC": tcc,
            "R2": TPM.R2_1D(pred_valid, obs_valid),
            "STD_MODEL": pred_std,
            "STD_OBS": obs_std,
            "STD_RATIO": std_ratio,
        })
    return row


def get_station_meta(station_df):
    """整理站点元数据索引，供日尺度和季节尺度验证共用。"""
    return (
        station_df.copy()
        .assign(station_id=lambda df: df["station_id"].astype(str))
        .drop_duplicates("station_id")
        .set_index("station_id")
    )


def station_seasonal_mean(df, seasons=Season_List, season_months=Season_Months):
    """
    先按季节年计算站点季节平均。

    DJF 按跨年冬季处理：12 月归到下一年的 DJF，例如 2000-12、2001-01、2001-02
    共同组成 season_year=2001 的 DJF。起止时间造成的非完整季节会被剔除。
    """
    work_df = df.copy()
    month = work_df.index.month
    year = work_df.index.year

    season = pd.Series(index=work_df.index, dtype=object)
    season.loc[month.isin([3, 4, 5])] = "MAM"
    season.loc[month.isin([6, 7, 8])] = "JJA"
    season.loc[month.isin([9, 10, 11])] = "SON"
    season.loc[month.isin([12, 1, 2])] = "DJF"
    season_year = pd.Series(year, index=work_df.index)
    season_year.loc[month == 12] = season_year.loc[month == 12] + 1

    labels = pd.DataFrame({
        "season": season,
        "season_year": season_year,
        "month": month,
    }, index=work_df.index).dropna(subset=["season"])

    month_sets = labels.groupby(["season", "season_year"])["month"].agg(lambda values: set(values))
    complete_groups = {
        (season_name, int(season_year_value))
        for (season_name, season_year_value), months in month_sets.items()
        if months == season_months[season_name]
    }

    grouped = work_df.groupby([labels["season"], labels["season_year"]]).mean()
    seasonal_df = {}
    for season_name in seasons:
        rows = [
            season_year_value
            for grouped_season, season_year_value in grouped.index
            if grouped_season == season_name and (grouped_season, int(season_year_value)) in complete_groups
        ]
        if rows:
            season_data = grouped.loc[(season_name, rows), :].copy()
            season_data.index = pd.Index([int(idx[1]) for idx in season_data.index], name="season_year")
            seasonal_df[season_name] = season_data.sort_index()
        else:
            seasonal_df[season_name] = pd.DataFrame(columns=work_df.columns, index=pd.Index([], name="season_year"))
    return seasonal_df


def station_seasonal_daily_subset(df, seasons=Season_List):
    """按季节筛选逐日数据，不先做季节年平均。"""
    month = df.index.month
    seasonal_df = {}
    seasonal_df["MAM"] = df.loc[month.isin([3, 4, 5])].copy()
    seasonal_df["JJA"] = df.loc[month.isin([6, 7, 8])].copy()
    seasonal_df["SON"] = df.loc[month.isin([9, 10, 11])].copy()
    seasonal_df["DJF"] = df.loc[month.isin([12, 1, 2])].copy()
    return {season: seasonal_df[season] for season in seasons}


def _seasonal_mode_info(seasonal_mode):
    mode = str(seasonal_mode).strip().lower()
    aliases = {
        "mean": "seasonal_mean",
        "seasonalmean": "seasonal_mean",
        "seasonal_mean": "seasonal_mean",
        "daily": "seasonal_daily",
        "seasonaldaily": "seasonal_daily",
        "seasonal_daily": "seasonal_daily",
    }
    mode = aliases.get(mode, mode)
    if mode == "seasonal_mean":
        return mode, "SeasonalMean", "Seasonal Mean"
    if mode == "seasonal_daily":
        return mode, "SeasonalDaily", "Seasonal Daily"
    raise ValueError(f"Unsupported seasonal_mode: {seasonal_mode}")


def _get_seasonal_data(case_df, obs_df, seasons=Season_List, seasonal_mode="seasonal_mean"):
    mode, _, _ = _seasonal_mode_info(seasonal_mode)
    if mode == "seasonal_mean":
        return station_seasonal_mean(case_df, seasons=seasons), station_seasonal_mean(obs_df, seasons=seasons)
    return station_seasonal_daily_subset(case_df, seasons=seasons), station_seasonal_daily_subset(obs_df, seasons=seasons)


def evaluate_station_metrics(
    case,
    var,
    case_df,
    obs_df,
    station_df,
    outdir,
    suffix="",
    scale_label="daily",
    station_metrics=Station_Metrics,
):
    """逐站点计算模式已提取序列与观测序列的验证指标。"""
    common_dates = case_df.index.intersection(obs_df.index)
    common_stations = [station for station in case_df.columns if station in obs_df.columns]
    if len(common_dates) == 0:
        raise ValueError(f"{case}_{var} {scale_label}: 模式与观测没有共同日期。")
    if len(common_stations) == 0:
        raise ValueError(f"{case}_{var} {scale_label}: 模式与观测没有共同站点。")

    station_meta = get_station_meta(station_df)
    rows = []
    for station_id in common_stations:
        metrics = calc_station_metrics(
            case_df.loc[common_dates, station_id].to_numpy(dtype=float),
            obs_df.loc[common_dates, station_id].to_numpy(dtype=float),
            station_metrics=station_metrics,
        )
        meta = station_meta.loc[station_id] if station_id in station_meta.index else pd.Series(dtype=float)
        rows.append({
            "station_id": station_id,
            "lat": meta.get("lat", np.nan),
            "lon": meta.get("lon", np.nan),
            "i": meta.get("i", np.nan),
            "j": meta.get("j", np.nan),
            **metrics,
        })

    metrics_df = pd.DataFrame(rows)
    metrics_df = metrics_df[["station_id", "lat", "lon", "i", "j", "N"] + station_metrics]
    savepath = f"{outdir}/{case}_{var}{suffix}_StationMetrics.csv"
    metrics_df.to_csv(savepath, index=False, na_rep="NaN")
    print(f"    Saved station metrics: {savepath}")
    return savepath


def evaluate_station_seasonal_metrics(
    case,
    var,
    case_df,
    obs_df,
    station_df,
    outdir,
    seasons=Season_List,
    station_metrics=Station_Metrics,
    seasonal_mode="seasonal_mean",
):
    """按季节计算验证指标。

    seasonal_mean: 先按季节年平均，再用季节年序列计算指标。
    seasonal_daily: 直接使用该季节内所有逐日数据计算指标。
    """
    _, mode_token, mode_label = _seasonal_mode_info(seasonal_mode)
    case_seasonal, obs_seasonal = _get_seasonal_data(case_df, obs_df, seasons=seasons, seasonal_mode=seasonal_mode)
    station_meta = get_station_meta(station_df)
    rows = []

    for season in seasons:
        case_season_df = case_seasonal[season]
        obs_season_df = obs_seasonal[season]
        common_index = case_season_df.index.intersection(obs_season_df.index)
        common_stations = [station for station in case_season_df.columns if station in obs_season_df.columns]
        if len(common_index) == 0 or len(common_stations) == 0:
            print(f"    Warning: {case}_{var}_{mode_token}_{season}: no common seasonal data.")
            continue

        for station_id in common_stations:
            metrics = calc_station_metrics(
                case_season_df.loc[common_index, station_id].to_numpy(dtype=float),
                obs_season_df.loc[common_index, station_id].to_numpy(dtype=float),
                station_metrics=station_metrics,
            )
            meta = station_meta.loc[station_id] if station_id in station_meta.index else pd.Series(dtype=float)
            rows.append({
                "seasonal_mode": mode_token,
                "season": season,
                "station_id": station_id,
                "lat": meta.get("lat", np.nan),
                "lon": meta.get("lon", np.nan),
                "i": meta.get("i", np.nan),
                "j": meta.get("j", np.nan),
                **metrics,
            })

    metrics_df = pd.DataFrame(rows)
    out_cols = ["seasonal_mode", "season", "station_id", "lat", "lon", "i", "j", "N"] + station_metrics
    if metrics_df.empty:
        metrics_df = pd.DataFrame(columns=out_cols)
    else:
        metrics_df = metrics_df[out_cols]
    savepath = f"{outdir}/{case}_{var}_{mode_token}_StationMetrics.csv"
    metrics_df.to_csv(savepath, index=False, na_rep="NaN")
    print(f"    Saved {mode_label} station metrics: {savepath}")

    for season in seasons:
        season_df = metrics_df.loc[metrics_df["season"] == season].drop(columns=["season"], errors="ignore")
        season_savepath = f"{outdir}/{case}_{var}_{mode_token}_{season}_StationMetrics.csv"
        season_df.to_csv(season_savepath, index=False, na_rep="NaN")
        print(f"    Saved {mode_label} {season} station metrics: {season_savepath}")
    return savepath


def plot_station_obs_model_scatter(
    case,
    var,
    case_df,
    obs_df,
    metrics_df,
    figoutdir,
    scale_label="Daily",
    suffix="",
    var_label=Var_Label,
    label_by_site=True,
    excluded_stations=EXCLUDED_PLOT_STATIONS,
):
    """绘制所有站点合并后的观测-模式散点图，并添加 1:1 线。"""
    print(f"    ---- Plotting station scatter: {case}, {var}, {scale_label} ----")
    common_dates = case_df.index.intersection(obs_df.index)
    common_stations = [station for station in case_df.columns if station in obs_df.columns]
    common_stations = _filter_excluded_stations(common_stations, excluded_stations=excluded_stations)
    if len(common_dates) == 0:
        raise ValueError(f"{case}_{var} {scale_label}: 模式与观测没有共同日期，无法绘图。")
    if len(common_stations) == 0:
        raise ValueError(f"{case}_{var} {scale_label}: 模式与观测没有共同站点，无法绘图。")

    pred_all = case_df.loc[common_dates, common_stations].to_numpy(dtype=float).ravel()
    obs_all = obs_df.loc[common_dates, common_stations].to_numpy(dtype=float).ravel()
    valid_all = np.isfinite(pred_all) & np.isfinite(obs_all)
    pred = pred_all[valid_all]
    obs = obs_all[valid_all]
    if pred.size == 0:
        raise ValueError(f"{case}_{var} {scale_label}: 没有有效的模式-观测配对数据，无法绘图。")
    print(f"        valid points: {pred.size}, stations: {len(common_stations)}")

    metrics_for_plot = metrics_df.copy()
    if "station_id" in metrics_for_plot.columns:
        metrics_for_plot["station_id"] = metrics_for_plot["station_id"].astype(str)
        metrics_for_plot = metrics_for_plot.loc[metrics_for_plot["station_id"].isin(common_stations)]
    mean_rmse = metrics_for_plot["RMSE"].mean(skipna=True)
    mean_bias = metrics_for_plot["BIAS"].mean(skipna=True)
    mean_rbias = metrics_for_plot["RBAIS"].mean(skipna=True)
    mean_mae = metrics_for_plot["MAE"].mean(skipna=True)

    vmin = np.nanmin([np.nanmin(obs), np.nanmin(pred)])
    vmax = np.nanmax([np.nanmax(obs), np.nanmax(pred)])
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError(f"{case}_{var}: 坐标范围无效，无法绘图。")
    if vmin == vmax:
        pad = 1.0 if vmin == 0 else abs(vmin) * 0.05
    else:
        pad = (vmax - vmin) * 0.05
    lims = [vmin - pad, vmax + pad]

    save_dir = _make_figure_type_dir(figoutdir, "Obs_vs_Model_Scatter", case, var)
    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    if label_by_site:
        cmap = plt.get_cmap("tab20")
        plotted_sites = 0
        for idx, station_id in enumerate(common_stations):
            station_obs = obs_df.loc[common_dates, station_id].to_numpy(dtype=float)
            station_pred = case_df.loc[common_dates, station_id].to_numpy(dtype=float)
            station_valid = np.isfinite(station_obs) & np.isfinite(station_pred)
            if not bool(station_valid.any()):
                continue
            ax.scatter(
                station_obs[station_valid],
                station_pred[station_valid],
                s=8,
                alpha=0.32,
                color=cmap(idx % cmap.N),
                edgecolors="none",
                rasterized=True,
                label=str(station_id),
            )
            plotted_sites += 1
    else:
        plotted_sites = 0
        ax.scatter(obs, pred, s=4, alpha=0.18, c="#2F6B9A", edgecolors="none", rasterized=True)
    ax.plot(lims, lims, color="black", lw=1.2, ls="--", label="_nolegend_" if label_by_site else "1:1")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"Observed {var_label.get(var, var)}")
    ax.set_ylabel(f"{MODEL_LABEL} {var_label.get(var, var)}")
    ax.set_title(f"{case} {var}: Station {scale_label} Scatter")
    ax.grid(True, ls=":", lw=0.6, alpha=0.6)
    if label_by_site and plotted_sites > 0:
        legend_ncol = max(1, int(np.ceil(plotted_sites / 24)))
        ax.legend(
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            title="Site",
            fontsize=7,
            title_fontsize=8,
            ncol=legend_ncol,
            markerscale=1.4,
        )
    else:
        ax.legend(frameon=False, loc="upper left")
    stat_text = (
        f"N = {pred.size}\n"
        f"Mean RMSE = {mean_rmse:.3g}\n"
        f"Mean BIAS = {mean_bias:.3g}\n"
        f"Mean RBAIS = {mean_rbias:.3g}%\n"
        f"Mean MAE = {mean_mae:.3g}"
    )
    ax.text(
        0.97, 0.03, stat_text,
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.85, "boxstyle": "round,pad=0.35"},
    )

    savepath = f"{save_dir}/{case}_{var}{suffix}_Obs_vs_Model_Scatter.{FIGFMT}"
    fig.savefig(savepath, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved station scatter: {savepath}")
    return savepath


def plot_station_timeseries(
    case,
    var,
    case_df,
    obs_df,
    metrics_df,
    figoutdir,
    suffix="",
    scale_label="Daily",
    var_label=Var_Label,
    excluded_stations=EXCLUDED_PLOT_STATIONS,
):
    """
    绘制每个站点的观测和模式时间序列。

    图片保存到 {figoutdir}/Station_Timeseries/{case}/{var}/，文件名为 {station_id}.png。
    """
    print(f"    ---- Plotting station time series: {case}, {var}, {scale_label} ----")
    common_dates = case_df.index.intersection(obs_df.index)
    common_stations = [station for station in case_df.columns if station in obs_df.columns]
    common_stations = _filter_excluded_stations(common_stations, excluded_stations=excluded_stations)
    if len(common_dates) == 0:
        raise ValueError(f"{case}_{var}: 模式与观测没有共同日期，无法绘制逐站时间序列。")
    if len(common_stations) == 0:
        raise ValueError(f"{case}_{var}: 模式与观测没有共同站点，无法绘制逐站时间序列。")

    required_cols = {"station_id", "RMSE", "BIAS", "TCC"}
    missing_cols = required_cols - set(metrics_df.columns)
    if missing_cols:
        raise ValueError(f"metrics_df 缺少逐站时间序列标注所需列: {sorted(missing_cols)}")

    metrics_lookup = metrics_df.copy()
    metrics_lookup["station_id"] = metrics_lookup["station_id"].astype(str)
    metrics_lookup = metrics_lookup.drop_duplicates("station_id").set_index("station_id")

    save_dir = os.path.join(
        _make_figure_type_dir(figoutdir, "Station_Timeseries", case, var),
        _safe_path_token(scale_label),
    )
    os.makedirs(save_dir, exist_ok=True)
    saved_paths = []
    print(f"        common dates: {len(common_dates)}, stations: {len(common_stations)}")

    for station_id in common_stations:
        plot_df = pd.DataFrame({
            "OBS": obs_df.loc[common_dates, station_id],
            "Model": case_df.loc[common_dates, station_id],
        }).replace([np.inf, -np.inf], np.nan)
        if scale_label.startswith("Seasonal Mean") and len(plot_df.index) > 1:
            numeric_years = pd.to_numeric(pd.Index(plot_df.index), errors="coerce")
            if np.isfinite(numeric_years).all():
                full_years = pd.Index(
                    range(int(np.nanmin(numeric_years)), int(np.nanmax(numeric_years)) + 1),
                    name=plot_df.index.name,
                )
                plot_df.index = numeric_years.astype(int)
                plot_df = plot_df.reindex(full_years)
        if plot_df.dropna(how="all").empty:
            print(f"    Warning: no valid time series data for station {station_id}, skip.")
            continue

        metric_row = metrics_lookup.loc[station_id] if station_id in metrics_lookup.index else pd.Series(dtype=float)
        stat_text = (
            f"RMSE = {_format_metric_value(metric_row.get('RMSE', np.nan))}\n"
            f"BIAS = {_format_metric_value(metric_row.get('BIAS', np.nan))}\n"
            f"TCC = {_format_metric_value(metric_row.get('TCC', np.nan), '.3f')}"
        )

        fig, ax = plt.subplots(figsize=(11.5, 4.2))
        ax.plot(plot_df.index, plot_df["OBS"], color="black", lw=0.8, alpha=0.85, label="OBS")
        ax.plot(plot_df.index, plot_df["Model"], color="#2F6B9A", lw=0.8, alpha=0.82, label=MODEL_LABEL)
        ax.set_title(f"{case} {var}: Station {station_id} {scale_label}")
        ax.set_xlabel("Date" if scale_label == "Daily" else "Season year")
        ax.set_ylabel(var_label.get(var, var))
        ax.grid(True, ls=":", lw=0.55, alpha=0.55)
        ax.legend(frameon=False, loc="upper left", ncol=2)
        ax.text(
            0.985, 0.965, stat_text,
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.88, "boxstyle": "round,pad=0.35"},
        )
        fig.autofmt_xdate()
        savepath = f"{save_dir}/{station_id}{suffix}.{FIGFMT}"
        fig.savefig(savepath, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(savepath)
        print(f"        Saved station {station_id} time series: {savepath}")

    print(f"    Saved station time series: {len(saved_paths)} files -> {save_dir}")
    return saved_paths


def plot_station_taylor_diagram(
    case,
    var,
    metrics_df,
    figoutdir,
    scale_label="Daily",
    suffix="",
    max_std_ratio=None,
    positive_corr_only=True,
    excluded_stations=EXCLUDED_PLOT_STATIONS,
):
    """基于逐站点 TCC 和 STD_RATIO 绘制归一化 Taylor 图。

    默认仅绘制正相关区域（R > 0），并使用笛卡尔坐标手绘 Taylor 图。
    """
    print(f"    ---- Plotting station Taylor diagram: {case}, {var}, {scale_label} ----")
    required_cols = {"TCC", "STD_RATIO"}
    missing_cols = required_cols - set(metrics_df.columns)
    if missing_cols:
        print(f"    Warning: missing Taylor metrics {sorted(missing_cols)}, skip Taylor diagram.")
        return None

    plot_df = metrics_df.copy()
    if "station_id" in plot_df.columns:
        plot_df["station_id"] = plot_df["station_id"].astype(str)
        plot_df = plot_df.loc[~plot_df["station_id"].isin({str(station) for station in excluded_stations})]
    plot_df["TCC"] = pd.to_numeric(plot_df["TCC"], errors="coerce")
    plot_df["STD_RATIO"] = pd.to_numeric(plot_df["STD_RATIO"], errors="coerce")
    if "N" in plot_df.columns:
        plot_df["N"] = pd.to_numeric(plot_df["N"], errors="coerce")
        plot_df = plot_df.loc[plot_df["N"] >= 2]
    else:
        plot_df["N"] = 1

    plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["TCC", "STD_RATIO"])
    plot_df = plot_df.loc[plot_df["STD_RATIO"] > 0]
    rmax = 3.0 if max_std_ratio is None else float(max_std_ratio)
    plot_df = plot_df.loc[plot_df["STD_RATIO"] <= rmax]
    if positive_corr_only:
        plot_df = plot_df.loc[plot_df["TCC"] > 0]
    else:
        plot_df = plot_df.loc[plot_df["TCC"].between(-1.0, 1.0, inclusive="both")]
    if plot_df.empty:
        corr_note = " with R > 0" if positive_corr_only else ""
        print(f"    Warning: no valid Taylor points{corr_note} for {case}_{var} {scale_label}.")
        return None
    print(f"        Taylor points: {len(plot_df)}, rmax: {rmax}")

    corr = plot_df["TCC"].clip(-1.0, 1.0).to_numpy(dtype=float)
    std_ratio = plot_df["STD_RATIO"].to_numpy(dtype=float)
    theta = np.arccos(corr)

    theta_max = np.pi / 2 if positive_corr_only else np.pi
    fig, ax = plt.subplots(figsize=(8.1, 7.5))
    fig.subplots_adjust(left=0.10, right=0.95, bottom=0.11, top=0.91)

    xmin = 0.0 if positive_corr_only else -rmax
    ax.set_xlim(xmin, rmax)
    ax.set_ylim(0.0, rmax)
    ax.set_aspect("equal", adjustable="box")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_color("0.1")
    ax.spines["left"].set_color("0.1")

    radius_ticks = _make_radius_ticks(rmax)
    ax.set_xticks(radius_ticks)
    ax.set_yticks(radius_ticks)
    ax.set_xticklabels([_format_numeric_tick(tick) for tick in radius_ticks])
    ax.set_yticklabels([_format_numeric_tick(tick) for tick in radius_ticks])
    ax.tick_params(axis="both", direction="in", length=6, width=1.0, labelsize=12)

    theta_grid = np.linspace(0.0, theta_max, 800)
    reference_radius = 1.0

    for radius in radius_ticks[1:]:
        x_circle = radius * np.cos(theta_grid)
        y_circle = radius * np.sin(theta_grid)
        is_reference_circle = abs(radius - reference_radius) < 1.0e-8
        circle_style = "-" if is_reference_circle else (":" if abs((radius / 0.5) - round(radius / 0.5)) < 1.0e-8 else "-")
        circle_color = "0.1" if is_reference_circle else "0.35"
        circle_lw = 1.6 if is_reference_circle else 0.9
        ax.plot(x_circle, y_circle, linestyle=circle_style, color=circle_color, linewidth=circle_lw, zorder=1)

    if positive_corr_only:
        corr_ticks = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0])
    else:
        corr_ticks = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0])
    corr_ticks = corr_ticks[(corr_ticks >= (-1.0 if not positive_corr_only else 0.0)) & (corr_ticks <= 1.0)]
    for corr_tick in corr_ticks:
        tick_theta = math.acos(float(np.clip(corr_tick, -1.0, 1.0)))
        if tick_theta > theta_max + 1.0e-12:
            continue
        x_ray = np.array([0.0, rmax * math.cos(tick_theta)])
        y_ray = np.array([0.0, rmax * math.sin(tick_theta)])
        ax.plot(x_ray, y_ray, color="0.87", linewidth=0.8, zorder=0)

        x_outer = rmax * math.cos(tick_theta)
        y_outer = rmax * math.sin(tick_theta)
        tick_dx = -0.025 * rmax * math.cos(tick_theta)
        tick_dy = -0.025 * rmax * math.sin(tick_theta)
        ax.plot([x_outer, x_outer + tick_dx], [y_outer, y_outer + tick_dy], color="0.1", linewidth=1.0, zorder=3)

        label_r = rmax * 1.04
        x_label = label_r * math.cos(tick_theta)
        y_label = label_r * math.sin(tick_theta)
        rotation = math.degrees(tick_theta) - 90.0
        ax.text(
            x_label,
            y_label,
            _format_corr_tick(float(corr_tick)),
            ha="center",
            va="center",
            rotation=rotation,
            rotation_mode="anchor",
            fontsize=12,
        )

    outer_x = rmax * np.cos(theta_grid)
    outer_y = rmax * np.sin(theta_grid)
    ax.plot(outer_x, outer_y, color="0.1", linewidth=1.2, zorder=3)
    ax.plot([xmin, rmax], [0.0, 0.0], color="0.1", linewidth=1.2, zorder=3)
    ax.plot([0.0, 0.0], [0.0, rmax], color="0.1", linewidth=1.2, zorder=3)

    grid_points = 600
    xs = np.linspace(xmin, rmax, grid_points)
    ys = np.linspace(0.0, rmax, grid_points)
    xx, yy = np.meshgrid(xs, ys)
    rr = np.sqrt(xx**2 + yy**2)
    mask = (rr > rmax + 1.0e-10) | (yy < 0.0)
    crmse_norm = np.sqrt((xx - reference_radius) ** 2 + yy**2)
    crmse_norm = np.ma.masked_where(mask, crmse_norm)

    contour_levels = np.arange(0.5, rmax + 0.001, 0.5)
    ax.contour(xx, yy, crmse_norm, levels=contour_levels, colors="0.82", linewidths=0.9, zorder=0)

    crmsd_label_angle = math.radians(100.0)
    crmsd_label_rotation = math.degrees(crmsd_label_angle)
    for level in contour_levels:
        label_x = reference_radius + level * math.cos(crmsd_label_angle)
        label_y = level * math.sin(crmsd_label_angle)
        if label_x < xmin or label_x > rmax or label_y < 0.0 or label_y > rmax:
            continue
        if math.hypot(label_x, label_y) > rmax + 1.0e-10:
            continue
        ax.text(
            label_x,
            label_y,
            f"{level:.1f}",
            ha="center",
            va="center",
            rotation=crmsd_label_rotation,
            rotation_mode="anchor",
            fontsize=9,
            color="0.65",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.2},
            zorder=2,
        )
    ax.text(reference_radius * 0.50, min(rmax * 0.72, reference_radius * 1.08), "CRMSD", color="0.72", fontsize=9, rotation=28)

    ax.scatter([reference_radius], [0.0], color="k", marker="*", s=260, zorder=6, clip_on=False)
    x_point = std_ratio * np.cos(theta)
    y_point = std_ratio * np.sin(theta)
    ax.scatter(
        x_point,
        y_point,
        s=42,
        color="#2F6B9A",
        alpha=0.72,
        edgecolors="white",
        linewidths=0.55,
        zorder=7,
        label=MODEL_LABEL,
    )

    ax.set_xlabel("Normalized standard deviation", fontsize=13)
    ax.set_ylabel("")
    fig.suptitle(f"{case} {var}: Station {scale_label} Taylor Diagram", fontsize=13, y=0.98)

    corr_theta = math.acos(0.68 if positive_corr_only else 0.75)
    corr_r = rmax * 1.10
    ax.text(
        corr_r * math.cos(corr_theta),
        corr_r * math.sin(corr_theta),
        "Correlation",
        ha="center",
        va="center",
        rotation=math.degrees(corr_theta) - 90.0,
        rotation_mode="anchor",
        fontsize=13,
    )

    legend_handles = [
        Line2D([0], [0], marker="*", color="k", lw=0, markersize=14, label="OBS"),
        Line2D([0], [0], marker="o", color="#2F6B9A", markerfacecolor="#2F6B9A", markeredgecolor="white", lw=0, markersize=8, label=MODEL_LABEL),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.01),
        borderaxespad=0.0,
        frameon=True,
        title="Experiment",
        fontsize=9,
        title_fontsize=9,
    )

    save_dir = _make_figure_type_dir(figoutdir, "Taylor_Diagram", case, var)
    savepath = f"{save_dir}/{case}_{var}{suffix}_Taylor_Diagram.{FIGFMT}"
    fig.savefig(savepath, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved station Taylor diagram: {savepath}")
    return savepath


def plot_station_spatial_map(case, var, metrics_df, metric, mapcfg, label, savepath):
    """绘制站点指标空间分布图，三角形位置为站点经纬度，颜色为指定指标。"""
    metric_data = _metric_for_data(var, metric)
    metric_col = _find_metric_col(metrics_df, metric_data)
    required_cols = {"station_id", "lat", "lon", metric_col}
    missing_cols = required_cols - set(metrics_df.columns)
    if missing_cols:
        print(f"    Warning: missing spatial {metric} ({metric_data}) map columns {sorted(missing_cols)}, skip map.")
        return None

    plot_df = metrics_df.copy()
    plot_df["station_id"] = plot_df["station_id"].astype(str)
    plot_df = plot_df.loc[~plot_df["station_id"].isin({str(station) for station in EXCLUDED_PLOT_STATIONS})]
    for col in ["lat", "lon", metric_col]:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["lat", "lon", metric_col])
    if plot_df.empty:
        print(f"    Warning: no valid station {metric} ({metric_data}) values for {case}_{var}, skip spatial map.")
        return None

    metric_values = plot_df[metric_col].to_numpy(dtype=float)
    fig = plt.figure(figsize=(9, 6), layout='constrained')
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal(central_longitude=114.3, central_latitude=29.5))

    # Inline from Plot_Yangtze_BaseMap_Lambert for the current station-map call:
    # data_in=None and sign_mask=None, so gridded shading and significance marks are not executed.
    ax.set_extent([106.2, 123.2, 24.2, 34.15], crs=ccrs.PlateCarree())
    ax.add_feature(
        cfeature.OCEAN,
        facecolor=ax.get_facecolor(),
        edgecolor="none",
        zorder=15,
    )
    taiwan_gdf = gpd.read_file(f"{TPC.BASEDATA}/Taiwan.gpkg").set_crs("EPSG:4326", allow_override=True)
    ax.add_geometries(
        taiwan_gdf.geometry,
        crs=ccrs.PlateCarree(),
        edgecolor="none",
        facecolor=".9",
        linewidth=0.4,
        zorder=30,
    )
    water_gdf = gpd.read_file(f"{TPC.BASEDATA}/SC_WATER.gpkg").set_crs("EPSG:4326", allow_override=True)
    ax.add_geometries(
        water_gdf.geometry,
        crs=ccrs.PlateCarree(),
        edgecolor="black",
        facecolor="none",
        linewidth=0.4,
        zorder=30,
    )
    draw_maps(
        get_adm_maps(country="中华人民共和国"),
        ax=ax,
        color="k",
        zorder=49,
        linewidth=0.3,
    )
    if label is not None:
        ax.text(
            0.98,
            0.98,
            f"{label}",
            ha="right",
            va="top",
            fontsize=22,
            transform=ax.transAxes,
            fontweight="bold",
            zorder=30,
        )
    ax.set_aspect("auto")

    china = get_adm_maps(country="中华人民共和国", level="国", record="first", only_polygon=True)
    ax.add_geometries(china, crs=ccrs.PlateCarree(), facecolor=".9", edgecolor=None, zorder=29)
    ax.scatter(
        plot_df["lon"].to_numpy(dtype=float),
        plot_df["lat"].to_numpy(dtype=float),
        c=metric_values,
        s=132,
        marker="^",
        cmap=mapcfg.cmap,
        vmin=float(mapcfg.levs[0][0]),
        vmax=float(mapcfg.levs[0][-1]),
        edgecolors="black",
        linewidths=0.35,
        alpha=0.92,
        transform=ccrs.PlateCarree(),
        zorder=80,
    )

    fig.savefig(savepath, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"        Saved station {metric} map: {savepath}")
    return savepath


def plot_station_metric_spatial_maps(
    case,
    var,
    figoutdir,
    outdir,
    metric="RMSE",
    n_jobs=16,
    share_colorbar=True,
):
    """绘制各季节站点指标空间分布图。"""
    target = "StationMetricsMap"
    seasons = TU.get_seasons()
    var_info = TYCM.Variable_Infos(var)
    FigOutDir_var = f'{figoutdir}/{target}/{case}/{var}/{metric}'
    os.makedirs(FigOutDir_var, exist_ok=True)
    varInfo = TPC.varInfo(longname=var_info['longname'], unit=var_info['bunit'], abbr=var_info['abbr'])
    label = f"{varInfo.longname} ({varInfo.unit})"
    tasks = []
    cbar_tasks = []
    cbar_modes = {}
    print(f"    ---- Plotting seasonal station {metric} spatial maps: {case}, {var} ----")
    for season in seasons:
        for mode_token in ["SeasonalMean"]: #, "SeasonalDaily"
            metrics_path = f"{outdir}/{case}_{var}_{mode_token}_{season}_StationMetrics.csv"
            print(f"        Reading seasonal metrics: {metrics_path}")
            metrics_df = read_station_metrics(case, var, outdir, suffix=f"_{mode_token}_{season}")
            mapcfg = _station_metric_mapcfg(var, metric, mode=mode_token)
            map_savepath = f"{FigOutDir_var}/{case}_{var}_{mode_token}_{season}_Station_Map_{metric}.{FIGFMT}"
            tasks.append((plot_station_spatial_map, (case, var, metrics_df, metric, mapcfg, season, map_savepath)))
            cbar_modes[mode_token] = mapcfg
            if not share_colorbar:
                colorbar_path = f"{FigOutDir_var}/{case}_{var}_{mode_token}_{season}_Station_Map_Colorbar_{metric}.{FIGFMT}"
                cbar_tasks.append((TPCB.plot_spatial_cbar_core_V, (mapcfg, 6, label, colorbar_path, 18, 18, 0.05, 'both','{:5.1f}')))
    mode_token = "Daily"
    metrics_path = f"{outdir}/{case}_{var}_StationMetrics.csv"
    print(f"        Reading daily metrics: {metrics_path}")
    metrics_df = read_station_metrics(case, var, outdir)
    mapcfg = _station_metric_mapcfg(var, metric, mode=mode_token)
    map_savepath = f"{FigOutDir_var}/{case}_{var}_{mode_token}_Station_Map_{metric}.{FIGFMT}"
    tasks.append((plot_station_spatial_map, (case, var, metrics_df, metric, mapcfg, mode_token, map_savepath)))
    cbar_modes[mode_token] = mapcfg
    if not share_colorbar:
        colorbar_path = f"{FigOutDir_var}/{case}_{var}_{mode_token}_Station_Map_Colorbar_{metric}.{FIGFMT}"
        cbar_tasks.append((TPCB.plot_spatial_cbar_core_V, (mapcfg, 6, label, colorbar_path, 18, 18, 0.05, 'both','{:5.1f}')))

    if share_colorbar:
        for mode_token, mapcfg in cbar_modes.items():
            colorbar_path = f"{FigOutDir_var}/{case}_{var}_{mode_token}_Station_Map_Colorbar_{metric}.{FIGFMT}"
            cbar_tasks.append((TPCB.plot_spatial_cbar_core_V, (mapcfg, 6, label, colorbar_path, 18, 18, 0.05, 'both','{:5.1f}')))
    tasks.extend(cbar_tasks)

    # 并行执行
    ntasks = len(tasks)
    print(f"    Parallel station map tasks: {ntasks} (maps: {ntasks - len(cbar_tasks)}, colorbars: {len(cbar_tasks)}, n_jobs: {n_jobs})")
    with Parallel(n_jobs=n_jobs, backend="loky",  pre_dispatch="2*n_jobs", return_as="generator") as parallel:
        gen = parallel(
            delayed(func)(*args) for func, args in tasks
        )
        res_list = [p for p in tqdm(gen, total= ntasks,
                                        desc=f"    ➠ Parallel", unit="task",
                                        dynamic_ncols=True)]
    print("    All parallel station maps done.")


def _station_metric_values_for_box(metrics_df, var, metric, excluded_stations=EXCLUDED_PLOT_STATIONS):
    """整理箱型图所需的一维站点指标数组。"""
    metric_col = _find_metric_col(metrics_df, metric, var=var)
    plot_df = metrics_df.copy()
    if "station_id" in plot_df.columns:
        plot_df["station_id"] = plot_df["station_id"].astype(str)
        plot_df = plot_df.loc[~plot_df["station_id"].isin({str(station) for station in excluded_stations})]
    values = pd.to_numeric(plot_df[metric_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return values.dropna().to_numpy(dtype=float)


def _read_station_reference_metrics(reference_file, case, var, metric):
    """读取 SeasonMean 箱型图右侧参考指标。"""
    if reference_file is None or not os.path.exists(reference_file):
        return {}

    rows = []
    with open(reference_file, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        header = [col.strip() for col in header]
        for row in reader:
            if not row or all(str(item).strip() == "" for item in row):
                continue
            # 兼容误写成 Case,Var,Metric,Season,,CMFDv2,ERA5-Land 的数据行
            if len(row) == len(header) + 1 and len(row) > 4 and str(row[4]).strip() == "":
                row = row[:4] + row[5:]
            if len(row) < len(header):
                row = row + [""] * (len(header) - len(row))
            if len(row) > len(header):
                row = row[:len(header)]
            rows.append(row)
    ref_df = pd.DataFrame(rows, columns=header)
    colmap = {str(col).strip().lower(): col for col in ref_df.columns}
    required_cols = ["var", "metric", "season"]
    missing_cols = [col for col in required_cols if col not in colmap]
    if missing_cols:
        raise ValueError(f"{reference_file} 缺少必要列: {missing_cols}")

    var_col = colmap["var"]
    metric_col = colmap["metric"]
    season_col = colmap["season"]
    case_col = colmap.get("case", None)
    cmfd_col = colmap.get("cmfdv2", None)
    era5_col = colmap.get("era5-land", None)

    work_df = ref_df.copy()
    mask = (
        work_df[var_col].astype(str).str.upper().eq(str(var).upper())
        & work_df[metric_col].astype(str).str.upper().eq(str(metric).upper())
    )
    if case_col is not None:
        case_values = work_df[case_col].fillna("All").astype(str)
        mask = mask & (case_values.str.upper().isin(["ALL", str(case).upper()]))
    work_df = work_df.loc[mask]

    ref_points = {}
    for _, row in work_df.iterrows():
        season = str(row[season_col]).upper()
        season_points = {}
        if cmfd_col is not None and pd.notna(row[cmfd_col]):
            season_points["CMFDv2"] = float(row[cmfd_col])
        if era5_col is not None and pd.notna(row[era5_col]):
            season_points["ERA5-Land"] = float(row[era5_col])
        if season_points:
            ref_points[season] = season_points
    return ref_points


def _plot_station_metric_boxplot_core(case, var, metric, box_data, labels, mode_label, ylabel, savepath, boxlevs=None, ref_points=None):
    """绘制单张站点指标箱型图。"""
    if not any(len(values) > 0 for values in box_data):
        print(f"    Warning: no valid {metric} values for {case}_{var}_{mode_label}, skip boxplot.")
        return None

    fig = plt.figure(figsize=(5.2, 6))
    # Fixed axes rectangle: keeps the boxplot frame itself identical across metrics.
    ax = fig.add_axes([0.24, 0.12, 0.70, 0.83])
    # colors =  ["#B8B8B8"] + ["#A5BFE4"] * (len(labels) - 1)   #
    colors =  ["#A5BFE4"] * len(labels)
    box = ax.boxplot(
        box_data,
        labels=labels,
        widths=0.45,
        patch_artist=True,
        showfliers=False,
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "black",
            "markeredgecolor": "black",
            "markersize": 4.5,
        },
        medianprops={"color": "black", "linewidth": 1.2},
        whiskerprops={"color": "black", "linewidth": 1.0},
        capprops={"color": "black", "linewidth": 1.0},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.0)
        patch.set_alpha(0.88)

    if str(metric).upper() in {"BIAS", "RBAIS"}:
        ax.axhline(0, color="black", linestyle="--", linewidth=1.25, alpha=0.75, zorder=1)
    if boxlevs is not None:
        boxlevs = np.asarray(boxlevs, dtype=float)
        boxlevs = boxlevs[np.isfinite(boxlevs)]
        if boxlevs.size >= 2:
            ax.set_ylim(float(boxlevs[0]), float(boxlevs[-1]))
            ax.set_yticks(boxlevs)
            ax.set_yticklabels([_format_numeric_tick(tick) for tick in boxlevs])
    if ref_points:
        nref = 0
        for ipos, label_name in enumerate(labels, start=1):
            season = str(label_name).upper()
            if season == "DAILY" or season not in ref_points:
                continue
            if "CMFDv2" in ref_points[season]:
                ax.scatter(ipos + 0.38, ref_points[season]["CMFDv2"],
                           marker="*", s=120, color="#D7191C", edgecolors="black", linewidths=0.35,
                           zorder=10, clip_on=False)
                nref += 1
            if "ERA5-Land" in ref_points[season]:
                ax.scatter(ipos + 0.38, ref_points[season]["ERA5-Land"],
                           marker="^", s=72, color="#FFDF20", edgecolors="black", linewidths=0.35,
                           zorder=10, clip_on=False)
                nref += 1
        ax.set_xlim(0.5, len(labels) + 0.75)
        handles = [
            Line2D([0], [0], marker="*", color="none", markerfacecolor="#D7191C",
                   markeredgecolor="black", markersize=10, label="CMFDv2"),
            Line2D([0], [0], marker="^", color="none", markerfacecolor="#FFDF20",
                   markeredgecolor="black", markersize=8, label="ERA5-Land"),
        ]
        ax.legend(handles=handles, frameon=False, loc="best", fontsize=16)
        print(f"        Plot reference markers: {nref}")
    ax.set_ylabel(ylabel, fontweight="bold", fontsize=18)
    # ax.set_title(f"{case} {var}: Station {metric} ({mode_label})", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.55)
    ax.tick_params(axis="both", labelsize=16)

    # Keep the saved canvas fixed. Do not use bbox_inches="tight" here; it changes
    # the apparent axes-frame size when labels or legends differ across metrics.
    fig.savefig(savepath, dpi=DPI)
    plt.close(fig)
    print(f"    Saved station {metric} boxplot: {savepath}")
    return savepath


def plot_station_metric_boxplots(
    case,
    var,
    figoutdir,
    outdir,
    metric="RMSE",
    seasons=None,
    reference_file=None,
):
    """
    绘制站点指标箱型图。

    每个指标输出两张图：
    1) Daily + SeasonalMean
    2) Daily + SeasonalDaily
    """
    target = "StationMetricsBox"
    seasons = TU.get_seasons() if seasons is None else seasons
    var_info = TYCM.Variable_Infos(var)
    varInfo = TPC.varInfo(longname=var_info['longname'], unit=var_info['bunit'], abbr=var_info['abbr'])
    ylabel = f"{metric} ({varInfo.unit})" if str(metric).upper() in {"RMSE", "BIAS", "MAE", "CRESM", "NMAE"} else str(metric)
    FigOutDir_var = f'{figoutdir}/{target}/{case}/{var}/{metric}'
    os.makedirs(FigOutDir_var, exist_ok=True)
    print(f"    ---- Plotting station {metric} boxplots: {case}, {var} ----")

    daily_path = f"{outdir}/{case}_{var}_StationMetrics.csv"
    print(f"        Reading daily metrics: {daily_path}")
    daily_df = read_station_metrics(case, var, outdir)
    metric_data = _metric_for_data(var, metric)
    if str(metric_data).upper() != str(metric).upper():
        print(f"        Metric column mapping: {var} {metric} -> {metric_data}")
    daily_values = _station_metric_values_for_box(daily_df, var, metric)
    saved_paths = []

    for mode_token, mode_label in [("SeasonalMean", "Daily_vs_SeasonalMean"),
                                   ]: #("SeasonalDaily", "Daily_vs_SeasonalDaily")
        # box_data = [daily_values]
        # labels = ["Daily"]
        box_data = []
        labels = []
        boxlevs = _station_metric_boxlevs(var, metric, mode=mode_token)
        ref_points = {}
        if mode_token == "SeasonalMean":
            ref_points = _read_station_reference_metrics(reference_file, case, var, metric)
            if ref_points:
                print(f"        Loaded reference metrics for {var} {metric}: {sorted(ref_points.keys())}")
        for season in seasons:
            metrics_path = f"{outdir}/{case}_{var}_{mode_token}_{season}_StationMetrics.csv"
            if not os.path.exists(metrics_path):
                print(f"    Warning: missing seasonal metrics, skip {mode_token} {season} box: {metrics_path}")
                box_data.append(np.array([], dtype=float))
                labels.append(season)
                continue
            print(f"        Reading seasonal metrics: {metrics_path}")
            metrics_df = read_station_metrics(case, var, outdir, suffix=f"_{mode_token}_{season}")
            box_data.append(_station_metric_values_for_box(metrics_df, var, metric))
            labels.append(season)

        savepath = f"{FigOutDir_var}/{case}_{var}_{mode_label}_Station_Box_{metric}.{FIGFMT}"
        path = _plot_station_metric_boxplot_core(case, var, metric, box_data, labels, mode_label, ylabel, savepath,
                                                boxlevs=boxlevs, ref_points=ref_points)
        if path is not None:
            saved_paths.append(path)

    print(f"    Saved station {metric} boxplots: {len(saved_paths)} files")
    return saved_paths


def Merge_Plot_Station_Validation(
    case,
    figoutdir,
    rows_config=None,
    map_mode="Daily",
    box_mode_label="Daily_vs_SeasonalMean",
    output_name=None,
):
    """
    合并站点验证关键图。

    图形类型固定为 2 行 x 3 列：
    每行依次为：空间图(+右侧 colorbar), boxplot, boxplot。

    rows_config 用于选择每行放置哪些变量和指标，例如：
        rows_config = [
            {"var": "T2m", "map_metric": "MAE", "box_metrics": ["RMSE", "MAE"]},
            {"var": "Prec", "map_metric": "MAE", "box_metrics": ["RMSE", "MAE"]},
        ]

    默认保持原图：T2m/Prec 的 Daily MAE 空间图和 RMSE/MAE box。
    """
    if rows_config is None:
        rows_config = [
            {"var": "T2m", "map_metric": "MAE",  "box_metrics": ["RMSE", "Bias"]},
            {"var": "Prec", "map_metric": "MAE", "box_metrics": ["RMSE", "Bias"]},
        ]
    if len(rows_config) != 2:
        raise ValueError("rows_config must contain exactly two row configs for a 2-row figure.")

    normalized_rows = []
    for irow, row_cfg in enumerate(rows_config, start=1):
        if not isinstance(row_cfg, dict):
            raise TypeError(f"rows_config[{irow - 1}] must be a dict.")
        var = row_cfg.get("var")
        map_metric = row_cfg.get("map_metric")
        box_metrics = row_cfg.get("box_metrics")
        if var is None or map_metric is None or box_metrics is None:
            raise ValueError(
                f"rows_config[{irow - 1}] must define 'var', 'map_metric', and 'box_metrics'."
            )
        if len(box_metrics) != 2:
            raise ValueError(f"rows_config[{irow - 1}]['box_metrics'] must contain exactly two metrics.")
        normalized_rows.append({
            "var": str(var),
            "map_metric": str(map_metric),
            "box_metrics": [str(box_metrics[0]), str(box_metrics[1])],
        })

    target_map = "StationMetricsMap"
    target_box = "StationMetricsBox"
    FigOutDir_out = f"{figoutdir}/Manuscript"
    os.makedirs(FigOutDir_out, exist_ok=True)

    map_crop = {"left": 0.01, "top": 0.01, "right": 0.01, "bottom": 0.01}
    cbar_crop = {"left": 0.01, "top": 0.018, "right": 0.02, "bottom": 0.018}
    box_crop = {"left": 0.01, "top": 0.01, "right": 0.01, "bottom": 0.01}

    def _check_image(path_in):
        if not os.path.exists(path_in):
            raise FileNotFoundError(f"Required station validation panel image not found: {path_in}")
        return path_in

    rows = []
    for row_cfg in normalized_rows:
        var = row_cfg["var"]
        map_metric = row_cfg["map_metric"]
        box_metric_1, box_metric_2 = row_cfg["box_metrics"]

        map_path = _check_image(
            f"{figoutdir}/{target_map}/{case}/{var}/{map_metric}/"
            f"{case}_{var}_{map_mode}_Station_Map_{map_metric}.{FIGFMT}"
        )
        cbar_path = _check_image(
            f"{figoutdir}/{target_map}/{case}/{var}/{map_metric}/"
            f"{case}_{var}_{map_mode}_Station_Map_Colorbar_{map_metric}.{FIGFMT}"
        )
        box_path_1 = _check_image(
            f"{figoutdir}/{target_box}/{case}/{var}/{box_metric_1}/"
            f"{case}_{var}_{box_mode_label}_Station_Box_{box_metric_1}.{FIGFMT}"
        )
        box_path_2 = _check_image(
            f"{figoutdir}/{target_box}/{case}/{var}/{box_metric_2}/"
            f"{case}_{var}_{box_mode_label}_Station_Box_{box_metric_2}.{FIGFMT}"
        )

        map_img = TIT.crop_image_from_path(map_path, crop_params=map_crop, mode="ratio")
        cbar_img = TIT.crop_image_from_path(cbar_path, crop_params=cbar_crop, mode="ratio")
        cbar_img = TIT.adjust_image_to_ref_canvas(
            target_img=cbar_img,
            ref_img=map_img,
            axis="height",
        )
        map_panel = TIT.merge_images_Row(
            rows_images=[[map_img, cbar_img]],
            cols_space=[[0.005]],
            rows_space=[],
            box_space={"left": 0, "top": 0, "right": 0, "bottom": 0},
            background_color="#FFFFFF",
            space_mode="ratio",
            alignment=["left"],
            draw_ticks=False,
            font_path="/home/wumej22/.local/share/fonts/NotoSans-Bold.ttf",
        )

        box_img_1 = TIT.crop_image_from_path(box_path_1, crop_params=box_crop, mode="ratio")
        box_img_2 = TIT.crop_image_from_path(box_path_2, crop_params=box_crop, mode="ratio")
        box_img_1 = TIT.adjust_image_to_ref_canvas(
            target_img=box_img_1,
            ref_img=map_panel,
            axis="height",
        )
        box_img_2 = TIT.adjust_image_to_ref_canvas(
            target_img=box_img_2,
            ref_img=map_panel,
            axis="height",
        )
        rows.append([map_panel, box_img_1, box_img_2])

    panel_texts = {
        "label1": {"x": 0.020, "y": 0.25, "fontsize": 0.04, "text": "2-m air temperature", "ha": "center", "va": "center", "color": "black", "fontweight": "bold", "rotation": -90},
        "label2": {"x": 0.020, "y": 0.75, "fontsize": 0.04, "text": "Precipitation", "ha": "center", "va": "center", "color": "black", "fontweight": "bold", "rotation": -90},
        "(a)": {"x": 0.045, "y": 0.015, "fontsize": 0.032, "text": "(a)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        "(b)": {"x": 0.585, "y": 0.030, "fontsize": 0.032, "text": "(b)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        "(c)": {"x": 0.822, "y": 0.030, "fontsize": 0.032, "text": "(c)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        "(d)": {"x": 0.045, "y": 0.515, "fontsize": 0.032, "text": "(d)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        "(e)": {"x": 0.585, "y": 0.530, "fontsize": 0.032, "text": "(e)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
        "(f)": {"x": 0.822, "y": 0.530, "fontsize": 0.032, "text": "(f)", "ha": "left", "va": "top", "color": "black", "fontweight": "bold"},
    }

    big_img = TIT.merge_images_Row(
        rows_images=rows,
        cols_space=[[0.010, 0.010], [0.010, 0.010]],
        rows_space=[0.02],
        box_space={"left": 0.08, "top": 0.01, "right": 0.01, "bottom": 0.01},
        background_color="#FFFFFF",
        space_mode="ratio",
        alignment=["left", "left"],
        draw_ticks=False,
        tick_step=0.01,
        texts=panel_texts,
        font_path="/home/wumej22/.local/share/fonts/NotoSans-Bold.ttf",
    )

    if output_name is None:
        row_tokens = []
        for row_cfg in normalized_rows:
            row_tokens.append(
                f"{row_cfg['var']}_{map_mode}Map{row_cfg['map_metric']}_"
                f"Box{'-'.join(row_cfg['box_metrics'])}"
            )
        output_name = f"{case}_StationValidation_{'_'.join(row_tokens)}"
    savepath_png = f"{FigOutDir_out}/{output_name}.{FIGFMT}"
    TIT.save(big_img, savepath_png, dpi=DPI)
    savepath_pdf = f"{FigOutDir_out}/{output_name}.pdf"
    TIT.save(big_img, savepath_pdf, dpi=DPI, page_size="original")
    print(f"    Saved station validation merged figure: {savepath_png}")
    print(f"    Saved station validation merged figure: {savepath_pdf}")
    return savepath_png, savepath_pdf



def plot_station_seasonal_scatters(
    case,
    var,
    case_df,
    obs_df,
    figoutdir,
    outdir,
    seasons=Season_List,
    seasonal_mode="seasonal_mean",
):
    """绘制季节尺度观测-模式散点图。"""
    _, mode_token, mode_label = _seasonal_mode_info(seasonal_mode)
    print(f"    ---- Plotting seasonal station scatters: {case}, {var}, {mode_label} ----")
    case_seasonal, obs_seasonal = _get_seasonal_data(case_df, obs_df, seasons=seasons, seasonal_mode=seasonal_mode)
    saved_count = 0
    for season in seasons:
        metrics_path = f"{outdir}/{case}_{var}_{mode_token}_{season}_StationMetrics.csv"
        if not os.path.exists(metrics_path):
            print(f"    Warning: missing seasonal metrics, skip scatter: {metrics_path}")
            continue
        print(f"        Reading seasonal metrics: {metrics_path}")
        metrics_df = read_station_metrics(case, var, outdir, suffix=f"_{mode_token}_{season}")
        plot_station_obs_model_scatter(
            case, var,
            case_seasonal[season], obs_seasonal[season], metrics_df, figoutdir,
            scale_label=f"{mode_label} {season}",
            suffix=f"_{mode_token}_{season}",
        )
        saved_count += 1
    print(f"    Saved seasonal station scatters: {saved_count} files")


def plot_station_seasonal_timeseries(
    case,
    var,
    case_df,
    obs_df,
    figoutdir,
    outdir,
    seasons=Season_List,
    seasonal_mode="seasonal_mean",
):
    """绘制与季节尺度指标计算一致的逐站点时间序列。"""
    _, mode_token, mode_label = _seasonal_mode_info(seasonal_mode)
    print(f"    ---- Plotting seasonal station time series: {case}, {var}, {mode_label} ----")
    case_seasonal, obs_seasonal = _get_seasonal_data(case_df, obs_df, seasons=seasons, seasonal_mode=seasonal_mode)
    saved_paths = []
    for season in seasons:
        metrics_path = f"{outdir}/{case}_{var}_{mode_token}_{season}_StationMetrics.csv"
        if not os.path.exists(metrics_path):
            print(f"    Warning: missing seasonal metrics, skip time series: {metrics_path}")
            continue
        print(f"        Reading seasonal metrics: {metrics_path}")
        metrics_df = read_station_metrics(case, var, outdir, suffix=f"_{mode_token}_{season}")
        paths = plot_station_timeseries(
            case,
            var,
            case_seasonal[season],
            obs_seasonal[season],
            metrics_df,
            figoutdir,
            suffix=f"_{mode_token}_{season}",
            scale_label=f"{mode_label} {season}",
        )
        saved_paths.extend(paths)
    print(f"    Saved seasonal station time series: {len(saved_paths)} files")
    return saved_paths


def plot_station_seasonal_taylor_diagrams(
    case,
    var,
    figoutdir,
    outdir,
    seasons=Season_List,
    seasonal_mode="seasonal_mean",
):
    """绘制各季节验证指标对应的 Taylor 图。"""
    _, mode_token, mode_label = _seasonal_mode_info(seasonal_mode)
    print(f"    ---- Plotting seasonal station Taylor diagrams: {case}, {var}, {mode_label} ----")
    saved_count = 0
    for season in seasons:
        metrics_path = f"{outdir}/{case}_{var}_{mode_token}_{season}_StationMetrics.csv"
        if not os.path.exists(metrics_path):
            print(f"    Warning: missing seasonal metrics, skip Taylor diagram: {metrics_path}")
            continue
        print(f"        Reading seasonal metrics: {metrics_path}")
        metrics_df = read_station_metrics(case, var, outdir, suffix=f"_{mode_token}_{season}")
        path = plot_station_taylor_diagram(
            case,
            var,
            metrics_df,
            figoutdir,
            scale_label=f"{mode_label} {season}",
            suffix=f"_{mode_token}_{season}",
        )
        if path is not None:
            saved_count += 1
    print(f"    Saved seasonal station Taylor diagrams: {saved_count} files")
