#!/share/home/dq048/anaconda3/envs/plot_env/bin/python
import os
import sys
import time
import dask
from tqdm import tqdm
import xarray as xr
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor

# 自定义工具箱
import ToolBoxes.Utils as TU

# ========== Config ==========
BufferZone = 15  # 缓冲区宽度，单位：grid

def correct_temperature(T_obs, h_obs, h_ref, lapse_rate=6.5):
    """
    温度随地形矫正函数（适用于 3D 温度场 + 2D 高程场）
    参数
    ----
    T_obs : array-like 或 xarray.DataArray
        观测或模拟温度，三维 (time, lat, lon)，单位：°C 或 K（保持一致即可）。
    h_obs : array-like 或 xarray.DataArray
        实际观测点或网格点海拔高度，二维 (lat, lon)，单位：米。
    h_ref : array-like 或 xarray.DataArray
        参考高度，二维 (lat, lon) 或标量，单位：米。
    lapse_rate : float, 可选
        递减率，单位 °C/km，默认 6.5°C/km。
    """
    # 将递减率从 °C/km 转换为 °C/m
    gamma = lapse_rate / 1000.0  # °C per meter
    # 计算高度差场 (lat,lon)
    # 当 h_ref 为标量时，也会被正确广播
    delta_h = h_obs - h_ref  # shape (lat,lon)  or broadcastable to that
    # 计算温度修正量 (lat,lon)
    delta_T = gamma * delta_h
    # 加回到三维温度场上，利用广播机制：
    # NumPy: 会在 time 维上自动广播 (lat,lon) → (time,lat,lon)
    # xarray: 会根据坐标名自动对齐并广播
    T_ref = T_obs + delta_T
    print(f" - Correcting temperature with lapse rate: {lapse_rate} °C/km")

    return T_ref


def resample_time(ds, timefreq, rspmethod="mean"):
    if timefreq.lower() == "daily":
        if rspmethod == "mean":
            ds = ds.resample(time="D").mean(skipna=True)
        elif rspmethod == "sum":
            ds = ds.resample(time="D").sum(skipna=True)
        elif rspmethod == "max":
            ds = ds.resample(time="D").max(skipna=True)
        elif rspmethod == "min":
            ds = ds.resample(time="D").min(skipna=True)
        else:
            raise ValueError(f"Unknown rspmethod: {rspmethod}")
    elif timefreq.lower() == "monthly":
        if rspmethod == "mean":
            ds = ds.resample(time="MS").mean(skipna=True)
        elif rspmethod == "sum":
            ds = ds.resample(time="MS").sum(skipna=True)
        elif rspmethod == "max":
            ds = ds.resample(time="MS").max(skipna=True)
        elif rspmethod == "min":
            ds = ds.resample(time="MS").min(skipna=True)
        else:
            raise ValueError(f"Unknown rspmethod: {rspmethod}")
    elif timefreq.lower() == "seasonal":
        if rspmethod == "mean":
            ds = ds.resample(time="QS-DEC").mean(skipna=True)
        elif rspmethod == "sum":
            ds = ds.resample(time="QS-DEC").sum(skipna=True)
        elif rspmethod == "max":
            ds = ds.resample(time="QS-DEC").max(skipna=True)
        elif rspmethod == "min":
            ds = ds.resample(time="QS-DEC").min(skipna=True)
        else:
            raise ValueError(f"Unknown rspmethod: {rspmethod}")
    elif timefreq.lower() == "monthly_diurnal":
        if rspmethod == "mean":
            ds = ds.resample(time="MS").mean(skipna=True)
        elif rspmethod == "sum":
            ds = ds.resample(time="MS").sum(skipna=True)
        elif rspmethod == "max":
            ds = ds.resample(time="MS").max(skipna=True)
        elif rspmethod == "min":
            ds = ds.resample(time="MS").min(skipna=True)
        else:
            raise ValueError(f"Unknown rspmethod: {rspmethod}")
    elif timefreq.lower() == "seasonal_diurnal":
        if rspmethod == "mean":
            ds = ds.resample(time="QS-DEC").mean(skipna=True)
        elif rspmethod == "sum":
            ds = ds.resample(time="QS-DEC").sum(skipna=True)
        elif rspmethod == "max":
            ds = ds.resample(time="QS-DEC").max(skipna=True)
        elif rspmethod == "min":
            ds = ds.resample(time="QS-DEC").min(skipna=True)
        else:
            raise ValueError(f"Unknown rspmethod: {rspmethod}")
    elif timefreq.lower() == "yearly":
        if rspmethod == "mean":
            ds = ds.resample(time="YS").mean(skipna=True)
        elif rspmethod == "sum":
            ds = ds.resample(time="YS").sum(skipna=True)
        elif rspmethod == "max":
            ds = ds.resample(time="YS").max(skipna=True)
        elif rspmethod == "min":
            ds = ds.resample(time="YS").min(skipna=True)
        else:
            raise ValueError(f"Unknown rspmethod: {rspmethod}")

    return ds



def read_data(args):
    """读取 NetCDF 文件，并根据 resample_freq 重采样。"""
    file = args.get('file')
    bufferzone = args.get('bufferzone', BufferZone)
    level_index = args.get('level_index', None)

    # ⚠️ 用上下文管理器打开，处理完马上关闭文件句柄
    with xr.open_dataset(file, engine="h5netcdf") as ds:
        # 统一维度名
        rename_map = {}
        if "south_north" in ds.dims: rename_map["south_north"] = "y"
        if "west_east"   in ds.dims: rename_map["west_east"]   = "x"
        if "bottom_top" in ds.dims: rename_map["bottom_top"]   = "level"
        if rename_map:
            ds = ds.rename(rename_map)

        if level_index is not None and "level" in ds.dims:
            ds = ds.isel(level=level_index).squeeze(drop=True)
            ds.attrs["selected_level_dim"] = "level"
            ds.attrs["selected_level_indices"] = [int(i) for i in level_index]
        
        if bufferzone and bufferzone > 0 and "x" in ds.dims and "y" in ds.dims:
            ds = ds.isel(x=slice(bufferzone, -bufferzone), y=slice(bufferzone, -bufferzone))

        # 关键：把数据加载到内存；离开 with 后文件会被关闭
        ds = ds.load()
    
    return ds



def Process_CWRF(ds, varname, varindata):
    """处理 CWRF 数据集，进行必要的预处理和重采样。"""
    # 单位转换
    if varname == 'T2m':
        ds = ds.rename({varindata:varname})
        ds[varname] = ds[varname] - 273.15  # K → °C
        # 合理数据范围检查，将异常值设置为 NaN
        ds[varname] = ds[varname].where((ds[varname] >= -100) & (ds[varname] <= 200), np.nan)
        ds[varname].attrs.update({'units':'degC', 'long_name':'2m temperature'})
    elif varname == 'T2m-Max':
        ds = ds.rename({varindata:varname})
        ds[varname] = ds[varname] - 273.15  # K → °C
        # 合理数据范围检查，将异常值设置为 NaN
        ds[varname] = ds[varname].where((ds[varname] >= -100) & (ds[varname] <= 200), np.nan)
        ds[varname].attrs.update({'units':'degC', 'long_name':'2m maximum temperature'})
    elif varname == 'T2m-Min':
        ds = ds.rename({varindata:varname})
        ds[varname] = ds[varname] - 273.15
        # 合理数据范围检查，将异常值设置为 NaN
        ds[varname] = ds[varname].where((ds[varname] >= -100) & (ds[varname] <= 200), np.nan)
        ds[varname].attrs.update({'units':'degC', 'long_name':'2m minimum temperature'})
    elif varname == 'Prec':
        # 若确认是 mm/hour，可改成 mm/day：ds['PRAVG'] = ds['PRAVG'] * 24
        ds = ds.rename({varindata:varname})
        ds[varname] = ds[varname] * 86400  # mm/s → mm/day
        # 合理数据范围检查，将异常值设置为 NaN
        ds[varname] = ds[varname].where((ds[varname] >= 0) & (ds[varname] <= 2000), np.nan)
        ds[varname].attrs.update({'units':'mm/day','long_name':'precipitation'})
    elif varname == 'Prec-Max':
        ds = ds.rename({varindata:varname})
        ds[varname] = ds[varname] * 86400
        ds[varname] = ds[varname].where((ds[varname] >= 0) & (ds[varname] <= 2000), np.nan)
        ds[varname].attrs.update({'units':'mm/day','long_name':'maximum precipitation'})
    elif varname == 'Prec-Min':
        ds = ds.rename({varindata:varname})
        ds[varname] = ds[varname] * 86400
        ds[varname] = ds[varname].where((ds[varname] >= 0) & (ds[varname] <= 2000), np.nan)
        ds[varname].attrs.update({'units':'mm/day','long_name':'minimum precipitation'})
    elif varname == 'LHF':
        ds = ds.rename({varindata:varname})
        ds[varname] = ds[varname] 
        ds[varname] = ds[varname].where((ds[varname] >= -1000) & (ds[varname] <= 1000), np.nan)
        ds[varname].attrs.update({'units':'W/m2','long_name':'latent heat flux'})
    elif varname == 'SHF':
        ds = ds.rename({varindata:varname})
        ds[varname] = ds[varname] 
        ds[varname] = ds[varname].where((ds[varname] >= -1000) & (ds[varname] <= 1000), np.nan)
        ds[varname].attrs.update({'units':'W/m2','long_name':'sensible heat flux'})
    elif varname == 'CloudFra':
        ds = ds.rename({varindata:varname})
        ds[varname] = ds[varname] * 100  # 比例尺转换
        ds[varname] = ds[varname].where((ds[varname] >= 0) & (ds[varname] <= 100), np.nan)
        ds[varname].attrs.update({'units':'%','long_name':'cloud fraction'})
    elif varname == 'RH':
        ds = ds.rename({varindata:varname})
        ds[varname] = ds[varname] 
        ds[varname] = ds[varname].where((ds[varname] >= 0) & (ds[varname] <= 100), np.nan)
        ds[varname].attrs.update({'units':'%','long_name':'relative humidity at 2m'})
    elif varname == 'Q2m':
        ds = ds.rename({varindata:varname})
        ds[varname] = ds[varname] * 1000  # kg/kg → g/kg
        ds[varname] = ds[varname].where((ds[varname] >= 0) & (ds[varname] <= 1000), np.nan)
        ds[varname].attrs.update({'units':'g/kg','long_name':'specific humidity at 2m'})
    elif varname == 'U10':
        ds = ds.rename({varindata:varname})
        ds[varname] = ds[varname] 
        ds[varname] = ds[varname].where((ds[varname] >= -300) & (ds[varname] <= 300), np.nan)
        ds[varname].attrs.update({'units':'m/s','long_name':'10m U wind component'})
    elif varname == 'V10':
        ds = ds.rename({varindata:varname})
        ds[varname] = ds[varname] 
        ds[varname] = ds[varname].where((ds[varname] >= -300) & (ds[varname] <= 300), np.nan)
        ds[varname].attrs.update({'units':'m/s','long_name':'10m V wind component'})
    elif varname == 'PBLH':
        ds = ds.rename({varindata:varname})
        ds[varname] = ds[varname].where((ds[varname] >= 0) & (ds[varname] <= 1000000), np.nan)
        ds[varname].attrs.update({'units':'m','long_name':'planetary boundary layer height'})
    elif varname == 'Height':
        ds = ds.rename({varindata:varname})
        ds[varname] = ds[varname].where((ds[varname] >= 0) & (ds[varname] <= 1000000), np.nan)
        ds[varname].attrs.update({'units':'m','long_name':'geopotential height at pressure level'})
    elif varname == 'Theta':
        ds = ds.rename({varindata:varname})
        ds[varname] = ds[varname] 
        ds[varname] = ds[varname].where((ds[varname] >= 200) & (ds[varname] <= 400), np.nan)
        ds[varname].attrs.update({'units':'K','long_name':'potential temperature at pressure level'})
    elif varname in ["U"]:
        ds = ds.rename({varindata:varname})
        ds[varname] = ds[varname] 
        ds[varname] = ds[varname].where((ds[varname] >= -300) & (ds[varname] <= 300), np.nan)
        ds[varname].attrs.update({'units':'m/s','long_name':'U wind component at pressure level'})
    elif varname in ["V"]:
        ds = ds.rename({varindata:varname})
        ds[varname] = ds[varname] 
        ds[varname] = ds[varname].where((ds[varname] >= -300) & (ds[varname] <= 300), np.nan)
        ds[varname].attrs.update({'units':'m/s','long_name':'V wind component at pressure level'})
    elif varname in ["W"]:
        ds = ds.rename({varindata:varname})
        ds[varname] = ds[varname] 
        ds[varname] = ds[varname].where((ds[varname] >= -300) & (ds[varname] <= 300), np.nan)
        ds[varname].attrs.update({'units':'m/s','long_name':'vertical wind component at pressure level'})
    elif varname in ["T"]:
        ds = ds.rename({varindata:varname})
        ds[varname] = ds[varname] - 273.15  # K → °C
        ds[varname] = ds[varname].where((ds[varname] >= -100) & (ds[varname] <= 200), np.nan)
        ds[varname].attrs.update({'units':'degC','long_name':'temperature at pressure level'})
    else:
        raise ValueError(f"Unknown variable name: {varname}")
    
    return ds




def read_CWRFPOST(path, starttime, endtime, casename, varname, varindata, filename,
                  timefreq="daily", rspmethod= "mean", bufferzone=BufferZone, level_index=None):
    """读取 CPOST 处理后的 CWRF 数据，并按 timefreq 重采样。"""

    # 1) 年份范围更稳妥
    s = pd.to_datetime(starttime)
    e = pd.to_datetime(endtime)
    years = range(s.year, e.year + 1)
    
    # 收集存在的文件
    files = []
    for y in years:
        if timefreq.lower() in ["daily"]:
            fp = f"{path}/Daily/{casename}_{y}_{filename}_hourly_daily_{rspmethod}.nc"
            if os.path.exists(fp):
                files.append(fp)
        elif timefreq.lower() in ["monthly"]:
            fp = f"{path}/Monthly/{casename}_{y}_{filename}_hourly_monthly_{rspmethod}.nc"
            if os.path.exists(fp):
                files.append(fp)
        elif timefreq.lower() in ["monthly_diurnal"]:
            fp = f"{path}/MonthlyDiurnal/{casename}_{y}_{filename}_hourly_monthly_diurnal_{rspmethod}.nc"
            if os.path.exists(fp):
                files.append(fp)
        elif timefreq.lower() in ["seasonal"]:
            fp = f"{path}/Seasonal/{casename}_{y}_{filename}_hourly_seasonal_{rspmethod}.nc"
            if os.path.exists(fp):
                files.append(fp)
        elif timefreq.lower() in ["seasonal_diurnal"]:
            fp = f"{path}/SeasonalDiurnal/{casename}_{y}_{filename}_hourly_seasonal_diurnal_{rspmethod}.nc"
            if os.path.exists(fp):
                files.append(fp)
        else:
            raise ValueError(f"Unknown timefreq: {timefreq}")
    if not files:
        print(fp, "\n\n")
        raise FileNotFoundError("No input files found for the given time range.")

    # 并行读取
    args_list = []
    for i, f in enumerate(files):
        readfunc = read_data
        if level_index is not None:
            args = {'file': f, 'level_index': level_index, 'bufferzone': bufferzone}
        else:
            args = {'file': f, 'bufferzone': bufferzone}
        args_list.append(args)
    ntasks = len(files)
    with Parallel(n_jobs=12, backend="loky", verbose=0,
                  pre_dispatch="2*n_jobs", return_as="generator") as parallel:
        gen = parallel(
            delayed(readfunc)(
                task
            )
            for i, task in enumerate(args_list)
        )
        results = [p for p in tqdm(gen, total=ntasks,
                                  desc=f"    ➠ Reading data", unit="file",
                                  dynamic_ncols=True, mininterval=0.1, leave=False)]

        ds = xr.concat(results, dim="time")
    ds = Process_CWRF(ds, varname, varindata)
    # 4) 裁切到目标时间范围
    if "time" in ds.dims or "time" in ds.coords:
        ds = ds.sel(time=slice(s, e))
        ds = ds.sortby("time")
    # ds = resample_time(ds, timefreq, rspmethod)
    #只返回目标变量
    # ds.to_netcdf(f"./cases/{casename}_{varname}_daily.nc", format='NETCDF4')
    get_reusable_executor().shutdown(wait=True, kill_workers=True)
    return ds[varname].squeeze(drop=True)



def read_CWRF_Info(path, bufferzone=BufferZone):
    wrfinp = xr.open_dataset(path).rename({'Times':'time','south_north':'y','west_east':'x'})
    wrfinp = wrfinp.isel(x=slice(bufferzone,-bufferzone), y=slice(bufferzone,-bufferzone))
    lon2d = wrfinp['XLONG'].values.squeeze()
    lat2d = wrfinp['XLAT'].values.squeeze()
    mapfac_mx = wrfinp['MAPFAC_MX'].values.squeeze()
    mapfac_my = wrfinp['MAPFAC_MY'].values.squeeze()
    scwater = wrfinp['SC_WATER'].values.squeeze()
    lakedp = wrfinp['DPLAKE'].values.squeeze()
    wrfinp.close()
    return lon2d, lat2d, mapfac_mx, mapfac_my, scwater, lakedp



def Process_CMFD(ds, varname):
    """处理 CMFD 数据集，进行必要的预处理和重采样。"""
    # 单位转换
    if varname == 'T2m':
        ds[varname] = ds[varname] - 273.15
        ds[varname].attrs.update({'units':'degC','long_name':'2m temperature'})
    elif varname == 'Prec':
        ds[varname] = ds[varname] * 24        # mm/h → mm/day
        ds[varname].attrs.update({'units':'mm/day','long_name':'precipitation'})
    elif varname == 'RH':
        ds[varname] = ds[varname] 
        ds[varname].attrs.update({'units':'%','long_name':'relative humidity at 2m'})
    elif varname == 'Q2m':
        ds[varname] = ds[varname] * 1000  # kg/kg → g/kg
        ds[varname].attrs.update({'units':'g/kg','long_name':'specific humidity at 2m'})
    else:
        raise ValueError(f"Unknown variable name: {varname}")
    return ds



def Process_CMFDv2(ds, varname):
    """处理 CMFD 数据集，进行必要的预处理和重采样。"""
    # 单位转换
    if varname == 'T2m':
        ds[varname] = ds[varname] - 273.15
        ds[varname].attrs.update({'units':'degC','long_name':'2m temperature'})
    elif varname == 'Prec':
        ds[varname] = ds[varname] * 86400        # mm/h → mm/day
        ds[varname].attrs.update({'units':'mm/day','long_name':'precipitation'})
    elif varname == 'RH':
        ds[varname] = ds[varname] 
        ds[varname].attrs.update({'units':'%','long_name':'relative humidity at 2m'})
    elif varname == 'Q2m':
        ds[varname] = ds[varname] * 1000  # kg/kg → g/kg
        ds[varname].attrs.update({'units':'g/kg','long_name':'specific humidity at 2m'})
    else:
        raise ValueError(f"Unknown variable name: {varname}")
    return ds



def Process_CDMet(ds, varname):
    # 单位转换
    if varname == 'T2m':
        ds[varname] = ds[varname] - 273.15
        ds[varname].attrs.update({'units':'degC','long_name':'2m temperature'})
    elif varname == 'Prec':
        ds[varname] = ds[varname]        #mm/day
        ds[varname].attrs.update({'units':'mm/day','long_name':'precipitation'})
    else:
        raise ValueError(f"Unknown variable name: {varname}")
    return ds



def Process_ERA5LAND(ds, varname):
    # 单位转换
    if varname == 'T2m':
        ds[varname] = ds[varname] - 273.15
        ds[varname].attrs.update({'units':'degC','long_name':'2m temperature'})
    elif varname == 'Prec':
        ds[varname] = ds[varname] * 1000 * 24        # m/h → mm/day
        ds[varname].attrs.update({'units':'mm/day','long_name':'precipitation'})
    elif varname == 'Q2m':
        ds[varname] = ds[varname] * 1000  # kg/kg → g/kg
        ds[varname].attrs.update({'units':'g/kg','long_name':'specific humidity at 2m'})
    else:
        raise ValueError(f"Unknown variable name: {varname}")
    return ds



def Process_CN051(ds, varname):
    # 单位转换
    if varname == 'T2m':
        ds[varname] = ds[varname]
        ds[varname].attrs.update({'units':'degC','long_name':'2m temperature'})
    elif varname == 'Prec':
        ds[varname] = ds[varname]         # mm/day
        ds[varname].attrs.update({'units':'mm/day','long_name':'precipitation'})
    else:
        raise ValueError(f"Unknown variable name: {varname}")
    return ds



def Process_MSWX(ds, varname):
    # 单位转换
    if varname == 'T2m':
        ds[varname] = ds[varname]
        ds[varname].attrs.update({'units':'degC','long_name':'2m temperature'})
    elif varname == 'Prec':
        ds[varname] = ds[varname] * 8        # mm/3h → mm/day
        ds[varname].attrs.update({'units':'mm/day','long_name':'precipitation'})
    else:
        raise ValueError(f"Unknown variable name: {varname}")
    return ds



def read_RefData(path, starttime, endtime, refname, varname, suffix, timefreq, bufferzone=BufferZone):
    """读取参考数据，并按 timefreq 重采样。"""
    print(f"    - Reading {refname} {varname} data...")
    t0 = time.time()
    # 年份范围更稳妥
    s = pd.to_datetime(starttime)
    e = pd.to_datetime(endtime)
    years = range(s.year, e.year + 1)
    # 收集存在的文件
    files = []
    for y in years:
        fp = f"{path}/{refname}_{suffix}_{varname}_{y}_daily_mean.nc"
        if os.path.exists(fp):
            files.append(fp)
        else:
            print(f"  - Warning: File {fp} does not exist, skipping.")

    if not files:
        raise FileNotFoundError("No input files found for the given time range.")
    args = [{'file': f, 'bufferzone': bufferzone} for f in files]
    # 并行读取
    results = Parallel(n_jobs=10, backend="loky", verbose=0,
                       pre_dispatch="2*n_jobs")(
        delayed(read_data)(arg) for arg in args
    )

    ds = xr.concat(results, dim="time")
    # 4) 裁切到目标时间范围
    if "time" in ds.dims or "time" in ds.coords:
        ds = ds.sel(time=slice(s, e))
    else:
        raise ValueError("Dataset has no 'time' coordinate/dimension.")

    # 5) 重采样（一次性进行，避免跨年窗口被截断）
    freq_map = {
        "seasonly": "QS-DEC",   # 季度起点为 12 月，对应 DJF/MAM/JJA/SON
        "monthly": "MS",       # 月初
        "daily": "1D",
        "hourly": "1H",
        None: None,
    }

    if timefreq not in freq_map:
        raise ValueError(f"Unknown timefreq='{timefreq}'. Choose from {list(k for k in freq_map if k)}.")

    freq = freq_map[timefreq]
    if freq is not None:
        # 可按需设置 label/closed；默认 label='left', closed='left'
        ds = ds.resample(time=freq).mean(skipna=True)
    ds = ds.sortby("time")

    # 6) 处理不同数据集的特定变量
    if refname == "CMFD":
        ds = Process_CMFD(ds, varname)
    elif refname == "CMFDv2":
        ds = Process_CMFDv2(ds, varname)
    elif refname == "ERA5-Land":
        ds = Process_ERA5LAND(ds, varname)
    elif refname == "CN05.1":
        ds = Process_CN051(ds, varname)
    elif refname == "MSWX":
        ds = Process_MSWX(ds, varname)
    elif refname == "CDMet":
        ds = Process_CDMet(ds, varname)
    
    # 计算掩模
    mask2d = ds[varname].notnull().any(dim='time')

    print(f"     ➔ Time Spent: {time.time() - t0:.2f} seconds")
    get_reusable_executor().shutdown(wait=True, kill_workers=True)
    return ds, mask2d



def read_seasonal_metrics(casename, varname, reflist, outdir):
    """读取季节性指标数据。"""
    ds_dict = {}
    seasons = TU.get_seasons()
    for season in seasons:
        ds_dict[season] = {}
        for refname in reflist:
            refpath = f"{outdir}/{casename}/Perform_Area_{casename}_{refname}_{varname}_{season}.nc"
            refds = xr.open_dataset(refpath)
            ds_dict[season][refname] = refds['clim_ref'].values
            ds_dict[season][casename] = refds['clim_case'].values
            ds_dict[season][f"{casename}-{refname}"] = refds['bias'].values
    return ds_dict



def read_extreme_metrics(casename, reflist, outdir):
    """读取极端天气事件指标数据。"""
    ds_dict = {}
    events = ['Cold', 'ColdWet', 'ColdDry', 'Hot', 'HotWet', 'HotDry', 'Wet', 'Dry']
    for refname in reflist:
        ds_dict[refname] = {}
        for event in events:
            refpath = f"{outdir}/{refname}/Extreme_{refname}.nc"
            refds = xr.open_dataset(refpath)
            ds_dict[refname][event] = refds[event]
    refpath = f"{outdir}/{casename}/Extreme_{casename}.nc"
    refds = xr.open_dataset(refpath)
    ds_dict[casename] = {}
    for event in events:
        ds_dict[casename][event] = refds[event]
    return ds_dict



def read_seasonal_significance(caselist, varname, checkmethod, outdir):
    """读取季节性显著性分析结果。"""
    ds_dict = {}
    seasons = TU.get_seasons()
    for season in seasons:
        ds_dict[season] = {}
        filepath = f"{outdir}/Analysis/Significance_{caselist[0]}-{caselist[1]}_{varname}_seasonalmean_{season}_{checkmethod}.nc"
        print(f"    - Reading: {os.path.relpath(filepath, os.getcwd())}")
        if os.path.exists(filepath):
            ds = xr.open_dataset(filepath)
            ds_dict[season] = ds
        else:
            print(f"  - Warning: Seasonal significance file {filepath} does not exist, skipping.")
    return ds_dict



def read_hourly_significance(caselist, varname, season, keep_hours, checkmethod, outdir):
    """
    读取季节性显著性分析结果，仅保留指定小时（默认 0,6,12,18）。
    返回结构：ds_dict[season] = <包含子集小时的 Dataset>
    """
    ds_dict = {}
    fp = f"{outdir}/RegClimImpact/Significance_RegClimImpact_{caselist[0]}-{caselist[1]}_{varname}_seasonaldiurnalmean_{season}_{checkmethod}.nc"
    if not os.path.exists(fp):
        print(f"  - Warning: Seasonal significance file {fp} does not exist, skipping.")
    # 读文件（一次）
    ds = xr.open_dataset(fp)
    if "hour" not in ds.dims:
        print(f"  - Warning: {fp} has no 'hour' dimension, skipping.")
        ds.close()
    # 与文件中的 hour 交集，避免 KeyError
    avail = ds["hour"].values.tolist()
    keep = [h for h in keep_hours if h in avail]
    if len(keep) == 0:
        print(f"  - Warning: none of {keep_hours} found in {fp}, available hours: {avail[:8]}{'...' if len(avail)>8 else ''}")
        ds.close()
    if len(keep) < len(keep_hours):
        missing = [h for h in keep_hours if h not in avail]
        print(f"  - Info: hours missing in {fp}: {missing}; keeping {keep}")
    # 子集化并载入内存，随后关闭文件句柄
    ds_sub = ds.sel(hour=keep).load()
    ds.close()
    # 也可排序为 0,6,12,18 的顺序（若文件中顺序不同）
    order = np.argsort(np.array(keep))
    ds_sub = ds_sub.isel(hour=order)

    for h in sorted(keep):
        h_bjt_str = TU.UTC_to_BJT_str(h)
        ds_dict[h_bjt_str] = ds_sub.sel(hour=h).load()

    return ds_dict




def read_hourly_significance_PressureLevel(caselist, varname, season, level, keep_hours, checkmethod, outdir):
    """
    读取季节性显著性分析结果，仅保留指定小时（默认 0,6,12,18）。
    返回结构：ds_dict[season] = <包含子集小时的 Dataset>
    """
    ds_dict = {}
    fp = f"{outdir}/PressureLevel/{level}hPa/Significance_PressureLevel_{caselist[0]}-{caselist[1]}_{varname}_{level}hPa_seasonaldiurnalmean_{season}_{checkmethod}.nc"
    if not os.path.exists(fp):
        print(f"  - Warning: Seasonal significance file {fp} does not exist, skipping.")
    # 读文件（一次）
    ds = xr.open_dataset(fp)
    if "hour" not in ds.dims:
        print(f"  - Warning: {fp} has no 'hour' dimension, skipping.")
        ds.close()
    # 与文件中的 hour 交集，避免 KeyError
    avail = ds["hour"].values.tolist()
    keep = [h for h in keep_hours if h in avail]
    if len(keep) == 0:
        print(f"  - Warning: none of {keep_hours} found in {fp}, available hours: {avail[:8]}{'...' if len(avail)>8 else ''}")
        ds.close()
    if len(keep) < len(keep_hours):
        missing = [h for h in keep_hours if h not in avail]
        print(f"  - Info: hours missing in {fp}: {missing}; keeping {keep}")
    # 子集化并载入内存，随后关闭文件句柄
    ds_sub = ds.sel(hour=keep).load()
    ds.close()
    # 也可排序为 0,6,12,18 的顺序（若文件中顺序不同）
    order = np.argsort(np.array(keep))
    ds_sub = ds_sub.isel(hour=order)

    for h in sorted(keep):
        h_bjt_str = TU.UTC_to_BJT_str(h)
        ds_dict[h_bjt_str] = ds_sub.sel(hour=h).load()

    return ds_dict



def read_mask(filepath):
    """读取掩模数据。"""
    if os.path.exists(filepath):
        maskds = xr.open_dataset(filepath)
        mask = maskds['mask'].values
    else:
        print(f"  - Warning: Mask file {filepath} does not exist, using default mask.")
    return mask



def read_newnc(filepath: str) -> xr.Dataset:
    """读取新的 NetCDF 文件。"""
    if os.path.exists(filepath):
        return xr.open_dataset(filepath)
    else:
        raise FileNotFoundError(f"File {filepath} not found.")



def save_newnc(savepath: str, in_dict: dict[str, list], coords: dict[str, np.ndarray]) -> None:
    """
    将逐格点指标保存为 NetCDF。 
    savepath: 保存路径
    in_dict: 包含指标数据的字典，键为变量名，值为对应的指标数据列表。 varname: [coord_list, data_narray]
    coords: 坐标字典，键为坐标名，值为对应的坐标数组。
    例如：coords = {'lat': lat_arr, 'lon': lon_arr}
    """
    os.system(f"rm -f {savepath}") if os.path.exists(savepath) else None
    ds = xr.Dataset(coords=coords)
    for varname, varinfo in in_dict.items():
        coord, metric_data = varinfo
        for c in coord:
            if c not in ds.coords:
                raise ValueError(f"Coordinate '{c}' not found in dataset coordinates.")
        ds = ds.assign({varname: (coord, metric_data)})
    os.remove(savepath) if os.path.exists(savepath) else None
    ds.to_netcdf(savepath, format='NETCDF4')
    print(f"    ➠ Saved to {os.path.relpath(savepath, os.getcwd())}")



def save_excel(savepath: str, in_df: pd.DataFrame | dict[str, pd.DataFrame]) -> None:
    """将区域平均/标量指标保存为 Excel（包括 ACC、S_dist 及面积加权均值等）。"""
    os.system(f"rm -f {savepath}") if os.path.exists(savepath) else None
    if isinstance(in_df, dict):
        with pd.ExcelWriter(savepath) as writer:
            for sheet_name, df in in_df.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        in_df.to_excel(savepath, index=False)
    print(f"    ➠ Saved to {os.path.relpath(savepath, os.getcwd())}")

    
    
def read_tif_as_da(
        path: str,
        band: int | None = None,
        mask_nodata: bool = True,
        squeeze: bool = True,
    ) -> xr.DataArray:
    """
    从 GeoTIFF 读取数据为 xarray.DataArray。

    参数
    ----
    path : str
        .tif 文件路径
    band : int | None
        读取哪一个波段（从 1 开始计数）。
        - None: 保留所有波段（dims: band, y, x）
        - 1/2/...: 只取其中一个波段
    mask_nodata : bool
        是否把 nodata 像元转为 NaN（使用掩膜）
    squeeze : bool
        如果只有一个波段，是否自动去掉 band 维度

    返回
    ----
    xr.DataArray
        dims 一般为 (y, x) 或 (band, y, x)，带有空间坐标和 CRS 信息。
    """
    import rioxarray as rxr

    # masked=True 会根据 nodata 生成掩膜
    da = rxr.open_rasterio(path, masked=mask_nodata)

    # 选择指定 band（注意 rioxarray 的 band 是从 1 开始）
    if band is not None:
        da = da.sel(band=band)

    # 如果只有一个 band，且希望去掉 band 维度
    if squeeze and "band" in da.dims and da.sizes["band"] == 1:
        da = da.isel(band=0, drop=True)

    # 给个更友好的名字
    if da.name is None:
        da.name = "pop"

    return da

    