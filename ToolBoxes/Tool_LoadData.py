import gc
import os
import time
import warnings
import numpy as np
import xarray as xr
import pandas as pd
from typing import Literal, Optional, Tuple, Dict, Any, Sequence, List

# 自定义模块
import ToolBoxes.Utils as TU
import ToolBoxes.Tool_InputOutput as TIO

warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.set_option("display.max_rows", None)
pd.set_option("display.expand_frame_repr", False)


def load_OneVar_AtSurface(
                        varname: str,
                        timefreq: str,
                        vardict: Dict[str, Any],
                        IO_dict: Dict[str, Any],
) -> xr.DataArray:
    """
    加载 CWRF 数据，并进行时间选择

    Parameters
    ----------
    casename : str
        实验名称
    varname : str
        变量名称
    datadir : str
        数据目录
    time_sel : Sequence[pd.Timestamp], optional
        需要选择的时间点序列，默认为 None，即不进行时间选择
    Returns
    -------
    xarr : xr.DataArray
        加载并处理后的数据
    """
    t0 = time.time()
    CasePath = IO_dict["CasePath"]
    StartTime = IO_dict["StartTime"]
    EndTime = IO_dict["EndTime"]
    BufferZone = IO_dict["BufferZone"]

    print(f"   → Load variable '{varname}' at surface...")
    Lake_data = TIO.read_CWRFPOST(CasePath["Lake"], StartTime, EndTime, "Lake", varname, timefreq=timefreq,
                                varindata=vardict[varname]['VarInData'],
                                filename=vardict[varname]['FileName'],
                                rspmethod=vardict[varname]['RspMethod'],
                                bufferzone=BufferZone)
                                
    NoLake_data = TIO.read_CWRFPOST(CasePath["NoLake"], StartTime, EndTime, "NoLake", varname, timefreq=timefreq,
                                    varindata=vardict[varname]['VarInData'],
                                    filename=vardict[varname]['FileName'],
                                    rspmethod=vardict[varname]['RspMethod'],
                                    bufferzone=BufferZone)

    print(f"     ➔ Time Spent: {time.time() - t0:.2f} seconds")

    return Lake_data, NoLake_data



def load_OneVar_AtPressure(
                        varname: str,
                        timefreq: str,
                        vardict: Dict[str, Any],
                        pressure_levels: List[float],
                        IO_dict: Dict[str, Any],
                        press_level: float = None,
) -> xr.DataArray:
    """
    加载 CWRF 数据，并进行时间选择
    Parameters
    ----------
    casename : str
        实验名称
    varname : str
        变量名称
    datadir : str
        数据目录
    time_sel : Sequence[pd.Timestamp], optional
        需要选择的时间点序列，默认为 None，即不进行时间选择
    Returns
    -------
    xarr : xr.DataArray
        加载并处理后的数据
    """
    t0 = time.time()
    CasePath = IO_dict["CasePath"]
    StartTime = IO_dict["StartTime"]
    EndTime = IO_dict["EndTime"]
    BufferZone = IO_dict["BufferZone"]

    if press_level is None:
        level_index = None
    else:
        # 确保 press_level 是一个列表
        if not isinstance(press_level, list):
            press_level = [press_level]
        # 获取对应的层次索引
        level_index = []
        for p in press_level:
            try:
                idx = pressure_levels.index(p)
            except ValueError:
                raise ValueError(f"指定的压强层 {p} 不在 pressure_levels 中！")
            level_index.append(idx)
    print(f"   → Load variable '{varname}' at pressure level {press_level} hPa...")
    Lake_data = TIO.read_CWRFPOST(CasePath["Lake"], StartTime, EndTime, "Lake", varname, timefreq=timefreq,
                                   varindata=vardict[varname]['VarInData'],
                                   filename=vardict[varname]['FileName'],
                                   level_index=level_index,
                                   rspmethod=vardict[varname]['RspMethod'],
                                   bufferzone=BufferZone)

    NoLake_data = TIO.read_CWRFPOST(CasePath["NoLake"], StartTime, EndTime, "NoLake", varname, timefreq=timefreq,
                                   varindata=vardict[varname]['VarInData'],
                                   filename=vardict[varname]['FileName'],
                                   level_index=level_index,
                                   rspmethod=vardict[varname]['RspMethod'],
                                   bufferzone=BufferZone)

    print(f"     ➔ Time Spent: {time.time() - t0:.2f} seconds")

    return Lake_data, NoLake_data