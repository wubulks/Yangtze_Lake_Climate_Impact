#!/stu01/wumej22/Anaconda3/envs/cwrf_env/bin/python
import os
import time
import warnings
import matplotlib
import numpy as np
import xarray as xr
import pandas as pd

# 自定义模块
import ToolBoxes.Utils as TU
import ToolBoxes.Tool_InputOutput as TIO
import ToolBoxes.Tool_PerformanceMetrics as TPM
import Func_WarmAdvection as FWA
import Func_ModelValidation as FMV
import Func_RegClimImpact as FRCI
import ToolBoxes.Tool_ExtremeEventMetrics as TEEM

matplotlib.use('Agg')  # 不显示图，只保存
warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.set_option("display.max_rows", None)
pd.set_option("display.expand_frame_repr", False)
time1 = time.time()

# ========== Option ==========
StartTime = "2000-01-01"
# EndTime = "2022-12-31"
EndTime = "2024-12-31"
# EndTime = "2002-12-31"

# ========== Config ==========
BufferZone = 15  # 缓冲区宽度，单位：grid

Flag_ModelEvaluate     = True   # 是否进行模式评估
Flag_ExtremeEvent_Calc = False   # 是否进行参考数据极端事件计算
Flag_Season_Plot       = True    # 是否进行季节验证图绘制
Flag_Extreme_Plot      = False   # 是否进行极端事件分析绘图


#------------------------------------------------------------------------
# 基础配置
#------------------------------------------------------------------------
# 创建目录
# ========== Path ==========
CasePath = {
    "Lake": "/hydata01/wumej22/Experiment/CWRF/Yangtze_C/ERA5_history/Lake",
    # "NoLake": "/hydata01/wumej22/Experiment/CWRF/Yangtze_C/ERA5_history/NoLake"
    # "CUOFF": "/hydata01/wumej22/Experiment/CWRF/Yangtze_C/ERA5_history/CUOFF",
    }

DataOutDir = "/hydata01/wumej22/Experiment/Process/Yangtze_C_6km_r1/Cases"
# FigOutDir = "/hydata01/wumej22/Experiment/Process/Yangtze_C_6km_r1/Figures"
FigOutDir = "/hydata01/wumej22/Experiment/Process/Yangtze_C_6km_r1/Figures"  
cwrfinp_path = "/hydata01/wumej22/Experiment/Process/Yangtze_C_6km_r1/BaseData/wrfinput_d01.2000"
Station_Dir = "/home/wumej22/hydata/Experiment/CWRF/Yangtze_C/RefData/StationObs/Data"

RefPath = {
    # "CMFD": ["/hydata01/wumej22/Experiment/CWRF/Yangtze_C/RefData/CMFD/Data/merged_by_year", [2000, 2018]],
    # "CN05.1": ["/hydata01/wumej22/Experiment/CWRF/Yangtze_C/RefData/CN05.1/Data/merged_by_year", [2000, 2021]],
    "CMFDv2": ["/hydata01/wumej22/Experiment/CWRF/Yangtze_C/RefData/CMFDV2/Data/merged_by_year", [2000, 2024]],
    # "CDMet": ["/hydata01/wumej22/Experiment/CWRF/Yangtze_C/RefData/CDMet/Data/merged_by_year", [2000, 2020]],
    # "MSWX": ["/hydata01/wumej22/Experiment/CWRF/Yangtze_C/RefData/MSWX/Data/merged_by_year", [2000, 2020]],
    "ERA5-Land": ["/hydata01/wumej22/Experiment/CWRF/Yangtze_C/RefData/ERA5LAND/Data/merged_by_year", [2000, 2024]],
    # "MSWEP": ["/hydata01/wumej22/Experiment/CWRF/Yangtze_C/RefData/MSWX/Data/merged_by_year", [2000, 2020]],
    # "ET_Xu": ["/hydata01/wumej22/Experiment/CWRF/Yangtze_C/RefData/ET_Xu/Data/merged_by_year", [2000, 2020]],
    }


Vardict = {
    'T2m': {
            'refdata': ['CMFDv2', "ERA5-Land", ], # "CN05.1", 'CDMet', "MSWX", 
            'VarInData': 'AT2M', 'FileName': 'AT2M',
            'RspMethod': 'mean', 'CheckMethod': 'Paired_t-test',
           },
    'Prec': {
            'refdata': ['CMFDv2', "ERA5-Land", ], # 'CDMet', "MSWX", 
            'VarInData': 'PRAVG', 'FileName': 'PRAVG',
            'RspMethod': 'mean', 'CheckMethod': 'Wilcoxon_signed_rank',
           },
    # 'RH': {
    #         'refdata': ['CMFDv2', "CN05.1",], # 'CDMet', "MSWX", "ERA5-Land", 
    #         'VarInData': 'RH', 'FileName': 'RH',
    #         'RspMethod': 'mean', 'CheckMethod': 'Paired_t-test',
    #        },
    'Q2m': {
            'refdata': ['CMFDv2', "ERA5-Land"], # 'CDMet', "MSWX", 
            'VarInData': 'AQ2M', 'FileName': 'AQ2M',
            'RspMethod': 'mean', 'CheckMethod': 'Paired_t-test',
           },
    }



os.makedirs(DataOutDir, exist_ok=True)
os.makedirs(FigOutDir, exist_ok=True)
os.makedirs(f"{DataOutDir}/ModelValidation", exist_ok=True)
os.makedirs(f"{FigOutDir}/ModelValidation", exist_ok=True)
DataOutDir_var = f"{DataOutDir}/ModelValidation"
FigOutDir_var = f"{FigOutDir}/ModelValidation"

# os.makedirs(f"{DataOutDir}/ModelValidation_NCP", exist_ok=True)
# os.makedirs(f"{FigOutDir}/ModelValidation_NCP", exist_ok=True)
# DataOutDir_var = f"{DataOutDir}/ModelValidation_NCP"
# FigOutDir_var = f"{FigOutDir}/ModelValidation_NCP"

# os.makedirs(f"{DataOutDir}/ModelValidation_CN05.1", exist_ok=True)
# os.makedirs(f"{FigOutDir}/ModelValidation_CN05.1", exist_ok=True)
# DataOutDir_var = f"{DataOutDir}/ModelValidation_CN05.1"
# FigOutDir_var = f"{FigOutDir}/ModelValidation_CN05.1"

# 开始验证
dx, dy = 6000, 6000  # 网格分辨率，单位：m
lon2d, lat2d, mapfac_mx, mapfac_my, scwater, lakedp = TIO.read_CWRF_Info(cwrfinp_path, bufferzone=BufferZone)
cwrfarea = TU.get_cwrf_grid_area(DX=dx, DY=dy, MAPFAC_MX=mapfac_mx, MAPFAC_MY=mapfac_my)
lkinfos = TU.get_lake_area_mask(scwater, lakedp, DX=dx, DY=dy, MAPFAC_MX=mapfac_mx, MAPFAC_MY=mapfac_my, nc_path=f"{DataOutDir}/lake_mask_with_dist.nc")
lkinfos['area'] = cwrfarea


########################################################
# 计算部分
########################################################
if Flag_ModelEvaluate:
    varmask = []
    for var in Vardict.keys():
        mask = None
        print(f"\n\n★ Evaluation: {var}")
        refds = {}
        for refname in Vardict[var]['refdata']:
            ds, mask2d  = TIO.read_RefData(RefPath[refname][0], StartTime, EndTime, refname, var, suffix='Yangtze_C', timefreq="daily", bufferzone=BufferZone)
            refds[refname] = ds
            mask = (mask & mask2d) if mask is not None else mask2d
        mask = mask.fillna(False).astype(bool)
        for case in CasePath.keys():
            print(f"\n    Processing case: {case}")
            caseds = TIO.read_CWRFPOST(CasePath[case], StartTime, EndTime, case, var, timefreq="daily", 
                                      varindata=Vardict[var]['VarInData'],
                                      filename=Vardict[var]['FileName'],
                                      rspmethod=Vardict[var]['RspMethod'],
                                      bufferzone=BufferZone)
            TPM.model_evaluation(case, var, caseds, refds, mask, cwrfarea, DataOutDir_var)
        varmask.append(mask)
    maskall = np.logical_and.reduce(varmask)
    savepath = f"{DataOutDir_var}/mask_all.nc"
    in_dict = {"mask": [["y","x"], maskall]}
    coords = {"y": caseds.y, "x": caseds.x}
    TIO.save_newnc(savepath=savepath, in_dict=in_dict, coords=coords)



# 极端事件分析
if Flag_ExtremeEvent_Calc:
    # Vardict中T2m和RH列表的交集
    print("")
    common_refs = set(Vardict['T2m']['refdata']) & set(Vardict['RH']['refdata'])
    mask = None
    refds = {}
    # 参考数据读取与mask制作
    for refname in common_refs:
        refds[refname] = {}
        t2m_ds, mask2d = TIO.read_RefData(RefPath[refname][0], StartTime, EndTime, refname, 'T2m', suffix='Yangtze_C', timefreq="daily", bufferzone=BufferZone)
        mask = (mask & mask2d) if mask is not None else mask2d
        mask = mask.fillna(False).astype(bool)
        rh_ds, mask2d = TIO.read_RefData(RefPath[refname][0], StartTime, EndTime, refname, 'RH', suffix='Yangtze_C', timefreq="daily", bufferzone=BufferZone)
        mask = (mask & mask2d) if mask is not None else mask2d
        mask = mask.fillna(False).astype(bool)
        refds[refname]['T2m'] = t2m_ds
        refds[refname]['RH'] = rh_ds
        rh_ds = refds[refname]['RH']
        t2m_ds = refds[refname]['T2m']
        TEEM.cal_extreme_event(refname, rh_ds, t2m_ds, mask, DataOutDir_var)

    for case in CasePath.keys():
        print(f"\n  ✯ Extreme Events: {case}")
        t2m_ds = TIO.read_CWRFPOST(CasePath[case], StartTime, EndTime, case, "T2m", timefreq="daily",
                                 varindata=Vardict['T2m']['VarInData'],
                                 filename=Vardict['T2m']['FileName'],
                                 rspmethod=Vardict['T2m']['RspMethod'],
                                 bufferzone=BufferZone)
        rh_ds = TIO.read_CWRFPOST(CasePath[case], StartTime, EndTime, case, "RH", timefreq="daily",
                                    varindata=Vardict['RH']['VarInData'],
                                    filename=Vardict['RH']['FileName'],
                                    rspmethod=Vardict['RH']['RspMethod'],
                                    bufferzone=BufferZone)
        TEEM.cal_extreme_event(case, rh_ds, t2m_ds, mask, DataOutDir_var)



########################################################
# 绘图部分
########################################################
#-- 季节验证 --
if Flag_Season_Plot:
    for varname in Vardict.keys():
        print(f"\n\n★ Seasonal Analysis: {varname}")
        for casename in CasePath.keys():
            print(f"   ✯ {casename} ")
            FMV.Plot_ModelValidation_Seasonal(casename, Vardict[varname]['refdata'], varname, lon2d, lat2d, DataOutDir_var, FigOutDir_var, lkinfos)
            FMV.Merge_ModelValidation_Seasonal(casename, Vardict[varname]['refdata'], varname, FigOutDir_var)


#-- 极端事件 --
if Flag_Extreme_Plot:
    print(f"\n\n★ Extreme Events Analysis")
    common_refs = set(Vardict['T2m']) & set(Vardict['RH'])
    common_refs = sorted(common_refs)
    for casename in CasePath.keys():
        print(f"   ✯ {casename} ")
        FMV.Plot_ModelValidation_Extreme(casename, common_refs, lon2d, lat2d, figfmt, DataOutDir_var, FigOutDir_var)
