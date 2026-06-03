#!/stu01/wumej22/Anaconda3/envs/cwrf_env/bin/python
import gc
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
import ToolBoxes.Tool_LoadData as TLD
import Func_WarmAdvection as FWA
import Func_RegClimImpact as FRCI
import Func_ExtremeEvent as FEE
import Func_PressureLevel as FPL
import Func_SpatialMap as FSM
import Func_Manuscript as FM
import Func_Review1 as FR1
import Func_CouplingRelationship as FCR
import Func_CalIndex as FCI
import Func_Additonal as FA

matplotlib.use('Agg')  # 不显示图，只保存
warnings.filterwarnings("ignore", category=RuntimeWarning)
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
pd.set_option("display.max_rows", None)
pd.set_option("display.expand_frame_repr", False)
time1 = time.time()

# ========== Option ==========
StartTime = "2000-01-01"
EndTime = "2024-12-31"

# 计算
Flag_RegClimImpact_Seasonal_Calc  = False    # 是否进行区域气候影响显著性检验
Flag_RegClimImpact_Diurnal_Calc   = False    # 是否进行区域气候影响显著性检验
Flag_RegClimImpact_Addtional_Info = False    # 是否进行区域气候影响显著性检验结果的额外信息计算（如受影响区域面积、受影响程度等）
Flag_WarmAdvection_Calc           = False    # 是否进行热平流计算
Flag_PressureLevel_Calc           = False    # 是否进行不同气压层计算
Flag_ExtremeEvent_Calc            = False     # 是否进行极端事件分析计算
Flag_ExtremeEvent_Test            = False     # 是否进行极端事件分析结果统计和显著性检验计算
Flag_ExtremeEvent_Addtional_Calc  = False     # 是否进行极端事件联合发生概率计算
Flag_CouplingTest_Calc            = False    # 是否进行耦合性检验计算

# 绘图
Flag_RegClimImpact_Plot           = False     # 是否进行区域气候影响显著性检验绘图
Flag_WarmAdvection_Plot           = False     # 是否进行热平流绘图  (不需要)
Flag_PressureLevel_Plot           = False     # 是否进行不同气压层绘图 
Flag_SpatialMap_Plot              = False      # 是否进行空间分布图绘图
Flag_ExtremeEventFreq_Plot        = False     # 是否进行极端事件分析绘图
Flag_ExtremeEventIntensity_Plot   = False     # 是否进行极端事件强度分析绘图 (不需要)
Flag_ExtremeEvent_Addtional_Plot  = False     # 是否进行极端事件联合发生概率绘图
Flag_CouplingTest_Plot            = False     # 是否进行耦合性检验绘图
Flag_Method_Plot                  = False     # 是否进行极端事件定义方法示意图绘图

# manuscript
Flag_Merge_Fig_1                  = False     # 是否合并图1: 区域气候效应图(气温diff, 气温diurnal, 相对湿度diff, 相对湿度diurnal)
Flag_Merge_Fig_2                  = False     # 是否合并图2: 影响机制图
Flag_Merge_Fig_3                  = False     # 是否合并图3: 合事件变化
Flag_Merge_Fig_4                  = False     # 是否合并图4: 热湿耦合图
Flag_Merge_Fig_5                  = False     # 是否合并图5: 
Flag_Merge_Fig_S1                 = False     # 是否合并图S1: 模式验证季节性图
Flag_Merge_Fig_S5                 = False     # 是否合并图S5: 不同季节LHF和SHF图
Flag_Merge_Fig_S6                 = False     # 是否合并图S6: 极端事件空间分布图
Flag_Merge_Fig_S7                 = False     # 是否合并图S7: CUOFF和NCP图
Flag_Merge_Fig_S8                 = False     # 是否合并图S8: 极端事件定义示意图

# Review 1
Flag_Merge_Review1_Figs           = True     # 是否合并Review1的图

Flag_Calculate_Indexes            = False     # 是否计算各类指数（CWRF技能指数，湖泊调节指数等）

# ========== Config ==========
BufferZone = 15  # 缓冲区宽度，单位：grid

# ========== Varlist ==========
vardict = {
    "T2m"      : {'VarInData': "AT2M" ,  "FileName": "AT2M" ,   "RspMethod": "mean",  "CheckMethod": "Paired_t-test",             },
    "T2m-Max"  : {'VarInData': "T2MAX",  "FileName": "T2MAX",   "RspMethod": "max",   "CheckMethod": "Paired_t-test",              },
    "T2m-Min"  : {'VarInData': "T2MIN",  "FileName": "T2MIN",   "RspMethod": "min",   "CheckMethod": "Paired_t-test",              },
    "LHF"      : {'VarInData': "ALFX" ,  "FileName": "ALFX" ,   "RspMethod": "mean",  "CheckMethod": "Paired_t-test",             },
    "SHF"      : {'VarInData': "AHFX" ,  "FileName": "AHFX" ,   "RspMethod": "mean",  "CheckMethod": "Paired_t-test",             },
    "RH"       : {'VarInData': "RH"   ,  "FileName": "RH"   ,   "RspMethod": "mean",  "CheckMethod": "Paired_t-test",             },
    "Prec"     : {'VarInData': "PRAVG",  "FileName": "PRAVG",   "RspMethod": "mean",  "CheckMethod": "Wilcoxon_signed-rank_test", },
    "Prec-Max" : {'VarInData': "PRMAX",  "FileName": "PRMAX",   "RspMethod": "max",   "CheckMethod": "Wilcoxon_signed-rank_test",  },
    "Q2m"      : {'VarInData': "AQ2M" ,  "FileName": "AQ2M" ,   "RspMethod": "mean",  "CheckMethod": "Paired_t-test",             },
    "T"        : {'VarInData': "tk",     "FileName": "temp",    "RspMethod": "mean",  "CheckMethod": "Paired_t-test",            },
    "U"        : {'VarInData': "u_met",  "FileName": "uv_met",  "RspMethod": "mean",  "CheckMethod": "Paired_t-test",            },
    "V"        : {'VarInData': "v_met",  "FileName": "uv_met",  "RspMethod": "mean",  "CheckMethod": "Paired_t-test",            },
    "UV"       : {'VarInData': "uv_met", "FileName": "uv_met",  "RspMethod": "mean",  "CheckMethod": "Paired_t-test",            },
    "Q"        : {'VarInData': "Q",      "FileName": "Q",       "RspMethod": "mean",  "CheckMethod": "Paired_t-test",            },
    "W"        : {'VarInData': "W",      "FileName": "W",       "RspMethod": "mean",  "CheckMethod": "Paired_t-test",            },
    "CloudFra" : {'VarInData': "CLDFRA", "FileName": "CLDFRA",  "RspMethod": "mean",  "CheckMethod": "Paired_t-test",            },
    "Height"   : {'VarInData': "height", "FileName": "height",  "RspMethod": "mean",  "CheckMethod": "Paired_t-test",            },
    "PBLH"     : {'VarInData': "PBLH",   "FileName": "PBLH",    "RspMethod": "mean",  "CheckMethod": "Paired_t-test",            },
    "Theta"    : {'VarInData': "theta",  "FileName": "temp",    "RspMethod": "mean",  "CheckMethod": "Paired_t-test",            },
    "U10"      : {'VarInData': "u_10",   "FileName": "uv_10",   "RspMethod": "mean",  "CheckMethod": "Paired_t-test",            },
    "V10"      : {'VarInData': "v_10",   "FileName": "uv_10",   "RspMethod": "mean",  "CheckMethod": "Paired_t-test",            },
    "UV10"     : {'VarInData': "uv_10",  "FileName": "uv_10",   "RspMethod": "mean",  "CheckMethod": "Paired_t-test",            },
}
Var2D = ["T2m", "LHF", "SHF", "RH", "Q2m", ] #"Prec", "Prec-Max", "UV10", "PBLH",  "T2m-Max", "T2m-Min", 
# Var2D = ["T2m","RH",]
# Var3D = [ "T", "W", "CloudFra", "Height", "Theta"]  #"U", "V", "T",
Var3D = [ "UV"]  #"U", "V", "T",
SpatialVar2D = [ "PBLH",  "UV",] #"PBLH", "UV10",  "UV", "UV10",
# SpatialVar2D = ["T2m", "T2m-Max", "T2m-Min", "LHF", "SHF", "RH", "Prec", "Prec-Max", "Q2m", "PBLH"]

pressure_levels=[1000, 975, 950, 925, 900, 875, 850, 825, 800, 750, 700, 650,
                 600, 550, 500, 450, 400, 350, 300, 250, 200, 150, 100, 50]

# test_levels=[1000, 975, 950, 925, 900, 875, 850, 825, 800, 750, 700, 650,
#                 600, 550, 500, 450, 400, 350, 300, 250, 200]
test_levels=[1000, ]

event_dicts = {
        'Cold':    ['Freq', 'exT_Cold',],
        'Dry':     ['Freq', 'exRH_Dry',],
        'Hot':     ['Freq', 'exT_Hot',],
        'Wet':     ['Freq', 'exRH_Wet',],
        'ColdDry': ['Freq', 'exTw_ColdDry',],
        'HotDry':  ['Freq', 'exTw_HotDry',],
        'ColdWet': ['Freq', 'exTw_ColdWet',],
        'HotWet':  ['Freq', 'exTw_HotWet',],
    }
eventslist = ['Cold', 'Dry', 'Hot', 'Wet', 'ColdDry', 'HotDry', 'ColdWet', 'HotWet']



#------------------------------------------------------------------------
# 基础配置
#------------------------------------------------------------------------
# 创建目录
# ========== Path ==========
CasePath = {
    "Lake": "/hydata01/wumej22/Experiment/CWRF/Yangtze_C/ERA5_history/Lake",
    "NoLake": "/hydata01/wumej22/Experiment/CWRF/Yangtze_C/ERA5_history/NoLake"
    }

DataOutDir = "/hydata01/wumej22/Experiment/Process/Yangtze_C_6km_r1/Cases"
FigOutDir = "/hydata01/wumej22/Experiment/Process/Yangtze_C_6km_r1/Figures"
cwrfinp_path = "/hydata01/wumej22/Experiment/Process/Yangtze_C_6km_r1/BaseData/wrfinput_d01.2000"

os.makedirs(DataOutDir, exist_ok=True)
os.makedirs(FigOutDir, exist_ok=True)
os.makedirs(f"{FigOutDir}/Review_1", exist_ok=True)
os.makedirs(f"{DataOutDir}/RegClimImpact", exist_ok=True)
os.makedirs(f"{DataOutDir}/ExtremeAnalysis", exist_ok=True)
os.makedirs(f"{DataOutDir}/ExtremeAnalysis/Excel", exist_ok=True)

# 开始验证
dx, dy = 6000, 6000  # 网格分辨率，单位：m
lon2d, lat2d, mapfac_mx, mapfac_my, scwater, lakedp = TIO.read_CWRF_Info(cwrfinp_path, bufferzone=BufferZone)
cwrfarea = TU.get_cwrf_grid_area(DX=dx, DY=dy, MAPFAC_MX=mapfac_mx, MAPFAC_MY=mapfac_my)
lkinfos = TU.get_lake_area_mask(scwater, lakedp, DX=dx, DY=dy, MAPFAC_MX=mapfac_mx, MAPFAC_MY=mapfac_my, nc_path=f"{DataOutDir}/lake_mask_with_dist.nc")
lkinfos['area'] = cwrfarea
cwrfarea_sum = np.nansum(cwrfarea[lkinfos['ocean']==0])
print(f"CWRF total area (excluding ocean): {cwrfarea_sum/1e6/1e4:.2f} x10^4 km²")

# 数据列表
IO_dict = {
    "StartTime": StartTime,
    "EndTime": EndTime,
    "CasePath": CasePath,
    "DataOutDir": DataOutDir,
    "FigOutDir": FigOutDir,
    "lkinfos": lkinfos,
    "BufferZone": BufferZone,
}

########################################################
# 计算部分
########################################################
if Flag_RegClimImpact_Seasonal_Calc:
    ''' 进行区域气候影响显著性检验计算 '''
    print("\n\nStart Regional Climate Impact Seasonal Significance Calculation...")
    for varname in Var2D:
        print(f"  Processing {varname} ...")
        if varname == "UV10":
            Lake_U10, NoLake_U10 = TLD.load_OneVar_AtSurface("U10", "seasonal", vardict, IO_dict=IO_dict) 
            Lake_V10, NoLake_V10 = TLD.load_OneVar_AtSurface("V10", "seasonal", vardict, IO_dict=IO_dict) 
            Lake_data = np.hypot(Lake_U10, Lake_V10).rename("UV10")
            NoLake_data = np.hypot(NoLake_U10, NoLake_V10).rename("UV10")
            print(type(Lake_data), Lake_data.time)
            del Lake_U10, NoLake_U10, Lake_V10, NoLake_V10
        else:
            Lake_data, NoLake_data = TLD.load_OneVar_AtSurface(varname, "seasonal", vardict, IO_dict=IO_dict) 
        FRCI.RegClimImpactSignificanceOfChange_seasonal( xarr1=Lake_data, xarr2=NoLake_data,
                                                       var=varname,  outdir=DataOutDir,
                                                       caselist=["Lake","NoLake"], 
                                                       checkmethod=vardict[varname]['CheckMethod'],
                                                       rspmethod=vardict[varname]['RspMethod'])
        del Lake_data, NoLake_data
        gc.collect()
    print(f"   √ Completed Regional Climate Impact Seasonal Significance Calculation.\n\n")



if Flag_RegClimImpact_Diurnal_Calc:
    ''' 进行区域气候影响显著性检验计算 '''
    print("\n\nStart Regional Climate Impact Diurnal Significance Calculation...")
    for varname in Var2D:
        print(f"  Processing {varname} ...")
        if varname == "UV10":
            Lake_U10, NoLake_U10 = TLD.load_OneVar_AtSurface("U10", "seasonal_diurnal", vardict, IO_dict=IO_dict) 
            Lake_V10, NoLake_V10 = TLD.load_OneVar_AtSurface("V10", "seasonal_diurnal", vardict, IO_dict=IO_dict) 
            Lake_data = np.hypot(Lake_U10, Lake_V10).rename("UV10")
            NoLake_data = np.hypot(NoLake_U10, NoLake_V10).rename("UV10")
            print(type(Lake_data), Lake_data.time)
            del Lake_U10, NoLake_U10, Lake_V10, NoLake_V10
        else:
            Lake_data, NoLake_data = TLD.load_OneVar_AtSurface(varname, "seasonal_diurnal", vardict, IO_dict=IO_dict) 
        FRCI.RegClimImpactSignificanceOfChange_diurnal(xarr1=Lake_data, xarr2=NoLake_data,
                                                      var=varname, outdir=DataOutDir,
                                                      caselist=["Lake","NoLake"],
                                                      checkmethod=vardict[varname]['CheckMethod'],
                                                      rspmethod=vardict[varname]['RspMethod'])
        del Lake_data, NoLake_data
        gc.collect()
    print(f"   √ Completed Regional Climate Impact Diurnal Significance Calculation.\n\n")



if Flag_RegClimImpact_Addtional_Info:
    print("\n\nStart Regional Climate Impact Significance Testing Info...")
    FigOutDir_var = f"{FigOutDir}/RegClimImpact"
    os.makedirs(FigOutDir_var, exist_ok=True)
    for varname in Var2D:
        print(f"  Processing {varname} ...")
        checkmethod = vardict[varname]['CheckMethod']
        FA.RegClimImpact_Additional(
            varname=varname, caselist=["Lake","NoLake"],
            checkmethod=checkmethod,
            OutDir=DataOutDir, FigOutDir=FigOutDir_var,
            lkinfos=lkinfos,
            onlysig=True
        )




if Flag_PressureLevel_Calc:
    ''' 不同气压层绘图 '''
    print("\n\nStart Pressure Level Calculation...")
    FigOutDir_level = f"{FigOutDir}/PressureLevel"
    os.makedirs(FigOutDir_level, exist_ok=True)
    for level in test_levels:
        for varname in Var3D:
            print(f"  Processing at {level} hPa ...")
            if varname == "UV":
                Lake_U, NoLake_U = TLD.load_OneVar_AtPressure("U", "seasonal", vardict, pressure_levels, IO_dict=IO_dict, press_level=level)
                Lake_V, NoLake_V = TLD.load_OneVar_AtPressure("V", "seasonal", vardict, pressure_levels, IO_dict=IO_dict, press_level=level)
                Lake_data = np.hypot(Lake_U, Lake_V).rename("UV")
                NoLake_data = np.hypot(NoLake_U, NoLake_V).rename("UV")
                print(type(Lake_data), Lake_data.time)
                del Lake_U, NoLake_U, Lake_V, NoLake_V
            else:
                Lake_data, NoLake_data = TLD.load_OneVar_AtPressure(varname, "seasonal", vardict, pressure_levels, IO_dict=IO_dict, press_level=level)
            FPL.PressureLevelSignificanceOfChange_seasonal(xarr1=Lake_data, xarr2=NoLake_data, level=level,
                                                           checkmethod=vardict[varname]['CheckMethod'],
                                                           var=varname, OutDir=DataOutDir,
                                                           caselist=["Lake","NoLake"],
                                                           rspmethod=vardict[varname]['RspMethod'])
            del Lake_data, NoLake_data
            gc.collect()
        print(f"   √ Completed plotting at {level} hPa.\n")

    for level in test_levels:
        for varname in Var3D:
            print(f"  Processing at {level} hPa ...")
            if varname == "UV":
                Lake_U, NoLake_U = TLD.load_OneVar_AtPressure("U", "seasonal_diurnal", vardict, pressure_levels, IO_dict=IO_dict, press_level=level)
                Lake_V, NoLake_V = TLD.load_OneVar_AtPressure("V", "seasonal_diurnal", vardict, pressure_levels, IO_dict=IO_dict, press_level=level)
                Lake_data = np.hypot(Lake_U, Lake_V).rename("UV")
                NoLake_data = np.hypot(NoLake_U, NoLake_V).rename("UV")
                print(type(Lake_data), Lake_data.time)
                del Lake_U, NoLake_U, Lake_V, NoLake_V
            else:
                Lake_data, NoLake_data = TLD.load_OneVar_AtPressure(varname, "seasonal_diurnal", vardict, pressure_levels, IO_dict=IO_dict, press_level=level)
            FPL.PressureLevelSignificanceOfChange_diurnal(xarr1=Lake_data, xarr2=NoLake_data, level=level,
                                                           checkmethod=vardict[varname]['CheckMethod'],
                                                           var=varname, OutDir=DataOutDir,
                                                           caselist=["Lake","NoLake"],
                                                           rspmethod=vardict[varname]['RspMethod'])
            del Lake_data, NoLake_data
            gc.collect()
        print(f"   √ Completed plotting at {level} hPa.\n")

    # Lake_Theta, NoLake_Theta = TLD.load_OneVar_AtPressure("Theta", "seasonal", vardict, pressure_levels, IO_dict=IO_dict, press_level=pressure_levels)
    # Lake_Height, NoLake_Height = TLD.load_OneVar_AtPressure("Height", "seasonal", vardict, pressure_levels, IO_dict=IO_dict, press_level=pressure_levels)
    # Lake_Temp, NoLake_Temp = TLD.load_OneVar_AtPressure("T", "seasonal", vardict, pressure_levels, IO_dict=IO_dict, press_level=pressure_levels)
    # Lake = xr.merge([Lake_Theta, Lake_Height, Lake_Temp])
    # NoLake = xr.merge([NoLake_Theta, NoLake_Height, NoLake_Temp])
    # FPL.Cal_StaticStability_and_DThetaDz(caselist=["Lake","NoLake"], case1=Lake, case2=NoLake, pressures=pressure_levels, outdir=DataOutDir)



if Flag_ExtremeEvent_Calc:
    ''' 进行极端事件分析计算 '''
    print("\n\nStart Extreme Event Calculation...")
    LakeT2m, NoLakeT2m = TLD.load_OneVar_AtSurface("T2m", "daily", vardict, IO_dict=IO_dict) 
    LakeRH, NoLakeRH = TLD.load_OneVar_AtSurface("RH", "daily", vardict, IO_dict=IO_dict) 
    NoLake = xr.merge([NoLakeT2m, NoLakeRH])
    Lake = xr.merge([LakeT2m, LakeRH])
    del LakeT2m, LakeRH, NoLakeT2m, NoLakeRH,
    FEE.CalExtremeEventThres(xarr=NoLake, refname="NoLake", outdir=DataOutDir)
    FEE.IdentifyExtremeEvents(xarr_in=NoLake, xarr_ref=NoLake, caselist=["NoLake", "NoLake"], outdir=DataOutDir)
    FEE.IdentifyExtremeEvents(xarr_in=Lake, xarr_ref=NoLake, caselist=["Lake", "NoLake"], outdir=DataOutDir)
    FEE.CountExtremeEvents(caselist=["NoLake", "NoLake"], outdir=DataOutDir)
    FEE.CountExtremeEvents(caselist=["Lake", "NoLake"], outdir=DataOutDir)
    # FEE.Calculate_seasonal_extreme_Intensity(caselist=["NoLake", "NoLake"], outdir=DataOutDir)
    # FEE.Calculate_seasonal_extreme_Intensity(caselist=["Lake", "NoLake"], outdir=DataOutDir)
    FEE.Calculate_extreme_Intensity(casename="NoLake", xarr_in=NoLake, refname="NoLake", outdir=DataOutDir)
    FEE.Calculate_extreme_Intensity(casename="Lake", xarr_in=Lake, refname="NoLake", outdir=DataOutDir)
    del Lake, NoLake
    gc.collect()
    print(f"   √ Completed Extreme Event Calculation.\n\n")



if Flag_ExtremeEvent_Test:
    ''' 进行极端事件分析结果统计和显著性检验计算 '''
    print("\n\nStart Extreme Event Analysis Significance Testing...")
    filepath = f"{DataOutDir}/ExtremeAnalysis/Extreme_Events_Lake_ref_NoLake_seasonal_event_count.nc"
    Lake = TIO.read_newnc(filepath)
    filepath = f"{DataOutDir}/ExtremeAnalysis/Extreme_Events_NoLake_ref_NoLake_seasonal_event_count.nc"
    NoLake = TIO.read_newnc(filepath)
    FEE.SignificanceOfExtremeEvents_Freq(xarr1=Lake, xarr2=NoLake, events=eventslist, caselist=["Lake", "NoLake"], outdir=DataOutDir)
    # for event in eventslist:
    #     event_vars = event_dicts[event][1:]
    #     filepath = f"{DataOutDir}/ExtremeAnalysis/Intensity_{event}_exT_exRH_exTw_Lake_ref_NoLake_seasonmean.nc"
    #     Lake = TIO.read_newnc(filepath)
    #     filepath = f"{DataOutDir}/ExtremeAnalysis/Intensity_{event}_exT_exRH_exTw_NoLake_ref_NoLake_seasonmean.nc"
    #     NoLake = TIO.read_newnc(filepath)
    #     FEE.SignificanceOfExtremeEvents_Intensity(xarr1=Lake, xarr2=NoLake, event=event, event_vars=event_vars, caselist=["Lake", "NoLake"], outdir=DataOutDir)
    gc.collect()
    print(f"   √ Completed Extreme Event Analysis.\n\n")



if Flag_ExtremeEvent_Addtional_Calc:
    ''' 进行极端事件联合发生概率计算 '''
    print("\n\nStart Extreme Event Additional Calculation...")
    # FEE.compute_hotwet_probability(caselist=["Lake","NoLake"], checkmethod="Wilcoxon_signed-rank_test", OutDir=DataOutDir, lkinfos=lkinfos)
    FEE.ComputeWetGivenHotProbability(caselist=["Lake","NoLake"], outdir=DataOutDir, refcase="NoLake")
    # FEE.compute_affected_population(caselist=["Lake","NoLake"], OutDir=DataOutDir, checkmethod="Wilcoxon_signed-rank_test", lkinfos=lkinfos)
    # FEE.summarize_affected_population_by_city(caselist=["Lake","NoLake"], OutDir=DataOutDir, checkmethod="Wilcoxon_signed-rank_test", lkinfos=lkinfos)
    print(f"   √ Completed Extreme Event Additional Calculation.\n\n")



if Flag_CouplingTest_Calc:
    ''' 进行耦合性检验计算 '''
    print("\n\nStart Coupling Test Calculation...")
    outdir = f"{DataOutDir}/CouplingTest"
    os.makedirs(outdir, exist_ok=True)
    # 读取数据
    LakeT2m, NoLakeT2m = TLD.load_OneVar_AtSurface("T2m", "daily", vardict, IO_dict=IO_dict) 
    LakeRH, NoLakeRH = TLD.load_OneVar_AtSurface("RH", "daily", vardict, IO_dict=IO_dict) 
    LakeQ2m, NoLakeQ2m = TLD.load_OneVar_AtSurface("Q2m", "daily", vardict, IO_dict=IO_dict) 
    NoLake = xr.merge([NoLakeT2m, NoLakeRH, NoLakeQ2m])
    Lake = xr.merge([LakeT2m, LakeRH, LakeQ2m])
    FCR.CalCouplingMetrics(casename="NoLake", xarr=NoLake, var1="T2m", var2="RH", outdir=outdir)
    FCR.CalCouplingMetrics(casename="NoLake", xarr=NoLake, var1="T2m", var2="Q2m", outdir=outdir)
    FCR.CalCouplingMetrics(casename="Lake", xarr=Lake, var1="T2m", var2="RH", outdir=outdir)
    FCR.CalCouplingMetrics(casename="Lake", xarr=Lake, var1="T2m", var2="Q2m", outdir=outdir)
    FCR.CalCouplingMetrics_Analyze(caselist=["Lake", "NoLake"], var1="T2m", var2="RH", outdir=outdir, lkinfos=lkinfos)
    FCR.CalCouplingMetrics_Analyze(caselist=["Lake", "NoLake"], var1="T2m", var2="Q2m", outdir=outdir, lkinfos=lkinfos)
    # FCR.CalUpperTailDependence(casename="NoLake", refname="NoLake", event1="Hot", event2="HotWet", DataOutDir=DataOutDir, outdir=outdir)
    # FCR.CalUpperTailDependence(casename="NoLake", refname="NoLake", event1="Wet", event2="HotWet", DataOutDir=DataOutDir, outdir=outdir)
    # FCR.CalUpperTailDependence(casename="Lake", refname="NoLake", event1="Hot", event2="HotWet", DataOutDir=DataOutDir, outdir=outdir)
    # FCR.CalUpperTailDependence(casename="Lake", refname="NoLake", event1="Wet", event2="HotWet", DataOutDir=DataOutDir, outdir=outdir)
    # FCR.TailDependence_Analyze(caselist=["Lake", "NoLake"], event1="Hot", event2="HotWet", outdir=outdir, lkinfos=lkinfos)
    # FCR.TailDependence_Analyze(caselist=["Lake", "NoLake"], event1="Wet", event2="HotWet", outdir=outdir, lkinfos=lkinfos)
    # FCR.Normalization_Variable(casename="NoLake", xarr_in=NoLake, var="T2m", timefreq="daily", outdir=outdir)
    # FCR.Normalization_Variable(casename="NoLake", xarr_in=NoLake, var="RH", timefreq="daily", outdir=outdir)
    # FCR.Normalization_Variable(casename="NoLake", xarr_in=NoLake, var="Q2m", timefreq="daily", outdir=outdir)
    # FCR.Normalization_Variable(casename="Lake", xarr_in=Lake, var="T2m", timefreq="daily", outdir=outdir)
    # FCR.Normalization_Variable(casename="Lake", xarr_in=Lake, var="RH", timefreq="daily", outdir=outdir)
    # FCR.Normalization_Variable(casename="Lake", xarr_in=Lake, var="Q2m", timefreq="daily", outdir=outdir)
    del Lake, NoLake, LakeT2m, NoLakeT2m, LakeRH, NoLakeRH, LakeQ2m, NoLakeQ2m
    gc.collect()
    print(f"   √ Completed Coupling Test Calculation.\n\n")


if Flag_WarmAdvection_Calc:
    ''' 进行热平流分析 需要读取T, U, V三个变量 '''    
    print("\n\nStart Warm Advection Calculation...")
    for level in test_levels:
        print(f"  Processing at {level} hPa ...")
        Lake_T, NoLake_T = TLD.load_OneVar_AtPressure("T", "daily", vardict, pressure_levels, IO_dict=IO_dict, press_level=level)
        Lake_U, NoLake_U = TLD.load_OneVar_AtPressure("U", "daily", vardict, pressure_levels, IO_dict=IO_dict, press_level=level)
        Lake_V, NoLake_V = TLD.load_OneVar_AtPressure("V", "daily", vardict, pressure_levels, IO_dict=IO_dict, press_level=level)
        Lake = {"T": Lake_T, "U": Lake_U, "V": Lake_V}
        NoLake = {"T": NoLake_T, "U": NoLake_U, "V": NoLake_V}
        FWA.WarmAdvectionSignificanceOfChange_seasonal(caselist=["Lake","NoLake"], case1=Lake, case2=NoLake,
                                                    dx=dx, dy=dy, checkmethod='Paired_t-test', outdir=DataOutDir, level=level)
        del Lake_T, Lake_U, Lake_V, NoLake_T, NoLake_U, NoLake_V, Lake, NoLake
        gc.collect()
    print(f"   √ Completed Warm Advection Calculation.\n\n")






########################################################
# 绘图部分
########################################################
if Flag_RegClimImpact_Plot:
    ''' 区域气候影响显著性检验绘图 '''
    print("\n\nStart Regional Climate Impact Significance Testing Plotting...")
    FigOutDir_var = f"{FigOutDir}/RegClimImpact"
    os.makedirs(FigOutDir_var, exist_ok=True)
    for varname in Var2D:
        print(f"  Processing {varname} ...")
        checkmethod = vardict[varname]['CheckMethod']
        FRCI.Plot_RegClimImpact(
            varname=varname, caselist=["Lake","NoLake"],
            lon2d=lon2d, lat2d=lat2d,
            checkmethod=checkmethod,
            OutDir=DataOutDir,
            FigOutDir=FigOutDir_var,
            lkinfos=lkinfos,
            onlysig=True
        )
        FRCI.Depth_Dependece_Impacts(
            varname=varname, caselist=["Lake","NoLake"],
            lon2d=lon2d, lat2d=lat2d,
            checkmethod=checkmethod,
            OutDir=DataOutDir,
            FigOutDir=FigOutDir_var,
            lkinfos=lkinfos,
            onlysig=True
        )
        # Lake, NoLake = TLD.load_OneVar_AtSurface(varname, "daily", vardict, IO_dict=IO_dict) 
        # FRCI.probability_density_distribution(
        #     varname=varname, caselist=["Lake","NoLake"],
        #     xarr1=Lake, xarr2=NoLake,
        #     checkmethod=checkmethod,
        #     OutDir=DataOutDir,
        #     FigOutDir=FigOutDir_var,
        #     lkinfos=lkinfos,
        #     onlysig=True
        # )
        FRCI.Merge_RegClimImpact(varname=varname, caselist=["Lake","NoLake"],
                                 checkmethod=checkmethod, FigOutDir=FigOutDir_var,
                                 onlysig=True)
        print(f"   √ Completed plotting for {varname}.\n")



if Flag_PressureLevel_Plot:
    ''' 不同气压层绘图 '''
    print("\n\nStart Pressure Level Plotting...")
    FigOutDir_level = f"{FigOutDir}/PressureLevel"
    os.makedirs(FigOutDir_level, exist_ok=True)
    for level in test_levels:
        print(f"  Processing at {level} hPa ...")
        for varname in Var3D:
            checkmethod = vardict[varname]['CheckMethod']
            FPL.Plot_PressureLevel(
                varname=varname, caselist=["Lake","NoLake"], level=level, 
                lon2d=lon2d, lat2d=lat2d,
                checkmethod=checkmethod,
                OutDir=DataOutDir,
                FigOutDir=FigOutDir_level,
                lkinfos=lkinfos,
                onlysig=True
            )
            # FPL.Merge_PressureLevel(varname=varname, caselist=["Lake","NoLake"], level=level, 
            #                         checkmethod=checkmethod, FigOutDir=FigOutDir_level, onlysig=True)
                                    
        print(f"   √ Completed plotting at {level} hPa for {varname}.\n")

    # FPL.Plot_StaticStability_and_DThetaDz(caselist=["Lake","NoLake"],levels=pressure_levels, OutDir=DataOutDir, FigOutDir=FigOutDir, lkinfos=lkinfos)



if Flag_SpatialMap_Plot:
    ''' 空间分布图绘图 '''
    print("\n\nStart Spatial Map Plotting...")
    FigOutDir_map = f"{FigOutDir}/SpatialMap"
    os.makedirs(FigOutDir_map, exist_ok=True)
    for varname in SpatialVar2D:
        print(f"  Processing {varname} ...")
        # checkmethod = vardict[varname]['CheckMethod']
        # FSM.Plot_Diurnal_Cycle(caselist=["Lake","NoLake"], varname=varname,
        #                        checkmethod=checkmethod, OutDir=DataOutDir,
        #                        FigOutDir=FigOutDir_map, lkinfos=lkinfos)
        for casename in ["Lake","NoLake"]:
            if varname == "UV10":
                Lake_U10, NoLake_U10 = TLD.load_OneVar_AtSurface("U10", "seasonal", vardict, IO_dict=IO_dict) 
                Lake_V10, NoLake_V10 = TLD.load_OneVar_AtSurface("V10", "seasonal", vardict, IO_dict=IO_dict) 
                FSM.Plot_SpatialMap_UV(
                    casename = casename,
                    varname = varname,
                    U_data = Lake_U10 if casename=="Lake" else NoLake_U10,
                    V_data = Lake_V10 if casename=="Lake" else NoLake_V10,
                    lon2d=lon2d, lat2d=lat2d,
                    FigOutDir=FigOutDir_map,
                    lkinfos=lkinfos,
                )
                FSM.Merge_SpatialMap_2DVar(
                    casename = casename,
                    varname = varname,
                    FigOutDir=FigOutDir_map,
                )
            elif varname == "UV":
                for level in test_levels:
                    Lake_U, NoLake_U = TLD.load_OneVar_AtPressure("U", "seasonal", vardict, pressure_levels, IO_dict=IO_dict, press_level=level)
                    Lake_V, NoLake_V = TLD.load_OneVar_AtPressure("V", "seasonal", vardict, pressure_levels, IO_dict=IO_dict, press_level=level)
                    FSM.Plot_SpatialMap_UV(
                        casename = casename,
                        varname = varname,
                        U_data = Lake_U if casename=="Lake" else NoLake_U,
                        V_data = Lake_V if casename=="Lake" else NoLake_V,
                        lon2d=lon2d, lat2d=lat2d,
                        FigOutDir=FigOutDir_map,
                        lkinfos=lkinfos,
                        level = level,
                    )
                    FSM.Merge_SpatialMap_2DVar(
                        casename = casename,
                        varname = varname,
                        FigOutDir=FigOutDir_map,
                        level = level,
                    )
            else:
                Lake, NoLake = TLD.load_OneVar_AtSurface(varname, "seasonal", vardict, IO_dict=IO_dict) 
                FSM.Plot_SpatialMap_2DVar(
                    casename = casename,
                    varname = varname,
                    xarr_in = Lake if casename=="Lake" else NoLake,
                    lon2d=lon2d, lat2d=lat2d,
                    FigOutDir=FigOutDir_map,
                    lkinfos=lkinfos,
                )
                FSM.Merge_SpatialMap_2DVar(
                    casename = casename,
                    varname = varname,
                    FigOutDir=FigOutDir_map,
                )

        print(f"   √ Completed plotting for {varname}.\n")



if Flag_ExtremeEventFreq_Plot:
    ''' 极端事件分析绘图 '''
    print("\n\nStart Extreme Event Frequency Plotting...")
    FigOutDir_event = f"{FigOutDir}/ExtremeEventFreq"
    os.makedirs(FigOutDir_event, exist_ok=True)
    FEE.plot_extreme_event_define(FigOutDir=FigOutDir_event)
    for event in eventslist:
        print(f"  Processing {event} ...")
        event_vars = event_dicts[event][1:]
        FEE.Plot_ExtremeEvents_Freq(
            event=event, event_vars=event_vars, caselist=["Lake","NoLake"],
            lon2d=lon2d, lat2d=lat2d,
            checkmethod="Wilcoxon_signed-rank_test",
            OutDir=DataOutDir,
            FigOutDir=FigOutDir_event,
            lkinfos=lkinfos,
            onlysig=True
        )
    #     print(f"   √ Completed plotting for {event}.\n")
    FEE.Plot_ExtremeEvents_Additional_Freq(event_dicts, caselist=["Lake","NoLake"],
                                            lon2d=lon2d, lat2d=lat2d,
                                            checkmethod="Wilcoxon_signed-rank_test",
                                            OutDir=DataOutDir, FigOutDir=FigOutDir_event,
                                            lkinfos=lkinfos, onlysig=True)
    FEE.Merge_ExtremeEvents_Freq(event_dicts, FigOutDir=FigOutDir_event,
                                 checkmethod="Wilcoxon_signed-rank_test", onlysig=True)

    # ------------ Liang commits ------------
    FEE.Plot_ExtremeEvents_Freq_New(caselist=["Lake","NoLake"],
                                    checkmethod="Wilcoxon_signed-rank_test",
                                    OutDir=DataOutDir, FigOutDir=FigOutDir_event,
                                    lkinfos=lkinfos, onlysig=True)
    ### FEE.Plot_ExtremeEvents_Freq_Liang(caselist=["Lake","NoLake"],
    ###                                 checkmethod="Wilcoxon_signed-rank_test",
    ###                                 OutDir=DataOutDir, FigOutDir=FigOutDir_event,
    ###                                 lkinfos=lkinfos, onlysig=True)
    FEE.Merge_Plot_ExtremeEvents_Freq_New(caselist=["Lake","NoLake"],
                                        checkmethod="Wilcoxon_signed-rank_test",
                                        OutDir=DataOutDir,
                                         FigOutDir=FigOutDir_event,
                                            lkinfos=lkinfos,
                                         onlysig=True)
    FEE.Plot_ExtremeEvents_Freq_HeatMap(caselist=["Lake","NoLake"],
                                        checkmethod="Wilcoxon_signed-rank_test",
                                        OutDir=DataOutDir, FigOutDir=FigOutDir_event,
                                        lkinfos=lkinfos, onlysig=True)
    FEE.Merge_Plot_ExtremeEvents_Freq_HeatMap(caselist=["Lake","NoLake"],
                                        checkmethod="Wilcoxon_signed-rank_test",
                                        OutDir=DataOutDir,
                                         FigOutDir=FigOutDir_event,
                                            lkinfos=lkinfos,
                                         onlysig=True)
    FEE.PlotWetGivenHotProbabilityDiff(caselist=["Lake","NoLake"],
                                       lon2d=lon2d, lat2d=lat2d,
                                       lkinfos=lkinfos, outdir=DataOutDir, 
                                       figoutdir=FigOutDir_event,
                                       refcase="NoLake")



if Flag_ExtremeEventIntensity_Plot:
    ''' 极端事件分析绘图 '''
    print("\n\nStart Extreme Event Intensity Plotting...")
    FigOutDir_event = f"{FigOutDir}/ExtremeEventIntensity"
    os.makedirs(FigOutDir_event, exist_ok=True)
    # for event in eventslist:
    #     print(f"  Processing {event} ...")
    #     event_vars = event_dicts[event][1:]
    #     FEE.Plot_ExtremeEvents_Intensity(
    #         event=event, event_vars=event_vars, caselist=["Lake","NoLake"],
    #         lon2d=lon2d, lat2d=lat2d,
    #         checkmethod="Wilcoxon_signed-rank_test",
    #         OutDir=DataOutDir,
    #         FigOutDir=FigOutDir_event,
    #         lkinfos=lkinfos,
    #         onlysig=True
    #     )
    #     print(f"   √ Completed plotting for {event}.\n")
    # FEE.Plot_ExtremeEvents_Additional_Intensity(event_dicts, caselist=["Lake","NoLake"],
    #                                         lon2d=lon2d, lat2d=lat2d,
    #                                         checkmethod="Wilcoxon_signed-rank_test",
    #                                         OutDir=DataOutDir, FigOutDir=FigOutDir_event,
    #                                         lkinfos=lkinfos, onlysig=True)
    # FEE.Merge_ExtremeEvents_Intensity(event_dicts, FigOutDir=FigOutDir_event,
    #                                  checkmethod="Wilcoxon_signed-rank_test", onlysig=True)


if Flag_ExtremeEvent_Addtional_Plot:
    ''' 极端事件分析绘图 '''
    print("\n\nStart Extreme Event Additional Plotting...")
    FigOutDir_event = f"{FigOutDir}/ExtremeEventAddtional"
    os.makedirs(FigOutDir_event, exist_ok=True)
    # FEE.Plot_HotWet_Probability(
    #     caselist=["Lake","NoLake"],
    #     lon2d=lon2d, lat2d=lat2d,
    #     checkmethod="Wilcoxon_signed-rank_test",
    #     OutDir=DataOutDir,
    #     FigOutDir=FigOutDir_event,
    #     lkinfos=lkinfos,
    #     onlysig=True
    # )
    # FEE.Merge_HotWet_Probability(
    #     caselist=["Lake","NoLake"],
    #     FigOutDir=FigOutDir_event
    # )
    FEE.plot_affected_population(
        caselist=["Lake","NoLake"],
        lon2d=lon2d, lat2d=lat2d,
        OutDir=DataOutDir,
        checkmethod="Wilcoxon_signed-rank_test",
        lkinfos=lkinfos,
        FigOutDir=FigOutDir_event,
    )


if Flag_WarmAdvection_Plot:
    ''' 热平流绘图 '''
    print("\n\nStart Warm Advection Plotting...")
    figoutdir = f"{FigOutDir}/WarmAdvection"
    os.makedirs(figoutdir, exist_ok=True)
    for level in test_levels:
        FWA.Plot_WarmAdvection(caselist=["Lake","NoLake"], level=level, lon2d=lon2d, lat2d=lat2d,
                               checkmethod="Paired_t-test", OutDir=DataOutDir, FigOutDir=figoutdir, 
                               lkinfos=lkinfos, onlysig=True)
        FWA.Merge_WarmAdvectionSignificance(FigOutDir=figoutdir, caselist=["Lake","NoLake"],
                                            level=level, checkmethod="Paired_t-test", onlysig=True)


if Flag_CouplingTest_Plot:
    ''' 耦合性检验绘图 '''
    print("\n\nStart Coupling Test Plotting...")
    outdir = f"{DataOutDir}/CouplingTest"
    figoutdir = f"{FigOutDir}/CouplingTest"
    os.makedirs(figoutdir, exist_ok=True)
    FCR.Plot_CouplingMetrics_Analyze(caselist=["Lake", "NoLake"], var1="T2m", var2="RH", lon2d=lon2d, lat2d=lat2d, lkinfos=lkinfos, outdir=outdir, figoutdir=figoutdir)
    # FCR.Plot_CouplingMetrics_Analyze(caselist=["Lake", "NoLake"], var1="T2m", var2="Q2m", lon2d=lon2d, lat2d=lat2d, lkinfos=lkinfos, outdir=outdir, figoutdir=figoutdir)
    # FCR.Merge_CouplingMetrics_Cases(caselist=["Lake", "NoLake"], var1="T2m", var2="RH", figoutdir=figoutdir)
    # FCR.Merge_CouplingMetrics_Cases(caselist=["Lake", "NoLake"], var1="T2m", var2="Q2m", figoutdir=figoutdir)
    # FCR.Plot_Coupling_state(caselist=["Lake","NoLake"], var1="T2m", var2="RH", timefreq="daily", checkmethod="Wilcoxon_signed-rank_test", lon2d=lon2d, lat2d=lat2d, OutDir=DataOutDir, FigOutDir=figoutdir,lkinfos=lkinfos,onlysig=True)
    FEE.Plot_exT_exRH_Coupling_state(
        caselist=["Lake","NoLake"],
        refname="NoLake",
        lon2d=lon2d, lat2d=lat2d, 
        OutDir=DataOutDir,
        FigOutDir=figoutdir,
        checkmethod="Wilcoxon_signed-rank_test",
        lkinfos=lkinfos,
        onlysig=True
    )



if Flag_Method_Plot:
    print("\n\nStart Method Plotting...")
    # FEE.Plot_method_of_define_extreme_event(OutDir=DataOutDir, FigOutDir=FigOutDir)
    FEE.Plot_method_of_define_extreme_event_New(OutDir=DataOutDir, FigOutDir=FigOutDir)





#############################
# Manuscript Figures
#############################
if Flag_Merge_Fig_1:
    ''' 合并图1 '''
    print("\nStart Merging Figure 1...")
    FM.Merge_Fig_1_RegClimImpact(FigOutDir=FigOutDir)
    print(f"   √ Completed Merging Figure 1.\n\n")


if Flag_Merge_Fig_2:
    ''' 合并图2 '''
    print("\nStart Merging Figure 2...")
    FM.Merge_Fig_2_Influence_Mechanism(FigOutDir=FigOutDir)
    print(f"   √ Completed Merging Figure 2.\n\n")


if Flag_Merge_Fig_3:
    ''' 合并图3 '''
    print("\nStart Merging Figure 3...")
    FM.Merge_Fig_3_ExtremeEventFreq(FigOutDir=FigOutDir)
    print(f"   √ Completed Merging Figure 3.\n\n")


if Flag_Merge_Fig_4:
    ''' 合并图4 '''
    print("\nStart Merging Figure 4...")
    FM.Merge_Fig_4_HotWetCoupling(FigOutDir=FigOutDir)
    print(f"   √ Completed Merging Figure 4.\n\n")


if Flag_Merge_Fig_5:
    ''' 合并图5 '''
    print("\nStart Merging Figure 5...")
    FM.Merge_Fig_5_AffectedPopulation(FigOutDir=FigOutDir)
    print(f"   √ Completed Merging Figure 5.\n\n")


if Flag_Merge_Fig_S1:
    ''' 合并图S1 '''
    print("\nStart Merging Figure S1...")
    FM.Merge_Fig_S1_ModelValidation(FigOutDir=FigOutDir)
    print(f"   √ Completed Merging Figure S1.\n\n")


if Flag_Merge_Fig_S5:
    ''' 合并图S5 '''
    print("\nStart Merging Figure S5...")
    FM.Merge_Fig_S5_RegClimImpact(FigOutDir=FigOutDir)
    print(f"   √ Completed Merging Figure S5.\n\n")


if Flag_Merge_Fig_S6:
    ''' 合并图S6 '''
    print("\nStart Merging Figure S6...")
    FM.Merge_Fig_S6_ExtremeEvents(FigOutDir=FigOutDir)
    print(f"   √ Completed Merging Figure S6.\n\n")


if Flag_Merge_Fig_S7:
    ''' 合并图S7 '''
    print("\nStart Merging Figure S7...")
    FM.Merge_Fig_S7_CUOFF_NCP(FigOutDir=FigOutDir)
    print(f"   √ Completed Merging Figure S7.\n\n")


if Flag_Merge_Fig_S8:
    ''' 合并图S8 '''
    print("\nStart Merging Figure S8...")
    FM.Merge_Fig_S8_define_extreme_event(FigOutDir=FigOutDir)
    print(f"   √ Completed Merging Figure S8.\n\n")




if Flag_Merge_Review1_Figs:
    ''' 合并Review1的图 '''
    print("\nStart Merging Review 1 Figures...")
    FR1.make_figdir(FigOutDir=FigOutDir)
    review1_figs = [
        ("Figure_1", FR1.Fig_1),
        ("Figure_2", FR1.Fig_2),
        ("Figure_3", FR1.Fig_3),
        ("Figure_4", FR1.Fig_4),
        ("Figure_5", FR1.Fig_5),
        ("Figure_S1", FR1.Fig_S1),
        ("Figure_S2", FR1.Fig_S2),
        ("Figure_S3", FR1.Fig_S3),
        ("Figure_S4", FR1.Fig_S4),
        ("Figure_S5", FR1.Fig_S5),
        ("Figure_S6", FR1.Fig_S6),
        ("Figure_S7", FR1.Fig_S7),
        ("Figure_S8", FR1.Fig_S8),
        ("Figure_S9", FR1.Fig_S9),
        ("Figure_S10", FR1.Fig_S10),
        ("Figure_S11", FR1.Fig_S11),
        ("Figure_S12", FR1.Fig_S12),
        ("Figure_S13", FR1.Fig_S13),
        ("Figure_S14", FR1.Fig_S14),
        ("Figure_S15", FR1.Fig_S15),
        ("Figure_S16", FR1.Fig_S16),
        ("Figure_S17", FR1.Fig_S17),
        ("Figure_S18", FR1.Fig_S18),
    ]
    for fig_label, fig_func in review1_figs:
        print(f"  Processing {fig_label} ...")
        fig_func(FigOutDir=FigOutDir, FigLabel=fig_label)
        print(f"   √ Completed Merging Review 1 {fig_label}.\n")
    print(f"   √ Completed Merging Review 1 Figures.\n\n")




########################################################
if Flag_Calculate_Indexes:
    ''' 计算各类指数 '''
    print("\n\nStart Calculating Various Indexes...")
    FCI.CWRFSkill_Index(OutDir=DataOutDir, lkinfos=lkinfos)
    FCI.LakeRegClimImpact(OutDir=DataOutDir, lkinfos=lkinfos)
    print(f"   √ Completed Calculating Various Indexes.\n\n")



