import cmaps
from typing import Tuple
from matplotlib import colors as mcolors
import pandas as pd
import numpy as np


def Variable_Infos(varname: str) -> dict:
    """
    获取变量的标准化绘图信息。
    遵循 SCI 期刊格式：
    1. 单位使用正体 (\mathrm)，且推荐使用负指数形式 (m s^-1)。
    2. 变量名使用斜体，下标中的描述性文字使用正体。
    """
    vars_dict = {
        # --- 温度相关 ---
        'T2m': {
            'longname': '2-m air temperature',
            'unit': r'°C',  # 摄氏度标准写法
            'bunit': r'$\mathbf{^\circ C}$',
            'abbr': r'$\mathrm{T2m}$'      # T斜体，2m正体
        },
        'T2m-Max': {
            'longname': 'Max 2-m air temperature',
            'unit': r'°C',
            'bunit': r'$\mathbf{^\circ C}$',
            'abbr': r'$\mathrm{T2m}_{\mathrm{max}}$'  # max为描述词，需正体
        },
        'T2m-Min': {
            'longname': 'Min 2-m air temperature',
            'unit': r'°C',
            'bunit': r'$\mathbf{^\circ C}$',
            'abbr': r'$\mathrm{T2m}_{\mathrm{min}}$'
        },
        'TSK': {
            'longname': 'Skin temperature',
            'unit': r'°C',
            'bunit': r'$\mathbf{^\circ C}$',
            'abbr': r'$\mathrm{T2m}_{\mathrm{skin}}$'
        },
        'WA': {
            'longname': 'Warm advection',
            'unit': r'°C\,\mathrm{day}^{-1}', # 负指数形式
            'bunit': r'$\mathbf{^\circ C\,day^{-1}}$',
            'abbr': r'$\mathrm{T2m}_{\mathrm{adv}}$'
        },
        'T': {
            'longname': 'Air temperature',
            'unit': r'°C',
            'bunit': r'$\mathbf{^\circ C}$',
            'abbr': r'$\mathrm{T}$'
        },
        'Theta': {
            'longname': 'Potential temperature',
            'unit': r'$\mathrm{K}$',
            'bunit': r'$\mathbf{K}$',
            'abbr': r'$\theta$' # 使用希腊字母
        },

        # --- 降水与湿度 ---
        'Prec': {
            'longname': 'Precipitation',
            'unit': r'$\mathrm{mm\,day^{-1}}$', # 加上 \, 增加间隙
            'bunit': r'$\mathbf{mm\,day^{-1}}$',
            'abbr': r'$\mathrm{Prec}$'
        },
        'Prec-Max': {
            'longname': 'Max precipitation',
            'unit': r'$\mathrm{mm\,day^{-1}}$',
            'bunit': r'$\mathbf{mm\,day^{-1}}$',
            'abbr': r'$\mathrm{Prec}_{\mathrm{max}}$'
        },
        'Q2m': {
            'longname': 'Specific humidity',
            'unit': r'$\mathrm{g\,kg^{-1}}$',
            'bunit': r'$\mathbf{g\,kg^{-1}}$',
            'abbr': r'$\mathrm{q2m}$' # 比湿通常用小写 q
        },
        'RH': {
            'longname': 'Relative humidity',
            'unit': r'$\%$', # % 在 LaTeX 中需要转义
            'bunit': r'$\mathbf{\%}$',
            'abbr': r'$\mathrm{RH}$' # RH 是缩写，应为正体
        },
        'CloudFra': {
            'longname': 'Cloud fraction',
            'unit': r'$\%$',
            'bunit': r'$\mathbf{\%}$',
            'abbr': r'$\mathrm{CF}$'
        },

        # --- 通量 ---
        'LHF': {
            'longname': 'Latent heat flux',
            'unit': r'$\mathrm{W\,m^{-2}}$',
            'bunit': r'$\mathbf{W\,m^{-2}}$',
            'abbr': r'$\mathrm{LHF}$' 
        },
        'SHF': {
            'longname': 'Sensible heat flux',
            'unit': r'$\mathrm{W\,m^{-2}}$',
            'bunit': r'$\mathbf{W\,m^{-2}}$',
            'abbr': r'$\mathrm{SHF}$'
        },

        # --- 风与动力 ---
        'U10': {
            'longname': '10-m zonal wind',
            'unit': r'$\mathrm{m\,s^{-1}}$',
            'bunit': r'$\mathbf{m\,s^{-1}}$',
            'abbr': r'$\mathrm{U}_{\mathrm{10}}$'
        },
        'V10': {
            'longname': '10-m meridional wind',
            'unit': r'$\mathrm{m\,s^{-1}}$',
            'bunit': r'$\mathbf{m\,s^{-1}}$',
            'abbr': r'$\mathrm{V}_{\mathrm{10}}$'
        },
        'UV10': {
            'longname': '10-m wind speed',
            'unit': r'$\mathrm{m\,s^{-1}}$',
            'bunit': r'$\mathbf{m\,s^{-1}}$',
            'abbr': r'$\mathrm{Wind}_{\mathrm{10}}$' # 或者 |\mathbf{V}_{10}|
        },
        'U': {
            'longname': 'Zonal wind speed',
            'unit': r'$\mathrm{m\,s^{-1}}$',
            'bunit': r'$\mathbf{m\,s^{-1}}$',
            'abbr': r'$\mathrm{U}$'
        },
        'V': {
            'longname': 'Meridional wind speed',
            'unit': r'$\mathrm{m\,s^{-1}}$',
            'bunit': r'$\mathbf{m\,s^{-1}}$',
            'abbr': r'$\mathrm{V}$'
        },
        'UV': {
            'longname': 'Wind speed',
            'unit': r'$\mathrm{m\,s^{-1}}$',
            'bunit': r'$\mathbf{m\,s^{-1}}$',
            'abbr': r'$\mathrm{Wind}$'
        },
        'W': {
            'longname': 'Vertical velocity',
            # 注意：如果原本是 Pa/s，对应的物理符号通常是 omega (\omega)
            # 如果是 m/s，符号是 w。此处保留你原来的 Pa/s 但修正写法
            'unit': r'$\mathrm{m\,s^{-1}}$', 
            'bunit': r'$\mathbf{m\,s^{-1}}$',
            'abbr': r'$\omega$' 
        },

        # --- 高度与稳定度 ---
        'PBLH': {
            'longname': 'Planetary boundary layer height',
            'unit': r'$\mathrm{m}$',
            'bunit': r'$\mathbf{m}$',
            'abbr': r'$\mathrm{PBLH}$'
        },
        'Height': {
            'longname': 'Geopotential height',
            'unit': r'$\mathrm{m}$',
            'bunit': r'$\mathbf{m}$',
            'abbr': r'$\mathrm{Z}$' # 位势高度通常用 Z 表示
        },
        'dTheta': {
            'longname': 'Vertical gradient of potential temp.',
            'unit': r'$\mathrm{K\,km^{-1}}$',
            'bunit': r'$\mathbf{K\,km^{-1}}$',
            'abbr': r'$\partial\theta/\partial z$' # 使用偏微分符号
        },
        'StaticStability': {
            'longname': 'Static stability',
            'unit': r'$\mathrm{K\,hPa^{-1}}$',
            'bunit': r'$\mathbf{K\,hPa^{-1}}$',
            'abbr': r'$\partial\theta/\partial p$'
        },
    }

    if varname in vars_dict.keys():
        return vars_dict[varname]
    else:
        # 建议打印可用键值，方便调试
        valid_keys = ", ".join(vars_dict.keys())
        raise ValueError(f"Variable '{varname}' is not defined. Available options: {valid_keys}")



def ExtremeEvent_Infos(eventname: str) -> dict:
    events_dict = {
        'Hot': {
            'longname': 'extreme hot',
            'unit': r'$\mathrm{days\,year^{-1}}$',
            'bunit': r'$\mathbf{days\,year^{-1}}$',
            'abbr': 'Hot'
        },
        'Cold': {
            'longname': 'extreme cold',
            'unit': r'$\mathrm{days\,year^{-1}}$',
            'bunit': r'$\mathbf{days\,year^{-1}}$',
            'abbr': 'Cold'
        },
        'Wet': {
            'longname': 'extreme wet',
            'unit': r'$\mathrm{days\,year^{-1}}$',
            'bunit': r'$\mathbf{days\,year^{-1}}$',
            'abbr': 'Wet'
        },
        'Dry': {
            'longname': 'extreme dry',
            'unit': r'$\mathrm{days\,year^{-1}}$',
            'bunit': r'$\mathbf{days\,year^{-1}}$',
            'abbr': 'Dry'
        },
        'ColdWet': {
            'longname': 'compound cold-wet',
            'unit': r'$\mathrm{days\,year^{-1}}$',
            'bunit': r'$\mathbf{days\,year^{-1}}$',
            'abbr': 'Cold-Wet'
        },
        'ColdDry': {
            'longname': 'compound cold-dry',
            'unit': r'$\mathrm{days\,year^{-1}}$',
            'bunit': r'$\mathbf{days\,year^{-1}}$',
            'abbr': 'Cold-Dry'
        },
        'HotWet': {
            'longname': 'compound hot-wet',
            'unit': r'$\mathrm{days\,year^{-1}}$',
            'bunit': r'$\mathbf{days\,year^{-1}}$',
            'abbr': 'Hot-Wet'
        },
        'HotDry': {
            'longname': 'compound hot-dry',
            'unit': r'$\mathrm{days\,year^{-1}}$',
            'bunit': r'$\mathbf{days\,year^{-1}}$',
            'abbr': 'Hot-Dry'
        },
        'exT': {
            'longname': 'extreme temperature intensity',
            'unit': '°C',
            'bunit': r'$\mathbf{^\circ C}$',
            'abbr': 'exT'
        },
        'exP': {
            'longname': 'extreme precipitation intensity',
            'unit': r'$\mathrm{mm\,day^{-1}}$',
            'bunit': r'$\mathbf{mm\,day^{-1}}$',
            'abbr': 'exP'
        },
        'Freq': {
            'longname': 'extreme event frequency',
            'unit': r'$\mathrm{days\,year^{-1}}$',
            'bunit': r'$\mathbf{days\,year^{-1}}$',
            'abbr': 'Freq'
        },
        'exT_Hot': {
            'longname': 'extreme heat intensity',
            'unit': '°C',
            'bunit': r'$\mathbf{^\circ C}$',
            'abbr': 'exT-Hot'
        },
        'exT_Cold': {
            'longname': 'extreme cold intensity',
            'unit': '°C',
            'bunit': r'$\mathbf{^\circ C}$',
            'abbr': 'exT-Cold'
        },
        'exRH_Wet': {
            'longname': 'extreme wet intensity',
            'unit': '%',
            'bunit': r'$\mathbf{\%}$',
            'abbr': 'exRH-Wet'
        },
        'exRH_Dry': {
            'longname': 'extreme drought intensity',
            'unit': '%',
            'bunit': r'$\mathbf{\%}$',
            'abbr': 'exRH-Dry'
        },
        'exTw_HotWet': {
            'longname': 'extreme hot-wet intensity',
            'unit': '°C',
            'bunit': r'$\mathbf{^\circ C}$',
            'abbr': 'exTw-HotWet'
        },
        'exTw_HotDry': {
            'longname': 'extreme hot-dry intensity',
            'unit': '°C',
            'bunit': r'$\mathbf{^\circ C}$',
            'abbr': 'exTw-HotDry'
        },
        'exTw_ColdWet': {
            'longname': 'extreme cold-wet intensity',
            'unit': '°C',
            'bunit': r'$\mathbf{^\circ C}$',
            'abbr': 'exTw-ColdWet'
        },
        'exTw_ColdDry': {
            'longname': 'extreme cold-dry intensity',
            'unit': '°C',
            'bunit': r'$\mathbf{^\circ C}$',
            'abbr': 'exTw-ColdDry'
        },
    }
    if eventname in events_dict:
        return events_dict[eventname]
    else:
        raise ValueError(f"Extreme event {eventname} is not defined.")



def Coupling_Infos(name: str) -> dict:
    level_dicts = {
        'pearson_r': {
            'longname': 'Pearson correlation coefficient',
            'unit': '',
            'bunit': '',
            'abbr': 'R_pearson'
        },
        'spearman_r': {
            'longname': 'Spearman correlation coefficient',
            'unit': '',
            'bunit': '',
            'abbr': 'R_spearman'
        },
        'mutual_info': {
            'longname': 'Mutual information',
            'unit': '',
            'bunit': '',
            'abbr': 'MI'
        },
        'r2': {
            'longname': 'Coefficient of determination',
            'unit': '',
            'bunit': '',
            'abbr': 'R2'
        },
        'kendall_tau': {
            'longname': 'Kendall’s τ coefficient',
            'unit': '',
            'bunit': '',
            'abbr': 'Kendall’s τ'
        },
        'lambda_u': {
            'longname': 'Upper tail dependence coefficient',
            'unit': '',
            'bunit': '',
            'abbr': '$\lambda_U$'
        },
        'cov': {
            'longname': 'Covariance',
            'unit': '',
            'bunit': '',
            'abbr': 'Cov'
        },
        'utdc': {
            'longname': 'Upper tail dependence coefficient',
            'unit': '',
            'bunit': '',
            'abbr': 'UTDC'
        },
    }
    if name in level_dicts:
        return level_dicts[name]
    else:
        raise ValueError(f"Coupling metric {name} is not defined.")



#--------------- Plot Utils -------------------
def Seasonal_Mean_Cmap(varname: str) -> dict:
    level_dicts = {
        'T2m': {
            'levels': {
                'DJF': [np.linspace(0, 15, 11),5],
                'MAM': [np.linspace(5, 25, 11),5],
                'JJA': [np.linspace(15, 35, 11),5],
                'SON': [np.linspace(5, 25, 11),5],
            },
            'unit': '°C',
            'cmap': cmaps.NCV_jaisnd
        },
        'TSK': {
            'levels': {
                'DJF': [np.linspace(0, 15, 11),5],
                'MAM': [np.linspace(5, 25, 11),5],
                'JJA': [np.linspace(15, 35, 11),5],
                'SON': [np.linspace(5, 25, 11),5],
            },
            'unit': '°C',
            'cmap': cmaps.NCV_jaisnd
        },
        'Prec': {
            'levels': {
                'DJF': [np.linspace(0, 13, 14),14],
                'MAM': [np.linspace(0, 13, 14),14],
                'JJA': [np.linspace(0, 13, 14),14],
                'SON': [np.linspace(0, 13, 14),14],
                # 'DJF': [[0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,], 16],
                # 'MAM': [[0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,], 16],
                # 'JJA': [[0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,], 16],
                # 'SON': [[0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,], 16],
            },
            'unit': 'mm/day',
            'cmap': cmaps.WhiteBlueGreenYellowRed
            # 'cmap': mcolors.ListedColormap(['#F2F2F2', '#B2DFEE', '#9AC0CD', '#44B0D5', '#00A3F7', 
            #                                 '#99FF33', '#00C700', '#008F00', '#003F00', '#FFCC00', 
            #                                 '#FF8F00', '#FF0000', '#D70000', '#FF00FF', '#800080']),
        },
        'PBLH': {
            'levels': {
                'DJF': [np.linspace(0, 1000, 11),5],
                'MAM': [np.linspace(0, 1000, 11),5],
                'JJA': [np.linspace(0, 1000, 11),5],
                'SON': [np.linspace(0, 1000, 11),5],
            },
            'unit': 'm',
            'cmap': cmaps.cmocean_matter
        },
        'UV10': {
            'levels': {
                'DJF': [np.linspace(0, 5, 11), 6],
                'MAM': [np.linspace(0, 5, 11), 6],
                'JJA': [np.linspace(0, 5, 11), 6],
                'SON': [np.linspace(0, 5, 11), 6],
            },
            'unit': 'm/s',
            'cmap': cmaps.cmocean_tempo
        },
        'RH': {
            'levels': {
                'DJF': [np.linspace(40, 100, 13),5],
                'MAM': [np.linspace(40, 100, 13),5],
                'JJA': [np.linspace(40, 100, 13),5],
                'SON': [np.linspace(40, 100, 13),5],
            },
            'unit': '%',
            'cmap': cmaps.WhiteBlueGreenYellowRed
        },
        'Q2m': {
            'levels': {
                'DJF': [np.linspace(0, 10, 11),5],
                'MAM': [np.linspace(5, 20, 11),5],
                'JJA': [np.linspace(10, 25, 11),5],
                'SON': [np.linspace(5, 20, 11),5],
            },
            'unit': 'g/kg',
            'cmap': cmaps.WhiteBlueGreenYellowRed
        },
        'UV': {
            'levels': {
                'DJF': [np.linspace(0, 5, 6), 6],
                'MAM': [np.linspace(0, 5, 6), 6],
                'JJA': [np.linspace(0, 5, 6), 6],
                'SON': [np.linspace(0, 5, 6), 6],
            },
            'unit': 'm/s',
            'cmap': cmaps.cmocean_tempo
        }
    }
    if varname in level_dicts.keys():
        return level_dicts[varname]
    else:
        raise ValueError(f"Variable {varname} is not defined.")



def Seasonal_Diff_Cmap(varname: str) -> dict:
    level_dicts = {
        'T2m': {
            'levels': {
                # 'DJF': [[-7, -5, -3, -1,  1, 3, 5, 7], 5],
                # 'MAM': [[-7, -5, -3, -1,  1, 3, 5, 7], 5],
                # 'JJA': [[-7, -5, -3, -1,  1, 3, 5, 7], 5],
                # 'SON': [[-7, -5, -3, -1,  1, 3, 5, 7], 5],
                'DJF': [np.linspace(-7, 7, 201), 5],
                'MAM': [np.linspace(-7, 7, 201), 5],
                'JJA': [np.linspace(-7, 7, 201), 5],
                'SON': [np.linspace(-7, 7, 201), 5],
            },
            'unit': '°C',
            'cmap': cmaps.BlueWhiteOrangeRed
        },
        'TSK': {
            'levels': {
                'DJF': [np.linspace(-7, 7, 11),5],
                'MAM': [np.linspace(-7, 7, 11),5],
                'JJA': [np.linspace(-7, 7, 11),5],
                'SON': [np.linspace(-7, 7, 11),5],
            },
            'unit': '°C',
            'cmap': cmaps.BlueWhiteOrangeRed
        },
        'Prec': {
            'levels': {
                # 'DJF': [[-7, -5, -3, -1,  1, 3, 5, 7], 5],
                # 'MAM': [[-7, -5, -3, -1,  1, 3, 5, 7], 5],
                # 'JJA': [[-7, -5, -3, -1,  1, 3, 5, 7], 5],
                # 'SON': [[-7, -5, -3, -1,  1, 3, 5, 7], 5],
                'DJF': [np.linspace(-7, 7, 201), 5],
                'MAM': [np.linspace(-7, 7, 201), 5],
                'JJA': [np.linspace(-7, 7, 201), 5],
                'SON': [np.linspace(-7, 7, 201), 5],
            },
            'unit': 'mm/day',
            'cmap': cmaps.BlueWhiteOrangeRed
            # 'cmap': mcolors.ListedColormap(['#1C326B','#2A62AD', '#488ECA', '#9CF8FF', '#FFFFFF', '#FDEEAC', '#FF7953', '#EE5A29', '#97161A'])
        },
        'PBLH': {
            'levels': {
                'DJF': [np.linspace(-300, 300, 13),5],
                'MAM': [np.linspace(-300, 300, 13),5],
                'JJA': [np.linspace(-300, 300, 13),5],
                'SON': [np.linspace(-300, 300, 13),5],
            },
            'unit': 'm',
            'cmap': cmaps.BlueWhiteOrangeRed
        },
        'UV10': {
            'levels': {
                'DJF': [np.linspace(-3, 3, 13),5],
                'MAM': [np.linspace(-3, 3, 13),5],
                'JJA': [np.linspace(-3, 3, 13),5],
                'SON': [np.linspace(-3, 3, 13),5],
            },
            'unit': 'm/s',
            'cmap': cmaps.BlueWhiteOrangeRed
        },
        'RH': {
            'levels': {
                'DJF': [np.linspace(-15, 15, 13),7],
                'MAM': [np.linspace(-15, 15, 13),7],
                'JJA': [np.linspace(-15, 15, 13),7],
                'SON': [np.linspace(-15, 15, 13),7],
            },
            'unit': '%',
            'cmap': cmaps.BlueWhiteOrangeRed
        },
        'Q2m': {
            'levels': {
                # 'DJF': [[-5, -4, -3, -2, -1,  1, 2, 3, 4, 5], 5],
                # 'MAM': [[-5, -4, -3, -2, -1,  1, 2, 3, 4, 5], 5],
                # 'JJA': [[-5, -4, -3, -2, -1,  1, 2, 3, 4, 5], 5],
                # 'SON': [[-5, -4, -3, -2, -1,  1, 2, 3, 4, 5], 5],
                'DJF': [np.linspace(-5, 5, 201), 5],
                'MAM': [np.linspace(-5, 5, 201), 5],
                'JJA': [np.linspace(-5, 5, 201), 5],
                'SON': [np.linspace(-5, 5, 201), 5],
            },
            'unit': 'g/kg',
            'cmap': cmaps.BlueWhiteOrangeRed
        },
    }
    if varname in level_dicts.keys():
        return level_dicts[varname]
    else:
        raise ValueError(f"Variable {varname} is not defined.")




def Daily_Cmap(varname: str) -> dict:
    level_dicts = {
        'T2m': {
            'levels': {
                'rmse': [np.linspace(0, 7, 8),5],
                'bias': [np.linspace(-3, 3, 7),5],
                'pcc': [np.linspace(0.6, 1, 7),5]
            },
            'unit': '°C',
            'cmap': {
                'rmse': cmaps.WhiteYellowOrangeRed,
                'bias': cmaps.BlueWhiteOrangeRed,
                'pcc': cmaps.WhiteBlue
            },
        },

        'TSK': {
            'levels': {
                'rmse': [np.linspace(0, 7, 8),5],
                'bias': [np.linspace(-3, 3, 7),5],
                'pcc': [np.linspace(0.6, 1, 7),5]
            },
            'unit': '°C',
            'cmap': {
                'rmse': cmaps.WhiteYellowOrangeRed,
                'bias': cmaps.BlueWhiteOrangeRed,
                'pcc': cmaps.WhiteBlue
            },
        },

        'Prec': {
            'levels': {
                'rmse': [np.linspace(0, 20, 11),5],
                'bias': [np.linspace(-5, 5, 11),5],
                'pcc': [np.linspace(0, 0.6, 11),5]
            },
            'unit': 'mm/day',
            'cmap': {
                'rmse': cmaps.WhiteYellowOrangeRed,
                'bias': cmaps.BlueWhiteOrangeRed,
                'pcc': cmaps.WhiteBlue
            },
        }
    }
    if varname in level_dicts.keys():
        return level_dicts[varname]
    else:
        raise ValueError(f"Variable {varname} is not defined.")

######################################################################


def Seasonal_RegClimImpact_Cmap(varname: str) -> dict:
    level_dicts = {
        'T2m': {
            'diff_maplevs_seasonal': [np.linspace(-0.3, 0.3, 11), 7],
            'diff_maplevs_annual': [np.linspace(-0.3, 0.3, 11), 7],
            'diff_boxlevs': [np.linspace(-0.3, 0.3, 11), 5],
            'rc_maplevs_seasonal': [np.linspace(-5, 5, 11), 5],
            'rc_maplevs_annual': [np.linspace(-5, 5, 11), 5],
            'rc_boxlevs': [np.linspace(-2, 2, 11), 5],
            'diff_cmap': cmaps.BlueWhiteOrangeRed,
            # 'diff_cmap': cmaps.MPL_PuOr.reversed(),
            'rc_cmap': cmaps.MPL_RdBu.reversed(),
        },
        'T2m-Max': {
            'diff_maplevs_seasonal': [np.linspace(-0.8, 0.8, 11), 5],
            'diff_maplevs_annual': [np.linspace(-0.8, 0.8, 11), 5],
            'diff_boxlevs': [np.linspace(-0.5, 0.5, 11), 5],
            'rc_maplevs_seasonal': [np.linspace(-5, 5, 11), 5],
            'rc_maplevs_annual': [np.linspace(-5, 5, 11), 5],
            'rc_boxlevs': [np.linspace(-2, 2, 11), 5],
            'diff_cmap': cmaps.BlueWhiteOrangeRed,
            # 'diff_cmap': cmaps.MPL_PuOr.reversed(),
            'rc_cmap': cmaps.MPL_RdBu.reversed(),
        },
        'T2m-Min': {
            'diff_maplevs_seasonal': [np.linspace(-0.8, 0.8, 11), 5],
            'diff_maplevs_annual': [np.linspace(-0.8, 0.8, 11), 5],
            'diff_boxlevs': [np.linspace(-0.5, 0.5, 11), 5],
            'rc_maplevs_seasonal': [np.linspace(-5, 5, 11), 5],
            'rc_maplevs_annual': [np.linspace(-5, 5, 11), 5],
            'rc_boxlevs': [np.linspace(-15, 15, 11), 5],
            'diff_cmap': cmaps.BlueWhiteOrangeRed,
            # 'diff_cmap': cmaps.MPL_PuOr.reversed(),
            'rc_cmap': cmaps.MPL_RdBu.reversed(),
        },
        'Prec': {
            'diff_maplevs_seasonal': [np.linspace(-0.6, 0.6, 11), 7],
            'diff_maplevs_annual': [np.linspace(-0.6, 0.6, 11), 7],
            'diff_boxlevs': [np.linspace(-2, 2, 11), 7],
            'rc_maplevs_seasonal': [np.linspace(-5, 5, 11), 5],
            'rc_maplevs_annual': [np.linspace(-5, 5, 11), 5],
            'rc_boxlevs': [np.linspace(-50, 50, 11), 5],
            'diff_cmap': cmaps.BlueWhiteOrangeRed,
            # 'diff_cmap': cmaps.MPL_PuOr.reversed(),
            'rc_cmap': cmaps.MPL_RdBu.reversed(),
        },
        'Prec-Max': {
            'diff_maplevs_seasonal': [np.linspace(-10, 10, 11), 5],
            'diff_maplevs_annual': [np.linspace(-10, 10, 11), 5],
            'rc_maplevs_seasonal': [np.linspace(-5, 5, 11), 5],
            'rc_maplevs_annual': [np.linspace(-5, 5, 11), 5],
            'diff_boxlevs': [np.linspace(-10, 10, 11), 5],
            'rc_boxlevs': [np.linspace(-5, 5, 11), 5],
            'diff_cmap': cmaps.BlueWhiteOrangeRed,
            # 'diff_cmap': cmaps.MPL_PuOr.reversed(),
            'rc_cmap': cmaps.MPL_RdBu.reversed(),
        },
        'LHF': {
            'diff_maplevs_seasonal': [np.linspace(-40, 40, 11), 5],
            'diff_maplevs_annual': [np.linspace(-50, 50, 11), 7],
            'diff_boxlevs': [np.linspace(-10, 10, 11), 7],
            'rc_maplevs_seasonal': [np.linspace(-40, 40, 11), 5],
            'rc_maplevs_annual': [np.linspace(-25, 25, 11), 5],
            'rc_boxlevs': [np.linspace(-12, 12, 11), 5],
            'diff_cmap': cmaps.BlueWhiteOrangeRed,
            # 'diff_cmap': cmaps.MPL_PuOr.reversed(),
            'rc_cmap': cmaps.MPL_RdBu.reversed(),
        },
        'SHF': {
            'diff_maplevs_seasonal': [np.linspace(-30, 30, 11),5],
            'diff_maplevs_annual': [np.linspace(-30, 30, 11),5],
            'diff_boxlevs': [np.linspace(-10, 10, 11),5],
            'rc_maplevs_seasonal': [np.linspace(-25, 25, 11), 5],
            'rc_maplevs_annual': [np.linspace(-25, 25, 11), 5],
            'rc_boxlevs': [np.linspace(-12, 12, 11), 5],
            'diff_cmap': cmaps.BlueWhiteOrangeRed,
            # 'diff_cmap': cmaps.MPL_PuOr.reversed(),
            'rc_cmap': cmaps.MPL_RdBu.reversed(),
        },
        'RH': {
            'diff_maplevs_seasonal': [np.linspace(-2, 2, 11), 5],
            'diff_maplevs_annual': [np.linspace(-2, 2, 11), 5],
            'diff_boxlevs': [np.linspace(-2, 2, 11), 5],
            'rc_maplevs_seasonal': [np.linspace(-5, 5, 11), 5],
            'rc_maplevs_annual': [np.linspace(-5, 5, 11), 5],
            'rc_boxlevs': [np.linspace(-3, 3, 11), 5],
            'diff_cmap': cmaps.BlueWhiteOrangeRed,
            # 'diff_cmap': cmaps.MPL_PuOr.reversed(),
            'rc_cmap': cmaps.MPL_RdBu.reversed(),
        },
        'Q2m': {
            'diff_maplevs_seasonal': [np.linspace(-0.4, 0.4, 11), 5],
            'diff_maplevs_annual': [np.linspace(-0.4, 0.4, 11), 5],
            'rc_maplevs_seasonal': [np.linspace(-5, 5, 11), 5],
            'rc_maplevs_annual': [np.linspace(-5, 5, 11), 5],
            'diff_boxlevs': [np.linspace(-0.4, 0.4, 11), 5],
            'rc_boxlevs': [np.linspace(-5, 5, 11), 5],
            'diff_cmap': cmaps.BlueWhiteOrangeRed,
            # 'diff_cmap': cmaps.MPL_PuOr.reversed(),
            'rc_cmap': cmaps.MPL_RdBu.reversed(),
        },
        'U10': {
            'diff_maplevs_seasonal': [np.linspace(-0.4, 0.4, 11), 5],
            'diff_maplevs_annual': [np.linspace(-0.4, 0.4, 11), 5],
            'rc_maplevs_seasonal': [np.linspace(-60, 60, 11), 5],
            'rc_maplevs_annual': [np.linspace(-60, 60, 11), 5],
            'diff_boxlevs': [np.linspace(-2, 2, 11), 5],
            'rc_boxlevs': [np.linspace(-60, 60, 11), 5],
            'diff_cmap': cmaps.BlueWhiteOrangeRed,
            # 'diff_cmap': cmaps.MPL_PuOr.reversed(),
            'rc_cmap': cmaps.MPL_RdBu.reversed(),
        },
        'V10': {
            'diff_maplevs_seasonal': [np.linspace(-0.4, 0.4, 11), 5],
            'diff_maplevs_annual': [np.linspace(-0.4, 0.4, 11), 5],
            'rc_maplevs_seasonal': [np.linspace(-60, 60, 11), 5],
            'rc_maplevs_annual': [np.linspace(-60, 60, 11), 5],
            'diff_boxlevs': [np.linspace(-2, 2, 11), 5],
            'rc_boxlevs': [np.linspace(-60, 60, 11), 5],
            'diff_cmap': cmaps.BlueWhiteOrangeRed,
            # 'diff_cmap': cmaps.MPL_PuOr.reversed(),
            'rc_cmap': cmaps.MPL_RdBu.reversed(),
        },
        'UV10': {
            'diff_maplevs_seasonal': [np.linspace(-1, 1, 11), 5],
            'diff_maplevs_annual': [np.linspace(-1, 1, 11), 5],
            'rc_maplevs_seasonal': [np.linspace(-15, 15, 11), 5],
            'rc_maplevs_annual': [np.linspace(-15, 15, 11), 5],
            'diff_boxlevs': [np.linspace(-2, 2, 11), 5],
            'rc_boxlevs': [np.linspace(-15, 15, 11), 5],
            'diff_cmap': cmaps.BlueWhiteOrangeRed,
            # 'diff_cmap': cmaps.MPL_PuOr.reversed(),
            'rc_cmap': cmaps.cmocean_matter,
        },
        'PBLH': {
            'diff_maplevs_seasonal': [np.linspace(-200, 200, 11), 5],
            'diff_maplevs_annual': [np.linspace(-200, 200, 11), 5],
            'rc_maplevs_seasonal': [np.linspace(-40, 40, 11), 5],
            'rc_maplevs_annual': [np.linspace(-40, 40, 11), 5],
            'diff_boxlevs': [np.linspace(-200, 200, 11), 5],
            'rc_boxlevs': [np.linspace(-40, 40, 11), 5],
            'diff_cmap': cmaps.BlueWhiteOrangeRed,
            # 'diff_cmap': cmaps.MPL_PuOr.reversed(),
            'rc_cmap': cmaps.MPL_RdBu.reversed(),
        },
        
    }
    if varname in level_dicts.keys():
        return level_dicts[varname]
    else:
        raise ValueError(f"Variable {varname} is not defined.")



def Diurnal_RegClimImpact_Cmap(varname: str) -> dict:
    level_dicts = {
        'T2m': {
            'roselevels': [np.linspace(-1.5, 1.5, 11), 5],
            'maplevels': [np.linspace(-0.4, 0.4, 11), 5],
            'kdelevels': [np.linspace(-1.25, 1.25, 11), 5],
            'rclevels': [np.linspace(-5, 5, 11), 5],
            'diff_cmap': cmaps.BlueWhiteOrangeRed,
            # 'diff_cmap': cmaps.MPL_PuOr.reversed(),
            'rcmap': cmaps.MPL_RdBu.reversed(),
        },
        'T2m-Max': {
            'roselevels': [np.linspace(-2.0, 2.0, 11), 5],
            'maplevels': [np.linspace(-1.5, 1.5, 11), 5],
            'kdelevels': [np.linspace(-2.0, 2.0, 11), 5],
            'rclevels': [np.linspace(-5, 5, 11), 5],
            'diff_cmap': cmaps.BlueWhiteOrangeRed,
            # 'diff_cmap': cmaps.MPL_PuOr.reversed(),
            'rcmap': cmaps.MPL_RdBu.reversed(),
        },
        'T2m-Min': {
            'roselevels': [np.linspace(-3.5, 3.5, 11), 5],
            'maplevels': [np.linspace(-1.5, 1.5, 11), 5],
            'kdelevels': [np.linspace(-3.5, 3.5, 11), 5],
            'rclevels': [np.linspace(-5, 5, 11), 5],
            'diff_cmap': cmaps.BlueWhiteOrangeRed,
            # 'diff_cmap': cmaps.MPL_PuOr.reversed(),
            'rcmap': cmaps.MPL_RdBu.reversed(),
        },
        'LHF': {
            'roselevels': [np.linspace(-50, 50, 11), 5],
            'maplevels': [np.linspace(-40, 40, 11), 5],
            'kdelevels': [np.linspace(-80, 80, 11), 5],
            'rclevels': [np.linspace(-5, 5, 11), 5],
            'diff_cmap': cmaps.BlueWhiteOrangeRed,
            # 'diff_cmap': cmaps.MPL_PuOr.reversed(),
            'rcmap': cmaps.MPL_RdBu.reversed(),
        },
        'SHF': {
            'roselevels': [np.linspace(-50, 50, 11), 5],
            'maplevels': [np.linspace(-20, 20, 11), 5],
            'kdelevels': [np.linspace(-20, 20, 11), 5],
            'rclevels': [np.linspace(-5, 5, 11), 5],
            'diff_cmap': cmaps.BlueWhiteOrangeRed,
            # 'diff_cmap': cmaps.MPL_PuOr.reversed(),
            'rcmap': cmaps.MPL_RdBu.reversed(),
        },
        'RH': {
            'roselevels': [np.linspace(-5, 5, 11), 5],
            'maplevels': [np.linspace(-2, 2, 11), 5],
            'kdelevels': [np.linspace(-4, 4, 11), 5],
            'rclevels': [np.linspace(-5, 5, 11), 5],
            'diff_cmap': cmaps.BlueWhiteOrangeRed,
            # 'diff_cmap': cmaps.MPL_PuOr.reversed(),
            'rcmap': cmaps.MPL_RdBu.reversed(),
        },
        'Prec': {
            'roselevels': [np.linspace(-2, 2, 11), 5],
            'maplevels': [np.linspace(-10, 10, 11), 5],
            'kdelevels': [np.linspace(-10, 10, 11), 5],
            'rclevels': [np.linspace(-5, 5, 11), 5],
            'diff_cmap': cmaps.BlueWhiteOrangeRed,
            # 'diff_cmap': cmaps.MPL_PuOr.reversed(),
            'rcmap': cmaps.MPL_RdBu.reversed(),
        },
        'Prec-Max': {
            'roselevels': [np.linspace(-2, 2, 11), 5],
            'maplevels': [np.linspace(-10, 10, 11), 5],
            'kdelevels': [np.linspace(-10, 10, 11), 5],
            'rclevels': [np.linspace(-5, 5, 11), 5],
            'diff_cmap': cmaps.BlueWhiteOrangeRed,
            # 'diff_cmap': cmaps.MPL_PuOr.reversed(),
            'rcmap': cmaps.MPL_RdBu.reversed(),
        },
        'Q2m': {
            'roselevels': [np.linspace(-0.4, 0.4, 11), 5],
            'maplevels': [np.linspace(-0.4, 0.4, 11), 5],
            'kdelevels': [np.linspace(-0.4, 0.4, 11), 5],
            'rclevels': [np.linspace(-5, 5, 11), 5],
            'diff_cmap': cmaps.BlueWhiteOrangeRed,
            # 'diff_cmap': cmaps.MPL_PuOr.reversed(),
            'rcmap': cmaps.MPL_RdBu.reversed(),
        },
        'U10': {
            'roselevels': [np.linspace(-2, 2, 11), 5],
            'maplevels': [np.linspace(-1, 1, 11), 5],
            'kdelevels': [np.linspace(-2, 2, 11), 5],
            'rclevels': [np.linspace(-60, 60, 11), 5],
            'diff_cmap': cmaps.BlueWhiteOrangeRed,
            # 'diff_cmap': cmaps.MPL_PuOr.reversed(),
            'rcmap': cmaps.MPL_RdBu.reversed(),
        },
        'V10': {
            'roselevels': [np.linspace(-2, 2, 11), 5],
            'maplevels': [np.linspace(-1, 1, 11), 5],
            'kdelevels': [np.linspace(-2, 2, 11), 5],
            'rclevels': [np.linspace(-60, 60, 11), 5],
            'diff_cmap': cmaps.BlueWhiteOrangeRed,
            # 'diff_cmap': cmaps.MPL_PuOr.reversed(),
            'rcmap': cmaps.MPL_RdBu.reversed(),
        },
        'PBLH': {
            'roselevels': [np.linspace(-300, 300, 11), 5],
            'maplevels': [np.linspace(-150, 150, 11), 5],
            'kdelevels': [np.linspace(-200, 200, 11), 5],
            'rclevels': [np.linspace(-40, 40, 11), 5],
            'diff_cmap': cmaps.BlueWhiteOrangeRed,
            # 'diff_cmap': cmaps.MPL_PuOr.reversed(),
            'rcmap': cmaps.MPL_RdBu.reversed(),
        },
        'UV10': {
            'roselevels': [np.linspace(-0.4, 0.4, 11), 5],
            'diff_maplevs_seasonal': [np.linspace(-1, 1, 11), 5],
            'diff_maplevs_annual': [np.linspace(-1, 1, 11), 5],
            'rc_maplevs_seasonal': [np.linspace(-10, 10, 11), 5],
            'rc_maplevs_annual': [np.linspace(-15, 15, 11), 5],
            'diff_boxlevs': [np.linspace(-2, 2, 11), 5],
            'rc_boxlevs': [np.linspace(-10, 10, 11), 5],
            'diff_cmap': cmaps.BlueWhiteOrangeRed,
            # 'diff_cmap': cmaps.MPL_PuOr.reversed(),
            'rc_cmap': cmaps.MPL_RdBu.reversed(),
        },
    }
    if varname in level_dicts.keys():
        return level_dicts[varname]
    else:
        raise ValueError(f"Variable {varname} is not defined.")




def Seasonal_ExtremeEvents_Freq_Cmap(varname: str) -> dict:
    level_dicts = {
        'Hot': {
            'diff_maplevs_seasonal': [np.linspace(-6, 6, 11), 5],
            'diff_maplevs_annual': [np.linspace(-10, 10, 11), 5],
            'rc_maplevs_seasonal': [np.linspace(-15, 15, 11), 5],
            'rc_maplevs_annual': [np.linspace(-20, 20, 11), 5],
            'diff_boxlevs': [np.linspace(-5, 5, 11), 5],
            'rc_boxlevs': [np.linspace(-20, 20, 11), 5],
            'diff_cmap': cmaps.BlueWhiteOrangeRed,
            # 'diff_cmap': cmaps.MPL_PuOr.reversed(),
            'rc_cmap': cmaps.MPL_RdBu.reversed(),
        },
        'Cold': {
            'diff_maplevs_seasonal': [np.linspace(-6, 6, 11), 5],
            'diff_maplevs_annual': [np.linspace(-10, 10, 11), 5],
            'rc_maplevs_seasonal': [np.linspace(-20, 20, 11), 5],
            'rc_maplevs_annual': [np.linspace(-20, 20, 11), 5],
            'diff_boxlevs': [np.linspace(-4, 4, 11), 5],
            'rc_boxlevs': [np.linspace(-25, 25, 11), 5],
            'diff_cmap': cmaps.BlueWhiteOrangeRed,
            # 'diff_cmap': cmaps.MPL_PuOr.reversed(),
            'rc_cmap': cmaps.MPL_RdBu.reversed(),
        },
        'Wet': {
            'diff_maplevs_seasonal': [np.linspace(-8, 8, 11), 5],
            'diff_maplevs_annual': [np.linspace(-10, 10, 11), 5],
            'rc_maplevs_seasonal': [np.linspace(-30, 30, 11), 5],
            'rc_maplevs_annual': [np.linspace(-15, 15, 11), 5],
            'diff_boxlevs': [np.linspace(-8, 8, 11), 5],
            'rc_boxlevs': [np.linspace(-40, 40, 11), 5],
            'diff_cmap': cmaps.BlueWhiteOrangeRed,
            # 'diff_cmap': cmaps.MPL_PuOr.reversed(),
            'rc_cmap': cmaps.MPL_RdBu.reversed(),
        },
        'Dry': {
            'diff_maplevs_seasonal': [np.linspace(-6, 6, 11), 5],
            'diff_maplevs_annual': [np.linspace(-12, 12, 11), 5],
            'rc_maplevs_seasonal': [np.linspace(-15, 15, 11), 5],
            'rc_maplevs_annual': [np.linspace(-30, 30, 11), 5],
            'diff_boxlevs': [np.linspace(-20, 20, 11), 5],
            'rc_boxlevs': [np.linspace(-30, 30, 11), 5],
            'diff_cmap': cmaps.BlueWhiteOrangeRed,
            # 'diff_cmap': cmaps.MPL_PuOr.reversed(),
            'rc_cmap': cmaps.MPL_RdBu.reversed(),
        },
        'HotWet': {
            'diff_maplevs_seasonal': [np.linspace(-8, 8, 11), 5],
            'diff_maplevs_annual': [np.linspace(-2, 2, 11), 5],
            'rc_maplevs_seasonal': [np.linspace(-100, 100, 11), 5],
            'rc_maplevs_annual': [np.linspace(-100, 100, 11), 5],
            'diff_boxlevs': [np.linspace(-2, 2, 11), 5],
            'rc_boxlevs': [np.linspace(-100, 100, 11), 5],
            'diff_cmap': cmaps.BlueWhiteOrangeRed,
            # 'diff_cmap': cmaps.MPL_PuOr.reversed(),
            'rc_cmap': cmaps.MPL_RdBu.reversed(),
        },
        'HotDry': {
            'diff_maplevs_seasonal': [np.linspace(-4, 4, 11), 5],
            'diff_maplevs_annual': [np.linspace(-8, 8, 11), 5],
            'rc_maplevs_seasonal': [np.linspace(-80, 80, 11), 5],
            'rc_maplevs_annual': [np.linspace(-80, 80, 11), 5],
            'diff_boxlevs': [np.linspace(-4, 4, 11), 5],
            'rc_boxlevs': [np.linspace(-80, 80, 11), 5],
            'diff_cmap': cmaps.BlueWhiteOrangeRed,
            # 'diff_cmap': cmaps.MPL_PuOr.reversed(),
            'rc_cmap': cmaps.MPL_RdBu.reversed(),
        },
        'ColdWet': {
            'diff_maplevs_seasonal': [np.linspace(-4, 4, 11), 5],
            'diff_maplevs_annual': [np.linspace(-6, 6, 11), 5],
            'rc_maplevs_seasonal': [np.linspace(-20, 20, 11), 5],
            'rc_maplevs_annual': [np.linspace(-24, 24, 11), 5],
            'diff_boxlevs': [np.linspace(-4, 4, 11), 5],
            'rc_boxlevs': [np.linspace(-50, 50, 11), 5],
            'diff_cmap': cmaps.BlueWhiteOrangeRed,
            # 'diff_cmap': cmaps.MPL_PuOr.reversed(),
            'rc_cmap': cmaps.MPL_RdBu.reversed(),
        },
        'ColdDry': {
            'diff_maplevs_seasonal': [np.linspace(-4, 4, 11), 5],
            'diff_maplevs_annual': [np.linspace(-4, 4, 11), 5],
            'rc_maplevs_seasonal': [np.linspace(-15, 15, 11), 5],
            'rc_maplevs_annual': [np.linspace(-50, 50, 11), 5],
            'diff_boxlevs': [np.linspace(-50, 50, 11), 5],
            'rc_boxlevs': [np.linspace(-100, 100, 11), 5],
            'diff_cmap': cmaps.BlueWhiteOrangeRed,
            # 'diff_cmap': cmaps.MPL_PuOr.reversed(),
            'rc_cmap': cmaps.MPL_RdBu.reversed(),
        },
        'Annual': {
            'diff_boxlevs': [np.linspace(-10, 10, 11), 5],
            'rc_boxlevs': [np.linspace(-70, 70, 11), 5],
        },
    }
    if varname in level_dicts.keys():
        return level_dicts[varname]
    else:
        raise ValueError(f"Variable {varname} is not defined.")




def Seasonal_ExtremeEvents_Intensity_Cmap(varname: str) -> dict:
    level_dicts = {
        'exT_Hot': {
            'diff_maplevs_seasonal': [np.linspace(-1, 1, 11), 5],
            'diff_maplevs_annual': [np.linspace(-0.8, 0.8, 11), 5],
            'rc_maplevs_seasonal': [np.linspace(-55, 55, 11), 5],
            'rc_maplevs_annual': [np.linspace(-16, 16, 11), 5],
            'diff_boxlevs': [np.linspace(-0.6, 0.6, 11), 5],
            'rc_boxlevs': [np.linspace(-40, 40, 11), 5],
            'diff_cmap': cmaps.BlueWhiteOrangeRed,
            'rc_cmap': cmaps.MPL_RdBu.reversed(),
        },
        'exT_Cold': {
            'diff_maplevs_seasonal': [np.linspace(-0.6, 0.6, 11), 5],
            'diff_maplevs_annual': [np.linspace(-0.4, 0.4, 11), 5],
            'rc_maplevs_seasonal': [np.linspace(-24, 24, 11), 5],
            'rc_maplevs_annual': [np.linspace(-30, 30, 11), 5],
            'diff_boxlevs': [np.linspace(-0.6, 0.6, 11), 5],
            'rc_boxlevs': [np.linspace(-30, 30, 11), 5],
            'diff_cmap': cmaps.BlueWhiteOrangeRed,
            'rc_cmap': cmaps.MPL_RdBu.reversed(),
        },
        'exRH_Wet': {
            'diff_maplevs_seasonal': [np.linspace(-1, 1, 11), 5],
            'diff_maplevs_annual': [np.linspace(-0.6, 0.6, 11), 5],
            'rc_maplevs_seasonal': [np.linspace(-30, 30, 11), 5],
            'rc_maplevs_annual': [np.linspace(-12, 12, 11), 5],
            'diff_boxlevs': [np.linspace(-2, 2, 11), 5],
            'rc_boxlevs': [np.linspace(-40, 40, 11), 5],
            'diff_cmap': cmaps.BlueWhiteOrangeRed,
            'rc_cmap': cmaps.MPL_RdBu.reversed(),
        },
        'exRH_Dry': {
            'diff_maplevs_seasonal': [np.linspace(-3, 3, 11), 5],
            'diff_maplevs_annual': [np.linspace(-1.2, 1.2, 11), 5],
            'rc_maplevs_seasonal': [np.linspace(-50, 50, 11), 5],
            'rc_maplevs_annual': [np.linspace(-30, 30, 11), 5],
            'diff_boxlevs': [np.linspace(-2, 2, 11), 5],
            'rc_boxlevs': [np.linspace(-30, 30, 11), 5],
            'diff_cmap': cmaps.BlueWhiteOrangeRed,
            'rc_cmap': cmaps.MPL_RdBu.reversed(),
        },
        'exTw_HotWet': {
            'diff_maplevs_seasonal': [np.linspace(-0.8, 0.8, 11), 5],
            'diff_maplevs_annual': [np.linspace(-0.6, 0.6, 11), 5],
            'rc_maplevs_seasonal': [np.linspace(-60, 60, 11), 5],
            'rc_maplevs_annual': [np.linspace(-20, 20, 11), 5],
            'diff_boxlevs': [np.linspace(-0.8, 0.8, 11), 5],
            'rc_boxlevs': [np.linspace(-50, 50, 11), 5],
            'diff_cmap': cmaps.BlueWhiteOrangeRed,
            'rc_cmap': cmaps.MPL_RdBu.reversed(),
        },
        'exTw_HotDry': {
            'diff_maplevs_seasonal': [np.linspace(-0.6, 0.6, 11), 5],
            'diff_maplevs_annual': [np.linspace(-3.0, 3.0, 11), 5],
            'rc_maplevs_seasonal': [np.linspace(-150, 150, 11), 5],
            'rc_maplevs_annual': [np.linspace(-40, 40, 11), 5],
            'diff_boxlevs': [np.linspace(-0.6, 0.6, 11), 5],
            'rc_boxlevs': [np.linspace(-300, 300, 11), 5],
            'diff_cmap': cmaps.BlueWhiteOrangeRed,
            'rc_cmap': cmaps.MPL_RdBu.reversed(),
        },
        'exTw_ColdWet': {
            'diff_maplevs_seasonal': [np.linspace(-1.5,1.5, 11), 5],
            'diff_maplevs_annual': [np.linspace(-2.0,2.0, 11), 5],
            'rc_maplevs_seasonal': [np.linspace(-70, 70, 11), 5],
            'rc_maplevs_annual': [np.linspace(-40, 40, 11), 5],
            'diff_boxlevs': [np.linspace(-0.8, 0.8, 11), 5],
            'rc_boxlevs': [np.linspace(-90, 90, 11), 5],
            'diff_cmap': cmaps.BlueWhiteOrangeRed,
            'rc_cmap': cmaps.MPL_RdBu.reversed(),
        },
        'exTw_ColdDry': {
            'diff_maplevs_seasonal': [np.linspace(-1, 1, 11), 5],
            'diff_maplevs_annual': [np.linspace(-0.6, 0.6, 11), 5],
            'rc_maplevs_seasonal': [np.linspace(-40, 40, 11), 5],
            'rc_maplevs_annual': [np.linspace(-30, 30, 11), 5],
            'diff_boxlevs': [np.linspace(-0.6, 0.6, 11), 5],
            'rc_boxlevs': [np.linspace(-40, 40, 11), 5],
            'diff_cmap': cmaps.BlueWhiteOrangeRed,
            'rc_cmap': cmaps.MPL_RdBu.reversed(),
        },
        'Annual': {
            'diff_boxlevs': [np.linspace(-3, 3, 11), 5],
            'rc_boxlevs': [np.linspace(-50, 50, 11), 5],
        },
    }

    if varname in level_dicts.keys():
        return level_dicts[varname]
    else:
        raise ValueError(f"Variable {varname} is not defined.")



def Seasonal_CaseDiff_Timeseries_Line_ColorRange(varname: str) -> dict:
    level_dicts = {
        'T2m': {
            'all': [np.linspace(-0.3, 0.3, 11), 7],
            'onlysign': [np.linspace(-0.5, 0.5, 11), 5],
            'rc': [np.linspace(-3, 3, 11), 7],
        },
        'T2m-Max': {
            'all': [np.linspace(-0.3, 0.3, 11), 7],
            'onlysign': [np.linspace(-1, 1, 11), 5],
            'rc': [np.linspace(-3, 3, 11), 5],
        },
        'T2m-Min': {
            'all': [np.linspace(-0.3, 0.3, 11), 7],
            'onlysign': [np.linspace(-1, 1, 11), 5],
            'rc': [np.linspace(-3, 3, 11), 5],
        },
        'Prec': {
            'all': [np.linspace(-0.8, 0.8, 11), 5],
            'onlysign': [np.linspace(-2, 2, 11), 5],
            'rc': [np.linspace(-15, 15, 11), 5],
        },
        'Prec-Max': {
            'all': [np.linspace(-20, 20, 11), 5],
            'onlysign': [np.linspace(-50, 50, 11), 5],
            'rc': [np.linspace(-20, 20, 11), 5],
        },
        'LHF': {
            'all': [np.linspace(-5, 5, 11), 5],
            'onlysign': [np.linspace(-50, 50, 11), 5],
            'rc': [np.linspace(-10, 10, 11), 5],
        },
        'SHF': {
            'all': [np.linspace(-2.5, 2.5, 11),  5],
            'onlysign': [np.linspace(-30, 30, 11),  5],
            'rc': [np.linspace(-5, 5, 11),  5],
        },
        'RH': {
            'all': [np.linspace(-1, 1, 11), 5],
            'onlysign': [np.linspace(-2, 2, 11), 5],
            'rc': [np.linspace(-1, 1, 11), 5],
        },
        'Q2m': {
            'all': [np.linspace(-0.4, 0.4, 11), 5],
            'onlysign': [np.linspace(-0.4, 0.4, 11), 5],
            'rc': [np.linspace(-10, 10, 11), 5],
        },
    }

    if varname in level_dicts.keys():
        return level_dicts[varname]
    else:
        raise ValueError(f"Variable {varname} is not defined.")



def Seasonal_Extreme_Timeseries_Line_ColorRange(varname: str) -> dict:
    level_dicts = {
        'Hot': {
            'all': [np.linspace(-6, 6, 11), 5],
            'onlysign': [np.linspace(-20, 20, 11), 5],
            'rc': [np.linspace(-3, 3, 11), 7],
        },
        'PrecWet': {
            'all': [np.linspace(-6, 6, 11), 5],
            'onlysign': [np.linspace(-20, 20, 11), 5],
            'rc': [np.linspace(-3, 3, 11), 5],
        },
        'PrecDry': {
            'all': [np.linspace(-6, 6, 11), 5],
            'onlysign': [np.linspace(-20, 20, 11), 5],
            'rc': [np.linspace(-3, 3, 11), 5],
        },
        'HotWet': {
            'all': [np.linspace(-6, 6, 11), 5],
            'onlysign': [np.linspace(-20, 20, 11), 5],
            'rc': [np.linspace(-20, 20, 11), 5],
        },
        'HotDry': {
            'all': [np.linspace(-6, 6, 11), 5],
            'onlysign': [np.linspace(-20, 20, 11), 5],
            'rc': [np.linspace(-20, 20, 11), 5],
        },
    }

    if varname in level_dicts.keys():
        return level_dicts[varname]
    else:
        raise ValueError(f"Variable {varname} is not defined.")



def Seasonal_ExtremeEvents_Freq_Range(varname: str) -> dict:
    level_dicts = {
        'Hot': {
            'levels': [np.linspace(-20, 20, 11), 5],
            'cmap': cmaps.BlueWhiteOrangeRed,
        },
        'PrecWet': {
            'levels': [np.linspace(-20, 20, 11), 5],
            'cmap': cmaps.BlueWhiteOrangeRed,
        },
        'PrecDry': {
            'levels': [np.linspace(-20, 20, 11), 5],
            'cmap': cmaps.BlueWhiteOrangeRed,
        },
        'HotWet': {
            'levels': [np.linspace(-20, 20, 11), 5],
            'cmap': cmaps.BlueWhiteOrangeRed,
        },
        'HotDry': {
            'levels': [np.linspace(-20, 20, 11), 5],
            'cmap': cmaps.BlueWhiteOrangeRed,
        },
    }
    if varname in level_dicts.keys():
        return level_dicts[varname]
    else:
        raise ValueError(f"Variable {varname} is not defined.")



def Seasonal_Extreme_Timeseries_Line_ColorRange(varname: str) -> dict:
    level_dicts = {
        'Hot': {
            'all': [np.linspace(-6, 6, 11), 5],
            'onlysign': [np.linspace(-20, 20, 11), 5],
            'rc': [np.linspace(-3, 3, 11), 7],
        },
        'PrecWet': {
            'all': [np.linspace(-6, 6, 11), 5],
            'onlysign': [np.linspace(-20, 20, 11), 5],
            'rc': [np.linspace(-3, 3, 11), 5],
        },
        'PrecDry': {
            'all': [np.linspace(-6, 6, 11), 5],
            'onlysign': [np.linspace(-20, 20, 11), 5],
            'rc': [np.linspace(-3, 3, 11), 5],
        },
        'HotWet': {
            'all': [np.linspace(-6, 6, 11), 5],
            'onlysign': [np.linspace(-20, 20, 11), 5],
            'rc': [np.linspace(-20, 20, 11), 5],
        },
        'HotDry': {
            'all': [np.linspace(-6, 6, 11), 5],
            'onlysign': [np.linspace(-20, 20, 11), 5],
            'rc': [np.linspace(-20, 20, 11), 5],
        },
    }

    if varname in level_dicts.keys():
        return level_dicts[varname]
    else:
        raise ValueError(f"Variable {varname} is not defined.")




def Coupling_Change_Cmap(varname: str) -> dict:
    level_dicts = {
        'pearson_r': {
            'levels': [np.linspace(-0.02, 0.02, 11), 5],
            'cmap': cmaps.MPL_PuOr.reversed(),
        },        
        'spearman_r': {
            'levels': [np.linspace(-0.02, 0.02, 11), 5],
            'cmap': cmaps.MPL_PuOr.reversed(),
        },
        'mutual_info': {
            'levels': [np.linspace(-0.02, 0.02, 11), 5],
            'cmap': cmaps.MPL_PuOr.reversed(),
        },
        'r2': {
            'levels': [np.linspace(-0.02, 0.02, 11), 5],
            'cmap': cmaps.MPL_PuOr.reversed(),
        },
        'lambda_u':{
            'levels': [np.linspace(-0.02, 0.02, 11), 5],
            'cmap': cmaps.MPL_PuOr.reversed(),
        },
        'kendall_tau':{
            'levels': [np.linspace(-0.02, 0.02, 11), 5],
            'cmap': cmaps.MPL_PuOr.reversed(),
        },
        'cov':{
            'levels': [np.linspace(-5, 5, 11), 5],
            'cmap': cmaps.MPL_PuOr.reversed(),
        }
    }
    if varname in level_dicts.keys():
        return level_dicts[varname]
    else:
        raise ValueError(f"Variable {varname} is not defined.")



def Seasonal_PressureLevel_Cmap(varname: str) -> dict:
    level_dicts = {
        'WA': {
            'diff_maplevs_seasonal': [np.linspace(-3, 3, 11), 7],
            'diff_maplevs_annual': [np.linspace(-3, 3, 11), 7],
            'diff_boxlevs': [np.linspace(-3, 3, 11), 5],
            'maplevs_seasonal': [np.linspace(-20, 20, 11), 5],
            'maplevs_annual': [np.linspace(-20, 20, 11), 5],
            'boxlevs': [np.linspace(-200, 200, 11), 5],
            'rc_maplevs_seasonal': [np.linspace(-200, 200, 11), 5],
            'rc_maplevs_annual': [np.linspace(-200, 200, 11), 5],
            'rc_boxlevs': [np.linspace(-200, 200, 11), 5],
            'cmap': cmaps.BlueWhiteOrangeRed,
            'diff_cmap': cmaps.MPL_PuOr.reversed(),
            'rc_cmap': cmaps.MPL_RdBu.reversed(),
        },
        'U': {
            'diff_maplevs_seasonal': [np.linspace(-3, 3, 11), 7],
            'diff_maplevs_annual': [np.linspace(-3, 3, 11), 7],
            'diff_boxlevs': [np.linspace(-3, 3, 11), 5],
            'maplevs_seasonal': [np.linspace(-10, 10, 11), 5],
            'maplevs_annual': [np.linspace(-10, 10, 11), 5],
            'boxlevs': [np.linspace(-100, 100, 11), 5],
            'rc_maplevs_seasonal': [np.linspace(-100, 100, 11), 5],
            'rc_maplevs_annual': [np.linspace(-100, 100, 11), 5],
            'rc_boxlevs': [np.linspace(-100, 100, 11), 5],
            'cmap': cmaps.BlueWhiteOrangeRed,
            'diff_cmap': cmaps.MPL_PuOr.reversed(),
            'rc_cmap': cmaps.MPL_RdBu.reversed(),
        },
        'V': {
            'diff_maplevs_seasonal': [np.linspace(-3, 3, 11), 7],
            'diff_maplevs_annual': [np.linspace(-3, 3, 11), 7],
            'diff_boxlevs': [np.linspace(-3, 3, 11), 5],
            'maplevs_seasonal': [np.linspace(-10, 10, 11), 5],
            'maplevs_annual': [np.linspace(-10, 10, 11), 5],
            'boxlevs': [np.linspace(-100, 100, 11), 5],
            'rc_maplevs_seasonal': [np.linspace(-100, 100, 11), 5],
            'rc_maplevs_annual': [np.linspace(-100, 100, 11), 5],
            'rc_boxlevs': [np.linspace(-100, 100, 11), 5],
            'cmap': cmaps.BlueWhiteOrangeRed,
            'diff_cmap': cmaps.MPL_PuOr.reversed(),
            'rc_cmap': cmaps.MPL_RdBu.reversed(),
        },
        'UV': {
            'diff_maplevs_seasonal': [np.linspace(-1, 1, 11), 5],
            'diff_maplevs_annual': [np.linspace(-1, 1, 11), 5],
            'diff_boxlevs': [np.linspace(-2, 2, 11), 5],
            'maplevs_seasonal': [np.linspace(-10, 10, 11), 5],
            'maplevs_annual': [np.linspace(-10, 10, 11), 5],
            'boxlevs': [np.linspace(-100, 100, 11), 5],
            'rc_maplevs_seasonal': [np.linspace(-15, 15, 11), 5],
            'rc_maplevs_annual': [np.linspace(-15, 15, 11), 5],
            'rc_boxlevs': [np.linspace(-15, 15, 11), 5],
            'diff_cmap': cmaps.BlueWhiteOrangeRed,
            # 'diff_cmap': cmaps.MPL_PuOr.reversed(),
            'rc_cmap': cmaps.MPL_RdBu.reversed(),
            'roselevels': [np.linspace(-0.4, 0.4, 11), 5],
        },
        'W': {
            'diff_maplevs_seasonal': [np.linspace(-3, 3, 11), 7],
            'diff_maplevs_annual': [np.linspace(-3, 3, 11), 7],
            'diff_boxlevs': [np.linspace(-3, 3, 11), 5],
            'maplevs_seasonal': [np.linspace(-10, 10, 11), 5],
            'maplevs_annual': [np.linspace(-10, 10, 11), 5],
            'boxlevs': [np.linspace(-100, 100, 11), 5],
            'rc_maplevs_seasonal': [np.linspace(-100, 100, 11), 5],
            'rc_maplevs_annual': [np.linspace(-100, 100, 11), 5],
            'rc_boxlevs': [np.linspace(-100, 100, 11), 5],
            'cmap': cmaps.BlueWhiteOrangeRed,
            'diff_cmap': cmaps.MPL_PuOr.reversed(),
            'rc_cmap': cmaps.MPL_RdBu.reversed(),
        },
        'T': {
            'diff_maplevs_seasonal': [np.linspace(-0.3, 0.3, 11), 7],
            'diff_maplevs_annual': [np.linspace(-0.3, 0.3, 11), 7],
            'diff_boxlevs': [np.linspace(-3, 3, 11), 5],
            'maplevs_seasonal': [np.linspace(-20, 20, 11), 5],
            'maplevs_annual': [np.linspace(-20, 20, 11), 5],
            'boxlevs': [np.linspace(-200, 200, 11), 5],
            'rc_maplevs_seasonal': [np.linspace(-5, 5, 11), 5],
            'rc_maplevs_annual': [np.linspace(-5, 5, 11), 5],
            'rc_boxlevs': [np.linspace(-200, 200, 11), 5],
            'cmap': cmaps.BlueWhiteOrangeRed,
            'diff_cmap': cmaps.MPL_PuOr.reversed(),
            'rc_cmap': cmaps.MPL_RdBu.reversed(),
        },
        'Theta': {
            'diff_maplevs_seasonal': [np.linspace(-0.3, 0.3, 11), 7],
            'diff_maplevs_annual': [np.linspace(-0.3, 0.3, 11), 7],
            'diff_boxlevs': [np.linspace(-3, 3, 11), 5],
            'maplevs_seasonal': [np.linspace(-20, 20, 11), 5],
            'maplevs_annual': [np.linspace(-20, 20, 11), 5],
            'boxlevs': [np.linspace(-200, 200, 11), 5],
            'rc_maplevs_seasonal': [np.linspace(-0.1, 0.1, 11), 5],
            'rc_maplevs_annual': [np.linspace(-0.1, 0.1, 11), 5],
            'rc_boxlevs': [np.linspace(-200, 200, 11), 5],
            'cmap': cmaps.BlueWhiteOrangeRed,
            'diff_cmap': cmaps.MPL_PuOr.reversed(),
            'rc_cmap': cmaps.MPL_RdBu.reversed(),
        },
        'CloudFra': {
            'diff_maplevs_seasonal': [np.linspace(-1.5, 1.5, 11), 5],
            'diff_maplevs_annual': [np.linspace(-1, 1, 11), 5],
            'diff_boxlevs': [np.linspace(-5, 5, 11), 5],
            'maplevs_seasonal': [np.linspace(-20, 20, 11), 5],
            'maplevs_annual': [np.linspace(-20, 20, 11), 5],
            'boxlevs': [np.linspace(-50, 50, 11), 5],
            'rc_maplevs_seasonal': [np.linspace(-10, 10, 11), 5],
            'rc_maplevs_annual': [np.linspace(-10, 10, 11), 5],
            'rc_boxlevs': [np.linspace(-50, 50, 11), 5],
            'cmap': cmaps.BlueWhiteOrangeRed,
            'diff_cmap': cmaps.MPL_PuOr.reversed(),
            'rc_cmap': cmaps.MPL_RdBu.reversed(),
        },
        'Height': {
            'diff_maplevs_seasonal': [np.linspace(-0.5, 0.5, 11), 7],
            'diff_maplevs_annual': [np.linspace(-0.5, 0.5, 11), 7],
            'diff_boxlevs': [np.linspace(-3, 3, 11), 5],
            'maplevs_seasonal': [np.linspace(-20, 20, 11), 5],
            'maplevs_annual': [np.linspace(-20, 20, 11), 5],
            'boxlevs': [np.linspace(-200, 200, 11), 5],
            'rc_maplevs_seasonal': [np.linspace(-1, 1, 11), 5],
            'rc_maplevs_annual': [np.linspace(-1, 1, 11), 5],
            'rc_boxlevs': [np.linspace(-200, 200, 11), 5],
            'cmap': cmaps.BlueWhiteOrangeRed,
            'diff_cmap': cmaps.MPL_PuOr.reversed(),
            'rc_cmap': cmaps.MPL_RdBu.reversed(),
        },

    }
    if varname in level_dicts.keys():
        return level_dicts[varname]
    else:
        raise ValueError(f"Variable {varname} is not defined.")



def ExtremeEvents_LinePlot_Range(varname: str) -> dict:
    level_dicts = {
        'Hot': {
            'season_area': [np.linspace(-16, 16, 11), 5],
            'season_duration': [np.linspace(-0.02, 0.02, 11), 5],
            'annual_area': [np.linspace(-10, 10, 11), 5],
            'annual_duration': [np.linspace(-0.02, 0.02, 11), 5],
        },
        'HotWet': {
            'season_area': [np.linspace(-16, 16, 11), 5],
            'season_duration': [np.linspace(-0.02, 0.02, 11), 5],
            'annual_area': [np.linspace(-10, 10, 11), 5],
            'annual_duration': [np.linspace(-0.02, 0.02, 11), 5],
        },
        'HotDry': {
            'season_area': [np.linspace(-16, 16, 11), 5],
            'season_duration': [np.linspace(-0.02, 0.02, 11), 5],
            'annual_area': [np.linspace(-10, 10, 11), 5],
            'annual_duration': [np.linspace(-0.02, 0.02, 11), 5],
        },
        'ColdWet': {
            'season_area': [np.linspace(-16, 16, 11), 5],
            'season_duration': [np.linspace(-0.02, 0.02, 11), 5],
            'annual_area': [np.linspace(-10, 10, 11), 5],
            'annual_duration': [np.linspace(-0.02, 0.02, 11), 5],
        },
        'ColdDry': {
            'season_area': [np.linspace(-16, 16, 11), 5],
            'season_duration': [np.linspace(-0.02, 0.02, 11), 5],
            'annual_area': [np.linspace(-10, 10, 11), 5],
            'annual_duration': [np.linspace(-0.02, 0.02, 11), 5],
        },
        'Cold': {
            'season_area': [np.linspace(-16, 16, 11), 5],
            'season_duration': [np.linspace(-0.02, 0.02, 11), 5],
            'annual_area': [np.linspace(-10, 10, 11), 5],
            'annual_duration': [np.linspace(-0.02, 0.02, 11), 5],
        },
    }
    if varname in level_dicts.keys():
        return level_dicts[varname]
    else:
        raise ValueError(f"Variable {varname} is not defined.")



def Affected_Population_Cmap(event: str) -> dict:
    level_dicts = {
        'Hot': {
            'grid': [np.linspace(0, 30000, 7), 5],
            'city': [np.linspace(0, 3000, 13), 5],
            'cmap': cmaps.cmocean_matter,
        },
        'Cold': {
            'grid': [np.linspace(0, 30000, 7), 5],
            'city': [np.linspace(0, 3000, 13), 5],
            'cmap': cmaps.cmocean_matter,
        },
        'Wet': {
            'grid': [np.linspace(0, 30000, 7), 5],
            'city': [np.linspace(0, 3000, 13), 5],
            'cmap': cmaps.cmocean_matter,
        },
        'Dry': {
            'grid': [np.linspace(0, 30000, 7), 5],
            'city': [np.linspace(0, 3000, 13), 5],
            'cmap': cmaps.cmocean_matter,
        }, 
        'HotWet': {
            'grid': [np.linspace(0, 20000, 7), 5],
            'city': [np.linspace(0, 3000, 13), 5],
            'cmap': cmaps.cmocean_matter,
        },
        'HotDry': {
            'grid': [np.linspace(0, 20000, 7), 5],
            'city': [np.linspace(0, 3000, 13), 5],
            'cmap': cmaps.cmocean_matter,
        },
        'ColdWet': {
            'grid': [np.linspace(0, 20000, 7), 5],
            'city': [np.linspace(0, 3000, 13), 5],
            'cmap': cmaps.cmocean_matter,
        },
        'ColdDry': {
            'grid': [np.linspace(0, 20000, 7), 5],
            'city': [np.linspace(0, 3000, 13), 5],
            'cmap': cmaps.cmocean_matter,
        },
    }
    if event in level_dicts.keys():
        return level_dicts[event]
    else:
        raise ValueError(f"Event {event} is not defined.")





def season_linecolor(season: str) -> str:
    linecolors = {
        'DJF': '#9C0C5E',
        'MAM': '#FF684A',
        'JJA': '#FFA96B',
        'SON': '#FFE8B3',
    }
    if season in linecolors.keys():
        return linecolors[season]
    else:
        raise ValueError(f"Season {season} is not defined.")
    

def season_marker(season: str) -> str:
    markers = {
        'DJF': '.',
        'MAM': 'o',
        'JJA': '^',
        'SON': 'D',
    }
    if season in markers.keys():
        return markers[season]
    else:
        raise ValueError(f"Season {season} is not defined.")



def CaseDiff_Quantiled_Colors(name: str) -> str:
    linecolors = {
        'Mean': '#9C0C5E',
        'Mid': '#FF684A',
        'Out': '#FFA96B',
    }
    if name in linecolors.keys():
        return linecolors[name]
    else:
        raise ValueError(f"name {name} is not defined.")
    


def CaseRC_Quantiled_Colors(name: str) -> str:
    linecolors = {
        'Mean': '#C60069',
        'Mid': '#53589A',
        'Out': '#009AD2',
    }
    if name in linecolors.keys():
        return linecolors[name]
    else:
        raise ValueError(f"name {name} is not defined.")
    

def HotWet_Coupling_Cmap():
    region_colors = ["#283c63", "#fbe8d3", "#f73859"]  # Blue for Down, Light Blue for None, Red for Up
    cmap = mcolors.ListedColormap(region_colors)
    return cmap


def MapPlot_CropParams_noColorbar():
    return {"left": 0.005, "top": 0.005, "right": 0.005, "bottom": 0.005}


def RosePlot_CropParams_noColorbar():
    return {"left": 0.005, "top": 0.005, "right": 0.005, "bottom": 0.005}


def MapCbar_CropParams_V():
    return {"left": 0.005, "top": 0.005, "right": 0.000, "bottom": 0.005}


def MapCbar_CropParams_H():
    return {"left": 0.005, "top": 0.005, "right": 0.005, "bottom": 0.000}


def Extreme_Events_BoxPlot_CropParams():
    return {"left": 0.005, "top": 0.005, "right": 0.005, "bottom": 0.005}


def Extreme_RegClimImpact_BoxPlot_CropParams():
    return {"left": 0.005, "top": 0.005, "right": 0.005, "bottom": 0.005}


def Merge_Fig_Space_Params():
    return {'left': 0.01, 'top': 0.01, 'right': 0.01, 'bottom': 0.01}


def Zero_space_CropParams():
    return {'left': 0.0, 'top': 0.0, 'right': 0.0, 'bottom': 0.0}
