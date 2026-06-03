import os
import warnings
import matplotlib as mpl
from matplotlib import font_manager
from typing import List, Any
from dataclasses import dataclass

# 自定义工具箱
import ToolBoxes.Utils as TU

mpl.use('Agg')  # 不显示图，只保存
warnings.filterwarnings("ignore", category=RuntimeWarning)


def apply_global_plot_style() -> None:
    for font_name in [
        "NotoSans-Regular.ttf",
        "NotoSans-Bold.ttf",
        "NotoSansSC-Regular.ttf",
        "NotoSansSC-Bold.ttf",
    ]:
        font_path = os.path.join("/home/wumej22/.local/share/fonts", font_name)
        if os.path.exists(font_path):
            font_manager.fontManager.addfont(font_path)

    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Noto Sans', 'Noto Sans SC', 'DejaVu Sans', 'Arial']
    mpl.rcParams['mathtext.fontset'] = 'custom'
    mpl.rcParams['mathtext.rm'] = 'Noto Sans'
    mpl.rcParams['mathtext.it'] = 'Noto Sans:italic'
    mpl.rcParams['mathtext.bf'] = 'Noto Sans:bold'
    mpl.rcParams['mathtext.default'] = 'rm'
    mpl.rcParams['axes.unicode_minus'] = False


apply_global_plot_style()



#***************************************
DPI_high = 1200  # 输出图片的分辨率
DPI_medium = 600  # 输出图片的分辨率
DPI_low  = 300  # 输出图片的分辨率
FIGFMT = 'png'  # 输出图片的格式

HighThresPercentile = 85
LowThresPercentile  = 15
ThresWindows        = 7     # set is {x-7, ..., x, ..., x+7}
fdr_alpha           = 0.2
#***************************************


#*********************画图配置类*******************************#
@dataclass
class mapConfig:
    levs: List[float]
    cmap: Any

@dataclass
class boxConfig:
    diff_boxlevs: List[float]
    rc_boxlevs: List[float]

@dataclass 
class roseConfig:
    roselevs: List[float]
    colors_dict: Any

@dataclass
class varInfo:
    longname: str
    abbr: str
    unit: str

@dataclass
class eventState:
    up: Any
    down: Any
    zero: Any
    none: Any

#*************************************************************#

#**********************基础数据路径**************************#
BASEDATA = '/hydata01/wumej22/Experiment/Process/Yangtze_C_6km_r1/BaseData'
# ******************************************************* #
