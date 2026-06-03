
#!/stu01/wumej22/Anaconda3/envs/cwrf_env/bin/python

import argparse
import time
from pathlib import Path
from typing import Dict, List
import os
import shapely.geometry as sgeom
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib

matplotlib.use("Agg")  # 非交互式后端
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from cnmaps import get_adm_maps
import geopandas as gpd
import cmaps

# ------------------- 1. 默认绘图配置 -------------------
DefaultDict = dict(
    dem_coarsen=1,        # DEM 采样倍率 (>1 表示降采样，1=原分辨率)
    draw_lake=True,      # 是否绘制湖泊 (cartopy 自带湖泊数据)
    draw_river=False,     # 是否绘制河流 (cartopy 自带河流数据)
    draw_province=False,   # 是否绘制省级行政区 (cnmaps)
    draw_country=True,    # 是否绘制国界 (cnmaps)
    draw_city=False,      # 是否绘制市级行政区 (cnmaps，绘制较慢)
    shapefile='/stu01/wumej22/data/ERA5/yangtze_res_shp/YangtzeBasin.shp',       # 额外 shapefile 路径
    deeplake='/hydata01/wumej22/Experiment/Process/Yangtze_C_6km_r1/BaseData/yangtze_deep_lake.gpkg',
    shallowlake='/hydata01/wumej22/Experiment/Process/Yangtze_C_6km_r1/BaseData/yangtze_shallow_lake.gpkg',
    scwater='/hydata01/wumej22/Experiment/Process/Yangtze_C_6km_r1/BaseData/SC_WATER.gpkg',
)

# ------------------- 2. 命令行参数 -------------------
def parse_args():
    p = argparse.ArgumentParser(description="生成带地形的 WRF 网格示意图", formatter_class=argparse.RawTextHelpFormatter)

    # ---- WRF 网格参数 ----
    p.add_argument("--casename", default="Yangtze", help="案例名称\n")
    p.add_argument("--proj", default="LAMBERT", choices=["LAMBERT"], help="水平投影类型\n")
    p.add_argument("--RefLat", type=float, default=29.5, help="投影中心纬度 (°)")
    p.add_argument("--RefLon", type=float, default=114.3, help="投影中心经度 (°)")
    p.add_argument("--True_Lat1", type=float, default=27, help="投影第一标准纬线 (°)")
    p.add_argument("--True_Lat2", type=float, default=33, help="投影第二标准纬线 (°)")
    p.add_argument("--dx_WE", type=float, default=6000.0, help="水平网格距 (东西向, m)")
    p.add_argument("--dy_SN", type=float, default=6000.0, help="水平网格距 (南北向, m)")
    p.add_argument("--EdgeNum_WE", type=int, default=316, help="东西向网格点数 (e_we)")
    p.add_argument("--EdgeNum_SN", type=int, default=217, help="南北向网格点数 (e_sn)")
    p.add_argument("--savepath", type=str, default="./Figures/StudyArea.png", help="输出文件保存路径")
    p.add_argument("--BdyWidth", type=int, default=15, help="WRF 边界缓冲区")
    p.add_argument("--topodir", type=str, default="/stu01/wumej22/CWRF/CWPS_GEOG/geog_new/geog_wm_modified_lake/topo_30s/", help="DEM 目录")
    
    p.add_argument("--plotcfg", nargs="*", metavar="KEY=VAL", default=[], help="修改绘图参数")
    return p.parse_args()



# ------------------- 3. 工具函数 -------------------
def parse_plot_cfg(DefaultDict, items):
    newDict = DefaultDict.copy()
    dictkeys = set(newDict.keys())
    for item in items:
        if "=" not in item: continue
        k, v = item.split("=", 1)
        v_str, k_str = v.strip(), k.strip().lower()
        if k_str not in dictkeys: continue
        if k_str in ("draw_lake", "draw_river", "draw_province", "draw_country", "draw_city"):
            values = v_str.lower() == "true"
        elif k_str == "dem_coarsen":
            values = int(v_str)
        else:
            values = v_str
        newDict[k_str] = values
    return newDict



def cal_corner_coords(g, proj):
    x0, y0 = -((g["e_we"] - 1) / 2) * g["dx"], -((g["e_sn"] - 1) / 2) * g["dy"]
    x1, y1 = x0 + g["e_we"] * g["dx"], y0 + g["e_sn"] * g["dy"]
    bdx, bdy = g["BdyWidth"] * g["dx"], g["BdyWidth"] * g["dy"]
    lam = dict(left=x0, right=x1, bottom=y0, top=y1)
    inner = dict(left=x0+bdx, right=x1-bdx, bottom=y0+bdy, top=y1-bdy)
    geo = ccrs.PlateCarree()
    wgs = {k: geo.transform_point(*pt, src_crs=proj) for k, pt in
           dict(left_bottom=(lam["left"], lam["bottom"]), right_bottom=(lam["right"], lam["bottom"]),
                left_top=(lam["left"], lam["top"]), right_top=(lam["right"], lam["top"])).items()}
    return lam, wgs, inner



def read_elevation(topodir, corner_wgs, fac):
    known_lat, known_lon, ddeg = -89.99583, -179.99583, 0.00833333
    tile_x = 1200
    tile_y = 1200
    tile_bdr = 3
    nx_full = ny_full = tile_x + 2 * tile_bdr
    
    dtype_tile = ">i2"
    lons_c, lats_c = [v[0] for v in corner_wgs.values()], [v[1] for v in corner_wgs.values()]
    tx_min, tx_max = [(f(lons_c) - known_lon) / ddeg for f in (min, max)]
    ty_min, ty_max = [(f(lats_c) - known_lat) / ddeg for f in (min, max)]
    
    xstarts = list(range(1, 43200, tile_x))
    ystarts = list(range(1, 21600, tile_y))
    
    xid = [i for i, xs in enumerate(xstarts) if xs <= tx_max and (xs + tile_x - 1) >= tx_min]
    yid = [j for j, ys in enumerate(ystarts) if ys <= ty_max and (ys + tile_y - 1) >= ty_min]
    
    big = np.zeros((len(yid) * tile_y, len(xid) * tile_x), np.int16)
    
    for ix, i in enumerate(xid):
        for iy, j in enumerate(yid):
            xs, ys = xstarts[i], ystarts[j]
            # 这里也需要修正：xe = xs + tile_x - 1, ye = ys + tile_y - 1
            xe, ye = xs + tile_x - 1, ys + tile_y - 1
            path = Path(topodir) / f"{xs:05d}-{xe:05d}.{ys:05d}-{ye:05d}"
            if path.exists():
                tile = np.fromfile(path, dtype=dtype_tile).reshape(ny_full, nx_full)[tile_bdr:-tile_bdr, tile_bdr:-tile_bdr]
                big[iy * tile_y : (iy + 1) * tile_y, ix * tile_x : (ix + 1) * tile_x] = tile
                
    lons = known_lon + (xstarts[xid[0]] - 1 + np.arange(big.shape[1])) * ddeg
    lats = known_lat + (ystarts[yid[0]] - 1 + np.arange(big.shape[0])) * ddeg
    
    da = xr.DataArray(big, dims=("lat", "lon"), coords=dict(lat=lats, lon=lons))
    if fac > 1: 
        da = da.coarsen(lat=fac, lon=fac, boundary="trim", coord_func="mean").mean().astype("float32")
    return da



# ------------------- 4. 投影辅助函数 (刻度控制) -------------------

def _lambert_ticks(ax, ticks, tick_location, line_constructor, n_samples=400):
    minlon, maxlon, minlat, maxlat = ax.get_extent(ccrs.PlateCarree())
    lon_c, lat_c = np.array([minlon, maxlon, maxlon, minlon]), np.array([minlat, minlat, maxlat, maxlat])
    corners = ax.projection.transform_points(ccrs.Geodetic(), lon_c, lat_c)[..., :2]
    corners = corners[~np.isnan(corners).any(axis=1)]
    if len(corners) == 0: return [], []
    pminx, pmaxx, pminy, pmaxy = corners[:,0].min(), corners[:,0].max(), corners[:,1].min(), corners[:,1].max()
    tick_positions, tick_labels = [], []
    for t in ticks:
        lonlat = line_constructor(t, n_samples, (minlon, maxlon, minlat, maxlat))
        proj_pts = ax.projection.transform_points(ccrs.Geodetic(), lonlat[:, 0], lonlat[:, 1])[..., :2]
        pts = proj_pts[~np.isnan(proj_pts).any(axis=1)]
        if pts.shape[0] == 0: continue
        if tick_location == 'bottom':
            idx = np.argmin(pts[:,1]); tick_positions.append(float(pts[idx,0]))
        elif tick_location == 'left':
            idx = np.argmin(pts[:,0]); tick_positions.append(float(pts[idx,1]))
        tick_labels.append(t)
    return tick_positions, tick_labels



def lambert_xticks(ax, ticks):
    original_xlim = ax.get_xlim()
    xticks, xtlabels = _lambert_ticks(ax, ticks, 'bottom', lambda t, n, b: np.vstack((np.zeros(n) + t, np.linspace(b[2], b[3], n))).T)
    if xticks:
        ax.xaxis.tick_bottom()
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{lon:g}°E" if lon >= 0 else f"{abs(lon):g}°W" for lon in xtlabels])
        ax.set_xlim(original_xlim)
    return xticks, xtlabels



def lambert_yticks(ax, ticks):
    original_ylim = ax.get_ylim()
    yticks, ytlabels = _lambert_ticks(ax, ticks, 'left', lambda t, n, b: np.vstack((np.linspace(b[0], b[1], n), np.zeros(n) + t)).T)
    if yticks:
        ax.yaxis.tick_left()
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{lat:g}°N" if lat >= 0 else f"{abs(lat):g}°S" for lat in ytlabels])
        ax.set_ylim(original_ylim)
    return yticks, ytlabels



# ------------------- 5. 绘图主函数 -------------------
def create_map(grid, proj, corner_lam, inner, elev_da, cfg, args):
    elev_land = elev_da.where(elev_da > 0)
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=proj)

    # 5-1 地形底色
    cm = cmaps.WhiteBlueGreenYellowRed
    X, Y = np.meshgrid(elev_land.lon, elev_land.lat)
    pcm = ax.pcolormesh(X, Y, elev_land, cmap=cm, shading="auto", vmax=2000, vmin=0, rasterized=True, transform=ccrs.PlateCarree(), zorder=0)
    plt.colorbar(pcm, ax=ax, shrink=0.6, pad=0.03, label="Elevation (m)")

    # 5-2 行政区划与要素
    if cfg["draw_province"]:
        ax.add_geometries(get_adm_maps(level="省", engine="geopandas").geometry, ccrs.PlateCarree(), fc="none", ec="k", lw=0.6)
    if cfg["draw_country"]:
        ax.add_geometries(get_adm_maps(level="国", engine="geopandas").geometry, ccrs.PlateCarree(), fc="none", ec="k", lw=1.0)
    # if cfg["shapefile"]:
    #     ax.add_geometries(gpd.read_file(cfg["shapefile"]).geometry, ccrs.PlateCarree(), fc="none", ec="red", lw=2)
    
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#49E0E1", zorder=1)
    if cfg["draw_lake"]:
        shp_water = gpd.read_file(cfg["scwater"]).to_crs(epsg=4326)
        ax.add_geometries(shp_water.geometry, ccrs.PlateCarree(), fc="#1020b2", ec="#1020b2", lw=0.5, zorder=3)

    # 5-5 缓冲区蒙版 (阴影与 Hatch)
    L, R, B, T = corner_lam["left"], corner_lam["right"], corner_lam["bottom"], corner_lam["top"]
    il, ir, ib, it = inner["left"], inner["right"], inner["bottom"], inner["top"]
    vertices = [(L, B), (R, B), (R, T), (L, T), (L, B), (il, ib), (il, it), (ir, it), (ir, ib), (il, ib)]
    codes = [MplPath.MOVETO] + [MplPath.LINETO]*3 + [MplPath.CLOSEPOLY]
    path = MplPath(vertices, codes + codes)
    ax.add_patch(PathPatch(path, facecolor='none', edgecolor='black', hatch='//', alpha=0.3, linewidth=0, zorder=5))
    ax.plot([il, ir, ir, il, il], [ib, ib, it, it, ib], transform=proj, color="black", linewidth=1.2, zorder=6)

    # 设定显示范围为内框
    ax.set_extent([il, ir, ib, it], crs=proj)

    # ------------------- 5-6 经纬度控制 (修正部分) -------------------
    # 计算内框地理范围
    geo_extent = ccrs.PlateCarree().transform_points(proj, np.array([il, ir]), np.array([ib, it]))
    lon_min, lon_max = geo_extent[:, 0].min(), geo_extent[:, 0].max()
    lat_min, lat_max = geo_extent[:, 1].min(), geo_extent[:, 1].max()
    
    # 计算步长与 Ticks
    lon_step = max(2, round((lon_max - lon_min)/4))
    lat_step = max(2, round((lat_max - lat_min)/4))
    lon_ticks = np.arange(np.floor(lon_min), np.ceil(lon_max)+1, lon_step)
    lat_ticks = np.arange(np.floor(lat_min), np.ceil(lat_max)+1, lat_step)
    lon_ticks = [108, 110, 112, 114, 116, 118, 120, 122]
    lat_ticks = [25, 27, 29, 31, 33]

    # 绘制 Ticks 标签
    lambert_xticks(ax, lon_ticks)
    lambert_yticks(ax, lat_ticks)
    
    # 绘制网格线并对齐 Ticks
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=0.6, color='gray', alpha=0.5, linestyle='--')
    gl.xlocator = matplotlib.ticker.FixedLocator(lon_ticks)
    gl.ylocator = matplotlib.ticker.FixedLocator(lat_ticks)
    ax.tick_params(axis='both', which='major', labelsize=10, pad=8)

    # 5-7 标记点
    ax.scatter(120.193, 31.225,marker="*", s=65,fc="yellow", ec="yellow", lw=0.6,zorder=20,transform=ccrs.PlateCarree())
    ax.text(120.193, 30.889, "Lake Taihu\n(2 m)", ha="center", va="top", fontsize=10, weight="bold", color="k", zorder=21,transform=ccrs.PlateCarree(),)

    ax.scatter(118.9, 29.519,marker="*", s=65,fc="yellow", ec="yellow", lw=0.6,zorder=20,transform=ccrs.PlateCarree())
    ax.text(118.9, 29.329, "Lake Qiandaohu\n(30 m)", ha="center", va="top", fontsize=10, weight="bold", color="k", zorder=21,transform=ccrs.PlateCarree(),)

    ax.scatter(111.3, 32.732, marker="*", s=65,fc="yellow", ec="yellow", lw=0.6,zorder=20,transform=ccrs.PlateCarree())
    ax.text(111.3, 32.995, "Danjiangkou Reservoir\n(73 m)", ha="center", va="bottom", fontsize=10, weight="bold", color="k", zorder=21,transform=ccrs.PlateCarree(),)

    ax.scatter(116.356, 29.099,marker="*", s=65,fc="yellow", ec="yellow", lw=0.6,zorder=20,transform=ccrs.PlateCarree())
    ax.text(115.000, 28.95, "Lake Poyang\n(7 m)", ha="center", va="top", fontsize=10, weight="bold", color="k", zorder=21, transform=ccrs.PlateCarree())

    # ax.scatter(111.96, 28.95,marker="^", s=65,fc="yellow", ec="yellow", lw=0.6,zorder=20,transform=ccrs.PlateCarree())
    # ax.text(111.96, 28.6, "Representative\npoints", ha="center", va="top", fontsize=10, weight="bold", color="k", zorder=21,transform=ccrs.PlateCarree(),)

    plt.savefig(args.savepath or f"./{grid['casename']}_map.png", dpi=600, bbox_inches="tight")
    plt.close()



if __name__ == "__main__":
    time0 = time.time()
    args = parse_args()
    cfg = parse_plot_cfg(DefaultDict, args.plotcfg)
    grid = dict(casename=args.casename, lat_0=args.RefLat, lon_0=args.RefLon, lat_1=args.True_Lat1, lat_2=args.True_Lat2,
                dx=args.dx_WE, dy=args.dy_SN, BdyWidth=args.BdyWidth, e_we=args.EdgeNum_WE, e_sn=args.EdgeNum_SN)
    proj = ccrs.LambertConformal(central_longitude=grid["lon_0"], central_latitude=grid["lat_0"], standard_parallels=(grid["lat_1"], grid["lat_2"]))
    lam, wgs, inner = cal_corner_coords(grid, proj)
    elev = read_elevation(args.topodir, wgs, cfg["dem_coarsen"])
    create_map(grid, proj, lam, inner, elev, cfg, args)

    time1 = time.time()
    print(f"完成绘图，耗时 {time1 - time0:.2f} 秒")