#!/stu01/wumej22/Anaconda3/envs/cwrf_env/bin/python
"""
Compute annual population exposure to lake-induced Hot-Wet event increases.

Definition used here:
    exposure_mask(year, y, x) =
        HotWet_days_Lake(year, y, x) - HotWet_days_NoLake(year, y, x) > threshold

The default threshold is 1 day. Population is kept at LandScan resolution for
aggregation: annual LandScan pixels are summed by their precomputed CWRF mesh
index, then the selected CWRF meshes are totaled for each year.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
from tqdm import tqdm

import ToolBoxes.Tool_DataPrepare as TDP
import ToolBoxes.Tool_PlotAreaMap as TPAM
import ToolBoxes.Tool_PlotBar as TPBH
import ToolBoxes.Tool_PlotColorBar as TPCB
import ToolBoxes.Tool_PlotConfig as TPC
import ToolBoxes.Tool_PlotRadialHistogram as TPRH
import ToolBoxes.Tool_YangtzeColorMap as TYCM

mpl.use("Agg")
FIGFMT = TPC.FIGFMT


# =========================
# User-configurable settings
# =========================

START_YEAR = 2000
END_YEAR = 2024
THRESHOLD_DAYS = 1.0
BUFFERZONE = 15
EVENT_VAR = "HotWet_Identified"

EXTREME_DIR = Path(
    "/hydata01/wumej22/Experiment/Process/Yangtze_C_6km_r1/Cases/ExtremeAnalysis"
)
LANDSCAN_DIR = Path(
    "/hydata01/wumej22/Experiment/CWRF/Yangtze_C/RefData/Landscan"
)
POPULATION_TEMPLATE = str(
    LANDSCAN_DIR / "landscan/nc/landscan-global-{year}-population_sel.nc"
)
MESH_PATH = LANDSCAN_DIR / "Yangtze_C_mesh_43200_21600_sel.nc"
CITYMESH_PATH = LANDSCAN_DIR / "CityMesh_on_Landscan.nc"
CITY_GPKG_PATH = Path(
    "/hydata01/wumej22/Experiment/Process/Yangtze_C_6km_r1/BaseData/china_yangtze_en.gpkg"
)

LAKE_EVENT_PATH = EXTREME_DIR / "Extreme_Events_Lake_ref_NoLake_identified_normalize.nc"
NOLAKE_EVENT_PATH = EXTREME_DIR / "Extreme_Events_NoLake_ref_NoLake_identified_normalize.nc"

OUTDIR = EXTREME_DIR
OUTPUT_PREFIX = "Exposure_HotWet_LakeMinusNoLake_gt1day_annual"
COMPUTE_ADMIN_STATS = True
WRITE_ADMIN_GPKG = True
COMPUTE_MULTIYEAR_MEAN = True
PLOT_MULTYEAR_MEAN = True
FIGOUTDIR = Path(
    "/hydata01/wumej22/Experiment/Process/Yangtze_C_6km_r1/Figures/ExtremeEventAddtional"
)


def _sort_lat_ascending(ds: xr.Dataset) -> xr.Dataset:
    if "lat" in ds.coords and ds["lat"].values[0] > ds["lat"].values[-1]:
        return ds.sortby("lat")
    return ds


def load_mesh(mesh_path: str, bufferzone: int) -> tuple[np.ndarray, xr.DataArray]:
    with xr.open_dataset(mesh_path) as ds_raw:
        ds = _sort_lat_ascending(ds_raw)
        mesh = ds["meshnum"].values
        if bufferzone > 0:
            mesh = mesh[bufferzone:-bufferzone, bufferzone:-bufferzone]
        mesh_int = mesh.astype(np.int64, copy=False)
        elmindex_da = ds["elmindex"].load()
        elmindex_da = elmindex_da.astype(np.int64)
    return mesh_int, elmindex_da


def load_citymesh(citymesh_path: str) -> xr.DataArray:
    with xr.open_dataset(citymesh_path) as ds_raw:
        ds = _sort_lat_ascending(ds_raw)
        citymesh_da = ds["mesh"].load()
    return citymesh_da.astype(np.int64)


def load_city_geodata(city_gpkg_path: str):
    try:
        import geopandas as gpd
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "geopandas is required for city/province statistics. "
            "Run this script with the project environment, e.g. "
            "/stu01/wumej22/Anaconda3/envs/cwrf_env/bin/python."
        ) from exc

    gdf = gpd.read_file(city_gpkg_path)
    required = {"code", "pr_adcode", "pr_name_en", "ct_name_en", "geometry"}
    missing = required.difference(gdf.columns)
    if missing:
        raise KeyError(f"Missing required columns in {city_gpkg_path}: {sorted(missing)}")
    for col in ["code", "ct_adcode", "pr_adcode", "cn_adcode"]:
        if col in gdf.columns:
            gdf[col] = gdf[col].astype(int)
    return gdf


def annual_hotwet_days(path: str, varname: str, year: int) -> xr.DataArray:
    with xr.open_dataset(path) as ds:
        if varname not in ds:
            raise KeyError(f"{varname!r} not found in {path}")
        da = ds[varname].sel(time=str(year))
        if da.sizes.get("time", 0) == 0:
            raise ValueError(f"No time records found for {year} in {path}")
        annual = da.sum(dim="time", skipna=True).load()
    return annual.astype(np.float32)


def population_by_mesh(pop_path: str, elmindex: np.ndarray, minlength: int) -> tuple[np.ndarray, xr.DataArray]:
    with xr.open_dataset(pop_path) as ds_raw:
        ds = _sort_lat_ascending(ds_raw)
        if "population" not in ds:
            raise KeyError(f"'population' not found in {pop_path}")
        pop_da = ds["population"].fillna(0.0).load()

    pop = pop_da.values.astype(np.float64, copy=False)
    if pop.shape != elmindex.shape:
        raise ValueError(
            f"Population shape {pop.shape} does not match elmindex shape {elmindex.shape} "
            f"for {pop_path}"
        )

    elm_flat = elmindex.ravel()
    pop_flat = pop.ravel()
    valid = elm_flat > 0
    mesh_pop = np.bincount(
        elm_flat[valid],
        weights=pop_flat[valid],
        minlength=minlength,
    )
    return mesh_pop, pop_da


def summarize_city_population(
    *,
    year: int,
    threshold_days: float,
    city_gdf,
    citymesh: np.ndarray,
    elmindex: np.ndarray,
    pop: np.ndarray,
    selected_meshes: np.ndarray,
    minlength: int,
) -> list[dict[str, float | int | str]]:
    if citymesh.shape != pop.shape:
        raise ValueError(
            f"City mesh shape {citymesh.shape} does not match population shape {pop.shape}"
        )

    city_flat = citymesh.ravel()
    elm_flat = elmindex.ravel()
    pop_flat = pop.ravel().astype(np.float64, copy=False)
    valid_city = city_flat > 0
    max_city_code = int(max(np.nanmax(citymesh), np.nanmax(city_gdf["code"].values)))
    city_minlength = max_city_code + 1

    total_by_city = np.bincount(
        city_flat[valid_city],
        weights=pop_flat[valid_city],
        minlength=city_minlength,
    )

    selected = np.zeros(minlength, dtype=bool)
    if selected_meshes.size:
        selected[selected_meshes] = True
    exposed = valid_city & (elm_flat > 0) & selected[elm_flat]
    exposed_by_city = np.bincount(
        city_flat[exposed],
        weights=pop_flat[exposed],
        minlength=city_minlength,
    )

    rows = []
    for city in city_gdf.itertuples(index=False):
        code = int(city.code)
        total_pop = float(total_by_city[code]) if code < total_by_city.size else 0.0
        exposed_pop = float(exposed_by_city[code]) if code < exposed_by_city.size else 0.0
        exposed_pct = exposed_pop / total_pop * 100.0 if total_pop > 0 else 0.0
        rows.append(
            {
                "year": int(year),
                "threshold_days": float(threshold_days),
                "code": code,
                "pr_adcode": int(city.pr_adcode),
                "pr_name_en": city.pr_name_en,
                "ct_name_en": city.ct_name_en,
                "total_pop": total_pop,
                "exposed_pop": exposed_pop,
                "exposed_pct": exposed_pct,
            }
        )
    return rows


def build_exposure_pop_cwrf(
    selected_meshes: np.ndarray,
    mesh_int: np.ndarray,
    mesh_pop: np.ndarray,
) -> np.ndarray:
    out = np.zeros(mesh_int.shape, dtype=np.float64)
    if selected_meshes.size == 0:
        return out

    selected = np.zeros(mesh_pop.shape[0], dtype=bool)
    selected[selected_meshes] = True
    mask = (mesh_int > 0) & selected[mesh_int]

    # In this mesh file each cropped CWRF grid cell has a unique mesh id. This
    # assignment also works if that remains true in future regenerated files.
    out[mask] = mesh_pop[mesh_int[mask]]
    return out


def build_multiyear_admin_mean(city_df: pd.DataFrame, city_gdf):
    city_mean = (
        city_df.groupby(["code", "pr_adcode", "pr_name_en", "ct_name_en"], as_index=False)
        .agg(
            total_pop=("total_pop", "mean"),
            exposed_pop=("exposed_pop", "mean"),
        )
    )
    city_mean["exposed_pct"] = np.where(
        city_mean["total_pop"] > 0,
        city_mean["exposed_pop"] / city_mean["total_pop"] * 100.0,
        0.0,
    )

    province_mean = (
        city_mean.groupby(["pr_adcode", "pr_name_en"], as_index=False)
        .agg(
            n_cities=("code", "count"),
            total_pop=("total_pop", "sum"),
            exposed_pop=("exposed_pop", "sum"),
        )
    )
    province_mean["exposed_pct"] = np.where(
        province_mean["total_pop"] > 0,
        province_mean["exposed_pop"] / province_mean["total_pop"] * 100.0,
        0.0,
    )

    city_geom_cols = [
        "code",
        "ct_adcode",
        "ct_name",
        "pr_adcode",
        "pr_name",
        "cn_adcode",
        "cn_name",
        "pr_name_en",
        "ct_name_en",
        "geometry",
    ]
    city_mean_gdf = city_mean.merge(
        city_gdf[city_geom_cols],
        on="code",
        how="left",
        suffixes=("", "_geom"),
    )
    for col in ["pr_adcode", "pr_name_en", "ct_name_en"]:
        geom_col = f"{col}_geom"
        if geom_col in city_mean_gdf.columns:
            city_mean_gdf = city_mean_gdf.drop(columns=[geom_col])

    import geopandas as gpd

    city_mean_gdf = gpd.GeoDataFrame(city_mean_gdf, geometry="geometry", crs=city_gdf.crs)
    province_geom = city_gdf.dissolve(
        by=["pr_adcode", "pr_name_en"],
        as_index=False,
    )[["pr_adcode", "pr_name_en", "geometry"]]
    province_mean_gdf = province_mean.merge(
        province_geom,
        on=["pr_adcode", "pr_name_en"],
        how="left",
    )
    province_mean_gdf = gpd.GeoDataFrame(
        province_mean_gdf,
        geometry="geometry",
        crs=city_gdf.crs,
    )
    return city_mean_gdf, province_mean_gdf


def prepare_mean_gdf_for_affected_population_plot(city_mean_gdf):
    gdf = city_mean_gdf.copy()
    gdf["up_pop"] = gdf["exposed_pop"]
    gdf["down_pop"] = 0.0
    gdf["all_pop"] = gdf["exposed_pop"]
    gdf["up_pct"] = gdf["exposed_pct"]
    gdf["down_pct"] = 0.0
    gdf["all_pct"] = gdf["exposed_pct"]
    return gdf


def plot_multiyear_mean_exposure(city_mean_gdf, figoutdir: Path) -> None:
    print("    ➠ Plotting multiyear mean HotWet exposure...")
    target = "Exposure_HotWet"
    event = "HotWet"
    checkmethod = f"{START_YEAR}-{END_YEAR}_mean"
    figoutdir_var = figoutdir / "Single"
    figoutdir_var.mkdir(parents=True, exist_ok=True)

    userlevels = [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000]
    levels_info = TYCM.Affected_Population_Cmap(event)
    citycfg = TPC.mapConfig(levs=levels_info["city"], cmap=levels_info["cmap"])

    gdf = prepare_mean_gdf_for_affected_population_plot(city_mean_gdf)
    gdf, gdf_pry = TDP.preparation_for_radial_histogram(gdf, "up", "pr_name_en")
    gdf_df = gdf[
        ["pr_name_en", "ct_name_en", "up_pop", "down_pop", "all_pop", "up_pop_K", "up_pct", "down_pct", "all_pct"]
    ].copy()
    print(gdf_df.sort_values(by="up_pct", ascending=False))
    print(gdf_pry.sort_values(by="up_pop_K", ascending=False))

    tasks = []
    savepath = figoutdir_var / f"{target}_Map_{event}-up_{checkmethod}-shpfile.{FIGFMT}"
    kargs = {"city_name_col": "ct_name_en", "city_linewidth": 0.4, "city_edgecolor": "grey"}
    tasks.append((TPAM.plot_city_stat_map_lambert, (gdf, "up_pop_K", citycfg, str(savepath), kargs)))

    label = "Affected population (Thousands, K)"
    savepath = figoutdir_var / f"{target}_Map_{event}_HColorbar-shpfile_{checkmethod}.{FIGFMT}"
    tasks.append((TPCB.plot_spatial_cbar_core_H, (citycfg, 9.0, label, str(savepath), 24, 30, 0.03, "max", "{:.0f}")))

    savepath = figoutdir_var / f"{target}_Bar_{event}-up_{checkmethod}-shpfile.{FIGFMT}"
    tasks.append(
        (
            TPBH.plot_stacked_bar_discrete_cmap,
            (
                gdf_pry,
                userlevels,
                citycfg.cmap,
                str(savepath),
                "pr_name_en",
                ["up_pop_K"],
                ["Hubei", "Hunan", "Jiangsu", "Anhui", "Jiangxi", "Henan", "Fujian", "Shaanxi", "Guizhou", "Zhejiang", "Chongqing", "Sichuan"],
            ),
        )
    )

    savepath = figoutdir_var / f"{target}_RadialHistogram_{event}-up_{checkmethod}.{FIGFMT}"
    kargs = {
        "stack_on": False,
        "color_by_colname": "up_pop_K",
        "color_cmap": citycfg.cmap,
        "cmap": citycfg.cmap,
        "levels": userlevels,
        "sort_by_Total": True,
        "sort_ascending": False,
        "inner_circle_radius": 5000,
        "legend_on": False,
        "ylims": [0, 3100],
        "offset_pry_text": -500,
        "offset_inner": -100,
        "unit_sec_text": "K",
        "primary_cats": ["Hubei", "Hunan", "Jiangsu", "Anhui", "Jiangxi", "Henan", "Fujian", "Shaanxi", "Guizhou", "Zhejiang", "Chongqing", "Sichuan"],
        "blank_length": 2,
        "circle_on": False,
        "circle_linewidth": 1,
        "radii": [0, 1000, 2000, 3000],
    }
    tasks.append((TPRH.radial_histogram, (gdf_df, "pr_name_en", "ct_name_en", ["up_pop_K"], str(savepath), kargs)))

    ntasks = len(tasks)
    with Parallel(n_jobs=8, backend="loky", pre_dispatch="2*n_jobs", return_as="generator") as parallel:
        gen = parallel(delayed(func)(*args) for func, args in tasks)
        _ = [
            p
            for p in tqdm(
                gen,
                total=ntasks,
                desc="    ➠ Parallel",
                unit="task",
                dynamic_ncols=True,
            )
        ]
    print("    All parallel plots done.")


def main() -> None:
    years = np.arange(START_YEAR, END_YEAR + 1, dtype=np.int32)
    outdir = Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading CWRF-LandScan mesh mapping...")
    mesh_int, elmindex_da = load_mesh(str(MESH_PATH), BUFFERZONE)
    elmindex = elmindex_da.values.astype(np.int64, copy=False)
    max_mesh_id = int(max(np.nanmax(mesh_int), np.nanmax(elmindex)))
    minlength = max_mesh_id + 1

    if COMPUTE_ADMIN_STATS:
        print("Loading city mesh and administrative boundaries...")
        citymesh_da = load_citymesh(str(CITYMESH_PATH))
        citymesh = citymesh_da.values.astype(np.int64, copy=False)
        city_gdf = load_city_geodata(str(CITY_GPKG_PATH))
        if citymesh.shape != elmindex.shape:
            raise ValueError(
                f"City mesh shape {citymesh.shape} does not match elmindex shape {elmindex.shape}"
            )
    else:
        citymesh = None
        city_gdf = None

    ny, nx = mesh_int.shape
    lake_days_all = np.zeros((years.size, ny, nx), dtype=np.float32)
    nolake_days_all = np.zeros((years.size, ny, nx), dtype=np.float32)
    diff_days_all = np.zeros((years.size, ny, nx), dtype=np.float32)
    exposure_mask_all = np.zeros((years.size, ny, nx), dtype=np.int8)
    exposure_pop_cwrf_all = np.zeros((years.size, ny, nx), dtype=np.float64)

    rows = []
    city_rows = []

    for yi, year in enumerate(years):
        print(f"Processing {year}...")
        lake_days = annual_hotwet_days(str(LAKE_EVENT_PATH), EVENT_VAR, int(year)).values
        nolake_days = annual_hotwet_days(str(NOLAKE_EVENT_PATH), EVENT_VAR, int(year)).values
        if lake_days.shape != mesh_int.shape or nolake_days.shape != mesh_int.shape:
            raise ValueError(
                f"Event grid shape mismatch for {year}: "
                f"Lake={lake_days.shape}, NoLake={nolake_days.shape}, mesh={mesh_int.shape}"
            )

        diff_days = lake_days - nolake_days
        exposure_mask = diff_days > THRESHOLD_DAYS
        selected_meshes = np.unique(mesh_int[exposure_mask & (mesh_int > 0)])

        pop_path = POPULATION_TEMPLATE.format(year=int(year))
        if not os.path.exists(pop_path):
            raise FileNotFoundError(f"Annual LandScan file not found: {pop_path}")
        mesh_pop, pop_da = population_by_mesh(pop_path, elmindex, minlength)
        exposure_pop_cwrf = build_exposure_pop_cwrf(selected_meshes, mesh_int, mesh_pop)

        if COMPUTE_ADMIN_STATS:
            city_rows.extend(
                summarize_city_population(
                    year=int(year),
                    threshold_days=THRESHOLD_DAYS,
                    city_gdf=city_gdf,
                    citymesh=citymesh,
                    elmindex=elmindex,
                    pop=pop_da.values,
                    selected_meshes=selected_meshes,
                    minlength=minlength,
                )
            )

        total_exposed = float(mesh_pop[selected_meshes].sum()) if selected_meshes.size else 0.0
        lake_days_all[yi] = lake_days
        nolake_days_all[yi] = nolake_days
        diff_days_all[yi] = diff_days
        exposure_mask_all[yi] = exposure_mask.astype(np.int8)
        exposure_pop_cwrf_all[yi] = exposure_pop_cwrf

        rows.append(
            {
                "year": int(year),
                "threshold_days": float(THRESHOLD_DAYS),
                "n_exposed_cwrf_cells": int(np.count_nonzero(exposure_mask)),
                "n_exposed_meshes": int(selected_meshes.size),
                "exposed_population": total_exposed,
                "population_file": pop_path,
            }
        )

    coords = {
        "year": years,
        "y": np.arange(ny),
        "x": np.arange(nx),
    }
    ds_out = xr.Dataset(
        data_vars={
            "lake_hotwet_days": (("year", "y", "x"), lake_days_all),
            "nolake_hotwet_days": (("year", "y", "x"), nolake_days_all),
            "lake_minus_nolake_hotwet_days": (("year", "y", "x"), diff_days_all),
            "exposure_mask": (("year", "y", "x"), exposure_mask_all),
            "exposure_population_cwrf": (("year", "y", "x"), exposure_pop_cwrf_all),
            "total_exposed_population": (
                ("year",),
                np.array([row["exposed_population"] for row in rows], dtype=np.float64),
            ),
            "n_exposed_cwrf_cells": (
                ("year",),
                np.array([row["n_exposed_cwrf_cells"] for row in rows], dtype=np.int32),
            ),
            "n_exposed_meshes": (
                ("year",),
                np.array([row["n_exposed_meshes"] for row in rows], dtype=np.int32),
            ),
        },
        coords=coords,
        attrs={
            "description": (
                "Annual population exposure to lake-induced Hot-Wet event increases. "
                "Exposure is defined where Lake annual Hot-Wet days minus NoLake "
                "annual Hot-Wet days exceeds threshold_days."
            ),
            "event_var": EVENT_VAR,
            "threshold_days": float(THRESHOLD_DAYS),
            "lake_event_path": str(LAKE_EVENT_PATH),
            "nolake_event_path": str(NOLAKE_EVENT_PATH),
            "population_template": POPULATION_TEMPLATE,
            "mesh_path": str(MESH_PATH),
            "bufferzone": int(BUFFERZONE),
            "population_aggregation": (
                "LandScan pixels are summed by precomputed elmindex mesh id; "
                "no spatial averaging or interpolation is applied."
            ),
        },
    )
    ds_out["exposure_mask"].attrs.update(
        long_name="Lake-induced Hot-Wet exposure mask",
        flag_values=np.array([0, 1], dtype=np.int8),
        flag_meanings="not_exposed exposed",
    )
    ds_out["exposure_population_cwrf"].attrs.update(
        long_name="Annual LandScan population in exposed CWRF mesh cells",
        units="persons",
        note="Summing this field over y,x gives total_exposed_population.",
    )
    ds_out["total_exposed_population"].attrs.update(
        long_name="Total annual exposed population",
        units="persons",
    )

    nc_path = outdir / f"{OUTPUT_PREFIX}.nc"
    csv_path = outdir / f"{OUTPUT_PREFIX}.csv"
    ds_out.to_netcdf(nc_path)
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"Saved NetCDF: {nc_path}")
    print(f"Saved CSV:    {csv_path}")

    if COMPUTE_ADMIN_STATS:
        city_df = pd.DataFrame(city_rows)
        province_df = (
            city_df.groupby(["year", "threshold_days", "pr_adcode", "pr_name_en"], as_index=False)
            .agg(
                n_cities=("code", "count"),
                total_pop=("total_pop", "sum"),
                exposed_pop=("exposed_pop", "sum"),
            )
        )
        province_df["exposed_pct"] = np.where(
            province_df["total_pop"] > 0,
            province_df["exposed_pop"] / province_df["total_pop"] * 100.0,
            0.0,
        )

        city_csv_path = outdir / f"{OUTPUT_PREFIX}_by_city.csv"
        province_csv_path = outdir / f"{OUTPUT_PREFIX}_by_province.csv"
        city_df.to_csv(city_csv_path, index=False)
        province_df.to_csv(province_csv_path, index=False)
        print(f"Saved city CSV:     {city_csv_path}")
        print(f"Saved province CSV: {province_csv_path}")

        if WRITE_ADMIN_GPKG:
            import geopandas as gpd

            city_geom_cols = [
                "code",
                "ct_adcode",
                "ct_name",
                "pr_adcode",
                "pr_name",
                "cn_adcode",
                "cn_name",
                "pr_name_en",
                "ct_name_en",
                "geometry",
            ]
            city_year_gdf = city_df.merge(
                city_gdf[city_geom_cols],
                on="code",
                how="left",
                suffixes=("", "_geom"),
            )
            for col in ["pr_adcode", "pr_name_en", "ct_name_en"]:
                geom_col = f"{col}_geom"
                if geom_col in city_year_gdf.columns:
                    city_year_gdf = city_year_gdf.drop(columns=[geom_col])
            city_year_gdf = gpd.GeoDataFrame(
                city_year_gdf,
                geometry="geometry",
                crs=city_gdf.crs,
            )

            province_geom = city_gdf.dissolve(
                by=["pr_adcode", "pr_name_en"],
                as_index=False,
            )[["pr_adcode", "pr_name_en", "geometry"]]
            province_year_gdf = province_df.merge(
                province_geom,
                on=["pr_adcode", "pr_name_en"],
                how="left",
            )
            province_year_gdf = gpd.GeoDataFrame(
                province_year_gdf,
                geometry="geometry",
                crs=city_gdf.crs,
            )

            city_gpkg_path = outdir / f"{OUTPUT_PREFIX}_by_city.gpkg"
            province_gpkg_path = outdir / f"{OUTPUT_PREFIX}_by_province.gpkg"
            city_year_gdf.to_file(city_gpkg_path, encoding="utf-8", driver="GPKG")
            province_year_gdf.to_file(province_gpkg_path, encoding="utf-8", driver="GPKG")
            print(f"Saved city GPKG:     {city_gpkg_path}")
            print(f"Saved province GPKG: {province_gpkg_path}")

        if COMPUTE_MULTIYEAR_MEAN:
            city_mean_gdf, province_mean_gdf = build_multiyear_admin_mean(city_df, city_gdf)

            city_mean_csv_path = outdir / f"{OUTPUT_PREFIX}_{START_YEAR}-{END_YEAR}_mean_by_city.csv"
            province_mean_csv_path = outdir / f"{OUTPUT_PREFIX}_{START_YEAR}-{END_YEAR}_mean_by_province.csv"
            city_mean_gpkg_path = outdir / f"{OUTPUT_PREFIX}_{START_YEAR}-{END_YEAR}_mean_by_city.gpkg"
            province_mean_gpkg_path = outdir / f"{OUTPUT_PREFIX}_{START_YEAR}-{END_YEAR}_mean_by_province.gpkg"

            pd.DataFrame(city_mean_gdf.drop(columns="geometry")).to_csv(city_mean_csv_path, index=False)
            pd.DataFrame(province_mean_gdf.drop(columns="geometry")).to_csv(province_mean_csv_path, index=False)
            city_mean_gdf.to_file(city_mean_gpkg_path, encoding="utf-8", driver="GPKG")
            province_mean_gdf.to_file(province_mean_gpkg_path, encoding="utf-8", driver="GPKG")

            print(f"Saved multiyear mean city CSV:     {city_mean_csv_path}")
            print(f"Saved multiyear mean province CSV: {province_mean_csv_path}")
            print(f"Saved multiyear mean city GPKG:     {city_mean_gpkg_path}")
            print(f"Saved multiyear mean province GPKG: {province_mean_gpkg_path}")

            if PLOT_MULTYEAR_MEAN:
                plot_multiyear_mean_exposure(city_mean_gdf, FIGOUTDIR)


if __name__ == "__main__":
    main()
