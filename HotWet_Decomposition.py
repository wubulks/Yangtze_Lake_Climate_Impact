#!/stu01/wumej22/Anaconda3/envs/cwrf_env/bin/python
import gc
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cmaps
import numpy as np
import pandas as pd
import xarray as xr

import ToolBoxes.Utils as TU
import ToolBoxes.Tool_InputOutput as TIO
import ToolBoxes.Tool_PlotConfig as TPC
import ToolBoxes.Tool_PlotAreaMap as TPAM
import ToolBoxes.Tool_PlotColorBar as TPCB
import ToolBoxes.Tool_SignificanceTest as TST

warnings.filterwarnings("ignore", category=RuntimeWarning)
TPC.apply_global_plot_style()

FIGFMT = TPC.FIGFMT
DPI = TPC.DPI_medium


# ========== Option ==========
StartTime = "2000-01-01"
EndTime = "2024-12-31"
DecompSeasons = ["ALL", "MAM", "JJA", "SON", "DJF"]
ExtremeRefCase = "NoLake"
HotFlagVar = "Hot_Flag"
WetFlagVar = "Wet_Flag"
MinHWEvents = 5
Flag_Plot = True
Flag_Region_Significance = True
BootstrapSamples = 1000
BootstrapCI = 0.95
BootstrapRandomSeed = 20260525
SignificanceAlpha = 0.05

# ========== Config ==========
BufferZone = 15
dx, dy = 6000.0, 6000.0

# ========== Path ==========
DataOutDir = "/hydata01/wumej22/Experiment/Process/Yangtze_C_6km_r1/Cases"
FigOutDir = "/hydata01/wumej22/Experiment/Process/Yangtze_C_6km_r1/Figures"
cwrfinp_path = "/hydata01/wumej22/Experiment/Process/Yangtze_C_6km_r1/BaseData/wrfinput_d01.2000"


@dataclass(frozen=True)
class Config:
    start_time: str
    end_time: str
    seasons: List[str]
    extreme_ref_case: str
    hot_flag_var: str
    wet_flag_var: str
    min_hw_events: int
    buffer_zone: int
    dx: float
    dy: float
    data_out_dir: str
    fig_out_dir: str
    cwrfinp_path: str


def season_select(da: xr.DataArray, season: str) -> xr.DataArray:
    if season == "ALL":
        return da
    return da.sel(time=da.time.dt.season == season)


def build_lake_info(cfg: Config) -> tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    lon2d, lat2d, mapfac_mx, mapfac_my, scwater, lakedp = TIO.read_CWRF_Info(
        cfg.cwrfinp_path,
        bufferzone=cfg.buffer_zone,
    )
    area = TU.get_cwrf_grid_area(
        DX=cfg.dx,
        DY=cfg.dy,
        MAPFAC_MX=mapfac_mx,
        MAPFAC_MY=mapfac_my,
    )
    mask_path = f"{cfg.data_out_dir}/lake_mask_with_dist.nc"
    lkinfos = TU.get_lake_area_mask(
        scwater,
        lakedp,
        DX=cfg.dx,
        DY=cfg.dy,
        MAPFAC_MX=mapfac_mx,
        MAPFAC_MY=mapfac_my,
        nc_path=mask_path,
    )
    lkinfos["area"] = area
    return lon2d, lat2d, lkinfos


def read_case_flags(case: str, cfg: Config) -> xr.Dataset:
    filepath = (
        f"{cfg.data_out_dir}/ExtremeAnalysis/"
        f"Extreme_Events_{case}_ref_{cfg.extreme_ref_case}_identified.nc"
    )
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Existing extreme-event file not found: {filepath}")

    print(f"Read existing hot/wet flags: {filepath}")
    with xr.open_dataset(filepath) as ds:
        for varname in [cfg.hot_flag_var, cfg.wet_flag_var]:
            if varname not in ds:
                raise KeyError(f"{varname} not found in {filepath}")
        out = xr.Dataset({
            "H": (ds[cfg.hot_flag_var].sel(time=slice(cfg.start_time, cfg.end_time)) == 1).load(),
            "W": (ds[cfg.wet_flag_var].sel(time=slice(cfg.start_time, cfg.end_time)) == 1).load(),
        })
    out.attrs.update({
        "case": case,
        "source_file": filepath,
        "hot_flag_var": cfg.hot_flag_var,
        "wet_flag_var": cfg.wet_flag_var,
    })
    return out


def event_counts(flags: xr.Dataset, season: str) -> xr.Dataset:
    flags = season_select(flags, season)
    valid = np.isfinite(flags["H"]) & np.isfinite(flags["W"])
    h = flags["H"] & valid
    w = flags["W"] & valid
    hw = h & w
    return xr.Dataset({
        "N": valid.sum("time").astype("int32"),
        "n_H": h.sum("time").astype("int32"),
        "n_W": w.sum("time").astype("int32"),
        "n_HW": hw.sum("time").astype("int32"),
    })


def probabilities_from_counts(counts: xr.Dataset, prefix: str) -> xr.Dataset:
    n = counts["N"].where(counts["N"] > 0)
    p_h = (counts["n_H"] / n).rename(f"P_{prefix}_H")
    p_w = (counts["n_W"] / n).rename(f"P_{prefix}_W")
    p_hw = (counts["n_HW"] / n).rename(f"P_{prefix}_HW")
    r_hw = (p_hw / (p_h * p_w)).rename(f"R_{prefix}")
    return xr.merge([
        counts["N"].rename(f"N_{prefix}"),
        counts["n_H"].rename(f"n_{prefix}_H"),
        counts["n_W"].rename(f"n_{prefix}_W"),
        counts["n_HW"].rename(f"n_{prefix}_HW"),
        p_h,
        p_w,
        p_hw,
        r_hw,
    ])


def log_ratio(a: xr.DataArray, b: xr.DataArray) -> xr.DataArray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(a) - np.log(b)


def spatial_decomposition(lake_counts: xr.Dataset, nolake_counts: xr.Dataset, cfg: Config) -> xr.Dataset:
    lake = probabilities_from_counts(lake_counts, "Lake")
    nolake = probabilities_from_counts(nolake_counts, "NoLake")

    c_h = log_ratio(lake["P_Lake_H"], nolake["P_NoLake_H"]).rename("C_H")
    c_w = log_ratio(lake["P_Lake_W"], nolake["P_NoLake_W"]).rename("C_W")
    c_r = log_ratio(lake["R_Lake"], nolake["R_NoLake"]).rename("C_R")
    d_hw = log_ratio(lake["P_Lake_HW"], nolake["P_NoLake_HW"]).rename("Delta_lnP_HW")
    residual = (d_hw - c_h - c_w - c_r).rename("decomposition_residual")

    low_freq_mask = (
        (lake_counts["n_HW"] < cfg.min_hw_events)
        | (nolake_counts["n_HW"] < cfg.min_hw_events)
        | (lake_counts["N"] <= 0)
        | (nolake_counts["N"] <= 0)
    ).rename("low_frequency_mask")
    finite_mask = np.isfinite(c_h) & np.isfinite(c_w) & np.isfinite(c_r) & np.isfinite(d_hw)
    valid_mask = (~low_freq_mask) & finite_mask

    terms = [c_h, c_w, c_r, d_hw, residual]
    terms = [term.where(valid_mask) for term in terms]
    out = xr.merge([lake, nolake, low_freq_mask.astype("int8"), valid_mask.astype("int8").rename("valid_decomposition_mask"), *terms])
    for name in ["C_H", "C_W", "C_R", "Delta_lnP_HW"]:
        out[name].attrs["units"] = "1"
    out["C_H"].attrs["long_name"] = "hot-event frequency contribution"
    out["C_W"].attrs["long_name"] = "wet-event frequency contribution"
    out["C_R"].attrs["long_name"] = "hot-wet dependence contribution"
    out["Delta_lnP_HW"].attrs["long_name"] = "total change in log compound hot-wet probability"
    out.attrs.update({
        "method": "Delta ln P(HW) = Delta ln P(H) + Delta ln P(W) + Delta ln R",
        "R_definition": "R = P(HW) / (P(H) * P(W))",
        "spatial_mask": f"masked where either experiment has n_HW < {cfg.min_hw_events}",
        "significance_test": "not applied",
    })
    return out


def area_weighted_prob(counts: xr.Dataset, region_mask: np.ndarray, area: np.ndarray) -> Dict[str, float]:
    valid_grid = region_mask & np.isfinite(area) & (counts["N"].values > 0)
    if not np.any(valid_grid):
        return {"P_H": np.nan, "P_W": np.nan, "P_HW": np.nan, "R": np.nan, "N_eff": np.nan, "n_HW_area": np.nan}

    weights = area[valid_grid]
    n = counts["N"].values[valid_grid]
    denom = np.sum(weights * n)
    p_h = np.sum(weights * counts["n_H"].values[valid_grid]) / denom
    p_w = np.sum(weights * counts["n_W"].values[valid_grid]) / denom
    p_hw = np.sum(weights * counts["n_HW"].values[valid_grid]) / denom
    r = p_hw / (p_h * p_w) if p_h > 0 and p_w > 0 else np.nan
    return {
        "P_H": float(p_h),
        "P_W": float(p_w),
        "P_HW": float(p_hw),
        "R": float(r),
        "N_eff": float(denom / np.sum(weights)),
        "n_HW_area": float(np.sum(weights * counts["n_HW"].values[valid_grid]) / np.sum(weights)),
    }


def decompose_probabilities(lake: Dict[str, float], nolake: Dict[str, float]) -> Dict[str, float]:
    c_h = np.log(lake["P_H"]) - np.log(nolake["P_H"])
    c_w = np.log(lake["P_W"]) - np.log(nolake["P_W"])
    c_r = np.log(lake["R"]) - np.log(nolake["R"])
    d_hw = np.log(lake["P_HW"]) - np.log(nolake["P_HW"])
    denom_abs = np.abs(c_h) + np.abs(c_w) + np.abs(c_r)
    return {
        "Delta_lnP_HW": d_hw,
        "C_H": c_h,
        "C_W": c_w,
        "C_R": c_r,
        "residual": d_hw - c_h - c_w - c_r,
        "F_H_abs": np.abs(c_h) / denom_abs if denom_abs > 0 else np.nan,
        "F_W_abs": np.abs(c_w) / denom_abs if denom_abs > 0 else np.nan,
        "F_R_abs": np.abs(c_r) / denom_abs if denom_abs > 0 else np.nan,
        "F_H_signed": c_h / d_hw if d_hw != 0 else np.nan,
        "F_W_signed": c_w / d_hw if d_hw != 0 else np.nan,
        "F_R_signed": c_r / d_hw if d_hw != 0 else np.nan,
    }


def area_decomposition_for_region(
    lake_counts: xr.Dataset,
    nolake_counts: xr.Dataset,
    region_name: str,
    region_mask: np.ndarray,
    area: np.ndarray,
    season: str,
) -> Dict[str, float | str]:
    lake = area_weighted_prob(lake_counts, region_mask, area)
    nolake = area_weighted_prob(nolake_counts, region_mask, area)
    terms = decompose_probabilities(lake, nolake)
    return {
        "season": season,
        "region": region_name,
        "P_Lake_H": lake["P_H"],
        "P_Lake_W": lake["P_W"],
        "P_Lake_HW": lake["P_HW"],
        "R_Lake": lake["R"],
        "P_NoLake_H": nolake["P_H"],
        "P_NoLake_W": nolake["P_W"],
        "P_NoLake_HW": nolake["P_HW"],
        "R_NoLake": nolake["R"],
        **terms,
        "n_HW_Lake_area_mean": lake["n_HW_area"],
        "n_HW_NoLake_area_mean": nolake["n_HW_area"],
        "N_eff_Lake": lake["N_eff"],
        "N_eff_NoLake": nolake["N_eff"],
    }


def yearly_region_decomposition_terms(
    lake_flags: xr.Dataset,
    nolake_flags: xr.Dataset,
    season: str,
    region_mask: np.ndarray,
    area: np.ndarray,
) -> pd.DataFrame:
    time = season_select(lake_flags["H"], season)["time"]
    years = np.unique(time.dt.year.values)
    rows = []
    for year in years:
        lake_year = lake_flags.sel(time=lake_flags.time.dt.year == year)
        nolake_year = nolake_flags.sel(time=nolake_flags.time.dt.year == year)
        lake_counts = event_counts(lake_year, season)
        nolake_counts = event_counts(nolake_year, season)
        lake = area_weighted_prob(lake_counts, region_mask, area)
        nolake = area_weighted_prob(nolake_counts, region_mask, area)
        terms = decompose_probabilities(lake, nolake)
        rows.append({"year": int(year), **terms})
    return pd.DataFrame(rows)


def add_region_significance(
    row: Dict[str, float | str],
    yearly_terms: pd.DataFrame,
    seed_offset: int,
) -> Dict[str, float | str]:
    out = dict(row)
    zero = np.zeros(len(yearly_terms), dtype=float)
    for idx, term in enumerate(["C_H", "C_W", "C_R", "Delta_lnP_HW"]):
        values = yearly_terms[term].to_numpy(dtype=float)
        obs, pval, low, high = TST.paired_bootstrap(
            values,
            zero,
            n_sample=BootstrapSamples,
            ci=BootstrapCI,
            alternative="two-sided",
            center_null=True,
            random_state=BootstrapRandomSeed + seed_offset * 10 + idx,
        )
        out[f"{term}_yearmean"] = obs
        out[f"{term}_p"] = pval
        out[f"{term}_ci_low"] = low
        out[f"{term}_ci_high"] = high
        out[f"{term}_sig"] = bool(np.isfinite(pval) and pval < SignificanceAlpha)
    out["significance_method"] = "paired_bootstrap_by_year"
    out["bootstrap_samples"] = BootstrapSamples
    out["bootstrap_ci"] = BootstrapCI
    return out


def build_regions(lkinfos: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    ocean = lkinfos["ocean"] == 1
    land = ~ocean
    regions = {
        "land": land,
        "lake_grid": (lkinfos["all"] == 1) & land,
    }
    if "dist_m_all" in lkinfos:
        regions["near_lake_100km"] = (lkinfos["dist_m_all"] <= 100000.0) & land
        regions["far_lake_gt100km"] = (lkinfos["dist_m_all"] > 100000.0) & land
    return regions


def symmetric_levels(data: np.ndarray, nlev: int = 11) -> np.ndarray:
    vmax = np.nanpercentile(np.abs(data), 98)
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    return np.linspace(-vmax, vmax, nlev)


def plot_spatial_decomposition(ds: xr.Dataset, lon2d: np.ndarray, lat2d: np.ndarray, lkinfos: Dict[str, np.ndarray], out_fig_dir: str, token: str) -> None:
    vars_plot = ["C_H", "C_W", "C_R", "Delta_lnP_HW"]
    titles = {
        "C_H": "Hot frequency contribution",
        "C_W": "Wet frequency contribution",
        "C_R": "Hot-wet dependence contribution",
        "Delta_lnP_HW": "Total log-probability change",
    }
    labels = {
        "C_H": r"$C_H$ contribution to $\Delta \ln P(HW)$",
        "C_W": r"$C_W$ contribution to $\Delta \ln P(HW)$",
        "C_R": r"$C_R$ contribution to $\Delta \ln P(HW)$",
        "Delta_lnP_HW": r"$\Delta \ln P(HW)$",
    }
    figoutdir_var = f"{out_fig_dir}/Single"
    os.makedirs(figoutdir_var, exist_ok=True)

    ocean = lkinfos["ocean"] == 1
    for varname in vars_plot:
        data = ds[varname].where(~xr.DataArray(ocean, dims=("y", "x"))).values
        levels = symmetric_levels(data)
        mapcfg = TPC.mapConfig(levs=[levels, 5], cmap=cmaps.MPL_PuOr.reversed())
        target = f"HotWetDecomposition_{varname}"
        map_path = f"{figoutdir_var}/{target}_Lake_minus_NoLake_{token}.{FIGFMT}"
        cbar_path = f"{figoutdir_var}/{target}_Lake_minus_NoLake_{token}_VColorbar.{FIGFMT}"
        TPAM.plot_categorical_map(
            data.copy(),
            target,
            lon2d,
            lat2d,
            map_path,
            None,
            titles[varname],
            lkinfos,
            mapcfg,
            None,
        )
        TPCB.plot_spatial_cbar_core_V(
            mapcfg,
            6.0,
            labels[varname],
            cbar_path,
            14,
            18,
            0.04,
            "both",
            "{:5.2f}",
        )
        print(f"Saved spatial decomposition map: {map_path}")


def save_outputs_for_season(
    season: str,
    lake_flags: xr.Dataset,
    nolake_flags: xr.Dataset,
    cfg: Config,
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    lkinfos: Dict[str, np.ndarray],
    out_data_dir: str,
    out_fig_dir: str,
) -> pd.DataFrame:
    print(f"\nStart hot-wet decomposition for season: {season}")
    lake_counts = event_counts(lake_flags, season)
    nolake_counts = event_counts(nolake_flags, season)
    spatial = spatial_decomposition(lake_counts, nolake_counts, cfg)
    spatial.attrs.update({
        "season": season,
        "start_time": cfg.start_time,
        "end_time": cfg.end_time,
        "hot_flag_var": cfg.hot_flag_var,
        "wet_flag_var": cfg.wet_flag_var,
        "extreme_ref_case": cfg.extreme_ref_case,
    })

    token = f"{season}_ref{cfg.extreme_ref_case}_{cfg.start_time.replace('-', '')}_{cfg.end_time.replace('-', '')}"
    nc_path = f"{out_data_dir}/HotWet_Decomposition_Lake-NoLake_{token}.nc"
    spatial.to_netcdf(nc_path, format="NETCDF4")
    print(f"Saved spatial decomposition data: {nc_path}")
    if Flag_Plot:
        plot_spatial_decomposition(spatial, lon2d, lat2d, lkinfos, out_fig_dir, token)

    regions = build_regions(lkinfos)
    rows = []
    for idx, (region_name, region_mask) in enumerate(regions.items()):
        row = area_decomposition_for_region(
            lake_counts=lake_counts,
            nolake_counts=nolake_counts,
            region_name=region_name,
            region_mask=region_mask,
            area=lkinfos["area"],
            season=season,
        )
        if Flag_Region_Significance:
            yearly_terms = yearly_region_decomposition_terms(
                lake_flags=lake_flags,
                nolake_flags=nolake_flags,
                season=season,
                region_mask=region_mask,
                area=lkinfos["area"],
            )
            row = add_region_significance(row, yearly_terms, seed_offset=idx)
        rows.append(row)

    summary = pd.DataFrame(rows)
    return summary


def main() -> None:
    cfg = Config(
        start_time=StartTime,
        end_time=EndTime,
        seasons=DecompSeasons,
        extreme_ref_case=ExtremeRefCase,
        hot_flag_var=HotFlagVar,
        wet_flag_var=WetFlagVar,
        min_hw_events=MinHWEvents,
        buffer_zone=BufferZone,
        dx=dx,
        dy=dy,
        data_out_dir=DataOutDir,
        fig_out_dir=FigOutDir,
        cwrfinp_path=cwrfinp_path,
    )
    invalid_seasons = [season for season in cfg.seasons if season not in ["ALL", "MAM", "JJA", "SON", "DJF"]]
    if invalid_seasons:
        raise ValueError(f"Invalid DecompSeasons: {invalid_seasons}")

    out_data_dir = f"{cfg.data_out_dir}/HotWetDecomposition"
    out_fig_dir = f"{cfg.fig_out_dir}/HotWetDecomposition"
    os.makedirs(out_data_dir, exist_ok=True)
    os.makedirs(out_fig_dir, exist_ok=True)

    lon2d, lat2d, lkinfos = build_lake_info(cfg)
    lake_flags = read_case_flags("Lake", cfg)
    nolake_flags = read_case_flags("NoLake", cfg)

    summary_dfs = []
    for season in cfg.seasons:
        summary_dfs.append(
            save_outputs_for_season(
                season=season,
                lake_flags=lake_flags,
                nolake_flags=nolake_flags,
                cfg=cfg,
                lon2d=lon2d,
                lat2d=lat2d,
                lkinfos=lkinfos,
                out_data_dir=out_data_dir,
                out_fig_dir=out_fig_dir,
            )
        )

    summary = pd.concat(summary_dfs, ignore_index=True)
    csv_path = (
        f"{out_data_dir}/HotWet_Decomposition_RegionSummary_Lake-NoLake_"
        f"ref{cfg.extreme_ref_case}_{cfg.start_time.replace('-', '')}_{cfg.end_time.replace('-', '')}.csv"
    )
    summary.to_csv(csv_path, index=False, na_rep="NaN")
    print(f"Saved region decomposition summary: {csv_path}")


if __name__ == "__main__":
    main()
