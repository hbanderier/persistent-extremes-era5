import datetime

import colormaps
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import xarray as xr
from jetutils.anyspell import get_persistent_jet_spells, subset_around_onset, extend_spells, make_daily
from jetutils.clustering import Experiment, RAW_REALSPACE
from jetutils.data import DataHandler, open_da
from jetutils.definitions import (
    DATADIR,
    YEARS,
    compute,
    get_region,
    infer_direction,
    polars_to_xarray,
    xarray_to_polars,
)
from jetutils.jet_finding import JetFindingExperiment, haversine_from_dl, average_jet_categories, spells_from_cross, spells_from_cross_catd, iterate_over_year_maybe_member, gather_normal_da_jets, track_jets
from jetutils.plots import gather_normal_da_jets_wrapper, interp_jets_to_zero_one
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from pathlib import Path
from tqdm import tqdm
import polars.selectors as cs

summer_filter = (
    pl.datetime_range(
        start=pl.datetime(1959, 1, 1),
        end=pl.datetime(2023, 1, 1),
        closed="left",
        interval="6h",
        eager=True,
        time_unit="ms",
    )
    .rename("time")
    .to_frame()
    .filter(pl.col("time").dt.month().is_in([6, 7, 8, 9]))
)
summer = summer_filter["time"]
summer_daily = summer.filter(summer.dt.hour() == 0)

dh = DataHandler("/storage/workspaces/giub_meteo_impacts/ci01/ERA5/plev/high_wind/6H/results/7")
exp = JetFindingExperiment(dh)
ds = exp.ds
all_jets_one_df = exp.find_jets(force=False, alignment_thresh=0.6, base_s_thresh=0.55, int_thresh_factor=0.35, hole_size=10)
all_jets_one_df = exp.categorize_jets(None, ["s", "theta"], force=False, n_init=5, init_params="k-means++", mode="week").cast({"time": pl.Datetime("ms")})

props_uncat = exp.props_as_df(False).cast({"time": pl.Datetime("ms")})
props_as_df = average_jet_categories(props_uncat, polar_cutoff=0.5)

props_summer = summer_filter.join(props_as_df, on="time")
phat_filter = (pl.col("is_polar") < 0.5) | ((pl.col("is_polar") > 0.5) & (pl.col("int") > 5e8))

phat_jets = all_jets_one_df.filter(phat_filter)
phat_jets_catd = phat_jets.with_columns(**{"jet ID": (pl.col("is_polar") > 0.5).cast(pl.UInt32())})
phat_props = props_uncat.filter(phat_filter)
phat_props_catd = average_jet_categories(phat_props, polar_cutoff=0.5)

# def create_jet_relative_clim(jets, path, da, suffix=""):
#     jets = jets.with_columns(pl.col("time").dt.round("1d"))
#     jets = jets.with_columns(jets.group_by("time", maintain_order=True).agg(pl.col("jet ID").rle_id())["jet ID"].explode())
#     indexer = iterate_over_year_maybe_member(jets, da)
#     to_average = []
#     for idx1, idx2 in tqdm(indexer, total=len(YEARS)):
#         jets_ = jets.filter(*idx1)
#         da_ = da.sel(**idx2)
#         try:
#             jets_with_interp = gather_normal_da_jets(jets_, da_, half_length=20)
#         except (KeyError, ValueError):
#             break
#         varname = da_.name + "_interp"
#         jets_with_interp = interp_jets_to_zero_one(jets_with_interp, [varname, "is_polar"], n_interp=30)
#         jets_with_interp = jets_with_interp.group_by("time", pl.col("is_polar") > 0.5, "norm_index", "n", maintain_order=True).agg(pl.col(varname).mean() )
#         to_average.append(jets_with_interp)
#     to_average = pl.concat(to_average)
#     clim = to_average.group_by(pl.col("time").dt.ordinal_day().alias("dayofyear"), "is_polar", "norm_index", "n").agg(pl.col(varname).mean()).sort("dayofyear", "is_polar", "norm_index", "n")
#     clim_ds = polars_to_xarray(clim, ["dayofyear", "is_polar", "n", "norm_index"])
#     clim_ds.to_netcdf(path.joinpath(f"{da.name}{suffix}_relative_clim.nc"))
    

# args = ["all", None, *get_region(exp.ds), "all"]
# da_T = open_da("ERA5", "plev", "t300", "dailymean", *args)
# da_T = compute(da_T)
# create_jet_relative_clim(phat_jets, exp.path, da_T, suffix="_phat")
# del da_T
# da_T = open_da("ERA5", "surf", "t2m", "dailymean", *args)
# da_T = compute(da_T)
# create_jet_relative_clim(phat_jets, exp.path, da_T, suffix="_phat")
# del da_T
# da_tp = open_da("ERA5", "surf", "tp", "dailysum", *args)
# da_tp = compute(da_tp)
# create_jet_relative_clim(phat_jets, exp.path, da_tp, suffix="_phat")
# del da_tp
# da_apvs = open_da("ERA5", "thetalev", "apvs", "dailyany", *args)
# da_apvs = compute(da_apvs)
# create_jet_relative_clim(phat_jets, exp.path, da_apvs, suffix="_phat")
# del da_apvs
# da_cpvs = open_da("ERA5", "thetalev", "cpvs", "dailyany", *args)
# da_cpvs = compute(da_cpvs)
# create_jet_relative_clim(phat_jets, exp.path, da_cpvs, suffix="_phat")
# del da_cpvs
# da_pv = open_da("ERA5", "thetalev", "PV", "dailymean", *args)
# da_pv = compute(da_pv)
# create_jet_relative_clim(phat_jets, exp.path, da_pv, suffix="_phat")
# del da_pv


diffs = (
    phat_props_catd[["jet", "time", "mean_lon", "mean_lat"]]
    .group_by("jet")
    .agg(
        pl.col("time"),
        cs.numeric().diff().abs(),
        (
            pl.col("time").diff().cast(pl.Float32())
            / pl.duration(
                seconds=1, time_unit=props_as_df["time"].dtype.time_unit
            ).cast(pl.Float32())
        ).alias("dt"),
        pl.col("mean_lat").alias("actual_lat"),
    )
    .explode(cs.all().exclude("jet"))
    .sort("time", "jet", descending=[False, True])
    .with_columns(
        com_speed=(
            haversine_from_dl(pl.col("actual_lat"), pl.col("mean_lon"), pl.col("mean_lat"))
            / pl.col("dt")
        )
        .cast(pl.Float32())
    )
    .rolling("time", period="2d", offset="-1d", group_by="jet")
    .agg(pl.col("com_speed").mean())
)
phat_props_catd = phat_props_catd.join(diffs[["time", "jet", "com_speed"]], on=["time", "jet"])
phat_props_catd_summer = summer_filter.join(phat_props_catd, on="time")


# cross_phat = track_jets(phat_jets, n_next=3)
# cross_phat.write_parquet(exp.path.joinpath("cross_phat.parquet"))
# cross_catd = track_jets(phat_jets_catd, n_next=3)
# cross_catd.write_parquet(exp.path.joinpath("cross_catd.parquet"))

cross_phat = pl.read_parquet(exp.path.joinpath("cross_phat.parquet"))
cross_catd = pl.read_parquet(exp.path.joinpath("cross_catd.parquet"))

spells_list_catd = spells_from_cross_catd(cross_catd, season=summer, q_STJ=0.93, q_EDJ=0.7)
spells_list = spells_from_cross(phat_jets, cross_phat, dis_polar_thresh=0.15, dist_thresh=2e5, season=summer, q_STJ=0.992, q_EDJ=0.97)

spells_from_jet_daily_stj_cs = get_persistent_jet_spells(
    phat_props_catd_summer,
    "com_speed",
    jet="STJ",
    q=0.8,
    minlen=datetime.timedelta(days=6),
    fill_holes=datetime.timedelta(hours=24),
).with_columns(spell_of=pl.lit("STJ"))
spells_from_jet_daily_edj_cs = get_persistent_jet_spells(
    phat_props_catd_summer,
    "com_speed",
    jet="EDJ",
    q=0.77,
    minlen=datetime.timedelta(days=6),
    fill_holes=datetime.timedelta(hours=24),
).with_columns(spell_of=pl.lit("EDJ"))
# spells_list = spells_list | {
#     "STJ_com": spells_from_jet_daily_stj_cs.cast({"time": pl.Datetime("ms"), "relative_time": pl.Duration("ms")}),
#     "EDJ_com": spells_from_jet_daily_edj_cs.cast({"time": pl.Datetime("ms"), "relative_time": pl.Duration("ms")}),
# }
# spells_list = spells_list | {f"{key}_catd": val for key, val in spells_list_catd.items()}
spells_list = {f"{key}_catd": val for key, val in spells_list_catd.items()}

for name, spell in spells_list.items():
    print(name, spell["spell"].n_unique())

kwargs = dict(time_before=datetime.timedelta(hours=24), time_after=datetime.timedelta(hours=24))
daily_spells_list = {a: make_daily(b, "spell", ["len", "spell_of"]) for a, b in spells_list.items()}


args = ["all", None, *get_region(ds), "all"]
da_T = open_da(
    "ERA5",
    "surf",
    "t2m",
    "dailymean",
    *args,
)
da_T = compute(da_T.sel(time=summer_daily), progress_flag=True)
da_T_upper = open_da(
    "ERA5",
    "plev",
    "t300",
    "dailymean",
    *args,
)
da_T_upper = compute(da_T_upper.sel(time=summer_daily), progress_flag=True)
da_tp = open_da(
    "ERA5",
    "surf",
    "tp",
    "dailysum",
    *args,
)
da_tp = compute(da_tp.sel(time=summer_daily), progress_flag=True)
da_apvs = open_da(
    "ERA5",
    "thetalev",
    "apvs",
    "dailyany",
    *args,
)
da_apvs = compute(da_apvs.sel(time=summer_daily), progress_flag=True)
da_cpvs = open_da(
    "ERA5",
    "thetalev",
    "cpvs",
    "dailyany",
    *args,
)
da_cpvs = compute(da_cpvs.sel(time=summer_daily), progress_flag=True)
da_pv = open_da(
    "ERA5",
    "thetalev",
    "PV",
    "dailymean",
    *args,
)
da_pv = compute(da_pv.sel(time=summer_daily), progress_flag=True)


def symmetrize_p(
    pvals: xr.DataArray | np.ndarray, direction: int, q: float
) -> np.ndarray:
    if direction == 0:
        return np.amin([pvals * 2, (1 - pvals) * 2], axis=0) < q
    if direction == 1:
        return np.asarray(pvals) > 1 - q
    return np.asarray(pvals) < q


basepath = Path("/storage/homefs/hb22g102/persistent-extremes-era5/Results/jet_rel_comp")

import os
for file in basepath.glob("*catd*.nc"):
   os.remove(file)

clims = {
    "t2m": xr.open_dataarray(exp.path.joinpath("t2m_phat_relative_clim.nc")),
    "t_up": xr.open_dataarray(exp.path.joinpath("t300_phat_relative_clim.nc")),
    "tp": xr.open_dataarray(exp.path.joinpath("tp_phat_relative_clim.nc")) * 1000,
    "apvs" : xr.open_dataarray(exp.path.joinpath("apvs_phat_relative_clim.nc")) * 100,
    "cpvs": xr.open_dataarray(exp.path.joinpath("cpvs_phat_relative_clim.nc")) * 100, 
    "pv": xr.open_dataarray(exp.path.joinpath("PV_phat_relative_clim.nc")) * 1e6,
}

variable_dict = {
    "t2m": da_T,
    "t_up": da_T_upper,
    "tp": da_tp * 1000,
    "apvs": da_apvs * 100,
    "cpvs": da_cpvs * 100,
    "pv": da_pv * 1e6,
}
long_names = {
    "t2m": "2m temperature [K]",
    "t_up": "Upper level temperature [K]",
    "tp": "Daily accum. precip. [mm]",
    "apvs": r"Anticyclonic PV streamer freq [$\%$]",
    "cpvs": r"Cyclonic PV streamer freq [$\%$]",
    "pv": "Potential vorticity [PVU]",
}

n_bootstraps = 100
days_around = [1, 3, 7]

for day_around in days_around:
    for spells_of in spells_list:
        spells_from_jet = daily_spells_list[spells_of]
        spells_from_jet = extend_spells(spells_from_jet)
        spells_from_jet = subset_around_onset(spells_from_jet, around_onset=datetime.timedelta(days=day_around))
        jets = summer_daily.rename("time").to_frame().join(all_jets_one_df, on="time")
        times = spells_from_jet
        for i, (varname, da) in enumerate(tqdm(variable_dict.items())):
            clim = clims[varname]
            ofile = Path(
                f"/storage/homefs/hb22g102/persistent-extremes-era5/Results/jet_rel_comp/{da.name}_interp_spells_of_{spells_of}_JJAS_{day_around=}.nc"
            )
            if ofile.is_file():
                continue
            jets_during_spells_with_interp_norm_ds = gather_normal_da_jets_wrapper(
                jets, times, da, n_bootstraps=n_bootstraps, clim=clim
            )
            try:
                to_plot = jets_during_spells_with_interp_norm_ds[da.name + "_interp"]
                pvals = jets_during_spells_with_interp_norm_ds["pvals"]
                to_plot.to_netcdf(ofile)
                pvals.to_netcdf(f"/storage/homefs/hb22g102/persistent-extremes-era5/Results/jet_rel_comp/{da.name}_interp_spells_of_{spells_of}_JJAS_pvals_{day_around=}.nc")
            except KeyError:
                to_plot = jets_during_spells_with_interp_norm_ds
                to_plot.to_netcdf(ofile)
