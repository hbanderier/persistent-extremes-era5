import datetime

import colormaps
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import xarray as xr
from jetutils.anyspell import get_persistent_jet_spells, mask_from_spells_pl, subset_around_onset, get_persistent_spell_times_from_som, get_spells, extend_spells, gb_index, make_daily
from jetutils.clustering import Experiment, RAW_REALSPACE, labels_to_centers
from jetutils.data import DataHandler, open_da, data_path, compute_anomalies_pl, coarsen_da, compute_anomalies_ds
from jetutils.definitions import (
    DATADIR,
    FIGURES,
    PRETTIER_VARNAME,
    YEARS,
    KAPPA,
    compute,
    get_region,
    infer_direction,
    polars_to_xarray,
    xarray_to_polars,
    labels_to_mask,
    extract_season_from_df,
    DUNCANS_REGIONS_NAMES,
    UNITS,
    N_WORKERS,
    do_rle,
    do_rle_fill_hole,
    squarify,
)
from jetutils.jet_finding import JetFindingExperiment, jet_integral_haversine, find_all_jets, haversine_from_dl, jet_position_as_da, add_normals, gather_normal_da_jets, average_jet_categories, connected_from_cross, get_double_jet_index, iterate_over_year_maybe_member, track_jets, average_jet_categories_v2, spells_from_cross, spells_from_cross_catd
from jetutils.plots import COLORS, COLORS_EXT, PINKPURPLE, WERNLI_FLAIR, WERNLI_FLAIR_LEVELS, Clusterplot, gather_normal_da_jets_wrapper, _gather_normal_da_jets_wrapper, make_transparent, honeycomb_panel, plot_seasonal, interp_jets_to_zero_one
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from pathlib import Path
from tqdm import tqdm
import polars.selectors as cs


all_times = (
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
)
summer_filter = (
    all_times
    .filter(pl.col("time").dt.month().is_in([6, 7, 8, 9]))
    .filter(pl.col("time").dt.ordinal_day() > 166)
)
summer = summer_filter["time"]
summer_daily = summer.filter(summer.dt.hour() == 0)
big_summer = all_times.filter(pl.col("time").dt.month().is_in([6, 7, 8, 9]))
big_summer_daily = big_summer.filter(big_summer["time"].dt.hour() == 0)
dh = DataHandler("/storage/workspaces/giub_meteo_impacts/ci01/ERA5/plev/high_wind/6H/results/7")
exp = JetFindingExperiment(dh)
ds = exp.ds
all_jets_one_df = exp.find_jets(force=False, alignment_thresh=0.6, base_s_thresh=0.55, int_thresh_factor=0.35, hole_size=10)
all_jets_one_df = exp.categorize_jets(None, ["s", "theta"], force=False, n_init=5, init_params="k-means++", mode="week").cast({"time": pl.Datetime("ms")})

props_uncat = exp.props_as_df(False).cast({"time": pl.Datetime("ms")})
props_as_df = average_jet_categories(props_uncat, polar_cutoff=0.5)

props_summer = summer_filter.join(props_as_df, on="time")
phat_filter = (pl.col("is_polar") < 0.5) | ((pl.col("is_polar") > 0.5) & (pl.col("int") > 1.5e8))

phat_jets = all_jets_one_df.filter((pl.col("is_polar").mean().over(["time", "jet ID"]) < 0.5) | ((pl.col("is_polar").mean().over(["time", "jet ID"]) > 0.5) & (pl.col("int").mode().first().over(["time", "jet ID"]) > 1.5e8)))
phat_jets_catd = phat_jets.with_columns(**{"jet ID": (pl.col("is_polar").mean().over(["time", "jet ID"]) > 0.5).cast(pl.UInt32())})

def create_jet_relative_dataset(jets, path, da, suffix="", half_length: float = 20.):
    jets = jets.with_columns(pl.col("time").dt.round("1d"))
    jets = jets.with_columns(jets.group_by("time", maintain_order=True).agg(pl.col("jet ID").rle_id())["jet ID"].explode())
    indexer = iterate_over_year_maybe_member(jets, da)
    to_average = []
    varname = da.name + "_interp"
    for idx1, idx2 in tqdm(indexer, total=len(YEARS)):
        jets_ = jets.filter(*idx1)
        da_ = da.sel(**idx2)
        try:
            jets_with_interp = gather_normal_da_jets(jets_, da_, half_length=half_length)
        except (KeyError, ValueError) as e:
            print(e)
            break
        jets_with_interp = interp_jets_to_zero_one(jets_with_interp, [varname, "is_polar"], n_interp=30)
        jets_with_interp = jets_with_interp.group_by("time", pl.col("is_polar").mean().over(["time", "jet ID"]) > 0.5, "norm_index", "n", maintain_order=True).agg(pl.col(varname).mean())
        to_average.append(jets_with_interp)
    pl.concat(to_average).write_parquet(path.joinpath(f"{da.name}{suffix}_relative.parquet"))
    
    
args = ["all", None, -100, 60, 0, 90]

args = ["all", None, *get_region(exp.ds)]
da_blocks = open_da("ERA5", "surf", "blocks", "dailymean", *args)
da_blocks = compute(da_blocks)
create_jet_relative_dataset(phat_jets_catd, exp.path, da_blocks, suffix="_phat_catd")
del da_blocks
# da_T = open_da("ERA5", "surf", "t2m", "dailymean", *args)
# da_T = compute(da_T)
# create_jet_relative_dataset(phat_jets_catd, exp.path, da_T, suffix="_phat_catd")
# del da_T

# da_tp = open_da("ERA5", "surf", "tp", "dailysum", *args)
# da_tp = compute(da_tp)
# create_jet_relative_dataset(phat_jets_catd, exp.path, da_tp, suffix="_phat_catd")
# del da_tp

# da_pv = open_da("ERA5", "thetalev", "PV330", "dailymean", *args)
# da_pv = compute(da_pv).rename("PV330")
# create_jet_relative_dataset(phat_jets_catd, exp.path, da_pv, suffix="_phat_catd")
# del da_pv

# da_pv = open_da("ERA5", "thetalev", "PV350", "dailymean", *args)
# da_pv = compute(da_pv).rename("PV350")
# create_jet_relative_dataset(phat_jets_catd, exp.path, da_pv, suffix="_phat_catd")
# del da_pv

# da_theta2pvu = open_da("ERA5", "surf", "theta2PVU", "dailymean", *args)
# da_theta2pvu = compute(da_theta2pvu)
# create_jet_relative_dataset(phat_jets_catd, exp.path, da_theta2pvu, suffix="_phat_catd")
# del da_theta2pvu

# varnames_rwb = ["APVO"]
# for var in varnames_rwb:
#     da_rwb = open_da("ERA5", "thetalev", var, "dailyany", *args)
#     da_rwb = compute(da_rwb)
#     create_jet_relative_dataset(phat_jets_catd, exp.path, da_rwb, suffix="_phat_catd")
#     del da_rwb
    
# dh_z500 = DataHandler.from_specs(
#     "ERA5",
#     "plev",
#     "z",
#     "dailymean",
#     (1959, 2023),
#     None,
#     levels=500
# )
# opath = dh_z500.path.joinpath("blocks.nc")
# if opath.is_file:
#     blocks = xr.open_dataarray(opath).rename("Blocks")
# else:
#     da_z500 = dh_z500.da.rename("z500").load() / 9.8
#     da_z500_zero = da_z500
#     da_z500_s = da_z500.assign_coords(lat=da_z500.lat.values + 15).sel(lat=da_z500.lat <= 90 - 15)
#     da_z500_s2 = da_z500.assign_coords(lat=da_z500.lat.values + 30).sel(lat=da_z500.lat <= 90 - 30)
#     da_z500_n = da_z500.assign_coords(lat=da_z500.lat.values - 15).sel(lat=da_z500.lat >= 15)
#     GHGS = (da_z500_zero - da_z500_s) / 15 > 0
#     GHGN = (da_z500_n - da_z500_zero) / 15 < -10
#     GHGS2 = (da_z500_s - da_z500_s2) / 15 < -5
#     blocks = (GHGS & GHGN & GHGS2).rename("Blocks")
#     blocks = compute(blocks, progress_flag=True)
#     blocks.to_netcdf(opath)
    
# create_jet_relative_dataset(phat_jets_catd, exp.path, blocks, suffix="_phat_catd")
