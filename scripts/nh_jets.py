from jetutils.definitions import *
from jetutils.data import *
from jetutils.jet_finding import *
from jetutils.anyspell import *
from jetutils.plots import *
from jetutils.geospatial import *


def create_jet_relative_dataset(jets, path, da, suffix="", half_length: float = 20.):
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
    
    
    
run = "ctrl"

dh = DataHandler.from_specs("Henrik_data", run, ("high_wind", ["u", "v", "s", "theta"]), "6H", minlon=-80, maxlon=40, minlat=15, maxlat=80, levels=20000)
exp = JetFindingExperiment(dh)
basepath_ctrl = exp.path
all_jets_one_df = exp.find_jets(force=False, n_coarsen=1, smooth_s=5, alignment_thresh=0.55, base_s_thresh=0.55, int_thresh_factor=0.2, hole_size=5)
theta300 = open_da("Henrik_data", run, ("high_wind", ["s", "theta"]), "6H", minlon=-80, maxlon=40, minlat=15, maxlat=80, levels=30000).rename({"s": "s300", "theta": "theta300"})
all_jets_one_df = add_feature_for_cat(all_jets_one_df, "s300", theta300, ofile_ajdf=exp.path.joinpath("all_jets_one_df.parquet"), force=False)
all_jets_one_df = add_feature_for_cat(all_jets_one_df, "theta300", theta300, ofile_ajdf=exp.path.joinpath("all_jets_one_df.parquet"), force=False)
all_jets_one_df = exp.categorize_jets(None, ["s300", "theta300"], force=False, n_init=15, init_params="k-means++", mode="month").cast({"time": pl.Datetime("ms")})
props_uncat = exp.props_as_df(categorize=False).cast({"time": pl.Datetime("ms")})

all_times = props_uncat["time"].unique().sort().to_frame()
summer_filter = (
    all_times
    .filter(pl.col("time").dt.month().is_in([6, 7, 8, 9]))
    .filter(pl.col("time").dt.ordinal_day() > 166)
)
summer = summer_filter["time"]
summer_daily = summer.filter(summer.dt.hour() == 0)
big_summer = all_times.filter(pl.col("time").dt.month().is_in([6, 7, 8, 9]))
big_summer_daily = big_summer.filter(big_summer["time"].dt.hour() == 0)

phat_filter = (pl.col("is_polar") < 0.5) | ((pl.col("is_polar") > 0.5) & (pl.col("int") > 1.e8))
phat_jets = all_jets_one_df.filter((pl.col("is_polar").mean().over(["time", "jet ID"]) < 0.5) | ((pl.col("is_polar").mean().over(["time", "jet ID"]) > 0.5) & (pl.col("int").mode().first().over(["time", "jet ID"]) > 1.e8)))

jets_ctrl = phat_jets.with_columns(**{"jet ID": (pl.col("is_polar").mean().over(["time", "jet ID"]) > 0.5).cast(pl.UInt32())})
phat_props = props_uncat.filter(phat_filter)
phat_props_catd = average_jet_categories(phat_props, polar_cutoff=0.5)
phat_props_catd = phat_props_catd.join(phat_props_catd.rolling("time", period="2d", group_by="jet").agg(**{f"{col}_var": pl.col(col).var() for col in ["mean_lon", "mean_lat", "mean_s", "s_star"]}), on=["time", "jet"])
props_ctrl = phat_props_catd.clone()
props_summer_ctrl = summer_filter.join(props_ctrl, on="time")

opath_cross_ctrl = basepath_ctrl.joinpath("cross_catd.parquet")
if opath_cross_ctrl.is_file():
    cross_catd_ctrl = pl.read_parquet(opath_cross_ctrl)
else:
    cross_catd_ctrl = track_jets(jets_ctrl, n_next=1)
    cross_catd_ctrl.write_parquet(opath_cross_ctrl)
    
spells_list_ctrl = spells_from_cross_catd(cross_catd_ctrl, season=summer, q_STJ=0.9, q_EDJ=0.885, minlen=datetime.timedelta(days=4))

for name, spell in spells_list_ctrl.items():
    print(name, spell["spell"].n_unique())
daily_spells_list_ctrl = {a: make_daily(b, "spell", ["len", "spell_of"]) for a, b in spells_list_ctrl.items()}



args = ["all", None, -100, 60, 0, 90]
path = exp.path

da = open_da("Henrik_data", run, ("high_wind", "PTTEND"), "6H", *args, levels=30000)
da = compute(da)
create_jet_relative_dataset(jets_ctrl, path, da)
del da

da = open_da("Henrik_data", run, ("high_wind", "DTCOND"), "6H", *args, levels=30000)
da = compute(da)
create_jet_relative_dataset(jets_ctrl, path, da)
del da

da = open_da("Henrik_data", run, ("high_wind", "theta"), "6H", *args, levels=30000)
da = compute(da)
create_jet_relative_dataset(jets_ctrl, path, da)
del da

da = open_da("Henrik_data", run, "t_low", "6H", *args, levels=100000)
da = compute(da)
create_jet_relative_dataset(jets_ctrl, path, da)
del da

da = open_da("Henrik_data", run, "z", "6H", *args)
da = compute(da)
create_jet_relative_dataset(jets_ctrl, path, da)
del da


run = "dobl"

dh = DataHandler.from_specs("Henrik_data", run, ("high_wind", ["u", "v", "s", "theta"]), "6H", minlon=-80, maxlon=40, minlat=15, maxlat=80, levels=20000)
exp = JetFindingExperiment(dh)
basepath_dobl = exp.path
all_jets_one_df = exp.find_jets(force=False, n_coarsen=1, smooth_s=5, alignment_thresh=0.55, base_s_thresh=0.55, int_thresh_factor=0.2, hole_size=5)
theta300 = open_da("Henrik_data", run, ("high_wind", ["s", "theta"]), "6H", minlon=-80, maxlon=40, minlat=15, maxlat=80, levels=30000).rename({"s": "s300", "theta": "theta300"})
all_jets_one_df = add_feature_for_cat(all_jets_one_df, "s300", theta300, ofile_ajdf=exp.path.joinpath("all_jets_one_df.parquet"), force=False)
all_jets_one_df = add_feature_for_cat(all_jets_one_df, "theta300", theta300, ofile_ajdf=exp.path.joinpath("all_jets_one_df.parquet"), force=False)
all_jets_one_df = exp.categorize_jets(None, ["s300", "theta300"], force=False, n_init=15, init_params="k-means++", mode="month").cast({"time": pl.Datetime("ms")})
props_uncat = exp.props_as_df(categorize=False).cast({"time": pl.Datetime("ms")})

all_times = props_uncat["time"].unique().sort().to_frame()
summer_filter = (
    all_times
    .filter(pl.col("time").dt.month().is_in([6, 7, 8, 9]))
    .filter(pl.col("time").dt.ordinal_day() > 166)
)
summer = summer_filter["time"]
summer_daily = summer.filter(summer.dt.hour() == 0)
big_summer = all_times.filter(pl.col("time").dt.month().is_in([6, 7, 8, 9]))
big_summer_daily = big_summer.filter(big_summer["time"].dt.hour() == 0)

phat_filter = (pl.col("is_polar") < 0.5) | ((pl.col("is_polar") > 0.5) & (pl.col("int") > 1.e8))
phat_jets = all_jets_one_df.filter((pl.col("is_polar").mean().over(["time", "jet ID"]) < 0.5) | ((pl.col("is_polar").mean().over(["time", "jet ID"]) > 0.5) & (pl.col("int").mode().first().over(["time", "jet ID"]) > 1.e8)))

jets_dobl = phat_jets.with_columns(**{"jet ID": (pl.col("is_polar").mean().over(["time", "jet ID"]) > 0.5).cast(pl.UInt32())})
phat_props = props_uncat.filter(phat_filter)
phat_props_catd = average_jet_categories(phat_props, polar_cutoff=0.5)
phat_props_catd = phat_props_catd.join(phat_props_catd.rolling("time", period="2d", group_by="jet").agg(**{f"{col}_var": pl.col(col).var() for col in ["mean_lon", "mean_lat", "mean_s", "s_star"]}), on=["time", "jet"])
props_dobl = phat_props_catd.clone()
props_summer_dobl = summer_filter.join(props_dobl, on="time")

opath_cross_dobl = basepath_dobl.joinpath("cross_catd.parquet")
if opath_cross_dobl.is_file():
    cross_catd_dobl = pl.read_parquet(opath_cross_dobl)
else:
    cross_catd_dobl = track_jets(jets_ctrl, n_next=1)
    cross_catd_dobl.write_parquet(opath_cross_dobl)
    
spells_list_dobl = spells_from_cross_catd(cross_catd_dobl, season=summer, q_STJ=0.9, q_EDJ=0.885, minlen=datetime.timedelta(days=4))

for name, spell in spells_list_dobl.items():
    print(name, spell["spell"].n_unique())
daily_spells_list_dobl = {a: make_daily(b, "spell", ["len", "spell_of"]) for a, b in spells_list_dobl.items()}


args = ["all", None, -100, 60, 0, 90]
path = exp.path

da = open_da("Henrik_data", run, ("high_wind", "PTTEND"), "6H", *args, levels=30000)
da = compute(da)
create_jet_relative_dataset(jets_dobl, path, da)
del da

da = open_da("Henrik_data", run, ("high_wind", "DTCOND"), "6H", *args, levels=30000)
da = compute(da)
create_jet_relative_dataset(jets_dobl, path, da)
del da

da = open_da("Henrik_data", run, ("high_wind", "theta"), "6H", *args, levels=30000)
da = compute(da)
create_jet_relative_dataset(jets_dobl, path, da)
del da

da = open_da("Henrik_data", run, "t_low", "6H", *args, levels=100000)
da = compute(da)
create_jet_relative_dataset(jets_dobl, path, da)
del da

da = open_da("Henrik_data", run, "z", "6H", *args)
da = compute(da)
create_jet_relative_dataset(jets_dobl, path, da)
del da