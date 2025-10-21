from jetutils.definitions import *
from jetutils.data import *
from jetutils.jet_finding import *
from jetutils.anyspell import *
from jetutils.plots import *


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

dh = DataHandler.from_specs("Henrik_data", run, ("high_wind", ["u", "v", "s", "theta", "lev"]), "6H", minlon=-80, maxlon=40, minlat=15, maxlat=80)
exp = JetFindingExperiment(dh)
exp.find_jets(force=False, n_coarsen=1, smooth_s=5, alignment_thresh=0.55, base_s_thresh=0.55, int_thresh_factor=0.2, hole_size=5)
# jets = exp.categorize_jets(None, ["s", "theta"], force=False, n_init=15, init_params="k-means++", mode="week").cast({"time": pl.Datetime("ms")})
# dh_mid = DataHandler.from_specs("Henrik_data", run, ("mid_wind", "s"), "6H", minlon=-80, maxlon=40, minlat=15, maxlat=80)
jets = exp.categorize_jets(None, ["s", "theta"], force=False, n_init=15, init_params="k-means++", mode="week").cast({"time": pl.Datetime("ms")})

props_uncat = exp.props_as_df(categorize=False)
props_as_df = exp.props_as_df(force=0)

phat_filter = (pl.col("is_polar") < 0.5) | ((pl.col("is_polar") > 0.5) & (pl.col("int") > 1.e8))

phat_jets = jets.filter((pl.col("is_polar").mean().over(["time", "jet ID"]) < 0.5) | ((pl.col("is_polar").mean().over(["time", "jet ID"]) > 0.5) & (pl.col("int").mode().first().over(["time", "jet ID"]) > 1.e8)))
phat_jets_catd = phat_jets.with_columns(**{"jet ID": (pl.col("is_polar").mean().over(["time", "jet ID"]) > 0.5).cast(pl.UInt32())})
phat_props = props_uncat.filter(phat_filter)
phat_props_catd = average_jet_categories(phat_props, polar_cutoff=0.5)

jets_ctrl = phat_jets_catd.clone()
props_ctrl = phat_props_catd.clone()



args = ["all", None, -100, 60, 0, 90]
path = exp.path

da = open_da("Henrik_data", run, ("high_wind", "PTTEND"), "6H", *args)
da = compute(da)
create_jet_relative_dataset(jets_ctrl, path, da)
del da

da = open_da("Henrik_data", run, ("high_wind", "DTCOND"), "6H", *args)
da = compute(da)
create_jet_relative_dataset(jets_ctrl, path, da)
del da

da = open_da("Henrik_data", run, ("high_wind", "theta"), "6H", *args)
da = compute(da)
create_jet_relative_dataset(jets_ctrl, path, da)
del da

da = open_da("Henrik_data", run, "z", "6H", *args)
da = compute(da)
create_jet_relative_dataset(jets_ctrl, path, da)
del da


run = "dobl"

dh = DataHandler.from_specs("Henrik_data", run, ("high_wind", ["u", "v", "s", "theta", "lev"]), "6H", minlon=-80, maxlon=40, minlat=15, maxlat=80)
exp = JetFindingExperiment(dh)
exp.find_jets(force=False, n_coarsen=1, smooth_s=5, alignment_thresh=0.55, base_s_thresh=0.55, int_thresh_factor=0.2, hole_size=5)
jets = exp.categorize_jets(None, ["s", "theta"], force=False, n_init=15, init_params="k-means++", mode="week").cast({"time": pl.Datetime("ms")})

props_as_df = exp.props_as_df(force=0)
props_uncat = exp.props_as_df(categorize=False)

phat_filter = (pl.col("is_polar") < 0.5) | ((pl.col("is_polar") > 0.5) & (pl.col("int") > 1.e8))

phat_jets = jets.filter((pl.col("is_polar").mean().over(["time", "jet ID"]) < 0.5) | ((pl.col("is_polar").mean().over(["time", "jet ID"]) > 0.5) & (pl.col("int").mode().first().over(["time", "jet ID"]) > 1.e8)))
phat_jets_catd = phat_jets.with_columns(**{"jet ID": (pl.col("is_polar").mean().over(["time", "jet ID"]) > 0.5).cast(pl.UInt32())})
phat_props = props_uncat.filter(phat_filter)
phat_props_catd = average_jet_categories(phat_props, polar_cutoff=0.5)

jets_dobl = phat_jets_catd.clone()
props_dobl = phat_props_catd.clone()


args = ["all", None, -100, 60, 0, 90]
path = exp.path

da = open_da("Henrik_data", run, ("high_wind", "PTTEND"), "6H", *args)
da = compute(da)
create_jet_relative_dataset(jets_dobl, path, da)
del da

da = open_da("Henrik_data", run, ("high_wind", "DTCOND"), "6H", *args)
da = compute(da)
create_jet_relative_dataset(jets_dobl, path, da)
del da

da = open_da("Henrik_data", run, ("high_wind", "theta"), "6H", *args)
da = compute(da)
create_jet_relative_dataset(jets_dobl, path, da)
del da

da = open_da("Henrik_data", run, "z", "6H", *args)
da = compute(da)
create_jet_relative_dataset(jets_dobl, path, da)
del da