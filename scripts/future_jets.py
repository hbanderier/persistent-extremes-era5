import xarray as xr
from jetstream_hugo.data import DataHandler
from jetstream_hugo.jet_finding import JetFindingExperiment, get_double_jet_index


ds_cesm = xr.open_dataset("/storage/workspaces/giub_meteo_impacts/ci01/CESM2/high_wind/historical/ds.zarr", engine="zarr")
ds_cesm = ds_cesm.reset_coords("time_bnds", drop=True).drop_dims("nbnd")
ds_cesm = ds_cesm.chunk({"member": 1, "time": 100, "lat": -1, "lon": -1})
dh = DataHandler.from_basepath_and_da("/storage/workspaces/giub_meteo_impacts/ci01/CESM2/high_wind/historical/results", ds_cesm)
exp = JetFindingExperiment(dh)
exp.find_jets()

ds_low = xr.open_dataset("/storage/workspaces/giub_meteo_impacts/ci01/CESM2/mid_wind/historical/ds.zarr", engine="zarr")
ds_low = ds_low.reset_coords("time_bnds", drop=True).drop_dims("nbnd")
ds_low = ds_low.chunk({"member": 1, "time": 100, "lat": -1, "lon": -1})

all_jets_one_df = exp.categorize_jets(ds_low["s"])
exp.track_jets()
props_as_df = exp.props_as_df(True)
exp.jet_position_as_da()
jet_pos_da = exp.jet_position_as_da()
props_as_df = get_double_jet_index(props_as_df, jet_pos_da)
props_as_df.write_parquet(exp.path.joinpath("props_as_df_extras.parquet"))