import numpy as np
import xarray as xr
from jetstream_hugo.definitions import KAPPA
from jetstream_hugo.data import DataHandler
from jetstream_hugo.jet_finding import JetFindingExperiment, get_double_jet_index


def main():
    for period in ["historical", "ssp370"]:
        ds_cesm = xr.open_dataset(f"/storage/workspaces/giub_meteo_impacts/ci01/CESM2/high_wind/{period}/ds.zarr", engine="zarr")
        ds_cesm = ds_cesm.reset_coords("time_bnds", drop=True).drop_dims("nbnd")
        ds_cesm = ds_cesm.chunk({"member": 1, "time": 100, "lat": -1, "lon": -1})
        ds_cesm["theta"] = (ds_cesm["t"] * (1000 / ds_cesm["lev"]) ** KAPPA).astype(np.float32)
        ds_cesm = ds_cesm.drop_vars("t")
        dh = DataHandler.from_basepath_and_da(f"/storage/workspaces/giub_meteo_impacts/ci01/CESM2/high_wind/{period}/results", ds_cesm)
        exp = JetFindingExperiment(dh)
        exp.find_jets()

        ds_low = xr.open_dataset(f"/storage/workspaces/giub_meteo_impacts/ci01/CESM2/mid_wind/{period}/ds.zarr", engine="zarr")
        ds_low = ds_low.reset_coords("time_bnds", drop=True).drop_dims("nbnd")
        ds_low = ds_low.chunk({"member": 1, "time": 100, "lat": -1, "lon": -1})

        exp.categorize_jets(ds_low["s"])
        exp.track_jets()
        props_as_df = exp.props_as_df(True)
        exp.jet_position_as_da()
        jet_pos_da = exp.jet_position_as_da()
        props_as_df = get_double_jet_index(props_as_df, jet_pos_da)
        props_as_df.write_parquet(exp.path.joinpath("props_as_df_extras.parquet"))


if __name__ == "__main__":
    main()