from jetstream_hugo.definitions import DATADIR
from jetstream_hugo.jet_finding import JetFindingExperiment
from jetstream_hugo.data import DataHandler
import xarray as xr

if __name__ == '__main__':
    # exp = JetFindingExperiment(DataHandler(f"{DATADIR}/ERA5/plev/high_wind/6H/results/6"))
    # dh_low = DataHandler.from_specs("ERA5", "plev", "mid_wind", "6H", "all", None, -80, 40, 15, 80, "all")
    # exp.find_jets()
    # exp.categorize_jets(dh_low.da["s"])
    # exp.track_jets()
    # exp.props_as_df()
    ds_cesm = xr.open_dataset("/storage/workspaces/giub_meteo_impacts/ci01/CESM2/flat_wind/ds.zarr", engine="zarr")
    ds_cesm = ds_cesm.chunk({"member": 1, "time": 100, "lat": -1, "lon": -1})
    dh = DataHandler.from_basepath_and_da("/storage/workspaces/giub_meteo_impacts/ci01/CESM2/flat_wind/results", ds_cesm)
    exp_cesm = JetFindingExperiment(dh)
    jets_cesm = exp_cesm.find_jets()