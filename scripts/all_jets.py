from jetstream_hugo.jet_finding import JetFindingExperiment
from jetstream_hugo.data import DataHandler
from multiprocessing import freeze_support
import xarray as xr

if __name__ == '__main__':
    freeze_support()
    ds_cesm = xr.open_dataset("/storage/workspaces/giub_meteo_impacts/ci01/CESM2/flat_wind/ds.zarr", engine="zarr")
    ds_cesm = ds_cesm.chunk({"member": 1, "time": 100, "lat": -1, "lon": -1})
    dh = DataHandler(ds_cesm, "/storage/workspaces/giub_meteo_impacts/ci01/CESM2/flat_wind/results")
    exp_cesm = JetFindingExperiment(dh)
    jets_cesm, _, _ = exp_cesm.track_jets()
    props_cesm = exp_cesm.props_as_df(True)