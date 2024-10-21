from jetstream_hugo.jet_finding import *
from jetstream_hugo.definitions import *
from jetstream_hugo.plots import *
from jetstream_hugo.clustering import *
from jetstream_hugo.data import *


ds_cesm = xr.open_dataset("/storage/workspaces/giub_meteo_impacts/ci01/CESM2/flat_wind/ds.zarr", engine="zarr")
ds_cesm = ds_cesm.chunk({"member": 1, "time": 100, "lat": -1, "lon": -1})
da_cesm = ds_cesm["s"].sel(time=ds_cesm.time.dt.season=="JJA")
dh = DataHandler(da_cesm, "/storage/workspaces/giub_meteo_impacts/ci01/CESM2/flat_wind/results")
exp_s_cesm = Experiment(dh)
exp_s_cesm.load_da(progress=True)

kwargs_som = dict(
    nx=8,
    ny=6,
    metric="euclidean",
    PBC=True,
    return_type=RAW_REALSPACE,
    force=False,
    learning_rate = 0.02,
    learning_rateN=0.002,
)

net, centers, labels = exp_s_cesm(**kwargs_som)