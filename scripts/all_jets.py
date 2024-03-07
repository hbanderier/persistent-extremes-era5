#!/bin/python3

import xarray as xr
from jetstream_hugo.definitions import DATADIR, save_pickle
from jetstream_hugo.data import open_da
from jetstream_hugo.jet_finding import JetFinder, preprocess, cluster_wind_speed, jets_from_mask


ds = xr.Dataset()
ds["s"] = open_da("ERA5", "plev", "s", "6H", (1940, 2022), None, -80, 30, 20, 80, [200, 250, 300]).load()
ds["u"] = open_da("ERA5", "plev", "u", "6H", (1940, 2022), None, -80, 30, 20, 80, [200, 250, 300]).load()
ds["v"] = open_da("ERA5", "plev", "v", "6H", (1940, 2022), None, -80, 30, 20, 80, [200, 250, 300]).load()
qss = xr.open_dataarray(f"{DATADIR}/ERA5/plev/s/6H/results/qs_clim.nc")

jet_finder = JetFinder(
    preprocess=preprocess,
    cluster = cluster_wind_speed,
    refine_jets=jets_from_mask,
)

jets = jet_finder.call(ds, thresholds=qss[20, :] / 4, processes=50)
save_pickle(jets, f"{DATADIR}/ERA5/plev/all_jets.pkl")