#!/bin/python3

import xarray as xr
from jetstream_hugo.clustering import Experiment

exp_s = Experiment(
    "ERA5", "plev", "s", "6H", (1940, 2023), None, -80, 30, 20, 75
)
thresholds = xr.open_dataarray("/storage/workspaces/giub_meteo_impacts/ci01/ERA5/plev/s/6H/results/q80_clim.nc")
exp_s.find_jets(thresholds=thresholds, processes=30, chunksize=1)
