#!/bin/python3

import xarray as xr
from jetstream_hugo.definitions import DATADIR
from jetstream_hugo.jet_finding import MultiVarExperiment


exp = MultiVarExperiment("ERA5", "plev", ["u", "v", "s"], "6H", (1940, 2022), None, -80, 30, 20, 80, [200, 250, 300])
qss = xr.open_dataarray(f"{DATADIR}/ERA5/plev/results/4/qs_clim.nc")

all_jets, where_are_jets, all_jets_one_array = exp.find_jets(thresholds=qss[20, :].reset_coords("quantile", drop=True), processes=48)
