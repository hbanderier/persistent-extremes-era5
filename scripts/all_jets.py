#!/bin/python3

import xarray as xr
from jetstream_hugo.definitions import DATADIR
from jetstream_hugo.jet_finding import MultiVarExperiment


exp = MultiVarExperiment("ERA5", "plev", ["u", "v", "s"], "6H", "all", None, -80, 30, 20, 80, [150, 200, 250, 300, 350])

thresholds = xr.open_dataarray(f"{DATADIR}/ERA5/plev/results/qs_clim.nc")
thresholds = thresholds[20, :].reset_coords("quantile", drop=True)

all_jets, where_are_jets, all_jets_one_array = exp.find_jets(thresholds=thresholds, processes=48, chunksize=200)
