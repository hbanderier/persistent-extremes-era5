#!/bin/python3

from tqdm import trange
from pathlib import Path
import xarray as xr
DATADIR = "/storage/workspaces/giub_meteo_impacts/ci01"

source = Path(DATADIR).joinpath("ERA5/plev/tmp/uvfull")
destu = Path(DATADIR).joinpath("ERA5/plev/u/6H")
destv = Path(DATADIR).joinpath("ERA5/plev/v/6H")
for year in trange(1959, 2023):
    yearstr = str(year).zfill(4)
    for month in range(1, 13):
        monthstr = str(month).zfill(2)
        filename = f"{yearstr}{monthstr}.nc"
        thisdestu = destu.joinpath(filename)
        thisdestv = destv.joinpath(filename)
        if thisdestu.is_file() and thisdestv.is_file():
            continue
        ds = xr.open_dataset(source.joinpath(filename)).rename(level="lev")
        ds["u"].to_netcdf(thisdestu)
        ds["v"].to_netcdf(thisdestv)
        ds.close()