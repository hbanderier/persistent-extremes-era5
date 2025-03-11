# import packages
import intake
import numpy as np
from pathlib import Path
from dask.diagnostics import ProgressBar

## Configure
var = ["U", "V"]
outpath = Path("/storage/workspaces/giub_meteo_impacts/ci01/CESM2/for_cra")
## change this!

col_url = (
    "https://ncar-cesm2-lens.s3-us-west-2.amazonaws.com/catalogs/aws-cesm2-le.json"
)
catalog = intake.open_esm_datastore(col_url)

catalog_subset = catalog.search(variable=var, frequency='daily', forcing_variant="cmip6")
dsets = catalog_subset.to_dataset_dict(storage_options={'anon':True})

ds_past = dsets["atm.historical.daily.cmip6"]
ds_future = dsets["atm.ssp370.daily.cmip6"]

minlon, maxlon, minlat, maxlat = -30, 40, 20, 80
ds_past_ns = (
    ds_past
    .isel(member_id=[0, 1, 2, 3, 4, 5])
    .isel(time=np.isin(ds_past.time.dt.year, np.arange(1970, 2000)))
    .isel(lev=-1)
    .sel(lon=slice(minlon, maxlon))
    .sel(lat=slice(minlat, maxlat))
)
ds_future_ns = (
    ds_future
    .isel(member_id=[0, 1, 2, 3, 4, 5])
    .isel(time=np.isin(ds_future.time.dt.year, np.arange(2070, 2100)))
    .isel(lev=-1)
    .sel(lon=slice(minlon, maxlon))
    .sel(lat=slice(minlat, maxlat))
)

with ProgressBar():
    ds_past_ns = ds_past_ns.load()
ds_past_ns.to_netcdf(outpath.joinpath("past.nc"))

with ProgressBar():
    ds_future_ns = ds_future_ns.load()
ds_future_ns.to_netcdf(outpath.joinpath("future.nc"))