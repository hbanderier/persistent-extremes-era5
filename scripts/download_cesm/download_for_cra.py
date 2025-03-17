# Configure file behaviour at the start of the main function !!

# import packages
import intake
import numpy as np
import xarray as xr
from pathlib import Path
from dask.diagnostics import ProgressBar


def standardize(da: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    if (da.lon.max() > 180) and (da.lon.min() >= 0): # we prefer longitude to go from -180 to +180 rather than 0 to 360
        da = da.assign_coords(lon=(((da.lon + 180) % 360) - 180))
        da = da.sortby("lon")
    if np.diff(da.lat.values)[0] < 0: # and increasing latitudes
        da = da.reindex(lat=da.lat[::-1])
    return da


def main():
    ## Configure
    varname = "RAIN"
    component = "lnd" # for land variables like RAIN, "atm" for atmospheric variables like wind, and "ocn" for ocean variables
    forcing_variant = "cmip6" # other option is "smbb", which stands for "SMoothed Biomass Burning"
    out_path = Path("/Users/bandelol/Documents/code_local/data/cesm-test")
    minlon, maxlon, minlat, maxlat = -30, 40, 20, 80
    years = {
        "past": np.arange(1970, 2000),
        "future": np.arange(2070, 2100),
    }
    levels = None # if you need to subselect levels from 3D data, then set their index / indices here, otherwise leave as None
    members = np.arange(5)
    out_name_past = f"{varname}_{len(members)}members_{years['past'][0]}-{years['past'][-1]}.nc"
    out_name_future = f"{varname}_{len(members)}members_{years['future'][0]}-{years['future'][-1]}.nc"
    
    col_url = (
        "https://ncar-cesm2-lens.s3-us-west-2.amazonaws.com/catalogs/aws-cesm2-le.json"
    )
    catalog = intake.open_esm_datastore(col_url)

    catalog_subset = catalog.search(variable=varname, frequency='daily', forcing_variant=forcing_variant)
    dsets = catalog_subset.to_dataset_dict(storage_options={'anon':True})

    ds_past = dsets[f"{component}.historical.daily.{forcing_variant}"]
    ds_future = dsets[f"{component}.ssp370.daily.{forcing_variant}"]

    ds_past_ns = (
        standardize(ds_past)
        .isel(member_id=members)
        .isel(time=np.isin(ds_past.time.dt.year, years["past"]))
        .sel(lon=slice(minlon, maxlon))
        .sel(lat=slice(minlat, maxlat))
    )
    ds_future_ns = (
        standardize(ds_future)
        .isel(member_id=members)
        .isel(time=np.isin(ds_future.time.dt.year, years["future"]))
        .sel(lon=slice(minlon, maxlon))
        .sel(lat=slice(minlat, maxlat))
    )
    if levels is not None and "lev" in ds_past_ns.dims:
        ds_past_ns = ds_past_ns.isel(lev=levels)
        ds_future_ns = ds_future_ns.isel(lev=levels)

    with ProgressBar():
        ds_past_ns = ds_past_ns.load()
    ds_past_ns.to_netcdf(out_path.joinpath(out_name_past))
    del ds_past_ns # free up memory

    with ProgressBar():
        ds_future_ns = ds_future_ns.load()
    ds_future_ns.to_netcdf(out_path.joinpath(out_name_future))
    
    
if __name__ == "__main__": 
    main()
""" 
this weird syntax is good practice, to distinguish python files meant to be run and 
those meant to provide functions for other scripts (i.e. files that are a part of a python module)
"""