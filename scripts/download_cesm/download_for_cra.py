# import packages
import intake
import numpy as np
import xarray as xr
from pathlib import Path
from dask.diagnostics import ProgressBar


## Configure
var = ["U", "V"]
outpath = Path("/storage/workspaces/giub_meteo_impacts/ci01/CESM2/for_cra")
minlon, maxlon, minlat, maxlat = -30, 40, 20, 80
## change this!


def standardize(da: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    """
    This section standardizes the names of things. You can choose to add or remove or modify this dict, it's form you in the end
    """
    standard_dict = {
        "member_id": "member",
        "U": "u",
        "u_component_of_wind": "u",
        "V": "v",
        "V500": "v",
        "v_component_of_wind": "v",
        "T": "t",
        "t2m": "t",
        "pt": "theta",
        "PRECL": "tp",
        "Z3": "z",
    }
    if isinstance(da, xr.Dataset):
        for key, value in standard_dict.items():
            if key in da:
                da = da.rename({key: value})
            else:
                pass
    elif isinstance(da, xr.DataArray):
        for key, value in standard_dict.items():
            if key in da.coords:
                da = da.rename({key: value})
            else:
                pass
    """
    This section standardizes the lon and lat grid to a format most people prefer
    """
    if (da.lon.max() > 180) and (da.lon.min() >= 0): # we prefer longitude to go from -180 to +180 rather than 0 to 360
        da = da.assign_coords(lon=(((da.lon + 180) % 360) - 180))
        da = da.sortby("lon")
    if np.diff(da.lat.values)[0] < 0: # and increasing latitudes
        da = da.reindex(lat=da.lat[::-1])
    """
    This section only transforms the data from 64 bits to 32 to save space. You don't lose much information doing this and save a lot of memory. Usually worth it!
    """
    if isinstance(da, xr.Dataset):
        for var in da.data_vars:
            if "chunksizes" in da[var].encoding and da[var].chunks is None:
                chunks = da[var].encoding["chunksizes"]
                chunks = chunks if chunks is not None else "auto"
                da[var] = da[var].chunk(chunks)
            if da[var].dtype == np.float64:
                da[var] = da[var].astype(np.float32)
            elif da[var].dtype == np.int64:
                da[var] = da[var].astype(np.int32)
    else:
        if "chunksizes" in da.encoding and da.chunks is None:
            chunks = da.encoding["chunksizes"]
            chunks = chunks if chunks is not None else "auto"
            da = da.chunk(chunks)
        if da.dtype == np.float64:
            da = da.astype(np.float32)
        elif da.dtype == np.int64:
            da = da.astype(np.int32)
    return da.unify_chunks()


def main():
    col_url = (
        "https://ncar-cesm2-lens.s3-us-west-2.amazonaws.com/catalogs/aws-cesm2-le.json"
    )
    catalog = intake.open_esm_datastore(col_url)

    catalog_subset = catalog.search(variable=var, frequency='daily', forcing_variant="cmip6")
    dsets = catalog_subset.to_dataset_dict(storage_options={'anon':True})

    ds_past = dsets["atm.historical.daily.cmip6"]
    ds_future = dsets["atm.ssp370.daily.cmip6"]

    ds_past_ns = (
        standardize(ds_past)
        .isel(member=[0, 1, 2, 3, 4, 5])
        .isel(time=np.isin(ds_past.time.dt.year, np.arange(1970, 2000)))
        .isel(lev=-1)
        .sel(lon=slice(minlon, maxlon))
        .sel(lat=slice(minlat, maxlat))
    )
    ds_future_ns = (
        standardize(ds_future)
        .isel(member=[0, 1, 2, 3, 4, 5])
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
    
    
if __name__ == "__main__":
    main()