# Configure file behaviour at the start of the main function !!

# import packages
from jetutils.definitions import DATADIR, compute
from jetutils.data import standardize, smooth
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
    for varname in ["PRECL", "TS"]:
        component = "atm" # for land variables like RAIN, "atm" for atmospheric variables like wind, and "ocn" for ocean variables
        forcing_variant = "cmip6" # other option is "smbb", which stands for "SMoothed Biomass Burning"
        out_path = Path(DATADIR, "CESM2", varname)
        minlon, maxlon, minlat, maxlat = None, None, 0, 90
        levels = None
        years = {
            "past": np.arange(1970, 2010),
            "future": np.arange(2060, 2100),
        }
        experiment_dict = {
            "past": "historical",
            "future": "ssp370",
        }

        col_url = (
            "https://ncar-cesm2-lens.s3-us-west-2.amazonaws.com/catalogs/aws-cesm2-le.json"
        )
        catalog = intake.open_esm_datastore(col_url)

        catalog_subset = catalog.search(variable=varname, frequency='daily', forcing_variant=forcing_variant)
        dsets = catalog_subset.to_dataset_dict(storage_options={'anon':True})
        
        for period in experiment_dict:

            ds = dsets[f"{component}.{experiment_dict[period]}.daily.{forcing_variant}"]

            ds = (
                standardize(ds)
                .reset_coords("time_bnds", drop=True)
                .squeeze()
                .isel(time=np.isin(ds.time.dt.year, years[period]))
                .sel(lon=slice(minlon, maxlon))
                .sel(lat=slice(minlat, maxlat))
            )
            if levels is not None and "lev" in ds.dims:
                ds = ds.isel(lev=levels)

            opath = out_path.joinpath(experiment_dict[period])
            opath.mkdir(parents=True, exist_ok=True)
            ds = ds[varname].drop_encoding()
            ds = ds.chunk({"time": 600})
            print()
            saved = ds.to_zarr(opath.joinpath("da.zarr"), compute=False, mode="w")
            with ProgressBar():
                saved.compute()
                
            ds = xr.open_zarr(opath.joinpath("da.zarr"))
        
            clim = ds.groupby("time.dayofyear").mean()
            clim = smooth(clim, {'dayofyear': ('win', 15)})
            clim = compute(clim, progress_flag=True)
            clim.to_zarr(opath.joinpath("clim.zarr"), mode="w")
            
            anom = ds.groupby("time.dayofyear") - clim
            anom = compute(anom, progress_flag=True)
            anom.to_zarr(opath.joinpath("anom.zarr"), mode="w")
    
    
if __name__ == "__main__": 
    main()
""" 
this weird syntax is good practice, to distinguish python files meant to be run and 
those meant to provide functions for other scripts (i.e. files that are a part of a python module)
"""