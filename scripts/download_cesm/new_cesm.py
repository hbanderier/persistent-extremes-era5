from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from jetstream_hugo.definitions import DATADIR, KAPPA, compute
from jetstream_hugo.data import standardize, flatten_by, extract
import numpy as np 
from netCDF4 import Dataset
import xarray as xr
import h5py
import argparse
import traceback


varname_to_search_dict = {
    "high_wind": ["U", "V", "T"],
    "mid_wind": ["U500", "V500"],
    "PRECL": "PRECL",
    "TS": "TS",
}
levels_dict = {
    "high_wind": list(range(14, 18)),
    "mid_wind": "all",
    "PRECL": "all",
    "TS": "all",
}
experiment_dict = {
    "past": "BHISTcmip6",
    "future": "BSSP370cmip6",
}
yearbounds = {
    "past": np.arange(1960, 2021, 10),
    "future": np.arange(2045, 2106, 10),
}
for key, val in yearbounds.items():
    yearbounds[key][-1] = val[-1] - 5
timebounds = {key: [f"{year1}0101-{year2 - 1}1231" for year1, year2 in pairwise(val)] for key, val in yearbounds.items()}

members = [f"{year}.{str(number).zfill(3)}" for year, number in zip(range(1001, 1201, 20), range(1, 11))]
for startyear in [1231, 1251, 1281, 1301]:
    members.extend(f"{startyear}.{str(number).zfill(3)}" for number in range(1, 11))
    
season = None
minlon = -180
maxlon = 180
minlat = 0
maxlat = 90
    
    
def get_url(varname: str, period: str, member: str, timebounds: str):
    experiment = experiment_dict[period]
    h = 6 if varname in ["U", "V", "T"] else 1

    return fr"https://tds.ucar.edu/thredds/fileServer/datazone/campaign/cgd/cesm/CESM2-LE/atm/proc/tseries/day_1/{varname}/b.e21.{experiment}.f09_g17.LE2-{member}.cam.h{h}.{varname}.{timebounds}.nc?api-token=ayhBFVYTOtGi2LM2cHDn6DjFCoKeCAqt69z8Ezt4#mode=bytes"


def downloader(varname: str | list, period: str, member: str, timebounds: str, odir: Path | str):
    opath = Path(odir).joinpath(f"{member}-{timebounds}.nc")
    if opath.is_file():
        return f"{member}, {timebounds} already existed"
    varname_to_search = varname_to_search_dict[varname]
    if isinstance(varname_to_search, Sequence):
        ds = xr.merge(
                [
                    xr.open_dataset(get_url(var, period, member, timebounds), engine="h5netcdf")[var]
                    for var in varname_to_search
                ]
            )
    else:
        ds = xr.open_dataset(get_url(varname_to_search, period, member, timebounds), engine="h5netcdf")[varname_to_search]
    ds = standardize(ds)
    ds = extract(
        ds,
        season=season,
        minlon=minlon,
        maxlon=maxlon,
        minlat=minlat,
        maxlat=maxlat,
        levels=levels_dict[varname],
        # members=[i],
    )
    if varname in ["high_wind", "mid_wind"]:
        ds["s"] = ds["s"] = np.sqrt(ds["u"] ** 2 + ds["v"] ** 2)
    ds = compute(ds, progress_flag=True, n_workers=1)
    if varname == "high_wind":
        ds = flatten_by(ds, "s")
        ds["theta"] = (1000 / ds["lev"]) ** KAPPA * ds["t"]
        ds = ds.drop_vars("t")
    for varname in (list(ds.data_vars) + ["lat", "lon"]):
        ds[varname] = ds[varname].astype(np.float32)
    ds = ds.to_netcdf(opath)
    return f"Retrieved {member}, {timebounds}"
    

def main():
    parser=argparse.ArgumentParser(description="sample argument parser")
    parser.add_argument("period", choices=["past", "future"])
    parser.add_argument("variable", choices=["high_wind", "mid_wind"])
    parser.add_argument("n_workers", default=10, type=int)
    args=parser.parse_args()
    period = args.period
    variable = args.variable   
    n_workers = args.n_workers   
    
    odir = Path(f"{DATADIR}/CESM2/{variable}/{period}")
    odir.mkdir(exist_ok=True)
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(downloader, variable, period, member, timebounds_, odir) for member in members for timebounds_ in timebounds[period]
        ]
        for f in as_completed(futures):
            try:
                print(f.result())
            except Exception as e:
                print(traceback.format_exc())
                print("could not retrieve")
                
if __name__ == "__main__":
    main()
