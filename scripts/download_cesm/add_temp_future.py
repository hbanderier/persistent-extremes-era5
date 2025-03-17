import datetime
from itertools import pairwise
from pathlib import Path
from jetstream_hugo.definitions import DATADIR, compute
from jetstream_hugo.data import standardize
import numpy as np 
import xarray as xr

experiment_dict = {
    "past": "BHISTcmip6",
    "future": "BSSP370cmip6",
}
yearbounds = {
    "past": np.arange(1960, 2021, 10),
    "future": np.arange(2045, 2106, 10),
}
yearbounds["past"][-1] = yearbounds["past"][-1] - 5
yearbounds["future"][-1] = yearbounds["future"][-1] - 4
timebounds = {key: [f"{year1}0101-{year2 - 1}1231" for year1, year2 in pairwise(val)] for key, val in yearbounds.items()}

members = [f"{year}.{str(number).zfill(3)}" for year, number in zip(range(1001, 1201, 20), range(1, 11))]
for startyear in [1231, 1251, 1281, 1301]:
    members.extend(f"{startyear}.{str(number).zfill(3)}" for number in range(1, 11))
    
members2 = [f"r{number}i{year}p1f1" for year, number in zip(range(1001, 1201, 20), range(1, 11))]
for startyear in [1231, 1251, 1281, 1301]:
    members2.extend(f"r{number}i{startyear}p1f1" for number in range(1, 11))
    
season = None
minlon = -180
maxlon = 180
minlat = 0
maxlat = 90
    
    
def get_url(varname: str, period: str, member: str, timebounds: str):
    experiment = experiment_dict[period]
    h = 6 if varname in ["U", "V", "T"] else 1

    return fr"https://tds.ucar.edu/thredds/fileServer/datazone/campaign/cgd/cesm/CESM2-LE/atm/proc/tseries/day_1/{varname}/b.e21.{experiment}.f09_g17.LE2-{member}.cam.h{h}.{varname}.{timebounds}.nc?api-token=ayhBFVYTOtGi2LM2cHDn6DjFCoKeCAqt69z8Ezt4#mode=bytes"

basepath = Path(f"{DATADIR}/CESM2/high_wind/ssp370")
var = "T"
period = "future"
for member1, member2 in zip(members, members2):
    da = []
    for tb in timebounds["future"]:
        da.append(
            standardize(xr.open_dataset(
                get_url(var, period, member1,tb),            
                engine="h5netcdf"
            )[var])
        )
    da = xr.concat(da, "time")
    ds = xr.open_mfdataset(basepath.glob(f"{member2}-????.nc"))

    for coord in ["lon", "lat", "lev"]:
        da[coord] = da[coord].astype(np.float32)

    da["time"] = da.indexes["time"].to_datetimeindex(time_unit="us") + datetime.timedelta(hours=12)
    ds["time"] = ds.indexes["time"].to_datetimeindex(time_unit="us")
    da_ = da.sel(time=ds.time.values, lon=ds.lon.values, lat=ds.lat.values)
    da_ = compute(da_.sel(lev=ds["lev"]), progress_flag=True)
    da_.to_netcdf(basepath.joinpath(f"{member2}.nc"))
    print(member1, "done")
