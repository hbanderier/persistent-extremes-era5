import time
from itertools import pairwise
from pathlib import Path
from jetutils.definitions import DATADIR
import numpy as np 
from tqdm import tqdm
from urllib.request import urlretrieve
from urllib.error import HTTPError, ContentTooShortError

experiment_dict = {
    "past": "BHISTcmip6",
    "future": "BSSP370cmip6",
}
yearbounds = {
    "past": np.arange(1970, 2011, 10),
    "future": np.arange(2055, 2106, 10),
}
levels_dict = {
    "high_wind": list(range(14, 18)),
    "mid_wind": 20,
    "PRECL": "all",
    "TS": "all",
}
# yearbounds["past"][-1] = yearbounds["past"][-1] - 5
yearbounds["future"][-1] = yearbounds["future"][-1] - 4
timebounds = {key: [f"{year1}0101-{year2 - 1}1231" for year1, year2 in pairwise(val)] for key, val in yearbounds.items()}
years = {key: [list(range(max(year1, 2060) if key == "future" else year1, year2)) for year1, year2 in pairwise(val)] for key, val in yearbounds.items()}

members = [f"{year}.{str(number).zfill(3)}" for year, number in zip(range(1001, 1201, 20), range(1, 11))]
for startyear in [1231, 1251, 1281, 1301]:
    members.extend(f"{startyear}.{str(number).zfill(3)}" for number in range(1, 11))
    
members2 = [f"r{number}i{year}p1f1" for year, number in zip(range(1001, 1201, 20), range(1, 11))]
for startyear in [1231, 1251, 1281, 1301]:
    members2.extend(f"r{number}i{startyear}p1f1" for number in range(1, 11))
    
season = None
minlon = -80
maxlon = 40
minlat = 15
maxlat = 80
    
def get_url(varname: str, period: str, member: str, timebounds: str, minlon: float, maxlon: float, minlat: float, maxlat: float, year: int):
    experiment = experiment_dict[period]
    h = 6 if varname in ["U", "V", "T"] else 1

    base_url = f"https://tds.gdex.ucar.edu/thredds/ncss/grid/files/d651056/CESM2-LE/atm/proc/tseries/day_1/{varname}/b.e21.{experiment}.f09_g17.LE2-{member}.cam.h{h}.{varname}.{timebounds}.nc"
    specs = f"var={varname}&north={maxlat}&west={minlon}&east={maxlon}&south={minlat}&horizStride=1&time_start={year}-01-01T00:00:00Z&time_end={year}-12-31T00:00:00Z&&&accept=netcdf4ext"
    return f"{base_url}?{specs}"


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)
        
        
def _download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urlretrieve(url, filename=output_path, reporthook=t.update_to)
        

def download_url(url, output_path, retries = 10):
    while retries > 0:
        try:
            _download_url(url, output_path)
            return 
        except (HTTPError, ContentTooShortError):
            retries = retries - 1
            time.sleep(1.0)
            continue
    _download_url(url, output_path)


basepath = Path(f"{DATADIR}/CESM2/high_wind/ssp370")
scratchdir = Path("/storage/workspaces/giub_meteo_impacts/ci01/CESM2/high_wind/ssp370/raw")
var = "T"
period = "future"


for member1, member2 in zip(members, members2):
    for years_, tb in zip(years["future"], timebounds["future"]):
        for year in years_:
            url = get_url(var, period, member1, tb, minlon, maxlon, minlat, maxlat, year)
            scratchpath = scratchdir.joinpath(f"{member2}-{year}.nc")
            if scratchpath.is_file():
                continue
            download_url(url, scratchpath)