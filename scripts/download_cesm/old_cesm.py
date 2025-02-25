from itertools import product
from pathlib import Path
import numpy as np
import intake
from jetstream_hugo.data import extract, flatten_by, standardize
from jetstream_hugo.definitions import KAPPA, DATADIR, compute
import argparse

parser=argparse.ArgumentParser(description="sample argument parser")
parser.add_argument("year")
parser.add_argument("varname", choices=["high_wind", "mid_wind"])
args=parser.parse_args()
year = int(args.year)
varname = args.varname

varname_to_search_dict = {
    "high_wind": ["U", "V", "T"],
    "mid_wind": ["U", "V"],
    "PRECL": "PRECL",
    "TS": "TS",
}
levels_dict = {
    "high_wind": list(range(14, 18)),
    "mid_wind": 20,
    "PRECL": "all",
    "TS": "all",
}

url = 'https://raw.githubusercontent.com/NCAR/cesm2-le-aws/main/intake-catalogs/aws-cesm2-le.json'
component = "atm"
frequency = "daily"
forcing_variant = "cmip6"
varname_to_search = varname_to_search_dict[varname]
period = year
season = None
minlon = -80
maxlon = 40
minlat = 15
maxlat = 80
levels = levels_dict[varname]
members = "all"
reduce_da = True
experiment = "historical" if year < 2016 else "ssp370"
basepath = Path(f"{DATADIR}/CESM2/{varname}/{experiment}")
basepath.mkdir(exist_ok=True, parents=True)

indexers = [component, experiment, frequency, forcing_variant]
indexers = [np.atleast_1d(indexer) for indexer in indexers]
indexers = [[str(idx_) for idx_ in indexer] for indexer in indexers]
indexers = list(product(*indexers))
ensemble_keys = [".".join(indexer) for indexer in indexers]

catalog = intake.open_esm_datastore(url)
catalog_subset = catalog.search(
    variable=varname_to_search,
    component=component,
    experiment=experiment,
    frequency=frequency,
    forcing_variant=forcing_variant,
)
dsets = catalog_subset.to_dataset_dict(
    xarray_open_kwargs={"consolidated": False},
    storage_options={"anon": True}
)
ds = dsets[ensemble_keys[0]]
ds = standardize(ds)
ds = extract(
    ds,
    period=period,
    season=season,
    minlon=minlon,
    maxlon=maxlon,
    minlat=minlat,
    maxlat=maxlat,
    levels=levels,
    members=members,
)
if varname in ["high_wind", "mid_wind"]:
    ds["s"] = np.sqrt(ds["u"] ** 2 + ds["v"] ** 2)
for varname in (list(ds.data_vars) + ["lat", "lon"]):
    ds[varname] = ds[varname].astype(np.float32)

for member in ds.member.values:
    with open("invalid.txt", "r") as fn:
        invalids = fn.read().split("\n")
    if f"{member}-{year}" in invalids:
        continue
    opath = basepath.joinpath(f"{member}-{year}.nc")
    if opath.is_file():
        continue
    try:
        ds_ = compute(ds.sel(member=member, time=ds.time.dt.year==year), n_workers=2, progress_flag=False)
        ds_ = flatten_by(ds_, "s")
    except Exception: # I know I know. I think it can only be ValueError (flatten) or aiobotocore.response.AioReadTimeoutError (compute), but it takes too long to test
        with open("invalid.txt", "a") as fn:
            fn.write(f"{member}-{year}\n")
        continue
    if varname == "high_wind":
        ds_["theta"] = ((1000 / ds_["lev"]) ** KAPPA * ds_["t"]).astype(np.float32)
        ds_ = ds_.drop_vars("t")
    ds_.to_netcdf(opath)
    print(member, year)