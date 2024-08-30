from itertools import product
from pathlib import Path
import numpy as np
import intake
from dask.diagnostics import ProgressBar
from jetstream_hugo.data import extract, flatten_by
from jetstream_hugo.definitions import DATADIR
import argparse

parser=argparse.ArgumentParser(description="sample argument parser")
parser.add_argument("year")
args=parser.parse_args()
year = int(args.year)

url = 'https://raw.githubusercontent.com/NCAR/cesm2-le-aws/main/intake-catalogs/aws-cesm2-le.json'
varname = ["U", "V"]
varname_to_search = varname
basepath = Path(f"{DATADIR}/CESM2/results")
component = "atm"
experiment = "historical" if year < 2016 else "ssp370"
frequency = "daily"
forcing_variant = "cmip6"
period = year
season = None
minlon = -80
maxlon = 40
minlat = 15
maxlat = 80
levels = list(range(13, 19))
members = "all"
reduce_da = True

indexers = [component, experiment, frequency, forcing_variant]
indexers = [np.atleast_1d(indexer) for indexer in indexers]
indexers = [[str(idx_) for idx_ in indexer] for indexer in indexers]
indexers = list(product(*indexers))
ensemble_keys = [".".join(indexer) for indexer in indexers]

basepath = Path(f"{DATADIR}/CESM2/flat_wind")
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

ds["s"] = np.sqrt(ds["u"] ** 2 + ds["v"] ** 2)

for member in ds.member.values:
    with open("invalid.txt", "r") as fn:
        invalids = fn.read().split("\n")
    if f"{member}-{year}" in invalids:
        continue
    opath = basepath.joinpath(f"{member}-{year}.nc")
    if opath.is_file():
        continue
    ds_ = ds.sel(member=member, time=ds.time.dt.year==year).compute(n_workers=2)
    try:
        ds_ = flatten_by(ds_, "s")
    except ValueError: # some broken members
        with open("invalid.txt", "a") as fn:
            fn.write(f"{member}-{year}\n")
        continue
    ds_.to_netcdf(opath)
    print(member, year)
    
