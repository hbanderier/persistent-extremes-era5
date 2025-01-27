from itertools import product
from pathlib import Path
import numpy as np
import xarray as xr
import intake
from jetstream_hugo.data import extract, flatten_by, standardize
from jetstream_hugo.definitions import DATADIR, compute, KAPPA
import argparse

varname_to_search_dict = {
    "high_wind": ["U", "V", "T"],
    "mid_wind": ["U", "V"],
}
levels_dict = {
    "high_wind": list(range(14, 18)),
    "mid_wind": [20],
}
experiment_dict = {
    "past": "historical",
    "future": "ssp370",
}
yearbounds_dict = {
    "past": (1964, 2015),
    "future": (2049, 2100),
}

parser=argparse.ArgumentParser(description="sample argument parser")
parser.add_argument("period", choices=["past", "future"])
parser.add_argument("variable", choices=["high_wind", "mid_wind", "PRECL"])
args=parser.parse_args()
period = args.period
variable = args.variable

url = 'https://raw.githubusercontent.com/NCAR/cesm2-le-aws/main/intake-catalogs/aws-cesm2-le.json'
component = "atm"
experiment = experiment_dict[period]
yearbounds = yearbounds_dict[period]
varname_to_search = varname_to_search_dict[variable]
levels = levels_dict.get(variable, None)
out_path = Path(f"{DATADIR}/CESM2/{variable}/{period}.zarr")
frequency = "daily"
forcing_variant = "cmip6"
season = None
minlon = -180
maxlon = 180
minlat = 0
maxlat = 90
members = "all"
reduce_da = True

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

if out_path.is_dir():
    ds_on_disk = xr.open_dataset(out_path)
    years_on_disk = np.unique(ds_on_disk.time.dt.year.values)
else:
    years_on_disk = []

for i, year in enumerate(range(*yearbounds)):
    if year in years_on_disk:
        continue
    ds_ = extract(
        ds,
        period=[year],
        season=season,
        minlon=minlon,
        maxlon=maxlon,
        minlat=minlat,
        maxlat=maxlat,
        levels=levels,
        # members=[i],
    )
    ds_["s"] = np.sqrt(ds_["u"] ** 2 + ds_["v"] ** 2)
    ds_ = compute(ds_, progress_flag=True)
    ds_ = flatten_by(ds_, "s")
    if variable == "high_wind":
        ds_["theta"] = (1000 / ds_["lev"]) ** KAPPA * ds_["t"]
        ds_ = ds_.drop_vars("t")
    for varname in (list(ds_.data_vars) + ["lat", "lon"]):
        ds_[varname] = ds_[varname].astype(np.float32)
    if i == 0:
        ds_.to_zarr(out_path, mode="w")
    else:
        ds_.to_zarr(out_path, append_dim="time")
    print(year)