from pathlib import Path
import numpy as np
from tqdm import trange
from jetstream_hugo.data import extract, flatten_by, standardize
from jetstream_hugo.definitions import DATADIR, compute
import argparse

import xarray as xr

ds = xr.open_zarr(
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
    chunks=None,
    storage_options=dict(token="anon"),
)
ar_full_37_1h = ds.sel(
    time=slice(ds.attrs["valid_time_start"], ds.attrs["valid_time_stop"])
)

base_ds = (
    ar_full_37_1h[["u_component_of_wind", "v_component_of_wind"]]
    .sel(
        time=ar_full_37_1h.time.dt.hour % 6 == 0,
        latitude=ar_full_37_1h.latitude >= 0,
        level=[175, 200, 225, 250, 300, 350],
    )
    .isel(longitude=slice(None, None, 2), latitude=slice(None, None, 2))
)

parser = argparse.ArgumentParser(description="sample argument parser")
parser.add_argument("year")
args = parser.parse_args()
year = int(args.year)

base_path = Path(f"{DATADIR}/ERA5/plev/flat_wind/dailymean")
for month in trange(1, 13):
    month_str = str(month).zfill(2)
    opath = base_path.joinpath(f"{year}{month_str}.nc")
    if opath.is_file():
        print(f"Already had {year}{month}")
        continue
    ds = _compute(
        base_ds.sel(
            time=(base_ds.time.dt.year == year) & (base_ds.time.dt.month == month)
        ),
        progress=True,
    )
    ds = standardize(ds)
    ds["s"] = np.sqrt(ds["u"] ** 2 + ds["v"] ** 2)
    ds = flatten_by(ds, "s")
    ds.to_netcdf(opath)
    print(f"Completed {year}{month}")
