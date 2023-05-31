import os
import platform

import numpy as np

try:
    import cupy as cp  # won't work on cpu nodes
except ImportError:
    pass
import pickle as pkl
from pathlib import Path
from typing import Any, Optional, Tuple, Union, Iterable

import pandas as pd
import xarray as xr
from cdo import Cdo
from nptyping import Float, Int, NDArray, Object, Shape

pf = platform.platform()
if pf.find("cray") >= 0:
    NODE = "DAINT"
    DATADIR = "/scratch/snx3000/hbanderi/data/persistent"
    N_WORKERS=16
    MEMORY_LIMIT='4GiB'
elif platform.node()[:4] == "clim":
    NODE = "CLIM"
    DATADIR = "/scratch2/hugo"
    N_WORKERS=8
    MEMORY_LIMIT='4GiB'
elif pf.find("el7") >= 0:  # find better later
    NODE = "UBELIX"
    DATADIR = "/storage/scratch/users/hb22g102"
    os.environ["CDO"] = "/storage/homefs/hb22g102/mambaforge/envs/env11/bin/cdo"
    N_WORKERS=16
    MEMORY_LIMIT='4GiB'
    
CLIMSTOR = "/mnt/climstor/ecmwf/era5/raw"

DATERANGEPL = pd.date_range("19590101", "20211231")
YEARSPL = np.unique(DATERANGEPL.year)
DATERANGEPL_SUMMER = DATERANGEPL[np.isin(DATERANGEPL.month, [6, 7, 8])]

DATERANGEPL_EXT = pd.date_range("19400101", "20221231")
YEARSPL_EXT = np.unique(DATERANGEPL_EXT.year)
DATERANGEPL_EXT_SUMMER = DATERANGEPL_EXT[np.isin(DATERANGEPL_EXT.month, [6, 7, 8])]

DATERANGEML = pd.date_range("19770101", "20211231")

WINDBINS = np.arange(0, 25, 0.5)
LATBINS = np.arange(15, 75.1, 0.5)
LONBINS = np.arange(-90, 30, 1)
DEPBINS = np.arange(-25, 25.1, 0.5)

ZOO = [
    "Lat",
    "Int",
    "Shar",
    "Lats",
    "Latn",
    "Tilt",
    "Lon",
    "Lonw",
    "Lone",
    "Dep",
    "Mea",
]

REGIONS = [
    "South", "West", "Balkans", "Scandinavia", "Russia", "Arctic"
]

SMALLNAME = {"Geopotential": "z", "Wind": "s", "Temperature": "t", "Precipitation": "tp"}  # Wind speed

RADIUS = 6.371e6  # m
OMEGA = 7.2921e-5  # rad.s-1
KAPPA = 0.2854
R_SPECIFIC_AIR = 287.0500676


def filenamesml(
    y: int, m: int, d: int
) -> str:  # Naming conventions of the files on climstor (why are they so different?)
    return [
        f"{CLIMSTOR}/ML/data/{str(y)}/P{str(y)}{str(m).zfill(2)}{str(d).zfill(2)}_{str(h).zfill(2)}"
        for h in range(0, 24, 6)
    ]


def filenamessfc(
    y: int, m: int, d: int
) -> str:  # Naming conventions of the files on climstor (why are they so different?)
    return [
        f"{CLIMSTOR}/SFC/data/{str(y)}/an_sfc_ERA5_{str(y)}-{str(m).zfill(2)}-{str(d).zfill(2)}.nc"
    ]


def filenamespl(y: int, m: int, d: int) -> str:
    return [
        f"{CLIMSTOR}/PL/data/an_pl_ERA5_{str(y)}-{str(m).zfill(2)}-{str(d).zfill(2)}.nc"
    ]  # returns iterable to have same call signature as filenamescl(y, m, d)


def filenamegeneric(y: int, m: int, folder: int) -> str:
    return [f"{DATADIR}/{folder}/{y}{str(m).zfill(2)}.nc"]


def _fn(date: pd.Timestamp, which: str) -> str:
    if which == "ML":
        return filenamesml(date.year, date.month, date.day)
    elif which == "PL":
        return filenamespl(date.year, date.month, date.day)
    elif which == "SFC":
        return filenamessfc(date.year, date.month, date.day)
    else:
        return filenamegeneric(date.year, date.month, which)


# instead takes pandas.timestamp (or iterable of _) as input
def fn(date: Union[list, NDArray, pd.DatetimeIndex, pd.Timestamp], which):
    if isinstance(date, (list, NDArray, pd.DatetimeIndex)):
        filenames = []
        for d in date:
            filenames.extend(_fn(d, which))
        return filenames
    elif isinstance(date, pd.Timestamp):
        return _fn(date, which)
    else:
        raise TypeError(f"Invalid type : {type(date)}")
    

def degcos(x: float) -> float:
    return np.cos(x / 180 * np.pi)


def degsin(x: float) -> float:
    return np.sin(x / 180 * np.pi)


def load_pickle(filename: str | Path) -> Any:
    with open(filename, "rb") as handle:
        to_ret = pkl.load(handle)
    return to_ret


def save_pickle(to_save: Any, filename: str | Path) -> None:
    with open(filename, "wb") as handle:
        pkl.dump(to_save, handle)


def CIequal(str1: str, str2: str) -> bool:
    """case-insensitive string equality check

    Args:
        str1 (str): first string
        str2 (str): second string

    Returns:
        bool: case insensitive string equality
    """
    return str1.casefold() == str2.casefold()


def hotspells_mask(filename: str = 'hotspells.csv', daysbefore: int = 21, daysafter: int = 5, timerange: NDArray | pd.DatetimeIndex | xr.DataArray = None, names: Iterable = None) -> xr.DataArray:
    """Returns timeseries mask of hotspells in several regions in `timerange` as a xr.DataArray with two dimensions and coordinates. It has shape (len(timerange), n_regions). n_regions is either inferred from the file or from the len of names if it is provided

    Args:
        filename (str, optional): path to hotspell center dates. Defaults to 'hotspells.csv'.
        daysbefore (int, optional): how many days before the center will the mask extend (inclusive). Defaults to 21.
        daysafter (int, optional): how many days after the center will the mask extend (inclusive). Defaults to 5.
        timerange (NDArray | pd.DatetimeIndex | xr.DataArray, optional): the time range to mask. Defaults to DATERANGEPL.
        names (Iterable, optional): names of the regions. See body for default values.

    Returns:
        xr.DataArray: at position (day, region) this is True if this day is part of a hotspell in this region
    """
    if names is None:
        names = REGIONS
    if timerange is None:
        timerange = DATERANGEPL
    else:
        try:
            timerange = timerange.values
        except AttributeError:
            pass
        timerange = pd.DatetimeIndex(timerange).floor(freq='1D')
    list_of_dates = np.loadtxt(filename, delimiter=",", dtype=np.datetime64)
    assert(len(names) == list_of_dates.shape[1])
    data = np.zeros((len(timerange), len(names)), dtype=bool)
    coords = {
        'time': timerange,
        'region': names
    }
    data = xr.DataArray(data, coords=coords)
    for i, dates in enumerate(list_of_dates.T):
        dates = np.sort(dates)
        dates = dates[
            ~(np.isnat(dates) | (np.datetime_as_string(dates, unit="Y") == "2022"))
        ]
        for date in dates:
            tsta = date - np.timedelta64(daysbefore, "D")
            tend = date + np.timedelta64(daysafter, "D")
            data.loc[tsta:tend, names[i]] = True
    return data


def get_hostpells_v2(filename: str = 'hotspells_v2.csv', lag_behind: int = 10, regions: list = None) -> list:
    if regions is None:
        regions = REGIONS
    hotspells_raw = pd.read_csv(filename)
    hotspells = []
    maxlen = 0
    maxnhs = 0
    for i, key in enumerate(regions):
        hotspells.append([])
        for line in hotspells_raw[f'dates{i + 1}']:
            if line == '-999':
                continue
            dateb, datee = [np.datetime64(d) for d in line.split('/')]
            dateb -= np.timedelta64(lag_behind, 'D')
            hotspells[-1].append(pd.date_range(dateb, datee, freq='1D'))
            maxlen = max(maxlen, len(hotspells[-1][-1]))
        maxnhs = max(maxnhs, len(hotspells[-1]))
    return hotspells, maxnhs, maxlen


def apply_hotspells_mask_v2(hotspells: list, da: xr.DataArray, maxlen: int = None, maxnhs: int = None, regions: list = None, lag_behind: int = 10) -> xr.DataArray:
    if regions is None:
        regions = REGIONS
    assert len(regions) == len(hotspells)
    if maxlen is None or maxnhs is None:
        maxlen = 0
        maxnhs = 0
        for region in hotspells:
            maxnhs = max(maxnhs, len(region))
            for hotspell in region:
                maxlen = max(maxlen, len(hotspell))
    data = np.zeros((da.shape[1], len(hotspells), maxnhs, maxlen))
    data[:] = np.nan
    da_masked = xr.DataArray(
        data, 
        coords={
            list(da.coords)[1]: np.arange(da.shape[1]), 'region': regions, 'hotspell': np.arange(maxnhs), 'day_after_beg': np.arange(maxlen) - lag_behind,
        },
    )
    for i, regionhs in enumerate(hotspells):
        for j, hotspell in enumerate(regionhs):
            try:
                da_masked[:, i, j, :len(hotspell)] = da.sel(time=hotspells[i][j].values).values.T
            except KeyError:
                ...
    return da_masked


def autocorrelation(path: Path, time_steps: int = 50) -> Path:
    ds = xr.open_dataset(path)
    name = path.parts[-1].split(".")[0]
    parent = path.parent
    autocorrs = {}
    for i, varname in enumerate(ds):
        if varname.split("_")[-1] == "climatology":
            continue
        autocorrs[varname] = ("lag", np.empty(time_steps))
        for j in range(time_steps):
            autocorrs[varname][1][j] = xr.corr(
                ds[varname], ds[varname].shift(time=j)
            ).values
    autocorrsda = xr.Dataset(autocorrs, coords={"lag": np.arange(time_steps)})
    opath = parent.joinpath(f"{name}_autocorrs.nc")
    autocorrsda.to_netcdf(opath)
    return opath  # a great swedish metal bEnd


def Hurst_exponent(path: Path, subdivs: int = 11) -> Path:
    ds = xr.open_dataset(path)
    subdivs = [2**n for n in range(11)]
    lengths = [len(ds.time) // n for n in subdivs]
    all_lengths = np.repeat(lengths, subdivs)
    N_chunks = np.sum(subdivs)
    Hurst = {}
    for i, varname in enumerate(ds.data_vars):
        adjusted_ranges = []
        for n_chunks, n in zip(subdivs, lengths):
            start = 0
            aranges = []
            for k in range(n_chunks):
                end = start + n
                series = ds[varname].isel(time=np.arange(start, end)).values
                mean = np.mean(series)
                std = np.std(series)
                series -= mean
                series = np.cumsum(series)
                raw_range = series.max() - series.min()
                aranges.append(raw_range / std)
            adjusted_ranges.append(np.mean(aranges))
        coeffs = np.polyfit(np.log(lengths), np.log(adjusted_ranges), deg=1)
        Hurst[varname] = [coeffs[0], np.exp(coeffs[1])]
    parent = path.parent
    name = path.parts[-1].split(".")[0]
    opath = parent.joinpath(f"{name}_Hurst.pkl")
    with open(opath, "wb") as handle:
        pkl.dump(Hurst, handle)
    return opath

def searchsortednd(a: NDArray, v: NDArray, **kwargs) -> NDArray:  # https://stackoverflow.com/questions/40588403/vectorized-searchsorted-numpy + reshapes
    orig_shapex, nx = v.shape[:-1], v.shape[-1]
    orig_shapea, na = a.shape[:-1], a.shape[-1]
    assert np.all(orig_shapex == orig_shapea)
    
    m = np.prod(orig_shapex)
    a = a.reshape(m, na)
    v = v.reshape(m, nx)
    max_num = np.maximum(np.nanmax(a) - np.nanmin(a), np.nanmax(v) - np.nanmin(v)) + 1
    r = max_num * np.arange(m)[:, None]
    p = np.searchsorted((a + r).ravel(), (v + r).ravel(), side="left", **kwargs).reshape(m, -1)
    return (p - na * (np.arange(m)[:, None])).reshape((*orig_shapex, -1))


def field_significance(to_test: xr.DataArray, take_from: NDArray | xr.DataArray, n_sam: int, n_sel: int = 100, thresh_up=True) -> Tuple[xr.DataArray, xr.DataArray]:
    indices = np.random.rand(n_sel, take_from.shape[0]).argpartition(n_sam, axis=1)[:, :n_sam]
    if isinstance(take_from, xr.DataArray):
        take_from = take_from.values
    empirical_distribution = np.mean(np.take(take_from, indices, axis=0), axis=1)
    nocorr = to_test > np.quantile(empirical_distribution, q=0.95, axis=0)

    # FDR correction
    idxs = np.argsort(empirical_distribution, axis=0)
    xcdf = np.take_along_axis(empirical_distribution, idxs, axis=0)
    ycdf = np.cumsum(idxs, axis=0) / np.sum(idxs, axis=0)
    ss = searchsortednd(xcdf.transpose((1, 2, 0)), to_test.values[:, :, None])
    ss[ss == n_sel] = n_sel - 1
    p = np.take_along_axis(ycdf.transpose((1, 2, 0)), ss, axis=-1).flatten()
    argp = np.argsort(p)
    p = p[argp]
    numvalid = len(p)
    bh_line = 0.1 * np.arange(1, numvalid + 1) / numvalid
    fdrcorr = np.zeros(len(p), dtype=bool)
    if thresh_up:
        above = ((1 - p)[::-1] < bh_line)[::-1]
        c = np.argmax(above)
        fdrcorr[argp[c:]] = True
    else:
        under = p < bh_line
        c = len(under) - np.argmax(under[::-1]) - 1
        fdrcorr[argp[:c]] = True
    fdrcorr = to_test.copy(data=fdrcorr.reshape(to_test.shape))
    return nocorr, fdrcorr


