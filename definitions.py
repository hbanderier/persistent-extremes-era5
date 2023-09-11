import os
import platform
from time import perf_counter

import numpy as np
try:
    import cupy as cp  # won't work on cpu nodes
except ImportError:
    pass
import pickle as pkl
from pathlib import Path
from itertools import combinations, combinations_with_replacement, permutations, product
from typing import Any, Optional, Tuple, Union, Iterable

from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics import pairwise_distances, roc_auc_score
from sklearn.metrics.pairwise import haversine_distances
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import pandas as pd
import xarray as xr
from skimage.filters import frangi
from tqdm import trange, tqdm
from cdo import Cdo
from nptyping import Float, Int, NDArray, Object, Shape
from multiprocessing import Pool
from functools import partial
from scipy.signal import find_peaks
from scipy.stats import norm
from numba import njit

pf = platform.platform()
if pf.find("cray") >= 0:
    NODE = "DAINT"
    DATADIR = "/scratch/snx3000/hbanderi/data/persistent"
    N_WORKERS = 16
    MEMORY_LIMIT = "4GiB"
elif platform.node()[:4] == "clim":
    NODE = "CLIM"
    DATADIR = "/scratch2/hugo"
    N_WORKERS = 8
    MEMORY_LIMIT = "4GiB"
elif pf.find("el7") >= 0:  # find better later
    NODE = "UBELIX"
    DATADIR = "/storage/scratch/users/hb22g102"
    os.environ["CDO"] = "/storage/homefs/hb22g102/mambaforge/envs/env11/bin/cdo"
    N_WORKERS = 8
    MEMORY_LIMIT = "4GiB"
else:
    NODE = "LOCAL"
    N_WORKERS = 8
    DATADIR = "../data"
    MEMORY_LIMIT = "2GiB"


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

REGIONS = ["S-W", "West", "S-E", "North", "East", "N-E"]

SMALLNAME = {
    "Geopotential": "z",
    "Wind": "s",
    "Temperature": "t",
    "Precipitation": "tp",
}  # Wind speed

PRETTIER_VARNAME = {
    'mean_lon': 'Avg. Longitude',
    'mean_lat': 'Avg. Latitude',
    'Lon': 'Lon. of max. speed',
    'Lat': 'Lat. of max. speed',
    'Spe': 'Max. speed',
    'lon_ext': 'Extent in lon.',
    'lat_ext': 'Extent in lat.',
    'tilt': 'Tilt',
    'int_over_europe': 'Int. speed over Europe',
    'int_all': 'Integrated speed',
    'persistence': 'Persistence',
}

LATEXY_VARNAME = {
    'mean_lon': '$\overline{\lambda}$',
    'mean_lat': '$\overline{\phi}$',
    'Lon': '$\lambda_{s^*}$',
    'Lat': '$\phi_{s^*}$',
    'Spe': '$s^*$',
    'lon_ext': '$\Delta \lambda$',
    'lat_ext': '$\Delta \phi$',
    'tilt': r'$\overline{\frac{\mathrm{d}\phi}{\mathrm{d}\lambda}}$',
    'int_over_europe': '$\int_{\mathrm{Eur.}} s \mathrm{d}\lambda$',
    'int_all': '$\int s \mathrm{d}\lambda$',
    'persistence': '$\Delta t$',
}

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


def hotspells_mask(
    filename: str = "hotspells.csv",
    daysbefore: int = 21,
    daysafter: int = 5,
    timerange: NDArray | pd.DatetimeIndex | xr.DataArray = None,
    names: Iterable = None,
) -> xr.DataArray:
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
        timerange = pd.DatetimeIndex(timerange).floor(freq="1D")
    list_of_dates = np.loadtxt(filename, delimiter=",", dtype=np.datetime64)
    assert len(names) == list_of_dates.shape[1]
    data = np.zeros((len(timerange), len(names)), dtype=bool)
    coords = {"time": timerange, "region": names}
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


def get_hostpells_v2(
    filename: str = "hotspells_v2.csv", lag_behind: int = 10, regions: list = None
) -> list:
    if regions is None:
        regions = REGIONS
    hotspells_raw = pd.read_csv(filename)
    hotspells = []
    maxlen = 0
    maxnhs = 0
    for i, key in enumerate(regions):
        hotspells.append([])
        for line in hotspells_raw[f"dates{i + 1}"]:
            if line == "-999":
                continue
            dateb, datee = [np.datetime64(d) for d in line.split("/")]
            dateb -= np.timedelta64(lag_behind, "D")
            hotspells[-1].append(pd.date_range(dateb, datee, freq="1D"))
            maxlen = max(maxlen, len(hotspells[-1][-1]))
        maxnhs = max(maxnhs, len(hotspells[-1]))
    return hotspells, maxnhs, maxlen


def apply_hotspells_mask_v2(
    hotspells: list,
    ds: xr.Dataset,
    maxlen: int = None,
    maxnhs: int = None,
    regions: list = None,
    lag_behind: int = 10,
) -> xr.Dataset:
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
    hotspell_length = np.zeros((len(regions), maxnhs))
    hotspell_length[:] = np.nan
    for i, hss in enumerate(hotspells):
        hotspell_length[i, :len(hss)] = [len(hs) - lag_behind for hs in hss]
    data = {}
    other_coord = list(ds.coords.items())[1]
    for varname in ds.data_vars:
        data[varname] = (
            (other_coord[0], 'region', 'hotspell', 'day_after_beg'), 
            np.zeros((ds[varname].shape[1], len(hotspells), maxnhs, maxlen))
        )
        data[varname][1][:] = np.nan
    ds_masked = xr.Dataset(
        data,
        coords={
            other_coord[0]: other_coord[1].values,
            "region": regions,
            "hotspell": np.arange(maxnhs),
            "day_after_beg": np.arange(maxlen) - lag_behind,
        },
    )
    for varname in ds.data_vars:
        for i, regionhs in enumerate(hotspells):
            for j, hotspell in enumerate(regionhs):
                try:
                    ds_masked[varname][:, i, j, :len(hotspell)] = ds[varname].sel(
                        time=hotspells[i][j].values
                    ).values.T
                except KeyError:
                    ...
    ds_masked = ds_masked.assign_coords({'hotspell_length': (('region', 'hotspell'), hotspell_length)})
    return ds_masked


def get_hotspell_mask(da_time: xr.DataArray | NDArray, num_lags: int = 1) -> xr.DataArray:
    hotspells = get_hostpells_v2(lag_behind=num_lags)[0]
    if isinstance(da_time, xr.DataArray):
        da_time = da_time.values
    hs_mask = np.zeros((len(da_time), len(REGIONS), num_lags))
    hs_mask = xr.DataArray(hs_mask, coords={'time': da_time, 'region': REGIONS, 'lag': np.arange(num_lags)})
    for i, region in enumerate(REGIONS):
        for hotspell in hotspells[i]:
            try:
                hs_mask.loc[hotspell[:10], region, np.arange(num_lags)] += np.eye(num_lags) * len(hotspell)
            except KeyError:
                ...
    return hs_mask


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


def infer_sym(to_plot: Any) -> bool:
    max = np.amax(to_plot)
    min = np.amin(to_plot)
    sym = (np.sign(max) == -np.sign(min)) and (
        np.abs(np.log10(np.abs(max)) - np.log10(np.abs(min))) <= 1
    )
    try:
        return sym.item()
    except AttributeError:
        return sym


def searchsortednd(
    a: NDArray, x: NDArray, **kwargs
) -> (
    NDArray
):  # https://stackoverflow.com/questions/40588403/vectorized-searchsorted-numpy + reshapes
    orig_shapex, nx = x.shape[1:], x.shape[0]
    _, na = a.shape[1:], a.shape[0]
    m = np.prod(orig_shapex)
    a = a.reshape(na, m)
    x = x.reshape(nx, m)
    max_num = np.maximum(np.nanmax(a) - np.nanmin(a), np.nanmax(x) - np.nanmin(x)) + 1
    r = max_num * np.arange(m)[None, :]
    p = (
        np.searchsorted((a + r).ravel(order="F"), (x + r).ravel(order="F"), side="left")
        .reshape(m, nx)
        .T
    )
    return (p - na * (np.arange(m)[None, :])).reshape((nx, *orig_shapex))


def fdr_correction(p: NDArray, q: float = 0.02):
    pshape = p.shape
    p = p.ravel()
    num_p = len(p)
    fdrcorr = np.zeros(num_p, dtype=bool)
    argp = np.argsort(p)
    p = p[argp]
    line_below = q * np.arange(num_p) / (num_p - 1)
    line_above = line_below + (1 - q)
    fdrcorr[argp] = (p >= line_above) | (p <= line_below)
    return fdrcorr.reshape(pshape)


def field_significance(
    to_test: xr.DataArray,
    take_from: NDArray | xr.DataArray,
    n_sel: int = 100,
    q: float = 0.02,
) -> Tuple[xr.DataArray, xr.DataArray]:
    n_sam = to_test.shape[0]
    indices = np.random.rand(n_sel, take_from.shape[0]).argpartition(n_sam, axis=1)[
        :, :n_sam
    ]
    if isinstance(take_from, xr.DataArray):
        take_from = take_from.values
    empirical_distribution = []
    cs = 500
    for ns in range(0, n_sam, cs):
        end = min(ns + cs, n_sam)
        empirical_distribution.append(
            np.mean(np.take(take_from, indices[:, ns:end], axis=0), axis=1)
        )
    sym = infer_sym(empirical_distribution)
    empirical_distribution = np.mean(empirical_distribution, axis=0)
    q = q / 2 if sym else q
    p = norm.cdf(
        to_test.mean(dim="time").values,
        loc=np.mean(empirical_distribution, axis=0),
        scale=np.std(empirical_distribution, axis=0),
    )
    nocorr = (p > (1 - q)) | (p < q)
    return nocorr, fdr_correction(p, q)


def one_ks_cumsum(b: NDArray, a: NDArray, q: float = 0.02, n_sam: int = None):
    if n_sam is None:
        n_sam = len(a)
    x = np.concatenate([a, b], axis=0)
    idxs_ks = np.argsort(x, axis=0)
    y1 = np.cumsum(idxs_ks < n_sam, axis=0) / n_sam
    y2 = np.cumsum(idxs_ks >= n_sam, axis=0) / n_sam
    d = np.amax(np.abs(y1 - y2), axis=0)
    p = np.exp(-(d**2) * n_sam)
    nocorr = (p < q).astype(int)
    return nocorr, fdr_correction(p, q)


def one_ks_searchsorted(b: NDArray, a: NDArray, q: float = 0.02, n_sam: int = None):
    if n_sam is None:
        n_sam = len(a)
    x = np.concatenate([a, b], axis=0)
    idxs_ks = np.argsort(x, axis=0)
    y1 = np.cumsum(idxs_ks < n_sam, axis=0) / n_sam
    y2 = np.cumsum(idxs_ks >= n_sam, axis=0) / n_sam
    d = np.amax(np.abs(y1 - y2), axis=0)
    p = np.exp(-(d**2) * n_sam)
    nocorr = (p < q).astype(int)
    return nocorr, fdr_correction(p, q)


def field_significance_v2(
    to_test: xr.DataArray,
    take_from: NDArray,
    n_sel: int = 100,
    q: float = 0.02,
    method: str = "cumsum",
    processes: int = N_WORKERS,
    chunksize: int = 2,
) -> Tuple[xr.DataArray, xr.DataArray]:
    # commented lines correspond to the slower searchsorted implementation. Currently using slightly less robust cumsum implementation
    nocorr = np.zeros((take_from.shape[1:]), dtype=int)
    fdrcorr = np.zeros((take_from.shape[1:]), dtype=int)
    a = to_test.values
    if method == "searchsorted":
        a = np.sort(a, axis=0)
        # b should be sorted as well but it's expensive to do it here, instead sort take_from before calling (since it's usually needed in many calls)
    n_sam = len(a)
    indices = np.random.rand(n_sel, take_from.shape[0]).argpartition(n_sam, axis=1)[
        :, :n_sam
    ]
    if method == "searchsorted":
        indices = np.sort(indices, axis=1)
        func = partial(one_ks_searchsorted, a=a, q=q, n_sam=n_sam)
    else:
        func = partial(one_ks_cumsum, a=a, q=q, n_sam=n_sam)

    with Pool(processes=processes) as pool:
        results = pool.map(
            func, (take_from[indices_] for indices_ in indices), chunksize=chunksize
        )
    nocorr, fdrcorr = zip(*results)
    nocorr = to_test[0].copy(data=np.sum(nocorr, axis=0) > (1 - q) * n_sel)
    fdrcorr = to_test[0].copy(data=np.sum(fdrcorr, axis=0) > (1 - q) * n_sel)
    return nocorr, fdrcorr


def labels_to_mask(labels: xr.DataArray | NDArray) -> NDArray:
    if isinstance(labels, xr.DataArray):
        labels = labels.values
    unique_labels = np.unique(labels)
    return labels[:, None] == unique_labels[None, :]


def compute_distance_matrix_(points, weights=None):
    if weights is None:
        weights = [1, 1]
    points_weighed = points[:, [1, 0]] * np.asarray(weights)[None, :]
    return haversine_distances(np.radians(points_weighed))


def find_jets_v2_(points, eps, weights=None, kind="AgglomerativeClustering"):
    dist_matrix = compute_distance_matrix_(points, weights=weights)
    if kind == "DBSCAN":
        model = DBSCAN(
            eps=eps,
            metric="precomputed",
        )
    elif kind == "AgglomerativeClustering":  # strategy pattern ?
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=eps,
            metric="precomputed",
            linkage="single",
        )
    labels = model.fit(dist_matrix).labels_
    masks = labels_to_mask(labels)
    return [points[mask] for mask in masks.T]


def update_jet(jet: NDArray):
    x, y, s = jet.T
    x_, c_ = np.unique(x, return_index=True)
    s_split = np.split(s, c_[1:])
    y_ = np.empty(len(s_split))
    s_ = np.empty(len(s_split))
    for i, sp in enumerate(s_split):
        j = np.argmax(sp) + c_[i]
        y_[i] = y[j]
        s_[i] = s[j]
    return np.stack([x_, y_, s_], axis=-1)


def find_jets_v4(
    X,
    lon: NDArray,
    lat: NDArray,
    eps: float = 5,
    height: float = 0.3,
    cutoff: int = 100,
) -> list:
    points = []
    res = lon[1] - lon[0]
    cutoff = cutoff / res
    X_norm = X / np.amax(X)
    X_prime = frangi(X_norm, black_ridges=False, sigmas=range(12, 23, 2), cval=1)
    for i, x in enumerate(X_prime.T):
        lo = lon[i]
        peaks = find_peaks(x, height=height, distance=10, width=1)[0]
        for peak in peaks:
            for plus in [-2, -1, 0, 1, 2]:
                try:
                    if x[peak + plus] > height:
                        points.append([lo, lat[peak + plus], x[peak + plus]])
                except IndexError:
                    pass
    for j, x in enumerate(X_prime):
        la = lat[j]
        peaks = find_peaks(x, height=height, distance=20000, width=5)[0]
        for peak in peaks:
            for plus in [-2, -1, 0, 1, 2]:
                try:
                    if x[peak + plus] > height:
                        points.append([lon[peak + plus], la, x[peak + plus]])
                except IndexError:
                    pass
    if len(points) == 0:
        return []
    points = np.atleast_2d(points)
    argsort = np.argsort(points[:, 0])
    points = points[argsort, :]
    eps = eps * np.radians(res)
    potential_jets = find_jets_v2_(points, eps=eps)

    jets = [jet for jet in potential_jets if np.sum(jet[:, 2]) > cutoff]
    sorted_order = np.argsort(
        [np.average(jet[:, 1], weights=jet[:, 2]) for jet in jets]
    )
    return [update_jet(jets[i]) for i in sorted_order]


def find_all_jets(
    da: xr.DataArray,
    X: NDArray = None,
    processes: int = N_WORKERS,
    chunksize: int = 10,
    **kwargs,
) -> Tuple[list, xr.DataArray]:
    lon = da.lon.values
    lat = da.lat.values
    func = partial(find_jets_v4, lon=lon, lat=lat, **kwargs)
    if X is None:
        X = da.values
    with Pool(processes=processes) as pool:
        alljets = list(tqdm(pool.imap(func, X, chunksize=chunksize)))
    return alljets


def jet_integral(jet: NDArray) -> float:
    return np.trapz(jet[:, 2])  # will fix the other one soon


def compute_jet_props(jets: list, eur_thresh: float = 19) -> Tuple[list, bool, bool]:
    props = []
    polys = []
    count = 1 if len(jets) > 0 else 0
    count_over_europe = 0
    for jet in jets:
        x, y, s = jet.T
        dic = {}
        dic["mean_lon"] = np.average(x, weights=s)
        dic["mean_lat"] = np.average(y, weights=s)
        dic["is_polar"] = dic["mean_lat"] > 45
        maxind = np.argmax(s)
        dic["Lon"] = x[maxind]
        dic["Lat"] = y[maxind]
        dic["Spe"] = s[maxind]
        dic["lon_ext"] = np.amax(x) - np.amin(x)
        dic["lat_ext"] = np.amax(y) - np.amin(y)
        p, r, _, _, _ = np.polyfit(x, y, w=s, deg=4, full=True)
        p = np.poly1d(p)
        polys.append((p, r))
        p = np.polyfit(x, y, w=s, deg=1)
        dic["tilt"] = p[0]
        try:
            dic["int_over_europe"] = jet_integral(jet[x > -10])
        except ValueError:
            dic["int_over_europe"] = 0
        dic["int_all"] = jet_integral(jet)
        count += np.any(
            [
                np.abs(other_dic["mean_lat"] - dic["mean_lat"]) > 15
                for other_dic in props
            ]
        )
        count_over_europe += dic["int_over_europe"] > eur_thresh
        props.append(dic)
    is_single = count == 1
    is_double = (count > 1) and (count_over_europe > 1)
    return props, is_double, is_single, polys


def compute_all_jet_props(
    all_jets: list,
    processes: int = N_WORKERS,
    chunk_size: int = 50,
    eur_thresh: float = 19,
) -> Tuple[list, NDArray, NDArray]:
    func = partial(compute_jet_props, eur_thresh=eur_thresh)
    with Pool(processes=processes) as pool:
        results = list(tqdm(pool.imap(func, all_jets, chunksize=chunk_size)))
    all_props, is_double, is_single, polys = zip(*results)
    return all_props, np.asarray(is_double), np.asarray(is_single), polys


def props_to_np(all_props: list, maxnjet: int = 2) -> NDArray:
    props_as_np = np.zeros((len(all_props), maxnjet, len(all_props[0][0])))
    for i, props in enumerate(all_props):
        for j, prop in enumerate(props):
            if j > 1:
                continue
            props_as_np[i, j, :] = [val for val in prop.values()]
    return props_as_np


def props_to_ds(all_props: list, time: NDArray | xr.DataArray = None, maxnjet: int = 3) -> xr.Dataset:
    if time is None:
        time = DATERANGEPL_SUMMER
    try:
        time_name = time.name
        time = time.values
    except AttributeError:
        time_name = 'time'
    assert len(time) == len(all_props)
    varnames = list(all_props[0][0].keys())
    ds = {}
    for varname in varnames:
        ds[varname] = ((time_name, 'jet'), np.zeros((len(time), maxnjet)))
        ds[varname][1][:] = np.nan
        for t in range(len(all_props)):
            for i in range(maxnjet):
                try:
                    props = all_props[t][i]
                except IndexError:
                    break
                ds[varname][1][t, i] = props[varname]
    ds = xr.Dataset(
        ds, 
        coords={time_name: time, 'jet': np.arange(maxnjet)}
    )
    return ds


def categorize_ds_jets(props_as_ds: xr.Dataset):
    time_name, time_val = list(props_as_ds.coords.items())[0]
    ds = xr.Dataset(coords={time_name: time_val, 'jet': ['subtropical', 'polar']})
    for varname in props_as_ds.data_vars:
        if varname == 'is_polar':
            continue
        cond = props_as_ds['is_polar']
        values = np.zeros((len(time_val), 2))
        values[:, 0] = props_as_ds[varname].where(1 - cond).mean(dim='jet').values
        values[:, 1] = props_as_ds[varname].where(cond).mean(dim='jet').values
        ds[varname] = ((time_name, 'jet'), values)
    return ds
    
    
def all_jets_to_one_array(all_jets: list):
    num_jets = [len(j) for j in all_jets]
    maxnjets = max(num_jets)
    num_indiv_jets = sum(num_jets)
    where_are_jets = np.full((len(all_jets), maxnjets, 2), fill_value=-1)
    all_jets_one_array = []
    k = 0
    l = 0
    for t, jets in enumerate(all_jets):
        for j, jet in enumerate(jets):
            l = k + len(jet)
            all_jets_one_array.append(jet)
            where_are_jets[t, j, :] = (k, l)
            k = l
    all_jets_one_array = np.concatenate(all_jets_one_array)
    return where_are_jets, all_jets_one_array


def one_array_to_all_jets(all_jets_one_array, where_are_jets):
    all_jets = []
    for where_are_jet in where_are_jets:
        all_jets.append([])
        for k, l in where_are_jet:
            all_jets[-1].append(all_jets_one_array[k:l])
    return all_jets


@njit
def isin(a, b):
    shape = a.shape
    a = a.ravel()
    n = len(a)
    result = np.full(n, False, dtype=np.bool_)
    set_b = set(b)
    for i in range(n):
        if a[i] in set_b:
            result[i] = True
    return result.reshape(shape)


@njit
def amin_ax0(a):
    result = np.zeros(a.shape[1])
    for i, a_ in enumerate(a.T):
        result[i] = np.amin(a_)
    return result


@njit
def amin_ax1(a):
    result = np.zeros(a.shape[0])
    for i, a_ in enumerate(a):
        result[i] = np.amin(a_)
    return result


@njit
def track_jets(all_jets_one_array, where_are_jets):
    factor: float = 0.2
    yearbreaks: int = 92
    guess_nflags: int = 1500
    all_jets_over_time = np.full(
        (guess_nflags, yearbreaks, 2), fill_value=len(where_are_jets), dtype=np.int32
    )
    last_valid_idx = np.full(guess_nflags, fill_value=yearbreaks, dtype=np.int32)
    for j in range(np.sum(where_are_jets[0, 0] >= 0)):
        all_jets_over_time[j, 0, :] = (0, j)
        last_valid_idx[j] = 0
    flags = np.full(where_are_jets.shape[:2], fill_value=guess_nflags, dtype=np.int32)
    last_flag = np.sum(where_are_jets[0, 0] >= 0) - 1
    flags[0, :last_flag + 1] = np.arange(last_flag + 1)
    for t, jet_idxs in enumerate(where_are_jets[1:]):  # can't really parallelize
        potentials = np.zeros(50, dtype=np.int32)
        from_ = max(0, last_flag - 30)
        times_to_test = np.take_along_axis(
            all_jets_over_time[from_ : last_flag + 1, :, 0],
            last_valid_idx[from_ : last_flag + 1, None],
            axis=1,
        ).flatten()
        potentials = (
            from_
            + np.where(
                isin(times_to_test, [t, t - 1, t - 2])
                & ((times_to_test // yearbreaks) == ((t + 1) // yearbreaks)).astype(
                    np.bool_
                )
            )[0]
        )
        num_valid_jets = np.sum(jet_idxs[:, 0] >= 0)
        dist_mat = np.zeros((len(potentials), num_valid_jets), dtype=np.float32)
        for i, jtt_idx in enumerate(potentials):
            t_jtt, j_jtt = all_jets_over_time[jtt_idx, last_valid_idx[jtt_idx]]
            k_jtt, l_jtt = where_are_jets[t_jtt, j_jtt]
            jet_to_try = all_jets_one_array[k_jtt:l_jtt, :2]
            for j in range(num_valid_jets):
                k, l = jet_idxs[j]
                jet = all_jets_one_array[k:l, :2]
                distances = np.sqrt(
                    np.sum(
                        (
                            np.radians(jet_to_try)[None, :, :]
                            - np.radians(jet)[:, None, :]
                        )
                        ** 2,
                        axis=-1,
                    )
                )
                # distances = haversine_distances(np.radians(jet_to_try), np.radians(jet))
                dist_mat[i, j] = np.mean(
                    np.array(
                        [
                            np.sum(amin_ax1(distances / len(jet_to_try))),
                            np.sum(amin_ax0(distances / len(jet))),
                        ]
                    )
                )
        connected_mask = dist_mat < factor
        flagged = np.zeros(num_valid_jets, dtype=np.bool_)
        for i, jtt_idx in enumerate(potentials):
            k_jtt, l_jtt = all_jets_over_time[jtt_idx, last_valid_idx[jtt_idx]]
            jet_to_try = all_jets_one_array[k_jtt:l_jtt, :2]
            js = np.argsort(dist_mat[i])
            for j in js:
                if not connected_mask[i, j]:
                    break
                if flagged[j]:
                    continue
                last_valid_idx[jtt_idx] = last_valid_idx[jtt_idx] + 1
                all_jets_over_time[jtt_idx, last_valid_idx[jtt_idx], :] = (t + 1, j)
                flagged[j] = True
                flags[t + 1, j] = jtt_idx
                break

        for j in range(num_valid_jets):
            if not flagged[j]:
                last_flag += 1
                all_jets_over_time[last_flag, 0, :] = (t + 1, j)
                last_valid_idx[last_flag] = 0
                flags[t + 1, j] = last_flag
                flagged[j] = True

    return all_jets_over_time, flags


def extract_props_over_time(jet, all_props):
    varnames = list(all_props[0][0].keys())
    props_over_time = {varname: np.zeros(len(jet)) for varname in varnames}
    for varname in varnames:
        for ti, (t, j) in enumerate(jet):
            props_over_time[varname][ti] = all_props[t][j][varname]
    return props_over_time


def add_persistence_to_props(ds_props: xr.Dataset, flags: NDArray):
    names = tuple(ds_props.coords.keys())
    num_jets = ds_props['mean_lon'].shape[1]
    jet_persistence_prop = flags[:, :num_jets].copy().astype(float)
    nan_flag = np.amax(flags)
    unique_flags, jet_persistence = np.unique(flags, return_counts=True)
    for i, flag in enumerate(unique_flags[:-1]):
        jet_persistence_prop[flags[:, :num_jets] == flag] = jet_persistence[i]
    jet_persistence_prop[flags[:, :num_jets] == nan_flag] = np.nan
    ds_props['persistence'] = (names, jet_persistence_prop)
    return ds_props
    
    
def comb_logistic_regression(y: NDArray, ds: xr.Dataset, all_combinations: list):
    coefs = np.zeros((len(all_combinations), len(all_combinations[0])))
    scores = np.zeros(len(all_combinations))
    for j, comb in enumerate(all_combinations):
        X = np.nan_to_num(np.stack([ds[varname][:, jet].values for varname, jet in comb], axis=1), nan=0)
        log = LogisticRegression().fit(X=X, y=y)
        coefs[j, :] = log.coef_[0]
        scores[j] = roc_auc_score(y, log.predict_proba(X)[:, 1])
    return coefs, scores


def all_logistic_regressions(ds: xr.Dataset, n_predictors: int, Y: xr.DataArray | NDArray):
    predictors = list(product(ds.data_vars, [0, 1]))
    all_combinations = list(combinations(predictors, n_predictors))
    func = partial(comb_logistic_regression, ds=ds, all_combinations=all_combinations)
    try:
        Y = Y.values
    except AttributeError:
        pass
    with Pool(processes=Y.shape[1]) as pool:
        results = list(tqdm(pool.imap(func, Y.T, chunksize=1)))
    coefs, scores = zip(*results)
    return np.stack(coefs, axis=0), np.stack(scores, axis=0)