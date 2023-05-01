import glob
import logging
import os
import platform
import time as timer

import contourpy
import numpy as np

try:
    import cupy as cp  # won't work on cpu nodes
except ImportError:
    pass
import pickle as pkl
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Optional, Tuple, Union, Iterable

import cartopy.crs as ccrs
import cartopy.feature as feat
import matplotlib as mpl
import pandas as pd
import xarray as xr
import xrft
from cdo import Cdo
from joblib import Parallel, delayed
from kmedoids import KMedoids
from matplotlib import cm
from matplotlib import path as mpath
from matplotlib import pyplot as plt
from matplotlib import ticker as mticker
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.colors import BoundaryNorm
from matplotlib.container import BarContainer
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from nptyping import Float, Int, NDArray, Object, Shape
from scipy import constants as co
from scipy import linalg
from scipy.stats import gaussian_kde
from scipy.stats import norm as normal_dist
from simpsom import SOMNet
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA as pca
from sklearn.metrics.pairwise import euclidean_distances

# logging.basicConfig(level=logging.DEBUG)
pf = platform.platform()
if pf.find("cray") >= 0:
    NODE = "DAINT"
    DATADIR = "/scratch/snx3000/hbanderi/data/persistent"
elif platform.node()[:4] == "clim":
    NODE = "CLIM"
    DATADIR = "/scratch/hugo"
elif pf.find("el7") >= 0:  # find better later
    NODE = "UBELIX"
    DATADIR = "/storage/scratch/users/hb22g102"
    os.environ["CDO"] = "/storage/homefs/hb22g102/mambaforge/envs/env11/bin/cdo"
CLIMSTOR = "/mnt/climstor/ecmwf/era5-new/raw"
cdo = Cdo()


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


RADIUS = 6.371e6  # m
OMEGA = 7.2921e-5  # rad.s-1
KAPPA = 0.2854
R_SPECIFIC_AIR = 287.0500676


def degcos(x: float) -> float:
    return np.cos(x / 180 * np.pi)


def degsin(x: float) -> float:
    return np.sin(x / 180 * np.pi)


DATERANGEPL = pd.date_range("19590101", "20211231")
YEARSPL = np.unique(DATERANGEPL.year)
DATERANGEML = pd.date_range("19770101", "20211231")
DATERANGEPL_SUMMER = DATERANGEPL[np.isin(DATERANGEPL.month, [6, 7, 8])]

WINDBINS = np.arange(0, 25, 0.5)
LATBINS = np.arange(15, 75.1, 0.5)
LONBINS = np.arange(-90, 30, 1)
DEPBINS = np.arange(-25, 25.1, 0.5)

COLORS5 = [  # https://coolors.co/palette/ef476f-ffd166-06d6a0-118ab2-073b4c
    "#ef476f",  # pinky red
    "#ffd166",  # yellow
    "#06d6a0",  # cyany green
    "#118ab2",  # light blue
    "#073b4c",  # dark blue
    "#F3722C",  # Orange
]

COLORS10 = [  # https://coolors.co/palette/f94144-f3722c-f8961e-f9844a-f9c74f-90be6d-43aa8b-4d908e-577590-277da1
    "#F94144",  # Vermilion
    "#F3722C",  # Orange
    "#F8961E",  # Atoll
    "#F9844A",  # Cadmium orange
    "#F9C74F",  # Caramel
    "#90BE6D",  # Lettuce green
    "#43AA8B",  # Bright Parrot Green
    "#4D908E",  # Abyss Green
    "#577590",  # Night Blue
    "#277DA1",  # Night Blue
]

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

COASTLINE = feat.NaturalEarthFeature(
    "physical", "coastline", "110m", edgecolor="black", facecolor="none"
)
BORDERS = feat.NaturalEarthFeature(
    "cultural",
    "admin_0_boundary_lines_land",
    "110m",
    edgecolor="grey",
    facecolor="none",
)

SMALLNAME = {"Geopotential": "z", "Wind": "s", "Temperature": "t"}  # Wind speed


def load_pickle(filename: str | Path) -> Any:
    with open(filename, "rb") as handle:
        to_ret = pkl.load(handle)
    return to_ret


def save_pickle(to_save: Any, filename: str | Path) -> None:
    with open(filename, "wb") as handle:
        pkl.dump(to_save, handle)


def make_boundary_path(
    minlon: float, maxlon: float, minlat: float, maxlat: float, n: int = 50
) -> mpath.Path:
    """Creates path to be used by GeoAxes.

    Args:
        minlon (float): minimum longitude
        maxlon (float): maximum longitude
        minlat (float): minimum latitude
        maxlat (float): maximum latitude
        n (int, optional): Interpolation points for each segment. Defaults to 50.

    Returns:
        boundary_path (mpath.Path): Boundary Path in flat projection
    """

    boundary_path = []
    # North (E->W)
    edge = [np.linspace(minlon, maxlon, n), np.full(n, maxlat)]
    boundary_path += [[i, j] for i, j in zip(*edge)]

    # West (N->S)
    edge = [np.full(n, maxlon), np.linspace(maxlat, minlat, n)]
    boundary_path += [[i, j] for i, j in zip(*edge)]

    # South (W->E)
    edge = [np.linspace(maxlon, minlon, n), np.full(n, minlat)]
    boundary_path += [[i, j] for i, j in zip(*edge)]

    # East (S->N)
    edge = [np.full(n, minlon), np.linspace(minlat, maxlat, n)]
    boundary_path += [[i, j] for i, j in zip(*edge)]

    boundary_path = mpath.Path(boundary_path)

    return boundary_path


def honeycomb_panel(
    ncol, nrow, ratio: int = None, subplot_kw: dict = None
) -> Tuple[Figure, NDArray[Any, Object]]:
    if ratio is None:
        fig = plt.figure(figsize=(20, 14))
    else:
        fig = plt.figure(figsize=(20, ratio * 20))
    gs = GridSpec(nrow, 2 * ncol + 1, hspace=0, wspace=0)
    axes = np.empty((ncol, nrow), dtype=object)
    if subplot_kw is None:
        subplot_kw = {}
    for i, j in product(range(ncol), range(nrow)):
        if j % 2 == 0:
            slice_x = slice(2 * i, 2 * i + 2)
        else:
            slice_x = slice(2 * i + 1, 2 * i + 2 + 1)
        axes[i, j] = fig.add_subplot(gs[nrow - j - 1, slice_x], **subplot_kw)
    return fig, axes


def clusterplot(
    nrow: int,
    ncol: int,
    to_plot: list,
    nlevels: int,
    cbar_extent: float,
    cbar_ylabel: str = None,
    clabels: Union[bool, list] = False,
    draw_labels: bool = False,
    lambert_projection: bool = False,
    cmap: str = "seismic",
    contours: bool = True,
    honeycomb: bool = False,
) -> Tuple[Figure, NDArray[Any, Object], Colorbar]:
    """Creates nice layout of plots with a common colorbar and color normalization

    Args:
        nrow (int): _description_
        ncol (int): _description_
        to_plot (list): _description_
        nlevels (int): _description_
        cbar_extent (float): _description_
        cbar_ylabel (str, optional): _description_. Defaults to None.
        clabels (bool, optional): _description_. Defaults to False.
        draw_labels: (bool, optional). Whether to draw labels. Defaults to False
        cmap (str, optional): _description_. Defaults to "seismic".

    Returns:
        fig (Figure): figure
        axes (npt.NDArray of Axes): axes
        cbar (Colorbar): colorbar
    """
    lon = to_plot[0]["lon"].values
    lat = to_plot[0]["lat"].values
    if lambert_projection:
        projection = ccrs.LambertConformal(
            central_longitude=np.mean(lon),
        )
    else:
        projection = ccrs.PlateCarree()
    if honeycomb:
        fig, axes = honeycomb_panel(
            nrow, ncol, None, subplot_kw={"projection": projection}
        )
    else:
        fig, axes = plt.subplots(
            nrow,
            ncol,
            figsize=(int(6.5 * ncol), int(4.5 * nrow)),
            subplot_kw={"projection": projection},
            constrained_layout=True,
        )
    if lambert_projection:
        extent = [np.amin(lon), np.amax(lon), np.amin(lat), np.amax(lat)]
        boundary = make_boundary_path(*extent)
    levels0 = np.delete(
        np.append(
            np.linspace(-cbar_extent, 0, nlevels), np.linspace(0, cbar_extent, nlevels)
        ),
        nlevels - 1,
    )
    levels = np.delete(levels0, nlevels - 1)
    cmap = mpl.colormaps[cmap]
    norm = BoundaryNorm(levels, cmap.N, extend="both")
    im = cm.ScalarMappable(norm=norm, cmap=cmap)
    axes = np.atleast_1d(axes).flatten()
    for i in range(len(to_plot)):
        axes[i].contourf(
            lon,
            lat,
            to_plot[i],
            transform=ccrs.PlateCarree(),
            levels=levels,
            cmap=cmap,
            norm=norm,
            extend="both",
        )
        if contours:
            cs = axes[i].contour(
                lon,
                lat,
                to_plot[i],
                transform=ccrs.PlateCarree(),
                levels=levels0,
                colors="k",
            )
        if lambert_projection:
            axes[i].set_boundary(boundary, transform=ccrs.PlateCarree())
        axes[i].add_feature(COASTLINE)
        axes[i].add_feature(BORDERS)
        if isinstance(clabels, bool) and clabels and contours:
            axes[i].clabel(cs)
        elif isinstance(clabels, list) and contours:
            axes[i].clabel(cs, levels=clabels)
        if draw_labels:
            gl = axes[i].gridlines(
                dms=False, x_inline=False, y_inline=False, draw_labels=True
            )
            gl.xlocator = mticker.FixedLocator([-90, -60, -30, 0, 30, 60])
            gl.ylocator = mticker.FixedLocator([20, 30, 40, 50, 60, 70])
            gl.xlines = (False,)
            gl.ylines = False
            plt.draw()
            for ea in gl.label_artists:
                current_pos = ea.get_position()
                if ea.get_text()[-1] in ["N", "S"]:
                    ea.set_visible(True)
                    continue
                if current_pos[1] > 4000000:
                    ea.set_visible(False)
                    continue
                ea.set_visible(True)
                ea.set_rotation(0)
                ea.set_position([current_pos[0], current_pos[1] - 200000])
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), spacing="proportional")
    cbar.ax.set_yticks(levels0)
    if cbar_ylabel is not None:
        cbar.ax.set_ylabel(cbar_ylabel)
    return fig, axes, cbar


def cdf(timeseries: Union[xr.DataArray, NDArray]) -> Tuple[NDArray, NDArray]:
    """Computes the cumulative distribution function of a 1D DataArray

    Args:
        timeseries (xr.DataArray or npt.NDArray): will be cast to ndarray if DataArray.

    Returns:
        x (npt.NDArray): x values for plotting,
        y (npt.NDArray): cdf of the timeseries,
    """
    if isinstance(timeseries, xr.DataArray):
        timeseries = timeseries.values
    idxs = np.argsort(timeseries)
    y = np.cumsum(idxs) / np.sum(idxs)
    x = timeseries[idxs]
    return x, y


# Create histogram
def compute_hist(
    timeseries: xr.DataArray, season: str = None, bins: Union[NDArray, list] = LATBINS
) -> Tuple[NDArray, NDArray]:
    """small wrapper for np.histogram that extracts a season out of xr.DataArray

    Args:
        timeseries (xr.DataArray): _description_
        season (str): _description_
        bins (list or npt.NDArray): _description_

    Returns:
        bins (npt.NDArray): _description_
        counts (npt.NDArray): _description_
    """
    if season is not None and season != "Annual":
        timeseries = timeseries.isel(time=timeseries.time.dt.season == season)
    return np.histogram(timeseries, bins=bins)


def histogram(
    timeseries: xr.DataArray,
    ax: Axes,
    season: str = None,
    bins: Union[NDArray, list] = LATBINS,
    **kwargs,
) -> BarContainer:
    """Small wrapper to plot a histogram out of a time series

    Args:
        timeseries (xr.DataArray): _description_
        ax (Axes): _description_
        season (str, optional): _description_. Defaults to None.
        bins (Union[NDArray, list], optional): _description_. Defaults to LATBINS.

    Returns:
        BarContainer: _description_
    """
    hist = compute_hist(timeseries, season, bins)
    midpoints = (hist[1][1:] + hist[1][:-1]) / 2
    bars = ax.bar(midpoints, hist[0], width=hist[1][1] - hist[1][0], **kwargs)
    return bars


def kde(
    timeseries: xr.DataArray,
    season: str = None,
    bins: Union[NDArray, list] = LATBINS,
    scaled: bool = False,
    return_x: bool = False,
    **kwargs,
) -> Union[NDArray, Tuple[NDArray, NDArray]]:
    hist = compute_hist(timeseries, season, bins)
    midpoints = (hist[1][1:] + hist[1][:-1]) / 2
    norm = (hist[1][1] - hist[1][0]) * np.sum(hist[0])
    kde: NDArray = gaussian_kde(midpoints, weights=hist[0], **kwargs).evaluate(
        midpoints
    )
    if scaled:
        kde *= norm
    if return_x:
        return midpoints, kde
    return kde


def compute_anomaly(
    da: xr.DataArray,
    return_clim: bool = False,
    smooth_kmax: int = None,
) -> (
    xr.DataArray | Tuple[xr.DataArray, xr.DataArray]
):  # https://github.com/pydata/xarray/issues/3575
    """computes daily anomalies extracted using a (possibly smoothed) climatology

    Args:
        da (xr.DataArray):
        return_clim (bool, optional): whether to also return the climatology (possibly smoothed). Defaults to False.
        smooth_kmax (bool, optional): maximum k for fourier smoothing of the climatology. No smoothing if None. Defaults to None.

    Returns:
        anom (DataArray): _description_
        clim (DataArray, optional): climatology
    """
    if len(da["time"]) == 0:
        return da
    gb = da.groupby("time.dayofyear")
    clim = gb.mean(dim="time")
    if smooth_kmax:
        ft = xrft.fft(clim, dim="dayofyear")
        ft[: int(len(ft) / 2) - smooth_kmax] = 0
        ft[int(len(ft) / 2) + smooth_kmax :] = 0
        clim = xrft.ifft(
            ft, dim="freq_dayofyear", true_phase=True, true_amplitude=True
        ).real.assign_coords(dayofyear=clim.dayofyear)
    anom = (gb - clim).reset_coords("dayofyear", drop=True)
    if return_clim:
        return anom, clim  # when Im not using map_blocks
    return anom


def figtitle(
    minlon: str,
    maxlon: str,
    minlat: str,
    maxlat: str,
    season: str,
) -> str:
    """Write extend of a region lon lat box in a nicer way, plus season

    Args:
        minlon (str): minimum longitude
        maxlon (str): maximum longitude
        minlat (str): minimum latitude
        maxlat (str): maximum latitude
        season (str): season  

    Returns:
        str: Nice title
    """
    minlon, maxlon, minlat, maxlat = (
        float(minlon),
        float(maxlon),
        float(minlat),
        float(maxlat),
    )
    title = f'${np.abs(minlon):.1f}°$ {"W" if minlon < 0 else "E"} - '
    title += f'${np.abs(maxlon):.1f}°$ {"W" if maxlon < 0 else "E"}, '
    title += f'${np.abs(minlat):.1f}°$ {"S" if minlat < 0 else "N"} - '
    title += f'${np.abs(maxlat):.1f}°$ {"S" if maxlat < 0 else "N"} '
    title += season
    return title


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
        names = ["South", "West", "Balkans", "Scandinavia", "Russia", "Arctic"]
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
        'regions': names
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


@dataclass(init=False)
class Experiment(object):
    dataset: str
    variable: str
    level: int | str
    region: str
    minlon: int
    maxlon: int
    minlat: int
    maxlat: int
    path: str

    def __init__(
        self,
        dataset: str,
        variable: str,
        level: int | str,
        region: Optional[str] = None,
        smallname: Optional[str] = None,
        minlon: Optional[int | float] = None,
        maxlon: Optional[int | float] = None,
        minlat: Optional[int | float] = None,
        maxlat: Optional[int | float] = None,
        smooth: bool = False,
    ):
        self.dataset = dataset
        self.variable = variable
        self.level = str(level)
        if smallname is None:
            self.smallname = SMALLNAME[variable]
        else:
            self.smallname = smallname
        if region is None:
            if np.any([bound is None for bound in [minlon, maxlon, minlat, maxlat]]):
                raise ValueError(
                    "Specify a region with either a string or 4 ints / floats"
                )
            self.region = f"box_{int(minlon)}_{int(maxlon)}_{int(minlat)}_{int(maxlat)}"
        else:
            self.region = region

        try:
            self.minlon, self.maxlon, self.minlat, self.maxlat = [
                int(bound) for bound in self.region.split("_")[-4:]
            ]
        except ValueError:
            if self.region == "dailymean":
                self.minlon, self.maxlon, self.minlat, self.maxlat = (
                    -180,
                    179.5,
                    -90,
                    90,
                )
            else:
                raise ValueError(f"{region=}, wrong specifier")
        self.path = Path(DATADIR, self.dataset, self.variable, self.level, self.region)
        self.copy_content(cdo, smooth)

    def ifile(self, suffix: str = "") -> Path:
        underscore = "" if suffix == "" else "_"
        joinpath = f"{self.smallname}{underscore}{suffix}.nc"
        return self.path.parent.joinpath("dailymean").joinpath(joinpath)

    def ofile(self, suffix: str = "") -> Path:
        underscore = "" if suffix == "" else "_"
        joinpath = f"{self.smallname}{underscore}{suffix}.nc"
        return self.path.joinpath(joinpath)

    def open_da(
        self, suffix: str = "", season: list | str | None = None, **kwargs
    ) -> xr.DataArray:
        da = xr.open_dataset(self.ofile(suffix), **kwargs)[self.smallname]
        try:
            da = da.rename({"longitude": "lon", "latitude": "lat"})
        except ValueError:
            pass
        if isinstance(season, list):
            da.isel(time=np.isin(da.time.dt.month, season))
        elif isinstance(season, str):
            if season in ["DJF", "MAM", "JJA", "SON"]:
                da = da.isel(time=da.time.dt.season == season)
            else:
                raise ValueError(
                    f"Wrong season specifier : {season} is not a valid xarray season"
                )
        if (da.time[1] - da.time[0]).values > np.timedelta64(6, "h"):
            da = da.assign_coords({"time": da.time.values.astype("datetime64[D]")})
        try:
            units = da.attrs["units"]
        except KeyError:
            return da
        if units == "m**2 s**-2":
            da /= co.g
            da.attrs["units"] = "m"
        return da

    def detrend(self, n_workers: int = 8):
        da = self.open_da(chunks={"time": -1, "lon": 20, "lat": 20})
        anom, clim = compute_anomaly(da, return_clim=True)
        anom = anom.compute(n_workers=n_workers)
        anom.to_netcdf(self.ofile("anomaly"))
        # clim = compute_climatology(da).compute(n_workers=n_workers)
        clim.to_netcdf(self.ofile("climatology"))
        anom = xrft.detrend(anom, "time", "linear").compute(n_workers=n_workers)
        anom.to_netcdf(self.ofile("detrended"))

    def get_winsize(self, da: xr.DataArray) -> Tuple[float, float, float]:
        resolution = (da.lon[1] - da.lon[0]).values.item()
        winsize = int(60 / resolution)
        halfwinsize = int(winsize / 2)
        return resolution, winsize, halfwinsize

    def smooth(self):
        if self.region == "dailymean":
            da = self.open_da()
            resolution, winsize, halfwinsize = self.get_winsize(da)
            da = da.pad(lon=halfwinsize, mode="wrap")
        else:
            da = self.open_da("bigger")
            resolution, winsize, halfwinsize = self.get_winsize(da)
        da = da.rolling(lon=winsize, center=True).mean()[:, :, halfwinsize:-halfwinsize]
        lon = da.lon
        da_fft = xrft.fft(da, dim="time")
        da_fft[np.abs(da_fft.freq_time) > 1 / 10 / 24 / 3600] = 0
        da = (
            xrft.ifft(da_fft, dim="freq_time", true_phase=True, true_amplitude=True)
            .real.assign_coords(time=da.time)
            .rename(self.smallname)
        )
        da.attrs["unit"] = "m/s"
        da["lon"] = lon
        da.to_netcdf(self.ofile("smooth"))

    def copy_content(self, cdo: Cdo, smooth: bool = False):
        if not self.path.is_dir():
            os.mkdir(self.path)
        ifile = self.ifile("")
        ofile = self.ofile("")
        if not ofile.is_file() and (not smooth and not self.region == "dailymean"):
            cdo.sellonlatbox(
                self.minlon,
                self.maxlon,
                self.minlat,
                self.maxlat,
                input=ifile.as_posix(),
                output=ofile.as_posix(),
            )
        elif not ofile.is_file() and smooth:
            ofile_bigger = self.ofile("bigger")
            cdo.sellonlatbox(
                self.minlon - 30,
                self.maxlon + 30,
                self.minlat,
                self.maxlat,
                input=ifile.as_posix(),
                output=ofile_bigger.as_posix(),
            )
            cdo.sellonlatbox(
                self.minlon,
                self.maxlon,
                self.minlat,
                self.maxlat,
                input=ofile_bigger.as_posix(),
                output=ofile.as_posix(),
            )
        to_iterate = ["detrended", "anomaly"]
        if smooth:
            to_iterate.append("smooth")
        for modified in to_iterate:
            ifile = self.ifile(modified)
            ofile = self.ofile(modified)
            if ofile.is_file():
                continue
            if ifile.is_file():
                cdo.sellonlatbox(
                    self.minlon,
                    self.maxlon,
                    self.minlat,
                    self.maxlat,
                    input=ifile.as_posix(),
                    output=ofile.as_posix(),
                )
                continue
            if modified in ["detrended", "anomaly"]:
                self.detrend()
            else:
                self.smooth()

    def to_absolute(
        self,
        da: xr.DataArray,
    ) -> (
        xr.DataArray
    ):  # TODO : deal with detrended anomalies, TODO : deal with other types of climatologies
        clim = self.open_da("climatology")
        return da.groupby("time.dayofyear") + clim


@dataclass(init=False)
class ClusteringExperiment(Experiment):
    midfix: str = "anomaly"
    season: list | str = None
    weigh: str = "sqrtcos"

    def __init__(
        self,
        dataset: str,
        variable: str,
        level: int | str,
        region: Optional[str] = None,
        smallname: Optional[str] = None,
        minlon: Optional[int | float] = None,
        maxlon: Optional[int | float] = None,
        minlat: Optional[int | float] = None,
        maxlat: Optional[int | float] = None,
        smooth: bool = False,
        midfix: str = "anomaly",
        season: list | str = None,
        weigh: str = "sqrtcos",
    ):
        super().__init__(
            dataset,
            variable,
            level,
            region,
            smallname,
            minlon,
            maxlon,
            minlat,
            maxlat,
            smooth,
        )
        self.midfix = midfix
        self.season = season
        if weigh is not None and weigh not in ["sqrtcos", "cos"]:
            raise ValueError(f"Wrong weigh specifier : {self.weigh}")
        self.weigh = weigh

    def prepare_for_clustering(self) -> Tuple[NDArray, xr.DataArray]:
        da = self.open_da(self.midfix, self.season)
        if CIequal(self.weigh, "sqrtcos"):
            da *= np.sqrt(degcos(da.lat))
        elif CIequal(self.weigh, "cos"):
            da *= degcos(da.lat)
        X = da.values.reshape(len(da.time), -1)
        return X, da

    def to_dataarray(
        self,
        centers: NDArray[Shape["*, *"], Float],
        da: xr.DataArray,
        n_pcas: Optional[int],
        coords: dict,
    ) -> xr.DataArray:
        centers = self.pca_inverse_transform(centers, n_pcas)

        shape = [len(coord) for coord in coords.values()]
        centers = xr.DataArray(centers.reshape(shape), coords=coords)
        if CIequal(self.weigh, "sqrtcos"):
            centers /= np.sqrt(degcos(da.lat))
        elif CIequal(self.weigh, "cos"):
            centers /= degcos(da.lat)
        return centers

    def compute_pcas(self, n_pcas: int, force: bool = False) -> str:
        glob_string = f"pca_*_{self.midfix}_{self.season}.pkl"
        logging.debug(glob_string)
        potential_paths = [
            Path(path) for path in glob.glob(self.path.joinpath(glob_string).as_posix())
        ]
        potential_paths = {
            path: int(path.parts[-1].split("_")[1]) for path in potential_paths
        }
        found = False
        logging.debug(potential_paths)
        for key, value in potential_paths.items():
            if value >= n_pcas:
                found = True
                break
        if found and not force:
            return key
        X, _ = self.prepare_for_clustering()
        pca_path = self.path.joinpath(f"pca_{n_pcas}_{self.midfix}_{self.season}.pkl")
        results = pca(n_components=n_pcas, whiten=True).fit(X)
        logging.debug(pca_path)
        with open(pca_path, "wb") as handle:
            pkl.dump(results, handle)
        return pca_path

    def pca_transform(
        self,
        X: NDArray[Shape["*, *"], Float],
        n_pcas: int = None,
    ) -> NDArray[Shape["*, *"], Float]:
        if n_pcas is not None:
            pca_path = self.compute_pcas(n_pcas)
            with open(pca_path, "rb") as handle:
                pca_results = pkl.load(handle)
            return pca_results.transform(X)[:, :n_pcas]
        return X

    def pca_inverse_transform(
        self,
        X: NDArray[Shape["*, *"], Float],
        n_pcas: int = None,
    ) -> NDArray[Shape["*, *"], Float]:
        if n_pcas is not None:
            pca_path = self.compute_pcas(n_pcas)
            with open(pca_path, "rb") as handle:
                pca_results = pkl.load(handle)
            diff_n_pcas = pca_results.n_components - n_pcas
            X = np.pad(X, [[0, 0], [0, diff_n_pcas]])
            return pca_results.inverse_transform(X)
        return X.reshape(X.shape[0], -1)

    def compute_opps(
        self,
        n_pcas: int = None,
        lag_max: int = 15,
        return_realspace: bool = False,
    ) -> Path | Tuple[NDArray, xr.DataArray, Path]:
        """
        I could GPU this fairly easily. Is it worth it tho ?
        """
        X, da = self.prepare_for_clustering()
        X = self.pca_transform(X, n_pcas)
        X = X.reshape((X.shape[0], -1))
        n_pcas = X.shape[1]
        opp_path: Path = self.path.joinpath(
            f"opp_{n_pcas}_{self.midfix}_{self.season}.pkl"
        )
        results = None
        if not opp_path.is_file():
            autocorrs = []
            for j in range(lag_max):
                autocorrs.append(
                    np.cov(X[j:], np.roll(X, j, axis=0)[j:], rowvar=False)[
                        n_pcas:, :n_pcas
                    ]
                )

            autocorrs = np.asarray(autocorrs)
            M = autocorrs[0] + np.sum(
                [autocorrs[i] + autocorrs[i].transpose() for i in range(1, lag_max)],
                axis=0,
            )

            invsqrtC0 = linalg.inv(linalg.sqrtm(autocorrs[0]))
            symS = invsqrtC0.T @ M @ invsqrtC0
            eigenvals, eigenvecs = linalg.eigh(symS)
            OPPs = autocorrs[0] @ (invsqrtC0 @ eigenvecs.T)
            idx = np.argsort(eigenvals)[::-1]
            eigenvals = eigenvals[idx]
            OPPs = OPPs[idx]
            results = {
                "eigenvals": eigenvals,
                "OPPs": OPPs,
            }
            with open(opp_path, "wb") as handle:
                pkl.dump(results, handle)
        if not return_realspace:
            return opp_path
        if results is None:
            with open(opp_path, "rb") as handle:
                results = pkl.load(handle)
                OPPs = results["OPPs"]
                eigenvals = results["eigenvals"]
        coords = {
            "OPP": np.arange(OPPs.shape[0]),
            "lat": da.lat.values,
            "lon": da.lon.values,
        }
        OPPs = self.to_dataarray(OPPs, da, n_pcas, coords)
        return eigenvals, OPPs, opp_path

    def opp_transform(
        self,
        X: NDArray[Shape["*, *"], Float],
        n_opps: int,
    ) -> NDArray[Shape["*, *"], Float]:
        opp_path = self.compute_opps(n_opps)
        with open(opp_path, "rb") as handle:
            opp_results = pkl.load(handle)
        if not X.shape[1] == n_opps:
            X = self.pca_transform(X, n_opps)
        OPPs = opp_results["OPPs"]
        X = X @ OPPs.T
        return X

    def opp_inverse_transform(
        self,
        X: NDArray[Shape["*, *"], Float],
        n_opps: int = None,
        to_realspace=False,
    ) -> NDArray[Shape["*, *"], Float]:
        opp_path = self.compute_opps(n_opps)
        with open(opp_path, "rb") as handle:
            opp_results = pkl.load(handle)
        OPPs = opp_results["OPPs"]
        X = X @ OPPs
        if to_realspace:
            return self.pca_inverse_transform(X)
        return X

    def cluster(
        self,
        n_clu: int,
        n_pcas: int = None,
        kind: str = "kmeans",
        return_centers: bool = True,
    ) -> str | Tuple[xr.DataArray, str]:
        X, da = self.prepare_for_clustering()
        X = self.pca_transform(X, n_pcas)
        if CIequal(kind, "kmeans"):
            results = KMeans(n_clu, n_init="auto")
            suffix = ""
        elif CIequal(kind, "kmedoids"):
            results = KMedoids(n_clu)
            suffix = "med"
        else:
            raise NotImplementedError(
                f"{kind} clustering not implemented. Options are kmeans and kmedoids"
            )
        picklepath = self.path.joinpath(
            f"k{suffix}_{n_clu}_{self.midfix}_{self.season}_{self.weigh}.pkl"
        )
        if picklepath.is_file():
            with open(picklepath, "rb") as handle:
                results = pkl.load(handle)
        else:
            results = results.fit(X)
            with open(picklepath, "wb") as handle:
                pkl.dump(results, handle)
        if not return_centers:
            return picklepath
        if isinstance(results, KMeans):
            centers = results.cluster_centers_
            coords = {
                "cluster": np.arange(centers.shape[0]),
                "lat": da.lat.values,
                "lon": da.lon.values,
            }
            centers = self.to_dataarray(da, n_pcas, centers, coords)
        elif isinstance(results, KMedoids):
            centers = (
                da.isel(time=results.medoids)
                .rename({"time": "cluster"})
                .assign_coords({"cluster": np.arange(len(results.medoids))})
            )
        return centers, picklepath

    def compute_som(
        self,
        nx: int,
        ny: int,
        n_pcas: int = None,
        OPP: bool = False,
        GPU: bool = False,
        return_centers: bool = False,
        train_kwargs: dict = None,
        **kwargs,
    ) -> SOMNet | Tuple[SOMNet, xr.DataArray]:
        if n_pcas is None and OPP:
            logging.warning("OPP flag will be ignored because n_pcas is set to None")

        output_path = self.path.joinpath(
            f"som_{nx}_{ny}_{self.midfix}_{self.season}_{self.weigh}{'_OPP' if OPP else ''}.npy"
        )
        if train_kwargs is None:
            train_kwargs = {}

        if output_path.is_file() and not return_centers:
            return output_path
        if OPP:
            X, da = self.prepare_for_clustering()
            X = self.opp_transform(X, n_opps=n_pcas)
        else:
            X, da = self.prepare_for_clustering()
            X = self.pca_transform(X, n_pcas=n_pcas)
        if GPU:
            try:
                X = cp.asarray(X)
            except NameError:
                GPU = False
        if output_path.is_file():
            net = SOMNet(nx, ny, X, GPU=GPU, PBC=True, load_file=output_path.as_posix())
        else:
            net = SOMNet(
                nx,
                ny,
                X,
                PBC=True,
                GPU=GPU,
                init="pca",
                **kwargs,
                # output_path=self.path.as_posix(),
            )
            net.train(**train_kwargs)
            net.save_map(output_path.as_posix())
        if not return_centers:
            return net
        centers = net._get(net.weights)
        logging.debug(centers)
        logging.debug(centers.shape)
        coords = {
            "x": np.arange(nx),
            "y": np.arange(ny),
            "lat": da.lat.values,
            "lon": da.lon.values,
        }
        if OPP:
            centers = self.opp_inverse_transform(centers, n_opps=n_pcas)
        centers = self.to_dataarray(centers, da, n_pcas, coords)
        return net, centers


def meandering(lines):
    m = 0
    for line in lines:  # typically few so a lopp is fine
        m += np.sum(np.sqrt(np.sum(np.diff(line, axis=0) ** 2, axis=1))) / 360
    return m


def one_ts(lon, lat, da):  # can't really vectorize this, have to parallelize
    m = []
    gen = contourpy.contour_generator(x=lon, y=lat, z=da)
    for lev in range(4900, 6205, 5):
        m.append(meandering(gen.lines(lev)))
    return np.amax(m)


@dataclass(init=False)
class ZooExperiment(object):
    def __init__(
        self,
        dataset: str,
        region: Optional[str],
        minlon: Optional[int | float],
        maxlon: Optional[int | float],
        minlat: Optional[int | float],
        maxlat: Optional[int | float],
    ):
        self.dataset = dataset
        self.exp_u = Experiment(
            dataset, "Wind", "Low", region, "u", minlon, maxlon, minlat, maxlat, True
        )
        self.exp_z = Experiment(
            dataset, "Geopotential", "500", region, None, minlon, maxlon, minlat, maxlat
        )
        self.region = self.exp_u.region
        self.minlon = self.exp_u.minlon
        self.maxlon = self.exp_u.maxlon
        self.minlat = self.exp_u.minlat
        self.maxlat = self.exp_u.maxlat
        self.da_wind = self.exp_u.open_da("smooth").squeeze().load()
        file_zonal_mean = self.exp_u.ofile("zonal_mean")
        if file_zonal_mean.is_file():
            self.da_wind_zonal_mean = xr.open_dataarray(file_zonal_mean)
        else:
            self.da_wind_zonal_mean = (
                xr.open_dataset(self.exp_u.ifile())[self.exp_u.smallname]
                .sel(lon=self.da_wind.lon, lat=self.da_wind.lat)
                .mean(dim="lon")
            )
            self.da_wind_zonal_mean.to_netcdf(file_zonal_mean)
        self.da_z = self.exp_z.open_da().squeeze()

    def compute_JLI(self) -> Tuple[xr.DataArray, xr.DataArray]:
        """Computes the Jet Latitude Index (also called Lat) as well as the wind speed at the JLI (Int)

        Args:

        Returns:
            Lat (xr.DataArray): Jet Latitude Index (see Woollings et al. 2010, Barriopedro et al. 2022)
            Int (xr.DataArray): Wind speed at the JLI (see Woollings et al. 2010, Barriopedro et al. 2022)
        """
        da_Lat = self.da_wind_zonal_mean
        LatI = da_Lat.argmax(dim="lat", skipna=True)
        self.Lat = xr.DataArray(
            da_Lat.lat[LatI.values.flatten()].values, coords={"time": da_Lat.time}
        ).rename("Lat")
        self.Lat.attrs["units"] = "degree_north"
        self.Int = da_Lat.isel(lat=LatI).reset_coords("lat", drop=True).rename("Int")
        self.Int.attrs["units"] = "m/s"
        return self.Lat, self.Int

    def compute_Shar(self) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
        """Computes sharpness and south + north latitudinal extent of the jet

        Args:

        Returns:
            Shar (xr.DataArray): Sharpness (see Woollings et al. 2010, Barriopedro et al. 2022)
            Lats (xr.DataArray): Southward latitudinal extent of the jet (see Woollings et al. 2010, Barriopedro et al. 2022)
            Latn (xr.DataArray): Northward latitudinal extent of the jet (see Woollings et al. 2010, Barriopedro et al. 2022)
        """
        da_Lat = self.da_wind_zonal_mean
        self.Shar = (self.Int - da_Lat.mean(dim="lat")).rename("Shar")
        self.Shar.attrs["units"] = self.Int.attrs["units"]
        difference_with_shar = da_Lat - self.Shar / 2
        roots = np.where(
            difference_with_shar.values[:, 1:] * difference_with_shar.values[:, :-1] < 0
        )
        hist = np.histogram(roots[0], bins=np.arange(len(da_Lat.time) + 1))[0]
        cumsumhist = np.append([0], np.cumsum(hist)[:-1])
        self.Lats = xr.DataArray(
            da_Lat.lat.values[roots[1][cumsumhist]],
            coords={"time": da_Lat.time},
            name="Lats",
        )
        self.Latn = xr.DataArray(
            da_Lat.lat.values[roots[1][cumsumhist + hist - 1]],
            coords={"time": da_Lat.time},
            name="Latn",
        )
        self.Latn[self.Latn < self.Lat] = da_Lat.lat[-1]
        self.Lats[self.Lats > self.Lat] = da_Lat.lat[0]
        self.Latn.attrs["units"] = "degree_north"
        self.Lats.attrs["units"] = "degree_north"
        return self.Shar, self.Lats, self.Latn

    def compute_Tilt(self) -> Tuple[xr.DataArray, xr.DataArray]:
        """Computes tilt and also returns the tracked latitudes

        Args:

        Returns:
            tuple[xr.DataArray, xr.DataArray]: _description_
        """
        self.trackedLats = (
            self.da_wind.isel(lat=0)
            .copy(data=np.zeros(self.da_wind.shape[::2]))
            .reset_coords("lat", drop=True)
            .rename("Tracked Latitudes")
        )
        self.trackedLats.attrs["units"] = "degree_north"
        lats = self.da_wind.lat.values
        twodelta = lats[2] - lats[0]
        midpoint = int(len(self.da_wind.lon) / 2)
        self.trackedLats[:, midpoint] = self.Lat
        iterator = zip(
            reversed(range(midpoint)), range(midpoint + 1, len(self.da_wind.lon))
        )
        for lonw, lone in iterator:
            for k, thislon in enumerate((lonw, lone)):
                otherlon = thislon - (
                    2 * k - 1
                )  # previous step in the iterator for either east (k=1, otherlon=thislon-1) or west (k=0, otherlon=thislon+1)
                mask = (
                    np.abs(
                        self.trackedLats[:, otherlon].values[:, None] - lats[None, :]
                    )
                    > twodelta
                )
                # mask = where not to look for a maximum. The next step (forward for east or backward for west) needs to be within twodelta of the previous (otherlon)
                da_wind_at_thislon = self.da_wind.isel(lon=thislon).values
                here = np.ma.argmax(
                    np.ma.array(da_wind_at_thislon, mask=mask),
                    axis=1,
                )
                self.trackedLats[:, thislon] = lats[here]
        self.Tilt = (
            self.trackedLats.polyfit(dim="lon", deg=1)
            .sel(degree=1)["polyfit_coefficients"]
            .reset_coords("degree", drop=True)
            .rename("Tilt")
        )
        self.Tilt.attrs["units"] = "degree_north/degree_east"
        return self.trackedLats, self.Tilt

    def compute_Lon(self) -> Tuple[xr.DataArray, xr.DataArray]:
        """_summary_

        Args:

        Returns:
            tuple[xr.DataArray, xr.DataArray]: _description_
        """
        self.Intlambda = self.da_wind.sel(lat=self.trackedLats).reset_coords(
            "lat", drop=True
        )
        Intlambdasq = self.Intlambda * self.Intlambda
        lons = xr.DataArray(
            self.da_wind.lon.values[None, :] * np.ones(len(self.da_wind.time))[:, None],
            coords={"time": self.da_wind.time, "lon": self.da_wind.lon},
        )
        self.Lon = (lons * Intlambdasq).sum(dim="lon") / Intlambdasq.sum(dim="lon")
        self.Lon.attrs["units"] = "degree_east"
        self.Lon = self.Lon.rename("Lon")
        return self.Intlambda, self.Lon

    def compute_Lonew(self) -> Tuple[xr.DataArray, xr.DataArray]:
        """_summary_

        Args:

        Returns:
            tuple[xr.DataArray, xr.DataArray]: _description_
        """
        Intlambda = self.Intlambda.values
        Mean = np.mean(Intlambda, axis=1)
        lon = self.da_wind.lon.values
        iLon = np.argmax(lon[None, :] - self.Lon.values[:, None] > 0, axis=1)
        basearray = Intlambda - Mean[:, None] < 0
        iLonw = (
            np.ma.argmin(
                np.ma.array(basearray, mask=lon[None, :] > self.Lon.values[:, None]),
                axis=1,
            )
            - 1
        )
        iLone = (
            np.ma.argmax(
                np.ma.array(basearray, mask=lon[None, :] <= self.Lon.values[:, None]),
                axis=1,
            )
            - 1
        )
        self.Lonw = xr.DataArray(
            lon[iLonw], coords={"time": self.da_wind.time}, name="Lonw"
        )
        self.Lone = xr.DataArray(
            lon[iLone], coords={"time": self.da_wind.time}, name="Lone"
        )
        self.Lonw.attrs["units"] = "degree_east"
        self.Lone.attrs["units"] = "degree_east"
        return self.Lonw, self.Lone

    def compute_Dep(self) -> xr.DataArray:
        """_summary_

        Args:

        Returns:
            xr.DataArray: _description_
        """
        phistarl = xr.DataArray(
            self.da_wind.lat.values[self.da_wind.argmax(dim="lat").values],
            coords={"time": self.da_wind.time.values, "lon": self.da_wind.lon.values},
        )
        self.Dep = (
            np.sqrt((phistarl - self.trackedLats) ** 2).sum(dim="lon").rename("Dep")
        )
        self.Dep.attrs["units"] = "degree_north"
        return self.Dep

    def compute_Mea(self, njobs: int = 8) -> xr.DataArray:
        lon = self.da_z.lon.values
        lat = self.da_z.lat.values
        self.Mea = Parallel(
            n_jobs=njobs, backend="loky", max_nbytes=1e6, verbose=0, batch_size=50
        )(
            delayed(one_ts)(lon, lat, self.da_z.sel(time=t).values)
            for t in self.da_z.time[:]
        )
        self.Mea = xr.DataArray(self.Mea, coords={"time": self.da_z.time}, name="Mea")
        return self.Mea

    def get_Zoo_path(self) -> str:
        return self.exp_u.path.joinpath("Zoo.nc")

    def compute_Zoo(self, detrend=False) -> str:
        logging.debug("Lat")
        _ = self.compute_JLI()
        logging.debug("Shar")
        _ = self.compute_Shar()
        logging.debug("Tilt")
        _ = self.compute_Tilt()
        logging.debug("Lon")
        _ = self.compute_Lon()
        logging.debug("Lonew")
        _ = self.compute_Lonew()
        logging.debug("Dep")
        _ = self.compute_Dep()
        logging.debug("Mea")
        _ = self.compute_Mea()

        Zoo = xr.Dataset(
            {
                "Lat": self.Lat,
                "Int": self.Int,
                "Shar": self.Shar,
                "Lats": self.Lats,
                "Latn": self.Latn,
                "Tilt": self.Tilt,
                "Lon": self.Lon,
                "Lonw": self.Lonw,
                "Lone": self.Lone,
                "Dep": self.Dep,
                "Mea": self.Mea,
            }
        ).dropna(
            dim="time"
        )  # dropna if time does not match between z and u (happens for NCEP)
        self.Zoopath = self.get_Zoo_path()
        if not detrend:
            Zoo.to_netcdf(self.Zoopath)
            return self.Zoopath
        for key, value in Zoo.data_vars.items():
            Zoo[f"{key}_anomaly"], Zoo[f"{key}_climatology"] = compute_anomaly(
                value, return_clim=1, smooth_kmax=3
            )
            Zoo[f"{key}_detrended"] = xrft.detrend(
                Zoo[f"{key}_anomaly"], dim="time", detrend_type="linear"
            )
        Zoo.to_netcdf(self.Zoopath)
        return self.Zoopath


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
