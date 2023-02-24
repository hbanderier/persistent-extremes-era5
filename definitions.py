import os
import glob
import platform
import numpy as np
import pandas as pd
import xarray as xr
import xrft
import pickle as pkl
import scipy.constants as co
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.path as mpath
from matplotlib.colors import BoundaryNorm
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.container import BarContainer
import cartopy.crs as ccrs
import cartopy.feature as feat
from scipy.stats import gaussian_kde, norm as normal_dist
from typing import Union, Any
from nptyping import NDArray, Object, Shape
from sklearn.cluster import KMeans


pf = platform.platform()
if pf.find("cray") >= 0:
    NODE = "daint"
elif platform.node()[:4] == "clim":
    NODE = "CLIM"
else:  # find better later
    NODE = "UBELIX"
DATADIR = (
    "/scratch/snx3000/hbanderi/data/persistent" if NODE == "daint" else "/scratch2/hugo"
)
CLIMSTOR = "/mnt/climstor/ecmwf/era5/raw"


def filenamesml(
    y: str, m: str, d: str
) -> str:  # Naming conventions of the files on climstor (why are they so different?)
    return [
        f"{CLIMSTOR}/ML/data/{str(y)}/P{str(y)}{str(m).zfill(2)}{str(d).zfill(2)}_{str(h).zfill(2)}"
        for h in range(0, 24, 6)
    ]


def filenamessfc(
    y: str, m: str, d: str
) -> str:  # Naming conventions of the files on climstor (why are they so different?)
    return [
        f"{CLIMSTOR}/SFC/data/an_sfc_ERA5_{str(y)}-{str(m).zfill(2)}-{str(d).zfill(2)}.nc"
    ]


def filenamespl(
    y: str, m: str, d: str
) -> str:
    return [
        f"{CLIMSTOR}/PL/data/an_pl_ERA5_{str(y)}-{str(m).zfill(2)}-{str(d).zfill(2)}.nc"
    ]  # returns iterable to have same call signature as filenamescl(y, m, d)


def filenamegeneric(
    y: str, m: str, folder: str
) -> str:
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


def fn(date: Union[list, NDArray, pd.DatetimeIndex, pd.Timestamp], which):  # instead takes pandas.timestamp (or iterable of _) as input
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
WINDBINS = np.arange(0, 25, 2)
LATBINS = np.arange(15, 75, 2.5)
LONBINS = np.arange(-90, 30, 3)
DEPBINS = np.arange(-25, 26, 1.5)

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
    "physical", "coastline", "10m", edgecolor="black", facecolor="none"
)
BORDERS = feat.NaturalEarthFeature(
    "cultural",
    "admin_0_boundary_lines_land",
    "10m",
    edgecolor="grey",
    facecolor="none",
)

SMALLNAME = {
    "Geopotential": "z",
    "Wind": "s",  # Wind speed
}


def clusterplot(
    nrow: int,
    ncol: int,
    to_plot: list,
    nlevels: int,
    cbar_extent: float,
    cbar_ylabel: str = None,
    clabels: bool = False,
    cmap: str = "seismic",
) -> tuple[Figure, NDArray[Any, Object], Colorbar]:
    """Creates nice layout of plots with a common colorbar and color normalization

    Args:
        nrow (int): _description_
        ncol (int): _description_
        to_plot (list): _description_
        nlevels (int): _description_
        cbar_extent (float): _description_
        cbar_ylabel (str, optional): _description_. Defaults to None.
        clabels (bool, optional): _description_. Defaults to False.
        cmap (str, optional): _description_. Defaults to "seismic".

    Returns:
        fig (Figure): figure
        axes (npt.NDArray of Axes): axes
        cbar (Colorbar): colorbar
    """
    lon = to_plot[0]["lon"].values
    lat = to_plot[0]["lat"].values
    projection = ccrs.LambertConformal(
        central_longitude=np.mean(lon),
    )
    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(int(6.5 * ncol), int(4.5 * nrow)),
        subplot_kw={"projection": projection},
        constrained_layout=True,
    )
    extent = [np.amin(lon), np.amax(lon), np.amin(lat), np.amax(lat)]
    boundary = make_boundary_path(*extent)
    levels0 = np.delete(
        np.append(
            np.linspace(-cbar_extent, 0, nlevels), np.linspace(0, cbar_extent, nlevels)
        ),
        nlevels - 1,
    )
    levels = np.delete(levels0, nlevels - 1)
    cmap = cm.get_cmap(cmap)
    norm = BoundaryNorm(levels, cmap.N, extend="both")
    im = cm.ScalarMappable(norm=norm, cmap=cmap)
    axes = axes.flatten()
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
        cs = axes[i].contour(
            lon,
            lat,
            to_plot[i],
            transform=ccrs.PlateCarree(),
            levels=levels0,
            colors="k",
        )
        axes[i].set_boundary(boundary, transform=ccrs.PlateCarree())
        axes[i].add_feature(COASTLINE)
        axes[i].add_feature(BORDERS)
        if clabels:
            axes[i].clabel(cs)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), spacing="proportional")
    cbar.ax.set_yticks(levels0)
    if cbar_ylabel is not None:
        cbar.ax.set_ylabel(cbar_ylabel)
    return fig, axes, cbar


def cdf(timeseries: Union[xr.DataArray, NDArray]) -> tuple[NDArray, NDArray]:
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
    x = timeseries.values[idxs]
    return x, y


### Create histogram
def compute_hist(
    timeseries: xr.DataArray, season: str = None, bins: Union[NDArray, list] = LATBINS
) -> tuple[NDArray, NDArray]:
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
) -> Union[NDArray, tuple[NDArray, NDArray]]:
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
    ds: xr.DataArray,
    return_clim: bool = False,
    smooth_kmax: bool = None,
) -> Union[
    xr.DataArray, tuple[xr.DataArray, xr.DataArray]
]:  # https://github.com/pydata/xarray/issues/3575
    """computes daily anomalies extracted using a (possibly smoothed) climatology

    Args:
        ds (DataArray):
        return_clim (bool, optional): whether to also return the climatology (possibly smoothed). Defaults to False.
        smooth_kmax (bool, optional): maximum k for fourier smoothing of the climatology. No smoothing if None. Defaults to None.

    Returns:
        anom (DataArray): _description_
        clim (DataArray, optional): climatology
    """
    if len(ds["time"]) == 0:
        return ds
    gb = ds.groupby("time.dayofyear")
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
        return anom, clim
    return anom


def detrend(
    dataset: str, variable: str, level: str, region: str, smallname: str = None, name: str = "full.nc"
) -> str:
    """creates a detrended dataset out of the specs

    Args:
        dataset (str): NCEP, ERA40, ERA5,
        variable (str):
        level (str): p level
        region (str): North_Atlantic or full (dailymean)
        smallname (str): variable name in the dataset. Defaults to SMALLNAME[variable] (see definition)
        name (str, optional): _description_. Defaults to "full.nc".

    Returns:
        str: path of the detrended file for ease of access
    """
    path = f"{DATADIR}/{dataset}/{variable}/{level}/{region}"
    ds = xr.open_dataset(f"{path}/{name}").rename({"longitude": "lon", "latitude": "lat"})
    if smallname is None:
        smallname = SMALLNAME[variable]
    if variable == "Geopotential":
        ds["z"] /= co.g
    da = ds[smallname].chunk({"time": -1, "lon": 41})
    anomaly = xr.map_blocks(compute_anomaly, da, template=da)
    detrended = xr.map_blocks(
        xrft.detrend, anomaly, args=("time", "linear"), template=da
    )
    anomaly.to_netcdf(f"{path}/{smallname}_anomaly.nc")
    detrended.to_netcdf(f"{path}/{smallname}_detrended.nc")
    return path


def create_grid_directory(
    cdo, dataset: str, variable: str, level: str,
    minlon: float, maxlon: float, minlat: float, maxlat: float
) -> str:
    basepath = f"{DATADIR}/{dataset}/{variable}/{level}"
    newdir = f"box_{minlon}_{maxlon}_{minlat}_{maxlat}"
    path = f"{basepath}/{newdir}"
    if os.path.isdir(path):
        return path
    os.mkdir(path)
    for basefile in ["detrended.nc", "anomaly.nc"]:
        filelist = glob.glob(f"{basepath}/dailymean/*_{basefile}")
        for ifile in filelist:
            ofile = f"{path}/{ifile.split('/')[-1]}"
            # print(ifile, ofile)
            cdo.sellonlatbox(
                minlon, maxlon, minlat, maxlat, input=ifile, output=ofile
            )
    return path


def do_kmeans(
    n_clu: int, path: str, smallname: str, season = "JJA", detrended = False, weigh: str = "sqrtcos"
) -> str:
    midname = "detrended" if detrended else "anomaly"
    da = xr.open_dataarray(f"{path}/{smallname}_{midname}.nc")
    if weigh=="sqrtcos":
        da *= np.sqrt(degcos(da.lat))
    elif weigh=="cos":
        da *= degcos(da.lat)
    tbt = da.values.reshape(len(da.time), -1)
    results = KMeans(n_clu, n_init="auto").fit(tbt)
    pklpath = f"{path}/k{n_clu}_{midname}_{season}_{weigh}.pkl"
    with open(pklpath, "wb") as handle:
        pkl.dump(results, handle)
    return pklpath


### Lat and Int
def compute_JLI(da_Lat: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """Computes the Jet Latitude Index (also called Lat) as well as the wind speed at the JLI (Int)

    Args:
        da_Lat (xr.DataArray): zonally averaged smoothed zonal wind time series

    Returns:
        Lat (xr.DataArray): Jet Latitude Index (see Woollings et al. 2010, Barriopedro et al. 2022)
        Int (xr.DataArray): Wind speed at the JLI (see Woollings et al. 2010, Barriopedro et al. 2022)
    """
    LatI = da_Lat.argmax(dim="lat", skipna=True)
    Lat = xr.DataArray(
        da_Lat.lat[LatI.values.flatten()].values, coords={"time": da_Lat.time}
    ).rename("Lat")
    Lat.attrs["units"] = "degree_north"
    Int = da_Lat.isel(lat=LatI).reset_coords("lat", drop=True).rename("Int")
    Int.attrs["units"] = "m/s"
    return Lat, Int


### Shar, Latn, Lats,
def compute_shar(
    da_Lat: xr.DataArray, Int: xr.DataArray, Lat: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Computes sharpness and south + north latitudinal extent of the jet

    Args:
        da_Lat (xr.DataArray): zonally averaged smoothed zonal wind time series
        Lat (xr.DataArray): Jet Latitude Index (see Woollings et al. 2010, Barriopedro et al. 2022)
        Int (xr.DataArray): Wind speed at the JLI (see Woollings et al. 2010, Barriopedro et al. 2022)

    Returns:
        Shar (xr.DataArray): Sharpness (see Woollings et al. 2010, Barriopedro et al. 2022)
        Lats (xr.DataArray): Southward latitudinal extent of the jet (see Woollings et al. 2010, Barriopedro et al. 2022)
        Latn (xr.DataArray): Northward latitudinal extent of the jet (see Woollings et al. 2010, Barriopedro et al. 2022)
    """
    Shar = (Int - da_Lat.mean(dim="lat")).rename("Shar")
    Shar.attrs["units"] = Int.attrs["units"]
    this = da_Lat - Shar / 2
    ouais = np.where(this.values[:, 1:] * this.values[:, :-1] < 0)
    hist = np.histogram(ouais[0], bins=np.arange(len(da_Lat.time) + 1))[0]
    cumsumhist = np.append([0], np.cumsum(hist)[:-1])
    Lats = xr.DataArray(
        da_Lat.lat.values[ouais[1][cumsumhist]],
        coords={"time": da_Lat.time},
        name="Lats",
    )
    Latn = xr.DataArray(
        da_Lat.lat.values[ouais[1][cumsumhist + hist - 1]],
        coords={"time": da_Lat.time},
        name="Latn",
    )
    Latn[Latn < Lat] = da_Lat.lat[-1]
    Lats[Lats > Lat] = da_Lat.lat[0]
    Latn.attrs["units"] = "degree_north"
    Lats.attrs["units"] = "degree_north"
    return Shar, Lats, Latn


### Tilt
def compute_Tilt(
    da: xr.DataArray, Lat: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray]:
    """Computes tilt and also returns the tracked latitudes

    Args:
        da (xr.DataArray): _description_
        Lat (xr.DataArray): _description_

    Returns:
        tuple[xr.DataArray, xr.DataArray]: _description_
    """
    trackedLats = (
        da.isel(lat=0)
        .copy(data=np.zeros(da.shape[::2]))
        .reset_coords("lat", drop=True)
        .rename("Tracked Latitudes")
    )
    trackedLats.attrs["units"] = "degree_north"
    lats = da.lat.values
    twodelta = lats[2] - lats[0]
    midpoint = int(len(da.lon) / 2)
    trackedLats[:, midpoint] = Lat
    iterator = zip(reversed(range(midpoint)), range(midpoint + 1, len(da.lon)))
    for lonw, lone in iterator:
        for k, thislon in enumerate((lonw, lone)):
            otherlon = thislon - (
                2 * k - 1
            )  # previous step in the iterator for either east (k=1, otherlon=thislon-1) or west (k=0, otherlon=thislon+1)
            mask = (
                np.abs(trackedLats[:, otherlon].values[:, None] - lats[None, :])
                > twodelta
            )
            # mask = where not to look for a maximum. The next step (forward for east or backward for west) needs to be within twodelta of the previous (otherlon)
            here = np.ma.argmax(
                np.ma.array(da.isel(lon=thislon).values, mask=mask), axis=1
            )
            trackedLats[:, thislon] = lats[here]
    Tilt = (
        trackedLats.polyfit(dim="lon", deg=1)
        .sel(degree=1)["polyfit_coefficients"]
        .reset_coords("degree", drop=True)
        .rename("Tilt")
    )
    Tilt.attrs["units"] = "degree_north/degree_east"
    return trackedLats, Tilt


### Lon
def compute_Lon(
    da: xr.DataArray, trackedLats: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray]:
    """_summary_

    Args:
        da (xr.DataArray): _description_
        trackedLats (xr.DataArray): _description_

    Returns:
        tuple[xr.DataArray, xr.DataArray]: _description_
    """
    Intlambda = da.sel(lat=trackedLats).reset_coords("lat", drop=True)
    Intlambdasq = Intlambda * Intlambda
    lons = xr.DataArray(
        da.lon.values[None, :] * np.ones(len(da.time))[:, None],
        coords={"time": da.time, "lon": da.lon},
    )
    Lon = (lons * Intlambdasq).sum(dim="lon") / Intlambdasq.sum(dim="lon")
    Lon.attrs["units"] = "degree_east"
    return Intlambda, Lon.rename("Lon")


### Lonw, Lone
def compute_Lonew(
    da: xr.DataArray, Intlambda: xr.DataArray, Lon: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray]:
    """_summary_

    Args:
        da (xr.DataArray): _description_
        Intlambda (xr.DataArray): _description_
        Lon (xr.DataArray): _description_

    Returns:
        tuple[xr.DataArray, xr.DataArray]: _description_
    """
    Intlambda = Intlambda.values
    Mean = np.mean(Intlambda, axis=1)
    lon = da.lon.values
    iLon = np.argmax(lon[None, :] - Lon.values[:, None] > 0, axis=1)
    basearray = Intlambda - Mean[:, None] < 0
    iLonw = (
        np.ma.argmin(
            np.ma.array(basearray, mask=lon[None, :] > Lon.values[:, None]), axis=1
        )
        - 1
    )
    iLone = (
        np.ma.argmax(
            np.ma.array(basearray, mask=lon[None, :] <= Lon.values[:, None]), axis=1
        )
        - 1
    )
    Lonw = xr.DataArray(lon[iLonw], coords={"time": da.time}, name="Lonw")
    Lone = xr.DataArray(lon[iLone], coords={"day": da.day}, name="Lone")
    Lonw.attrs["units"] = "degree_east"
    Lone.attrs["units"] = "degree_east"
    return Lonw, Lone


### Dep
def compute_Dep(da: xr.DataArray, trackedLats: xr.DataArray) -> xr.DataArray:
    """_summary_

    Args:
        da (xr.DataArray): _description_
        trackedLats (xr.DataArray): _description_

    Returns:
        xr.DataArray: _description_
    """
    phistarl = xr.DataArray(
        da.lat.values[da.argmax(dim="lat").values],
        coords={"time": da.time.values, "lon": da.lon.values},
    )
    Dep = np.sqrt((phistarl - trackedLats) ** 2).sum(dim="lon").rename("Dep")
    Dep.attrs["units"] = "degree_north"
    return Dep


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
