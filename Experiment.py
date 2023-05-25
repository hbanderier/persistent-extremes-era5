import os
import pickle as pkl
import logging
import glob
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Optional, Tuple, Iterable

import numpy as np
from nptyping import Float, Int, NDArray, Object, Shape
import xarray as xr
import xrft
from scipy import linalg, constants as co
from scipy.optimize import minimize
from simpsom import SOMNet
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA as pca
from joblib import Parallel, delayed
from kmedoids import KMedoids
import contourpy

try:
    import cupy as cp  # won't work on cpu nodes
except ImportError:
    pass


from definitions import NODE, DATADIR, SMALLNAME, DATERANGEPL, DATERANGEPL_EXT, LATBINS, cdo, degcos, degsin, setup_cdo, CIequal


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
    lon: NDArray
    lat: NDArray
    coslat: NDArray
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
    ) -> None:
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
        self.lon = None
        self.lat = None
        self.coslat = None
        self.copy_content(smooth)

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

        if self.lon is None or self.lat is None or self.coslat is None:
            self.lon = da.lon.values
            self.lat = da.lat.values
            self.coslat = degcos(np.meshgrid(np.ones(len(self.lon)), self.lat)[1])
        
        for daterange in (DATERANGEPL, DATERANGEPL_EXT):
            if len(da.time) == len(daterange):
                da = da.assign_coords({'time': daterange.values})
                break
            elif len(da.drop_duplicates(dim='time').time) == len(daterange):
                da = da.drop_duplicates(dim='time').assign_coords({'time': daterange.values})
                break
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

    def detrend(self, n_workers: int = 8) -> None:
        da = self.open_da(chunks={"time": -1, "lon": 30, "lat": 30})
        anom = da.map_blocks(compute_anomaly, template=da).compute(n_workers=n_workers)
        anom.to_netcdf(self.ofile("anomaly"))
        anom = anom.map_blocks(xrft.detrend, template=anom, args=["time", "linear"]).compute(n_workers=n_workers)
        anom.to_netcdf(self.ofile("detrended"))

    def get_winsize(self, da: xr.DataArray) -> Tuple[float, float, float]:
        resolution = (da.lon[1] - da.lon[0]).values.item()
        winsize = int(60 / resolution)
        halfwinsize = int(winsize / 2)
        return resolution, winsize, halfwinsize

    def smooth(self) -> None:
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

    def copy_content(self, smooth: bool = False):
        if not self.path.is_dir():
            os.mkdir(self.path)
        ifile = self.ifile("")
        ofile = self.ofile("")
        if not ofile.is_file() and (not smooth and not self.region == "dailymean"):
            setup_cdo()
            cdo.sellonlatbox(
                self.minlon,
                self.maxlon,
                self.minlat,
                self.maxlat,
                input=ifile.as_posix(),
                output=ofile.as_posix(),
            )
        elif not ofile.is_file() and smooth:
            setup_cdo()
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
                setup_cdo()
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


def compute_autocorrs(X: NDArray[Shape["*, *"], Float], lag_max: int) -> NDArray[Shape["*, *, *"], Float]:
    autocorrs = []
    i_max = X.shape[1]
    for i in range(lag_max):
        autocorrs.append(
            np.cov(X[i:], np.roll(X, i, axis=0)[i:], rowvar=False)[
                i_max:, :i_max
            ]
        )
    return np.asarray(autocorrs)

@dataclass(init=False)
class ClusteringExperiment(Experiment):
    midfix: str = "anomaly"
    season: list | str = None

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
    ) -> None:
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

    def prepare_for_clustering(self) -> Tuple[NDArray, xr.DataArray]:
        da = self.open_da(self.midfix, self.season)
        X = da.values.reshape(len(da.time), -1) * np.sqrt(self.coslat.flatten())[None, :]
        return X, da

    def to_dataarray(
        self,
        centers: NDArray[Shape["*, *"], Float],
        da: xr.DataArray,
        n_pcas: int = None,
        coords: dict | str = None,
    ) -> xr.DataArray:
        if n_pcas: # 0 pcas is also no transform
            centers = self.pca_inverse_transform(centers, n_pcas)
        if coords is None:
            coords = 'mode'
        if isinstance(coords, str):
            coords = {
                coords : np.arange(centers.shape[0]),
                'lat': da.lat.values,
                'lon': da.lon.values,
            }
        shape = [len(coord) for coord in coords.values()]
        centers = xr.DataArray(centers.reshape(shape), coords=coords)
        centers /= np.sqrt(degcos(da.lat))
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
        n_pcas: int,
    ) -> NDArray[Shape["*, *"], Float]:
        pca_path = self.compute_pcas(n_pcas)
        with open(pca_path, "rb") as handle:
            pca_results = pkl.load(handle)
        X = pca_results.transform(X)
        return X
    
    def pca_inverse_transform(
        self,
        X: NDArray[Shape["*, *"], Float],
        n_pcas: int,
    ) -> NDArray[Shape["*, *"], Float]:
        pca_path = self.compute_pcas(n_pcas)
        with open(pca_path, "rb") as handle:
            pca_results = pkl.load(handle)
        diff_n_pcas = pca_results.n_components - X.shape[1]
        X = np.pad(X, [[0, 0], [0, diff_n_pcas]])
        X = pca_results.inverse_transform(X)
        return X.reshape(X.shape[0], -1)
    
    def _compute_opps_T1(
            self, 
            X: NDArray, 
            lag_max: int,
        ) -> dict:
        autocorrs = compute_autocorrs(X, lag_max)
        M = np.trapz(autocorrs + autocorrs.transpose((0, 2, 1)), axis=0)

        invC0 = linalg.inv(autocorrs[0])
        eigenvals, eigenvecs = linalg.eigh(0.5 * invC0 @ M)
        OPPs = autocorrs[0] @ eigenvecs.T
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        OPPs = OPPs[idx]
        T1s = np.sum(OPPs.reshape(OPPs.shape[0], 1, 1, OPPs.shape[1]) @ autocorrs @ OPPs.reshape(OPPs.shape[0], 1, OPPs.shape[1], 1), axis=1).squeeze()
        T1s /= (OPPs.reshape(OPPs.shape[0], 1, OPPs.shape[1]) @ autocorrs[0] @ OPPs.reshape(OPPs.shape[0], OPPs.shape[1], 1)).squeeze()
        return {
            "T": T1s,
            "OPPs": OPPs,
        }
    
    def _compute_opps_T2(
            self, 
            X: NDArray, 
            lag_max: int
        ) -> dict:
        autocorrs = compute_autocorrs(X, lag_max)
        C0sqrt = linalg.sqrtm(autocorrs[0])
        C0minushalf = linalg.inv(C0sqrt)
        basis = linalg.orth(C0minushalf)

        def minus_T2(x) -> float:
            normxsq = linalg.norm(x) ** 2
            factor1 = x.T @ C0minushalf @ autocorrs @ C0minushalf @ x
            return - 2 * np.trapz(factor1 ** 2) / normxsq ** 2

        def minus_T2_gradient(x) -> NDArray[Shape["*"], Float]:
            normxsq = linalg.norm(x) ** 2
            factor1 = x.T @ C0minushalf @ autocorrs @ C0minushalf @ x
            factor2 = (C0minushalf @ (autocorrs + autocorrs.transpose((0, 2, 1))) @ C0minushalf) @ x
            numerator = 4 * np.trapz((factor1)[:, None] * factor2, axis=0)
            return - numerator / normxsq ** 2 - 4 * minus_T2(x) * x / normxsq ** 3

        def norm0(x) -> float:
            return 10 - linalg.norm(x) ** 2

        def jac_norm0(x) -> NDArray[Shape["*"], Float]:
            return - 2 * x
        
        Id = np.eye(X.shape[1])
        proj = Id.copy()
        OPPs = []
        T2s = []
        numsuc = 0
        while numsuc < 10:
            xmin, xmax = np.amin(basis, axis=0), np.amax(basis, axis=0)
            x0 = xmin + (xmax - xmin) * np.random.rand(len(xmax))
            res = minimize(minus_T2, x0, jac=minus_T2_gradient, method='SLSQP', constraints={'type': 'ineq', 'fun': norm0, 'jac': jac_norm0})
            if res.success:
                unit_x = res.x / linalg.norm(res.x)
                OPPs.append(C0sqrt @ unit_x)
                T2s.append(-res.fun)
                proj = Id - np.outer(unit_x, unit_x)
                autocorrs = proj @ autocorrs @ proj
                C0minushalf = proj @ C0minushalf @ proj
                numsuc += 1
        return {
            "T": np.asarray(T2s),
            "OPPs": np.asarray(OPPs),
        }

    def compute_opps(
        self,
        n_pcas: int = None,
        lag_max: int = 90,
        type: int = 1,
        return_realspace: bool = False,
    ) -> Tuple[Path, dict] | Tuple[NDArray, xr.DataArray, Path]:
        """
        I could GPU this fairly easily. Is it worth it tho ?
        """
        if type not in [1, 2]:
            raise ValueError(f'Wrong OPP type, pick 1 or 2')
        X, da = self.prepare_for_clustering()
        if n_pcas:
            X = self.pca_transform(X, n_pcas)
        X = X.reshape((X.shape[0], -1))
        n_pcas = X.shape[1]
        opp_path: Path = self.path.joinpath(
            f"opp_{n_pcas}_{self.midfix}_{self.season}_T{type}.pkl"
        )
        results = None
        if not opp_path.is_file():
            if type == 1:
                results = self._compute_opps_T1(X, n_pcas, lag_max)
            if type == 2:
                results = self._compute_opps_T2(X, n_pcas, lag_max)
            with open(opp_path, "wb") as handle:
                pkl.dump(results, handle)
        if results is None:
            with open(opp_path, "rb") as handle:
                results = pkl.load(handle)
        if not return_realspace:
            return opp_path, results
        OPPs = results["OPPs"]
        eigenvals = results["T"]
        coords = 'OPP'
        OPPs = self.to_dataarray(OPPs, da, n_pcas, coords)
        return eigenvals, OPPs, opp_path

    def opp_transform(
        self,
        X: NDArray[Shape["*, *"], Float],
        n_pcas: int,
        cutoff: int = None,
        type: int = 1,
    ) -> NDArray[Shape["*, *"], Float]:
        _, results = self.compute_opps(n_pcas, type=type)
        if not X.shape[1] == n_pcas:
            X = self.pca_transform(X, n_pcas)
        if cutoff is None:
            cutoff = n_pcas if type == 1 else 10
        OPPs = results["OPPs"][:cutoff]
        X = X @ OPPs.T
        return X

    def opp_inverse_transform(
        self,
        X: NDArray[Shape["*, *"], Float],
        n_pcas: int = None,
        cutoff: int = None,
        to_realspace=False,
        type: int = 1,
    ) -> NDArray[Shape["*, *"], Float]:
        _, results = self.compute_opps(n_pcas, type=type)
        if cutoff is None:
            cutoff = n_pcas if type == 1 else 10
        OPPs = results["OPPs"][:cutoff]
        X = X @ OPPs
        if to_realspace:
            return self.pca_inverse_transform(X)
        return X

    def cluster(
        self,
        n_clu: int,
        n_pcas: int,
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
            f"k{suffix}_{n_clu}_{self.midfix}_{self.season}.pkl"
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
            centers = self.to_dataarray(centers, da, n_pcas, coords)
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
            f"som_{nx}_{ny}_{self.midfix}_{self.season}{'_OPP' if OPP else ''}.npy"
        )
        if train_kwargs is None:
            train_kwargs = {}

        if output_path.is_file() and not return_centers:
            return output_path
        if OPP:
            X, da = self.prepare_for_clustering()
            X = self.opp_transform(X, n_pcas=n_pcas)
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
            centers = self.opp_inverse_transform(centers, n_pcas=n_pcas)
        centers = self.to_dataarray(centers, da, n_pcas, coords)
        return net, centers
    

def project_onto_clusters(X: NDArray | xr.DataArray, centers: NDArray | xr.DataArray, weighs:tuple = None) -> NDArray | xr.DataArray:
    
    if isinstance(X, NDArray): 
        if isinstance(centers, xr.DataArray): # always cast to type of X
            centers = centers.values
        if weighs is not None:
            X = np.swapaxes(np.swapaxes(X, weighs[0], -1) * weighs[1], -1, weighs[0]) # https://stackoverflow.com/questions/30031828/multiply-numpy-ndarray-with-1d-array-along-a-given-axis
        return np.tensordot(X, centers.T, axes=X.ndim - 1) # cannot weigh
        
    if isinstance(X, xr.DataArray):
        if isinstance(centers, NDArray):
            if centers.ndim == 4:
                centers = centers.reshape((centers.shape[0] * centers.shape[1], centers.shape[2], centers.shape[3]))
            coords = dict(centers=np.arange(centers.shape[0]), **{key: val for key, val in X.coords if key != 'time'})
            centers = xr.DataArray(centers, coords=coords)
        try:
            weighs = np.sqrt(degcos(X.lat))
            X *= weighs
            denominator = np.sum(weighs.values)
        except AttributeError:
            denominator = 1
        return X.dot(centers) / denominator    


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