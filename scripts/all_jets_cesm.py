import warnings
from itertools import product
from pathlib import Path

import numpy as np
import xarray as xr
from contourpy import contour_generator
from sklearn.cluster import AgglomerativeClustering
import dask

from jetstream_hugo.definitions import (
    DATADIR,
    RADIUS,
    COMPUTE_KWARGS,
    get_runs_fill_holes,
    labels_to_mask,
)
from jetstream_hugo.data import smooth
from jetstream_hugo.jet_finding import (
    compute_alignment,
    interp_xy_ds,
    jet_integral_haversine,
)
from dask.distributed import Client, progress

def flatten_by(ds: xr.Dataset, by: str = "-criterion") -> xr.Dataset:
    if "lev" not in ds.dims:
        return ds
    ope = np.nanargmin if by[0] == "-" else np.nanargmax
    by = by.lstrip("-")
    levmax = ds[by].reduce(ope, dim="lev")
    ds = ds.isel(lev=levmax).reset_coords("lev")  # but not drop
    ds["lev"] = ds["lev"].astype(np.float32)
    return ds


def preprocess(ds: xr.Dataset, smooth_s: float = None) -> xr.Dataset:
    # ds = flatten_by(ds, "s")
    # if (ds.lon[1] - ds.lon[0]) <= 0.75:
    #     ds = coarsen_da(ds, 1.5)
    if smooth_s is not None:
        for var in ["u", "v", "s"]:
            ds[var] = smooth(ds[var], smooth_map={"lon+lat": ("fft", smooth_s)})
    ds = ds.assign_coords(
        {
            "x": np.radians(ds["lon"]) * RADIUS,
            "y": RADIUS
            * np.log(
                (1 + np.sin(np.radians(ds["lat"])) / np.cos(np.radians(ds["lat"])))
            ),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ds["sigma"] = (
            ds["u"] * ds["s"].differentiate("y") - ds["v"] * ds["s"].differentiate("x")
        ) / ds["s"]
    fft_smoothing = 1.0 if ds["sigma"].min() < -0.0001 else 0.8
    ds["sigma"] = smooth(ds["sigma"], smooth_map={"lon+lat": ("fft", fft_smoothing)})
    return ds.reset_coords(["x", "y"], drop=True)


def find_jets(
    ds: xr.Dataset,
    template: xr.Dataset,
    wind_threshold: float = 23,
    jet_threshold: float = 1.0e8,
    alignment_threshold: float = 0.3,
    mean_alignment_threshold: float = 0.7,
    smooth_s: float = 0.3,
    hole_size: int = 1,
):
    max_n_jets, max_n_points = template["u"].shape
    if "threshold" in ds:
        wind_threshold = ds["threshold"].item()
        jet_threshold = jet_threshold * ds["threshold_ratio"].item()
    ds = preprocess(ds, smooth_s=smooth_s)
    lon, lat = ds.lon.values, ds.lat.values
    dx = lon[1] - lon[0]
    contours, types = contour_generator(
        x=lon, y=lat, z=ds["sigma"].values, line_type="SeparateCode", quad_as_tri=False
    ).lines(0.0)
    groups = []
    for contour, types_ in zip(contours, types):
        if len(contour) < 15:
            continue
        cyclic: bool = 79 in types_
        group = interp_xy_ds(ds, contour[::-1])
        group = compute_alignment(group)
        mask = (group["alignment"] > alignment_threshold) & (
            group["s"].values > wind_threshold
        )
        mask = mask.values
        indicess = get_runs_fill_holes(mask, hole_size=hole_size, cyclic=cyclic)
        for indices in indicess:
            indices = np.unique(indices)
            if len(indices) < 15:
                continue
            group_df = group.to_dataframe()
            for potential_to_drop in ["time", "member", ["ratio", "label"]]:
                try:
                    group_df = group_df.drop(columns=potential_to_drop)
                except KeyError:
                    pass
            group_df = group_df.iloc[indices]
            group_ = group_df[["lon", "lat"]].values.astype(np.float32)
            labels = (
                AgglomerativeClustering(
                    n_clusters=None, distance_threshold=dx * 1.9, linkage="single"
                )
                .fit(group_)
                .labels_
            )
            masks = labels_to_mask(labels)
            for mask in masks.T:
                groups.append(group_df.iloc[mask])
    jets = []
    for group_df in groups:
        bigjump = np.diff(group_df["lon"]) < -3 * dx
        if any(bigjump):
            here = np.where(bigjump)[0][0] + 1
            group_df = group_df.apply(np.roll, args=(-here,), raw=True)
        group_ = group_df[["lon", "lat", "s"]].values.astype(np.float32)
        jet_int = jet_integral_haversine(group_)
        mean_alignment = np.mean(group_df["alignment"].values)
        if jet_int > jet_threshold and mean_alignment > mean_alignment_threshold:
            jets.append(xr.Dataset.from_dataframe(group_df.reset_index()))
    try:
        jets = xr.concat(jets, dim="jet").drop_vars("points").rename(index="point").astype(np.float32)
        jets = jets.isel(jet=slice(max_n_jets), point=slice(max_n_points))
        return jets
    except ValueError: # Fail to concat because no jets: need to return something, maybe should be all nan instead
        return template


def find_all_jets(
    ds: xr.Dataset,
    basepath: Path,
    n_jets: int = 15,
    n_points: int = 230,
    **kwargs,
):
    def inner_find_all_jets(ds_block):
        extra_dims = {}
        for potential in ["member", "time"]:
            if potential in ds_block.dims:
                extra_dims[potential] = ds_block[potential].values
        to_ret = None
        # len_ = np.prod([len(co) for co in extra_dims.values()])
        iter_ = product(*list(extra_dims.values()))
        # iter_ = product(*list(extra_dims.values()))
        for vals in iter_:
            indexer = {dim: val for dim, val in zip(extra_dims, vals)}
            this_ds = ds_block.loc[indexer]
            these_jets = find_jets(this_ds, **kwargs)
            indexer = indexer | {"jet": these_jets.jet.values, "point": these_jets.point.values}
            if to_ret is None:
                to_ret = xr.Dataset({varname: template_da.copy() for varname in these_jets.data_vars})
            to_ret.loc[indexer] = these_jets
        first_time = ds_block.time[0]
        yr = str(first_time.dt.year).zfill(4)
        mo = str(first_time.dt.month).zfill(2)
        da = str(first_time.dt.day).zfill(2)
        to_ret.to_netcdf(basepath.joinpath(f"jets/{yr}{mo}{da}.nc"))
        return
    
    all_jets_one_ds = xr.map_blocks(inner_find_all_jets, ds, template=template)

    return all_jets_one_ds


def main():
    basepath = Path(f"{DATADIR}/CESM2/flat_wind")
    ds = xr.open_dataset(
        basepath.joinpath("ds.zarr"), 
        engine="zarr", 
        chunks={"member": -1, "time": 100, "lat": -1, "lon": -1}
    )
    ds = ds.reset_coords("time_bnds", drop=True)
    quantiles = xr.open_dataarray(basepath.joinpath("results/s_q.nc"))
    ds["threshold"] = ("time", quantiles[4, :].data)
    ds["threshold_ratio"] = ds["threshold"] / ds["threshold"].max().item()
    with Client(**COMPUTE_KWARGS):
        oe = find_all_jets(ds, basepath=basepath.joinpath("results/1"))
        oe = dask.persist(oe)
        progress(oe, notebook=False)
        oe = dask.compute(oe)
    
if __name__ == '__main__':
    main()