import os
from pathlib import Path

import dask.array as da
import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
import wavebreaking as wb
import xarray as xr
from jetutils.data import open_da
from jetutils.definitions import DATADIR, YEARS, compute
from jetutils.derived_quantities import compute_absolute_vorticity
from jetutils.geospatial import gather_normal_da_jets, interp_jets_to_zero_one
from jetutils.jet_finding import (
    DataHandler,
    JetFindingExperiment,
    add_feature_for_cat,
    iterate_over_year_maybe_member,
)
from scipy.signal.windows import lanczos
from tqdm import tqdm
from wavebreaking import to_xarray

os.environ["RUST_BACKTRACE"] = "full"


def convolve(in1, in2, mode="full", method="fft", axes=None):
    from scipy.signal import fftconvolve, oaconvolve
    from scipy.signal._signaltools import _init_freq_conv_axes

    in1 = da.asarray(in1)
    in2 = np.asarray(in2)

    # Checking for trivial cases and incorrect flags
    if in1.ndim == in2.ndim == 0:  # scalar inputs
        return in1 * in2
    elif in1.ndim != in2.ndim:
        raise ValueError("in1 and in2 should have the same dimensionality")
    elif in1.size == 0 or in2.size == 0:  # empty arrays
        return np.array([])

    if mode != "full" and mode != "same" and mode != "valid" and mode != "periodic":
        raise ValueError(
            "acceptable mode flags are 'valid', 'same', 'full' or 'periodic'"
        )
    if method not in ["fft", "oa"]:
        raise ValueError("acceptable method flags are 'oa', or 'fft'")

    # Pre-formatting or the the inputs, mainly for the `axes` argument
    in1, in2, axes = _init_freq_conv_axes(in1, in2, mode, axes, sorted_axes=False)

    # _init_freq_conv_axes calls a function that will swap out inputs if required
    # when mode == "valid".We want to avoid having in2 be a dask array thus check
    # to see if the inputs were swapped and raise an error.
    if isinstance(in1, np.ndarray):
        raise ValueError(
            "For 'valid' mode in1 has to be at least as large as in2 in every dimension"
        )

    s1 = in1.shape
    s2 = in2.shape

    # If all axe were removed by the preformatting we only have to rely
    # on multiplication broadcasting rules.
    if not len(axes):
        in_cv = in1 * in2
        # This is the "full" output that is also valid.
        # To get the "same" output we need to center in some dimensions.
        if mode == "same" or mode == "periodic":
            not_axes_but_s1_1 = [
                a
                for a in range(in1.ndim)
                if a not in axes and s1[a] == 1 and s2[a] != 1
            ]
            in_cv = in_cv[
                tuple(
                    (
                        slice((s2[a] - 1) // 2, (s2[a] - 1) // 2 + 1)
                        if a in not_axes_but_s1_1
                        else slice(None, None)
                    )
                    for a in range(in1.ndim)
                )
            ]
            return in_cv

    else:
        # This is kind of a hack but it works.
        not_axes_but_s1_1 = [
            a for a in range(in1.ndim) if a not in axes and s1[a] == 1 and s2[a] != 1
        ]
        if len(not_axes_but_s1_1) and (mode == "full" or mode == "valid"):
            new_shape = tuple(
                s1[i] for i in range(in1.ndim) if i not in not_axes_but_s1_1
            )
            in1 = in1.reshape(new_shape)
            for a in not_axes_but_s1_1:
                in1 = da.stack([in1] * s2[a], axis=a)
            return convolve(in1, in2, mode=mode, method=method, axes=axes)

        # Deals with the case where there is at least one axis a in which we do not
        # do the convolution that has s2[a] == s1[a] != 1
        not_axes_but_same_shape = [
            a for a in range(in1.ndim) if a not in axes and s1[a] == s2[a] != 1
        ]
        if len(not_axes_but_same_shape):
            to_rechunk = [a for a in not_axes_but_same_shape if len(in1.chunks[a]) != 1]
            new_chunk_size = tuple(
                -1 if a in to_rechunk else "auto" for a in range(in1.ndim)
            )
            in1 = in1.rechunk(new_chunk_size)

        depth = {i: s2[i] // 2 for i in axes}

        # Flags even dimensions and removes them by adding zeros
        # This is done to avoid from having some results show up twice
        # at the edge of blocks
        even_flag = np.r_[[1 - s2[a] % 2 if a in axes else 0 for a in range(in1.ndim)]]
        target_shape = np.asarray(s2)
        target_shape += even_flag

        if any(target_shape != np.asarray(s2)):
            # padding axes where in2 is even
            pad_width = tuple(
                (even_flag[a], 0) if a in axes else (0, 0) for a in range(in1.ndim)
            )
            in2 = da.pad(in2, pad_width)

        if mode != "valid":
            pad_width = tuple(
                (depth[i] - even_flag[i], depth[i]) if i in axes else (0, 0)
                for i in range(in1.ndim)
            )
            in1 = da.pad(in1, pad_width)

        if mode == "periodic":
            boundary = "periodic"
        else:
            boundary = 0

        cv_dict = {"oa": oaconvolve, "fft": fftconvolve}
        
        def cv_func(x):
            return cv_dict[method](x, in2, mode="same", axes=axes)

        complex_result = in1.dtype.kind == "c" or in2.dtype.kind == "c"

        if complex_result:
            dtype = "complex"
        else:
            dtype = "float"

        # Actualy does the convolution with all the parameters preformatted
        in_cv = in1.map_overlap(
            cv_func, depth=depth, boundary=boundary, trim=True, dtype=dtype
        )

        # The output as to be reduced depending on the `mode` argument
        if mode == "valid":
            output_slicing = tuple(
                (
                    slice(depth[i], s1[i] - (depth[i] - even_flag[i]), 1)
                    if i in depth.keys()
                    else slice(0, None)
                )
                for i in range(in1.ndim)
            )
            in_cv = in_cv[output_slicing]

        elif mode != "full":
            # Only have to undo the padding
            output_slicing = tuple(
                slice(p[0], -p[1]) if p != (0, 0) else slice(0, None) for p in pad_width
            )
            in_cv = in_cv[output_slicing]

    return in_cv


def create_jet_relative_dataset(
    jets,
    da,
    half_length: float = 2e6,
    dn: float = 1e5,
    n_interp: int = 30,
    in_meters: bool = True,
):
    indexer = iterate_over_year_maybe_member(jets, da)
    to_average = []
    varname = da.name + "_interp"
    for idx1, idx2 in tqdm(indexer, total=len(YEARS)):
        jets_ = jets.filter(*idx1)
        da_ = da.sel(**idx2)
        try:
            jets_with_interp = gather_normal_da_jets(
                jets_, da_, half_length=half_length, dn=dn, in_meters=in_meters
            )
        except (KeyError, ValueError) as e:
            print(e)
            break
        jets_with_interp = interp_jets_to_zero_one(
            jets_with_interp, [varname, "is_polar"], n_interp=n_interp
        )
        # jets_with_interp = jets_with_interp.group_by("time", pl.col("is_polar").mean().over(["time", "jet ID"]) > 0.5, "norm_index", "n", maintain_order=True).agg(pl.col(varname).mean())
        to_average.append(jets_with_interp)
    return pl.concat(to_average)


# block 1: compute zeta
for run in ["ctrl", "dobl"]:
    basepath_i = Path(DATADIR, f"Henrik_data/{run}/high_wind/6H")
    basepath_zeta = Path(DATADIR, f"Henrik_data/{run}/zeta/6H")
    basepath_zeta.mkdir(parents=True, exist_ok=True)

    glob = basepath_i.glob("????.nc")

    for f in glob:
        oname = f.name
        ofile = basepath_zeta.joinpath(oname)
        if not ofile.is_file():
            ds = xr.open_dataset(f)
            ds = ds[["u", "v"]]
            zeta = compute_absolute_vorticity(ds)
            zeta.to_netcdf(ofile)
            print("zeta", oname, run)

# block 2: compute EMF

for run in ["ctrl", "dobl"]:
    opath = Path(
        f"{DATADIR}/Henrik_data/{run}/high_wind/6H/results",
        "Eddy_uv300_natl_10days.zarr",
    )
    if opath.is_dir():
        continue
    half_len = 20
    uv = (
        xr.open_mfdataset(
            f"{DATADIR}/Henrik_data/{run}/high_wind/6H/*.nc",
            combine="nested",
            concat_dim="time",
        )[["u", "v"]]
        .sel(lon=slice(-80, 40), lat=slice(15, 80), lev=30000)
        .chunk("auto")
    )
    l_win = lanczos(2 * half_len + 1)[:, None, None]
    dims = uv.dims
    uv["ubar"] = (
        dims,
        (convolve(uv["u"].data, l_win)[half_len:-half_len] / l_win.sum()).astype(
            np.float32
        ),
    )
    uv["up"] = uv["u"] - uv["ubar"]
    uv["vbar"] = (
        dims,
        (convolve(uv["v"].data, l_win)[half_len:-half_len] / l_win.sum()).astype(
            np.float32
        ),
    )
    uv["vp"] = uv["v"] - uv["vbar"]

    del uv["ubar"]
    del uv["vbar"]
    del uv["u"]
    del uv["v"]
    uv = uv.chunk({"time": 1390, "lat": 72, "lon": 161})
    uv["EMF"] = uv["up"] * uv["vp"]
    res = uv.to_zarr(opath, compute=False)
    compute(res, progress=True)


# stage 3: compute wavebreaking
for run in ["ctrl", "dobl"]:
    basepath_zeta = Path(DATADIR, f"Henrik_data/{run}/zeta/6H")
    da_mflux = xr.open_dataset(
        f"{DATADIR}/Henrik_data/{run}/high_wind/6H/results/Eddy_uv300_natl_10days.zarr"
    )["EMF"]
    opath = Path(DATADIR, f"Henrik_data/{run}/rwb_index")
    opath.mkdir(exist_ok=True)
    for year in range(1969, 2021):
        zeta = xr.open_dataarray(basepath_zeta.joinpath(f"{year}.nc")).sel(lev=30000)
        zeta = zeta.assign_coords(
            lat=0.7 + np.arange(len(zeta.lat), dtype=np.float128) * 0.94
        ).rename("zeta")
        mflux = da_mflux.sel(time=zeta.time)
        mflux = mflux.assign_coords(
            lat=0.7 + np.arange(len(mflux.lat), dtype=np.float128) * 0.94
        ).rename("mflux")
        ofile = opath.joinpath(f"overturnings_{year}.parquet")
        if ofile.is_file():
            continue
        zeta_ = wb.calculate_smoothed_field(
            data=zeta,
            passes=5,
            weights=np.array([[0, 1, 0], [1, 2, 1], [0, 1, 0]]),
            mode="wrap",
        )
        out = wb.calculate_overturnings(
            data=zeta_ * 1e5,
            contour_levels=[7, 8, 9, 10],
            intensity=mflux,  # optional
            periodic_add=120,
        )
        out[0].to_parquet(ofile)
        
# stage 4: rwb das
varname = "ones"
dtype = {"ones": np.uint32, "intensity": np.float32, "mean_var": np.float32, "event_area": np.float32}[varname]
coords = {
    "time": pd.date_range("19690101", "20210101", freq="6h", inclusive="left"),
    "lat": np.arange(0, 90.5, 1),
    "lon": np.arange(-100, 60.5, 1),
}
shape = [len(co) for co in coords.values()]
dummy_da = xr.DataArray(np.zeros(shape, dtype=dtype), coords=coords)

if not Path(DATADIR, "Henrik_data", "dobl", "APVO/dailyany/full.nc").is_file():
    for run in ["ctrl", "dobl"]:
        all_events = {}
        basepath = Path(DATADIR, "Henrik_data", run, "rwb_index")
        files = list(sorted(basepath.glob("overturnings_????.parquet")))
        events = []
        for f in files:
            events.append(gpd.read_parquet(f))
        events = pd.concat(events)
        all_events["APVO"] = events[events.orientation == "anticyclonic"]
        all_events["CPVO"] = events[events.orientation == "cyclonic"]
        opath = Path(DATADIR, "Henrik_data", run)
        for name in ["APVO", "CPVO"]:
            da_rwb = to_xarray(dummy_da, all_events[name], flag=varname).rename(name)
            opath.joinpath(f"{name}/6H").mkdir(parents=True, exist_ok=True)
            da_rwb.to_netcdf(opath.joinpath(f"{name}/6H/full.nc"))
            da_rwb = da_rwb.resample(time="1D").any().astype(np.uint8)
            opath.joinpath(f"{name}/dailyany").mkdir(parents=True, exist_ok=True)
            da_rwb.to_netcdf(opath.joinpath(f"{name}/dailyany/full.nc"))

# stage 5: define jets (already computed probably)
both_jets = {}
both_paths = {}
for run in ["ctrl", "dobl"]:
    dh = DataHandler.from_specs("Henrik_data", run, ("high_wind", ["u", "v", "s", "theta"]), "6H", minlon=-80, maxlon=40, minlat=15, maxlat=80, levels=20000)
    exp = JetFindingExperiment(dh)
    all_jets_one_df = exp.find_jets(force=False, n_coarsen=1, smooth_s=5, alignment_thresh=0.55, base_s_thresh=0.55, int_thresh_factor=0.2, hole_size=5)
    theta300 = open_da("Henrik_data", run, ("high_wind", ["s", "theta"]), "6H", minlon=-80, maxlon=40, minlat=15, maxlat=80, levels=30000).rename({"s": "s300", "theta": "theta300"})
    all_jets_one_df = add_feature_for_cat(all_jets_one_df, "s300", theta300, ofile_ajdf=exp.path.joinpath("all_jets_one_df.parquet"), force=False)
    all_jets_one_df = add_feature_for_cat(all_jets_one_df, "theta300", theta300, ofile_ajdf=exp.path.joinpath("all_jets_one_df.parquet"), force=False)
    all_jets_one_df = exp.categorize_jets(None, ["s300", "theta300"], force=False, n_init=15, init_params="k-means++", mode="month").cast({"time": pl.Datetime("ms")})
    phat_filter = (pl.col("is_polar") < 0.5) | ((pl.col("is_polar") > 0.5) & (pl.col("int") > 1.e8))
    phat_jets = all_jets_one_df.filter((pl.col("is_polar").mean().over(["time", "jet ID"]) < 0.5) | ((pl.col("is_polar").mean().over(["time", "jet ID"]) > 0.5) & (pl.col("int").mode().first().over(["time", "jet ID"]) > 1.e8)))
    both_paths[run] = exp.path
    both_jets[run] = phat_jets.with_columns(**{"jet ID": (pl.col("is_polar").mean().over(["time", "jet ID"]) > 0.5).cast(pl.UInt32())})

# stage 6: Interpolate new fields
args = ["all", None, -100, 60, 0, 90]

to_do = (
    ("PTTEND300", ("high_wind", "PTTEND"), {"levels": 30000}),
    ("DTCOND300", ("high_wind", "DTCOND"), {"levels": 30000}),
    ("theta300", ("high_wind", "theta"), {"levels": 30000}),
    ("s300", ("high_wind", "s"), {"levels": 30000}),
    ("zeta300", "zeta", {"levels": 30000}),
    ("z500", "z", {}),
    ("t_low", "t_low", {"levels": 100000}),
    ("PTTEND500", ("heating", "PTTEND"), {"levels": 50000}),
    ("DTCOND500", ("heating", "DTCOND"), {"levels": 50000}),
    ("APVO", "APVO", {}),
    ("CPVO", "CPVO", {}),
)

for run in ["ctrl", "dobl"]:
    for huh in to_do:
        ofile = both_paths[run].joinpath(f"{da.name}_meters_relative.parquet")
        if ofile.is_file():
            continue
        rename, name, kwargs = huh
        da_ = open_da("Henrik_data", run, name, "6H", *args, **kwargs).rename(rename)
        da_ = compute(da_)
        interpd = create_jet_relative_dataset(both_jets[run], da_)
        del da_
        interpd.write_parquet(ofile)