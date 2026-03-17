import os
from pathlib import Path

import dask.array as darr
import numpy as np
import polars as pl
import xarray as xr
from jetutils.data import open_da, smooth
from jetutils.definitions import DATADIR, YEARS, compute, N_WORKERS, DEFAULT_VARNAME, degsin, OMEGA, C_P_AIR, KAPPA
from jetutils.derived_quantities import compute_absolute_vorticity
from jetutils.geospatial import (
    gather_normal_da_jets,
    interp_jets_to_zero_one,
    detect_contours,
    detect_overturnings,
    event_props,
    to_xarray_sjoin,
)
from jetutils.jet_finding import (
    DataHandler,
    JetFindingExperiment,
    add_feature_for_cat,
    iterate_over_year_maybe_member,
)
from scipy.signal.windows import lanczos
from tqdm import tqdm, trange

os.environ["RUST_BACKTRACE"] = "full"


def convolve(in1, in2, mode="full", method="fft", axes=None):
    from scipy.signal import fftconvolve, oaconvolve
    from scipy.signal._signaltools import _init_freq_conv_axes

    in1 = darr.asarray(in1)
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
                in1 = darr.stack([in1] * s2[a], axis=a)
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
            in2 = darr.pad(in2, pad_width)

        if mode != "valid":
            pad_width = tuple(
                (depth[i] - even_flag[i], depth[i]) if i in axes else (0, 0)
                for i in range(in1.ndim)
            )
            in1 = darr.pad(in1, pad_width)

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

# block 2: compute eddy stuff
for run in ["ctrl", "dobl"]:
    opath = Path(
        f"{DATADIR}/Henrik_data/{run}/high_wind/6H/results",
        "Eddy_NH_10days.zarr",
    )
    if opath.is_dir():
        continue
    half_len = 20
    ds = (
        xr.open_mfdataset(
            f"{DATADIR}/Henrik_data/{run}/high_wind/6H/*.nc",
            combine="nested",
            concat_dim="time",
        )[["u", "v", "theta"]]
        .sel(lat=slice(0, 90))
        .chunk("auto")
    )
    for var in ds.data_vars:
        ds[var] = ds[var].astype(np.float32)
    ds["omega"] = (
        xr.open_mfdataset(
            f"{DATADIR}/Henrik_data/{run}/vertical/6H/*.nc",
            combine="nested",
            concat_dim="time",
        )["omega"]
        .astype(np.float32)
        .sel(lat=slice(0, 90))
        .chunk("auto")
    )
    ds["phi"] = (
        xr.open_mfdataset(
            f"{DATADIR}/Henrik_data/{run}/z_high/6H/*.nc",
            combine="nested",
            concat_dim="time",
        )["Z3"]
        .astype(np.float32)
        .sel(lat=slice(0, 90))
        .chunk("auto")
    )
    l_win = lanczos(2 * half_len + 1)[:, None, None, None]
    dims = ds.dims
    for var in ds.data_vars:
        ds[f"{var}bar"] = (
            dims,
            (convolve(ds[var].data, l_win)[half_len:-half_len] / l_win.sum()).astype(
                np.float32
            ),
        )
        ds[f"{var}p"] = ds[var] - ds[f"{var}bar"]
        del ds[f"{var}bar"]
        del ds[var]
    ds = ds.chunk({"time": 1390, "lat": 72, "lon": 161})
    res = ds.to_zarr(opath, compute=False)
    compute(res, progress=True)
    
    
# block 3: EP Flux

for run in ["ctrl", "dobl"]:
    ipath = Path(
        f"{DATADIR}/Henrik_data/{run}/high_wind/6H/results",
        "Eddy_NH_10days.zarr",
    )
    ofile = Path(f"/storage/workspaces/giub_meteo_impacts/ci01/Henrik_data/{run}/high_wind/6H/results/F300.zarr")
    if ofile.is_dir():
        continue
    ds = xr.open_dataset(ipath).chunk("auto")
    ds["dthetadp"] = xr.open_mfdataset(f"{DATADIR}/Henrik_data/{run}/dthetadp/6H/*.nc")[DEFAULT_VARNAME].rename("dthetadp").astype(np.float32)
    ds["dudp"] = xr.open_mfdataset(f"{DATADIR}/Henrik_data/{run}/vertical/6H/*.nc")["dudp"].astype(np.float32)
    ds["u"] = xr.open_mfdataset(f"{DATADIR}/Henrik_data/{run}/high_wind/6H/*.nc")["u"].chunk("auto")
    
    ds = ds.sel(lev=30000)
    
    Gen_EP = {}

    gamma = (-KAPPA / ds.lev * (100000 / ds.lev) ** KAPPA * ds["dthetadp"].mean(["time", "lon", "lat"])).astype(np.float32)
    EAPE = (C_P_AIR * 0.5 * (ds.lev * 1e-5) ** (2 * KAPPA) * gamma * ds["thetap"] ** 2).astype(np.float32)
    S = (0.5 * (ds["up"] ** 2 + ds["vp"] ** 2 - EAPE)).astype(np.float32)
    f = (2 * OMEGA * degsin(ds.lat)).astype(np.float32)

    ## Base 2 * 3
    Gen_EP["F11"] = ds["up"] ** 2 - S
    Gen_EP["F12"] = ds["up"] * ds["vp"]
    Gen_EP["F13"] = ds["up"] * ds["omegap"] - ds["vp"] * ds["thetap"] * f / ds["dthetadp"]
    Gen_EP["F21"] = ds["up"] * ds["vp"]
    Gen_EP["F22"] = ds["vp"] ** 2 - S
    Gen_EP["F23"] = ds["vp"] * ds["omegap"] + ds["up"] * ds["thetap"] * f / ds["dthetadp"]

    ## Additional from original EP:
    Gen_EP["F12_extra"] = - ds["dudp"] * ds["vp"] * ds["thetap"] / ds["dthetadp"]
    # Gen_EP["F13_extra"] = (ds["u"] * degcos(ds.lat)).differentiate("lat") * ds["vp"] * ds["thetap"] / ds["dthetadp"] / RADIUS / degcos(ds.lat)

    # ## Additional from Plumb 85:
    # Gen_EP["F11_extra_"] = -(ds["vp"] * ds["phip"]).differentiate("lon") / f / RADIUS / degcos(ds.lat)
    # Gen_EP["F12_extra_"] = -(ds["up"] * ds["phip"]).differentiate("lon") / f / RADIUS / degcos(ds.lat)
    # Gen_EP["F13_extra_"] = -(ds["thetap"] * ds["phip"]).differentiate("lon") / RADIUS / degcos(ds.lat) / ds["dthetadp"]
    Gen_EP = xr.Dataset(Gen_EP).chunk({"time": 1390, "lat": 72, "lon": 161})
    res = Gen_EP.to_zarr(ofile, compute=False)
    compute(res, progress=True)

# block 4: WB

# levels = compute(xr.open_mfdataset(list(Path(DATADIR, "Henrik_data/ctrl/zeta/6H").glob("*0.nc")))["__xarray_dataarray_variable__"].quantile([0.25, 0.375, 0.5, 0.625, 0.75]), progress_flag=True).values.tolist()
# for run in ["ctrl", "dobl"]:
#     basepath_zeta = Path(DATADIR, f"Henrik_data/{run}/zeta/6H")
#     da_mflux = xr.open_dataset(
#         f"{DATADIR}/Henrik_data/{run}/high_wind/6H/results/Eddy_NH_10days.zarr"
#     ).sel(lev=30000)
#     da_mflux = (da_mflux["up"] * da_mflux["vp"]).rename("EMF")
#     opath = Path(DATADIR, "Henrik_data", run) 
#     opath_rwb = Path(DATADIR, f"Henrik_data/{run}/rwb_index")
#     opath_rwb.mkdir(exist_ok=True)
#     for year in trange(1969, 2021):
#         ofile = opath_rwb.joinpath(f"overturnings_{year}.parquet")
#         zeta = xr.open_dataarray(basepath_zeta.joinpath(f"{year}.nc")).sel(lev=30000)
#         zeta = zeta.assign_coords(
#             lat=(0.7 + np.arange(len(zeta.lat), dtype=np.float32) * 0.94).round(2)
#         ).rename("zeta").reset_coords(["lev", "loni", "lati"], drop=True) * 1e5
#         mflux = da_mflux.sel(time=zeta.time)
#         mflux = mflux.assign_coords(
#             lat=(0.7 + np.arange(len(mflux.lat), dtype=np.float32) * 0.94).round(2)
#         ).rename("mflux").reset_coords("lev", drop=True)
#         zeta = smooth(zeta, {"lon": ("win", 3), "lat": ("win", 3)})
#         zeta = compute(zeta)
#         mflux = smooth(mflux, {"lon": ("win", 3), "lat": ("win", 3)})
#         mflux = compute(mflux)
#         if ofile.is_file():
#             overturnings = pl.read_parquet(ofile)
#             overturnings_on_grid = None
#         else:
#             contours = detect_contours(zeta, levels, processes=N_WORKERS, ctx="fork")
#             overturnings = detect_overturnings(contours)
#             overturnings, overturnings_on_grid = event_props(overturnings, [zeta, mflux])
#             overturnings.write_parquet(ofile)
        
#         for orientation in ["cyclonic", "anticyclonic"]:
#             name = f"{orientation[0].upper()}AVO"
#             odir = opath.joinpath(f"{name}/6H")
#             odir.mkdir(parents=True, exist_ok=True)
#             ofile = odir.joinpath(f"{year}.nc")
#             if ofile.is_file():
#                 continue
#             df = overturnings.filter(pl.col("orientation") == orientation)
#             da = to_xarray_sjoin(zeta, events=df)
#             da.to_netcdf(ofile)
            
#             odir = opath.joinpath(f"{name}/dailyany")
#             odir.mkdir(parents=True, exist_ok=True)
#             da = da.any("level").resample(time="1D").any().astype(np.uint8)
#             da.to_netcdf(odir.joinpath(f"{year}.nc"))
            
# block 5: define jets (already computed probably)

both_jets = {}
both_paths = {}
for run in ["ctrl", "dobl"]:
    dh = DataHandler.from_specs(
        "Henrik_data",
        run,
        ("high_wind", ["u", "v", "s", "theta"]),
        "6H",
        minlon=-80,
        maxlon=40,
        minlat=15,
        maxlat=80,
        levels=20000,
    )
    exp = JetFindingExperiment(dh)
    all_jets_one_df = exp.find_jets(
        force=False,
        n_coarsen=1,
        smooth_s=5,
        alignment_thresh=0.55,
        base_s_thresh=0.55,
        int_thresh_factor=0.2,
        hole_size=5,
    )
    theta300 = open_da(
        "Henrik_data",
        run,
        ("high_wind", ["s", "theta"]),
        "6H",
        minlon=-80,
        maxlon=40,
        minlat=15,
        maxlat=80,
        levels=30000,
    ).rename({"s": "s300", "theta": "theta300"})
    all_jets_one_df = add_feature_for_cat(
        all_jets_one_df,
        "s300",
        theta300,
        ofile_ajdf=exp.path.joinpath("all_jets_one_df.parquet"),
        force=False,
    )
    all_jets_one_df = add_feature_for_cat(
        all_jets_one_df,
        "theta300",
        theta300,
        ofile_ajdf=exp.path.joinpath("all_jets_one_df.parquet"),
        force=False,
    )
    all_jets_one_df = exp.categorize_jets(
        None,
        ["s300", "theta300"],
        force=False,
        n_init=15,
        init_params="k-means++",
        mode="month",
    ).cast({"time": pl.Datetime("ms")})
    phat_filter = (pl.col("is_polar") < 0.5) | (
        (pl.col("is_polar") > 0.5) & (pl.col("int") > 1.0e8)
    )
    phat_jets = all_jets_one_df.filter(
        (pl.col("is_polar").mean().over(["time", "jet ID"]) < 0.5)
        | (
            (pl.col("is_polar").mean().over(["time", "jet ID"]) > 0.5)
            & (pl.col("int").mode().first().over(["time", "jet ID"]) > 1.0e8)
        )
    )
    both_paths[run] = exp.path
    both_jets[run] = phat_jets.with_columns(
        **{
            "jet ID": (pl.col("is_polar").mean().over(["time", "jet ID"]) > 0.5).cast(
                pl.UInt32()
            )
        }
    )

# stage 6: Interpolate new fields
args = ["all", None, -100, 60, 0, 90]

to_do = (
    ("AAVO", "AAVO", {}),
    ("CAVO", "CAVO", {}),
)

for run in ["ctrl", "dobl"]:
    for huh in to_do:
        rename, name, kwargs = huh
        ofile = both_paths[run].joinpath(f"{rename}_relative.parquet")
        if ofile.is_file():
            continue
        da_ = open_da("Henrik_data", run, name, "6H", *args, **kwargs).rename(rename)
        if rename in ["AAVO", "CAVO"]:
            da_ = da_.isel(lev=2)
        da_ = compute(da_)
        interpd = create_jet_relative_dataset(both_jets[run], da_)
        del da_
        interpd.write_parquet(ofile)
    Gen_EP = xr.open_dataset(f"/storage/workspaces/giub_meteo_impacts/ci01/Henrik_data/{run}/high_wind/6H/results/F300.zarr")
    for data_var in Gen_EP.data_vars:
        ofile = both_paths[run].joinpath(f"{data_var}_relative.parquet")
        if ofile.is_file():
            continue
        da_ = Gen_EP[data_var]
        interpd = create_jet_relative_dataset(both_jets[run], da_)
        del da_
        interpd.write_parquet(ofile)
    Gen_EP = xr.open_dataset(f"/storage/workspaces/giub_meteo_impacts/ci01/Henrik_data/{run}/high_wind/6H/results/F300_2x.zarr")
    for data_var in Gen_EP.data_vars:
        ofile = both_paths[run].joinpath(f"{data_var}_relative.parquet")
        if ofile.is_file():
            continue
        da_ = Gen_EP[data_var]
        interpd = create_jet_relative_dataset(both_jets[run], da_)
        del da_
        interpd.write_parquet(ofile)