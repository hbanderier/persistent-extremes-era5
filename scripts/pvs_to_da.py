from pathlib import Path
from tqdm import tqdm
import numpy as np
from jetutils.definitions import compute, YEARS, DATADIR
from jetutils.data import open_da, to_netcdf


for type_ in ["apvs", "cpvs"]:
    for year in tqdm(YEARS):
        opath = Path(DATADIR, f"ERA5/thetalev/{type_}/dailyany", f"{year}.nc")
        if opath.is_file():
            continue
        da = open_da("ERA5", "thetalev", type_, "6H", [year], None, None, None, None, None, "all").astype(np.int8).any("lev").resample(time="1D").any()
        da = compute(da)
        to_netcdf(da, opath)


# def to_xarray(events: st.GeoDataFrame, dummy_da: xr.DataArray, varname: str):
#     dummy_da = dummy_da.rename("dummy")
#     da_df = pl.from_pandas(dummy_da.to_dataframe().reset_index())
#     orig_times = da_df["time"]
#     timedtype = orig_times.dtype
#     da_df = da_df.drop("time", "dummy").cast({"lon": pl.Float32, "lat": pl.Float32})
#     da_df = da_df.unique(["lat", "lon"]).sort(["lat", "lon"]).with_columns(geometry=st.from_xy("lon", "lat"))
#     da_df = st.GeoDataFrame(da_df)
#     events = events.with_columns(pl.col("geometry").st.buffer(0.25))
#     if varname == "flag":
#         events = events.with_columns(flag=pl.lit(1))
#     events = (
#         events.select(["date", "geometry", varname])
#         .cast({varname: pl.UInt32 if varname == "flag" else pl.Float32})
#         .rename({"date": "time"})
#     )
#     events = events.cast({"time": timedtype})
#     events = events.filter(pl.col("time").is_in(orig_times))
#     dummy_da = xr.zeros_like(dummy_da, dtype=np.float32)
#     events = events.st.sjoin(da_df, on="geometry", how="inner", predicate="contains")
#     events = events.unique(["time", "lon", "lat"])
#     events_da = xr.DataArray.from_series(
#         events[["time", varname, "lat", "lon"]]
#         .to_pandas()
#         .set_index(["time", "lat", "lon"])[varname]
#     ).fillna(0)
#     dummy_da.loc[
#         {
#             "time": events_da.time.values,
#             "lat": events_da.lat.values,
#             "lon": events_da.lon.values,
#         }
#     ] = events_da
#     return dummy_da


# all_events = {}
# for level in range(310, 365, 5):
#     events = st.from_geopandas(gpd.read_parquet(f"/storage/workspaces/giub_meteo_impacts/ci01/ERA5/RWB_index/era5_pv_streamers_{level}K_1959-2022.parquet"))

#     tropospheric = events.filter(pl.col("mean_var") < pl.col("level"))
#     anticyclonic = tropospheric.filter(pl.col("intensity") >= pl.col("level"))
#     cyclonic = tropospheric.filter(pl.col("intensity") < pl.col("level"))
    
#     all_events[level] = {"anti": anticyclonic, "cycl": cyclonic}
    
    
# Path(f"{DATADIR}/ERA5/thetalev/apvs/6H/").mkdir(exist_ok=True, parents=True)
# Path(f"{DATADIR}/ERA5/thetalev/cpvs/6H/").mkdir(exist_ok=True, parents=True)
# varname = "flag"
# dtype = {"flag": np.uint32, "intensity": np.float32}[varname]
# for year in YEARS:
#     print(year)
#     # for month in range(1, 13):
#     ofile_anti = Path(f"{DATADIR}/ERA5/thetalev/apvs/6H/{year}.nc")
#     ofile_cycl = Path(f"{DATADIR}/ERA5/thetalev/cpvs/6H/{year}.nc")
#     if ofile_cycl.is_file() and ofile_anti.is_file():
#         continue
#     time_mask = (TIMERANGE.year == year)# & (TIMERANGE.month == month)
#     coords = {
#         "time": TIMERANGE[time_mask],
#         "lat": np.arange(0, 90.5, .5),
#         "lon": np.arange(-180, 180, .5),
#     }
#     shape = [len(co) for co in coords.values()]
#     dummy_da = xr.DataArray(np.zeros(shape, dtype=dtype), coords=coords)
#     anti_all_levs = {}
#     cycl_all_levs = {}
#     for lev, events in tqdm(all_events.items(), total=11):
#         anti_all_levs[lev] = to_xarray(events["anti"], dummy_da, varname)
#         cycl_all_levs[lev] = to_xarray(events["cycl"], dummy_da, varname)
#     anti_all_levs = xr.concat(anti_all_levs.values(), dim="lev").assign_coords(lev=list(anti_all_levs))
#     cycl_all_levs = xr.concat(cycl_all_levs.values(), dim="lev").assign_coords(lev=list(cycl_all_levs))
#     anti_all_levs.to_netcdf(ofile_anti)
#     cycl_all_levs.to_netcdf(ofile_cycl)
#     del anti_all_levs, cycl_all_levs
#     gc.collect()
    
    
# basepath = Path(DATADIR, "ERA5", "thetalev")
# for varname in ["apvs", "cpvs"]:
#     path = basepath.joinpath(varname)
#     dest_dir = path.joinpath("dailymean")
#     dest_dir.mkdir(exist_ok=True)
#     for orig_path in tqdm(path.joinpath("6H").glob("????.nc"), total=len(YEARS)):
#         dest_path = dest_dir.joinpath(orig_path.name)
#         if dest_path.is_file():
#             continue
#         da = xr.open_dataset(orig_path)
#         da = da.resample(time="1D").mean()
#         da = da.to_netcdf(dest_path)