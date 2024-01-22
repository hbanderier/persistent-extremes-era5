from pathlib import Path
import numpy as np
from jetstream_hugo.data import open_da
from jetstream_hugo.definitions import DATADIR


def find_indices(da_PV, level):
    switches = np.abs((da_PV <= level).astype(int).diff("lev"))
    good = switches.any("lev")
    above = switches.argmax("lev") + 1
    above = above.where(good, 0)
    below = (above - 1).where(good, 0)
    prefactor = (level - da_PV.isel(lev=above)) / (da_PV.isel(lev=below) - da_PV.isel(lev=above))
    minvar = da_PV.min("lev") >= level
    maxvar = da_PV.max("lev") <= level
    return prefactor, above, below, good, minvar, maxvar


def interpolate_isosurface(da, prefactor, below, above, good, minvar, maxvar):
    interp_level = (prefactor * (da.isel(lev=below) - da.isel(lev=above))) + da.isel(lev=above)
    # Handle missing values and instances where no values for surface exist above and below
    interp_level = interp_level.where(good)
    interp_level = interp_level.where(good | ~minvar, da.isel(lev=0))
    interp_level = interp_level.where(good | ~maxvar, da.isel(lev=-1))
    interp_level = interp_level.reset_coords('lev', drop=True)
    return interp_level


da_lev = None
varnames = ['u', 'v', 's', 'P', 'theta']
baseout = Path(DATADIR, "ERA5", "2PVU")
for year in range(1959, 2023):
    PV = open_da(
        "ERA5", "thetalev", "PV", "6H", year
    ).load()
    if da_lev is None:
        da_lev = PV.lev
    prefactor, above, below, good, minvar, maxvar = find_indices(PV, 2)    
    del PV
    for varname in varnames:
        outpath = baseout.joinpath(Path(varname, '6H', f'{year}.nc'))
        outpath.parent.mkdir(parents=True, exist_ok=True)
        if outpath.is_file():
            continue
        if varname == 'theta':
            da = da_lev
        else:
            da = open_da("ERA5", "thetalev", varname, "6H", year).load()
        da = interpolate_isosurface(da, prefactor, below, above, good, minvar, maxvar)
        da.astype(np.float32).to_netcdf(outpath)
        del da