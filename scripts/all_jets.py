from jetstream_hugo.definitions import DATADIR
from jetstream_hugo.jet_finding import JetFindingExperiment
from jetstream_hugo.data import DataHandler
import xarray as xr

if __name__ == '__main__':
    ds = xr.open_mfdataset(f"{DATADIR}/ERA5/plev/high_wind/6H/????.nc")
    dh = DataHandler(ds, f"{DATADIR}/ERA5/plev/high_wind/6H/results")
    exp = JetFindingExperiment(dh)
    exp.track_jets()
    exp.props_as_df(True)

    ds = xr.open_mfdataset(f"{DATADIR}/ERA5/plev/high_wind/dailymean/????.nc")
    dh = DataHandler(ds, f"{DATADIR}/ERA5/plev/high_wind/dailymean/results")
    exp = JetFindingExperiment(dh)
    exp.track_jets()
    exp.props_as_df(True)