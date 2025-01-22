from jetstream_hugo.definitions import DATADIR
from jetstream_hugo.jet_finding import JetFindingExperiment
from jetstream_hugo.data import DataHandler
import xarray as xr

if __name__ == '__main__':
    exp = JetFindingExperiment(DataHandler(f"{DATADIR}/ERA5/plev/high_wind/6H/results/6"))
    dh_low = DataHandler.from_specs("ERA5", "plev", "mid_wind", "6H", "all", None, -80, 40, 15, 80, "all")
    exp.find_jets()
    exp.categorize_jets(dh_low.da["s"])
    exp.track_jets()
    exp.props_as_df()
