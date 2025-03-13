from jetstream_hugo.definitions import DATADIR
from jetstream_hugo.jet_finding import JetFindingExperiment, get_double_jet_index
from jetstream_hugo.data import DataHandler
import xarray as xr

if __name__ == '__main__':
    exp = JetFindingExperiment(DataHandler(f"{DATADIR}/ERA5/plev/high_wind/6H/results/1"))
    dh_low = DataHandler.from_specs(
        "ERA5", "plev", "mid_wind", "6H", "all", None, -80, 40, 15, 80, "all"
    )
    ds = exp.ds
    all_jets_one_df = exp.find_jets()
    all_jets_one_df = exp.categorize_jets(dh_low.da["s"])
    all_jets_one_df, all_jets_over_time, flags = exp.track_jets()
    props_as_df_uncat = exp.props_as_df(False)
    props_as_df = exp.props_as_df(True)
    all_props_over_time = exp.props_over_time(all_jets_over_time, props_as_df_uncat)
    ds = exp.ds
    da = exp.ds["s"]
    jet_pos_da = exp.jet_position_as_da()
    props_as_df = get_double_jet_index(props_as_df, jet_pos_da)
    props_as_df.write_parquet(exp.path.joinpath("props_as_df_extras.parquet"))