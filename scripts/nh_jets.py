from jetutils.definitions import DATADIR
from jetutils.jet_finding import JetFindingExperiment, get_double_jet_index
from jetutils.data import DataHandler, open_da
from pathlib import Path


if __name__ == '__main__':
    ds = open_da("ERA5", "plev", "high_wind", "6H", "all", None, None, None, 0, 90, "all")
    basepath = Path(DATADIR, "ERA5", "plev", "high_wind", "6H", "results")
    dh = DataHandler.from_basepath_and_da(basepath, ds)
    exp = JetFindingExperiment(dh)
    
    ds = open_da("ERA5", "plev", "mid_wind", "6H", "all", None, None, None, 0, 90, "all")
    basepath = Path(DATADIR, "ERA5", "plev", "mid_wind", "6H", "results")
    dh_low = DataHandler.from_basepath_and_da(basepath, ds)
    
    all_jets_one_df = exp.find_jets()
    all_jets_one_df = exp.categorize_jets(dh_low.da["s"])
    all_jets_one_df, all_jets_over_time, flags = exp.track_jets()
    props_as_df_uncat = exp.props_as_df(False)
    props_as_df = exp.props_as_df(True)
    all_props_over_time = exp.props_over_time(all_jets_over_time, props_as_df_uncat)
    jet_pos_da = exp.jet_position_as_da()
    props_as_df = get_double_jet_index(props_as_df, jet_pos_da)
    props_as_df.write_parquet(exp.path.joinpath("props_as_df_extras.parquet"))
