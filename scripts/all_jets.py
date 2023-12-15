#!/bin/python3

from jetstream_hugo.clustering import *

exp_s = Experiment(
    "ERA5", "s", "6H", (1940, 2023), [5, 6, 7, 8, 9], -60, 70, 25, 85, 250, None, None, None
)

all_jets, where_are_jets, all_jets_one_array = exp_s.find_jets(62, 4)