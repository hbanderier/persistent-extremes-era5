#!/bin/python3

from jetstream_hugo.clustering import Experiment


exp_s = Experiment(
    "ERA5", "plev", "s", "6H", (1940, 2023), None, -80, 30, 20, 75
)
exp_s.find_jets(60, 10)
