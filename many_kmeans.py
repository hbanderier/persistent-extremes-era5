from itertools import product
from definitions import cluster, create_grid_directory
from cdo import Cdo
import click

cdo = Cdo()
@click.command()
@click.option("--kind", type=click.Choice(["kmeans", "kmedoids"], case_sensitive=False), default="kmeans", help="Which clustering type to perform")
@click.option("--season", type=click.Choice(["All", "DJF", "MAM", "JJA", "SON"], case_sensitive=False), default="All", help="Which season")
def main(kind, season):
    minlon = [-90, -60]
    maxlon = [30, 60, 90, 120]
    minlat = [15, 30]
    maxlat = [80]
    n_clu = [2, 3, 4, 5]

    packs = product(minlon, maxlon, minlat, maxlat, n_clu)
    season = season.upper()
    if season == "ALL":
        season = None
    for pack in packs:
        minlon, maxlon, minlat, maxlat, n_clu = pack
        print(pack)
        path = create_grid_directory(
            cdo, "ERA5", "Wind", 300, minlon, maxlon, minlat, maxlat
        )
        _ = cluster(
            n_clu, path, "s", season=season, detrended=True
        )


if __name__ == "__main__":
    main()