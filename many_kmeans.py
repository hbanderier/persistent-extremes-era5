from itertools import product
from definitions import do_kmeans, create_grid_directory
from cdo import Cdo

cdo = Cdo()

def main():
    minlon = [-90, -75, -60]
    maxlon = [30, 60]
    minlat = [15, 30]
    maxlat = [80]
    n_clu = [2, 3, 4, 5]

    packs = product(minlon, maxlon, minlat, maxlat, n_clu)
    for pack in packs:
        minlon, maxlon, minlat, maxlat, n_clu = pack
        print(pack)
        path = create_grid_directory(
            cdo, "ERA5", "Wind", 300, minlon, maxlon, minlat, maxlat
        )
        _ = do_kmeans(
            n_clu, path, "s", detrended=True
        )


if __name__ == "__main__":
    main()