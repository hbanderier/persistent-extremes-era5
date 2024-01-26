from pathlib import Path
from jetstream_hugo.definitions import DATADIR, YEARSPL
import cdsapi


def main():
    basepath = Path(f"{DATADIR}/ERA5/thetalev/")
    c = cdsapi.Client()
    for year in range(1959, 1960):
        path = basepath.joinpath(f"usv{year}JJA.nc")
        if path.is_file():
            continue
        c.retrieve(
            "reanalysis-era5-pressure-levels",
            {
                "product_type": "reanalysis",
                "format": "netcdf",
                "variable": ["u", "v", "pv"],
                "year": str(year),
                "month": [
                    "06", "07", "08",
                ],
                "day": [
                    "01", "02", "03",
                    "04", "05", "06",
                    "07", "08", "09",
                    "10", "11", "12",
                    "13", "14", "15",
                    "16", "17", "18",
                    "19", "20", "21",
                    "22", "23", "24",
                    "25", "26", "27",
                    "28", "29", "30",
                    "31",
                ],
                "time": [
                    "00:00", "06:00", "12:00",
                    "18:00",
                ],
                "pressure_level": [
                    "200", "225", "250", "300", "350", "500", "600", "700", "850"
                ],
                "area": [90, -80, 0, 70],
                "grid": "0.5/0.5",
                "format": "netcdf",
            },
            path)
        

if __name__ == "__main__":
    main()
