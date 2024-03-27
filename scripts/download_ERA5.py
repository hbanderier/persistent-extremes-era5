from pathlib import Path
from jetstream_hugo.definitions import DATADIR, YEARSPL
import cdsapi


def main():
    basepath = Path(f"{DATADIR}/ERA5/plev/")
    c = cdsapi.Client()
    for year in range(1959, 1960):
        for month in range(1, 13):
            month_str = str(month).zfill(2)
            path = basepath.joinpath(f"uv{year}{month_str}_allglobe.nc")
            if path.is_file():
                continue
            c.retrieve(
                "reanalysis-era5-pressure-levels",
                {
                    "product_type": "reanalysis",
                    "variable": [
                        "u_component_of_wind",
                        "v_component_of_wind",
                    ],
                    "pressure_level": [
                        "150",
                        "200",
                        "250",
                        "300",
                        "350",
                        "500",
                        "700",
                        "850",
                    ],
                    "year": str(year),
                    "month": month_str,
                    "day": [
                        "01",
                        "02",
                        "03",
                        "04",
                        "05",
                        "06",
                        "07",
                        "08",
                        "09",
                        "10",
                        "11",
                        "12",
                        "13",
                        "14",
                        "15",
                        "16",
                        "17",
                        "18",
                        "19",
                        "20",
                        "21",
                        "22",
                        "23",
                        "24",
                        "25",
                        "26",
                        "27",
                        "28",
                        "29",
                        "30",
                        "31",
                    ],
                    "time": [
                        "00:00",
                        "06:00",
                        "12:00",
                        "18:00",
                    ],
                    "grid": "1.0/1.0",
                    "format": "netcdf",
                },
                path,
            )


if __name__ == "__main__":
    main()
