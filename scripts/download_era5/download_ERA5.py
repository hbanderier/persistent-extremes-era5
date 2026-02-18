from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from jetutils.definitions import DATADIR
from cdsapi import Client

basepath = Path(f"{DATADIR}/ERA5/plev/uv/6H")
basepath.mkdir(parents=True, exist_ok=True)
suffix = ""
suffix = f"_{suffix}" if suffix != "" else ""

def retrieve(client: Client, request: dict, year: int, month: int | None = None):
    year = str(year).zfill(4)
    if month is not None:
        month = str(month).zfill(2)
        ofile = basepath.joinpath(f'{year}{month}{suffix}.nc')
    else:
        month = [str(i).zfill(2) for i in range(1, 13)]
        ofile = basepath.joinpath(f'{year}{suffix}.nc')
    if Path(ofile).is_file():
        return
    request.update({"year": year, "month": month})
    client.retrieve("reanalysis-era5-pressure-levels", request, ofile)
    return f"Retrieved {year}, {month}"
    
    
def main():
    request = {
        "product_type": ["reanalysis"],
        "variable": [
            "u_component_of_wind",
            "v_component_of_wind",
            "temperature",
        ],
        "year": "2023",
        "month": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12"
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
            "31"
        ],
        "time": [
            "00:00", "06:00", "12:00",
            "18:00"
        ],
        "pressure_level": ["250", "850"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": [90, -180, 0, 180],
        "grid": "0.5/0.5",
    }
    client = Client()
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [
            executor.submit(retrieve, client, request.copy(), year) for year in range(1973, 2025) # modify this if needed
        ]
        for f in as_completed(futures):
            try:
                print(f.result())
            except Exception:
                print("could not retrieve")

if __name__ == "__main__":
    main()
