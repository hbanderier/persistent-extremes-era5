from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from jetutils.definitions import DATADIR
from cdsapi import Client

basepath = Path(f"{DATADIR}/ERA5/plev/t300/6H")

def retrieve(client, request, year):
    year = str(year).zfill(4)
    ofile = basepath.joinpath(f'{year}.nc')
    if Path(ofile).is_file():
        return
    request.update({"year": year})
    client.retrieve("reanalysis-era5-pressure-levels", request, ofile)
    return f"Retrieved {year}"
    
    
def main():
    request = {
        "product_type": "reanalysis",
        "variable": [
            "temperature",
        ],
        "year": "1982",
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
        "pressure_level": "300",
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": [90, -180, 0, 180],
        "grid": "0.5/0.5",
    }
    client = Client()
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(retrieve, client, request.copy(), year) for year in range(1959, 2023) # modify this if needed
        ]
        for f in as_completed(futures):
            try:
                print(f.result())
            except:
                print("could not retrieve")
        

if __name__ == "__main__":
    main()
