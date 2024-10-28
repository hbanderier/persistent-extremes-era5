from pathlib import Path
from jetstream_hugo.definitions import DATADIR, YEARS
from concurrent.futures import ThreadPoolExecutor, as_completed
import calendar
import cdsapi

basepath = Path(f"{DATADIR}/ERA5/surf")


def retrieve(client, request, year, month):
    last_day = calendar.monthrange(year, month)[1]
    last_day = str(last_day).zfill(2)
    month = str(month).zfill(2)
    year = str(year).zfill(4)
    ofile = basepath.joinpath(f'raw/{year}{month}.nc')
    if Path(ofile).is_file():
        return
    request.update({"year": year, "month": month})
    client.retrieve("derived-era5-single-levels-daily-statistics", request, ofile)
    return f"Retrieved {year}, {month}"
    
    
def main():
    request = {
        "product_type": "reanalysis",
        "variable": [
            "2m_temperature",
            "mean_sea_level_pressure",
            "sea_surface_temperature",
            "total_precipitation",
        ],
        'year': '1959',
        'month': '01',
        'day': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
            '13', '14', '15',
            '16', '17', '18',
            '19', '20', '21',
            '22', '23', '24',
            '25', '26', '27',
            '28', '29', '30',
            '31',
        ],
        "daily_statistic": "daily_mean",
        "time_zone": "utc+00:00",
        "frequency": "6_hourly",
        "grid": "0.5/0.5",
        'format': 'netcdf',
        "area": [90, -180, 0, 180],
    }
    client = cdsapi.Client()
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(retrieve, client, request.copy(), year, month) for year in YEARS for month in range(1, 13)
        ]
        for f in as_completed(futures):
            try:
                print(f.result())
            except:
                print("could not retrieve")
        

if __name__ == "__main__":
    main()
