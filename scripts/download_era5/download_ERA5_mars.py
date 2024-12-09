from pathlib import Path
from jetstream_hugo.definitions import DATADIR, YEARS
from concurrent.futures import ThreadPoolExecutor, as_completed
import calendar
import cdsapi

basepath = Path(f"{DATADIR}/ERA5/2PVU/")


def retrieve(client, request, year, month):
    last_day = calendar.monthrange(year, month)[1]
    last_day = str(last_day).zfill(2)
    month = str(month).zfill(2)
    year = str(year).zfill(4)
    ofile = f'{basepath}/raw/{year}{month}.nc'
    if Path(ofile).is_file():
        return
    request.update({"date": f'{year}-{month}-01/to/{year}-{month}-{last_day}'})
    client.retrieve('reanalysis-era5-complete', request, ofile)
    return f"Retrieved {year}, {month}"
    
    
def main():
    request = { 
        'date'    : f'1959-01-01/to/1959-01-31',
        'levtype' : 'pv',
        "levelist": "2000",
        'param'   : '3/54/131/132',                  
        'stream'  : 'oper',                  
        'time'    : '00/to/23/by/6', 
        'type'    : 'an',
        'grid'    : '0.5/0.5', 
        'format'  : 'netcdf',
    }
    client = cdsapi.Client()
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(retrieve, client, request.copy(), year, month) for year in YEARS[40:] for month in range(1, 13)
        ]
        for f in as_completed(futures):
            try:
                print(f.result())
            except:
                print("could not retrieve")
        

if __name__ == "__main__":
    main()
