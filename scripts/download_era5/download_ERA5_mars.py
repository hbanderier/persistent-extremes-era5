from pathlib import Path
from jetutils.definitions import DATADIR, YEARS
from concurrent.futures import ThreadPoolExecutor, as_completed
import calendar
import cdsapi

basepath = Path(f"{DATADIR}/ERA5/thetalev/PV/6H")


def retrieve(client, request, year):
    year = str(year).zfill(4)
    ofile = basepath.joinpath(f"{year}.nc")
    if Path(ofile).is_file():
        return
    request.update({"date": f'{year}-01-01/to/{year}-12-31'})
    client.retrieve('reanalysis-era5-complete', request, ofile)
    return f"Retrieved {year}"
    
    
def main():
    request = { 
        'date'    : f'1959-01-01/to/1959-12-31',
        'levtype' : 'pt',
        "levelist": "350",
        'param'   : '60',                  
        'stream'  : 'oper',                  
        'time'    : '00/to/23/by/6', 
        'type'    : 'an',
        'grid'    : '0.5/0.5', 
        'format'  : 'netcdf',
    }
    client = cdsapi.Client()
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(retrieve, client, request.copy(), year) for year in YEARS
        ]
        for f in as_completed(futures):
            try:
                print(f.result())
            except:
                print("could not retrieve")
        

if __name__ == "__main__":
    main()
