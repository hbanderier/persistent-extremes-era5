import pandas as pd
import numpy as np
import os
from definitions import DATADIR, YEARSPL
import cdsapi


def main():
    basepath = f'{DATADIR}/ERA5/prec/surf/raw'
    c = cdsapi.Client()
    for year in range(1940, 2023):
        ofile = f'{basepath}/{year}.nc'
        if os.path.isfile(ofile):
            continue
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': 'total_precipitation',
                'year': str(year),
                'month': [
                    '01', '02', '03',
                    '04', '05', '06',
                    '07', '08', '09',
                    '10', '11', '12',
                ],
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
                'time': [
                    '00:00', '06:00', '12:00',
                    '18:00',
                ],
                'area': [90, -100, 0, 100],
                'grid': '0.5/0.5',
                'format': 'netcdf',
            },
            ofile)
        

if __name__ == '__main__':
    main()