import pandas as pd
import numpy as np
import os
from definitions import DATADIR, YEARSPL
import cdsapi


def main():
    c = cdsapi.Client()
    for year in range(1940, 2023):
        if os.path.isfile(f'{DATADIR}/ERA5/Wind/Multi/raw/v/{year}.nc'):
            continue
        c.retrieve(
            'reanalysis-era5-pressure-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': 'v_component_of_wind',
                'pressure_level': [
                    '300', '500', '700', '775', '850', '925'
                ],
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
            f'{DATADIR}/ERA5/Wind/Multi/raw/v/{year}.nc')
        

if __name__ == '__main__':
    main()
