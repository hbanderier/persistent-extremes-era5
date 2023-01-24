import numpy as np
import pandas as pd


DATADIR = "/scratch2/hugo/ERA5"
CLIMSTOR = "/mnt/climstor/ecmwf/era5/raw"

def filenamescm(y, m, d):  # Naming conventions of the files on climstor (why are they so different?)
    return [f"{CLIMSTOR}/ML/data/{str(y)}/P{str(y)}{str(m).zfill(2)}{str(d).zfill(2)}_{str(h).zfill(2)}" for h in range(0, 24, 6)]
def filenamecp(y, m, d):
    return [f"{CLIMSTOR}/PL/data/an_pl_ERA5_{str(y)}-{str(m).zfill(2)}-{str(d).zfill(2)}.nc"]  # returns iterable to have same call signature as filenamescl(y, m, d)
def filenamegeneric(y, m, folder):
    return [f"{DATADIR}/{folder}/{y}{str(m).zfill(2)}.nc"]

def _fn(date, which):
    if which == "ML":
        return filenamescm(date.year, date.month, date.day)
    elif which == "PL":
        return filenamecp(date.year, date.month, date.day)
    else:
        return filenamegeneric(date.year, date.month, which)
    
def fn(date, which):  # instead takes pandas.timestamp (or iterable of _) as input
    if isinstance(date, (list, np.ndarray, pd.DatetimeIndex)):
        filenames = []
        for d in date:
            filenames.extend(_fn(d, which))
        return filenames
    elif isinstance(date, pd.Timestamp):
        return _fn(date, which)
    else:
        raise RuntimeError(f"Invalid type : {type(date)}")

RADIUS = 6.371e6  # m
OMEGA = 7.2921e-5  # rad.s-1
KAPPA = 0.2854
R_SPECIFIC_AIR = 287.0500676

def degcos(x):
    return np.cos(x / 180 * np.pi)
def degsin(x):
    return np.sin(x / 180 * np.pi)

DATERANGEPL = pd.date_range("19590101", "20021231")
YEARSPL = np.unique(DATERANGEPL.year)
DATERANGEML = pd.date_range("19770101", "20211231")
WINDBINS = np.arange(0, 25, 2)
LATBINS = np.arange(15, 75, 2.5)
LONBINS = np.arange(-90, 30, 3)
DEPBINS = np.arange(-25, 26, 1.5)


def main():
    return


if __name__ == "__main__":
    main()