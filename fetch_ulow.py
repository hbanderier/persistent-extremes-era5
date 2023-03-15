from definitions import DATERANGEPL, DATADIR, fn
from cdo import Cdo
import os
cdo = Cdo(logging=True, logFile='cdo_commands.log')


def main():
    for path in fn(DATERANGEPL, "PL"):
        if not os.path.isfile(path):
            continue
        ofile = f'{DATADIR}/ERA5/Wind/Low/dailymean/' + path.split('/')[-1].split('_')[-1]
        if os.path.isfile(ofile):
            continue
        cdo.daymean(input=f'-vertmean -selname,u -sellevel,700,750,800,850 {path}', output=ofile)


if __name__ == "__main__":
    main()