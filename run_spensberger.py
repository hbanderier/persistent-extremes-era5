import netCDF4 as nc
import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
from jetstream_hugo.definitions import DATADIR
from jetaxis_detect.jet_detect import jet_detect

MAX_POINTS = 10000
MAX_LINES = 500

jet_detect.jetint_thres = 0.124e-8	# K-threshold for instantaneous ERA-Interim, mean winds 925-700 hPa, T84-resolution
jet_detect.searchrad = 1.5		# Maximum distance of points along the jet axis in grid point indices
jet_detect.minlen = 2.0e6		# Minimum lenght of the jet axes in meters
jet_detect.grid_cyclic_ew = False	# Is grid periodic in x-direction?

tvar, latvar, lonvar = 'time', 'lat', 'lon'
ufile, uvar = f"{DATADIR}/ERA5/tmp/u.nc", 'u'
vfile, vvar = f"{DATADIR}/ERA5/tmp/v.nc", 'v'

fu = nc.Dataset(ufile)
u = fu.variables[uvar][::].squeeze().data.astype(np.float32).copy()
time = fu.variables[tvar][::].data.astype(np.int16).copy()
lat = fu.variables[latvar][::].data.astype(np.float32).copy()
lon = fu.variables[lonvar][::].data.astype(np.float32).copy()
fu.close()

ny, nx = len(lat), len(lon)

# Load v-data
fv = nc.Dataset(vfile)
v = fv.variables[vvar][::].squeeze().data.astype(np.float16).copy()
fv.close()

dx = np.ones((ny,nx)) * 111.111e3 * np.cos(np.pi/180.0 * lat)[:,np.newaxis].astype(np.float32).copy()
dy = (np.ones((ny,nx)) * 111.111e3).astype(np.float32).copy()

# Detect jets
ja, jaoff = jet_detect.run_jet_detect(MAX_POINTS, MAX_LINES, u, v, dx, dy)