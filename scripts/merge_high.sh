#!/bin/bash


basepath="/storage/scratch/users/hb22g102/ERA5/Wind"
for year in $(seq -f "%04g" 1940 2022); do
    if [ ! -f "${basepath}/200_250/split/${year}u.nc" ]; then
        /storage/homefs/hb22g102/mambaforge/envs/env11/bin/cdo splitname "${basepath}/200_250/raw/${year}.nc" "${basepath}/200_250/split/${year}"
    fi
	for var in u v; do
        ofile="${basepath}/Multi2/${var}/${year}.nc"
	    echo ${ofile}
	    if [ ! -f ${ofile} ]; then
		    echo ${year}
            /storage/homefs/hb22g102/mambaforge/envs/env11/bin/cdo -sellevel,300,500,700,850 "${basepath}/Multi/regridded/${var}/${year}.nc" "${basepath}/Multi/regridded2/${var}/${year}.nc"
		    /storage/homefs/hb22g102/mambaforge/envs/env11/bin/cdo -invertlat -sellonlatbox,-180,180,-90,90 "${basepath}/200_250/split/${year}${var}.nc" "${basepath}/200_250/split2/${var}/${year}.nc"
            /storage/homefs/hb22g102/mambaforge/envs/env11/bin/cdo -b F32 merge "${basepath}/200_250/split2/${var}/${year}.nc" "${basepath}/Multi/regridded2/${var}/${year}.nc" "${basepath}/Multi2/${var}/${year}.nc"
 
	    fi
    done
done
#  -chname,latitude,lat -chname,longitude,lon
