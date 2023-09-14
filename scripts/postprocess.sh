#!/bin/bash


basepath="/storage/scratch/users/hb22g102/$1"
echo ${basepath}
for year in $(seq -f "%04g" 1940 2022); do
	ofile="${basepath}/regridded/$2/${year}.nc"
	echo ${ofile}
	if [ ! -f ${ofile} ]; then
		echo ${year}
		/storage/homefs/hb22g102/mambaforge/envs/env11/bin/cdo -invertlat -sellonlatbox,-180,180,-90,90 "${basepath}/raw/$2/${year}.nc" ${ofile}
	fi
done
#  -chname,latitude,lat -chname,longitude,lon