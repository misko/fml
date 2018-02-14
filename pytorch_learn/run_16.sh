#!/bin/bash

max_runs=1
runs=0

#lrs="0.001 0.0001"
lrs="0.001"
#radiuss="3 9 12 15"
radiuss="3 12"
layers="32"
includehs="0"
distpowers="0 1"
filters="4"
for lr in $lrs; do
for distpower in $distpowers; do
for layer in $layers; do
for filter in $filters; do
for radius in $radiuss; do
for includeh in $includehs; do
	python atomblaster.py --markersize 4 --distpower $distpower --nfilters $filter --nlayers $layer --lr $lr --radius $radius --includeh $includeh save.data &
	sleep 10 &
	runs=`expr ${runs} + 1`
	while [ "${runs}" -ge "${max_runs}" ]; do
		wait
		runs=0
	done
done
done
done
done
done
done

exit


#check if distpower 0/1 matters
python atomblaster.py --markersize 4 --distpower 0 --nfilters 4 --nlayers 1 --lr 0.001 --radius 3 --includeh 1 save.data &
python atomblaster.py --markersize 4 --distpower 1 --nfilters 4 --nlayers 1 --lr 0.001 --radius 3 --includeh 1 save.data &
python atomblaster.py --markersize 4 --distpower 2 --nfilters 4 --nlayers 1 --lr 0.001 --radius 3 --includeh 1 save.data &
python atomblaster.py --markersize 4 --distpower 3 --nfilters 4 --nlayers 1 --lr 0.001 --radius 3 --includeh 1 save.data &
wait
wait
wait 
wait
#check if distpower includeh matters
python atomblaster.py --markersize 4 --distpower 1 --nfilters 4 --nlayers 1 --lr 0.001 --radius 9 --includeh 1 save.data &
python atomblaster.py --markersize 4 --distpower 1 --nfilters 4 --nlayers 1 --lr 0.001 --radius 9 --includeh 0 save.data &
python atomblaster.py --markersize 4 --distpower 1 --nfilters 4 --nlayers 1 --lr 0.001 --radius 6 --includeh 1 save.data &
python atomblaster.py --markersize 4 --distpower 1 --nfilters 4 --nlayers 1 --lr 0.001 --radius 12 --includeh 1 save.data &
wait
wait
wait
wait
#check if distpower filters matters
python atomblaster.py --markersize 4 --distpower 1 --nfilters 8 --nlayers 1 --lr 0.001 --radius 9 --includeh 1 save.data &
python atomblaster.py --markersize 4 --distpower 1 --nfilters 16 --nlayers 1 --lr 0.001 --radius 9 --includeh 1 save.data &
python atomblaster.py --markersize 4 --distpower 1 --nfilters 4 --nlayers 2 --lr 0.001 --radius 9 --includeh 1 save.data &
wait
wait
wait

